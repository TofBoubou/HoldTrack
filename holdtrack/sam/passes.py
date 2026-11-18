"""Run the five SAM 3 passes on a video from the boxes of a detector.

    python -m holdtrack.sam.passes clip.mp4 --detections rf_2xl_detections.npz \
        --checkpoint weights/sam3.pt --out work/clip

No detector runs here. The caller gives the detections that start the passes and, for the
complete system, the detections of a second detector whose tracks carry the forward passes.
Tracks and masks are returned in the frame order of the input video. The backward pass runs
ByteTrack on the reversed video; the mirrored pass mirrors the existing ByteTrack tracks.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from holdtrack.artifacts import checked, write_new
from holdtrack.cache import atomic_json, file_hash
from holdtrack.detect.bank import DetectionBank
from holdtrack.inputs.mot import physical
from holdtrack.inputs.video import prepare_video
from holdtrack.sam.transforms import (
    reflect_boxes,
    reflect_mot,
    restore_mirrored_support,
    restore_reverse_clock,
    reverse_bank,
)

ROLES = ("BASE", "EXCL", "PROT", "reverse_BASE", "mirror_EXCL")


@dataclass(frozen=True)
class SamAssets:
    checkpoint: Path
    checkpoint_sha256: str
    sam3_root: Path | None = None
    sam2_predictor: Path | None = None
    device: str = "cuda"

    def validate(self):
        from holdtrack.sam._sequence import _installed_sources
        from holdtrack.sam.lifecycle import SourceBindings

        if self.device not in {"cpu", "cuda"}:
            raise ValueError("Choose cpu or cuda")
        checkpoint = checked(
            dict(path=str(self.checkpoint.expanduser().resolve()), sha256=self.checkpoint_sha256),
            Path.cwd(),
        )
        if (self.sam3_root is None) != (self.sam2_predictor is None):
            raise ValueError("Provide both official SAM source locations, or neither")
        sources = (
            _installed_sources()
            if self.sam3_root is None
            else SourceBindings(
                self.sam3_root.expanduser().resolve(), self.sam2_predictor.expanduser().resolve()
            )
        )
        sources.validate()
        return checkpoint, sources


@dataclass(frozen=True)
class PassOutput:
    role: str
    raw: Path
    projected: Path
    masks: Path | None
    receipt: Path
    source_projected: Path | None = None


def _reference(path: Path) -> dict:
    return dict(path=str(path.resolve()), sha256=file_hash(path))


def _associations(bank: DetectionBank, sequence, destination: Path) -> Path:
    from holdtrack.inputs.bytetrack import replay_npz

    raw = destination.with_suffix(".npz")
    bank.save(raw)
    replay_npz(
        raw,
        file_hash(raw),
        frame_count=len(sequence.paths),
        width=sequence.width,
        height=sequence.height,
        output=destination,
    )
    return destination


def prepare_passes(
    video: str | Path,
    detections: str | Path | DetectionBank,
    output: str | Path,
    *,
    assets: SamAssets,
    roles: tuple[str, ...] = ROLES,
    fps: float | None = None,
    associations: str | Path | None = None,
    support_detections: str | Path | DetectionBank | None = None,
    support_associations: str | Path | None = None,
) -> Path:
    """Validate the frames, the model files and the inputs of every pass, without loading a model.

    Detections have zero-based frames, boxes as x1, y1, x2, y2 in pixels and scores; an NPZ
    holds frames/boxes/scores or frame/boxes/score. ``associations`` is the ByteTrack MOT file
    of these detections. ``support_detections`` and ``support_associations`` give a second
    detector whose tracks carry the three forward passes: in the complete system the passes
    start on RF-DETR L at 1344 and are projected on the tracks of RF-DETR 2XL. The backward
    and mirrored passes stay on the tracks of the first detector.
    """
    from holdtrack.sam._sequence import _load_detections, _run_prepared

    if not roles or len(set(roles)) != len(roles) or not set(roles) <= set(ROLES):
        raise ValueError(f"Choose distinct pass roles from {ROLES}")
    checkpoint, sources = assets.validate()
    root = Path(output).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    sequence = prepare_video(video, root / "input", fps=fps)
    count = len(sequence.paths)
    bank = (
        detections
        if isinstance(detections, DetectionBank)
        else DetectionBank(*_load_detections(detections, count))
    )
    if (bank.frames >= count).any():
        raise ValueError("Detections fall outside this video's frame clock")
    original_count = len(bank.frames)
    bank, indices = bank.eligible(0.01)
    directory = root / "inputs"
    directory.mkdir()
    forward_bt = directory / "forward_ByteTrack.txt"
    if associations is None:
        _associations(bank, sequence, forward_bt)
    else:
        payload = Path(associations).read_bytes()
        if any(not 0 <= row[0] < count for _, row in physical(payload)):
            raise ValueError("Forward ByteTrack falls outside the original video clock")
        write_new(forward_bt, payload)
    projection_bt = forward_bt
    crossed_projection = support_detections is not None or support_associations is not None
    projection_bank = None
    if crossed_projection:
        projection_bt = directory / "projection_ByteTrack.txt"
        support = None
        if support_detections is not None:
            support = (
                support_detections
                if isinstance(support_detections, DetectionBank)
                else DetectionBank(*_load_detections(support_detections, count))
            )
            if (support.frames >= count).any():
                raise ValueError("Projection detections fall outside this video's clock")
            support, _ = support.eligible(0.01)
            projection_bank = directory / "projection_detections.npz"
            support.save(projection_bank)
        if support_associations is not None:
            payload = Path(support_associations).read_bytes()
            if any(not 0 <= row[0] < count for _, row in physical(payload)):
                raise ValueError("Projection ByteTrack falls outside the original video clock")
            write_new(projection_bt, payload)
        else:
            _associations(support, sequence, projection_bt)
        if support is not None:
            from holdtrack.geometry import bind_observations
            from holdtrack.steps.features import mot_bytes

            tracks = mot_bytes(projection_bt.read_bytes())
            _, proof = bind_observations(support.arrays(), tracks, tracks)
            if proof["ambiguous_score_linked_rows"]:
                raise ValueError("Projection support has ambiguous source detections")
    images = sequence.references()
    reversed_images = None
    reverse = reverse_indices = reverse_bt = None
    if "reverse_BASE" in roles:
        reverse, reverse_indices = reverse_bank(bank, count)
        reverse_bt = _associations(reverse, sequence, directory / "reverse_ByteTrack.txt")
        reversed_images = []
        reverse_directory = directory / "reverse_frames"
        reverse_directory.mkdir()
        for index, image in enumerate(reversed(images), 1):
            source = Path(image["path"])
            target = reverse_directory / f"{index:08d}{source.suffix.lower()}"
            try:
                target.symlink_to(source)
            except OSError:
                try:
                    os.link(source, target)
                except OSError:
                    shutil.copyfile(source, target)
            if file_hash(target) != image["sha256"]:
                raise ValueError("Reversed frame bytes changed")
            # Do not resolve the alias: the native sequence directory is reversed.
            reversed_images.append(dict(path=str(target), sha256=image["sha256"]))
    manifests = {}
    for role in roles:
        is_reverse, is_mirror = role == "reverse_BASE", role == "mirror_EXCL"
        selected = reverse if is_reverse else bank
        source_indices = reverse_indices if is_reverse else np.arange(len(bank.frames))
        boxes = reflect_boxes(selected.boxes, sequence.width) if is_mirror else selected.boxes
        rows = [
            [int(frame), int(index), *map(float, box), float(score)]
            for frame, index, box, score in zip(
                selected.frames, source_indices, boxes, selected.scores, strict=True
            )
        ]
        raw = directory / (role + "_detections.json")
        atomic_json(raw, rows)
        support = reverse_bt if is_reverse else forward_bt
        if is_mirror:
            support = directory / "mirror_ByteTrack.txt"
            write_new(support, reflect_mot(forward_bt.read_bytes(), sequence.width))
        manifest = directory / (role + ".json")
        atomic_json(
            manifest,
            dict(
                schema="holdtrack-rgb-stream-v1",
                video=Path(video).stem,
                voice="BASE" if is_reverse else "EXCL" if is_mirror else role,
                width=sequence.width,
                height=sequence.height,
                frame_count=count,
                images=reversed_images if is_reverse else images,
                raw_detections=_reference(raw),
                bytetrack=_reference(support),
                sam3_root=str(sources.sam3_root),
                sam2_predictor=str(sources.sam2_predictor),
                checkpoint=dict(path=str(checkpoint), sha256=assets.checkpoint_sha256),
                device=assets.device,
                mirror=is_mirror,
            ),
        )
        validation = _run_prepared(manifest, root / "not_written", validate_inputs_only=True)
        if validation["model_loaded"] or validation["frame_count"] != count:
            raise RuntimeError("The real pass input was not validated")
        manifests[role] = _reference(manifest)
    plan = root / "passes.json"
    atomic_json(
        plan,
        dict(
            schema="holdtrack-sam-passes-v1",
            roles=list(roles),
            manifests=manifests,
            frame_count=count,
            width=sequence.width,
            height=sequence.height,
            fps=sequence.fps,
            frame_order="natural filename order; original one-based MOT clock",
            decoded_video=sequence.decoded,
            forward_support=_reference(forward_bt),
            forward_projection_support=_reference(projection_bt),
            projection_detections=_reference(projection_bank) if projection_bank else None,
            crossed_forward_projection=crossed_projection,
            reverse_and_mirror_support="SAM seed detector; never the alternate projection bank",
            positive_native_floor=0.01,
            input_detections=original_count,
            eligible_detections=len(indices),
            discarded_by_positive_floor=original_count - len(indices),
            all_inputs_validated_before_model=True,
        ),
    )
    return plan


def run_prepared_passes(plan: str | Path, *, export_masks: bool = True) -> dict[str, PassOutput]:
    """Run complete voices serially and restore clocks/coordinates for consumers."""
    from holdtrack.masks import MaskWriter
    from holdtrack.sam._sequence import _run_prepared
    from holdtrack.sam.projection import project

    plan = Path(plan).resolve()
    configuration = json.loads(plan.read_text())
    if configuration["schema"] != "holdtrack-sam-passes-v1":
        raise ValueError("Expected holdtrack-sam-passes-v1")
    roles = configuration["roles"]
    if not roles or len(set(roles)) != len(roles) or not set(roles) <= set(ROLES):
        raise ValueError("The plan must contain distinct, known pass roles")
    root = plan.parent
    count, width = configuration["frame_count"], configuration["width"]
    original_support = checked(configuration["forward_support"], root).read_bytes()
    projection_support = checked(configuration["forward_projection_support"], root).read_bytes()
    outputs = {}
    for role in roles:
        if role not in ROLES:
            raise ValueError(f"Unknown SAM role: {role}")
        manifest = checked(configuration["manifests"][role], root)
        destination = root / role
        destination.mkdir(exist_ok=False)
        writer = MaskWriter(destination / "masks") if export_masks else None
        reverse, mirror = role == "reverse_BASE", role == "mirror_EXCL"

        def observe(frame, masks, scores, rows):
            if writer is None:
                return
            restored_frame = count - 1 - frame if reverse else frame
            if mirror:
                masks = {identity: mask[:, ::-1] for identity, mask in masks.items()}
            restored_rows = [(restored_frame, *row[1:]) for row in rows]
            writer.frame(
                restored_frame, masks, {int(k): float(v) for k, v in scores.items()}, restored_rows
            )

        start = time.perf_counter()
        try:
            result = _run_prepared(manifest, destination / "native", on_frame=observe)
        finally:
            # Native predictor methods contain reference cycles. End a completed
            # predictor's lifetime before loading the next policy, without
            # retaining several complete models in GPU memory.
            gc.collect()
            torch = sys.modules.get("torch")
            if torch is not None and torch.cuda.is_initialized():
                torch.cuda.empty_cache()
        raw, projected = Path(result["sam"]).read_bytes(), Path(result["mot"]).read_bytes()
        if reverse:
            raw, projected = (
                restore_reverse_clock(raw, count),
                restore_reverse_clock(projected, count),
            )
        elif mirror:
            raw = reflect_mot(raw, width)
            projected = restore_mirrored_support(original_support, projected, width)
        source_projected = projected
        if not reverse and not mirror and configuration["crossed_forward_projection"]:
            # Use raw mask boxes, never an already relabelled intermediate MOT.
            projected, _ = project(projection_support, raw)
        write_new(destination / "sam.txt", raw)
        write_new(destination / "mot.txt", projected)
        write_new(destination / "source_mot.txt", source_projected)
        mask_files = sorted(writer.directory.glob("*.json")) if writer else []
        if writer and [int(p.stem) for p in mask_files] != list(range(1, count + 1)):
            raise ValueError("Incomplete RLE image clock")
        torch = sys.modules.get("torch")
        gpu = (
            torch.cuda.get_device_name()
            if torch is not None and torch.cuda.is_initialized()
            else None
        )
        receipt = destination / "receipt.json"
        atomic_json(
            receipt,
            dict(
                role=role,
                voice=result["voice"],
                frames=count,
                seconds=time.perf_counter() - start,
                GPU=gpu,
                original_image_coordinates=True,
                original_one_based_clock=True,
                mirror_after_native_RGB_normalization=mirror,
                reverse_ByteTrack_recomputed=reverse,
                manifest=_reference(manifest),
                raw=_reference(destination / "sam.txt"),
                projected=_reference(destination / "mot.txt"),
                source_projected=_reference(destination / "source_mot.txt"),
                crossed_projection=bool(
                    not reverse and not mirror and configuration["crossed_forward_projection"]
                ),
                masks=[_reference(path) for path in mask_files],
                native_raw=_reference(Path(result["sam"])),
                native_projected=_reference(Path(result["mot"])),
            ),
        )
        outputs[role] = PassOutput(
            role,
            destination / "sam.txt",
            destination / "mot.txt",
            writer.directory if writer else None,
            receipt,
            destination / "source_mot.txt",
        )
    atomic_json(
        root / "receipt.json",
        dict(
            schema="holdtrack-sam-passes-complete-v1",
            plan=_reference(plan),
            passes={role: _reference(value.receipt) for role, value in outputs.items()},
        ),
    )
    return outputs


def run_passes(
    video,
    detections,
    output,
    *,
    assets: SamAssets,
    roles=ROLES,
    fps=None,
    associations=None,
    support_detections=None,
    support_associations=None,
    export_masks=True,
) -> dict[str, PassOutput]:
    """Prepare arbitrary pixels, play requested voices, return restored MOT/RLE paths."""
    plan = prepare_passes(
        video,
        detections,
        output,
        assets=assets,
        roles=roles,
        fps=fps,
        associations=associations,
        support_detections=support_detections,
        support_associations=support_associations,
    )
    return run_prepared_passes(plan, export_masks=export_masks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="MP4/video file or image directory")
    parser.add_argument(
        "--detections",
        required=True,
        help="detections that start the passes (NPZ, zero-based frames)",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--sam3-root")
    parser.add_argument("--sam2-predictor")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--roles", nargs="+", choices=ROLES, default=list(ROLES))
    parser.add_argument("--fps", type=float)
    parser.add_argument("--associations", help="ByteTrack tracks of the detections (MOT file)")
    parser.add_argument(
        "--support-detections",
        help="detections of a second detector whose tracks carry the forward passes (NPZ)",
    )
    parser.add_argument(
        "--support-associations", help="ByteTrack tracks of the second detector (MOT file)"
    )
    parser.add_argument("--no-masks", action="store_true")
    parser.add_argument(
        "--prepare-only", action="store_true", help="Validate all inputs without loading a model"
    )
    options = parser.parse_args()
    from holdtrack.weights import manifest

    assets = SamAssets(
        Path(options.checkpoint),
        options.checkpoint_sha256 or manifest()["sam3"]["sha256"],
        Path(options.sam3_root) if options.sam3_root else None,
        Path(options.sam2_predictor) if options.sam2_predictor else None,
        options.device,
    )
    plan = prepare_passes(
        options.video,
        options.detections,
        options.out,
        assets=assets,
        roles=tuple(options.roles),
        fps=options.fps,
        associations=options.associations,
        support_detections=options.support_detections,
        support_associations=options.support_associations,
    )
    if options.prepare_only:
        print(plan)
    else:
        run_prepared_passes(plan, export_masks=not options.no_masks)


if __name__ == "__main__":
    main()
