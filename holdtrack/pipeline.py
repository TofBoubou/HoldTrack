"""Run and resume the complete tracking pipeline."""

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from holdtrack import geometry
from holdtrack.artifacts import checked
from holdtrack.cache import StageCache, atomic_json, file_hash, fingerprint
from holdtrack.config import DEFAULT_PRESET, RunConfig, TrackSteps, preset
from holdtrack.detect.bank import DetectionBank
from holdtrack.fusion import FusionViews, fuse
from holdtrack.inputs.mot import physical
from holdtrack.steps.chain import (
    RefinementInputs,
    apply_box_offset,
    apply_gap_filling,
    apply_stitching,
)
from holdtrack.steps.features import features, mot_bytes
from holdtrack.steps.reliability import keep_mask, predict, subset
from holdtrack.steps.short_tracks import retained_observations
from holdtrack.vote import vote_bytes


@dataclass(frozen=True)
class CachedSequence:
    detections: Path
    associations: Path
    voices: dict[str, Path]
    views: dict[str, Path | None]
    head: dict
    steps: TrackSteps
    gap_detections: Path | None = None
    feature_to_gap_indices: tuple[int, ...] | None = None

    @classmethod
    def load(cls, path: str | Path) -> "CachedSequence":
        source = Path(path)
        payload = json.loads(source.read_text())
        if payload["schema"] != "holdtrack-run-inputs-v1":
            raise ValueError("Expected holdtrack-run-inputs-v1 cached inputs")

        def reference(value: dict) -> Path:
            return checked(value, source.parent)

        return cls(
            reference(payload["detections"]),
            reference(payload["associations"]),
            {voice: reference(value) for voice, value in payload["voices"].items()},
            {view: reference(value) if value else None for view, value in payload["views"].items()},
            payload["reliability_head"],
            TrackSteps.from_dict(payload["steps"]),
            reference(payload["gap_detections"]) if payload.get("gap_detections") else None,
            tuple(payload["feature_to_gap_indices"])
            if "feature_to_gap_indices" in payload
            else None,
        )


def _engine_signature() -> str:
    root = Path(__file__).parent
    return fingerprint(
        {str(path.relative_to(root)): file_hash(path) for path in sorted(root.rglob("*.py"))}
    )


def _save_arrays(path: Path, **arrays: np.ndarray) -> None:
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def _byte_stage(cache: StageCache, name: str, inputs: dict, produce) -> Path:

    def write_payload(path: Path) -> None:
        path.write_bytes(produce())

    return cache.stage(name, inputs, "tracks.txt", write_payload)


def replay_sequence(inputs: CachedSequence, cache: StageCache, configuration: RunConfig) -> Path:
    """Use complete detector and SAM intermediates through the normal cached pipeline."""
    engine = _engine_signature()
    detections = DetectionBank.load(inputs.detections)
    associations = mot_bytes(inputs.associations.read_bytes())
    for voice in configuration.voices:
        if voice not in inputs.voices:
            raise ValueError(f"The cache has no complete {voice} trajectory")
    voice_paths = {
        voice: cache.import_file("sam_" + voice.lower(), inputs.voices[voice])
        for voice in configuration.voices
    }
    voice_signature = {voice: file_hash(path) for voice, path in voice_paths.items()}

    def voted_payload() -> bytes:
        if len(voice_paths) == 1:
            return next(iter(voice_paths.values())).read_bytes()
        return vote_bytes({voice: path.read_bytes() for voice, path in voice_paths.items()})[0]

    voted_path = _byte_stage(
        cache, "vote", dict(engine=engine, voices=voice_signature), voted_payload
    )
    voted = voted_path.read_bytes()
    bank = detections.arrays()
    query, provenance = geometry.bind_observations(bank, associations, mot_bytes(voted))
    if provenance["ambiguous_score_linked_rows"]:
        raise ValueError("A track box has an ambiguous detector score")
    view_banks = {
        name: DetectionBank.load(path) if path else None for name, path in inputs.views.items()
    }
    views = FusionViews(view_banks, detections, inputs.associations.read_bytes())
    auxiliary = {
        name: views.arrays(name) if name in configuration.auxiliary_detectors else None
        for name in ("mirror", "dfine")
    }
    fused_path = _byte_stage(
        cache,
        "fusion",
        dict(
            engine=engine,
            vote=file_hash(voted_path),
            views={name: file_hash(path) if path else None for name, path in inputs.views.items()},
            auxiliary=configuration.auxiliary_detectors,
            mode=configuration.fusion_mode,
            extra_weight=configuration.extra_view_weight,
        ),
        lambda: fuse(voted, views, configuration),
    )
    fused = fused_path.read_bytes()

    def feature_arrays(path: Path) -> None:
        matrix, _, identities, _ = features(
            voted, fused, bank, associations, auxiliary["mirror"], auxiliary["dfine"]
        )
        _save_arrays(path, matrix=matrix, identities=identities)

    feature_path = cache.stage(
        "reliability_features",
        dict(
            engine=engine,
            vote=file_hash(voted_path),
            fusion=file_hash(fused_path),
            detections=file_hash(inputs.detections),
            associations=file_hash(inputs.associations),
            support={
                name: file_hash(inputs.views[name]) if inputs.views.get(name) else None
                for name in ("mirror", "dfine")
            },
        ),
        "features.npz",
        feature_arrays,
    )
    with np.load(feature_path, allow_pickle=False) as saved:
        matrix, identities = (saved["matrix"].copy(), saved["identities"].copy())
    probability_path = cache.stage(
        "reliability_probabilities",
        dict(engine=engine, features=file_hash(feature_path), head=inputs.head),
        "probabilities.npy",
        lambda path: _save_probabilities(path, predict(inputs.head, matrix)),
    )
    probabilities = np.load(probability_path, allow_pickle=False)
    reliable = keep_mask(probabilities, inputs.steps.temporal_threshold)
    reliability_path = _byte_stage(
        cache,
        "reliability",
        dict(
            engine=engine,
            source=file_hash(fused_path),
            probabilities=file_hash(probability_path),
            threshold=inputs.steps.temporal_threshold,
        ),
        lambda: subset(fused, reliable),
    )
    retained = retained_observations(identities, reliable, inputs.steps.min_track_length)
    length_path = _byte_stage(
        cache,
        "short_tracks",
        dict(
            engine=engine,
            reliability=file_hash(reliability_path),
            minimum=inputs.steps.min_track_length,
        ),
        lambda: subset(fused, retained),
    )
    stitch_path = _byte_stage(
        cache,
        "stitching",
        dict(
            engine=engine,
            source=file_hash(length_path),
            max_gap=inputs.steps.stitch_max_gap,
            max_distance=inputs.steps.stitch_max_distance,
        ),
        lambda: apply_stitching(length_path.read_bytes(), inputs.steps),
    )
    offset_path = _byte_stage(
        cache,
        "box_offset",
        dict(engine=engine, source=file_hash(stitch_path), offset=inputs.steps.box_offset),
        lambda: apply_box_offset(stitch_path.read_bytes(), inputs.steps),
    )
    gap_bank = DetectionBank.load(inputs.gap_detections) if inputs.gap_detections else detections
    source_indices = (
        np.asarray(inputs.feature_to_gap_indices, int)[query]
        if inputs.feature_to_gap_indices is not None
        else query
    )
    refinement = RefinementInputs(
        matrix, identities, source_indices, gap_bank.frames, gap_bank.boxes, gap_bank.scores
    )
    result = _byte_stage(
        cache,
        "gap_filling",
        dict(
            engine=engine,
            source=file_hash(offset_path),
            detections=file_hash(inputs.gap_detections or inputs.detections),
            retained_source_indices=source_indices[retained].tolist(),
            max_gap=inputs.steps.max_gap,
            score_min=inputs.steps.gap_score_min,
            score_max=inputs.steps.gap_score_max,
        ),
        lambda: apply_gap_filling(offset_path.read_bytes(), refinement, retained, inputs.steps),
    )
    atomic_json(
        cache.directory / "stages.json",
        dict(
            detections=str(inputs.detections),
            associations=str(inputs.associations),
            views={name: str(path) if path else None for name, path in inputs.views.items()},
            vote=str(voted_path),
            fusion=str(fused_path),
            tracks=str(result),
        ),
    )
    return result


def _save_probabilities(path: Path, probabilities: np.ndarray) -> None:
    with path.open("xb") as stream:
        np.save(stream, probabilities, allow_pickle=False)


def _produce_sequence(
    frames: Path,
    cache: StageCache,
    configuration: RunConfig,
    weights_dir: Path,
    detector_python: Path | None,
    dfine_source: Path | None,
    device: str,
    masks: Path | None = None,
) -> CachedSequence:
    from PIL import Image

    from holdtrack.detect.runner import detect_sequence
    from holdtrack.inputs.bytetrack import associate_predictions, mot_text, verify_native_backend
    from holdtrack.sam._sequence import _installed_sources, _run_prepared
    from holdtrack.weights import manifest, prepare_weight

    verify_native_backend()
    images = sorted(
        path for path in frames.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        raise ValueError("The frame directory contains no images")
    with Image.open(images[0]) as image:
        width, height = image.size
    image_references = [dict(path=str(path.resolve()), sha256=file_hash(path)) for path in images]
    detector_names = tuple(
        dict.fromkeys((configuration.principal_detector, *configuration.auxiliary_detectors))
    )
    checkpoints = {
        name: prepare_weight("yolo" if name == "mirror" else name, weights_dir)
        for name in detector_names
    }
    sam_checkpoint = prepare_weight("sam3", weights_dir)
    source_bindings = _installed_sources()
    detector_paths = {}
    for name in detector_names:
        detector_paths[name] = detect_sequence(
            name,
            images,
            checkpoints[name],
            cache,
            python=detector_python or Path(sys.executable),
            source=dfine_source,
            device=device,
        )
    detections = DetectionBank.load(detector_paths[configuration.principal_detector])
    eligible, eligible_indices = detections.eligible()
    raw_path = cache.stage(
        "principal_candidates",
        dict(raw=file_hash(detector_paths[configuration.principal_detector]), floor=0.01),
        "detections.npz",
        eligible.save,
    )

    def associate(path: Path) -> None:
        rows, _ = associate_predictions(
            eligible.frames,
            eligible.boxes,
            eligible.scores,
            frame_count=len(images),
            width=width,
            height=height,
        )
        observed = dict(
            frame=np.asarray([row[0] for row in rows], np.int32),
            identity=np.asarray([row[1] for row in rows], np.int64),
            boxes=np.asarray([row[2:6] for row in rows], np.float32).reshape(-1, 4),
            confidence=np.ones(len(rows), np.float32),
        )
        path.write_bytes(mot_text(observed).encode())

    association = cache.stage(
        "association",
        dict(
            raw=file_hash(raw_path),
            frames=len(images),
            width=width,
            height=height,
            ultralytics="8.4.64",
        ),
        "tracks.txt",
        associate,
    )
    _candidates = [
        [int(frame), index, *map(float, box), float(score)]
        for index, (frame, box, score) in enumerate(
            zip(eligible.frames, eligible.boxes, eligible.scores)
        )
    ]
    raw_json = cache.directory / "principal_candidates.json"
    atomic_json(raw_json, _candidates)
    voices = {}
    mask_voice = "PROT" if "PROT" in configuration.voices else configuration.voices[0]
    for voice in configuration.voices:

        def produce_voice(path: Path, selected_voice=voice) -> None:
            bindings = dict(
                schema="holdtrack-rgb-stream-v1",
                voice=selected_voice,
                video=frames.name,
                width=width,
                height=height,
                frame_count=len(images),
                images=image_references,
                raw_detections=dict(path=str(raw_json), sha256=file_hash(raw_json)),
                bytetrack=dict(path=str(association), sha256=file_hash(association)),
                sam3_root=str(source_bindings.sam3_root),
                sam2_predictor=str(source_bindings.sam2_predictor),
                checkpoint=dict(path=str(sam_checkpoint), sha256=manifest()["sam3"]["sha256"]),
                device=device,
            )
            bindings_path = path.parent / "inputs.json"
            atomic_json(bindings_path, bindings)
            observe = None
            if masks is not None and selected_voice == mask_voice:
                from holdtrack.masks import MaskWriter

                writer = MaskWriter(path.parent / "masks")

                def observe(frame, frame_masks, scores, rows):
                    writer.frame(
                        frame, frame_masks, {int(k): float(v) for k, v in scores.items()}, rows
                    )

            produced = _run_prepared(bindings_path, path.parent / "trajectory", on_frame=observe)
            shutil.copyfile(produced["mot"], path)

        voices[voice] = cache.stage(
            "sam_" + voice.lower(),
            dict(
                engine=_engine_signature(),
                images=image_references,
                raw=file_hash(raw_json),
                association=file_hash(association),
                checkpoint=manifest()["sam3"]["sha256"],
                voice=voice,
                device=device,
                **(dict(masks=True) if masks is not None and voice == mask_voice else {}),
            ),
            "tracks.txt",
            produce_voice,
        )
        if masks is not None and voice == mask_voice:
            exported = voices[voice].parent / "masks"
            if not exported.is_dir():
                raise FileNotFoundError("The cached pass holds no masks; use a new work directory")
            names = sorted(p.name for p in exported.iterdir())
            if not masks.exists():
                shutil.copytree(exported, masks)
            elif sorted(p.name for p in masks.iterdir()) != names:
                raise FileExistsError("The mask folder holds other files; choose a new path")
    head_path = prepare_weight(configuration.reliability_head, weights_dir)
    head = json.loads(head_path.read_text())
    return CachedSequence(
        raw_path,
        association,
        voices,
        {
            name: detector_paths.get(name)
            for name in ("yolo", "mirror", "dfine", "rf_2xl", "rf_large")
            if name != configuration.principal_detector
        },
        head,
        configuration.steps,
        detector_paths[configuration.principal_detector],
        tuple(map(int, eligible_indices)),
    )


def _produce_crossed(
    frames: Path,
    cache: StageCache,
    configuration: RunConfig,
    weights_dir: Path,
    detector_python: Path | None,
    dfine_source: Path | None,
    device: str,
    masks: Path | None = None,
) -> CachedSequence:
    """Start the tracker passes on the boxes of the seed detector and project the three forward
    passes on the tracks of the principal detector. The backward and the mirrored pass are
    written next to them, in ``passes/`` of the work directory."""
    from holdtrack.detect.runner import detect_sequence
    from holdtrack.sam.passes import ROLES, SamAssets, run_passes
    from holdtrack.weights import manifest, prepare_weight

    images = sorted(
        path for path in frames.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        raise ValueError("The frame directory contains no images")
    seed = configuration.seed_detector
    detector_names = tuple(
        dict.fromkeys((configuration.principal_detector, seed, *configuration.auxiliary_detectors))
    )
    checkpoints = {
        name: prepare_weight("yolo" if name == "mirror" else name, weights_dir)
        for name in detector_names
    }
    sam_checkpoint = prepare_weight("sam3", weights_dir)
    detector_paths = {
        name: detect_sequence(
            name,
            images,
            checkpoints[name],
            cache,
            python=detector_python or Path(sys.executable),
            source=dfine_source,
            device=device,
        )
        for name in detector_names
    }
    detections = DetectionBank.load(detector_paths[configuration.principal_detector])
    eligible, eligible_indices = detections.eligible()
    raw_path = cache.stage(
        "principal_candidates",
        dict(raw=file_hash(detector_paths[configuration.principal_detector]), floor=0.01),
        "detections.npz",
        eligible.save,
    )
    root = cache.directory / "passes"
    if not (root / "receipt.json").exists():
        if root.exists():
            raise FileExistsError(f"{root} holds unfinished passes; remove it or change --out")
        run_passes(
            frames,
            DetectionBank.load(detector_paths[seed]),
            root,
            assets=SamAssets(sam_checkpoint, manifest()["sam3"]["sha256"], device=device),
            roles=ROLES,
            support_detections=detections,
        )
    if masks is not None and not masks.exists():
        shutil.copytree(root / "PROT" / "masks", masks)
    head = json.loads(prepare_weight(configuration.reliability_head, weights_dir).read_text())
    return CachedSequence(
        raw_path,
        root / "inputs" / "projection_ByteTrack.txt",
        {voice: root / voice / "mot.txt" for voice in configuration.voices},
        {
            name: detector_paths.get(name)
            for name in ("yolo", "mirror", "dfine", "rf_2xl", "rf_large")
            if name != configuration.principal_detector
        },
        head,
        configuration.steps,
        detector_paths[configuration.principal_detector],
        tuple(map(int, eligible_indices)),
    )


def run(
    frames: str | Path | None,
    out: str | Path,
    *,
    config: str | RunConfig = DEFAULT_PRESET,
    work_dir: str | Path | None = None,
    weights_dir: str | Path = "weights",
    cached_inputs: str | Path | None = None,
    detector_python: str | Path | None = None,
    dfine_source: str | Path | None = None,
    device: str = "cuda",
    masks: str | Path | None = None,
) -> Path:
    """Run all stages or resume verified intermediates with the same numerical pipeline.

    ``masks``: folder that receives the masks of the guarded pass, one file per frame; the
    tracks are the same with or without it."""
    configuration = preset(config) if isinstance(config, str) else config
    output = Path(out)
    cache = StageCache(work_dir or output.parent / (output.stem + ".work"))
    with cache.locked():
        if cached_inputs:
            inputs = CachedSequence.load(cached_inputs)
        else:
            if frames is None:
                raise ValueError("Provide frames or complete cached inputs")
            crossed = configuration.seed_detector not in (None, configuration.principal_detector)
            inputs = (_produce_crossed if crossed else _produce_sequence)(
                Path(frames).resolve(),
                cache,
                configuration,
                Path(weights_dir),
                Path(detector_python) if detector_python else None,
                Path(dfine_source) if dfine_source else None,
                device,
                Path(masks) if masks else None,
            )
        result = replay_sequence(inputs, cache, configuration)
        payload = result.read_bytes()
        physical(payload)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            if output.read_bytes() != payload:
                raise FileExistsError("The output contains different tracks; choose a new path")
            return output
        temporary = output.with_name(output.name + ".partial")
        with temporary.open("xb") as stream:
            stream.write(payload)
        temporary.replace(output)
        return output
