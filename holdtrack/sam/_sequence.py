"""Validate RGB sequence inputs and run one memory policy."""

import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from holdtrack.artifacts import checked, read_json, write_new
from holdtrack.inputs.mot import physical
from holdtrack.sam import _loop as tracking
from holdtrack.sam import backend, lifecycle, model, projection


@dataclass(frozen=True)
class DetectionCandidate:
    index: int
    box: tuple[float, float, float, float]
    confidence: float


def _candidates(rows: list, frame_count: int) -> dict[int, list[DetectionCandidate]]:
    seen = set()
    result = {}
    for row in rows:
        if len(row) != 7 or not np.isfinite(row).all():
            raise ValueError("Expected frame, index, x1, y1, x2, y2, score")
        frame, index = row[:2]
        if (
            int(frame) != frame
            or int(index) != index
            or (not 0 <= frame < frame_count)
            or (index < 0)
        ):
            raise ValueError("Invalid zero-based frame or source index")
        key = (int(frame), int(index))
        if key in seen:
            raise ValueError("Duplicate source detection index")
        if row[4] <= row[2] or row[5] <= row[3] or (not 0 <= row[6] <= 1):
            raise ValueError("Invalid detection extents or score")
        seen.add(key)
        result.setdefault(int(frame), []).append(
            DetectionCandidate(int(index), tuple(map(float, row[2:6])), float(row[6]))
        )
    for items in result.values():
        items.sort(key=lambda item: item.index)
    return result


def _validate_inputs(manifest: str | Path) -> tuple:
    from PIL import Image

    path = Path(manifest)
    data = read_json(path)
    base = path.parent
    if data["schema"] != "holdtrack-rgb-stream-v1" or data["voice"] not in {"BASE", "EXCL", "PROT"}:
        raise ValueError("Unknown sequence schema or voice")
    if type(data.get("mirror", False)) is not bool:
        raise ValueError("Horizontal reflection must be an explicit boolean")
    width, height, frame_count = (data[key] for key in ("width", "height", "frame_count"))
    if any(type(value) is not int or value < 1 for value in [width, height, frame_count]):
        raise ValueError("Invalid sequence dimensions")
    paths = tuple(checked(reference, base) for reference in data["images"])
    if len(paths) != frame_count or len(set(map(lambda p: p.resolve(), paths))) != frame_count:
        raise ValueError("Image population differs")
    if len({path.parent.resolve() for path in paths}) != 1:
        raise ValueError("All frames must belong to one directory")
    for image in paths:
        with Image.open(image) as rgb:
            if rgb.size != (width, height):
                raise ValueError("RGB dimensions changed")
            rgb.verify()
    detections = _candidates(read_json(checked(data["raw_detections"], base)), frame_count)
    observed = checked(data["bytetrack"], base).read_bytes()
    if any((not 0 <= row[0] < frame_count for _, row in physical(observed))):
        raise ValueError("An associated box falls outside the image sequence")
    sources = lifecycle.SourceBindings(data["sam3_root"], data["sam2_predictor"])
    sources.validate()
    checkpoint = Path(data["checkpoint"]["path"])
    if not checkpoint.is_absolute():
        checkpoint = base / checkpoint
    return (data, paths, detections, observed, sources, checkpoint)


def _run_prepared(
    manifest: str | Path,
    output: str | Path,
    *,
    validate_inputs_only: bool = False,
    model_loader: Callable | None = None,
    on_frame: Callable | None = None,
    on_memory: Callable | None = None,
    stop_after: int | None = None,
):
    data, paths, detections, observed, sources, checkpoint = _validate_inputs(manifest)
    if validate_inputs_only:
        return {"frame_count": len(paths), "voice": data["voice"], "model_loaded": False}
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    loader = model.load_tracker if model_loader is None else model_loader
    options = dict(device=data["device"])
    if data.get("mirror", False):
        options["mirror"] = True
    predictor, _ = loader(checkpoint, data["checkpoint"]["sha256"], sources, paths, **options)
    state = predictor.init_state(
        str(paths[0].parent),
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
        async_loading_frames=False,
    )
    try:
        rows, _ = tracking._run_sequence(
            predictor,
            state,
            detections,
            video=data["video"],
            width=data["width"],
            height=data["height"],
            frame_count=len(paths),
            voice=data["voice"],
            on_frame=on_frame,
            on_memory=on_memory,
            stop_after=stop_after,
        )
        raw = backend.mot_bytes(rows)
        if stop_after is not None:
            observed = b"".join(
                line
                for line in observed.splitlines(keepends=True)
                if int(line.split(b",", 1)[0]) <= stop_after
            )
        payload, _ = projection.project(observed, raw)
        write_new(destination / "sam.txt", raw)
        write_new(destination / "mot.txt", payload)
        return {
            "voice": data["voice"],
            "frames": stop_after or len(paths),
            "sam": str(destination / "sam.txt"),
            "mot": str(destination / "mot.txt"),
        }
    finally:
        predictor.reset_state(state)


def _load_detections(
    path: str | Path, frame_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = Path(path)
    if source.suffix == ".npz":
        with np.load(source, allow_pickle=False) as bank:
            frame_key, score_key = ("frames", "scores") if "frames" in bank else ("frame", "score")
            frames, boxes, scores = (bank[key].copy() for key in [frame_key, "boxes", score_key])
    else:
        rows = np.loadtxt(source, delimiter=",", ndmin=2)
        if rows.shape[1] != 10:
            raise ValueError("Text detections must use ten-column MOT format")
        frames, scores = (rows[:, 0] - 1, rows[:, 6])
        boxes = rows[:, 2:6].copy()
        boxes[:, 2:] += boxes[:, :2]
    if frames.ndim != 1 or boxes.shape != (len(frames), 4) or scores.shape != frames.shape:
        raise ValueError("Detection array dimensions disagree")
    if (
        not np.isfinite(boxes).all()
        or not np.isfinite(scores).all()
        or (not np.isfinite(frames).all())
    ):
        raise ValueError("Detections must be finite")
    if (boxes[:, 2:] <= boxes[:, :2]).any() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Detection extents or scores are invalid")
    if (
        not np.equal(frames, frames.astype(np.int64)).all()
        or ((frames < 0) | (frames >= frame_count)).any()
    ):
        raise ValueError("Detection frames must refer to the supplied image sequence")
    return (frames.astype(np.int64), boxes, scores)


def _installed_sources() -> lifecycle.SourceBindings:
    roots = []
    for name in ["sam3", "sam2"]:
        package = importlib.util.find_spec(name)
        if package is None or package.origin is None:
            raise ImportError(f"Install the official {name} package before tracking")
        roots.append(Path(package.origin).parent)
    return lifecycle.SourceBindings(roots[0].parent, roots[1] / "sam2_video_predictor.py")


def _run_from_frames(
    frames: str | Path,
    detections: str | Path,
    voice: str,
    *,
    checkpoint: str | Path | None,
    device: str,
    masks: str | Path | None,
) -> bytes:
    import tempfile

    from PIL import Image

    from holdtrack.cache import atomic_json, file_hash
    from holdtrack.inputs.bytetrack import associate_predictions, mot_text, verify_native_backend
    from holdtrack.weights import manifest

    directory = Path(frames)
    images = sorted(
        path for path in directory.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        raise ValueError("The frame directory contains no images")
    with Image.open(images[0]) as image:
        width, height = image.size
    source_frames, boxes, scores = _load_detections(detections, len(images))
    verify_native_backend()
    associated, _ = associate_predictions(
        source_frames, boxes, scores, frame_count=len(images), width=width, height=height
    )
    observed = dict(
        frame=np.asarray([row[0] for row in associated], np.int32),
        identity=np.asarray([row[1] for row in associated], np.int64),
        boxes=np.asarray([row[2:6] for row in associated], np.float32).reshape(-1, 4),
        confidence=np.ones(len(associated), np.float32),
    )
    sources = _installed_sources()
    weights = manifest()["sam3"]
    checkpoint = Path(checkpoint) if checkpoint else Path("weights") / weights["filename"]
    writer = None
    if masks is not None:
        from holdtrack.masks import MaskWriter

        writer = MaskWriter(masks)
    with tempfile.TemporaryDirectory(prefix="holdtrack-sam-") as temporary:
        work = Path(temporary)
        raw = work / "detections.json"
        atomic_json(
            raw,
            [
                [int(frame), index, *map(float, box), float(score)]
                for index, (frame, box, score) in enumerate(zip(source_frames, boxes, scores))
            ],
        )
        association = work / "associations.txt"
        association.write_bytes(mot_text(observed).encode())
        bindings = dict(
            schema="holdtrack-rgb-stream-v1",
            voice=voice,
            video=directory.name,
            width=width,
            height=height,
            frame_count=len(images),
            images=[dict(path=str(path.resolve()), sha256=file_hash(path)) for path in images],
            raw_detections=dict(path=str(raw), sha256=file_hash(raw)),
            bytetrack=dict(path=str(association), sha256=file_hash(association)),
            sam3_root=str(sources.sam3_root),
            sam2_predictor=str(sources.sam2_predictor),
            checkpoint=dict(path=str(checkpoint.resolve()), sha256=weights["sha256"]),
            device=device,
        )
        path = work / "inputs.json"
        atomic_json(path, bindings)
        result = _run_prepared(
            path,
            work / "trajectory",
            on_frame=writer.frame if writer else None,
            on_memory=writer.memory if writer else None,
        )
        return Path(result["mot"]).read_bytes()
