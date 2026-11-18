"""Run YOLO26 and its mirrored detection view."""

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from holdtrack.detect.bank import DetectionBank


def predict(
    images: list[Path], checkpoint: Path, *, mirrored: bool = False, device: str = "cuda"
) -> DetectionBank:
    """Keep the one-to-one person head and restore mirrored source coordinates."""
    import torch
    from ultralytics import YOLO

    detector = YOLO("yolo26x.yaml", task="detect")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    detector.model.load_state_dict(state, strict=True)
    del state
    detector.model.eval().requires_grad_(False)
    if not detector.model.model[-1].end2end:
        raise ValueError("The YOLO26 detector must use its one-to-one head")
    frame_indices, boxes, scores = ([], [], [])
    options = dict(
        end2end=True,
        imgsz=1280,
        conf=0.01,
        iou=0.7,
        max_det=300,
        classes=[0],
        device=device,
        half=False,
        rect=False,
        batch=1,
        workers=0,
        augment=False,
        agnostic_nms=False,
        vid_stride=1,
        save=False,
        verbose=False,
    )

    def mirrored_results():
        for frame, path in enumerate(images):
            with Image.open(path) as original:
                result = detector.predict(
                    source=ImageOps.mirror(original.convert("RGB")), **options
                )[0]
            yield (frame, result)

    with tempfile.TemporaryDirectory(prefix="holdtrack-images-") as temporary:
        source = Path(temporary) / "frames.txt"
        source.write_text("".join(str(path.resolve()) + "\n" for path in images))
        expected = {str(path.resolve()): frame for frame, path in enumerate(images)}
        seen = set()
        with torch.inference_mode():
            results = (
                mirrored_results()
                if mirrored
                else (
                    (expected[str(Path(result.path).resolve())], result)
                    for result in detector.predict(source=str(source), stream=True, **options)
                )
            )
            for frame, result in results:
                if frame in seen:
                    raise ValueError("YOLO returned an image twice")
                seen.add(frame)
                prediction_boxes = result.boxes.xyxy.float().cpu().numpy()
                prediction_scores = result.boxes.conf.float().cpu().numpy()
                order = np.argsort(-prediction_scores, kind="stable")
                prediction_boxes, prediction_scores = (
                    prediction_boxes[order],
                    prediction_scores[order],
                )
                height, width = result.orig_shape
                if mirrored:
                    prediction_boxes = prediction_boxes.copy()
                    prediction_boxes[:, [0, 2]] = width - prediction_boxes[:, [2, 0]]
                valid = (
                    (prediction_boxes[:, 2:] > prediction_boxes[:, :2]).all(1)
                    & (
                        np.minimum(prediction_boxes[:, 2], width)
                        > np.maximum(prediction_boxes[:, 0], 0)
                    )
                    & (
                        np.minimum(prediction_boxes[:, 3], height)
                        > np.maximum(prediction_boxes[:, 1], 0)
                    )
                )
                boxes.append(prediction_boxes[valid])
                scores.append(prediction_scores[valid])
                frame_indices.append(np.full(int(valid.sum()), frame, np.int64))
        if seen != set(range(len(images))):
            raise ValueError("YOLO omitted an input image")
    return DetectionBank(
        np.concatenate(frame_indices), np.concatenate(boxes), np.concatenate(scores)
    )
