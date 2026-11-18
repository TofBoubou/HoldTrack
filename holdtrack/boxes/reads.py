"""Reads of a video by the person detector, from the pixels, for the box corrector.

The detector (RF-DETR, strict load, native postprocessing, no added NMS) reads each frame
several times: whole frame at a chosen input size, mirrored, by vertical bands, and on a
crop around each tracked box. Every read returns boxes in the coordinates of the original
frame. Frames are numbered from 1, in the order of ``images``. No annotation is read.

The corrector of the paper uses, in this order (see ``table.build_table``):
  whole frame            RF-DETR at 1120; RF-DETR at 1120, mirrored; YOLO; YOLO, mirrored;
                         D-FINE; RF-DETR at 1280
  bands                  three bands of 0.6 of the width at 1120 (left edges 0, 0.2, 0.4);
                         five bands of 0.4 of the width at 1280 (left edges 0 to 0.6 by 0.15)
  crops                  height 1.25 x the box, at 1120; height 1.7 x the box, mirrored, at 1120
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

BANDS_3 = ((0.0, 0.6), (0.2, 0.6), (0.4, 0.6))
BANDS_5 = ((0.0, 0.4), (0.15, 0.4), (0.3, 0.4), (0.45, 0.4), (0.6, 0.4))


def window(box, width: int, height: int, margin: float, largest: float = 4.0):
    """Crop with the aspect of the frame, centered on the box, ``margin`` times its
    height (at least height / largest, at most the frame), wide enough for the box."""
    h, w = box[3] - box[1], box[2] - box[0]
    crop_h = min(height, max(margin * h, height / largest, 1.3 * w * height / width))
    crop_w = crop_h * width / height
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    x0 = min(max(cx - crop_w / 2, 0), width - crop_w)
    y0 = min(max(cy - crop_h / 2, 0), height - crop_h)
    return int(round(x0)), int(round(y0)), int(round(x0 + crop_w)), int(round(y0 + crop_h))


class Reader:
    """RF-DETR person detector, ready to read frames, bands and crops."""

    def __init__(self, checkpoint: Path, specification: dict, device: str = "cuda"):
        import torch
        from rfdetr import RFDETR2XLarge
        from rfdetr.config import RFDETRLargeDeprecatedConfig, TrainConfig
        from rfdetr.datasets.coco import make_coco_transforms_square_div_64
        from rfdetr.detr import _namespace_from_configs
        from rfdetr.models.lwdetr import build_criterion_and_postprocessors, build_model

        configuration_class = (
            RFDETRLargeDeprecatedConfig
            if specification["architecture"] == "RFDETRLargeDeprecated"
            else RFDETR2XLarge._model_config_class
        )
        self.configuration = configuration_class(
            **dict(specification["model_config"], device=device, pretrain_weights=None)
        )
        training = TrainConfig(
            dataset_dir=".",
            output_dir=".",
            class_names=["person"],
            multi_scale=False,
            tensorboard=False,
            wandb=False,
        )
        namespace = _namespace_from_configs(self.configuration, training)
        namespace.pretrained_encoder = False
        self.model = build_model(namespace)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        self.model.load_state_dict(state, strict=True)
        del state
        self.model.eval().requires_grad_(False).to(device)
        self.postprocessor = build_criterion_and_postprocessors(namespace)[1].eval()
        self.device = device
        self.torch = torch
        self._make = make_coco_transforms_square_div_64
        self._transforms = {}
        self.skipped = 0

    def _tensor(self, image: Image.Image, resolution: int):
        torch = self.torch
        if resolution not in self._transforms:
            self._transforms[resolution] = self._make(
                "val",
                resolution,
                multi_scale=False,
                patch_size=self.configuration.patch_size,
                num_windows=self.configuration.num_windows,
            )
        width, height = image.size
        target = dict(
            boxes=torch.empty(0, 4),
            labels=torch.empty(0, dtype=torch.long),
            area=torch.empty(0),
            iscrowd=torch.empty(0, dtype=torch.long),
            orig_size=torch.tensor([height, width]),
            size=torch.tensor([height, width]),
        )
        tensor, _ = self._transforms[resolution](image, target)
        if tuple(tensor.shape) != (3, resolution, resolution):
            raise ValueError("RF-DETR preprocessing produced an unexpected shape")
        return tensor

    def detect(
        self, crops: list[Image.Image], resolution: int, floor: float = 0.05, batch: int = 4
    ):
        """Boxes and scores of the persons of each crop, in the coordinates of the crop."""
        torch = self.torch
        out = []
        with torch.inference_mode():
            for start in range(0, len(crops), batch):
                part = crops[start : start + batch]
                tensors = torch.stack([self._tensor(c, resolution) for c in part]).to(self.device)
                sizes = torch.tensor([[c.size[1], c.size[0]] for c in part], device=self.device)
                for restored in self.postprocessor(self.model(tensors), sizes):
                    if not (
                        torch.isfinite(restored["scores"]).all()
                        and torch.isfinite(restored["boxes"]).all()
                    ):
                        self.skipped += 1
                        out.append((np.zeros((0, 4), np.float32), np.zeros(0, np.float32)))
                        continue
                    keep = (restored["labels"] == 0) & (restored["scores"] >= floor)
                    out.append(
                        (
                            restored["boxes"][keep].float().cpu().numpy(),
                            restored["scores"][keep].float().cpu().numpy(),
                        )
                    )
        return out

    def full(
        self,
        images: list[Path],
        resolution: int | None = None,
        mirrored: bool = False,
        floor: float = 0.05,
        batch: int = 4,
    ):
        """Read of the whole frames: frames, boxes, scores."""
        resolution = resolution or self.configuration.resolution
        frames, boxes, scores = [], [], []
        for start in range(0, len(images), batch):
            part = []
            for path in images[start : start + batch]:
                with Image.open(path) as original:
                    image = original.convert("RGB")
                part.append(image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if mirrored else image)
            for offset, (image, (b, s)) in enumerate(
                zip(part, self.detect(part, resolution, floor, batch))
            ):
                if mirrored:
                    b = np.stack(
                        [image.size[0] - b[:, 2], b[:, 1], image.size[0] - b[:, 0], b[:, 3]], 1
                    )
                frames.append(np.full(len(b), start + offset + 1, np.int32))
                boxes.append(b)
                scores.append(s)
        return (
            np.concatenate(frames),
            np.concatenate(boxes).astype(np.float32),
            np.concatenate(scores).astype(np.float32),
        )

    def bands(
        self,
        images: list[Path],
        fractions=BANDS_3,
        resolution: int | None = None,
        floor: float = 0.05,
        batch: int = 4,
    ):
        """Read by vertical bands (left edge and width as shares of the frame width):
        list of (left, right, frames, boxes, scores), as ``table.match_bands`` takes it."""
        resolution = resolution or self.configuration.resolution
        store = [([], [], []) for _ in fractions]
        edges = None
        for index, path in enumerate(images):
            with Image.open(path) as original:
                image = original.convert("RGB")
            width, height = image.size
            edges = [
                (int(round(f0 * width)), min(width, int(round((f0 + fw) * width))))
                for f0, fw in fractions
            ]
            crops = [image.crop((x0, 0, x1, height)) for x0, x1 in edges]
            for k, ((x0, _), (b, s)) in enumerate(
                zip(edges, self.detect(crops, resolution, floor, batch))
            ):
                b = b.copy()
                b[:, [0, 2]] += float(x0)
                store[k][0].append(np.full(len(b), index + 1, np.int32))
                store[k][1].append(b)
                store[k][2].append(s)
        return [
            (
                float(x0),
                float(x1),
                np.concatenate(f),
                np.concatenate(b).astype(np.float32),
                np.concatenate(s).astype(np.float32),
            )
            for (x0, x1), (f, b, s) in zip(edges, store)
        ]

    def crops(
        self,
        images: list[Path],
        frames,
        boxes,
        margin: float = 1.25,
        mirrored: bool = False,
        resolution: int | None = None,
        floor: float = 0.05,
        batch: int = 8,
        largest: float = 4.0,
    ):
        """Read on a crop around each tracked box: for each row, the detection with the
        largest IoU with the box (NaN under 0.3), this IoU and its score, as
        ``table.row_view`` takes them."""
        resolution = resolution or self.configuration.resolution
        frames = np.asarray(frames).astype(int)
        ref = np.asarray(boxes, np.float32)
        n = len(frames)
        box, iou, score = (
            np.full((n, 4), np.nan, np.float32),
            np.zeros(n, np.float32),
            np.zeros(n, np.float32),
        )
        order = np.argsort(frames, kind="stable")
        values, start = np.unique(frames[order], return_index=True)
        start = list(start) + [n]
        for k, frame in enumerate(values):
            rows = order[start[k] : start[k + 1]]
            with Image.open(images[int(frame) - 1]) as original:
                image = original.convert("RGB")
            width, height = image.size
            windows = [window(ref[i].astype(float), width, height, margin, largest) for i in rows]
            parts = [image.crop(w) for w in windows]
            sizes = [p.size for p in parts]
            if mirrored:
                parts = [p.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for p in parts]
            for i, w, (cw, _), (b, s) in zip(
                rows, windows, sizes, self.detect(parts, resolution, floor, batch)
            ):
                b = b.astype(np.float64)
                if len(b) == 0:
                    continue
                if mirrored:
                    b = np.stack([cw - b[:, 2], b[:, 1], cw - b[:, 0], b[:, 3]], 1)
                b[:, [0, 2]] += w[0]
                b[:, [1, 3]] += w[1]
                a = ref[i].astype(float)
                iw = np.clip(np.minimum(a[2], b[:, 2]) - np.maximum(a[0], b[:, 0]), 0, None)
                ih = np.clip(np.minimum(a[3], b[:, 3]) - np.maximum(a[1], b[:, 1]), 0, None)
                inter = iw * ih
                io = inter / (
                    (a[2] - a[0]) * (a[3] - a[1])
                    + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
                    - inter
                    + 1e-9
                )
                m = int(io.argmax())
                if io[m] >= 0.3:
                    box[i], iou[i], score[i] = b[m], io[m], s[m]
        return box, iou, score
