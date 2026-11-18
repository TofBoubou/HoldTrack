"""Keypoints of each tracked person, from the pixels, by two public pose estimators.

``keypoints17``   ViTPose+ (17 body keypoints, COCO order), on a crop of aspect 0.75
                  around the box, enlarged by 1.25, resampled to 192 x 256; the heatmaps
                  are averaged with those of the mirrored crop.
``keypoints133``  RTMW-L (body, feet, face, hands; official ONNX file), on a crop that
                  widens the box by 30% on each side and by 10% above and below,
                  resampled to 288 x 384.
Both take the frames (numbered from 1, in the order of ``images``) and the boxes of the
track rows, and return (N, K, 3): x and y in pixels of the frame, and a score. Parts of a
crop that fall outside the frame are black. No annotation is read.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

MIRRORED_17 = (0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15)


def _crops(images, frames, boxes, window, width: int, height: int, batch: int):
    """Batches of (rows, crops as uint8, windows) in the order of the frames."""
    frames = np.asarray(frames).astype(int)
    order = np.argsort(frames, kind="stable")
    values, start = np.unique(frames[order], return_index=True)
    start = list(start) + [len(frames)]
    rows, crops, windows = [], [], []
    for k, frame in enumerate(values):
        with Image.open(images[int(frame) - 1]) as original:
            image = original.convert("RGB")
        for j in order[start[k] : start[k + 1]]:
            x0, y0, w, h = window(boxes[j])
            crops.append(
                np.asarray(
                    image.transform(
                        (width, height), Image.EXTENT, (x0, y0, x0 + w, y0 + h), Image.BILINEAR
                    )
                )
            )
            rows.append(j)
            windows.append((x0, y0, w, h))
        if len(rows) >= batch:
            yield np.array(rows), np.stack(crops), np.array(windows, np.float32)
            rows, crops, windows = [], [], []
    if rows:
        yield np.array(rows), np.stack(crops), np.array(windows, np.float32)


def _window17(b):
    cx, cy, w, h = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1]
    if w > 0.75 * h:
        h = w / 0.75
    else:
        w = 0.75 * h
    w, h = 1.25 * w, 1.25 * h
    return float(cx - w / 2), float(cy - h / 2), float(w), float(h)


def _window133(b, width=288, height=384):
    w, h = b[2] - b[0], b[3] - b[1]
    x0, x1, y0, y1 = b[0] - 0.3 * w, b[2] + 0.3 * w, b[1] - 0.1 * h, b[3] + 0.1 * h
    cw, ch, cx, cy = x1 - x0, y1 - y0, (x0 + x1) / 2, (y0 + y1) / 2
    if cw / ch > width / height:
        ch = cw * height / width
    else:
        cw = ch * width / height
    return float(cx - cw / 2), float(cy - ch / 2), float(cw), float(ch)


def keypoints17(
    images: list[Path],
    frames,
    boxes,
    model: str = "usyd-community/vitpose-plus-huge",
    mirror: bool = True,
    batch: int = 96,
    device: str = "cuda",
) -> np.ndarray:
    import torch
    from transformers import AutoImageProcessor, VitPoseForPoseEstimation

    net = VitPoseForPoseEstimation.from_pretrained(model).eval().to(device)
    height, width = net.config.backbone_config.image_size
    experts = int(getattr(net.config.backbone_config, "num_experts", 1))
    processor = AutoImageProcessor.from_pretrained(model)
    mean = torch.tensor(list(processor.image_mean), device=device).view(1, 3, 1, 1)
    std = torch.tensor(list(processor.image_std), device=device).view(1, 3, 1, 1)
    swap = torch.tensor(MIRRORED_17, device=device)
    boxes = np.asarray(boxes, np.float32).astype(np.float64)
    out = np.full((len(boxes), 17, 3), np.nan, np.float32)
    for rows, crops, windows in _crops(images, frames, boxes, _window17, width, height, batch):
        x = torch.from_numpy(crops).to(device).permute(0, 3, 1, 2).float().div_(255.0)
        x = (x - mean) / std
        n = len(x)
        index = (
            torch.zeros(2 * n if mirror else n, dtype=torch.long, device=device)
            if experts > 1
            else None
        )
        with (
            torch.inference_mode(),
            torch.autocast(device, dtype=torch.bfloat16, enabled=(device == "cuda")),
        ):
            maps = net(
                pixel_values=torch.cat([x, x.flip(3)]) if mirror else x, dataset_index=index
            ).heatmaps.float()
        if mirror:
            maps = 0.5 * (maps[:n] + maps[n:].flip(3)[:, swap])
        k, h, w = maps.shape[1:]
        flat = maps.reshape(n, k, -1)
        value, at = flat.max(2)
        py, px = at // w, at % w

        def pick(yy, xx):
            return flat.gather(
                2, (yy.clamp(0, h - 1) * w + xx.clamp(0, w - 1)).unsqueeze(2)
            ).squeeze(2)

        dx = 0.25 * torch.sign(pick(py, px + 1) - pick(py, px - 1)) * ((px > 0) & (px < w - 1))
        dy = 0.25 * torch.sign(pick(py + 1, px) - pick(py - 1, px)) * ((py > 0) & (py < h - 1))
        win = torch.from_numpy(windows).to(device)
        xs = win[:, 0:1] + (px + dx + 0.5) * win[:, 2:3] / w
        ys = win[:, 1:2] + (py + dy + 0.5) * win[:, 3:4] / h
        out[rows] = torch.stack([xs, ys, value], 2).cpu().numpy().astype(np.float32)
    return out


def keypoints133(images: list[Path], frames, boxes, onnx: Path, batch: int = 64) -> np.ndarray:
    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    session = ort.InferenceSession(
        str(onnx), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    width, height = 288, 384
    entry = session.get_inputs()[0]
    dynamic = not isinstance(entry.shape[0], int)
    names = [o.name for o in session.get_outputs()]
    mean = np.array([123.675, 116.28, 103.53], np.float32)
    std = np.array([58.395, 57.12, 57.375], np.float32)
    boxes = np.asarray(boxes, np.float32).astype(np.float64)
    out = np.full((len(boxes), 133, 3), np.nan, np.float32)
    for rows, crops, windows in _crops(images, frames, boxes, _window133, width, height, batch):
        x = np.ascontiguousarray(((crops.astype(np.float32) - mean) / std).transpose(0, 3, 1, 2))
        if dynamic:
            a, b = session.run(names, {entry.name: x})
        else:
            parts = [session.run(names, {entry.name: x[i : i + 1]}) for i in range(len(x))]
            a, b = np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
        sx, sy = (a, b) if a.shape[-1] == width * 2 else (b, a)
        px, py = sx.argmax(2) / 2.0, sy.argmax(2) / 2.0
        score = np.minimum(sx.max(2), sy.max(2))
        out[rows] = np.stack(
            [
                windows[:, 0:1] + px * windows[:, 2:3] / width,
                windows[:, 1:2] + py * windows[:, 3:4] / height,
                score,
            ],
            2,
        ).astype(np.float32)
    return out
