"""Boxes and time."""

import numpy as np

REFERENCE_FPS = 20.0


def frames(seconds, fps=REFERENCE_FPS):
    """Number of frames of a duration. The durations of the chain were set on videos at 20 frames per second."""
    return int(round(seconds * fps))


def xyxy(rows):
    """Corners of rows whose columns 2 to 5 are x, y, width, height."""
    return np.stack([rows[:, 2], rows[:, 3], rows[:, 2] + rows[:, 4], rows[:, 3] + rows[:, 5]], 1)


def iou_matrix(a, b):
    w = np.clip(
        np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None
    )
    h = np.clip(
        np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None
    )
    inter = w * h
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None] - inter + 1e-9)


def overlap_with_others(frame, boxes):
    """Largest IoU of each box with another box of its frame."""
    out = np.zeros(len(frame))
    for f in np.unique(frame):
        idx = np.where(frame == f)[0]
        if len(idx) > 1:
            m = iou_matrix(boxes[idx], boxes[idx])
            np.fill_diagonal(m, 0)
            out[idx] = m.max(1)
    return out
