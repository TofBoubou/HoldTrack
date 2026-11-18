"""Reverse or mirror the input of a pass, and put its output back in the frame order and view of the video."""

from __future__ import annotations

import numpy as np

from holdtrack.detect.bank import DetectionBank
from holdtrack.inputs.mot import physical


def reverse_bank(bank: DetectionBank, frame_count: int) -> tuple[DetectionBank, np.ndarray]:
    """Reverse the clock; keep proposal ordering within each original image."""
    if frame_count < 1 or (bank.frames >= frame_count).any():
        raise ValueError("Detections fall outside the video clock")
    frames = frame_count - 1 - bank.frames
    indices = np.argsort(frames, kind="stable")
    return DetectionBank(frames[indices], bank.boxes[indices], bank.scores[indices]), indices


def reflect_boxes(boxes: np.ndarray, width: int) -> np.ndarray:
    boxes = np.asarray(boxes)
    if width <= 0 or boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("Expected xyxy boxes and positive image width")
    if not np.isfinite(boxes).all() or (boxes[:, 2:] < boxes[:, :2]).any():
        raise ValueError("Invalid native boxes")
    result = boxes.copy()
    result[:, 0] = width - boxes[:, 2]
    result[:, 2] = width - boxes[:, 0]
    return result


def reflect_mot(payload: bytes, width: int) -> bytes:
    """Reflect x only; retain native frame/identity/score/other tokens."""
    chunks = []
    for line, row in physical(payload):
        columns = line.rstrip(b"\r\n").split(b",")
        # Order of the operations: (W - x) - w, not W - (x + w).
        columns[2] = f"{width - float(columns[2]) - float(columns[4]):.8f}".encode()
        chunks.append(b",".join(columns) + line[len(line.rstrip(b"\r\n")) :])
    return b"".join(chunks)


def restore_mirrored_support(reference: bytes, mirrored: bytes, width: int) -> bytes:
    """Use mirror identities and exact original bytes for every other field.

    Reflection reuses ByteTrack, so rows must correspond by physical index.
    A changed row, score or geometry fails; there is no IoU rematching here.
    """
    a, b = reference.splitlines(keepends=True), mirrored.splitlines(keepends=True)
    if len(a) != len(b):
        raise ValueError("Mirror physical support differs")
    result = []
    for i, (original, changed) in enumerate(zip(a, b, strict=True)):
        x = original.rstrip(b"\r\n").split(b",")
        y = changed.rstrip(b"\r\n").split(b",")
        if len(x) != len(y) or x[0] != y[0] or x[6:] != y[6:]:
            raise ValueError(f"Mirror native row differs at {i}")

        def box(columns):
            left, top, w, h = map(float, columns[2:6])
            return np.array([left, top, left + w, top + h])

        expected = reflect_boxes(box(x)[None], width)[0]
        if not np.allclose(expected, box(y), rtol=0, atol=2e-7):
            raise ValueError(f"Mirror native coordinates differ at {i}")
        if int(y[1]) < 1:
            raise ValueError("Mirror identity must be positive")
        x[1] = y[1]
        result.append(b",".join(x) + original[len(original.rstrip(b"\r\n")) :])
    payload = b"".join(result)
    physical(payload)
    return payload


def restore_reverse_clock(payload: bytes, frame_count: int) -> bytes:
    """Return one-based original frame numbers, sorted by frame and identity."""
    rows = []
    for line, row in physical(payload):
        if not 0 <= row[0] < frame_count:
            raise ValueError("Reverse MOT falls outside the video clock")
        frame = frame_count - row[0]
        _, rest = line.split(b",", 1)
        rows.append((frame, row[1], str(frame).encode() + b"," + rest))
    return b"".join(line for _, _, line in sorted(rows))
