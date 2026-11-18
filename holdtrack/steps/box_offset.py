"""Correct the systematic box offset."""

import numpy as np


def render(original, boxes):
    lines = original.splitlines(keepends=True)
    out = []
    changed = 0
    if len(lines) != len(boxes):
        raise ValueError("Fixed physical row support")
    for line, box in zip(lines, boxes):
        fields = line.rstrip(b"\r\n").split(b",")
        x, y, w, h = map(float, fields[2:6])
        old = np.array([x, y, x + w, y + h])
        if np.array_equal(box, old):
            out.append(line)
            continue
        if not np.isfinite(box).all() or not (box[2:] > box[:2]).all():
            raise ValueError("No inversion/zero extent;no clipping or epsilon")
        fields[2:6] = [
            f"{v:.8f}".encode() for v in [box[0], box[1], box[2] - box[0], box[3] - box[1]]
        ]
        ending = b"\r\n" if line.endswith(b"\r\n") else b"\n" if line.endswith(b"\n") else b""
        out.append(b",".join(fields) + ending)
        changed += 1
    payload = b"".join(out)
    if any(
        (
            a.split(b",")[:2] + a.split(b",")[6:] != b.split(b",")[:2] + b.split(b",")[6:]
            for a, b in zip(lines, payload.splitlines(keepends=True))
        )
    ):
        raise ValueError("Only geometry fields may change")
    return (payload, changed)
