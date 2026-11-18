"""Parse track observations while preserving serialized tokens."""

import numpy as np


def physical(payload):
    result = []
    seen = set()
    for line in payload.splitlines(keepends=True):
        columns = line.strip().split(b",")
        frame, tid = (int(columns[0]) - 1, int(columns[1]))
        x, y, w, h = map(float, columns[2:6])
        if (
            (frame, tid) in seen
            or tid <= 0
            or (not np.isfinite([x, y, w, h]).all())
            or (w <= 0)
            or (h <= 0)
        ):
            raise ValueError("Invalid MOT physical observation")
        seen.add((frame, tid))
        result.append((line, (frame, tid, x, y, x + w, y + h)))
    return result
