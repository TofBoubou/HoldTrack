"""Smooth the four edges of each box along its identity.

Some benchmarks annotate a box at key frames and interpolate between them, so an annotated
edge follows straight segments in time. A detector box carries a jitter from frame to frame
that such an annotation does not have. Each edge is replaced by the value, at the frame of
the row, of a line fitted to that edge over the rows of the same identity within `window`
seconds, with weights that decrease linearly with the distance in time. Where the window
is incomplete, at the ends of a track or beside a gap, a line fitted locally has no bias on
a person who moves, which a moving average has.
"""

import numpy as np


def smooth_edges(frames, identities, boxes, fps, window=0.2, minimum=3):
    """Return the boxes (N, 4), as x1, y1, x2, y2, with each edge smoothed along its identity.

    A row that has fewer than `minimum` rows of its identity inside the window keeps its box.
    """
    frames = np.asarray(frames).astype(np.int64)
    identities = np.asarray(identities).astype(np.int64)
    boxes = np.asarray(boxes, float)
    half_window = max(1, int(round(window * fps)))
    out = boxes.copy()
    order = np.lexsort((frames, identities))
    sorted_identities = identities[order]
    bounds = np.flatnonzero(np.r_[True, sorted_identities[1:] != sorted_identities[:-1], True])
    for start, end in zip(bounds[:-1], bounds[1:]):
        rows = order[start:end]
        if len(rows) < minimum:
            continue
        time = frames[rows].astype(float)
        delta = time[None, :] - time[:, None]
        weight = np.clip(1.0 - np.abs(delta) / (half_window + 1.0), 0.0, None)
        count = (weight > 0).sum(1)
        s0 = weight.sum(1)
        s1 = (weight * delta).sum(1)
        s2 = (weight * delta * delta).sum(1)
        sy = weight @ boxes[rows]
        sty = (weight * delta) @ boxes[rows]
        denominator = s0 * s2 - s1 * s1
        fitted = (count >= minimum) & (denominator > 1e-9)
        out[rows[fitted]] = (
            s2[fitted, None] * sy[fitted] - s1[fitted, None] * sty[fitted]
        ) / denominator[fitted, None]
    return out
