"""Short tracks that the base system removes, offered to the identity chain.

A candidate is an identity of the fusion output, before the track-level steps, of at
least four boxes, none of which is found in the final tracks (same frame, IoU of at
least 0.9). Output: (M, 6) = frame, candidate track number, left, top, right, bottom.
"""

from __future__ import annotations

import numpy as np

from .table import iou_matrix


def short_tracks(fusion: np.ndarray, frames: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """fusion: MOT rows of the fusion output (frame, identity, left, top, width, height)."""
    q = np.asarray(fusion, float)
    ff = q[:, 0].astype(np.int64)
    fi = q[:, 1].astype(np.int64)
    fb = np.stack([q[:, 2], q[:, 3], q[:, 2] + q[:, 4], q[:, 3] + q[:, 5]], 1)
    pr = np.asarray(frames).astype(np.int64)
    pb = np.asarray(boxes, float)
    seen = np.zeros(len(ff), bool)
    rows: dict[int, list[int]] = {}
    fusion_rows: dict[int, list[int]] = {}
    for i, x in enumerate(pr):
        rows.setdefault(int(x), []).append(i)
    for i, x in enumerate(ff):
        fusion_rows.setdefault(int(x), []).append(i)
    for x, ii in fusion_rows.items():
        jj = rows.get(x, [])
        if jj:
            seen[ii] = iou_matrix(fb[ii], pb[jj]).max(1) >= 0.9
    out = []
    for identity in np.unique(fi):
        s = np.where(fi == identity)[0]
        if seen[s].any() or len(s) < 4:
            continue
        s = s[np.argsort(ff[s])]
        out.append(np.concatenate([ff[s][:, None], np.full((len(s), 1), identity), fb[s]], 1))
    return np.concatenate(out) if out else np.zeros((0, 6))
