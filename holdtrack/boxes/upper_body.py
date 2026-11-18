"""Upper-body table of the track rows, from the 17 keypoints of the pose estimator.

Six columns per row: upper body readable (0 or 1), upper body alone (0 or 1), jump of
the anchor before this row (0 or 1), anchor x, anchor y, size.

  anchor    midpoint of the two shoulders, both with a score above 0.3
  size      max(0.6 x shoulder distance, 0.45 x distance from the shoulder midpoint to
            the hip midpoint when both hips have a score above 0.3, 0.08 x box height)
  readable  anchor defined, both shoulders with a score of at least 0.5, and at least
            two of the five face keypoints with a score above 0.3
  alone     readable, and no other row of the frame has its anchor closer than the size
  jump      between two consecutive frames of an identity, the anchor moves by at least
            the size
"""

from __future__ import annotations

import numpy as np


def upper_body(
    frames: np.ndarray, identities: np.ndarray, boxes: np.ndarray, keypoints: np.ndarray
) -> np.ndarray:
    fr = np.asarray(frames).astype(int)
    pid = np.asarray(identities).astype(int)
    b = np.asarray(boxes, float)
    kp = np.asarray(keypoints, float)
    n = len(fr)
    sc = kp[:, :, 2]
    defined = (sc[:, 5] > 0.3) & (sc[:, 6] > 0.3)
    ax = np.where(defined, (kp[:, 5, 0] + kp[:, 6, 0]) / 2, np.nan)
    ay = np.where(defined, (kp[:, 5, 1] + kp[:, 6, 1]) / 2, np.nan)
    shoulders = np.hypot(kp[:, 5, 0] - kp[:, 6, 0], kp[:, 5, 1] - kp[:, 6, 1])
    hips = (sc[:, 11] > 0.3) & (sc[:, 12] > 0.3)
    trunk = np.where(
        hips,
        np.hypot(ax - (kp[:, 11, 0] + kp[:, 12, 0]) / 2, ay - (kp[:, 11, 1] + kp[:, 12, 1]) / 2),
        0.0,
    )
    size = np.maximum(
        np.maximum(0.6 * shoulders, 0.45 * np.nan_to_num(trunk)), 0.08 * (b[:, 3] - b[:, 1])
    )
    readable = defined & (sc[:, 5] >= 0.5) & (sc[:, 6] >= 0.5) & ((sc[:, :5] > 0.3).sum(1) >= 2)
    alone = np.ones(n, bool)
    order = np.argsort(fr, kind="stable")
    _, start = np.unique(fr[order], return_index=True)
    start = list(start) + [n]
    for i in range(len(start) - 1):
        rows = order[start[i] : start[i + 1]]
        rows = rows[defined[rows]]
        if len(rows) < 2:
            continue
        d = np.hypot(ax[rows][:, None] - ax[rows][None], ay[rows][:, None] - ay[rows][None])
        np.fill_diagonal(d, np.inf)
        alone[rows] = d.min(1) >= size[rows]
    jump = np.zeros(n, bool)
    for q in np.unique(pid):
        rows = np.where(pid == q)[0]
        rows = rows[np.argsort(fr[rows])]
        consecutive = np.diff(fr[rows]) == 1
        move = np.hypot(np.diff(ax[rows]), np.diff(ay[rows]))
        s = np.zeros(len(rows), bool)
        s[1:] = consecutive & np.isfinite(move) & (move >= size[rows][1:])
        jump[rows] = s
    return np.stack([readable, readable & alone, jump, ax, ay, size], 1).astype(np.float32)
