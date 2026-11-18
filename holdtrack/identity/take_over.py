"""Step 11. When an identity takes over the body of another, the shorter run is the intruder.

Event: identity J has a box in frame t and none in t + 1; in t + 1 a box of another identity I covers the box of J
(IoU >= 0.7), and I is not simply going on (no box in t, or its box of t covers its box of t + 1 at less than 0.3).
The run of J before the event and the run of I after it are followed box to box (IoU >= 0.5). A run of at most 1.5
seconds, at least three times shorter than the other, takes the name of the other. Only the boxes are read.
"""

import collections

import numpy as np

from .geometry import frames as n_frames
from .geometry import xyxy

COVER_IOU = 0.7
CONTINUES_IOU = 0.3
RATIO = 3.0
RUN_MAX_S = 1.5
RUN_IOU = 0.5


def _iou(a, B):
    B = np.atleast_2d(B)
    iw = np.clip(np.minimum(a[2], B[:, 2]) - np.maximum(a[0], B[:, 0]), 0, None)
    ih = np.clip(np.minimum(a[3], B[:, 3]) - np.maximum(a[1], B[:, 1]), 0, None)
    i = iw * ih
    u = (a[2] - a[0]) * (a[3] - a[1]) + (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]) - i
    return i / np.maximum(u, 1e-9)


def take_over(video, identities):
    """Returns the new identities and the counts (events, runs renamed before and after, cancelled, boxes renamed)."""
    p = video.rows(identities)
    pf = p[:, 0].astype(int)
    pid = p[:, 1].astype(np.int64)
    box = xyxy(p)
    run_max = n_frames(RUN_MAX_S, video.fps)
    at = {(int(f), int(lab)): i for i, (f, lab) in enumerate(zip(pf, pid))}
    in_frame = collections.defaultdict(list)
    counts = collections.Counter()
    for i, f in enumerate(pf):
        in_frame[int(f)].append(i)

    def run(i, step):
        L = [i]
        lab = int(pid[i])
        f = int(pf[i])
        while True:
            n = at.get((f + step, lab))
            if n is None or _iou(box[L[-1]], box[n])[0] < RUN_IOU:
                break
            L.append(n)
            f += step
        return L

    proposed = {}
    for i in range(len(p)):
        t = int(pf[i])
        J = int(pid[i])
        if (t + 1, J) in at:
            continue
        K = in_frame.get(t + 1, [])
        if not K:
            continue
        for k, x in zip(K, _iou(box[i], box[K])):
            if x < COVER_IOU:
                continue
            id_i = int(pid[k])
            j = at.get((t, id_i))
            if j is not None and _iou(box[j], box[k])[0] >= CONTINUES_IOU:
                continue
            before = run(i, -1)
            after = run(k, +1)
            counts["events"] += 1
            if len(before) <= run_max and len(before) * RATIO <= len(after):
                for q in before:
                    proposed[q] = id_i
                counts["before"] += 1
            elif len(after) <= run_max and len(after) * RATIO <= len(before):
                for q in after:
                    proposed[q] = J
                counts["after"] += 1
    new = pid.copy()
    for q, m in proposed.items():
        new[q] = m
    for _ in range(30):
        key = pf.astype(np.int64) * 10**9 + new
        u, cnt = np.unique(key, return_counts=True)
        dup = set(u[cnt > 1].tolist())
        if not dup:
            break
        for q in list(proposed):
            if key[q] in dup:
                new[q] = pid[q]
                proposed.pop(q)
                counts["cancelled"] += 1
    counts["renamed"] = len(proposed)
    key = pf.astype(np.int64) * 10**9 + new
    assert len(np.unique(key)) == len(key) and (new > 0).all()
    return new, counts
