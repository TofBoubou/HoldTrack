"""Relay in place: an identity that ends and a new identity that starts shortly after at
the same place are merged. No appearance is read.

The relay is computed on the complete system: track rows and attached rows, gaps of one
or two frames filled, identities of fewer than four boxes dropped.
"""

from __future__ import annotations

import numpy as np

from .assemble import gaps

MAX_GAP_S = (
    5.0  # seconds between the last box of the first identity and the first box of the second
)
MAX_DISTANCE = 0.25  # distance of the two box centers, in mean box heights
MIN_LIFE_S = 1.5  # seconds of boxes that each identity must have
START_S = 0.25  # the second identity must start after this time


def system(frames, identities, boxes, attached):
    """Frames, identities and boxes of the complete system."""
    a = np.asarray(attached, float).reshape(-1, 7)
    fr = np.concatenate([np.asarray(frames).astype(np.int64), a[:, 0].astype(np.int64)])
    ids = np.concatenate(
        [np.asarray(identities).astype(np.int64).ravel(), a[:, 1].astype(np.int64)]
    )
    b = np.concatenate([np.asarray(boxes, float), a[:, 2:6].astype(float)])
    keep = np.ones(len(fr), bool)
    seen = set()
    for n, key in enumerate(zip(fr.tolist(), ids.tolist())):
        if key in seen:
            keep[n] = False
        seen.add(key)
    fr, ids, b = fr[keep], ids[keep], b[keep]
    f_new, i_new, b_new, _ = gaps(fr, ids, b)
    fr, ids, b = (
        np.concatenate([fr, f_new]),
        np.concatenate([ids, i_new]),
        np.concatenate([b, b_new]),
    )
    _, inverse, count = np.unique(ids, return_inverse=True, return_counts=True)
    long_enough = count[inverse] >= 4
    return fr[long_enough], ids[long_enough], b[long_enough]


def pairs(frames, identities, boxes, fps: float = 20.0):
    """Pairs (distance, ended identity, new identity, gap, boxes of each, last frame, first frame)."""
    fr = np.asarray(frames).astype(int)
    last_frame = int(fr.max())
    max_gap, min_boxes, start = (
        int(round(MAX_GAP_S * fps)),
        int(round(MIN_LIFE_S * fps)),
        int(round(START_S * fps)),
    )
    info = {}
    for q in np.unique(identities):
        k = np.where(identities == q)[0]
        k = k[np.argsort(fr[k], kind="stable")]
        if len(k) >= min_boxes:
            info[int(q)] = (int(fr[k[0]]), int(fr[k[-1]]), boxes[k[0]], boxes[k[-1]], len(k))
    candidates = []
    for a, (_, a1, _, a_last, na) in info.items():
        if a1 >= last_frame - 1:
            continue
        for b, (b0, _, b_first, _, nb) in info.items():
            if b == a or b0 <= start or not (1 <= b0 - a1 <= max_gap):
                continue
            height = ((a_last[3] - a_last[1]) + (b_first[3] - b_first[1])) / 2
            distance = float(
                np.hypot(
                    (a_last[0] + a_last[2] - b_first[0] - b_first[2]) / 2,
                    (a_last[1] + a_last[3] - b_first[1] - b_first[3]) / 2,
                )
                / max(height, 1)
            )
            if distance <= MAX_DISTANCE:
                candidates.append((distance, a, b, b0 - a1, na, nb, a1, b0))
    candidates.sort()
    used_a, used_b, out = set(), set(), []
    for c in candidates:
        if c[1] in used_a or c[2] in used_b:
            continue
        used_a.add(c[1])
        used_b.add(c[2])
        out.append(c)
    return out


def relay(frames, identities, boxes, attached, excluded=(), fps: float = 20.0):
    """Identities of the track rows and attached rows after the relay, and the pairs
    applied. ``excluded`` holds pairs (ended identity, new identity) to leave apart."""
    ids = np.asarray(identities).astype(np.int64).ravel()
    a = np.array(attached, dtype=float).reshape(-1, 7)
    f_sys, i_sys, b_sys = system(frames, ids, boxes, a)
    found = [c for c in pairs(f_sys, i_sys, b_sys, fps) if (c[1], c[2]) not in set(excluded)]
    new, new_attached = ids.copy(), a.copy()
    for _, kept, merged, *_ in found:
        new[new == merged] = kept
        if len(new_attached):
            new_attached[new_attached[:, 1].astype(np.int64) == merged, 1] = kept
    return new, new_attached, found
