"""Step 1. Merge whole identities that the appearance classifier of the video names as one person.

The classifier is trained on the video itself, one class per identity of the base tracker, and predicts each crop
out of block. An identity A takes the name of B when A and B never share a frame, A has enough crops, and B is the
best class of A other than A by a margin over the second.
"""

import collections

import numpy as np

from .segments import clean_segments

MARGIN = 0.0
MIN_CROPS = 5
CALIBRATION_MIN_CROPS = 50


def calibrate(log_probs, labels, classes, in_segment):
    """Temperature that minimises the cross-entropy between the classifier and the identities of the base tracker,
    on the crops of clean segments. Returns the calibrated log-probabilities and the temperature (None if too few)."""
    use = (~np.isnan(log_probs[:, 0])) & in_segment & np.isin(labels, classes)
    if use.sum() < CALIBRATION_MIN_CROPS:
        return log_probs, None
    y = np.searchsorted(classes, labels[use])
    best = (1e9, 1.0)
    for t in np.geomspace(0.5, 60, 48):
        z = log_probs[use] / t
        z = (
            z
            - np.log(np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True))
            - z.max(1, keepdims=True)
        )
        nll = -z[np.arange(len(y)), y].mean()
        if nll < best[0]:
            best = (nll, t)
    t = best[1]
    z = log_probs / t
    ok = ~np.isnan(z[:, 0])
    zz = z[ok]
    zz = (
        zz
        - np.log(np.exp(zz - zz.max(1, keepdims=True)).sum(1, keepdims=True))
        - zz.max(1, keepdims=True)
    )
    out = log_probs.copy()
    out[ok] = zz
    return out, float(t)


def merge(rows, crops_meta, log_probs, classes, margin=MARGIN, min_crops=MIN_CROPS, fps=20.0):
    """rows: frame, identity, x, y, width, height (one per box). crops_meta: one row per crop, columns frame,
    identity, x, y, width, height, overlap, segment, segment at 0.3, row of the box. log_probs: (crops, classes),
    NaN where the classifier gives nothing. Returns the new identities, the number of merges and the temperature."""
    frame = rows[:, 0].astype(int)
    identity = rows[:, 1].astype(int)
    if log_probs.shape[1] == 0:  # a short clip: no identity has enough crops for the classifier
        return identity.copy(), 0, None
    segment, found, _ = clean_segments(rows, 0.5, fps)
    log_probs = log_probs.astype(np.float32)
    classes = classes.astype(int)
    crop_segment = segment[crops_meta[:, 9].astype(int)]
    crop_label = crops_meta[:, 1].astype(int)
    assert (
        len(crops_meta) == len(log_probs) and (crops_meta[:, 7].astype(int) == crop_segment).all()
    ), "crops do not match the tracks"
    log_probs, temperature = calibrate(log_probs, crop_label, classes, crop_segment >= 0)
    ok = ~np.isnan(log_probs[:, 0])
    by_identity = collections.defaultdict(list)
    for k in range(len(found)):
        m = (crop_segment == k) & ok
        n = int(m.sum())
        if n:
            by_identity[found[k]["identity"]].append((n, log_probs[m].mean(0)))
    present = collections.defaultdict(set)
    for f, name in zip(frame, identity):
        present[int(name)].add(int(f))
    candidates = []
    for a, parts in by_identity.items():
        n = sum(p[0] for p in parts)
        if n < min_crops:
            continue
        score = sum(p[0] * p[1] for p in parts) / n
        order = [j for j in np.argsort(-score) if int(classes[j]) != a]
        if len(order) < 2:
            continue
        candidates.append((float(score[order[0]] - score[order[1]]), a, int(classes[order[0]])))
    new = identity.copy()
    merges = 0
    renamed = {}
    for gap, a, b in sorted(candidates, reverse=True):
        if gap < margin:
            break
        while b in renamed:
            b = renamed[b]
        if b == a or (present[a] & present[b]):
            continue
        new[new == a] = b
        present[b] |= present[a]
        renamed[a] = b
        merges += 1
    return new, merges, temperature
