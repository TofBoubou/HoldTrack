"""Add a matched detector view with a fixed weight."""

from collections import defaultdict

import numpy as np

from holdtrack.fusion.constants import ANCHOR_WEIGHT


def mix(data, new, p, weight):
    from holdtrack import geometry as geo

    native = data["options"][0]
    anchor = data["options"][1]
    N = len(anchor)
    assert native.shape == anchor.shape == (N, 4)
    if weight == 0:
        return (native.copy(), dict(matched=0, changed=0, weight0_exact=True))
    output = native.copy()
    groups = defaultdict(list)
    matches = {}
    for i, frame_index in enumerate(data["frames"]):
        groups[int(frame_index)].append(i)
    scores = data["rf_scores"][data["query"].astype(int)]
    for frame, ii in groups.items():
        ai = np.array([i for i in ii if scores[i] >= p["anchor_floor"]], int)
        bi = np.flatnonzero(
            (new["frame"] == frame)
            & (new["score"] >= p["new_floor"])
            & (new["boxes"][:, 2:] > new["boxes"][:, :2]).all(1)
        )
        ai = geo.canonical(anchor, ai)
        bi = geo.canonical(new["boxes"], bi)
        left, right = geo.match(geo.boxes_iou(anchor[ai], new["boxes"][bi]), p["IoU"])
        for i, j in zip(ai[left], bi[right]):
            matches[int(i)] = new["boxes"][j]
    for i, box in matches.items():
        auxiliary = [z[i] for z in data["options"][2:] if np.isfinite(z[i]).all()]
        output[i] = np.sum(
            np.stack([ANCHOR_WEIGHT * anchor[i], *auxiliary, weight * box]),
            axis=0,
            dtype=np.float64,
        ) / (ANCHOR_WEIGHT + len(auxiliary) + weight)
    assert np.isfinite(output).all() and (output[:, 2:] > output[:, :2]).all()
    return (
        output,
        dict(
            matched=len(matches),
            changed=int(np.any(output != native, axis=1).sum()),
            unmatched_kept_exact=True,
        ),
    )
