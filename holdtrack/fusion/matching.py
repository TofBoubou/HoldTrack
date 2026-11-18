"""Align auxiliary detector boxes with track observations."""

from collections import defaultdict

import numpy as np


def grouping(frames):
    out = defaultdict(list)
    for i, frame_index in enumerate(frames):
        out[int(frame_index)].append(i)
    return {frame_index: np.asarray(x, int) for frame_index, x in out.items()}


def matched_links(anchor, frames, view, floor, geo):
    result = {}
    group = grouping(view["frames"])
    for frame_index, indices in grouping(frames).items():
        vi = group.get(frame_index, np.array([], int))
        vi = vi[view["scores"][vi] >= floor]
        aa = geo.canonical(anchor, indices)
        bb = geo.canonical(view["boxes"], vi)
        a, b = geo.match(geo.boxes_iou(anchor[aa], view["boxes"][bb]), 0.7)
        for x, y in zip(aa[a], bb[b]):
            result[int(x)] = view["boxes"][y]
    return result
