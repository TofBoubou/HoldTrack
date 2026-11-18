"""Fuse auxiliary boxes around the principal detector."""

from collections import defaultdict
from types import SimpleNamespace

import numpy as np

from holdtrack.fusion.constants import (
    ANCHOR_WEIGHT,
    DFINE_SCORE_MIN,
    VIEW_MATCH_IOU,
    YOLO_SCORE_MIN,
)
from holdtrack.inputs.mot import physical


def _ending(line):
    return b"\r\n" if line.endswith(b"\r\n") else b"\n" if line.endswith(b"\n") else b""


def unchanged_tokens(source, candidate):
    original, changed = (source.splitlines(keepends=True), candidate.splitlines(keepends=True))
    if len(original) != len(changed):
        raise ValueError("P2 added/dropped a physical line")
    for a, b in zip(original, changed):
        aa, bb = (a.rstrip(b"\r\n").split(b","), b.rstrip(b"\r\n").split(b","))
        if aa[:2] + aa[6:] != bb[:2] + bb[6:] or _ending(a) != _ending(b):
            raise ValueError("Frame/ID/nongeometry/order/line ending changed")
    physical(candidate)
    return dict(
        same_lines=True,
        same_frames=True,
        same_ID=True,
        same_non_geometry_tokens_and_order=True,
        same_line_endings=True,
    )


fusion = SimpleNamespace(physical=physical, _ending=_ending, unchanged_tokens=unchanged_tokens)


def geometry(source, views):
    if set(views) != {"yolo", "flip", "DFINE"}:
        raise ValueError("Only three auxiliary views; RF already anchor")
    lines = fusion.physical(source)
    anchor = np.array([observation[2:6] for _, observation in lines])
    groups = defaultdict(list)
    for i, (_, observation) in enumerate(lines):
        groups[observation[0]].append(i)
    matches = {}
    for name, floor in [
        ("yolo", YOLO_SCORE_MIN),
        ("flip", YOLO_SCORE_MIN),
        ("DFINE", DFINE_SCORE_MIN),
    ]:
        v = views[name]
        linked = {}
        if v is not None:
            from holdtrack import geometry as geo

            for frame, indices in groups.items():
                ai = geo.canonical(anchor, indices)
                vi = geo.canonical(
                    v["boxes"], np.flatnonzero((v["frames"] == frame) & (v["scores"] >= floor))
                )
                a, b = geo.match(geo.boxes_iou(anchor[ai], v["boxes"][vi]), VIEW_MATCH_IOU)
                for x, y in zip(ai[a], vi[b]):
                    linked[int(x)] = v["boxes"][y]
        matches[name] = linked
    out = []
    for i, (line, _) in enumerate(lines):
        other = [matches[name][i] for name in ["yolo", "flip", "DFINE"] if i in matches[name]]
        if not other:
            out.append(line)
            continue
        box = np.sum(np.stack([ANCHOR_WEIGHT * anchor[i], *other]), axis=0, dtype=np.float64) / (
            ANCHOR_WEIGHT + len(other)
        )
        x, y, x2, y2 = box
        tokens = line.rstrip(b"\r\n").split(b",")
        tokens[2:6] = [f"{v:.8f}".encode() for v in [x, y, x2 - x, y2 - y]]
        out.append(b",".join(tokens) + fusion._ending(line))
    payload = b"".join(out)
    invariants = fusion.unchanged_tokens(source, payload)
    return (
        payload,
        dict(
            invariants=invariants,
            RF_anchor_weight=ANCHOR_WEIGHT,
            RF_auxiliary_count=0,
            matching_counts={k: len(v) for k, v in matches.items()},
        ),
    )
