"""Fuse YOLO and its complementary detection views."""

from collections import Counter, defaultdict

import numpy as np

from holdtrack import geometry as geo
from holdtrack.fusion.constants import DFINE_SCORE_MIN, VIEW_MATCH_IOU, YOLO_SCORE_MIN
from holdtrack.inputs.mot import physical


def observations(frames, boxes, scores):
    frames = np.asarray(frames)
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    scores = np.asarray(scores)
    if frames.ndim != 1 or scores.ndim != 1 or (not len(frames) == len(boxes) == len(scores)):
        raise ValueError("Incompatible raw observation arrays")
    if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
        raise ValueError("Nonfinite raw observation")
    if not np.equal(frames, frames.astype(np.int64)).all() or (frames < 0).any():
        raise ValueError("Noninteger/negative raw frame")
    if (boxes[:, 2:] <= boxes[:, :2]).any() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Invalid raw extent/score; no silent clipping or discard")
    return dict(frames=frames.astype(np.int64), boxes=boxes, scores=scores)


def inverse_flip(boxes, width):
    boxes = np.asarray(boxes)
    result = boxes.copy()
    result[:, 0] = width - boxes[:, 2]
    result[:, 2] = width - boxes[:, 0]
    return result


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


def fuse(source, flip, dfine):
    lines = physical(source)
    anchor = np.asarray([observation[2:6] for _, observation in lines], dtype=float).reshape(-1, 4)
    frames = defaultdict(list)
    for i, (_, observation) in enumerate(lines):
        frames[observation[0]].append(i)
    matches, reports = ({}, {})
    for name, view, threshold in [
        ("flip", flip, YOLO_SCORE_MIN),
        ("DFINE", dfine, DFINE_SCORE_MIN),
    ]:
        groups = defaultdict(list)
        if view is not None:
            view = observations(view["frames"], view["boxes"], view["scores"])
            for i in np.flatnonzero(view["scores"] >= threshold):
                groups[int(view["frames"][i])].append(int(i))
        linked = {}
        for frame, indices in frames.items():
            if not groups.get(frame):
                continue
            ai = geo.canonical(anchor, indices)
            vi = geo.canonical(view["boxes"], groups[frame])
            a, b = geo.match(geo.boxes_iou(anchor[ai], view["boxes"][vi]), VIEW_MATCH_IOU)
            for x, y in zip(ai[a], vi[b]):
                linked[int(x)] = view["boxes"][y].copy()
        matches[name] = linked
        reports[name] = dict(
            available=view is not None,
            source_threshold=threshold,
            eligible_source_rows=sum(map(len, groups.values())),
            matched_anchor_rows=len(linked),
            match_to_original_anchor=True,
        )
    output, counts = ([], Counter())
    for i, (line, _) in enumerate(lines):
        other = [matches[name][i] for name in ("flip", "DFINE") if i in matches[name]]
        counts["available_views_" + str(len(other))] += 1
        if not other:
            output.append(line)
            continue
        box = np.mean(np.stack([anchor[i], *other]), axis=0)
        if np.array_equal(box, anchor[i]):
            output.append(line)
            continue
        x, y, x2, y2 = map(float, box)
        if not np.isfinite(box).all() or x2 <= x or y2 <= y:
            raise ValueError("Invalid fused box, no fallback")
        row = line.rstrip(b"\r\n").split(b",")
        row[2:6] = [f"{v:.8f}".encode() for v in (x, y, x2 - x, y2 - y)]
        output.append(b",".join(row) + _ending(line))
        counts["coordinate_changed_rows"] += 1
    result = b"".join(output)
    invariant = unchanged_tokens(source, result)
    return (
        result,
        dict(
            rows_in=len(lines),
            rows_out=len(lines),
            counts=dict(counts),
            views=reports,
            invariance=invariant,
        ),
    )
