"""Fuse an RF detector view with the YOLO anchor."""

from collections import Counter, defaultdict

import numpy as np

from holdtrack import geometry as geo
from holdtrack.fusion.constants import DFINE_SCORE_MIN, VIEW_MATCH_IOU, YOLO_SCORE_MIN
from holdtrack.fusion.yolo import _ending, unchanged_tokens
from holdtrack.inputs.mot import physical


def weighted_box(anchor, matches, i):
    import numpy as np

    others = [matches[k][i] for k in ("flip", "DFINE") if i in matches[k]]
    if i not in matches["RF"]:
        return np.mean(np.stack([anchor, *others]), axis=0)
    terms = [np.asarray(anchor, dtype=np.float64)]
    terms.extend(np.asarray(x, dtype=np.float64) for x in others)
    terms.append(2 * np.asarray(matches["RF"][i], dtype=np.float64))
    return np.sum(np.stack(terms), axis=0, dtype=np.float64) / (3 + len(others))


def observations(frames, boxes, scores):
    columns = np.asarray(frames)
    x = np.asarray(boxes, dtype=float).reshape(-1, 4)
    s = np.asarray(scores)
    if (
        not len(columns) == len(x) == len(s)
        or not np.isfinite(x).all()
        or (not np.isfinite(s).all())
        or (x[:, 2:] < x[:, :2]).any()
        or ((s < 0) | (s > 1)).any()
        or (not np.equal(columns, columns.astype(np.int64)).all())
        or (columns < 0).any()
    ):
        raise ValueError("Invalid raw RF geometry/score")
    return dict(frames=columns.astype(np.int64), boxes=x, scores=s)


def fuse(source, flip, dfine, rf):
    lines = physical(source)
    anchor = np.asarray([observation[2:6] for _, observation in lines], dtype=float).reshape(-1, 4)
    frames = defaultdict(list)
    for i, (_, observation) in enumerate(lines):
        frames[observation[0]].append(i)
    matches, reports = ({}, {})
    for name, view, threshold in [
        ("flip", flip, YOLO_SCORE_MIN),
        ("DFINE", dfine, DFINE_SCORE_MIN),
        ("RF", rf, YOLO_SCORE_MIN),
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
        other = [matches[name][i] for name in ("flip", "DFINE", "RF") if i in matches[name]]
        counts["available_views_" + str(len(other))] += 1
        if not other:
            output.append(line)
            continue
        box = weighted_box(anchor[i], matches, i)
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


_native_fuse = fuse


def fuse(source, flip, dfine, rf):
    from holdtrack.fusion.yolo import observations as p2_observations

    for view in (flip, dfine):
        if view is not None:
            p2_observations(view["frames"], view["boxes"], view["scores"])
    return _native_fuse(source, flip, dfine, rf)
