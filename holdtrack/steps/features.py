"""Build temporal reliability inputs from current predictions."""

import io
from collections import defaultdict

import numpy as np

from holdtrack import geometry as geo
from holdtrack.fusion.anchor import unchanged_tokens
from holdtrack.inputs.mot import physical


def mot_bytes(payload, *, truth=False):
    frames, boxes, tokens, source_indices = ([], [], [], [])
    with io.StringIO(payload.decode()) as stream:
        for index, line in enumerate(stream):
            if not line.strip():
                continue
            row = line.strip().split(",")
            if truth and (float(row[6]) <= 0 or int(float(row[7])) != 1):
                continue
            frame = int(float(row[0])) - 1
            x, y, w, h = map(float, row[2:6])
            if frame < 0 or w <= 0 or h <= 0 or (not np.isfinite([x, y, w, h]).all()):
                raise ValueError("Invalid MOT observation")
            frames.append(frame)
            boxes.append((x, y, x + w, y + h))
            source_indices.append(index)
            tokens.append(tuple((v for i, v in enumerate(row) if i != 1)))
    return dict(
        frames=np.asarray(frames, dtype=int),
        boxes=np.asarray(boxes, dtype=float).reshape(-1, 4),
        non_ID=tokens,
        source_indices=np.asarray(source_indices, dtype=int),
    )


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


def blind_links(source, flip, dfine):
    lines = physical(source)
    anchor = np.asarray([observation[2:6] for _, observation in lines], dtype=float).reshape(-1, 4)
    frames = defaultdict(list)
    for i, (_, observation) in enumerate(lines):
        frames[observation[0]].append(i)
    matches, reports = ({}, {})
    for name, view, threshold in [("flip", flip, 0.3), ("DFINE", dfine, 0.5)]:
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
            a, b = geo.match(geo.boxes_iou(anchor[ai], view["boxes"][vi]), 0.7)
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
    return (lines, anchor, matches, reports)


def features(source, candidate, raw, bt, flip=None, dfine=None):
    indices, proof = geo.bind_observations(raw, bt, mot_bytes(source))
    if proof["ambiguous_score_linked_rows"]:
        raise ValueError("Ambiguous duplicate geometry/rawscore")
    unchanged_tokens(source, candidate)
    j9 = physical(candidate)
    lines, anchor, links, _ = blind_links(source, flip, dfine)
    scores = raw["scores"][indices].astype(float)
    support = np.asarray(
        [int(i in links["flip"]) + int(i in links["DFINE"]) for i in range(len(lines))]
    )
    frames = np.asarray([row[0] for _, row in j9])
    identities = np.asarray([row[1] for _, row in j9])
    track = defaultdict(list)
    for i, identity in enumerate(identities):
        track[int(identity)].append(i)
    neighbor = np.zeros(len(lines))
    groups = defaultdict(list)
    for i, (_, row) in enumerate(lines):
        groups[row[0]].append(i)
    for group in groups.values():
        a = geo.boxes_iou(anchor[group], anchor[group])
        np.fill_diagonal(a, 0)
        neighbor[group] = a.max(1)
    selected = np.arange(len(frames))
    result = []
    for i in selected:
        same = np.asarray(track[int(identities[i])], int)
        delta = frames[same] - frames[i]
        other = same[(np.abs(delta) <= 5) & (delta != 0)]
        result.append(
            [
                scores[i],
                neighbor[i],
                support[i],
                len(other),
                int((frames[other] < frames[i]).sum()),
                int((frames[other] > frames[i]).sum()),
                float(scores[other].mean()) if len(other) else scores[i],
                float(scores[other].max()) if len(other) else scores[i],
                float(support[other].mean()) if len(other) else float(support[i]),
                len(same),
            ]
        )
    X = np.asarray(result, float).reshape(-1, 10)
    if X.shape != (len(frames), 10) or not np.isfinite(X).all():
        raise ValueError("All-row features malformed")
    return (X, frames[selected], identities[selected], proof)
