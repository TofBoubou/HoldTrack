"""Match detector boxes without changing their ordering."""

from collections import Counter, defaultdict
from decimal import Decimal

import numpy as np
from scipy.optimize import linear_sum_assignment


def boxes_iou(a, b):
    a, b = (np.asarray(a, dtype=float).reshape(-1, 4), np.asarray(b, dtype=float).reshape(-1, 4))
    area_a = np.prod(np.maximum(0, a[:, 2:] - a[:, :2]), axis=1)
    area_b = np.prod(np.maximum(0, b[:, 2:] - b[:, :2]), axis=1)
    inter = np.maximum(
        0, np.minimum(a[:, None, 2:], b[None, :, 2:]) - np.maximum(a[:, None, :2], b[None, :, :2])
    ).prod(2)
    return inter / np.maximum(1e-15, area_a[:, None] + area_b[None, :] - inter)


def match(iou, alpha):
    iou = np.asarray(iou, dtype=float)
    if not iou.size:
        return (np.empty(0, dtype=int), np.empty(0, dtype=int))
    eligible = iou >= alpha
    score = np.where(eligible, min(iou.shape) + 1 + iou, 0)
    a, b = linear_sum_assignment(-score)
    keep = eligible[a, b]
    return (a[keep], b[keep])


def signature(frame, box):
    x1, y1, x2, y2 = map(float, box)
    return (int(frame), *(f"{v:.8f}" for v in (x1, y1, x2 - x1, y2 - y1)))


def mot(path, *, truth=False):
    frames, boxes, tokens, source_indices = ([], [], [], [])
    with open(path) as stream:
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


def bind_observations(raw, bt, identities):
    if bt["non_ID"] != identities["non_ID"]:
        raise ValueError("three-voice must preserve ByteTrack non-ID tokens and order")
    lookup = defaultdict(list)
    for index, (frame, box) in enumerate(zip(raw["frames"], raw["boxes"])):
        lookup[signature(frame, box)].append(index)
    cursor, linked, duplicates = (Counter(), [], [])
    ambiguous_score_rows = 0
    for index, (frame, box) in enumerate(zip(bt["frames"], bt["boxes"])):
        row = bt["non_ID"][index]
        key = (int(frame), *(f"{Decimal(v):.8f}" for v in row[1:5]))
        choices = lookup.get(key, [])
        k = cursor[key]
        if k >= len(choices):
            raise ValueError("ByteTrack row has no exact unused raw observation: " + str(key))
        linked.append(choices[k])
        cursor[key] += 1
        if len(choices) > 1:
            duplicates.append(dict(BT_row=index, equivalent_raw_indices=choices))
            ambiguous_score_rows += len({float(raw["scores"][i]) for i in choices}) > 1
    linked = np.asarray(linked, dtype=int)
    return (
        linked,
        dict(
            method="exact frame/8-decimal native MOT geometry multiset; source-order tie",
            duplicate_groups=duplicates,
            duplicate_observation_count=len(duplicates),
            ambiguous_score_linked_rows=ambiguous_score_rows,
            duplicate_indices_not_claimed_as_historic_track_idx=True,
            exact_matched_rows=len(linked),
            coordinate_changed_rows=0,
            unexpected_added_rows=0,
            three_voice_non_ID_tokens_and_order_exact=True,
        ),
    )


def canonical(boxes, indices):
    return np.asarray(sorted(indices, key=lambda i: (*map(float, boxes[i]), int(i))), dtype=int)
