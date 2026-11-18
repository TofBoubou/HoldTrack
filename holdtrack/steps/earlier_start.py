"""Start an identity earlier, from the masks that hold the person before its first box.

An identity starts at its first box. A pass of the mask tracker can hold the same person in
a mask before that frame. Two functions turn such masks into boxes of the identity:

`confirmed_mask_rows` reads the mask boxes of one or several passes and a bank of
detections. A mask that has no box in its frame becomes a box when a detection that covers
no existing box matches it; the box is the detection and the identity is the one of the
object.

`rows_before_birth` reads the mask boxes of one pass alone. In the seconds before the first
box of an identity, the mask of its object becomes a box when it has the size of that first
box and covers no existing box.

Neither function reads an annotation.
"""

import collections

import numpy as np
from scipy.optimize import linear_sum_assignment


def corners(rows):
    """Columns x, y, width, height (from column 2) as x1, y1, x2, y2."""
    return np.stack([rows[:, 2], rows[:, 3], rows[:, 2] + rows[:, 4], rows[:, 3] + rows[:, 5]], 1)


def overlaps(first, second):
    """IoU, share of each first box covered, share of each second box covered."""
    width = np.clip(
        np.minimum(first[:, None, 2], second[None, :, 2])
        - np.maximum(first[:, None, 0], second[None, :, 0]),
        0,
        None,
    )
    height = np.clip(
        np.minimum(first[:, None, 3], second[None, :, 3])
        - np.maximum(first[:, None, 1], second[None, :, 1]),
        0,
        None,
    )
    inter = width * height
    area_first = (first[:, 2] - first[:, 0]) * (first[:, 3] - first[:, 1])
    area_second = (second[:, 2] - second[:, 0]) * (second[:, 3] - second[:, 1])
    return (
        inter / (area_first[:, None] + area_second[None] - inter + 1e-9),
        inter / (area_first[:, None] + 1e-9),
        inter / (area_second[None] + 1e-9),
    )


def match(first, second):
    """Return, for each box of `second`, the index of its box of `first`, or -1.

    Boxes are matched one to one at an IoU of at least 0.5. A box of `second` left alone is
    then matched to a free box of `first` that contains 70% of it, if it covers 15% of that
    box.
    """
    out = -np.ones(len(second), np.int64)
    if not len(first) or not len(second):
        return out
    iou, cover_first, cover_second = overlaps(first, second)
    rows, columns = linear_sum_assignment(-iou)
    matched = iou[rows, columns] >= 0.5
    out[columns[matched]] = rows[matched]
    free_first = np.setdiff1d(np.arange(len(first)), rows[matched])
    free_second = np.where(out < 0)[0]
    if len(free_first) and len(free_second):
        block = np.ix_(free_first, free_second)
        score = np.where(
            (cover_second[block] >= 0.7) & (cover_first[block] >= 0.15), iou[block] + 1e-3, -1.0
        )
        rows, columns = linear_sum_assignment(-score)
        matched = score[rows, columns] > 0
        out[free_second[columns[matched]]] = free_first[rows[matched]]
    return out


def confirmed_mask_rows(
    rows,
    passes,
    detection_frames,
    detection_boxes,
    detection_scores,
    min_score=0.3,
    min_rows=5,
    share=0.8,
    places=("before",),
    final_identities=None,
):
    """Return the boxes (n, 8) that the masks of the passes add to the identities.

    rows: (m, 6+) frame, identity, x, y, width, height. passes: list of arrays (k, 6+)
    frame, object, x, y, width, height, the mask boxes of each pass, with the frame numbers
    of `rows`. The detections are the bank of the detector that started the tracks.

    An object carries an identity when at least `share` of its masks that lie on a box lie
    on boxes of that identity (`min_rows` boxes at least). A mask is free when it lies on no
    box and its identity has no box in the frame. A detection is free when its score is at
    least `min_score` and its IoU with every box of the frame is under 0.5. A free mask
    matched to a free detection proposes the detection for its identity. Between passes a
    detection goes to the identity that proposes it most, and an identity receives one
    detection per frame.

    places: where a box is kept relative to the boxes of its identity: "before" its first
    box, "inside", "after" its last box. final_identities: one identity per row of `rows`,
    decided after the objects were named; each added box then takes the final identity of
    the nearest box of its identity in time, and is dropped if that identity already has a
    box in the frame.

    Columns of the result: frame, identity, x, y, width, height, score of the detection,
    number of passes that agree.
    """
    rows = np.asarray(rows, float)
    row_frames = rows[:, 0].astype(np.int64)
    row_identities = rows[:, 1].astype(np.int64)
    row_boxes = corners(rows)
    detection_frames = np.asarray(detection_frames).astype(np.int64)
    detection_boxes = np.asarray(detection_boxes, float)
    detection_scores = np.asarray(detection_scores, float)
    valid = (
        np.isfinite(detection_boxes).all(1)
        & (detection_boxes[:, 2] > detection_boxes[:, 0])
        & (detection_boxes[:, 3] > detection_boxes[:, 1])
        & (detection_scores >= min_score)
    )
    rows_of_frame = collections.defaultdict(list)
    detections_of_frame = collections.defaultdict(list)
    for index, frame in enumerate(row_frames.tolist()):
        rows_of_frame[frame].append(index)
    for index in np.where(valid)[0].tolist():
        detections_of_frame[int(detection_frames[index])].append(index)
    present = set(zip(row_frames.tolist(), row_identities.tolist()))
    free_detections = {}
    proposals = collections.defaultdict(collections.Counter)
    for masks in passes:
        masks = np.asarray(masks, float)
        masks = masks[(masks[:, 4] > 1) & (masks[:, 5] > 1)]
        mask_frames = masks[:, 0].astype(np.int64)
        mask_objects = masks[:, 1].astype(np.int64)
        mask_boxes = corners(masks)
        mask_row = -np.ones(len(masks), np.int64)
        masks_of_frame = collections.defaultdict(list)
        for index, frame in enumerate(mask_frames.tolist()):
            masks_of_frame[frame].append(index)
        for frame, mask_index in masks_of_frame.items():
            row_index = rows_of_frame.get(frame)
            if row_index:
                mask_index = np.array(mask_index)
                row_index = np.array(row_index)
                found = match(row_boxes[row_index], mask_boxes[mask_index])
                mask_row[mask_index] = np.where(found >= 0, row_index[np.maximum(found, 0)], -1)
        identity_of = {}
        for obj in np.unique(mask_objects):
            carried = row_identities[mask_row[(mask_objects == obj) & (mask_row >= 0)]]
            if len(carried) >= min_rows:
                values, counts = np.unique(carried, return_counts=True)
                if counts.max() >= share * len(carried):
                    identity_of[int(obj)] = int(values[counts.argmax()])
        for frame, mask_index in masks_of_frame.items():
            mask_index = np.array(
                [
                    index
                    for index in mask_index
                    if mask_row[index] < 0
                    and int(mask_objects[index]) in identity_of
                    and (frame, identity_of[int(mask_objects[index])]) not in present
                ]
            )
            if not len(mask_index) or frame not in detections_of_frame:
                continue
            if frame not in free_detections:
                candidates = np.array(detections_of_frame[frame])
                row_index = rows_of_frame.get(frame)
                if row_index:
                    iou = overlaps(detection_boxes[candidates], row_boxes[np.array(row_index)])[0]
                    candidates = candidates[iou.max(1) < 0.5]
                free_detections[frame] = candidates
            candidates = free_detections[frame]
            if not len(candidates):
                continue
            found = match(detection_boxes[candidates], mask_boxes[mask_index])
            pairs = zip(mask_index[found >= 0].tolist(), candidates[found[found >= 0]].tolist())
            for index, detection in pairs:
                proposals[frame, detection][identity_of[int(mask_objects[index])]] += 1
    best = {}
    for (frame, detection), votes in proposals.items():
        identity, count = max(votes.items(), key=lambda vote: (vote[1], -vote[0]))
        current = best.get((frame, identity))
        if current is None or (count, detection_scores[detection]) > (
            current[1],
            detection_scores[current[0]],
        ):
            best[frame, identity] = (detection, count)
    added = [
        (
            frame,
            identity,
            detection_boxes[detection, 0],
            detection_boxes[detection, 1],
            detection_boxes[detection, 2] - detection_boxes[detection, 0],
            detection_boxes[detection, 3] - detection_boxes[detection, 1],
            detection_scores[detection],
            count,
        )
        for (frame, identity), (detection, count) in best.items()
    ]
    added.sort(key=lambda row: row[:2])
    added = np.array(added, float).reshape(-1, 8)
    keep = np.ones(len(added), bool)
    renamed = added[:, 1].copy()
    final = None if final_identities is None else np.asarray(final_identities).astype(np.int64)
    taken = set() if final is None else set(zip(row_frames.tolist(), final.tolist()))
    sorted_rows = {}
    for index in range(len(added)):
        identity = int(added[index, 1])
        if identity not in sorted_rows:
            own = np.where(row_identities == identity)[0]
            sorted_rows[identity] = own[np.argsort(row_frames[own], kind="stable")]
        own = sorted_rows[identity]
        times = row_frames[own]
        position = int(np.searchsorted(times, added[index, 0]))
        if position == 0:
            place, nearest = "before", own[0]
        elif position == len(times):
            place, nearest = "after", own[-1]
        else:
            place = "inside"
            closer_left = added[index, 0] - times[position - 1] <= times[position] - added[index, 0]
            nearest = own[position - 1] if closer_left else own[position]
        keep[index] = place in places
        if keep[index] and final is not None:
            key = (int(added[index, 0]), int(final[nearest]))
            if key in taken:
                keep[index] = False
            else:
                taken.add(key)
                renamed[index] = final[nearest]
    added[:, 1] = renamed
    return added[keep]


def rows_before_birth(rows, masks, fps, seconds=5.0):
    """Return the boxes (n, 6) that the masks of one pass add before the first box.

    rows: (m, 6+) frame, identity, x, y, width, height. masks: (k, 6+) frame, object, x, y,
    width, height, with the frame numbers of `rows`. An object carries an identity when 80%
    of its masks that lie on a box lie on boxes of that identity (5 boxes at least). In the
    `seconds` before the first box of the identity, the mask of the object becomes a box
    when its height is 0.7 to 1.4 times and its width 0.6 to 1.6 times those of that first
    box, and when its IoU with every box of the frame is under 0.5.
    """
    rows = np.asarray(rows, float)[:, :6]
    masks = np.asarray(masks, float)[:, :6]
    if not len(rows) or not len(masks):
        return np.zeros((0, 6))
    row_boxes = corners(rows)
    row_identities = rows[:, 1].astype(np.int64)
    mask_frames = masks[:, 0]
    mask_boxes = corners(masks)
    mask_objects = masks[:, 1].astype(np.int64)
    carried = np.full(len(masks), -1, np.int64)
    alone = np.ones(len(masks), bool)
    for frame in np.unique(mask_frames):
        mask_index = np.where(mask_frames == frame)[0]
        row_index = np.where(rows[:, 0] == frame)[0]
        if not len(row_index):
            continue
        iou = overlaps(mask_boxes[mask_index], row_boxes[row_index])[0]
        first, second = linear_sum_assignment(-iou)
        matched = iou[first, second] >= 0.5
        carried[mask_index[first[matched]]] = row_identities[row_index[second[matched]]]
        alone[mask_index[iou.max(1) >= 0.5]] = False
    added = {}
    for obj in np.unique(mask_objects):
        names = carried[(mask_objects == obj) & (carried >= 0)]
        if len(names) < 5:
            continue
        identity, count = collections.Counter(names.tolist()).most_common(1)[0]
        if count < 0.8 * len(names):
            continue
        own = np.where(row_identities == identity)[0]
        birth = own[rows[own, 0].argmin()]
        birth_frame = rows[birth, 0]
        birth_height = row_boxes[birth, 3] - row_boxes[birth, 1]
        birth_width = row_boxes[birth, 2] - row_boxes[birth, 0]
        earlier = (
            (mask_objects == obj)
            & alone
            & (mask_frames < birth_frame)
            & (mask_frames >= birth_frame - seconds * fps)
        )
        for index in np.where(earlier)[0]:
            height = mask_boxes[index, 3] - mask_boxes[index, 1]
            width = mask_boxes[index, 2] - mask_boxes[index, 0]
            if 0.7 <= height / birth_height <= 1.4 and 0.6 <= width / birth_width <= 1.6:
                added[(int(mask_frames[index]), int(identity))] = mask_boxes[index]
    return np.array(
        [
            [frame, identity, box[0], box[1], box[2] - box[0], box[3] - box[1]]
            for (frame, identity), box in added.items()
        ]
    ).reshape(len(added), 6)
