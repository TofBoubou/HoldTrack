"""Finish the tracks of a video once boxes and identities are decided.

In order: boxes that start an identity earlier, gaps filled under camera motion, edges
smoothed along each identity. The steps read tracks, mask boxes and detections; they read
no annotation.
"""

import numpy as np

from holdtrack.steps.camera_motion import fill_gaps
from holdtrack.steps.earlier_start import confirmed_mask_rows, rows_before_birth
from holdtrack.steps.smoothing import smooth_edges


def _append(frames, identities, boxes, added, frame_count):
    """Append the added rows (frame, identity, x, y, width, height) that are new and valid."""
    taken = set(zip(frames.tolist(), identities.tolist()))
    added = np.array(
        [
            row[:6]
            for row in added
            if (int(row[0]), int(row[1])) not in taken
            and 1 <= row[0] <= frame_count
            and row[4] > 0
            and row[5] > 0
        ]
    ).reshape(-1, 6)
    frames = np.concatenate([frames, added[:, 0].astype(np.int64)])
    identities = np.concatenate([identities, added[:, 1].astype(np.int64)])
    corners = np.c_[added[:, 2], added[:, 3], added[:, 2] + added[:, 4], added[:, 3] + added[:, 5]]
    return frames, identities, np.concatenate([boxes, corners]), len(added)


def finish(
    rows,
    fps,
    frame_count,
    motion,
    identities=None,
    forward_masks=None,
    backward_masks=None,
    detections=None,
    min_score=0.3,
    seconds_before=5.0,
    max_gap=1.5,
    window=0.2,
):
    """Return the frames, identities and boxes (x1, y1, x2, y2) of the finished tracks.

    rows: (m, 6+) frame, identity, x, y, width, height, frames numbered from 1.
    identities: one identity per row, decided after the tracks were built; the identities
    of `rows` are used when it is None. motion: {frame: 3x3 matrix}, see `camera_motion`.
    forward_masks, backward_masks: mask boxes (k, 6+) frame, object, x, y, width, height of
    the forward and backward passes. detections: (frames, boxes, scores) of the detector
    that started the tracks. Each input left as None skips the step that reads it.
    max_gap and window are in seconds; a window of 0 skips the smoothing.
    """
    rows = np.asarray(rows, float)
    frames = rows[:, 0].astype(np.int64)
    base_identities = rows[:, 1]
    boxes = np.stack([rows[:, 2], rows[:, 3], rows[:, 2] + rows[:, 4], rows[:, 3] + rows[:, 5]], 1)
    if identities is None:
        identities = base_identities.astype(np.int64)
    else:
        identities = np.asarray(identities).astype(np.int64)
        if len(identities) != len(frames):
            raise ValueError("one identity per row is required")
        if len(set(zip(frames.tolist(), identities.tolist()))) != len(frames):
            raise ValueError("an identity has two boxes in one frame")
    counts = {"earlier_start": 0, "before_birth": 0, "filled": 0}
    if detections is not None and forward_masks is not None:
        passes = [forward_masks] if backward_masks is None else [forward_masks, backward_masks]
        base = np.c_[
            frames,
            base_identities,
            boxes[:, 0],
            boxes[:, 1],
            boxes[:, 2] - boxes[:, 0],
            boxes[:, 3] - boxes[:, 1],
        ]
        added = confirmed_mask_rows(
            base,
            passes,
            detections[0],
            detections[1],
            detections[2],
            min_score=min_score,
            places=("before",),
            final_identities=identities,
        )
        frames, identities, boxes, counts["earlier_start"] = _append(
            frames, identities, boxes, added, frame_count
        )
    if forward_masks is not None:
        current = np.c_[
            frames,
            identities,
            boxes[:, 0],
            boxes[:, 1],
            boxes[:, 2] - boxes[:, 0],
            boxes[:, 3] - boxes[:, 1],
        ]
        added = rows_before_birth(current, forward_masks, fps, seconds=seconds_before)
        frames, identities, boxes, counts["before_birth"] = _append(
            frames, identities, boxes, added, frame_count
        )
    missing = [int(frame) for frame in np.unique(frames) if int(frame) not in motion]
    if missing:
        raise ValueError(f"no camera motion for frame {missing[0]}")
    new_frames, new_identities, new_boxes = fill_gaps(
        frames, identities, boxes, int(round(max_gap * fps)), motion
    )
    valid = (new_boxes[:, 2] > new_boxes[:, 0]) & (new_boxes[:, 3] > new_boxes[:, 1])
    frames = np.concatenate([frames, new_frames[valid]])
    identities = np.concatenate([identities, new_identities[valid]])
    boxes = np.concatenate([boxes, new_boxes[valid]])
    counts["filled"] = int(valid.sum())
    if window > 0:
        smoothed = smooth_edges(frames, identities, boxes, fps, window=window)
        usable = (
            (smoothed[:, 2] > smoothed[:, 0] + 1)
            & (smoothed[:, 3] > smoothed[:, 1] + 1)
            & np.isfinite(smoothed).all(1)
        )
        boxes = np.where(usable[:, None], smoothed, boxes)
    order = np.lexsort((identities, frames))
    return frames[order], identities[order], boxes[order], counts


def mot_lines(frames, identities, boxes):
    """Format finished tracks as MOT rows: frame, identity, x, y, width, height, 1, -1, -1, -1."""
    return [
        f"{frame:d},{identity:d},{box[0]:.2f},{box[1]:.2f},"
        f"{box[2] - box[0]:.2f},{box[3] - box[1]:.2f},1,-1,-1,-1"
        for frame, identity, box in zip(frames.tolist(), identities.tolist(), boxes.tolist())
    ]
