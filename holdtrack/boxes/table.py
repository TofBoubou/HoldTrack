"""Build the table of a video: one row per box of the tracks, with the box that each
detector view gives to the same person, and the occlusion flags of its edges.

A view is a set of detector boxes with scores. In each frame, the boxes of the tracks
are assigned one to one to the boxes of the view that pass its score floor, by maximum
total IoU; a pair is kept when its IoU is at least 0.5.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w = np.clip(
        np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None
    )
    h = np.clip(
        np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None
    )
    i = w * h
    return i / (
        ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
        + ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None]
        - i
        + 1e-9
    )


def _by_frame(frames: np.ndarray, index: np.ndarray | None = None) -> dict[int, np.ndarray]:
    index = np.arange(len(frames)) if index is None else index
    order = index[np.argsort(frames[index], kind="stable")]
    values, start = np.unique(frames[order], return_index=True)
    return {
        int(f): order[s : (start[n + 1] if n + 1 < len(start) else len(order))]
        for n, (f, s) in enumerate(zip(values, start))
    }


def match_view(
    frames, boxes, view_frames, view_boxes, view_scores, floor: float, minimum: float = 0.5
):
    """Box, IoU and score that the view gives to each row (NaN, 0 and 0 without a match).

    The mask boxes of a tracker pass are matched the same way, with ``minimum`` at 0.3
    and scores of one."""
    n = len(frames)
    out_box, out_iou, out_score = np.full((n, 4), np.nan), np.zeros(n), np.zeros(n)
    ok = (
        (view_scores >= floor)
        & (view_boxes[:, 2] > view_boxes[:, 0])
        & (view_boxes[:, 3] > view_boxes[:, 1])
    )
    view = _by_frame(view_frames, np.flatnonzero(ok))
    for f, rows in _by_frame(frames).items():
        if f not in view:
            continue
        jj = view[f]
        m = iou_matrix(boxes[rows], view_boxes[jj])
        r, c = linear_sum_assignment(-m)
        for x, y in zip(r, c):
            if m[x, y] >= minimum:
                out_box[rows[x]] = view_boxes[jj[y]]
                out_iou[rows[x]] = m[x, y]
                out_score[rows[x]] = view_scores[jj[y]]
    return out_box, out_iou, out_score


def edge_flags(frames, boxes, width: float | None = None, height: float | None = None):
    """Contact (largest IoU with another box of the frame) and one flag per edge:
    0 free, 1 covered by a neighbor in front (lower bottom edge), 2 by a neighbor behind,
    3 at the image border. Without the image size, the border is that of the boxes."""
    n = len(frames)
    contact, flags = np.zeros(n), np.zeros((n, 4), int)
    width = float(np.ceil(boxes[:, 2].max())) if width is None else float(width)
    height = float(np.ceil(boxes[:, 3].max())) if height is None else float(height)
    for rows in _by_frame(frames).values():
        if len(rows) < 2:
            continue
        b_all = boxes[rows]
        m = iou_matrix(b_all, b_all)
        np.fill_diagonal(m, 0)
        contact[rows] = m.max(1)
        for x in range(len(rows)):
            b = b_all[x]
            for y in range(len(rows)):
                if x == y:
                    continue
                o = b_all[y]
                front = 1 if o[3] > b[3] else 2
                vertical = (min(b[3], o[3]) - max(b[1], o[1])) / (b[3] - b[1])
                horizontal = (min(b[2], o[2]) - max(b[0], o[0])) / (b[2] - b[0])
                if vertical > 0.3:
                    if o[0] < b[0] < o[2] and (flags[rows[x], 0] == 0 or front == 1):
                        flags[rows[x], 0] = front
                    if o[0] < b[2] < o[2] and (flags[rows[x], 2] == 0 or front == 1):
                        flags[rows[x], 2] = front
                if horizontal > 0.3:
                    if o[1] < b[1] < o[3] and (flags[rows[x], 1] == 0 or front == 1):
                        flags[rows[x], 1] = front
                    if o[1] < b[3] < o[3] and (flags[rows[x], 3] == 0 or front == 1):
                        flags[rows[x], 3] = front
    border = np.stack(
        [boxes[:, 0] <= 2, boxes[:, 1] <= 2, boxes[:, 2] >= width - 2, boxes[:, 3] >= height - 2], 1
    )
    flags[border] = 3
    return contact, flags, width, height


def match_bands(frames, boxes, bands, width: float, floor: float = 0.3, margin: float = 8.0):
    """View read by vertical bands of the frame, merged into one view.

    ``bands`` is a list of (left, right, frames, boxes, scores). A row uses only the
    bands that hold its box entirely (with a margin, except at the image border); in a
    band, detections that touch an inner border of the band are dropped; among the
    bands that match the row, the one whose center is nearest to the box center is kept.
    """
    n = len(frames)
    best = np.full(n, np.inf)
    out_box, out_iou, out_score = np.full((n, 4), np.nan), np.zeros(n), np.zeros(n)
    rows_by_frame = _by_frame(frames)
    for left, right, b_frames, b_boxes, b_scores in bands:
        ok = (
            (b_scores >= floor)
            & (b_boxes[:, 2] > b_boxes[:, 0])
            & (b_boxes[:, 3] > b_boxes[:, 1])
            & ((b_boxes[:, 0] > left + 3) | (left <= 0))
            & ((b_boxes[:, 2] < right - 3) | (right >= width))
        )
        view = _by_frame(b_frames, np.flatnonzero(ok))
        inside = ((boxes[:, 0] >= left + margin) | (left <= 0)) & (
            (boxes[:, 2] <= right - margin) | (right >= width)
        )
        distance = np.abs((boxes[:, 0] + boxes[:, 2]) / 2 - (left + right) / 2)
        for f, rows in rows_by_frame.items():
            rows = rows[inside[rows]]
            if f not in view or len(rows) == 0:
                continue
            jj = view[f]
            m = iou_matrix(boxes[rows], b_boxes[jj])
            r, c = linear_sum_assignment(-m)
            for x, y in zip(r, c):
                if m[x, y] >= 0.5 and distance[rows[x]] < best[rows[x]]:
                    best[rows[x]] = distance[rows[x]]
                    out_box[rows[x]] = b_boxes[jj[y]]
                    out_iou[rows[x]] = m[x, y]
                    out_score[rows[x]] = b_scores[jj[y]]
    return out_box, out_iou, out_score


def row_view(box, iou, score, floor: float = 0.3):
    """View computed for each row (a detector read on a crop around the box): kept when
    its IoU with the box of the row is at least 0.5 and its score passes the floor."""
    box = np.asarray(box, float)
    ok = (
        np.isfinite(box[:, 0])
        & (iou >= 0.5)
        & (score >= floor)
        & (box[:, 2] > box[:, 0])
        & (box[:, 3] > box[:, 1])
    )
    return np.where(ok[:, None], box, np.nan), np.where(ok, iou, 0), np.where(ok, score, 0)


def _iou_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w = np.clip(np.minimum(a[:, 2], b[:, 2]) - np.maximum(a[:, 0], b[:, 0]), 0, None)
    h = np.clip(np.minimum(a[:, 3], b[:, 3]) - np.maximum(a[:, 1], b[:, 1]), 0, None)
    i = w * h
    return i / (
        (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
        + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        - i
        + 1e-9
    )


def enclosing(points: np.ndarray, usable: np.ndarray, least: int) -> np.ndarray:
    """Box that encloses the usable points of each row; NaN with fewer than ``least``."""
    x = np.where(usable, points[:, :, 0], np.nan)
    y = np.where(usable, points[:, :, 1], np.nan)
    with np.errstate(all="ignore"):
        b = np.stack([np.nanmin(x, 1), np.nanmin(y, 1), np.nanmax(x, 1), np.nanmax(y, 1)], 1)
    b[usable.sum(1) < least] = np.nan
    b[(b[:, 2] <= b[:, 0] + 1) | (b[:, 3] <= b[:, 1] + 1)] = np.nan
    return b


def body_views(boxes, mesh_box, points17, points133):
    """Three views of each row taken from the body: the box of the mesh, the box of the
    17 keypoints with a score above 0.3 (at least 4), and the box of the 133 keypoints
    with a score above 2.127 (at least 10). Returns boxes, IoUs and scores, (3, N, ...)."""
    import warnings

    boxes = np.asarray(boxes, float)
    n = len(boxes)
    k17, k133 = np.asarray(points17, float), np.asarray(points133, float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        views = [
            np.asarray(mesh_box, float),
            enclosing(k17, k17[:, :, 2] > 0.3, 4),
            enclosing(k133, k133[:, :, 2] > 2.127, 10),
        ]
    scores = [np.ones(n), k17[:, :, 2].mean(1), k133[:, :, 2].mean(1) / 5.0]
    ious, kept = [], []
    for b, sc in zip(views, scores):
        ok = np.isfinite(b[:, 0])
        io = np.zeros(n)
        io[ok] = _iou_rows(b[ok], boxes[ok])
        ious.append(np.where(ok, io, 0.0))
        kept.append(np.where(ok, sc, 0.0))
    return np.stack(views), np.stack(ious), np.stack(kept)


def build_table(
    track_rows,
    full_views,
    band_views,
    row_views,
    pass_masks,
    mesh_box=None,
    points17=None,
    points133=None,
    width=None,
    height=None,
):
    """Table of a video, as two dictionaries (rows and views; mask boxes).

    track_rows   MOT rows of the tracks of the base system, (N, 10)
    full_views   list of (frames, boxes, scores, score floor): detector reads of the full frame
    band_views   list of lists of (left, right, frames, boxes, scores): detector reads by vertical bands
    row_views    list of (boxes, IoUs, scores), one entry per row: detector reads on a crop around the box
    pass_masks   three arrays of MOT rows: the mask boxes of the three forward passes
    mesh_box     (N, 4) box of the body mesh, NaN where missing
    points17, points133   keypoints of the two pose estimators, (N, 17, 3) and (N, 133, 3)
    Without the mesh and the keypoints, the table holds the detector views only.
    The views are stored in this order: full frame, bands, crops, mesh, 17 keypoints, 133 keypoints.
    """
    q = np.asarray(track_rows, float)
    frames = q[:, 0].astype(np.int64)
    boxes = np.stack([q[:, 2], q[:, 3], q[:, 2] + q[:, 4], q[:, 3] + q[:, 5]], 1)
    n = len(q)
    contact, flags, width, height = edge_flags(frames, boxes, width, height)
    views = [match_view(frames, boxes, f, b, s, floor) for f, b, s, floor in full_views]
    views += [match_bands(frames, boxes, bands, width) for bands in band_views]
    views += [row_view(b, i, s) for b, i, s in row_views]
    if mesh_box is not None:
        body = body_views(boxes, mesh_box, points17, points133)
        views += [(body[0][k], body[1][k], body[2][k]) for k in range(3)]
    masks = np.full((3, n, 4), np.nan)
    masks_iou = np.zeros((3, n))
    for k, m in enumerate(pass_masks):
        m = np.asarray(m, float)
        mb = np.stack([m[:, 2], m[:, 3], m[:, 2] + m[:, 4], m[:, 3] + m[:, 5]], 1)
        masks[k], masks_iou[k], _ = match_view(
            frames, boxes, m[:, 0].astype(int), mb, np.ones(len(m)), 0.0, minimum=0.3
        )
    table = dict(
        frame=frames,
        pid=q[:, 1].astype(np.int64),
        ref=boxes,
        gt=np.full((n, 4), np.nan),
        gid=np.full(n, -1, np.int64),
        views=np.stack([v[0] for v in views]),
        views_iou=np.stack([v[1] for v in views]),
        views_score=np.stack([v[2] for v in views]),
        contact=contact,
        hidden=flags.astype(np.int64),
        W=np.float64(width),
        H=np.float64(height),
        fold=np.int64(0),
        tail=q[:, 6:],
    )
    return table, dict(masks=masks, masks_iou=masks_iou)
