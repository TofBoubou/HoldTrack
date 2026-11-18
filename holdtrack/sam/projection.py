"""Project segmentation identities onto detector trajectories."""

from __future__ import annotations

import io
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

IOU_MIN = 0.5
MAX_GAP = 30
HALF_WINDOW = 5
FLICKER_WINDOW = 5


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax2, ay2 = (a[:, 0] + a[:, 2], a[:, 1] + a[:, 3])
    bx2, by2 = (b[:, 0] + b[:, 2], b[:, 1] + b[:, 3])
    w = np.clip(
        np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None
    )
    h = np.clip(
        np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None
    )
    inter = w * h
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def relabel(det: np.ndarray, sam: np.ndarray) -> tuple[np.ndarray, dict]:
    labels = np.zeros(len(det), dtype=np.int64)
    matched_iou = []
    sam_by_frame = defaultdict(list)
    for j, frame in enumerate(sam[:, 0].astype(int)):
        sam_by_frame[frame].append(j)
    det_by_frame = defaultdict(list)
    for i, frame in enumerate(det[:, 0].astype(int)):
        det_by_frame[frame].append(i)
    sam_dropped = 0
    for frame, di in det_by_frame.items():
        sj = sam_by_frame.get(frame, [])
        if not sj:
            continue
        m = iou(det[di, 2:6], sam[sj, 2:6])
        rows, cols = linear_sum_assignment(-m)
        for observation, c in zip(rows, cols):
            if m[observation, c] >= IOU_MIN:
                labels[di[observation]] = int(sam[sj[c], 1])
                matched_iou.append(float(m[observation, c]))
    for frame, sj in sam_by_frame.items():
        used = {labels[i] for i in det_by_frame.get(frame, []) if labels[i]}
        sam_dropped += sum(1 for j in sj if int(sam[j, 1]) not in used)
    n_matched = int((labels > 0).sum())
    by_track = defaultdict(list)
    for i in range(len(det)):
        by_track[int(det[i, 1])].append(i)
    frame_used = defaultdict(set)
    for i in range(len(det)):
        if labels[i]:
            frame_used[int(det[i, 0])].add(int(labels[i]))
    propagated = 0
    fresh = 0
    offset = int(sam[:, 1].max()) if len(sam) else 0
    for track, idx in sorted(by_track.items()):
        idx = sorted(idx, key=lambda i: det[i, 0])
        anchors = [(int(det[i, 0]), int(labels[i])) for i in idx if labels[i]]
        anchor_frames = np.array([a[0] for a in anchors]) if anchors else np.array([], dtype=int)
        for i in idx:
            if labels[i]:
                continue
            frame = int(det[i, 0])
            choice = 0
            if len(anchor_frames):
                gaps = np.abs(anchor_frames - frame)
                k = int(np.argmin(gaps))
                if gaps[k] <= MAX_GAP:
                    candidate = anchors[k][1]
                    if candidate not in frame_used[frame]:
                        choice = candidate
            if choice:
                propagated += 1
            else:
                choice = offset + track
                fresh += 1
            labels[i] = choice
            frame_used[frame].add(choice)
    out = det.copy()
    out[:, 1] = labels
    census = {
        "det_rows": int(len(det)),
        "sam2_rows": int(len(sam)),
        "matched_rows": n_matched,
        "propagated_rows": propagated,
        "fresh_rows": fresh,
        "sam2_rows_dropped": int(sam_dropped),
        "matched_iou_median": float(np.median(matched_iou)) if matched_iou else None,
        "ids_det": int(len(np.unique(det[:, 1]))),
        "ids_sam2": int(len(np.unique(sam[:, 1]))),
        "ids_out": int(len(np.unique(labels))),
        "duplicate_id_in_frame": int(
            len(out) - len({(int(observation[0]), int(observation[1])) for observation in out})
        ),
    }
    return (out, census)


def smooth(det: np.ndarray, v1: np.ndarray) -> np.ndarray:
    labels = v1.copy()
    by_track = defaultdict(list)
    for i in range(len(det)):
        by_track[int(det[i, 1])].append(i)
    for idx in by_track.values():
        idx.sort(key=lambda i: det[i, 0])
        frames = det[idx, 0]
        for pos, i in enumerate(idx):
            lo = np.searchsorted(frames, frames[pos] - HALF_WINDOW, side="left")
            hi = np.searchsorted(frames, frames[pos] + HALF_WINDOW, side="right")
            counts = Counter(int(v1[j]) for j in idx[lo:hi])
            top = max(counts.values())
            leaders = sorted((k for k, c in counts.items() if c == top))
            labels[i] = int(v1[i]) if int(v1[i]) in leaders else leaders[0]
    while True:
        by_frame_label = defaultdict(list)
        for i in range(len(det)):
            by_frame_label[int(det[i, 0]), int(labels[i])].append(i)
        conflicts = [rows for rows in by_frame_label.values() if len(rows) > 1]
        if not conflicts:
            return labels
        for rows in conflicts:
            for i in rows:
                if labels[i] != v1[i]:
                    labels[i] = v1[i]


def _matrix(payload):
    if not payload.strip():
        return np.empty((0, 10), dtype=np.float64)
    rows = np.loadtxt(io.BytesIO(payload), delimiter=",", ndmin=2)
    if rows.ndim != 2 or rows.shape[1] < 7 or (not np.isfinite(rows).all()):
        raise ValueError("Invalid MOT matrix")
    return rows


def project(bt_payload, sam_payload):
    raw = _matrix(bt_payload)
    samraw = _matrix(sam_payload)
    if not len(raw):
        return (bt_payload, dict(rows=0, empty=True))
    order = np.lexsort((raw[:, 1], raw[:, 0]))
    det = raw[order]
    sam = samraw[np.lexsort((samraw[:, 1], samraw[:, 0]))]
    v1 = relabel(det, sam)[0][:, 1].astype(np.int64)
    v2 = smooth(det, v1)
    labels = np.empty(len(raw), dtype=np.int64)
    labels[order] = v2
    lines = bt_payload.split(b"\n")
    if lines[-1] == b"":
        lines.pop()
    if len(lines) != len(raw):
        raise ValueError("Source physical row count changed")
    result = []
    for line, label in zip(lines, labels):
        fields = line.split(b",")
        fields[1] = str(int(label)).encode()
        result.append(b",".join(fields))
    for a, b in zip(lines, result):
        x, y = (a.split(b","), b.split(b","))
        if x[:1] + x[2:] != y[:1] + y[2:]:
            raise ValueError("Changed non-ID source token")
    if len(
        {(int(observation[0]), int(identity)) for observation, identity in zip(raw, labels)}
    ) != len(raw):
        raise ValueError("Duplicate frame/identity")
    return (b"\n".join(result) + b"\n", dict(rows=len(raw), census_v1={}, census_v2={}))
