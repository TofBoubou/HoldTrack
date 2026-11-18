"""Keypoint features of the box corrector, one row per (track row, box edge).

126 columns, relative to the reference box of the stage and mirrored for the left edges: 57 from the 17
keypoints of the row, 37 from the same keypoints at the neighbouring frames of the identity and from the most
overlapping neighbour, 32 from the 133 keypoints.
"""

from __future__ import annotations

import warnings

import numpy as np

MIRROR = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15])


def most_overlapping(fr, ref):
    """For each row, the row of the same frame whose reference box overlaps most (IoU above 0.1), or -1,
    and that IoU."""
    n = len(fr)
    other = np.full(n, -1)
    overlap = np.zeros(n)
    o_ = np.argsort(fr, kind="stable")
    f_, st_ = np.unique(fr[o_], return_index=True)
    st_ = list(st_) + [n]
    for i_ in range(len(f_)):
        ii = o_[st_[i_] : st_[i_ + 1]]
        if len(ii) < 2:
            continue
        b = ref[ii]
        iw = np.clip(
            np.minimum(b[:, None, 2], b[None, :, 2]) - np.maximum(b[:, None, 0], b[None, :, 0]),
            0,
            None,
        )
        ih = np.clip(
            np.minimum(b[:, None, 3], b[None, :, 3]) - np.maximum(b[:, None, 1], b[None, :, 1]),
            0,
            None,
        )
        inter = iw * ih
        ar = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        M = inter / (ar[:, None] + ar[None] - inter + 1e-9)
        np.fill_diagonal(M, 0)
        k_ = M.argmax(1)
        overlap[ii] = M.max(1)
        other[ii] = np.where(M.max(1) > 0.1, ii[k_], -1)
    return other, overlap


def pose_features(video, table, ref, rows, edges, t1, t2, folder17, folder133):
    """t1, t2: the two confidence thresholds of the 133-keypoint features."""
    warnings.simplefilter("ignore")
    a = table
    z = np.load(f"{folder17}/{video}.npz")
    z2 = np.load(f"{folder133}/{video}.npz")
    for q in (z, z2):
        assert (q["frame"] == a["frame"]).all() and (q["pid"] == a["pid"]).all(), video
    w = (ref[:, 2] - ref[:, 0])[:, None]
    h = (ref[:, 3] - ref[:, 1])[:, None]
    kp = z["kp"].astype(float)
    g = (edges == 0)[:, None]
    # 57 features of the 17 keypoints
    x = ((kp[:, :, 0] - ref[:, 0:1]) / w)[rows]
    y = ((kp[:, :, 1] - ref[:, 1:2]) / h)[rows]
    s = kp[:, :, 2][rows]
    x = np.where(g, 1 - x[:, MIRROR], x)
    y = np.where(g, y[:, MIRROR], y)
    s = np.where(g, s[:, MIRROR], s)
    ok = s > 0.3
    with np.errstate(all="ignore"):
        xm = np.where(ok, x, np.nan)
        ym = np.where(ok, y, np.nan)
        ext = np.stack(
            [
                np.nanmin(xm, 1),
                np.nanmax(xm, 1),
                np.nanmin(ym, 1),
                np.nanmax(ym, 1),
                ok.sum(1),
                (s > 0.5).sum(1),
            ],
            1,
        )
    out = [x, y, s, ext]
    # 37 features: neighbouring frames of the same identity, and the most overlapping neighbour
    ok0 = kp[:, :, 2] > 0.3
    with np.errstate(all="ignore"):
        X0 = np.where(ok0, kp[:, :, 0], np.nan)
        Y0 = np.where(ok0, kp[:, :, 1], np.nan)
        E = np.stack([np.nanmin(X0, 1), np.nanmax(X0, 1), np.nanmin(Y0, 1), np.nanmax(Y0, 1)], 1)
        ms = kp[:, :, 2].mean(1)
    fr = a["frame"].astype(np.int64)
    pid = a["pid"].astype(np.int64)
    key = pid * 100000 + fr
    idx = {int(k): i for i, k in enumerate(key)}

    def rel(j_, rows_, left):
        o = j_ >= 0
        q = np.full((len(rows_), 4), np.nan)
        r_ = ref[rows_]
        q[o, 0] = (E[j_[o], 0] - r_[o, 0]) / w[rows_[o], 0]
        q[o, 1] = (E[j_[o], 1] - r_[o, 0]) / w[rows_[o], 0]
        q[o, 2] = (E[j_[o], 2] - r_[o, 1]) / h[rows_[o], 0]
        q[o, 3] = (E[j_[o], 3] - r_[o, 1]) / h[rows_[o], 0]
        return np.where(left, np.stack([1 - q[:, 1], 1 - q[:, 0], q[:, 2], q[:, 3]], 1), q)

    for d_ in (-4, -2, -1, 1, 2, 4):
        j_ = np.array([idx.get(int(k + d_), -1) for k in key])[rows]
        out.append(rel(j_, rows, g))
        out.append(np.where(j_ >= 0, ms[np.maximum(j_, 0)], np.nan)[:, None])
    other, overlap = most_overlapping(fr, ref)
    jv = other[rows]
    ov_ = jv >= 0
    anchor = np.full((len(rows), 2), np.nan)
    axv = (kp[:, 5, 0] + kp[:, 6, 0]) / 2
    ayv = (kp[:, 5, 1] + kp[:, 6, 1]) / 2
    anchor[ov_, 0] = (axv[jv[ov_]] - ref[rows[ov_], 0]) / w[rows[ov_], 0]
    anchor[ov_, 1] = (ayv[jv[ov_]] - ref[rows[ov_], 1]) / h[rows[ov_], 0]
    anchor[:, 0] = np.where(g[:, 0], 1 - anchor[:, 0], anchor[:, 0])
    out += [rel(jv, rows, g), anchor, overlap[rows][:, None]]
    # 32 features of the 133 keypoints
    k2 = z2["kp"].astype(float)
    x2 = ((k2[:, :, 0] - ref[:, 0:1]) / w)[rows]
    y2 = ((k2[:, :, 1] - ref[:, 1:2]) / h)[rows]
    s2 = k2[:, :, 2][rows]
    g1 = edges == 0

    def extent(cols, thr, score=True):
        okc = s2[:, cols] > thr
        xm_ = np.where(okc, x2[:, cols], np.nan)
        ym_ = np.where(okc, y2[:, cols], np.nan)
        with np.errstate(all="ignore"):
            a0 = np.nanmin(xm_, 1)
            a1 = np.nanmax(xm_, 1)
            q = [
                np.where(g1, 1 - a1, a0),
                np.where(g1, 1 - a0, a1),
                np.nanmin(ym_, 1),
                np.nanmax(ym_, 1),
            ]
        return np.stack(q + ([s2[:, cols].mean(1)] if score else []), 1)

    everything = np.arange(133)
    feet = np.arange(17, 23)
    face = np.arange(23, 91)
    left_hand = extent(np.arange(91, 112), t1)
    right_hand = extent(np.arange(112, 133), t1)
    counts = np.stack(
        [
            (s2[:, :17] > t1).sum(1),
            (s2[:, 17:23] > t1).sum(1),
            (s2[:, 91:] > t1).sum(1),
            (s2[:, 23:91] > t1).sum(1),
        ],
        1,
    )
    out += [
        extent(everything, t1, False),
        extent(everything, t2, False),
        extent(feet, t1),
        np.where(g, right_hand, left_hand),
        np.where(g, left_hand, right_hand),
        extent(face, t1),
        counts,
    ]
    r = np.concatenate([o.astype(np.float32) for o in out], 1)
    assert r.shape == (len(rows), 126), r.shape
    return r
