"""Body-mesh features of the box corrector, one row per (track row, box edge).

`mesh_features`: 161 columns, relative to the reference box of the stage: x and y of the 70 projected points
(140), the four edges of the mesh box (4), the extents of four groups of points (body without fingers and toes,
feet, right hand, left hand: 16), the estimated depth (1). For a left edge, x becomes 1 - x, left and right
points are exchanged, and so are the hands. Rows without a mesh are NaN.

`mesh_time_features`: 30 columns: the mesh box of the same identity at t-4, t-2, t-1, t+1, t+2, t+4 (24), and
the most overlapping neighbour: its mesh box (4), its relative depth (1), the IoU (1).
"""

from __future__ import annotations

import glob
import re

import numpy as np

from .pose_features import most_overlapping

_cache: dict = {}
_layout: dict = {}


def point_layout(names_file):
    """Names of the 70 points (from the public repository of the mesh model), their left-right exchange and
    the four groups of points."""
    key = str(names_file)
    if key not in _layout:
        names = re.findall(r'name="([^"]+)"', open(key).read())
        assert len(names) == 70
        mirror = np.array(
            [
                names.index(
                    n.replace("left_", "@").replace("right_", "left_").replace("@", "right_")
                )
                for n in names
            ]
        )
        assert (mirror[mirror] == np.arange(70)).all()

        def finger(n):
            return "finger" in n or "thumb" in n

        def foot(n):
            return "toe" in n or "heel" in n

        groups = [
            np.array([i for i, n in enumerate(names) if not (finger(n) or foot(n))]),
            np.array([i for i, n in enumerate(names) if foot(n)]),
            np.array([i for i, n in enumerate(names) if n.startswith("right_") and finger(n)]),
            np.array([i for i, n in enumerate(names) if n.startswith("left_") and finger(n)]),
        ]
        _layout[key] = (mirror, groups)
    return _layout[key]


def load(video, table, folder):
    """Arrays aligned on the rows of the table: 2D points (N, 70, 2), mesh box (N, 4), depth (N).
    NaN where the mesh is missing."""
    key = (video, str(folder))
    if key in _cache:
        return _cache[key]
    a = table
    n = len(a["frame"])
    kp = np.full((n, 70, 2), np.nan, np.float32)
    box = np.full((n, 4), np.nan, np.float32)
    depth = np.full(n, np.nan, np.float32)
    files = sorted(glob.glob(f"{folder}/{video}.npz"))
    assert files, ("no mesh for", video)
    for f in files:
        z = np.load(f)
        row = z["row"]
        assert (a["frame"][row] == z["frame"]).all() and (a["pid"][row] == z["pid"]).all(), (
            video,
            f,
        )
        kp[row] = z["kp2"]
        box[row] = z["mesh_box"]
        depth[row] = z["cam_t"][:, 2]
    _cache[key] = (kp, box, depth)
    return _cache[key]


def mesh_features(video, table, ref, rows, edges, folder, names_file):
    mirror, groups = point_layout(names_file)
    kp, box, depth = load(video, table, folder)
    r = ref[rows]
    w = (r[:, 2] - r[:, 0])[:, None]
    h = (r[:, 3] - r[:, 1])[:, None]
    g = (edges == 0)[:, None]
    x = (kp[rows][:, :, 0] - r[:, 0:1]) / w
    y = (kp[rows][:, :, 1] - r[:, 1:2]) / h
    m = box[rows].astype(float)
    mx0 = (m[:, 0:1] - r[:, 0:1]) / w
    mx1 = (m[:, 2:3] - r[:, 0:1]) / w
    my0 = (m[:, 1:2] - r[:, 1:2]) / h
    my1 = (m[:, 3:4] - r[:, 1:2]) / h
    out = [
        np.where(g, 1 - x[:, mirror], x),
        np.where(g, y[:, mirror], y),
        np.where(g, 1 - mx1, mx0),
        np.where(g, 1 - mx0, mx1),
        my0,
        my1,
    ]
    ex = []
    for c in groups:
        ex.append(
            (
                x[:, c].min(1, keepdims=True),
                x[:, c].max(1, keepdims=True),
                y[:, c].min(1, keepdims=True),
                y[:, c].max(1, keepdims=True),
            )
        )
    for k, k2 in ((0, 0), (1, 1), (2, 3), (3, 2)):  # group read for a left edge: hands exchanged
        a_, b_ = ex[k], ex[k2]
        out += [
            np.where(g, 1 - b_[1], a_[0]),
            np.where(g, 1 - b_[0], a_[1]),
            np.where(g, b_[2], a_[2]),
            np.where(g, b_[3], a_[3]),
        ]
    out.append(depth[rows][:, None].astype(float))
    q = np.concatenate(out, 1).astype(np.float32)
    assert q.shape == (len(rows), 161), q.shape
    return q


def mesh_time_features(video, table, ref, rows, edges, folder):
    a = table
    _, box, depth = load(video, table, folder)
    fr = a["frame"].astype(np.int64)
    pid = a["pid"].astype(np.int64)
    w = ref[:, 2] - ref[:, 0]
    h = ref[:, 3] - ref[:, 1]
    g = (edges == 0)[:, None]
    key = pid * 100000 + fr
    idx = {int(k): i for i, k in enumerate(key)}

    def rel(j_):
        o = j_ >= 0
        q = np.full((len(rows), 4), np.nan)
        r_ = ref[rows]
        m = box[np.maximum(j_, 0)].astype(float)
        q[o, 0] = (m[o, 0] - r_[o, 0]) / w[rows[o]]
        q[o, 1] = (m[o, 2] - r_[o, 0]) / w[rows[o]]
        q[o, 2] = (m[o, 1] - r_[o, 1]) / h[rows[o]]
        q[o, 3] = (m[o, 3] - r_[o, 1]) / h[rows[o]]
        return np.where(g, np.stack([1 - q[:, 1], 1 - q[:, 0], q[:, 2], q[:, 3]], 1), q)

    out = []
    for d_ in (-4, -2, -1, 1, 2, 4):
        out.append(rel(np.array([idx.get(int(k + d_), -1) for k in key])[rows]))
    other, overlap = most_overlapping(fr, ref)
    jv = other[rows]
    ov = jv >= 0
    dz = np.full(len(rows), np.nan)
    dz[ov] = (depth[jv[ov]] - depth[rows[ov]]) / depth[rows[ov]]
    out += [rel(jv), dz[:, None], overlap[rows][:, None]]
    q = np.concatenate(out, 1).astype(np.float32)
    assert q.shape == (len(rows), 30), q.shape
    return q
