"""Replay of the agreement module against the frozen identities, on the 57 annotated videos or on the test set.

For each video: the frozen identities before the two steps, then the identities expected after the link step
and after the box step. No annotation is read. The replay data are described in docs/REPRODUCE.md.

    python identity_agree.py <replay data> <annotated57|test35>
"""

import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from holdtrack.identity import agree

DATA = os.path.abspath(sys.argv[1])
name = sys.argv[2]
if name == "annotated57":
    V = f"{DATA}/annotated57/videos"
    E0, E1, E2 = (
        np.load(f"{DATA}/annotated57/identities/{k}.npz")
        for k in ("merge_two", "agree_links", "agree_boxes")
    )
else:
    V = f"{DATA}/test35/identity_steps/videos"
    E0, E1, E2 = (
        np.load(f"{DATA}/test35/identity_steps/work/{k}/ids.npz")
        for k in ("s05_merge_two", "s06_agree_links", "s07_agree_boxes")
    )


def clear(vis):
    vis = vis.astype(np.float64)
    return np.isfinite(vis[:, 0]) & (np.nanmax(vis[:, :4], 1) < 0.05)


d1 = d2 = n = 0
bad = []
ops = 0
for v in sorted(E0.files):
    t = np.loadtxt(f"{V}/{v}/tracks.txt", delimiter=",", ndmin=2)
    k = len(t)
    p = np.concatenate([t[:, :6], np.ones((k, 1)), -np.ones((k, 3))], 1)
    p[:, 1] = np.asarray(E0[v]).astype(np.int64).ravel()
    cl = clear(np.load(f"{V}/{v}/visibility.npy"))
    assert len(cl) == len(p), v
    bo = agree.project(np.loadtxt(f"{V}/{v}/passes/backward.txt", delimiter=","), p)
    mo = agree.project(np.loadtxt(f"{V}/{v}/passes/mirrored.txt", delimiter=","), p)
    log = []
    a = agree.link_step(p, cl, bo, mo, 20.0, log)
    q = p.copy()
    q[:, 1] = a
    b = agree.box_step(q, cl, bo, mo, 20.0, log)
    ops += len(log)
    x1 = int((a != np.asarray(E1[v]).astype(np.int64).ravel()).sum())
    x2 = int((b != np.asarray(E2[v]).astype(np.int64).ravel()).sum())
    d1 += x1
    d2 += x2
    n += len(p)
    if x1 or x2:
        bad.append((v[-4:], x1, x2))
    print(v, len(p), "rows | differences after the links", x1, "| after the boxes", x2, flush=True)
print(
    f"{name}: {len(E0.files)} videos, {n} rows, {ops} operations ; rows that differ from the frozen identities after the link step: {d1} ; after the box step: {d2} ; videos that differ: {bad}"
)
print(
    "module:",
    hashlib.sha256(open(agree.__file__, "rb").read()).hexdigest()[:16],
    "| VERDICT:",
    "IDENTICAL" if d1 == 0 and d2 == 0 else "DIFFERENT",
)
