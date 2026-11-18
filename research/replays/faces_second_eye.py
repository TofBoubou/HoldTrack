"""Replay of `holdtrack.faces.second_eye` against the frozen face tables, on the 92 videos (57 annotated, 35 test).

Shared faces, the table of the second model, the tables without shared faces and their projections are
recomputed and compared with the stored ones, array by array. The embeddings of the second model are the
stored ones. No annotation is read. The replay data are described in docs/REPRODUCE.md.

    python faces_second_eye.py <replay data>
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stored

from holdtrack.faces import second_eye as se

DATA = os.path.abspath(sys.argv[1])


def diff(a, b):
    assert sorted(a) == sorted(b), (sorted(a), sorted(b))
    return sum(
        int(
            not (
                np.asarray(a[k]).shape == np.asarray(b[k]).shape
                and np.array_equal(a[k], b[k], equal_nan=True)
            )
        )
        for k in a
    )


d = dict(
    shared=0,
    adaface=0,
    adaface_clean=0,
    adaface_clean_projected=0,
    arcface_clean=0,
    arcface_clean_projected=0,
)
n = 0
faces = 0
for SET, V in (
    (DATA + "/annotated57", DATA + "/annotated57/videos"),
    (DATA + "/test35", DATA + "/test35/identity_steps/videos"),
):
    for v in sorted(os.listdir(V)):
        F = SET + "/faces/{t}/" + v + ".npz"
        tr = np.loadtxt(f"{V}/{v}/tracks.txt", delimiter=",", ndmin=2)
        P = np.load(f"{SET}/pose133/{v}.npz")
        assert (P["frame"].astype(int) == tr[:, 0].astype(int)).all(), v
        sh, other, dist, eyes = se.shared_faces(P["frame"], P["kp"])
        z = stored.load(F.format(t="shared"))
        d["shared"] += int(
            not (
                np.array_equal(sh, z["shared"])
                and np.array_equal(other, z["other"])
                and np.array_equal(dist, z["dist"], equal_nan=True)
                and np.array_equal(eyes, z["eye_distance"], equal_nan=True)
            )
        )
        arc = stored.load(F.format(t="arcface"))
        b = stored.load(F.format(t="adaface_embeddings"))
        assert (b["rows"] == arc["rows"]).all(), v
        ada = se.adaface_table(arc, b["embedding"], b["norm_single"], b["mirror_cosine"])
        d["adaface"] += diff(ada, stored.load(F.format(t="adaface"))) > 0
        for name, t in (("adaface", ada), ("arcface", arc)):
            c = se.clean(t, sh)
            d[name + "_clean"] += diff(c, stored.load(F.format(t=name + "_clean"))) > 0
            p = se.project(c, tr[:, 0], tr[:, 1])
            d[name + "_clean_projected"] += (
                diff(p, stored.load(F.format(t=name + "_clean_projected"))) > 0
            )
        n += 1
        faces += len(arc["rows"])
print(f"{n} videos, {faces} faces ; videos whose output differs from the stored table: {d}")
print("VERDICT:", "IDENTICAL" if not any(d.values()) else "DIFFERENT")
