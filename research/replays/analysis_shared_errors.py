"""Replay of `holdtrack.analysis.shared_errors` on the 57 annotated videos against the counts of the paper.

The boxes are those of the base system; the five passes are the stored ones. As in the measurement, the
objects of the exclusive and guarded passes are first renamed to the objects of the native pass, one to one.
Reads the annotations of the 57 videos. The replay data are described in docs/REPRODUCE.md.

    python analysis_shared_errors.py <replay data> <work folder>
"""

import json
import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from holdtrack.analysis import shared_errors as se

DATA, T = (os.path.abspath(x) for x in sys.argv[1:3])
SET = DATA + "/annotated57"
V = SET + "/videos"
for d in ("exclusive", "guarded"):
    os.makedirs(f"{T}/{d}", exist_ok=True)


def map_to_base(base, other, offset=10**7):
    bl = sorted(set(base))
    ol = sorted(set(other))
    bi = {x: i for i, x in enumerate(bl)}
    oi = {x: i for i, x in enumerate(ol)}
    m = np.zeros((len(ol), len(bl)))
    for b, o in zip(base, other):
        m[oi[o], bi[b]] += 1
    r, c = linear_sum_assignment(-m)
    mp = {ol[i]: bl[j] for i, j in zip(r, c) if m[i, j] > 0}
    nxt = offset
    for o in ol:
        if o not in mp:
            mp[o] = nxt
            nxt += 1
    return [mp[o] for o in other]


videos = sorted(os.listdir(V))
for v in videos:
    base = np.loadtxt(f"{V}/{v}/passes/native.txt", delimiter=",")[:, 1].astype(int).tolist()
    for i, name in enumerate(("exclusive", "guarded")):
        if not os.path.exists(f"{T}/{name}/{v}.txt"):
            q = np.loadtxt(f"{V}/{v}/passes/{name}.txt", delimiter=",")
            q[:, 1] = map_to_base(base, q[:, 1].astype(int).tolist(), offset=(i + 1) * 10**7)
            with open(f"{T}/{name}/{v}.txt", "w") as o:
                for x in q:
                    o.write(f"{int(x[0])},{int(x[1])},{','.join(repr(float(y)) for y in x[2:6])}\n")
P = {k: V + "/{video}/passes/" + k + ".txt" for k in ("native", "mirrored", "backward")}
P.update({k: f"{T}/{k}/" + "{video}.txt" for k in ("exclusive", "guarded")})
P = {k: P[k] for k in ("native", "exclusive", "guarded", "mirrored", "backward")}
out = se.measure(
    videos,
    DATA + "/dancetrack/{video}/gt/gt.txt",
    P,
    V + "/{video}/tracks.txt",
    ("native", "exclusive", "guarded"),
    "backward",
)
se.show(out)
json.dump(out, open(f"{T}/report.json", "w"), indent=1)
ref = json.load(open(SET + "/reference/shared_errors.json"))
gap = 0
for k, c in out["pairs"].items():
    r = ref["pairs"][k]
    e = [
        c["boxes"] - r["boxes"],
        c["errors_a"] - r["errors_a"],
        c["errors_b"] - r["errors_b"],
        c["both"] - r["both"],
        c["same"] - r["same"],
        round(c["expected"] - r["expected"], 6),
        round(c["chance_independent"] * c["both"] - r["same_expected_independent"], 6),
        round(
            c["chance_uniform_neighbors"] * c["with_neighbor"]
            - r["same_expected_uniform_neighbors"],
            6,
        ),
        c["with_neighbor"] - r["with_neighbor"],
    ]
    gap += sum(abs(x) for x in e)
    if any(e):
        print("DIFFERENCE", k, e)
for k, c in out["passes"].items():
    gap += abs(c["boxes"] - ref["passes"][k]["boxes"]) + abs(
        c["errors"] - ref["passes"][k]["errors"]
    )
print(
    "sum of the differences with the counts of the paper:",
    gap,
    "| VERDICT:",
    "IDENTICAL" if gap == 0 else "DIFFERENT",
)
