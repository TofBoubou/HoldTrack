"""Do several passes of a tracker make the same identity errors?

    python -m holdtrack.analysis.shared_errors --videos v1,v2,... --gt "data/{video}/gt/gt.txt" \
        --pass native="out/native/{video}.txt" --pass backward="out/backward/{video}.txt" [--pass ...] \
        [--boxes "out/boxes/{video}.txt"] [--forward native,exclusive --backward backward] [--out report.json]

Every file is in MOT format (frame, identity, x, y, width, height, ...). The passes are compared on a common set of
boxes: those of --boxes if given, else the annotated boxes. In each frame a pass gives a box the object whose box
matches it one to one (IoU >= 0.5). The objects of a pass are matched one to one with the annotated persons of the
video; the pass is wrong on a box when its object belongs to another person, or to none.

For each pair of passes, on the boxes that both label, the report gives: the errors of each, the boxes where both
are wrong, the number expected if they were wrong independently within each video (boxes in contact and away
from it counted apart), and, among the boxes where both are wrong, the share where they give the same wrong
person, with a bootstrap interval over videos and two chance levels.
"""

import argparse
import collections
import itertools
import json

import numpy as np
from scipy.optimize import linear_sum_assignment


def xyxy(a):
    return np.stack([a[:, 2], a[:, 3], a[:, 2] + a[:, 4], a[:, 3] + a[:, 5]], 1)


def iou_mat(a, b):
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


def read(f):
    q = np.loadtxt(f, delimiter=",")
    return q.reshape(-1, q.shape[-1])


def project(q, p, min_iou=0.5):
    """Object of the pass q on each box of p (-1 if none)."""
    qb = xyxy(q)
    pb = xyxy(p)
    qf = q[:, 0].astype(int)
    pf = p[:, 0].astype(int)
    out = np.full(len(p), -1, np.int64)
    for fr in np.unique(pf):
        pi = np.where(pf == fr)[0]
        qi = np.where(qf == fr)[0]
        if not len(qi):
            continue
        iou = iou_mat(pb[pi], qb[qi])
        r, c = linear_sum_assignment(-iou)
        for a, b in zip(r, c):
            if iou[a, b] >= min_iou:
                out[pi[a]] = int(q[qi[b], 1])
    return out


def annotate(g, p, contact_iou=0.30):
    """For each box of p: its annotated person (-1 if none), whether this person is in contact in the frame, and the
    number of other annotated persons whose box overlaps hers."""
    gb = xyxy(g)
    gf = g[:, 0].astype(int)
    gid = g[:, 1].astype(int)
    pb = xyxy(p)
    pf = p[:, 0].astype(int)
    person = np.full(len(p), -1)
    contact = np.zeros(len(p), bool)
    neighbors = np.zeros(len(p), int)
    for f in np.unique(gf):
        gi = np.where(gf == f)[0]
        pi = np.where(pf == f)[0]
        if not len(pi):
            continue
        m = iou_mat(gb[gi], gb[gi])
        np.fill_diagonal(m, 0)
        iou = iou_mat(gb[gi], pb[pi])
        r, c = linear_sum_assignment(-iou)
        for a, b in zip(r, c):
            if iou[a, b] >= 0.5:
                person[pi[b]] = gid[gi[a]]
                contact[pi[b]] = m[a].max() > contact_iou
                first = [j for j in range(len(gi)) if gid[gi[j]] == gid[gi[a]]][0]
                neighbors[pi[b]] = int(
                    sum(1 for j in range(len(gi)) if gid[gi[j]] != gid[gi[a]] and m[first, j] > 0)
                )
    return person, contact, neighbors


def persons_of(objects, person):
    """Annotated person of the object of the pass on each box (-2: object without a person; -1: no box)."""
    ok = (objects >= 0) & (person >= 0)
    us = np.unique(objects[ok])
    gs = np.unique(person[ok])
    iu = {u: i for i, u in enumerate(us)}
    ig = {g: i for i, g in enumerate(gs)}
    M = np.zeros((len(us), len(gs)))
    for u, g in zip(objects[ok], person[ok]):
        M[iu[u], ig[g]] += 1
    r, c = linear_sum_assignment(-M)
    m = {us[a]: gs[b] for a, b in zip(r, c) if M[a, b] > 0}
    return np.array([m.get(u, -2) if u >= 0 else -1 for u in objects])


def measure(videos, gt, passes, boxes=None, forward=(), backward=None, seed=20261112, draws=4000):
    """videos: list of names; gt, boxes: path patterns with {video}; passes: dict name -> path pattern."""
    names = list(passes)
    pairs = list(itertools.combinations(names, 2))
    T = {k: collections.Counter() for k in pairs}
    U = collections.Counter()
    PV = {}
    for v in videos:
        g = read(gt.format(video=v))
        p = read(boxes.format(video=v)) if boxes else g
        person, contact, neighbors = annotate(g, p)
        L = {k: project(read(f.format(video=v)), p) for k, f in passes.items()}
        P = {k: persons_of(x, person) for k, x in L.items()}
        for k in names:
            ok = (L[k] >= 0) & (person >= 0)
            U[k + " boxes"] += int(ok.sum())
            U[k + " errors"] += int((ok & (P[k] != person)).sum())
        for a, b in pairs:
            ok = (L[a] >= 0) & (L[b] >= 0) & (person >= 0)
            ea = ok & (P[a] != person)
            eb = ok & (P[b] != person)
            both = ea & eb
            c = T[(a, b)]
            same = both & (P[a] == P[b]) & (P[a] >= 0)
            c["boxes"] += int(ok.sum())
            c["errors_a"] += int(ea.sum())
            c["errors_b"] += int(eb.sum())
            c["both"] += int(both.sum())
            c["same"] += int(same.sum())
            PV.setdefault(v, {})[f"{a}+{b}"] = [int(ok.sum()), int(both.sum()), int(same.sum())]
            for st in (False, True):
                m = ok & (contact == st)
                n = int(m.sum())
                if n:
                    c["expected"] += float((ea & m).sum()) * float((eb & m).sum()) / n
            ix = np.where(both)[0]
            for q in np.unique(person[ix]):
                m = ix[person[ix] == q]
                ca = collections.Counter(P[a][m][P[a][m] >= 0].tolist())
                cb = collections.Counter(P[b][m][P[b][m] >= 0].tolist())
                c["chance_independent"] += sum(ca[x] * cb[x] for x in ca) / len(m)
            k = neighbors[ix]
            near = k > 0
            c["with_neighbor"] += int(near.sum())
            c["chance_uniform"] += float((1.0 / k[near]).sum())
            c["same_with_neighbor"] += int((same[ix] & near).sum())
    rng = np.random.default_rng(seed)
    vs = sorted(PV)
    out = dict(passes={}, pairs={})
    for k in names:
        out["passes"][k] = dict(
            boxes=U[k + " boxes"],
            errors=U[k + " errors"],
            error_rate=U[k + " errors"] / max(1, U[k + " boxes"]),
        )
    for (a, b), c in T.items():
        key = f"{a}+{b}"
        M = np.array([PV[v][key] for v in vs], float)
        ix = rng.integers(0, len(vs), (draws, len(vs)))
        bs = M[ix, 2].sum(1) / np.maximum(M[ix, 1].sum(1), 1)
        out["pairs"][key] = dict(
            c,
            ratio_to_independent=c["both"] / max(c["expected"], 1e-9),
            b_wrong_when_a_wrong=c["both"] / max(1, c["errors_a"]),
            same_wrong_person=c["same"] / max(1, c["both"]),
            same_wrong_person_interval=[
                float(np.percentile(bs, 2.5)),
                float(np.percentile(bs, 97.5)),
            ],
            chance_independent=c["chance_independent"] / max(1, c["both"]),
            chance_uniform_neighbors=c["chance_uniform"] / max(1, c["with_neighbor"]),
        )
    if forward and backward:
        fw = [f"{a}+{b}" for a, b in pairs if a in forward and b in forward]
        bw = [
            f"{a}+{b}"
            for a, b in pairs
            if (a in forward and b == backward) or (b in forward and a == backward)
        ]

        def pool(ks, ix=None):
            M = np.array(
                [[sum(PV[v][k][1] for k in ks), sum(PV[v][k][2] for k in ks)] for v in vs], float
            )
            return (
                M[:, 1].sum() / M[:, 0].sum()
                if ix is None
                else M[ix, 1].sum(1) / np.maximum(M[ix, 0].sum(1), 1)
            )

        ix = rng.integers(0, len(vs), (draws, len(vs)))
        d = pool(fw, ix) - pool(bw, ix)
        out["forward_against_backward"] = dict(
            two_forward_passes=float(pool(fw)),
            forward_and_backward=float(pool(bw)),
            difference=float(pool(fw) - pool(bw)),
            difference_interval=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
        )
    out["per_video"] = PV
    return out


def show(out):
    print("pass | boxes | errors | rate")
    for k, c in out["passes"].items():
        print(f"  {k:12s} | {c['boxes']:8d} | {c['errors']:7d} | {100 * c['error_rate']:.2f} %")
    print(
        "pair | both wrong | expected if independent | ratio | same wrong person (95% interval) | chance, independent draws | chance, uniform among neighbors"
    )
    for k, c in out["pairs"].items():
        lo, hi = c["same_wrong_person_interval"]
        print(
            f"  {k:26s} | {c['both']:7d} | {c['expected']:9.1f} | x{c['ratio_to_independent']:4.1f} | {100 * c['same_wrong_person']:5.1f} % ({100 * lo:.1f} to {100 * hi:.1f}) | {100 * c['chance_independent']:5.1f} % | {100 * c['chance_uniform_neighbors']:5.1f} %"
        )
    if "forward_against_backward" in out:
        f = out["forward_against_backward"]
        print(
            f"same wrong person: two forward passes {100 * f['two_forward_passes']:.1f} %, a forward and the backward pass {100 * f['forward_and_backward']:.1f} %, difference {100 * f['difference']:.1f} points ({100 * f['difference_interval'][0]:.1f} to {100 * f['difference_interval'][1]:.1f})"
        )


def main():
    ap = argparse.ArgumentParser(
        description="Shared identity errors of several passes of a tracker."
    )
    ap.add_argument("--videos", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pass", dest="passes", action="append", required=True)
    ap.add_argument("--boxes")
    ap.add_argument("--forward", default="")
    ap.add_argument("--backward")
    ap.add_argument("--out")
    a = ap.parse_args()
    videos = open(a.videos[1:]).read().split() if a.videos.startswith("@") else a.videos.split(",")
    out = measure(
        videos,
        a.gt,
        dict(x.split("=", 1) for x in a.passes),
        a.boxes,
        [x for x in a.forward.split(",") if x],
        a.backward,
    )
    show(out)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
