"""Where does a tracker lose HOTA?

    python -m holdtrack.analysis.where_lost --videos v1,v2,... --gt "data/{video}/gt/gt.txt" --tracks "out/{video}.txt" [--out report.json]

Both files are in MOT format (frame, identity, x, y, width, height, ...). The output of the tracker is corrected with
the annotation in four ways, one or several at a time:

    identity  a run of boxes of a person under an identity that is not her main identity takes the main identity;
    box       a matched box is placed on the annotated box;
    missed    an annotated person without a matched box receives her annotated box under her main identity;
    false     a box matched to no annotated person is removed.

HOTA is computed for the 16 combinations, and the gain of all four is split among them with the Shapley value,
which does not depend on the order of the corrections and sums to the total gain. The same gain is split by
place, with three players: every correction at contact, every correction away from it, and the removal of false
boxes. A person is in contact in a frame if, in this frame or one of the ten before, her annotated box overlaps
that of another annotated person at an IoU above 0.3.

In each frame, boxes and annotated persons are matched one to one at the largest total IoU, and pairs at an IoU of
at least 0.5 are kept. The main identity of a person is the one that holds most of her matched boxes.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from itertools import permutations
from multiprocessing import Pool

import numpy as np
from scipy.optimize import linear_sum_assignment

CONTACT_IOU = 0.3
CONTACT_FRAMES = 10
CASES = {}
for id_i in (0, 1):
    for B_ in (0, 1):
        for M in (0, 1):
            for X in (0, 1):
                CASES[f"f{id_i}{B_}{M}{X}"] = (id_i, id_i, B_, B_, M, M, X)
for C in (0, 1):
    for A in (0, 1):
        for X in (0, 1):
            CASES[f"c{C}{A}{X}"] = (C, A, C, A, C, A, X)


def judge(root=None):
    if root:
        sys.path.insert(0, root)
    from holdtrack import evaluation

    return evaluation


def iou2(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    i = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return i / (
        ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
        + ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
        - i
        + 1e-9
    )


def decompose(fr0, pid, ref, gf, gg, gb, J):
    """HOTA statistics of one video for every combination of corrections. Boxes as x1, y1, x2, y2."""
    fr0 = np.asarray(fr0).astype(int)
    pid = np.asarray(pid).astype(np.int64)
    ref = np.asarray(ref, float)
    n0 = len(fr0)
    gt_rows = [
        (int(f), int(g), float(b[0]), float(b[1]), float(b[2]), float(b[3]))
        for f, g, b in zip(gf, gg, gb)
    ]
    touch = set()
    for f in np.unique(gf):
        k = np.where(gf == f)[0]
        if len(k) > 1:
            M = iou2(gb[k], gb[k])
            np.fill_diagonal(M, 0)
            for j in np.where(M.max(1) > CONTACT_IOU)[0]:
                touch.add((int(f), int(gg[k[j]])))

    def contact(f, g):
        return any((f - d, g) in touch for d in range(CONTACT_FRAMES + 1))

    gid = np.full(n0, -1)
    gtb = np.full((n0, 4), np.nan)
    ip = defaultdict(list)
    ig = defaultdict(list)
    for j, f in enumerate(fr0.tolist()):
        ip[f].append(j)
    for j, f in enumerate(gf.tolist()):
        ig[f].append(j)
    for f, P in ip.items():
        Gk = ig.get(f)
        if not Gk:
            continue
        M = iou2(ref[P], gb[Gk])
        r, c = linear_sum_assignment(-np.where(M >= 0.5, M, 0))
        for a, b in zip(r, c):
            if M[a, b] >= 0.5:
                gid[P[a]] = gg[Gk[b]]
                gtb[P[a]] = gb[Gk[b]]
    main = {}
    seg = np.full(n0, -1)
    wrong_c = np.zeros(n0, bool)
    wrong_n = np.zeros(n0, bool)
    ns = 0
    for g in np.unique(gid[gid >= 0]):
        k = np.where(gid == g)[0]
        k = k[np.argsort(fr0[k], kind="stable")]
        main[int(g)] = int(Counter(pid[k].tolist()).most_common(1)[0][0])
        s = 0
        for i in range(1, len(k) + 1):
            if i == len(k) or pid[k[i]] != pid[k[s]]:
                seg[k[s:i]] = ns
                ns += 1
                if pid[k[s]] != main[int(g)]:
                    (wrong_c if contact(int(fr0[k[s]]), int(g)) else wrong_n)[k[s:i]] = True
                s = i
    matched = gid >= 0
    at_contact = np.array([contact(int(f), int(g)) if g >= 0 else False for f, g in zip(fr0, gid)])
    seen = set(zip(fr0[matched].tolist(), gid[matched].tolist()))
    missed = [(int(f), int(g), b) for f, g, b in zip(gf, gg, gb) if (int(f), int(g)) not in seen]
    mc = [contact(f, g) for f, g, _ in missed]

    def build(name_c, name_n, box_c, box_n, miss_c, miss_n, no_false):
        ids = pid.copy()
        B = ref.copy()
        prio = np.zeros(n0, bool)
        for m, on in ((wrong_c, name_c), (wrong_n, name_n)):
            if on:
                for j in np.where(m)[0]:
                    ids[j] = main[int(gid[j])]
                    prio[j] = True
        for m, on in ((matched & at_contact, box_c), (matched & ~at_contact, box_n)):
            if on:
                B[m] = gtb[m]
        keep = matched if no_false else np.ones(n0, bool)
        src = np.where(keep)[0].tolist()
        fr = fr0[keep].tolist()
        ids = ids[keep].tolist()
        B = B[keep].tolist()
        prio = prio[keep].tolist()
        pg = [-1] * len(fr)
        for (f, g, b), c in zip(missed, mc):
            if (c and miss_c) or ((not c) and miss_n):
                fr.append(f)
                ids.append(main.get(g, 8000000 + g))
                B.append(list(b))
                prio.append(True)
                src.append(-1)
                pg.append(g)
        occ = defaultdict(list)
        for j, (f, i) in enumerate(zip(fr, ids)):
            occ[(f, i)].append(j)
        for (f, i), L in occ.items():
            if len(L) > 1:
                L.sort(key=lambda j: (not prio[j], j))
                for j in L[1:]:
                    ids[j] = (
                        (
                            9000000 + int(seg[src[j]])
                            if seg[src[j]] >= 0
                            else 9500000 + int(pid[src[j]]) % 400000
                        )
                        if src[j] >= 0
                        else 8000000 + pg[j]
                    )
        assert len(set(zip(fr, ids))) == len(fr), "two boxes of one identity in a frame"
        rows = [
            (int(f), int(i), float(b[0]), float(b[1]), float(b[2]), float(b[3]))
            for f, i, b in zip(fr, ids, B)
        ]
        return J.sequence_sufficient_statistics(gt_rows, rows)

    done = {}
    out = {}
    for k, c in CASES.items():
        if c not in done:
            done[c] = build(*c)
        out[k] = done[c]
    counts = dict(
        boxes=int(n0),
        matched=int(matched.sum()),
        matched_at_contact=int((matched & at_contact).sum()),
        false=int((~matched).sum()),
        wrong_identity_at_contact=int(wrong_c.sum()),
        wrong_identity_away=int(wrong_n.sum()),
        missed_at_contact=int(sum(mc)),
        missed_away=int(len(mc) - sum(mc)),
        annotated=int(len(gf)),
    )
    return out, counts


def one_video(a):
    v, gt, tracks, root = a
    J = judge(root)
    G = np.loadtxt(gt.format(video=v), delimiter=",")
    G = G.reshape(-1, G.shape[-1])
    T = np.loadtxt(tracks.format(video=v), delimiter=",")
    T = T.reshape(-1, T.shape[-1])
    gb = np.stack([G[:, 2], G[:, 3], G[:, 2] + G[:, 4], G[:, 3] + G[:, 5]], 1)
    tb = np.stack([T[:, 2], T[:, 3], T[:, 2] + T[:, 4], T[:, 3] + T[:, 5]], 1)
    return v, decompose(T[:, 0], T[:, 1], tb, G[:, 0].astype(int), G[:, 1].astype(int), gb, J)


def shapley(val, n):
    phi = [0.0] * n
    perms = list(permutations(range(n)))
    for p in perms:
        S = [0] * n
        for j in p:
            before = val(tuple(S))
            S[j] = 1
            phi[j] += val(tuple(S)) - before
    return [x / len(perms) for x in phi]


def measure(videos, gt, tracks, root=None, workers=2):
    J = judge(root)
    with Pool(workers) as pool:
        res = pool.map(one_video, [(v, gt, tracks, root) for v in videos])
    H = {
        c: 100 * float(J.combine_sufficient_statistics([r[1][0][c] for r in res])["HOTA"])
        for c in CASES
    }
    b = H["f0000"]
    counts = Counter()
    for r in res:
        counts.update(r[1][1])
    four = ("identity", "box", "missed", "false")
    place = ("at contact", "away from contact", "false boxes")
    return dict(
        HOTA=b,
        missing=100 - b,
        recovered=H["f1111"] - b,
        with_all=H["f1111"],
        alone=dict(zip(four, (H[c] - b for c in ("f1000", "f0100", "f0010", "f0001")))),
        last=dict(zip(four, (H["f1111"] - H[c] for c in ("f0111", "f1011", "f1101", "f1110")))),
        shapley=dict(zip(four, shapley(lambda S: H["f" + "".join(map(str, S))], 4))),
        place_alone=dict(zip(place, (H[c] - b for c in ("c100", "c010", "c001")))),
        place_shapley=dict(zip(place, shapley(lambda S: H["c" + "".join(map(str, S))], 3))),
        counts=dict(counts),
        cases=H,
    )


def show(o):
    def f(d):
        return " ; ".join(f"{k} {v:+.3f}" for k, v in d.items())

    print(
        "HOTA {:.3f} ; missing {:.3f} ; recovered by the four corrections {:+.3f} (HOTA {:.3f})".format(
            o["HOTA"], o["missing"], o["recovered"], o["with_all"]
        )
    )
    print("one correction alone   : " + f(o["alone"]))
    print("one correction last    : " + f(o["last"]))
    print("Shapley value          : " + f(o["shapley"]))
    print("by place, alone        : " + f(o["place_alone"]))
    print("by place, Shapley      : " + f(o["place_shapley"]))


def main():
    ap = argparse.ArgumentParser(
        description="Where a tracker loses HOTA: oracle corrections and Shapley shares."
    )
    ap.add_argument("--videos", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--out")
    ap.add_argument("--package-root")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    videos = open(a.videos[1:]).read().split() if a.videos.startswith("@") else a.videos.split(",")
    o = measure(videos, a.gt, a.tracks, a.package_root, a.workers)
    show(o)
    if a.out:
        json.dump(o, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
