"""Cut a name where its face changes, then link the parts.

A *fragment* is a run of rows of one identity without a gap, without a change of track, and entirely free of
contact (or entirely in contact). Inside a free fragment the tracker cannot have changed person. A name is cut
between two fragments when the faces before and after do not resemble each other; parts of at least `faces`
faces are then linked when they are each other's best resemblance, above a threshold fitted on the video.
Nothing here reads an annotation.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .pieces import FaceTable

GAP = 10  # frames
CONTACT = 0.3
SHORT = 5  # a free run shorter than this, between two contacts, joins the contact
OVERLAP = 5  # frames two linked parts may share
NEW_NAME = 6 * 10**6


def unit(X):
    return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-9)


def cosine(a, b):
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9))


def iou_matrix(a, b):
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


def contact(frame, box):
    """Strongest overlap of every row with another box of its frame."""
    c = np.zeros(len(frame))
    for f in np.unique(frame):
        ix = np.where(frame == f)[0]
        if len(ix) > 1:
            io = iou_matrix(box[ix], box[ix])
            np.fill_diagonal(io, 0)
            c[ix] = io.max(1)
    return c


def fragments(frame, ids, track, ct):
    """Fragment of every row, and the table of fragments (identity, first frame, last frame, rows, free)."""
    frag = np.full(len(frame), -1, np.int64)
    T = []
    k = 0
    for name in np.unique(ids):
        ix = np.where(ids == name)[0]
        ix = ix[np.argsort(frame[ix], kind="stable")]
        if not len(ix):
            continue
        cut = np.where((np.diff(frame[ix]) > GAP) | (np.diff(track[ix]) != 0))[0]
        st = np.r_[0, cut + 1]
        en = np.r_[cut + 1, len(ix)]
        for s, e in zip(st, en):
            P = ix[s:e]
            free = ct[P] < CONTACT
            b = np.r_[0, np.where(np.diff(free.astype(int)) != 0)[0] + 1, len(P)]
            runs = [[P[x:y], bool(free[x])] for x, y in zip(b[:-1], b[1:])]
            for r in runs:
                if r[1] and len(r[0]) < SHORT and len(runs) > 1:
                    r[1] = False
            merged = []
            for r in runs:
                if merged and merged[-1][1] == r[1]:
                    merged[-1][0] = np.concatenate([merged[-1][0], r[0]])
                else:
                    merged.append(r)
            for L, is_free in merged:
                frag[L] = k
                T.append(
                    (
                        int(name),
                        int(frame[L].min()),
                        int(frame[L].max()),
                        int(len(L)),
                        bool(is_free),
                    )
                )
                k += 1
    return frag, T


def moment_directions(X, fr, ids, half=30, step=10):
    """Principal directions of the local mean of the faces over time (balanced between the names)."""
    centres = np.arange(int(fr.min()), int(fr.max()) + 1, step)
    o = np.argsort(fr, kind="stable")
    frs = fr[o]
    M = []
    for c in centres:
        lo = np.searchsorted(frs, c - half, "left")
        hi = np.searchsorted(frs, c + half, "right")
        ix = o[lo:hi]
        if len(ix) < 6:
            continue
        u, inv, cnt = np.unique(ids[ix], return_inverse=True, return_counts=True)
        good = np.where(cnt >= 2)[0]
        if len(good) < 3:
            continue
        S = np.zeros((len(u), X.shape[1]), np.float64)
        np.add.at(S, inv, X[ix])
        M.append((S[good] / cnt[good, None]).mean(0))
    if len(M) < 3:
        return np.zeros((0, X.shape[1]), np.float32)
    M = np.array(M, np.float32)
    return np.linalg.svd(M - M.mean(0), full_matrices=False)[2]


def faces(table: FaceTable, rows: int, frame, track, directions=3):
    """Embeddings centred on the video, without the first directions of the moment, per row; their norm;
    the rows that have a face."""
    X = unit(table.embedding.astype(np.float32))
    E = np.zeros((rows, X.shape[1]), np.float32)
    W = np.zeros(rows, np.float32)
    has = np.zeros(rows, bool)
    idx = table.index
    if len(idx):
        g = X.mean(0)
        Y = X - g
        if len(X) >= 50:
            P = moment_directions(X, frame[idx], track[idx])[:directions]
            Y = Y - (Y @ P.T) @ P
            E[idx] = unit(Y).astype(np.float32)
        else:
            E[idx] = unit(Y)
        W[idx] = table.norm
        has[idx] = True
    return E, W, has


def rename(frame, ids, frag, group, holder_first=True, new_name=NEW_NAME):
    """New names from groups of fragments (group -1: fragment not classified).

    Every group takes the current name it carries most (optimal assignment), or a new name. The other rows
    follow, inside their current name, the closest classified row in time. When two rows of a frame end under
    one name, the classified one keeps it; at equal rank, with `holder_first`, the row that already carried it
    keeps it. The other takes back its former name if it is free in the frame, else a new name."""
    g = np.where(frag >= 0, group[np.maximum(frag, 0)], -1)
    new = ids.copy()
    cl = g >= 0
    if not cl.any():
        return new
    Gs = np.unique(g[cl])
    Ns = np.unique(ids)
    M = np.zeros((len(Gs), len(Ns)))
    gi = {x: i for i, x in enumerate(Gs)}
    ni = {x: i for i, x in enumerate(Ns)}
    np.add.at(M, (np.array([gi[x] for x in g[cl]]), np.array([ni[x] for x in ids[cl]])), 1)
    r, c = linear_sum_assignment(-M)
    name = {int(Gs[i]): int(new_name + i) for i in range(len(Gs))}
    for i, j in zip(r, c):
        if M[i, j] > 0:
            name[int(Gs[i])] = int(Ns[j])
    new[cl] = np.array([name[int(x)] for x in g[cl]])
    for N in Ns:
        ix = np.where(ids == N)[0]
        a = ix[cl[ix]]
        b = ix[~cl[ix]]
        if not len(a) or not len(b):
            continue
        o = np.argsort(frame[a], kind="stable")
        fa = frame[a][o]
        na = new[a][o]
        k = np.searchsorted(fa, frame[b])
        k0 = np.clip(k - 1, 0, len(fa) - 1)
        k1 = np.clip(k, 0, len(fa) - 1)
        pick = np.where(np.abs(fa[k0] - frame[b]) <= np.abs(fa[k1] - frame[b]), k0, k1)
        new[b] = na[pick]
    fresh = new_name + 10**5
    o = np.lexsort((new != ids, ~cl, frame)) if holder_first else np.lexsort((~cl, frame))
    seen = {}
    for i in o:
        key = (int(frame[i]), int(new[i]))
        if key not in seen:
            seen[key] = i
            continue
        if (int(frame[i]), int(ids[i])) not in seen and not (
            (frame == frame[i]) & (new == ids[i])
        ).any():
            new[i] = ids[i]
            seen[(int(frame[i]), int(ids[i]))] = i
        else:
            fresh += 1
            new[i] = fresh
            seen[(int(frame[i]), fresh)] = i
    return new


def _link_threshold(frame, frag, T, S, N, faces_min, floor):
    """Threshold of the links fitted on the video: resemblance between free fragments of at least `faces_min`
    faces that are present together, hence of different persons."""
    ks = [k for k in range(len(T)) if T[k][4] and N[k] >= faces_min]
    im = [set(frame[frag == k].tolist()) for k in ks]
    c = [
        cosine(S[ks[i]], S[ks[j]])
        for i in range(len(ks))
        for j in range(i + 1, len(ks))
        if im[i] & im[j]
    ]
    if len(c) < 20:
        return floor, len(c)
    return max(floor, float(np.quantile(c, 0.995)) if len(c) >= 200 else float(max(c))), len(c)


def cut_and_link_once(
    frame,
    track,
    box,
    ids,
    table: FaceTable,
    cut=0.146,
    link=0.470,
    faces_min=30,
    fit=True,
    rejoin=True,
    holder_first=True,
):
    """One pass. `frame`, `track` and `box` (x1, y1, x2, y2) are the rows of the tracks; `ids` their current
    names. Returns the new names and the log (cuts, links)."""
    n = len(ids)
    ct = contact(frame, box)
    frag, T = fragments(frame, ids, track, ct)
    E, W, has = faces(table, n, frame, track)
    K = len(T)
    S = np.zeros((K, E.shape[1]), np.float64)
    N = np.zeros(K, np.int64)
    for k in range(K):
        if T[k][4]:
            ix = np.where((frag == k) & has)[0]
            N[k] = len(ix)
            if len(ix):
                S[k] = (E[ix] * W[ix, None]).sum(0)
    segs = []
    log = dict(cuts=[], links=[])
    TL, pairs = _link_threshold(frame, frag, T, S, N, faces_min, link) if fit else (link, 0)
    log["link_threshold"] = round(TL, 3)
    log["pairs_present_together"] = pairs
    for name in np.unique(ids):
        ks = sorted(
            [k for k in range(K) if T[k][0] == name and T[k][4] and N[k] > 0], key=lambda k: T[k][1]
        )
        if not ks:
            continue
        cs = np.concatenate([np.zeros((1, S.shape[1])), np.cumsum(S[ks], 0)])
        ns = np.r_[0, np.cumsum(N[ks])]

        def split(lo, hi):
            best = None
            for i in range(lo + 1, hi):
                if ns[i] - ns[lo] < faces_min or ns[hi] - ns[i] < faces_min:
                    continue
                s = cosine(cs[i] - cs[lo], cs[hi] - cs[i])
                if best is None or s < best[0]:
                    best = (s, i)
            if best is not None and best[0] < cut:
                log["cuts"].append(
                    dict(
                        name=int(name),
                        after_frame=int(T[ks[best[1] - 1]][2]),
                        before_frame=int(T[ks[best[1]]][1]),
                        cosine=round(best[0], 3),
                    )
                )
                return split(lo, best[1]) + split(best[1], hi)
            return [(lo, hi)]

        parts = split(0, len(ks))
        for lo, hi in parts:
            kk = ks[lo:hi]
            t0 = min(T[k][1] for k in kk)
            t1 = max(T[k][2] for k in kk)
            segs.append(
                dict(
                    name=int(name),
                    frags=kk,
                    sum=cs[hi] - cs[lo],
                    n=int(ns[hi] - ns[lo]),
                    cut=len(parts) > 1,
                    t0=t0,
                    t1=t1,
                    frames=set(frame[(ids == name) & (frame >= t0) & (frame <= t1)].tolist()),
                )
            )
    G = [
        dict(members=[i], sum=s["sum"].copy(), n=s["n"], cut=s["cut"], frames=set(s["frames"]))
        for i, s in enumerate(segs)
    ]
    while True:
        ok = [i for i, g in enumerate(G) if g is not None and g["n"] >= faces_min]
        best = {}
        for x in range(len(ok)):
            for y in range(x + 1, len(ok)):
                A, B = G[ok[x]], G[ok[y]]
                if not (A["cut"] or B["cut"]) or len(A["frames"] & B["frames"]) > OVERLAP:
                    continue
                s = cosine(A["sum"], B["sum"])
                for u, w in ((ok[x], ok[y]), (ok[y], ok[x])):
                    if u not in best or s > best[u][0]:
                        best[u] = (s, w)
        cand = [
            (s, u, w)
            for u, (s, w) in best.items()
            if s >= TL and best.get(w, (None, None))[1] == u and u < w
        ]
        if not cand:
            break
        s, u, w = max(cand)
        A, B = G[u], G[w]
        log["links"].append(
            dict(
                cosine=round(s, 3),
                a=[
                    (segs[i]["name"], segs[i]["t0"], segs[i]["t1"], segs[i]["n"])
                    for i in A["members"]
                ],
                b=[
                    (segs[i]["name"], segs[i]["t0"], segs[i]["t1"], segs[i]["n"])
                    for i in B["members"]
                ],
            )
        )
        G[u] = dict(
            members=A["members"] + B["members"],
            sum=A["sum"] + B["sum"],
            n=A["n"] + B["n"],
            cut=True,
            frames=A["frames"] | B["frames"],
        )
        G[w] = None
    group = np.full(K, -1, np.int64)
    rest = {}  # a part that is linked to nothing stays with the other unlinked parts of its name
    for gi, g in enumerate(G):
        if g is None:
            continue
        if len(g["members"]) == 1:
            gi = rest.setdefault(segs[g["members"][0]]["name"], gi)
        g["gid"] = gi
    if (
        rejoin
    ):  # a linked group never present together with the unlinked rest of one of its names joins it
        for g in G:
            if g is None or len(g["members"]) < 2:
                continue
            names = sorted(
                set(segs[i]["name"] for i in g["members"]),
                key=lambda N_: (
                    -sum(len(segs[i]["frames"]) for i in g["members"] if segs[i]["name"] == N_)
                ),
            )
            for N_ in names:
                if N_ not in rest:
                    continue
                taken = set()
                for i in g["members"]:
                    if segs[i]["name"] == N_:
                        taken |= segs[i]["frames"]
                frames_rest = set(frame[ids == N_].tolist()) - taken
                frames_group = (
                    set().union(*[segs[i]["frames"] for i in g["members"] if segs[i]["name"] != N_])
                    if any(segs[i]["name"] != N_ for i in g["members"])
                    else set()
                )
                if len(frames_rest & frames_group) <= OVERLAP:
                    log.setdefault("rejoined", []).append(
                        dict(
                            name=int(N_),
                            group=[
                                (segs[i]["name"], segs[i]["t0"], segs[i]["t1"])
                                for i in g["members"]
                            ],
                        )
                    )
                    g["gid"] = rest[N_]
                    break
    for g in G:
        if g is None:
            continue
        for i in g["members"]:
            for k in segs[i]["frags"]:
                group[k] = g["gid"]
    new = rename(frame, ids, frag, group, holder_first) if log["links"] else ids.copy()
    key = frame.astype(np.int64) * 10**9 + new
    assert len(np.unique(key)) == len(key), "two rows of one frame under one name"
    log["changed"] = int((new != ids).sum())
    return new, log


def cut_and_link(frame, track, box, ids, table: FaceTable, turns=5, **settings):
    """Passes repeated while a pass links something (five at most). The fragments are always delimited by
    the tracks of origin; the faces are read on the table without shared faces."""
    frame = np.asarray(frame).astype(int)
    track = np.asarray(track).astype(np.int64)
    ids = np.asarray(ids).astype(np.int64)
    new, log = cut_and_link_once(frame, track, box, ids, table, **settings)
    logs = [log]
    for _ in range(turns - 1):
        if not logs[-1]["links"]:
            break
        new, log = cut_and_link_once(frame, track, box, new, table, **settings)
        logs.append(log)
    return new, logs
