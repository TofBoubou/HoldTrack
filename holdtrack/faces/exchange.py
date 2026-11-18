"""Exchange without a cut: two identities swapped over an interval that the faces find themselves.

For two identities present together, the readable faces of both are split in two clusters; the contiguous run
of faces that sit on the wrong cluster gives the interval. The exchange is kept when the faces of one side,
inside the interval, resemble the gallery of the other identity outside it. Its bounds are moved to the frames
of strongest overlap between the two boxes. With two face models, an interval found by one model is kept only
if the other validates it with the same thresholds. Nothing here reads an annotation.
"""

from __future__ import annotations

import numpy as np

from .pieces import READABLE_NORM, FaceTable, Rows

GALLERY_MIN = 12
SIDE_MIN = 12  # faces of a side inside the interval
FIRST = 0.4
MARGIN = 0.4
COMMON = 30  # frames where both identities are present
PASSES = 3


def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    i = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def best_run(s):
    """Contiguous run of maximal sum: (sum, first, last included)."""
    best = -1e18
    bi = bj = 0
    cur = 0.0
    ci = 0
    for n, x in enumerate(s):
        if cur <= 0:
            cur = float(x)
            ci = n
        else:
            cur += float(x)
        if cur > best:
            best = cur
            bi = ci
            bj = n
    return best, bi, bj


def two_means(X):
    n = len(X)
    rng = np.random.default_rng(0)
    k = rng.choice(n, min(n, 200), replace=False)
    S = X[k] @ X[k].T
    i, j = np.unravel_index(int(S.argmin()), S.shape)
    mu = np.stack([X[k[i]], X[k[j]]])
    lab = None
    for _ in range(20):
        l2 = (X @ mu.T).argmax(1)
        if lab is not None and (l2 == lab).all():
            break
        lab = l2
        if (lab == 0).sum() == 0 or (lab == 1).sum() == 0:
            return None
        mu = np.stack([X[lab == c].mean(0) for c in (0, 1)])
        mu /= np.maximum(np.linalg.norm(mu, axis=1, keepdims=True), 1e-9)
    return mu[0], mu[1], lab


class Reader:
    """The identities of one video read by one face model. `boxes` are the boxes of the rows (x1, y1, x2, y2);
    `short` the short-track rows (frame, identity, ...)."""

    def __init__(self, rows: Rows, boxes, ids, short, table: FaceTable, contact_limit=0.3):
        self.fr = rows.frame.astype(int)
        ct = rows.contact
        N0 = len(self.fr)
        E, q = table.dense(N0)
        lis = q >= READABLE_NORM
        self.B = boxes
        self.readable = bool(lis.sum() >= 50)
        self.ids = np.asarray(ids).astype(np.int64).ravel().copy()
        self.short = np.array(short, dtype=float).copy()
        self.usable = lis & (ct < contact_limit)
        if self.readable:
            Ec = E - E[lis].mean(0)
            Ec /= np.maximum(np.linalg.norm(Ec, axis=1, keepdims=True), 1e-9)
            self.Ec = Ec
            self.W = Ec * q[:, None]

    def gallery(self, L):
        if len(L) < GALLERY_MIN:
            return None
        keep = L
        for _ in range(3):
            g = self.W[keep].sum(0)
            g /= np.linalg.norm(g)
            s_ = self.Ec[L] @ g
            keep = L[s_ >= np.quantile(s_, 0.4)]
        g = self.W[keep].sum(0)
        return g / np.linalg.norm(g)

    def mean(self, L):
        g = self.W[L].sum(0)
        return g / max(np.linalg.norm(g), 1e-9)

    def index(self):
        fr, ids = self.fr, self.ids
        self.names = [int(x) for x in np.unique(ids)]
        self.rows_of = {}
        self.faces_of = {}
        for src in self.names:
            L = np.where(ids == src)[0]
            L = L[np.argsort(fr[L], kind="stable")]
            self.rows_of[src] = L
            self.faces_of[src] = L[self.usable[L]]
        self.with_gallery = [src for src in self.names if len(self.faces_of[src]) >= GALLERY_MIN]
        self.frames_of = {src: set(fr[self.rows_of[src]].tolist()) for src in self.names}
        for x in self.short:
            self.frames_of.setdefault(int(x[1]), set()).add(int(x[0]))

    def box(self, src, f):
        """Box of I at frame f, else its last known box at most 30 frames earlier; (box, present at f)."""
        L = self.rows_of[src]
        k = int(np.searchsorted(self.fr[L], f, side="right")) - 1
        if k < 0:
            return None, False
        d = f - int(self.fr[L[k]])
        if d == 0:
            return self.B[L[k]], True
        return (self.B[L[k]], False) if d <= 30 else (None, False)

    def contact(self, src, K, f1, f2):
        """Frame of strongest overlap between I and K in [f1, f2] (one of the two present); (IoU, frame)."""
        best = 0.0
        bf = None
        for f in range(f1, f2 + 1):
            bi, pi = self.box(src, f)
            bk, pk = self.box(K, f)
            if bi is None or bk is None or not (pi or pk):
                continue
            x = iou(bi, bk)
            if x > best:
                best = x
                bf = f
        return best, bf

    def judge(self, src, K, a, b):
        fr = self.fr
        d = {}
        for name, X in (("I", src), ("K", K)):
            u = self.faces_of[X]
            m = (fr[u] >= a) & (fr[u] <= b)
            d[name] = (u[m], u[~m])
        gI = self.gallery(d["I"][1])
        gK = self.gallery(d["K"][1])
        sides = {}
        for name, target, own in (("I", gK, gI), ("K", gI, gK)):
            inn = d[name][0]
            if len(inn) >= 4 and target is not None and own is not None:
                m = self.mean(inn)
                s1 = float(m @ target)
                e = s1 - float(m @ own)
                sides[name] = dict(
                    n=int(len(inn)),
                    s1=round(s1, 3),
                    margin=round(e, 3),
                    valid=bool(len(inn) >= SIDE_MIN and s1 >= FIRST and e >= MARGIN),
                )
            else:
                sides[name] = dict(n=int(len(inn)), s1=None, margin=None, valid=False)
        val = [c for c in sides.values() if c["valid"]]
        if not val:
            return None
        if any(
            (not c["valid"]) and c["margin"] is not None and c["margin"] < 0 for c in sides.values()
        ):
            return None
        return dict(
            I=int(src),
            K=int(K),
            a=int(a),
            b=int(b),
            sides=sides,
            strength=float(sum(c["n"] * c["margin"] for c in val)),
        )

    def bounds(self, c):
        fr = self.fr
        src, K, a, b = c["I"], c["K"], c["a"], c["b"]
        f = np.concatenate([fr[self.faces_of[src]], fr[self.faces_of[K]]])
        before = f[f < a]
        after = f[f > b]
        if len(before):
            x, t = self.contact(src, K, int(before.max()) + 1, a)
            if t is None:
                return False
            c["t_in"] = int(t)
            c["iou_in"] = round(x, 3)
        else:
            c["t_in"] = -1
            c["iou_in"] = None
        if len(after):
            x, t = self.contact(src, K, b + 1, int(after.min()))
            if t is None:
                return False
            c["t_out"] = int(t)
            c["iou_out"] = round(x, 3)
        else:
            c["t_out"] = 10**9
            c["iou_out"] = None
        return c["t_out"] > c["t_in"]

    def candidates(self):
        fr = self.fr
        out = []
        Ks = self.with_gallery
        for x in range(len(Ks)):
            for y in range(x + 1, len(Ks)):
                src, K = Ks[x], Ks[y]
                if len(self.frames_of[src] & self.frames_of[K]) < COMMON:
                    continue
                L = np.concatenate([self.faces_of[src], self.faces_of[K]])
                side = np.r_[
                    np.zeros(len(self.faces_of[src]), int), np.ones(len(self.faces_of[K]), int)
                ]
                o = np.argsort(fr[L], kind="stable")
                L = L[o]
                side = side[o]
                km = two_means(self.Ec[L])
                if km is None:
                    continue
                p = self.Ec[L] @ (km[0] - km[1])
                s = np.where(side == 0, p, -p)
                seen = set()
                for sign in (1.0, -1.0):
                    tot, i, j = best_run(sign * s)
                    if tot <= 0 or (i == 0 and j == len(L) - 1):
                        continue
                    a, b = int(fr[L[i]]), int(fr[L[j]])
                    if (a, b) in seen:
                        continue
                    seen.add((a, b))
                    c = self.judge(src, K, a, b)
                    if c is not None and self.bounds(c):
                        out.append(c)
        return sorted(out, key=lambda c: -c["strength"])

    def swap(self, c):
        src, K = c["I"], c["K"]
        m = (self.fr >= c["t_in"]) & (self.fr < c["t_out"])
        mi = m & (self.ids == src)
        mk = m & (self.ids == K)
        self.ids[mi] = K
        self.ids[mk] = src
        n = int(mi.sum() + mk.sum())
        if len(self.short):
            f = self.short[:, 0].astype(np.int64)
            i = self.short[:, 1].astype(np.int64)
            m = (f >= c["t_in"]) & (f < c["t_out"])
            m1 = m & (i == src)
            m2 = m & (i == K)
            self.short[m1, 1] = K
            self.short[m2, 1] = src
        return n


def _excluded(c, excluded):
    """`excluded`: (I, K, t_in, t_out) withdrawn by the check; a candidate of the same pair whose interval
    covers more than half of a withdrawn interval is not applied."""
    for i, k, a, b in excluded:
        if {i, k} == {c["I"], c["K"]}:
            lo = max(a, c["t_in"])
            hi = min(b, c["t_out"])
            inter = max(0, hi - lo)
            shorter = max(1, min(b - a, c["t_out"] - c["t_in"]))
            if inter > 0.5 * shorter:
                return True
    return False


def exchange(
    rows: Rows,
    boxes,
    ids0,
    short0,
    table_a: FaceTable,
    table_b: FaceTable,
    contact_limit=1.01,
    excluded=(),
):
    """Exchanges validated by the two models. Returns the identities, the short-track rows and the log."""
    ids0 = np.asarray(ids0).astype(np.int64).ravel()
    A = Reader(rows, boxes, ids0, short0, table_a, contact_limit)
    B = Reader(rows, boxes, ids0, short0, table_b, contact_limit)
    log = []
    if A.readable and B.readable:
        for turn in range(PASSES):
            A.index()
            B.index()
            cands = []
            for X, Y, name in ((A, B, "model A"), (B, A, "model B")):
                for c in X.candidates():
                    c2 = Y.judge(c["I"], c["K"], c["a"], c["b"])
                    if c2 is None:
                        continue
                    c["found_by"] = name
                    c["other_model"] = c2["sides"]
                    cands.append(c)
            cands.sort(key=lambda c: -c["strength"])
            taken = set()
            done = False
            for c in cands:
                if c["I"] in taken or c["K"] in taken:
                    continue
                if _excluded(c, excluded):
                    continue
                c["turn"] = turn + 1
                c["rows"] = A.swap(c)
                B.swap(c)
                taken |= {c["I"], c["K"]}
                log.append(c)
                done = True
            if not done:
                break
    return A.ids.copy(), A.short.copy(), log
