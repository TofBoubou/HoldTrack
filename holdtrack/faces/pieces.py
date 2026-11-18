"""Whole identities, pieces and merges read on the faces.

A video is given as arrays over the rows of its tracks: frame, track of origin, contact, and one or several
face tables (a row index, an embedding and a norm per face). Nothing here reads an annotation.

Gallery of a set of rows: sum of the centred embeddings weighted by their norm, over the readable faces away
from contact, trimmed three times to the 60% closest to the sum. A *whole* candidate moves an identity I onto
an identity K whose gallery it resembles; a *piece* candidate does the same for a stretch of I delimited by a
gap or a change of track. `apply` and `apply_rejudged` pose the candidates; `merge` joins identities that are
never present together and are each other's best resemblance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

READABLE_NORM = 14.0  # a face is readable from this norm of its raw embedding
GALLERY_MIN = 12  # faces of a gallery
ADVANCE = 0.12  # lead of the target over every other identity
FIRST = 0.4  # resemblance of the candidate to its target
MARGIN = 0.4  # lead of the candidate over what it leaves
PIECE_MIN = 30  # rows of a piece
GAP = 10  # frames of absence that delimit a piece
NEW_NAME = 7_300_000  # names given to short-track rows that would collide


@dataclass
class FaceTable:
    """Faces of one video: `index` is the row of each face, `embedding` its unit vector, `norm` the norm of
    the raw embedding. `disputed` marks the rows whose face is shared with a neighbour (optional)."""

    index: np.ndarray
    embedding: np.ndarray
    norm: np.ndarray
    disputed: np.ndarray | None = None
    eyes: np.ndarray | None = None  # distance between the eyes, in pixels

    @classmethod
    def load(cls, path: str, disputed: str | None = None) -> FaceTable:
        t = np.load(path)
        d = np.load(disputed)["dispute"].astype(bool) if disputed else None
        if "rows" in t.files:
            return cls(
                t["rows"],
                t["embedding"],
                t["norm_single"],
                d,
                t["eye_distance"] if "eye_distance" in t.files else None,
            )
        return cls(
            t["idx"], t["emb"], t["norme0"], d, t["yeux"] if "yeux" in t.files else None
        )  # tables of the replays

    def dense(self, rows: int):
        E = np.zeros((rows, 512), np.float32)
        q = np.zeros(rows, np.float32)
        E[self.index] = self.embedding.astype(np.float32)
        q[self.index] = self.norm
        return E, q


@dataclass
class Rows:
    """Rows of the tracks of one video, in the order of its tables."""

    frame: np.ndarray
    track: np.ndarray
    contact: np.ndarray


@dataclass
class Settings:
    """Tables and thresholds of one pass over the candidates.

    `main` and `second` are the two face models; `main_pieces` and `second_pieces` are the tables on which
    the pieces are read (the tables without shared faces), with their own contact limits."""

    main: FaceTable
    second: FaceTable | None = None
    main_pieces: FaceTable | None = None
    second_pieces: FaceTable | None = None
    merge_threshold: float = 0.7
    second_merge_threshold: float = 0.7
    fallback: bool = False  # a piece partly renamed by a whole exchange leaves the label of the rest of its identity
    no_piece_from_main_alone: bool = False
    no_whole: bool = False
    no_merge: bool = False
    rejudge: bool = False  # candidates recomputed after each operation
    piece_gallery_contact: float = 0.3
    piece_segment_contact: float = 0.6
    without_disputed: bool = False
    first: float = FIRST
    margin: float = MARGIN
    excluded: set = field(
        default_factory=set
    )  # (I, K, first frame, last frame) withdrawn by the check


def mode(x):
    u, c = np.unique(x, return_counts=True)
    return int(u[c.argmax()])


def galleries(rows: Rows, ids, table: FaceTable):
    """Trimmed gallery of every identity with 20 readable faces away from contact (embeddings not centred)."""
    fr = rows.frame.astype(int)
    ct = rows.contact
    E, q = table.dense(len(fr))
    lis = q >= READABLE_NORM
    W = E * q[:, None]
    gal = {}
    for src in np.unique(ids):
        L = np.where((ids == src) & lis & (ct < 0.3))[0]
        if len(L) < 20:
            continue
        keep = L
        for _ in range(3):
            g = W[keep].sum(0)
            g /= np.linalg.norm(g)
            s_ = E[L] @ g
            keep = L[s_ >= np.quantile(s_, 0.4)]
        g = W[keep].sum(0)
        gal[int(src)] = g / np.linalg.norm(g)
    return fr, gal


def merge(rows: Rows, ids, short, table: FaceTable, tau: float):
    """Identities never present together (short-track rows count in the presence) whose galleries resemble
    each other (tau) and are each other's best resemblance. Returns {old name: new name} and the merges."""
    fr, gal = galleries(rows, ids, table)
    Ks = sorted(gal)
    corr = {}
    if len(Ks) < 2:
        return corr, []
    pres = {
        K: set(fr[ids == K].tolist()) | set(int(x[0]) for x in short if int(x[1]) == K) for K in Ks
    }
    G = np.stack([gal[K] for K in Ks])
    S_ = G @ G.T
    np.fill_diagonal(S_, -9)
    best = S_.argmax(1)
    pairs = sorted(
        [
            (float(S_[i, j]), Ks[i], Ks[j])
            for i in range(len(Ks))
            for j in range(i + 1, len(Ks))
            if best[i] == j and best[j] == i and S_[i, j] >= tau and not (pres[Ks[i]] & pres[Ks[j]])
        ],
        reverse=True,
    )
    group = {K: [K] for K in Ks}
    head = {K: K for K in Ks}
    done = []
    for s_, src, K in pairs:
        a_, b_ = head[src], head[K]
        if a_ == b_:
            continue
        if set().union(*[pres[x] for x in group[a_]]) & set().union(*[pres[x] for x in group[b_]]):
            continue
        group[a_] += group[b_]
        for x in group[b_]:
            head[x] = a_
        del group[b_]
        done.append((round(s_, 3), src, K))
    for members in group.values():
        if len(members) > 1:
            lead = max(members, key=lambda x: len(pres[x]))
            for x in members:
                if x != lead:
                    corr[x] = lead
    return corr, done


def rename(ids, corr):
    return np.array([corr.get(int(i), int(i)) for i in ids], np.int64) if corr else ids


def candidates_of_table(
    rows: Rows,
    ids,
    table: FaceTable,
    gallery_contact=0.3,
    segment_contact=0.6,
    without_disputed=False,
    first=FIRST,
    margin=MARGIN,
):
    """Whole and piece candidates read on one table (embeddings centred on the readable faces of the video)."""
    fr = rows.frame.astype(int)
    ct = rows.contact
    pid = rows.track.astype(np.int64)
    N0 = len(fr)
    E, q = table.dense(N0)
    lis = q >= READABLE_NORM
    out = []
    if without_disputed and table.disputed is not None:
        lis &= ~table.disputed
    if lis.sum() < 50:
        return fr, out
    Ec = E - E[lis].mean(0)
    Ec /= np.maximum(np.linalg.norm(Ec, axis=1, keepdims=True), 1e-9)
    W = Ec * q[:, None]

    def gal(L):
        L = L[lis[L] & (ct[L] < gallery_contact)]
        if len(L) < GALLERY_MIN:
            return None
        keep = L
        for _ in range(3):
            g = W[keep].sum(0)
            g /= np.linalg.norm(g)
            s_ = Ec[L] @ g
            keep = L[s_ >= np.quantile(s_, 0.4)]
        g = W[keep].sum(0)
        return g / np.linalg.norm(g)

    def seg(L):
        L = L[lis[L] & (ct[L] < segment_contact)]
        if len(L) < 4:
            return None
        g = W[L].sum(0)
        return g / np.linalg.norm(g)

    U = [int(x) for x in np.unique(ids)]
    lin = {src: np.where(ids == src)[0] for src in U}
    G = {src: gal(lin[src]) for src in U}
    Ks = [src for src in U if G[src] is not None]
    C = []
    for src in U:
        L = lin[src]
        L = L[np.argsort(fr[L], kind="stable")]
        if G[src] is not None:
            C.append((src, L, "whole", G[src], None))
        if len(L) < 2:
            continue
        cut = np.where((np.diff(fr[L]) > GAP) | (np.diff(pid[L]) != 0))[0]
        st = np.r_[0, cut + 1]
        en = np.r_[cut + 1, len(L)]
        if len(st) < 2:
            continue
        for s, e in zip(st, en):
            P = L[s:e]
            if len(P) < PIECE_MIN:
                continue
            gP = gal(P)
            gR = gal(np.setdiff1d(L, P))
            if gP is not None and gR is not None:
                C.append((src, P, "piece", gP, gR))
    for src, Lc, kind, Gc, gR in C:
        a_ = int(fr[Lc].min())
        b_ = int(fr[Lc].max())
        fc = set(fr[Lc].tolist())
        outside = {}
        if kind == "piece":
            for X in U:
                if X == src:
                    continue
                g = gal(lin[X][(fr[lin[X]] < a_) | (fr[lin[X]] > b_)])
                if g is not None:
                    outside[X] = float(Gc @ g)
        for K in U:
            if K == src:
                continue
            Lk = lin[K]
            inside = Lk[(fr[Lk] >= a_) & (fr[Lk] <= b_)]
            common = len(fc & set(fr[inside].tolist()))
            if (kind == "whole" and common < 5) or (kind == "piece" and 1 <= common < 5):
                continue
            gK = gal(Lk[(fr[Lk] < a_) | (fr[Lk] > b_)])
            if gK is None:
                continue
            s1 = float(Gc @ gK)
            s2 = None
            if s1 < first:
                continue
            if common >= 5:
                sk = seg(inside)
                if sk is None or s1 - float(sk @ gK) < margin:
                    continue
                s2 = float(sk @ gK)
            if kind == "piece" and s1 - float(Gc @ gR) < margin:
                continue
            others = (
                [x for X, x in outside.items() if X != K]
                if kind == "piece"
                else [float(Gc @ G[X]) for X in Ks if X not in (src, K)]
            )
            out.append(
                dict(
                    I=src,
                    K=int(K),
                    a=a_,
                    b=b_,
                    s1=s1,
                    s2=s2,
                    own=float(Gc @ gR) if gR is not None else None,
                    advance=s1 - (max(others) if others else -1.0),
                    nI=int(len(Lc)),
                    nKin=int(len(inside)) if common >= 5 else 0,
                    kind=kind,
                    op="exchange" if common >= 5 else "move",
                    rows=Lc,
                    inside=inside if common >= 5 else np.zeros(0, int),
                )
            )
    return fr, out


def candidates(rows: Rows, ids, s: Settings):
    """Candidates of the main table; with a second table, their union (the stronger resemblance is kept)."""

    def of(table, pieces):
        fr, c = candidates_of_table(rows, ids, table, first=s.first, margin=s.margin)
        if pieces is not None:
            _, cp = candidates_of_table(
                rows,
                ids,
                pieces,
                s.piece_gallery_contact,
                s.piece_segment_contact,
                s.without_disputed,
                first=s.first,
                margin=s.margin,
            )
            c = [x for x in c if x["kind"] == "whole"] + [x for x in cp if x["kind"] == "piece"]
        return fr, c

    fr, c1 = of(s.main, s.main_pieces)
    if s.second is None:
        return fr, c1
    _, c2 = of(s.second, s.second_pieces)
    key = lambda c: (c["I"], c["K"], c["a"], c["b"], c["kind"])  # noqa: E731
    D1 = {key(c): dict(c, eye="1") for c in c1}
    for c in c2:
        k = key(c)
        if k not in D1:
            D1[k] = dict(c, eye="2")
        else:
            D1[k] = dict(
                c if c["s1"] > D1[k]["s1"] else D1[k],
                eye="both",
                advance=max(c["advance"], D1[k]["advance"]),
            )
    if s.no_piece_from_main_alone:
        return fr, [c for c in D1.values() if not (c["kind"] == "piece" and c["eye"] == "1")]
    return fr, list(D1.values())


def _ordered(cands, s: Settings, with_pieces):
    cands = [
        c
        for c in cands
        if c["advance"] >= ADVANCE
        and (with_pieces or c["kind"] == "whole")
        and not (s.no_whole and c["kind"] == "whole")
    ]
    return sorted([c for c in cands if c["kind"] == "whole"], key=lambda c: -c["s1"]) + sorted(
        [c for c in cands if c["kind"] == "piece"], key=lambda c: -c["s1"]
    )


def _kept(c, kept):
    """Second proof of a piece: `kept` is None (no sorting), "both" (both models propose it), a function of
    the candidate, or a dictionary (I, K, first frame, last frame) -> kept."""
    if c["kind"] != "piece" or kept is None:
        return True
    if kept == "both":
        return c.get("eye") == "both"
    if callable(kept):
        return bool(kept(c))
    return kept.get((c["I"], c["K"], c["a"], c["b"]), False)


def apply(rows: Rows, ids0, s: Settings, kept, with_pieces=True):
    """Poses the candidates computed once: whole identities first, then pieces, by decreasing resemblance."""
    fr, cands = candidates(rows, ids0, s)
    order = _ordered(cands, s, with_pieces)
    new = ids0.copy()
    fixed = np.zeros(len(ids0), bool)
    spans = []
    log = []
    lin0 = {int(src): np.where(ids0 == src)[0] for src in np.unique(ids0)}
    for c in order:
        P = c["rows"]
        Q = c["inside"]
        src = c["I"]
        K = c["K"]
        c2 = {k: x for k, x in c.items() if k not in ("rows", "inside")}
        note = lambda e, n=0: log.append(c2 | dict(state=e, changed=n, _P=P))  # noqa: E731
        if (int(src), int(K), int(c["a"]), int(c["b"])) in s.excluded:
            note("withdrawn by the check")
            continue
        if not _kept(c, kept):
            note("set aside by the sorting")
            continue
        if fixed[P].any() or (len(Q) and fixed[Q].any()):
            note("rows already identified")
            continue
        Lk = lin0[K]
        ko = Lk[(fr[Lk] < c["a"]) | (fr[Lk] > c["b"])]
        cur = mode(new[P])
        tgt = mode(new[ko])
        if tgt == cur:
            note("already under the right label")
            continue
        Qc = Q[new[Q] == tgt] if len(Q) else Q
        fp = set(fr[P].tolist())
        if fp & set(fr[np.setdiff1d(np.where(new == tgt)[0], Qc)].tolist()):
            note("duplicate")
            continue
        if len(Qc) and set(fr[Qc].tolist()) & set(
            fr[np.setdiff1d(np.where(new == cur)[0], P)].tolist()
        ):
            rest = np.setdiff1d(lin0[src], P)
            rest = rest[(fr[rest] < c["a"]) | (fr[rest] > c["b"])]
            cur2 = mode(new[rest]) if (s.fallback and c["kind"] == "piece" and len(rest)) else None
            if (
                cur2 is None
                or cur2 in (cur, tgt)
                or set(fr[Qc].tolist())
                & set(fr[np.setdiff1d(np.where(new == cur2)[0], P)].tolist())
            ):
                note("duplicate")
                continue
            cur = cur2
            c2["fallback"] = True
        new[P] = tgt
        fixed[P] = True
        if len(Qc):
            new[Qc] = cur
        spans.append((c["a"], c["b"], cur, tgt, len(Qc) > 0))
        note("applied", int(len(P) + len(Qc)))
    return new, spans, log


def apply_rejudged(rows: Rows, ids0, s: Settings, kept, with_pieces=True):
    """Poses the best candidate, recomputes the candidates on the current identities, and starts again. The
    rows of a posed candidate no longer move."""
    new = ids0.copy()
    fixed = np.zeros(len(ids0), bool)
    spans = []
    log = []
    seen = set()
    for turn in range(60):
        fr, cands = candidates(rows, new, s)
        order = _ordered(cands, s, with_pieces)
        done = False
        for c in order:
            P = c["rows"]
            Q = c["inside"]
            src = c["I"]
            K = c["K"]
            key = (int(src), int(K), int(c["a"]), int(c["b"]), c["kind"])
            c2 = {k: x for k, x in c.items() if k not in ("rows", "inside")}
            note = lambda e, n=0: log.append(c2 | dict(state=e, changed=n, _P=P, turn=turn + 1))  # noqa: E731
            if key in seen:
                continue
            if key[:4] in s.excluded:
                note("withdrawn by the check")
                seen.add(key)
                continue
            if not _kept(c, kept):
                note("set aside by the sorting")
                seen.add(key)
                continue
            if fixed[P].any() or (len(Q) and fixed[Q].any()):
                note("rows already identified")
                seen.add(key)
                continue
            if set(fr[P].tolist()) & set(fr[np.setdiff1d(np.where(new == K)[0], Q)].tolist()):
                note("duplicate")
                seen.add(key)
                continue
            if len(Q) and set(fr[Q].tolist()) & set(
                fr[np.setdiff1d(np.where(new == src)[0], P)].tolist()
            ):
                note("duplicate")
                seen.add(key)
                continue
            new[P] = K
            fixed[P] = True
            if len(Q):
                new[Q] = src
            spans.append((c["a"], c["b"], src, K, len(Q) > 0))
            note("applied", int(len(P) + len(Q)))
            done = True
            break
        if not done:
            break
    return new, spans, log


def rename_short(short, spans):
    """Short-track rows follow the operations: inside a span, the left label takes the target label, and the
    target label takes the left one when the operation is an exchange."""
    short = short.copy()
    if len(short) == 0:
        return short
    f = short[:, 0].astype(np.int64)
    for a_, b_, cur, tgt, exchange in spans:
        i = short[:, 1].astype(np.int64)
        m1 = (f >= a_) & (f <= b_) & (i == cur)
        m2 = (f >= a_) & (f <= b_) & (i == tgt)
        short[m1, 1] = tgt
        if exchange:
            short[m2, 1] = cur
    return short


def separate_short(rows: Rows, new, short):
    """A short-track row that lands on the (frame, name) of a row of the tracks, or of another short-track row,
    takes a new name. Returns the rows and the number of new names."""
    if len(short) == 0:
        return short, 0
    fr = rows.frame.astype(np.int64)
    taken = set(zip(fr.tolist(), np.asarray(new).astype(np.int64).tolist()))
    short = short.copy()
    n = 0
    name = max(NEW_NAME, int(np.asarray(new).max()), int(short[:, 1].max()))
    for k in range(len(short)):
        c = (int(short[k, 0]), int(short[k, 1]))
        if c in taken:
            name += 1
            short[k, 1] = name
            n += 1
            c = (c[0], name)
        taken.add(c)
    return short, n


def run(rows: Rows, ids0, short0, s: Settings, kept, with_pieces=True):
    """Whole identities, pieces, then merges. Returns the identities, the short-track rows, the log, the table
    of merged names and the merges."""
    new, spans, log = (apply_rejudged if s.rejudge else apply)(rows, ids0, s, kept, with_pieces)
    short = rename_short(short0, spans)
    if s.no_merge:
        short, n = separate_short(rows, new, short)
        if n:
            log.append(
                dict(
                    kind="separation",
                    op="short-track rows renamed to avoid a duplicate",
                    state="separation",
                    changed=n,
                    I=-1,
                    K=-1,
                    a=-1,
                    b=-1,
                    s1=0.0,
                    advance=0.0,
                    nI=n,
                    nKin=0,
                    _P=np.zeros(0, int),
                )
            )
        return new, short, log, {}, []
    c1, f1 = merge(rows, new, short, s.main, s.merge_threshold)
    new2 = rename(new, c1)
    short2 = short.copy()
    if len(short2):
        short2[:, 1] = rename(short2[:, 1].astype(np.int64), c1)
    if s.second is not None:
        c2, f2 = merge(rows, new2, short2, s.second, s.second_merge_threshold)
        if c2:
            new2 = rename(new2, c2)
            if len(short2):
                short2[:, 1] = rename(short2[:, 1].astype(np.int64), c2)
            c1 = {k: c2.get(x, x) for k, x in c1.items()} | {
                k: x for k, x in c2.items() if k not in c1
            }
            f1 = list(f1) + [tuple(x) + ("second model",) for x in f2]
    return new2, short2, log, c1, f1
