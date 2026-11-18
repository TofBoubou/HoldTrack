"""The face and a reader: a row takes the identity that its face designates when a pass of the tracker agrees.

For every row, the faces of its identity within a second and a half form a window; the window is compared with
the gallery of every identity and with its own gallery, recomputed without the rows close to the window. The
row is designated to the best other identity K when the lead is large and a forward pass holds K, or when the
lead is positive and both a contrary pass (backward or mirrored) and a forward pass hold K. The two thresholds
are read on the video, on the designated rows of its safe stretches. Nothing here reads an annotation.
"""

from __future__ import annotations

import collections

import numpy as np

from ..identity import stretches as safe_stretches
from ..identity.geometry import xyxy
from .cuts import contact, unit
from .pieces import READABLE_NORM, FaceTable

EYES = 6.0  # pixels between the eyes
GALLERY_CONTACT = 0.3
WINDOW_CONTACT = 0.6
GALLERY_MIN = 20
WINDOW_MIN = 4
HALF_WINDOW = 30  # frames
APART = 60  # frames around the window left out of its own gallery
PASS_MIN = 20


def load(table: FaceTable, rows: int):
    """Unit embeddings centred on the readable faces of the video, per row; readable rows."""
    X = unit(table.embedding.astype(np.float32))
    q = (table.norm >= READABLE_NORM) & (table.eyes >= EYES)
    if q.sum() >= GALLERY_MIN:
        X = unit(X - X[q].mean(0))
    E = np.zeros((rows, X.shape[1]), np.float32)
    good = np.zeros(rows, bool)
    E[table.index] = X
    good[table.index] = q
    return E, good


def galleries(E, good, ids, frame, c):
    G = {}
    for name in np.unique(ids):
        ix = np.where((ids == name) & good & (c < GALLERY_CONTACT))[0]
        if len(ix) >= GALLERY_MIN:
            G[int(name)] = (E[ix].sum(0), frame[ix], E[ix], ix)
    return G


def votes(E, good, ids, frame, c, G):
    """For every row: the best other identity, its lead over the row's own identity, the resemblance to its
    own identity. An identity without a gallery is compared with the second best gallery."""
    n = len(ids)
    K = np.full(n, -1, np.int64)
    lead = np.full(n, np.nan, np.float32)
    own = np.full(n, np.nan, np.float32)
    names = sorted(G)
    if len(names) < 2:
        return K, lead, own
    M = unit(np.stack([G[name][0] for name in names]))
    rank = {name: k for k, name in enumerate(names)}
    for name in np.unique(ids):
        al = np.where(ids == name)[0]
        al = al[np.argsort(frame[al])]
        ix = al[good[al] & (c[al] < WINDOW_CONTACT)]
        if len(ix) < WINDOW_MIN:
            continue
        fr = frame[ix]
        fa = frame[al]
        cs = np.concatenate([np.zeros((1, E.shape[1]), np.float32), np.cumsum(E[ix], 0)])
        a = np.searchsorted(fr, fa - HALF_WINDOW, "left")
        b = np.searchsorted(fr, fa + HALF_WINDOW, "right")
        ok = (b - a) >= WINDOW_MIN
        T = unit(cs[b] - cs[a])
        S = T @ M.T
        if name in G:
            gs, gf, ge = G[name][:3]
            o = np.argsort(gf)
            gf = gf[o]
            gc = np.concatenate([np.zeros((1, E.shape[1]), np.float32), np.cumsum(ge[o], 0)])
            lo = np.searchsorted(gf, fa - APART, "left")
            hi = np.searchsorted(gf, fa + APART, "right")
            rest = gs[None, :] - (gc[hi] - gc[lo])
            nb = len(gf) - (hi - lo)
            s_own = np.where(nb >= GALLERY_MIN // 2, (T * unit(rest)).sum(1), np.nan)
            S[:, rank[name]] = -9
        else:
            s_own = np.sort(S, 1)[:, -2]
        k = S.argmax(1)
        best = S[np.arange(len(al)), k]
        m = ok
        K[al[m]] = np.array(names)[k[m]]
        own[al[m]] = s_own[m]
        lead[al[m]] = (best - s_own)[m]
    return K, lead, own


def majority_name(x):
    y = x[x >= 0]
    if 2 * len(y) < len(x) or not len(y):
        return None
    k, m = collections.Counter(y.tolist()).most_common(1)[0]
    return k if 2 * m >= len(x) else None


def face_and_reader(video, ids, table: FaceTable):
    """New identities of the rows of `video` (a `holdtrack.identity.video.Video`) under the current names `ids`."""
    p = video.rows(ids)
    frame = p[:, 0].astype(int)
    ids = p[:, 1].astype(np.int64)
    box = xyxy(p)
    n = len(p)
    new = ids.copy()
    E, good = load(table, n)
    if good.sum() < GALLERY_MIN:
        return new
    c = contact(frame, box)
    G = galleries(E, good, ids, frame, c)
    K, lead, _ = votes(E, good, ids, frame, c, G)
    none = np.full(n, -1, np.int64)
    forward = video.forward
    readers = [
        video.backward if video.backward is not None else none,
        video.mirrored if video.mirrored is not None else none,
        forward[1],
        forward[2],
        forward[0],
    ]
    # a reader holds the vote of the face if it gives the rows of the window the name it gives the gallery of K
    gallery_name = [{k: majority_name(x[G[k][3]]) for k in G} for x in readers]
    rows_of = {int(name): np.where(ids == name)[0] for name in np.unique(ids)}
    rows_of = {name: z[np.argsort(frame[z])] for name, z in rows_of.items()}
    voted = (K >= 0) & ~np.isnan(lead)
    holds = np.zeros((5, n), bool)
    seg, _ = safe_stretches.stretches(video, p, "full")
    ref = lead[voted & (seg >= 0)]
    q95, q99 = (
        (np.quantile(ref, 0.95), np.quantile(ref, 0.99)) if len(ref) >= 200 else (np.inf, np.inf)
    )
    s1 = max(0.5, q99)
    s2 = max(0.0, q95)
    low = min(s1, s2)
    for i in np.where(voted & (lead >= low))[0]:
        z = rows_of[int(ids[i])]
        fz = frame[z]
        w = z[
            np.searchsorted(fz, frame[i] - HALF_WINDOW, "left") : np.searchsorted(
                fz, frame[i] + HALF_WINDOW, "right"
            )
        ]
        for r, x in enumerate(readers):
            g = gallery_name[r].get(int(K[i]))
            holds[r, i] = g is not None and majority_name(x[w]) == g
    forward_holds = holds[2:].any(0)
    contrary = holds[:2].any(0) & forward_holds
    da = voted & (lead >= s1) & forward_holds
    db = voted & (lead >= s2) & contrary
    target = np.where(da | db, K, -1)
    at = collections.defaultdict(dict)
    for i in range(n):
        at[int(frame[i])][int(ids[i])] = i
    done = set()
    order = sorted(np.where(target >= 0)[0], key=lambda i: -lead[i])
    for _ in range(3):  # a name freed by a renaming can be taken at the next pass
        before = len(done)
        for i in order:
            if i in done:
                continue
            f = int(frame[i])
            k = int(target[i])
            j = at[f].get(k)
            if j is None:
                if at[f].get(int(ids[i])) == i:
                    del at[f][int(ids[i])]
                new[i] = k
                at[f][k] = i
                done.add(i)
            elif j not in done and target[j] == ids[i]:
                new[i] = k
                new[j] = ids[i]
                at[f][k] = i
                at[f][int(ids[i])] = j
                done |= {i, j}
        if len(done) == before:
            break
    key = frame.astype(np.int64) * 10**9 + new
    assert len(np.unique(key)) == len(key) and (new > 0).all(), video.name
    return new


def keep_clear_of_short(frame, reference, new, short):
    """A row renamed K takes back its reference name if a short-track row already carries K in its frame."""
    r = np.asarray(reference).astype(np.int64).ravel()
    n = np.asarray(new).astype(np.int64).copy()
    taken = (
        set(zip(short[:, 0].astype(int).tolist(), short[:, 1].astype(int).tolist()))
        if len(short)
        else set()
    )
    changed = np.where(r != n)[0]
    back = [i for i in changed if (int(frame[i]), int(n[i])) in taken]
    n[back] = r[back]
    return n


def supported_by_a_pass(video, ids):
    """Second proof of a piece candidate: one of the five passes (backward, mirrored, exclusive, guarded,
    native) gives the same majority name, over at least half of the rows, to the rows of the piece and to the
    rows of its target outside the interval. Returns a function of the candidate."""
    p = video.rows(ids)
    frame = p[:, 0].astype(int)
    ids = p[:, 1].astype(np.int64)
    none = np.full(len(p), -1)
    forward = video.forward
    readers = [
        video.backward if video.backward is not None else none,
        video.mirrored if video.mirrored is not None else none,
        forward[1],
        forward[2],
        forward[0],
    ]

    def share(x):
        y = x[x >= 0]
        if not len(y) or not len(x):
            return None, 0.0
        k, n = collections.Counter(y.tolist()).most_common(1)[0]
        return k, n / len(x)

    def supported(c):
        src, K, a, b = int(c["I"]), int(c["K"]), int(c["a"]), int(c["b"])
        P = np.where((ids == src) & (frame >= a) & (frame <= b))[0]
        outside = np.where((ids == K) & ((frame < a) | (frame > b)))[0]
        for x in readers:
            (ni, pi), (nh, ph) = share(x[P]), share(x[outside])
            if ni is not None and ni == nh and pi >= 0.5 and ph >= 0.5:
                return True
        return False

    return supported
