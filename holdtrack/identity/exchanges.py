"""Steps 3 to 5. Exchanges of two identities at a contact, and merges that two appearance models agree on.

Two identities P and Q that touch may have exchanged their names during the contact. Two probes read it: each is a
logistic regression "P or Q" fitted on the crops before the contact, on one appearance model, and read on the
crops after it. An exchange is applied when both probes score it and a proposer shows it: the second pass of links
(step 3), then the backward pass (step 4).
"""

import collections

import numpy as np

from .geometry import frames as n_frames
from .geometry import xyxy

WINDOW_S = 15.0
MIN_CROPS = 15
LONG_S = 10.0
SHARED_S = 5.0
EPISODE_GAP_S = 0.5
CONTACT_IOU = 0.2
CLEAR_IOU = 0.1
THETA_SECOND = 0.6
THETA_BACKWARD = 0.5


def _iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = iw * ih
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def _probe(E, before_p, before_q, after_p, after_q):
    from sklearn.linear_model import LogisticRegression

    X = np.concatenate([E[before_p], E[before_q]])
    y = np.concatenate([np.zeros(len(before_p)), np.ones(len(before_q))])
    mu = X.mean(0)
    sd = X.std(0) + 1e-6
    m = LogisticRegression(C=1.0, max_iter=500).fit((X - mu) / sd, y)
    return 0.5 * (
        float(m.predict((E[after_p] - mu) / sd).mean())
        + float(1 - m.predict((E[after_q] - mu) / sd).mean())
    )


def probes_input(video, crops_meta, embedding, features):
    """Frames, boxes, row of each crop, and the two appearance models (the second reduced to 64 components)."""
    rows = video.rows()
    T = features.astype(np.float32)
    T = T - T.mean(0)
    _, _, vt = np.linalg.svd(T, full_matrices=False)
    return (
        rows[:, 0].astype(int),
        xyxy(rows),
        crops_meta[:, 9].astype(int),
        embedding.astype(np.float32),
        T @ vt[:64].T,
    )


def contact_events(ids, frame, box, crop_row, E1, E2, fps=20.0):
    """Contacts between two long identities, with the scores of the two probes for "they exchanged"."""
    window = n_frames(WINDOW_S, fps)
    long_ = n_frames(LONG_S, fps)
    shared = n_frames(SHARED_S, fps)
    gap = n_frames(EPISODE_GAP_S, fps)
    cf = frame[crop_row]
    cid = ids[crop_row]
    size = collections.Counter(ids.tolist())
    big = [lab for lab, n in size.items() if n >= long_]
    at = {
        lab: {int(f): i for f, i in zip(frame[ids == lab], np.where(ids == lab)[0])} for lab in big
    }
    out = []
    for a in range(len(big)):
        for b in range(a + 1, len(big)):
            P, Q = big[a], big[b]
            common = sorted(set(at[P]) & set(at[Q]))
            if len(common) < shared:
                continue
            io = {f: _iou(box[at[P][f]], box[at[Q][f]]) for f in common}
            touch = [f for f in common if io[f] > CONTACT_IOU]
            if not touch:
                continue
            episodes = []
            d = touch[0]
            prev = touch[0]
            for f in touch[1:]:
                if f - prev > gap:
                    episodes.append((d, prev))
                    d = f
                prev = f
            episodes.append((d, prev))
            clear = np.array([io.get(int(f), 0.0) <= CLEAR_IOU for f in cf])
            for t1, t2 in episodes:

                def lot(name, lo, hi, clear=clear):
                    return np.where((cid == name) & (cf >= lo) & (cf <= hi) & clear)[0]

                bp, bq, ap, aq = (
                    lot(P, t1 - window, t1 - 1),
                    lot(Q, t1 - window, t1 - 1),
                    lot(P, t2 + 1, t2 + window),
                    lot(Q, t2 + 1, t2 + window),
                )
                if min(len(bp), len(bq), len(ap), len(aq)) < MIN_CROPS:
                    continue
                out.append(
                    dict(
                        P=int(P),
                        Q=int(Q),
                        t1=int(t1),
                        t2=int(t2),
                        s1=_probe(E1, bp, bq, ap, aq),
                        s2=_probe(E2, bp, bq, ap, aq),
                        sets=(bp, bq, ap, aq),
                    )
                )
    return out


def _record(e, c, a):
    return dict(
        P=e["P"],
        Q=e["Q"],
        frames=[e["t1"], e["t2"]],
        probe_video=round(e["s1"], 2),
        probe_pretrained=round(e["s2"], 2),
        rows_changed=int((c != a).sum()),
    )


def confirmed_by_second_pass(a, b, frame, box, crop_row, E1, E2, theta=THETA_SECOND, fps=20.0):
    """Step 3. a: identities after the first pass of links; b: proposal of the second pass. The rows of P and Q take
    the split that b gives them when both probes score the exchange at least theta."""
    c = a.copy()
    done = []
    taken = set()

    def major(ix):
        return collections.Counter(b[crop_row[ix]].tolist()).most_common(1)[0][0]

    for e in sorted(
        contact_events(a, frame, box, crop_row, E1, E2, fps), key=lambda e: -min(e["s1"], e["s2"])
    ):
        if min(e["s1"], e["s2"]) < theta or e["P"] in taken or e["Q"] in taken:
            continue
        p_before, q_before, p_after, q_after = [major(x) for x in e["sets"]]
        if p_before == q_before or p_after != q_before or q_after != p_before:
            continue
        m = (a == e["P"]) | (a == e["Q"])
        n = c.copy()
        n[m & (b == p_before)] = e["P"]
        n[m & (b == q_before)] = e["Q"]
        key = frame.astype(np.int64) * 10**9 + n
        if len(np.unique(key)) != len(key):
            continue
        c = n
        taken |= {e["P"], e["Q"]}
        done.append(_record(e, c, a))
    return c, done


def proposed_by_backward_pass(
    a, backward, frame, box, crop_row, E1, E2, theta=THETA_BACKWARD, fps=20.0
):
    """Step 4. The backward pass says, frame by frame, whether P and Q are as before the contact or exchanged; the
    frames where it says nothing follow the nearest frame where it speaks."""
    c = a.copy()
    done = []
    taken = set()

    def major(ix):
        q = backward[crop_row[ix]]
        q = q[q >= 0]
        return collections.Counter(q.tolist()).most_common(1)[0][0] if len(q) else None

    for e in sorted(
        contact_events(a, frame, box, crop_row, E1, E2, fps), key=lambda e: -min(e["s1"], e["s2"])
    ):
        if min(e["s1"], e["s2"]) < theta or e["P"] in taken or e["Q"] in taken:
            continue
        p_before, q_before, p_after, q_after = [major(x) for x in e["sets"]]
        if (
            None in (p_before, q_before, p_after, q_after)
            or p_before == q_before
            or p_after != q_before
            or q_after != p_before
        ):
            continue
        P, Q = e["P"], e["Q"]
        ip = np.where(a == P)[0]
        iq = np.where(a == Q)[0]
        side = {}
        for i in ip:
            if backward[i] == p_before:
                side[int(frame[i])] = 0
            elif backward[i] == q_before:
                side[int(frame[i])] = 1
        for i in iq:
            if backward[i] == q_before:
                side.setdefault(int(frame[i]), 0)
            elif backward[i] == p_before:
                side.setdefault(int(frame[i]), 1)
        if not side:
            continue
        fk = np.array(sorted(side))
        ok = np.array([side[f] for f in fk])
        n = c.copy()
        for idx, other in ((ip, Q), (iq, P)):
            f = frame[idx]
            j = np.clip(np.searchsorted(fk, f), 0, len(fk) - 1)
            j0 = np.clip(j - 1, 0, len(fk) - 1)
            near = np.where(np.abs(fk[j] - f) <= np.abs(fk[j0] - f), j, j0)
            n[idx[ok[near] == 1]] = other
        key = frame.astype(np.int64) * 10**9 + n
        if len(np.unique(key)) != len(key):
            continue
        c = n
        taken |= {P, Q}
        done.append(_record(e, c, a))
    return c, done


def merge_two_models(ids, frame, base_identity, embedding, crop_row, log_probs, classes):
    """Step 5. An identity A is merged with B, the identity whose centre is nearest in the embedding of the video,
    when the classifier of step 1 also puts B first among the other identities and A and B never share a frame."""
    E = embedding.astype(np.float32)
    log_probs = log_probs.astype(np.float32)
    classes = classes.astype(int)
    new = ids.copy()
    ci = ids[crop_row]
    names = np.unique(ids)
    total = {int(lab): E[ci == lab].sum(0) for lab in names}
    count = {int(lab): int((ci == lab).sum()) for lab in names}
    present = collections.defaultdict(set)
    for f, lab in zip(frame, ids):
        present[int(lab)].add(int(f))
    size = {int(lab): int((ids == lab).sum()) for lab in names}
    C = {lab: total[lab] / (np.linalg.norm(total[lab]) + 1e-9) for lab in total if count[lab] >= 1}
    L = [lab for lab in C if count[lab] >= 3]
    cand = []
    if len(L) < 3:
        return new, []
    class_identity = np.array(
        [
            collections.Counter(ids[base_identity == c].tolist()).most_common(1)[0][0]
            for c in classes
        ]
    )
    Z = np.stack([C[lab] for lab in L])
    S = Z @ Z.T
    np.fill_diagonal(S, -9)
    for i, A in enumerate(L):
        o = np.argsort(-S[i])
        B = L[o[0]]
        m = float(S[i, o[0]] - S[i, o[1]])
        if S[i, o[0]] < 0.0 or m < 0.0:
            continue
        sj = log_probs[ci == A].mean(0)
        oj = [k for k in np.argsort(-sj) if class_identity[k] != A]
        if not oj or class_identity[oj[0]] != B:
            continue
        cand.append((m, A, B))
    renamed = {}
    done = []
    for m, A, B in sorted(cand, reverse=True):
        while B in renamed:
            B = renamed[B]
        while A in renamed:
            A = renamed[A]
        if A == B or (present[A] & present[B]):
            continue
        small, large = (A, B) if size[A] <= size[B] else (B, A)
        new[new == small] = large
        present[large] |= present[small]
        size[large] += size[small]
        renamed[small] = large
        done.append((int(small), int(large)))
    return new, done
