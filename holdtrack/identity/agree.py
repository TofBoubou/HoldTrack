"""Agreement of a backward pass and a mirrored pass: a training-free identity corrector for mask-based trackers.

Given the tracks of a tracker and two more passes of the same tracker (one on the reversed video, one on
horizontally mirrored frames, both with their boxes put back in the original frame), this module changes an
identity only where the two passes agree:

  1. link step: two tracklets of different identities are joined when both passes give them the same object and
     the links of the input tracks at both ends are supported by neither pass and contradicted by both;
  2. box step: a box that both passes place in another identity takes it if that identity has no box in the
     frame, or exchanges with the box that carries it if both passes place that box in the identity of the first.

Nothing is learned, no appearance or face model is used, and no annotation is read.

Inputs are arrays; `python -m holdtrack.identity.agree` reads MOT files. Durations are given in seconds and
converted with `fps`; the defaults are the values of the system measured on DanceTrack (20 frames per second),
where this module reproduces the identities of the system on every row (`research/replays/identity_agree.py`).
"""

import collections

import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------- geometry
def xyxy(a):
    """(frame, id, x, y, w, h, ...) rows -> boxes as (x1, y1, x2, y2)."""
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


def project(pass_rows, tracks, min_iou=0.5):
    """Object that a pass gives to each box of the tracks (-1 if none): one-to-one matching in each frame."""
    q = np.asarray(pass_rows, float)
    q = q.reshape(-1, q.shape[-1])
    qb = xyxy(q)
    pb = xyxy(tracks)
    qf = q[:, 0].astype(int)
    pf = tracks[:, 0].astype(int)
    out = np.full(len(tracks), -1, np.int64)
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


def clear_boxes(tracks, mask_boxes, reference_boxes=None, tolerance=0.05):
    """A box is clear when the box of its mask covers it to within `tolerance` of its size on each side.

    mask_boxes: (N, 4) as x1, y1, x2, y2, NaN where a box has no mask. reference_boxes: (N, 4) boxes whose
    overflow is measured (default: the boxes of the tracks)."""
    B = xyxy(tracks) if reference_boxes is None else np.asarray(reference_boxes, float)
    m = np.asarray(mask_boxes, float)
    w = B[:, 2] - B[:, 0]
    h = B[:, 3] - B[:, 1]
    dep = np.stack(
        [
            (m[:, 0] - B[:, 0]) / w,
            (m[:, 1] - B[:, 1]) / h,
            (B[:, 2] - m[:, 2]) / w,
            (B[:, 3] - m[:, 3]) / h,
        ],
        1,
    )
    return np.isfinite(dep[:, 0]) & (np.nanmax(dep, 1) < tolerance)


# ---------------------------------------------------------------- tracklets and candidate links
def tracklets(tracks, clear, min_len=10, max_gap=3, max_unclear=2):
    """Runs of clear boxes of one identity. Returns (tracklet index of each box, -1 if none; list of tracklets)."""
    pf = tracks[:, 0].astype(int)
    pid = tracks[:, 1].astype(int)
    seg = np.full(len(tracks), -1)
    S = []

    def close(cur):
        if len(cur) >= min_len:
            seg[cur] = len(S)
            S.append(
                dict(
                    label=int(pid[cur[0]]),
                    start=int(pf[cur[0]]),
                    end=int(pf[cur[-1]]),
                    n=len(cur),
                    lines=np.array(cur),
                )
            )

    for t in np.unique(pid):
        idx = np.where(pid == t)[0]
        idx = idx[np.argsort(pf[idx])]
        cur = []
        for i in idx:
            if clear[i]:
                if cur and pf[i] - pf[cur[-1]] > max_gap:
                    close(cur)
                    cur = []
                cur.append(i)
            elif cur and pf[i] - pf[cur[-1]] > max_unclear:
                close(cur)
                cur = []
        close(cur)
    return seg, S


def candidate_links(tracks, S, max_gap=200, max_dist=1.0, dist_slope=0.01, n_end=5):
    """Pairs (a, b) of tracklets proposed as the same person: the previous tracklet of the identity of b, and the
    last tracklet of every other identity that ends at most `max_gap` frames before b, at a distance of at most
    `max_dist` box heights plus `dist_slope` per frame of gap."""
    cx = tracks[:, 2] + tracks[:, 4] / 2
    yb = tracks[:, 3] + tracks[:, 5]
    hh = tracks[:, 5]
    by_label = collections.defaultdict(list)
    for k, s in enumerate(S):
        by_label[s["label"]].append(k)
    for lab in by_label:
        by_label[lab].sort(key=lambda k: S[k]["start"])

    def end_point(k, last):
        i = S[k]["lines"]
        j = i[-n_end:] if last else i[:n_end]
        return float(np.median(cx[j])), float(np.median(yb[j])), float(np.median(hh[j]))

    FIN = {k: end_point(k, True) for k in range(len(S))}
    DEB = {k: end_point(k, False) for k in range(len(S))}
    A = []
    B = []
    for b, sb in enumerate(S):
        cands = []
        i = by_label[sb["label"]].index(b)
        if i > 0:
            cands.append(by_label[sb["label"]][i - 1])
        for lab, ks in by_label.items():
            if lab == sb["label"]:
                continue
            prec = [k for k in ks if S[k]["end"] < sb["start"]]
            if not prec:
                continue
            a = prec[-1]
            e = sb["start"] - S[a]["end"]
            if e > max_gap:
                continue
            d = np.hypot(DEB[b][0] - FIN[a][0], DEB[b][1] - FIN[a][1]) / max(FIN[a][2], 1.0)
            if d <= max_dist + dist_slope * e:
                cands.append(a)
        for a in cands:
            A.append(a)
            B.append(b)
    return np.array(A, int), np.array(B, int)


def _majority(x):
    return collections.Counter(x.tolist()).most_common(1)[0][0] if len(x) else None


def support(S, A, B, objects, need_half):
    """For each link (a, b): share of the boxes of b to which the pass gives the main object of a (NaN if unknown).
    With need_half, a has a main object only if the pass labels at least half of its boxes."""
    maj = {}
    for k, s in enumerate(S):
        q = objects[s["lines"]]
        q = q[q >= 0]
        maj[k] = (
            _majority(q)
            if (len(q) and (not need_half or len(q) >= 0.5 * len(s["lines"])))
            else None
        )
    out = np.full(len(A), np.nan)
    for i, (a, b) in enumerate(zip(A, B)):
        q = objects[S[int(b)]["lines"]]
        q = q[q >= 0]
        if maj[int(a)] is not None and len(q):
            out[i] = float((q == maj[int(a)]).mean())
    return out


# ---------------------------------------------------------------- the link step
def clean_handover(S, A, B, same, back, mirr):
    """Links between two identities that both passes support, kept only when the handover is clean: on the side of
    b (resp. a), no link of the same identity is supported by a pass, and the nearest one is contradicted by both.
    Returns (forced, removed): boolean arrays over the links."""

    def sup(x):
        return np.nan_to_num(x, nan=0.0) >= 0.5

    def con(x):
        return np.isfinite(x) & (x < 0.5)

    des = sup(back) & sup(mirr) & ~same
    unsup = sup(back) | sup(mirr)
    both_con = con(back) & con(mirr)
    forced = np.zeros(len(A), bool)
    removed = np.zeros(len(A), bool)
    if not des.any():
        return forced, removed
    same_b = collections.defaultdict(list)
    same_a = collections.defaultdict(list)
    for i in range(len(A)):
        if same[i]:
            same_b[B[i]].append(i)
            same_a[A[i]].append(i)

    def free(L, key):
        if not L:
            return True
        if any(unsup[i] for i in L):
            return False
        return bool(both_con[min(L, key=key)])

    per_b = {}
    for i in np.where(des)[0]:
        if not free(same_b[B[i]], lambda j: -S[A[j]]["end"]):
            continue
        if not free(same_a[A[i]], lambda j: S[B[j]]["start"]):
            continue
        b = B[i]
        if b not in per_b or S[A[i]]["end"] > S[A[per_b[b]]]["end"]:
            per_b[b] = i
    per_a = {}
    for i in per_b.values():
        a = A[i]
        if a not in per_a or S[B[i]]["start"] < S[B[per_a[a]]]["start"]:
            per_a[a] = i
    for i in per_a.values():
        forced[i] = True
        for j in same_b[B[i]] + same_a[A[i]]:
            removed[j] = True
    return forced, removed


def apply_links(tracks, seg, S, A, B, score, same, keep_cost=-99.0, start_cost=0.0):
    """Assignment over the whole video: each tracklet takes at most one predecessor and one successor; chains of
    tracklets become identities. A box outside tracklets follows the nearest tracklet of its identity."""
    pf = tracks[:, 0].astype(int)
    pid = tracks[:, 1].astype(int)
    n = len(S)
    if n == 0:
        return pid.copy()
    M = np.full((n, 2 * n), -1e6)
    has_same = np.zeros(n, bool)
    for a, b, sc, me in zip(A, B, score, same):
        if np.isfinite(sc):
            has_same[b] |= bool(me)
            M[b, a] = sc
    for b in range(n):
        M[b, n + b] = keep_cost if has_same[b] else start_cost
    r, c = linear_sum_assignment(-M)
    pred = {int(b): int(a) for b, a in zip(r, c) if a < n and M[b, a] > -1e5}
    root = list(range(n))

    def R(x):
        while root[x] != x:
            x = root[x]
        return x

    for b, a in pred.items():
        root[R(b)] = R(a)
    ch = collections.defaultdict(list)
    for k in range(n):
        ch[R(k)].append(k)
    nearest = np.full(len(pid), -1)
    for lab in np.unique(pid):
        ks = [k for k in range(n) if S[k]["label"] == lab]
        if not ks:
            continue
        idx = np.where(pid == lab)[0]
        deb = np.array([S[k]["start"] for k in ks])
        fin = np.array([S[k]["end"] for k in ks])
        for i in idx:
            d = np.maximum(0, np.maximum(deb - pf[i], pf[i] - fin))
            nearest[i] = ks[int(d.argmin())]
    chain_of = np.array([R(k) if k >= 0 else -1 for k in nearest])
    new = pid.copy()
    taken = collections.defaultdict(set)
    for i in np.where(chain_of < 0)[0]:
        taken[int(pid[i])].add(int(pf[i]))
    for r_ in sorted(ch, key=lambda r_: -int((chain_of == r_).sum())):
        li = np.where(chain_of == r_)[0]
        fr = set(pf[li].tolist())
        vote = collections.Counter(pid[li].tolist())
        name = None
        for lab, _ in vote.most_common():
            if not (taken[int(lab)] & fr):
                name = int(lab)
                break
        if name is None:
            name = 4 * 10**6 + r_
        new[li] = name
        taken[name] |= fr
    key = pf.astype(np.int64) * 10**9 + new.astype(np.int64)
    u, cnt = np.unique(key, return_counts=True)
    dup = set(u[cnt > 1].tolist())
    if dup:
        seen = set()
        for i in np.argsort(seg < 0, kind="stable"):
            if key[i] in dup:
                if key[i] in seen:
                    new[i] = 5 * 10**6 + i
                seen.add(key[i])
    return new


def link_step(tracks, clear, back_obj, mirr_obj, fps=20.0, log=None):
    """Join tracklets of different identities that the backward and the mirrored pass both give to one object."""
    f = fps / 20.0
    seg, S = tracklets(
        tracks,
        clear,
        min_len=int(round(10 * f)),
        max_gap=int(round(3 * f)),
        max_unclear=int(round(2 * f)),
    )
    A, B = candidate_links(tracks, S, max_gap=int(round(200 * f)), dist_slope=0.01 / f)
    if not len(A):
        return tracks[:, 1].astype(np.int64).copy()
    same = np.array([S[a]["label"] == S[b]["label"] for a, b in zip(A, B)])
    back = support(S, A, B, back_obj, need_half=False)
    mirr = support(S, A, B, mirr_obj, need_half=True)
    forced, removed = clean_handover(S, A, B, same, back, mirr)
    score = np.where(same, 5.0, -10.0)
    score[forced] = 10.0
    score[removed] = np.nan
    new = apply_links(tracks, seg, S, A, B, score, same).astype(np.int64)
    if log is not None:
        for i in np.where(forced)[0]:
            a = S[int(A[i])]
            b = S[int(B[i])]
            log.append(
                dict(
                    step="link",
                    frm=dict(identity=a["label"], frames=[a["start"], a["end"]]),
                    to=dict(identity=b["label"], frames=[b["start"], b["end"]]),
                )
            )
    return new


# ---------------------------------------------------------------- the box step
def _one_to_one(objects, pid, sure, min_boxes):
    """Object of a pass -> identity, one to one, by mutual majority over the boxes of tracklets."""
    ok = sure & (objects >= 0)
    T = collections.Counter(zip(objects[ok].tolist(), pid[ok].tolist()))
    bu = {}
    bc = {}
    for (u, c), k in T.items():
        if u not in bu or k > T[(u, bu[u])]:
            bu[u] = c
        if c not in bc or k > T[(bc[c], c)]:
            bc[c] = u
    return {u: c for u, c in bu.items() if bc.get(c) == u and T[(u, c)] >= min_boxes}


def box_step(tracks, clear, back_obj, mirr_obj, fps=20.0, log=None):
    """Rename a box that both passes place in another identity (absent from the frame, or reciprocal exchange)."""
    f = fps / 20.0
    pf = tracks[:, 0].astype(int)
    pid = tracks[:, 1].astype(np.int64)
    seg, S = tracklets(
        tracks,
        clear,
        min_len=int(round(10 * f)),
        max_gap=int(round(3 * f)),
        max_unclear=int(round(2 * f)),
    )
    sure = seg >= 0
    new = pid.copy()
    nmin = int(round(20 * f))
    mb = _one_to_one(back_obj, pid, sure, nmin)
    mm = _one_to_one(mirr_obj, pid, sure, nmin)
    vb = np.array([mb.get(int(u), -1) for u in back_obj])
    vm = np.array([mm.get(int(u), -1) for u in mirr_obj])
    vote = np.where((vb == vm) & (vb >= 0), vb, -1)
    by_frame = collections.defaultdict(dict)
    for i in range(len(tracks)):
        by_frame[int(pf[i])][int(pid[i])] = i
    done = set()
    for i in np.where((vote >= 0) & (vote != pid))[0]:
        if i in done:
            continue
        c = int(vote[i])
        j = by_frame[int(pf[i])].get(c)
        if j is None:
            new[i] = c
            done.add(i)
            if log is not None:
                log.append(dict(step="box", frame=int(pf[i]), frm=int(pid[i]), to=c))
        elif vote[j] == pid[i]:
            new[i] = c
            new[j] = pid[i]
            done |= {i, j}
            if log is not None:
                log.append(dict(step="box exchange", frame=int(pf[i]), identities=[int(pid[i]), c]))
    key = pf.astype(np.int64) * 10**9 + new
    assert len(np.unique(key)) == len(key), "two boxes of one identity in a frame"
    return new


def correct(tracks, clear, backward_rows, mirrored_rows, fps=20.0, log=None):
    """Both steps, in the order of the system. tracks: (N, >=6) rows (frame, identity, x, y, w, h); clear: (N,) bool;
    backward_rows, mirrored_rows: rows of the two passes in the same format. Returns the corrected identities (N,)."""
    tracks = np.asarray(tracks, float).copy()
    back = project(backward_rows, tracks)
    mirr = project(mirrored_rows, tracks)
    tracks[:, 1] = link_step(tracks, clear, back, mirr, fps, log)
    return box_step(tracks, clear, back, mirr, fps, log)


def _read(path):
    return np.loadtxt(path, delimiter=",", ndmin=2)


def main(argv=None):
    """Correct the identities of a MOT file where a backward and a mirrored pass agree."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Correct identities where a backward and a mirrored pass agree. "
        "All files are in MOT format: frame, identity, x, y, width, height[, ...]."
    )
    parser.add_argument("--tracks", required=True, help="tracks to correct")
    parser.add_argument(
        "--mask-boxes", help="boxes of the masks, frames and identities of --tracks"
    )
    parser.add_argument(
        "--clear", help="instead of --mask-boxes: .npy, one boolean per row of --tracks"
    )
    parser.add_argument(
        "--backward", required=True, help="pass on the reversed video, original frames"
    )
    parser.add_argument("--mirrored", required=True, help="pass on mirrored frames, boxes put back")
    parser.add_argument(
        "--out", required=True, help="rows of --tracks with the corrected identities"
    )
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--log", help="JSON file listing the operations")
    options = parser.parse_args(argv)
    if (options.mask_boxes is None) == (options.clear is None):
        parser.error("give exactly one of --mask-boxes and --clear")
    with open(options.tracks) as stream:
        lines = [line for line in stream.read().splitlines() if line.strip()]
    tracks = _read(options.tracks)
    if len(tracks) != len(lines):
        parser.error("--tracks must hold one box per line")
    if options.clear:
        clear = np.load(options.clear).astype(bool)
        if len(clear) != len(tracks):
            parser.error("--clear must hold one boolean per row of --tracks")
    else:
        masks = _read(options.mask_boxes)
        row_of = {(int(r[0]), int(r[1])): i for i, r in enumerate(masks)}
        mask_boxes = np.full((len(tracks), 4), np.nan)
        for i, r in enumerate(tracks):
            j = row_of.get((int(r[0]), int(r[1])))
            if j is not None:
                x, y, w, h = masks[j, 2:6]
                mask_boxes[i] = [x, y, x + w, y + h]
        clear = clear_boxes(tracks, mask_boxes)
    log = []
    new = correct(tracks, clear, _read(options.backward), _read(options.mirrored), options.fps, log)
    with open(options.out, "w") as stream:
        for line, identity in zip(lines, new, strict=True):
            cells = line.split(",")
            cells[1] = str(int(identity))
            stream.write(",".join(cells) + "\n")
    if options.log:
        with open(options.log, "w") as stream:
            json.dump(log, stream, indent=1)
    changed = int((new != tracks[:, 1].astype(np.int64)).sum())
    print(
        f"{len(tracks)} boxes, {int(clear.sum())} clear, {changed} identities changed,"
        f" {len(log)} operations"
    )


if __name__ == "__main__":
    main()
