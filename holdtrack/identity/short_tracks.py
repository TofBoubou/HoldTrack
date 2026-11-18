"""Step 12. Attach the short tracks that the base system removed, when two proofs of different kinds agree.

Three proofs name an identity for a short track:
  passes      each of the five passes names the identity where it places at least half of the boxes of the track;
              the proof is the identity named by most passes (none on a tie);
  appearance  a logistic regression of the video on the appearance features of the crops, one class per identity;
              the proof is the best identity among those absent from the frame for at least half of the boxes;
  body        an identity has a box on the first box of the track in the frame before it (IoU >= 0.7) and none in
              its first frame, or the same after its end.
The track is attached to the identity that at least two proofs name. Its boxes are added where the identity is
absent from the frame.
"""

import collections
import os

import numpy as np

from .crops import short_rows
from .geometry import xyxy
from .stretches import stretches as safe_stretches
from .unsupported import match_objects
from .video import map_to_base, project, read_mot

BODY_IOU = 0.7
MIN_CROPS = 5


def _iou(a, B):
    B = np.atleast_2d(B)
    iw = np.clip(np.minimum(a[2], B[:, 2]) - np.maximum(a[0], B[:, 0]), 0, None)
    ih = np.clip(np.minimum(a[3], B[:, 3]) - np.maximum(a[1], B[:, 1]), 0, None)
    i = iw * ih
    return i / np.maximum(
        (a[2] - a[0]) * (a[3] - a[1]) + (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]) - i, 1e-9
    )


def votes(video, p, cand):
    """(5, M): identity voted by each pass (backward, mirrored, exclusive, guarded, native) for each box of the short
    tracks (-1: silent). The objects of a pass are matched with the identities on the safe stretches of the tracks."""
    if not len(cand):
        return np.zeros((5, 0), np.int64)
    if video.backward is None or video.mirrored is None:
        return np.full((5, len(cand)), -1, np.int64)
    pid = p[:, 1].astype(np.int64)
    stretch, _ = safe_stretches(video, p, "full")
    safe = stretch >= 0
    fw = video.forward
    M = [
        match_objects(x, pid, safe, video.fps)
        for x in (video.backward, video.mirrored, fw[1], fw[2], fw[0])
    ]
    table = np.concatenate([p[:, :6], cand])
    n = len(p)
    folder = os.path.join(video.folder, "passes")
    X = []
    for k in ("backward", "mirrored"):
        q = read_mot(os.path.join(folder, k + ".txt"))
        X.append(project(q, q[:, 1].astype(np.int64)[None], table)[0])
    P = {
        k: read_mot(os.path.join(folder, k + ".txt"))
        for k in ("native", "exclusive", "guarded", "vote")
    }
    base = P["native"][:, 1].astype(int).tolist()
    m = np.stack(
        [np.array(base)]
        + [
            np.array(map_to_base(base, P[k][:, 1].astype(int).tolist(), (i + 1) * 10**7))
            for i, k in enumerate(("exclusive", "guarded"))
        ]
    )
    out = project(P["vote"], m, table)
    XE = np.stack([X[0], X[1], out[1], out[2], out[0]])
    return np.stack([np.array([M[r].get(int(u), -1) for u in XE[r, n:]]) for r in range(5)])


def attach(video, identities, short_tracks, crops_meta, features, short_meta, short_features):
    """short_tracks: (M, 6) frame, track, x1, y1, x2, y2. crops_meta, features: crops of the tracks and their
    appearance features; short_meta, short_features: the same for the short tracks. Returns (K, 7): frame, identity,
    x1, y1, x2, y2, short track; and the counts."""
    from sklearn.linear_model import LogisticRegression

    p = video.rows(identities)
    cand = short_rows(short_tracks)
    counts = collections.Counter()
    VO = votes(video, p, cand)
    if not len(cand) or video.mirrored is None:
        return np.zeros((0, 7)), counts
    pf = p[:, 0].astype(int)
    pid = p[:, 1].astype(np.int64)
    box = xyxy(p)
    cf = cand[:, 0].astype(int)
    ck = cand[:, 1].astype(np.int64)
    cb = xyxy(cand)
    at = {}
    in_frame = collections.defaultdict(list)
    present = collections.defaultdict(set)
    for i, (f, lab) in enumerate(zip(pf, pid)):
        at[(int(f), int(lab))] = i
        in_frame[int(f)].append(i)
        present[int(f)].add(int(lab))
    tracks = {int(k): np.where(ck == k)[0][np.argsort(cf[ck == k])] for k in np.unique(ck)}
    Xp = features.astype(np.float32)
    yp = pid[crops_meta[:, 9].astype(int)]
    Xc = short_features.astype(np.float32)
    row = short_meta[:, 9].astype(int)
    assert len(Xp) == len(crops_meta) and len(Xc) == len(short_meta)
    names, nb = np.unique(yp, return_counts=True)
    keep = np.isin(yp, names[nb >= MIN_CROPS])
    clf = LogisticRegression(C=10, max_iter=300)
    clf.fit(Xp[keep], yp[keep])
    LP = clf.predict_log_proba(Xc)
    K = clf.classes_

    def body(b, t, step):
        near = in_frame.get(t + step, [])
        if not near:
            return None
        for k_, x in sorted(zip(near, _iou(b, box[near])), key=lambda z: -z[1]):
            if x < BODY_IOU:
                break
            lab = int(pid[k_])
            if (t, lab) in at:
                continue
            return lab
        return None

    added = []
    for k, idx in tracks.items():
        counts["tracks"] += 1
        named = []
        for r in range(5):
            C = collections.Counter(int(x) for x in VO[r, idx] if x >= 0).most_common(1)
            named.append(C[0][0] if C and C[0][1] * 2 >= len(idx) else -1)
        D = collections.Counter(x for x in named if x >= 0).most_common(2)
        by_passes = D[0][0] if D and not (len(D) > 1 and D[1][1] == D[0][1]) else None
        by_appearance = None
        crops = np.where(np.isin(row, idx))[0]
        if len(crops):
            note = LP[crops].mean(0)
            free = [
                j
                for j in np.argsort(-note)
                if sum(int(K[j]) not in present[int(cf[i])] for i in idx) * 2 >= len(idx)
            ]
            if free:
                by_appearance = int(K[free[0]])
        J = body(cb[idx[0]], int(cf[idx[0]]), -1)
        id_i = body(cb[idx[-1]], int(cf[idx[-1]]), +1)
        by_body = (
            J
            if (J is not None and (id_i is None or id_i == J))
            else id_i
            if (id_i is not None and J is None)
            else None
        )
        agree = [
            x[0]
            for x in ((by_passes, by_appearance), (by_passes, by_body), (by_appearance, by_body))
            if x[0] is not None and x[0] == x[1]
        ]
        if not agree or len(set(agree)) != 1:
            continue
        c = agree[0]
        counts["attached"] += 1
        for i in idx:
            f = int(cf[i])
            if c in present[f]:
                counts["boxes not added"] += 1
                continue
            present[f].add(c)
            added.append((f, c, *[float(z) for z in cb[i]], int(k)))
            counts["boxes added"] += 1
    return (np.array(added, np.float64) if added else np.zeros((0, 7))), counts
