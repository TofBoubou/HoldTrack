"""Final check of the late operations: take-overs and interval exchanges. No annotation is read.

A take-over gives a new identity B the name of an identity A that ended shortly before. It is removed when the
backward pass gives B another ancestor than A: in the 7.5 seconds before the birth of B, the object of the backward
pass that holds B was carried mostly by another identity than A, on at least 2.5 seconds of boxes.

An interval exchange swaps two identities I and K between two frames. It is removed when at least four of the five
passes (backward, mirrored, exclusive, guarded, native) keep the identities in place, and at least half of the faces
of the valid sides have the centre of the eyes inside the box of the other identity.

    relays(video, identities, pairs)                              -> pairs (A, B) to remove
    exchanges(video, identities, records, face_rows, eye_centers) -> records to remove

`video` is the folder of the video (tracks.txt, bytetrack.txt, passes/); `identities` has one entry per row of
tracks.txt and holds the identities before the operations are applied.
"""

import collections

import numpy as np

from .geometry import frames as n_frames
from .geometry import xyxy
from .video import Video

PASSES = ("backward", "mirrored", "exclusive", "guarded", "native")
WINDOW_S = 7.5
EDGE_S = 5.0
MIN_S = 2.5
KEEP_MIN = 4
FACE_SHARE_MIN = 0.5


def _video(video, fps=20.0):
    return video if isinstance(video, Video) else Video(video, fps)


def _majority(x):
    y = x[x >= 0]
    if not len(y) or not len(x):
        return None, 0.0
    k, n = collections.Counter(y.tolist()).most_common(1)[0]
    return int(k), n / len(x)


def relay_ancestor(frames, identities, backward, B, fps=20.0):
    """Identities that carried, before the birth of B, the object of the backward pass that holds B: [(identity, boxes)]."""
    window = n_frames(WINDOW_S, fps)
    edge = n_frames(EDGE_S, fps)
    ib = np.where(identities == B)[0]
    ib = ib[np.argsort(frames[ib])]
    born = int(frames[ib[0]])
    obj, _ = _majority(backward[ib[:edge]])
    if obj is None:
        return None
    k = np.where((backward == obj) & (frames < born) & (frames >= born - window))[0]
    return (
        [(int(a), int(b)) for a, b in collections.Counter(identities[k].tolist()).most_common(2)]
        if len(k)
        else None
    )


def relays(video, identities, pairs, fps=20.0):
    """Take-overs to remove among `pairs` [(A, B)]: B takes the name of A."""
    video = _video(video, fps)
    frames = video.frame
    F = np.asarray(identities).astype(np.int64).ravel()
    assert len(F) == len(frames)
    out = []
    for A, B in pairs:
        anc = (
            relay_ancestor(frames, F, video.backward, int(B), fps)
            if video.backward is not None
            else None
        )
        if anc and anc[0][0] != int(A) and anc[0][1] >= n_frames(MIN_S, fps):
            out.append((int(A), int(B)))
    return out


def exchanges(video, identities, records, face_rows, eye_centers):
    """Interval exchanges to remove. `records`: dicts with I, K, t_in, t_out (frames, t_out excluded) and valid = (side
    I valid, side K valid), in the order they are applied. `face_rows`: rows of tracks.txt that have a face;
    `eye_centers`: (N, 2), centre of the eyes of each row. Returns the records to remove."""
    video = _video(video)
    frames = video.frame
    box = xyxy(video.base)
    fw = video.forward
    none = np.full(video.n, -1, np.int64)
    L = dict(
        backward=video.backward if video.backward is not None else none,
        mirrored=video.mirrored if video.mirrored is not None else none,
        exclusive=fw[1],
        guarded=fw[2],
        native=fw[0],
    )
    F = np.asarray(identities).astype(np.int64).ravel().copy()
    n = len(F)
    assert n == len(frames)
    face = np.zeros(n, bool)
    face[np.asarray(face_rows, int)] = True
    c = np.asarray(eye_centers, float)
    out = []
    for o in records:
        id_i, K = int(o["I"]), int(o["K"])
        t0 = int(o["t_in"])
        t1 = int(o["t_out"])
        valid = o["valid"]
        inside = (frames >= t0) & (frames < t1)
        ii = np.where(inside & (F == id_i))[0]
        ki = np.where(inside & (F == K))[0]
        ko = np.where(~inside & (F == K))[0]
        io = np.where(~inside & (F == id_i))[0]
        keep = set()
        for name in PASSES:
            x = L[name]
            (ni, pi), (nd, pd), (nh, ph), (nih, pih) = (
                _majority(x[ii]),
                _majority(x[ki]),
                _majority(x[ko]),
                _majority(x[io]),
            )
            if (
                not (ni is not None and ni == nh and pi >= 0.5 and ph >= 0.5)
                and nd is not None
                and nd == nh
                and nd != ni
                and pd >= 0.5
                and ph >= 0.5
            ):
                keep.add(name)
            if (
                not (nd is not None and nd == nih and pd >= 0.5 and pih >= 0.5)
                and ni is not None
                and ni == nih
                and ni != nd
                and pi >= 0.5
                and pih >= 0.5
            ):
                keep.add(name)
        rows = {
            (int(lab), int(f)): i
            for i, (lab, f) in enumerate(zip(F, frames))
            if inside[i] and lab in (id_i, K)
        }
        faces = 0
        in_other = 0
        for name, other, ok in ((id_i, K, valid[0]), (K, id_i, valid[1])):
            if not ok:
                continue
            for i in (i for (lab, f), i in rows.items() if lab == name and face[i]):
                faces += 1
                j = rows.get((other, int(frames[i])))
                if j is not None:
                    in_other += int(
                        box[j][0] <= c[i][0] <= box[j][2] and box[j][1] <= c[i][1] <= box[j][3]
                    )
        if len(keep) >= KEEP_MIN and faces and in_other / faces >= FACE_SHARE_MIN:
            out.append(o)
        F[ii] = K
        F[ki] = id_i
    return out
