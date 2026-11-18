"""One video of the chain: its folder, read once.

    tracks.txt             frame, identity of the base tracker, x, y, width, height
    visibility.npy         (N, 10): four overflows of the box over the mask box, visible share, four occlusion
                           flags, contact
    upper_body.npy         (N, 6): upper body readable, upper body hers, anchor jump, anchor x, anchor y, size
    bytetrack.txt          MOT rows of ByteTrack
    passes/native.txt, exclusive.txt, guarded.txt, vote.txt
                           same rows and boxes as bytetrack.txt; column 2 is the identity given by the pass
    passes/backward.txt, mirrored.txt
                           MOT rows of the pass, in the frame of the original video

Every per-box array follows the order of tracks.txt.
"""

import collections
import os

import numpy as np
from scipy.optimize import linear_sum_assignment

from .geometry import iou_matrix, xyxy


def read_mot(path):
    q = np.loadtxt(path, delimiter=",")
    return q.reshape(-1, q.shape[-1])


def map_to_base(base, other, offset=10**7):
    """Objects of a pass renamed to the objects of the first pass, one to one (as in the vote)."""
    bl = sorted(set(base))
    ol = sorted(set(other))
    bi = {lab: i for i, lab in enumerate(bl)}
    oi = {lab: i for i, lab in enumerate(ol)}
    m = np.zeros((len(ol), len(bl)))
    for b, o in zip(base, other):
        m[oi[o], bi[b]] += 1
    r, c = linear_sum_assignment(-m)
    mp = {ol[i]: bl[j] for i, j in zip(r, c) if m[i, j] > 0}
    nxt = offset
    for o in ol:
        if o not in mp:
            mp[o] = nxt
            nxt += 1
    return [mp[o] for o in other]


def project(pass_rows, objects, rows):
    """Objects of a pass on each box of `rows` (-1 if none): one-to-one matching in each frame, IoU >= 0.5.
    `objects` is (K, rows of the pass); the result is (K, boxes)."""
    qb = xyxy(pass_rows)
    qf = pass_rows[:, 0].astype(int)
    pb = xyxy(rows)
    pf = rows[:, 0].astype(int)
    out = np.full((objects.shape[0], len(rows)), -1, np.int64)
    for f in np.unique(pf):
        pi = np.where(pf == f)[0]
        qi = np.where(qf == f)[0]
        if not len(qi):
            continue
        iou = iou_matrix(pb[pi], qb[qi])
        r, c = linear_sum_assignment(-iou)
        for a, b in zip(r, c):
            if iou[a, b] >= 0.5:
                out[:, pi[a]] = objects[:, qi[b]]
    return out


def majority(x):
    """Most frequent non-negative value of x (None if none)."""
    x = x[x >= 0]
    return collections.Counter(x.tolist()).most_common(1)[0][0] if len(x) else None


class Video:
    def __init__(self, folder, fps=20.0):
        self.folder = folder
        self.fps = fps
        self.name = os.path.basename(os.path.normpath(folder))
        self._cache = {}
        t = read_mot(os.path.join(folder, "tracks.txt"))
        self.base = t[:, :6].copy()
        self.frame = self.base[:, 0].astype(int)
        self.n = len(t)

    def rows(self, identities=None):
        """frame, identity, x, y, width, height, with the given identities (those of the base tracker if None)."""
        p = self.base.copy()
        if identities is not None:
            assert len(identities) == self.n
            p[:, 1] = identities
        return p

    def _get(self, key, make):
        if key not in self._cache:
            self._cache[key] = make()
        return self._cache[key]

    @property
    def visibility(self):
        return self._get(
            "visibility",
            lambda: np.load(os.path.join(self.folder, "visibility.npy")).astype(np.float64),
        )

    @property
    def upper_body(self):
        return self._get(
            "upper_body",
            lambda: np.load(os.path.join(self.folder, "upper_body.npy")).astype(np.float64),
        )

    @property
    def forward(self):
        """(5, N): native, exclusive, guarded (both renamed to the objects of the native pass), vote, ByteTrack."""

        def make():
            P = {
                k: read_mot(os.path.join(self.folder, "passes", k + ".txt"))
                for k in ("native", "exclusive", "guarded", "vote")
            }
            vote = P["vote"]
            base = P["native"][:, 1].astype(int).tolist()
            B = read_mot(os.path.join(self.folder, "bytetrack.txt"))
            assert (
                len(B) == len(vote)
                and (B[:, 0] == vote[:, 0]).all()
                and np.abs(B[:, 2:6] - vote[:, 2:6]).max() < 1e-6
            ), self.name
            m = np.stack(
                [np.array(base)]
                + [
                    np.array(map_to_base(base, P[k][:, 1].astype(int).tolist(), (i + 1) * 10**7))
                    for i, k in enumerate(("exclusive", "guarded"))
                ]
                + [vote[:, 1].astype(int), B[:, 1].astype(int)]
            )
            return project(vote, m, self.base)

        return self._get("forward", make)

    def _single(self, name):
        def make():
            f = os.path.join(self.folder, "passes", name + ".txt")
            if not os.path.exists(f):
                return None
            q = read_mot(f)
            return project(q, q[:, 1].astype(np.int64)[None], self.base)[0]

        return self._get(name, make)

    @property
    def backward(self):
        """Object of the backward pass on each box (None if the video has no backward pass)."""
        return self._single("backward")

    @property
    def mirrored(self):
        return self._single("mirrored")
