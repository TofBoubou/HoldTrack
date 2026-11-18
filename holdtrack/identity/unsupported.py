"""Step 10. A box whose identity no pass supports takes the identity that at least two passes give it.

Five passes are read: backward, mirrored, exclusive, guarded, native. The objects of each pass are matched one to
one with the identities, by mutual majority on the safe stretches. A box whose identity none of the five supports,
and that at least two place in one identity absent from the frame, takes this identity. With two identities named
by as many passes, nothing is done.
"""

import collections

import numpy as np

from .geometry import frames as n_frames
from .stretches import stretches as safe_stretches

MIN_S = 1.0


def match_objects(objects, identity, safe, fps=20.0):
    """object of the pass -> identity, one to one, by mutual majority on the safe boxes."""
    ok = safe & (objects >= 0)
    T = collections.Counter(zip(objects[ok].tolist(), identity[ok].tolist()))
    of_object = {}
    of_identity = {}
    for (u, c), k in T.items():
        if u not in of_object or k > T[(u, of_object[u])]:
            of_object[u] = c
        if c not in of_identity or k > T[(of_identity[c], c)]:
            of_identity[c] = u
    return {
        u: c
        for u, c in of_object.items()
        if of_identity.get(c) == u and T[(u, c)] >= n_frames(MIN_S, fps)
    }


def rename_unsupported(video, identities):
    """Returns the new identities and the counts (renamed, held by another box, tie)."""
    p = video.rows(identities)
    pf = p[:, 0].astype(int)
    pid = p[:, 1].astype(np.int64)
    new = pid.copy()
    counts = collections.Counter()
    if video.backward is None or video.mirrored is None:
        return new, counts
    stretch, _ = safe_stretches(video, p, "full")
    safe = stretch >= 0
    fw = video.forward
    passes = [video.backward, video.mirrored, fw[1], fw[2], fw[0]]
    votes = np.stack(
        [
            np.array([m.get(int(u), -1) for u in x])
            for x, m in ((x, match_objects(x, pid, safe, video.fps)) for x in passes)
        ]
    )
    present = collections.defaultdict(set)
    for f, lab in zip(pf, pid):
        present[int(f)].add(int(lab))
    for i in np.where(~(votes == pid[None, :]).any(0) & ((votes >= 0).sum(0) >= 2))[0]:
        cur = int(pid[i])
        fr = int(pf[i])
        best = collections.Counter(int(x) for x in votes[:, i] if x >= 0 and x != cur).most_common(
            2
        )
        if not best or best[0][1] < 2:
            continue
        if len(best) > 1 and best[1][1] == best[0][1]:
            counts["tie"] += 1
            continue
        c = best[0][0]
        if c in present[fr]:
            counts["held"] += 1
            continue
        new[i] = c
        present[fr].add(c)
        counts["renamed"] += 1
    key = pf.astype(np.int64) * 10**9 + new
    assert len(np.unique(key)) == len(key) and (new > 0).all()
    return new, counts
