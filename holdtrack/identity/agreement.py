"""Steps 6 to 9. Identities that two passes agree on.

Steps 6 and 7 are the agreement module (agree.py): links, then boxes, on which the backward pass and the mirrored
pass agree. Steps 8 and 9 add the forward passes: a box that the backward pass and one of three forward passes
(mirrored, exclusive, guarded) place in one identity c, other than its own,
  takes c if c is absent from the frame;
  exchanges with the box that holds c if that box is placed likewise in the identity of the first;
  else, if no forward pass votes for its present identity and the box that holds c was not changed, the two boxes
  exchange their identities.
Step 8 matches the objects of a pass with the identities on the safe stretches; step 9 on every box whose contact
is at most 0.5.
"""

import collections

import numpy as np

from . import agree
from .stretches import safe_rows, safety
from .stretches import stretches as safe_stretches
from .unsupported import match_objects


def _both(video):
    return video.backward is not None and video.mirrored is not None


def links(video, identities):
    """Step 6."""
    if not _both(video):
        return np.asarray(identities).astype(np.int64).copy()
    return agree.link_step(
        video.rows(identities),
        safe_rows(video.visibility, "full"),
        video.backward,
        video.mirrored,
        video.fps,
    )


def boxes(video, identities):
    """Step 7."""
    if not _both(video):
        return np.asarray(identities).astype(np.int64).copy()
    return agree.box_step(
        video.rows(identities),
        safe_rows(video.visibility, "full"),
        video.backward,
        video.mirrored,
        video.fps,
    )


def boxes_with_forward(video, identities, wide=False):
    """Step 8 (wide=False) and step 9 (wide=True). Returns the identities and the counts."""
    p = video.rows(identities)
    pf = p[:, 0].astype(int)
    pid = p[:, 1].astype(np.int64)
    new = pid.copy()
    counts = collections.Counter()
    if not _both(video):
        return new, counts
    safe = safety(video, "iou05")[0] if wide else safe_stretches(video, p, "full")[0] >= 0
    fw = video.forward

    def placed(objects):
        m = match_objects(objects, pid, safe, video.fps)
        return np.array([m.get(int(u), -1) for u in objects])

    back = placed(video.backward)
    forward = [placed(x) for x in (video.mirrored, fw[1], fw[2])]
    agreed = np.zeros(len(p), bool)
    for w in forward:
        agreed |= w == back
    vote = np.where(agreed & (back >= 0), back, -1)
    holder = collections.defaultdict(dict)
    for i in range(len(p)):
        holder[int(pf[i])][int(pid[i])] = i
    done = set()
    for i in np.where((vote >= 0) & (vote != pid))[0]:
        if i in done:
            continue
        c = int(vote[i])
        j = holder[int(pf[i])].get(c)
        if j is None:
            new[i] = c
            counts["renamed"] += 1
            done.add(i)
            holder[int(pf[i])][c] = i
        elif vote[j] == pid[i]:
            new[i] = c
            new[j] = pid[i]
            counts["exchanged"] += 2
            done |= {i, j}
        elif j not in done and not any(w[i] == pid[i] for w in forward):
            new[i] = c
            new[j] = pid[i]
            counts["exchanged, no forward pass for the present identity"] += 2
            done |= {i, j}
        else:
            counts["untouched"] += 1
    key = pf.astype(np.int64) * 10**9 + new
    assert len(np.unique(key)) == len(key) and (new > 0).all()
    return new, counts
