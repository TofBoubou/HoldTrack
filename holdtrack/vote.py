"""Combine the three identity trajectories."""

from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

VOICES = ("BASE", "EXCL", "PROT")


def map_to_base(base, other, offset=10**7):
    bl = sorted(set(base))
    ol = sorted(set(other))
    bi = {label: i for i, label in enumerate(bl)}
    oi = {label: i for i, label in enumerate(ol)}
    m = np.zeros((len(ol), len(bl)))
    for b, o in zip(base, other):
        m[oi[o], bi[b]] += 1
    observation, c = linear_sum_assignment(-m)
    mp = {ol[i]: bl[j] for i, j in zip(observation, c) if m[i, j] > 0}
    nxt = offset
    for o in ol:
        if o not in mp:
            mp[o] = nxt
            nxt += 1
    return [mp[o] for o in other]


def vote_ids(base, excl, prot, frames, mapper):
    if not len(base) == len(excl) == len(prot) == len(frames):
        raise ValueError("Unaligned vote rows")
    if max(base, default=0) >= 10**7 or len(set(zip(frames, base))) != len(base):
        raise ValueError("BASE namespace or frame-ID collision")
    e, q = (mapper(base, excl, offset=10**7), mapper(base, prot, offset=2 * 10**7))
    labels = []
    for b, x, y in zip(base, e, q):
        winner, count = Counter((b, x, y)).most_common(1)[0]
        labels.append(winner if count >= 2 else b)
    groups = defaultdict(list)
    for i, frame in enumerate(frames):
        groups[frame].append(i)
    resets = 0
    while True:
        changed = False
        for indices in groups.values():
            counts = Counter(labels[i] for i in indices)
            for i in indices:
                if counts[labels[i]] > 1 and labels[i] != base[i]:
                    labels[i] = base[i]
                    resets += 1
                    changed = True
        if not changed:
            break
    if len(set(zip(frames, labels))) != len(base):
        raise ValueError("Vote frame-ID collision")
    return (labels, resets)


def vote_bytes(data):
    if set(data) != set(VOICES):
        raise ValueError("Exactly three three-voice voices required")
    tokens = {
        k: [line.split(b",") for line in v.splitlines(keepends=True)] for k, v in data.items()
    }
    base = tokens["BASE"]
    for role, rows in tokens.items():
        if len(rows) != len(base):
            raise ValueError("Voice physical support differs")
        for a, b in zip(base, rows):
            if len(a) != len(b) or a[:1] + a[2:] != b[:1] + b[2:]:
                raise ValueError("Voice changes columns other than identity")
    mapper = map_to_base
    voter = vote_ids
    ids = {role: [int(float(row[1])) for row in rows] for role, rows in tokens.items()}
    frames = [int(float(row[0])) for row in base]
    labels, resets = voter(ids["BASE"], ids["EXCL"], ids["PROT"], frames, mapper)
    for row, label in zip(base, labels):
        row[1] = str(label).encode("ascii")
    return (
        b"".join(b",".join(row) for row in base),
        dict(rows=len(base), uniqueness_resets=resets, GT_opened=False),
    )
