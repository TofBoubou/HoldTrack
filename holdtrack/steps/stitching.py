"""Reconnect compatible track fragments."""

from collections import defaultdict


def stitch(pred, maxgap=60, maxd=0.5):
    out = {}
    nlinks = 0
    for v, rows in pred.items():
        T = defaultdict(dict)
        for frame_index, i, x1, y1, x2, y2 in rows:
            T[i][frame_index] = ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
        span = {i: (min(data), max(data)) for i, data in T.items()}
        cand = []
        for a, (s1, e1) in span.items():
            b1 = T[a][e1]
            for b, (s2, e2) in span.items():
                if b == a or not 0 < s2 - e1 <= maxgap:
                    continue
                b2 = T[b][s2]
                data = ((b1[0] - b2[0]) ** 2 + (b1[1] - b2[1]) ** 2) ** 0.5 / max(b1[3], 1)
                if data < maxd:
                    cand.append((data, a, b))
        cand.sort()
        used_end = set()
        used_start = set()
        parent = {}
        for data, a, b in cand:
            if a in used_end or b in used_start:
                continue
            used_end.add(a)
            used_start.add(b)
            parent[b] = a

        def root(i):
            while i in parent:
                i = parent[i]
            return i

        nlinks += len(parent)
        out[v] = [(frame_index, root(i), *b) for frame_index, i, *b in rows]
    return (out, nlinks)
