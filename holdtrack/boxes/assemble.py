"""Assemble the final tracks of a video from the corrected boxes and the identities.

The steps act on the MOT rows of the corrected boxes, in this order: identities of the
identity chain; short tracks that the chain attaches; final identities; gaps of one or
two frames filled; identities of fewer than four boxes dropped; boxes rounded to the
pixel. A row is never reordered, and every field other than the identity and the box is
copied character for character.
"""

from __future__ import annotations

from collections import Counter

import numpy as np


def join(rows: list[str], identities) -> list[str]:
    """Rows of the corrected boxes with the identity of each row replaced."""
    ids = np.asarray(identities).ravel()
    if len(ids) != len(rows) or not np.all(ids == np.round(ids)) or not (ids > 0).all():
        raise ValueError("One positive integer identity is needed for each row")
    out, keys = [], set()
    for row, k in zip(rows, ids.astype(np.int64)):
        f = row.split(",")
        key = (f[0], int(k))
        if key in keys:
            raise ValueError(f"Identity {key[1]} appears twice in frame {key[0]}")
        keys.add(key)
        out.append(",".join([f[0], str(int(k))] + f[2:]))
    return out


def add_attached(rows: list[str], attached) -> list[str]:
    """Add the attached rows (frame, identity, left, top, right, bottom, track), after
    the rows of their frame."""
    frames = np.array([int(r.split(",")[0]) for r in rows])
    keys = {tuple(r.split(",")[:2]) for r in rows}
    known = {r.split(",")[1] for r in rows}
    new: dict[int, list[str]] = {}
    for r in sorted(np.asarray(attached, float).reshape(-1, 7).tolist()):
        t, i = int(r[0]), int(r[1])
        x1, y1, x2, y2 = r[2:6]
        key = (str(t), str(i))
        if not (x2 > x1 and y2 > y1 and i > 0 and str(i) in known) or key in keys:
            raise ValueError(f"Attached row {key} is invalid or already present")
        keys.add(key)
        new.setdefault(t, []).append(
            ",".join(
                [
                    str(t),
                    str(i),
                    f"{x1:.8f}",
                    f"{y1:.8f}",
                    "%.8f" % (x2 - x1),
                    "%.8f" % (y2 - y1),
                    "1.00000000",
                    "-1",
                    "-1",
                    "-1",
                ]
            )
        )
    return _merge(rows, frames, new)


def _merge(rows, frames, new):
    out, k = [], 0
    for t in sorted(set(frames.tolist()) | set(new)):
        while k < len(rows) and frames[k] == t:
            out.append(rows[k])
            k += 1
        out += new.get(t, [])
    return out


def set_identities(
    rows: list[str], frames, before, after, attached_before, attached_after
) -> list[str]:
    """Rename each row from its identity before to its identity after. ``frames``,
    ``before`` and ``after`` follow the rows of the table; the two attached arrays hold
    the same rows in the same order."""
    table = {}
    for x, y, z in zip(
        np.asarray(frames).tolist(),
        np.asarray(before).ravel().tolist(),
        np.asarray(after).ravel().tolist(),
    ):
        table[(int(x), int(y))] = int(z)
    a0 = np.asarray(attached_before, float).reshape(-1, 7)
    a1 = np.asarray(attached_after, float).reshape(-1, 7)
    if a0.shape != a1.shape or not (a0[:, 0] == a1[:, 0]).all():
        raise ValueError("The attached rows before and after are not the same rows")
    for r0, r1 in zip(a0.tolist(), a1.tolist()):
        table.setdefault((int(r0[0]), int(r0[1])), int(r1[1]))
    out = []
    for row in rows:
        f = row.split(",")
        z = table[(int(f[0]), int(f[1]))]
        if z != int(f[1]):
            f[1] = str(z)
            row = ",".join(f)
        out.append(row)
    keys = [tuple(r.split(",")[:2]) for r in out]
    if len(set(keys)) != len(keys):
        raise ValueError("An identity appears twice in a frame")
    return out


def gaps(frames, identities, boxes, longest: int = 2, overlap: float = 0.5):
    """Rows that fill the gaps of one or two frames of an identity whose two boxes
    around the gap overlap at an IoU of at least 0.5: frame, identity, interpolated
    box, and the index of the row before the gap."""
    fr = np.asarray(frames).astype(np.int64)
    ids = np.asarray(identities).astype(np.int64)
    b_all = np.asarray(boxes, float)
    out_f, out_i, out_b, out_a = [], [], [], []
    for q in np.unique(ids):
        ii = np.where(ids == q)[0]
        ii = ii[np.argsort(fr[ii], kind="stable")]
        ff = fr[ii]
        for k in np.where((ff[1:] - ff[:-1] >= 2) & (ff[1:] - ff[:-1] <= longest + 1))[0]:
            a, b = b_all[ii[k]], b_all[ii[k + 1]]
            w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
            h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
            i = w * h
            c = i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)
            if c < overlap:
                continue
            n = int(ff[k + 1] - ff[k] - 1)
            for d in range(1, n + 1):
                out_f.append(int(ff[k] + d))
                out_i.append(int(q))
                out_b.append(a + (d / (n + 1)) * (b - a))
                out_a.append(int(ii[k]))
    return (
        np.array(out_f, np.int64),
        np.array(out_i, np.int64),
        np.array(out_b, float).reshape(-1, 4),
        np.array(out_a, np.int64),
    )


def fill_gaps(rows: list[str]) -> list[str]:
    q = np.array([[float(x) for x in r.split(",")[:6]] for r in rows])
    frames = q[:, 0].astype(np.int64)
    ids = q[:, 1].astype(np.int64)
    boxes = np.stack([q[:, 2], q[:, 3], q[:, 2] + q[:, 4], q[:, 3] + q[:, 5]], 1)
    f_new, i_new, b_new, before = gaps(frames, ids, boxes)
    new: dict[int, list[str]] = {}
    for t, i, x, a in sorted(zip(f_new.tolist(), i_new.tolist(), b_new.tolist(), before.tolist())):
        tail = rows[a].split(",")[6:]
        new.setdefault(t, []).append(
            ",".join(
                [
                    str(t),
                    str(i),
                    f"{x[0]:.8f}",
                    f"{x[1]:.8f}",
                    "%.8f" % (x[2] - x[0]),
                    "%.8f" % (x[3] - x[1]),
                ]
                + tail
            )
        )
    return _merge(rows, frames, new)


def drop_short(rows: list[str], least: int = 4) -> list[str]:
    ids = [r.split(",")[1] for r in rows]
    count = Counter(ids)
    return [r for r, i in zip(rows, ids) if count[i] >= least]


def round_boxes(rows: list[str]) -> list[str]:
    out = []
    for row in rows:
        t = row.split(",")
        x, y, w, h = (float(z) for z in t[2:6])
        x1, y1, x2, y2 = np.round([x, y, x + w, y + h])
        x2, y2 = max(x2, x1 + 1), max(y2, y1 + 1)
        t[2:6] = [f"{x1:.8f}", f"{y1:.8f}", "%.8f" % (x2 - x1), "%.8f" % (y2 - y1)]
        out.append(",".join(t))
    return out


def assemble(
    box_rows: list[str], chain_identities, chain_attached, final_identities, final_attached
) -> list[str]:
    """Final MOT rows of a video. ``box_rows`` are the rows of the corrected boxes."""
    frames = [int(r.split(",")[0]) for r in box_rows]
    rows = join(box_rows, chain_identities)
    rows = add_attached(rows, chain_attached)
    rows = set_identities(
        rows, frames, chain_identities, final_identities, chain_attached, final_attached
    )
    return round_boxes(drop_short(fill_gaps(rows)))
