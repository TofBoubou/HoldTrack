"""The face steps on rows without a readable face, and on two identities whose faces are exchanged."""

import numpy as np

from holdtrack.faces import cuts, exchange, pieces
from holdtrack.faces.pieces import FaceTable, Rows, Settings


def _rows(n=120):
    """Two dancers side by side, their boxes overlapping slightly, on n frames."""
    frame = np.r_[np.arange(1, n + 1), np.arange(1, n + 1)]
    track = np.r_[np.ones(n, np.int64), 2 * np.ones(n, np.int64)]
    box = np.r_[
        np.tile([100.0, 100.0, 180.0, 300.0], (n, 1)), np.tile([150.0, 100.0, 230.0, 300.0], (n, 1))
    ]
    return frame, track, box


def _table(rows, sides, seed=0):
    rng = np.random.default_rng(seed)
    centre = rng.normal(size=(2, 512))
    e = centre[sides] + 0.3 * rng.normal(size=(len(rows), 512))
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    return FaceTable(
        rows.astype(np.int32),
        e.astype(np.float16),
        np.full(len(rows), 20.0, np.float32),
        None,
        np.full(len(rows), 12.0, np.float32),
    )


def test_no_face_changes_nothing():
    frame, track, box = _rows()
    empty = FaceTable(
        np.zeros(0, np.int32), np.zeros((0, 512), np.float16), np.zeros(0, np.float32)
    )
    rows = Rows(frame, track, cuts.contact(frame, box))
    ids = track.copy()
    short = np.zeros((0, 7))
    new, short2, _, merged, _ = pieces.run(
        rows, ids, short, Settings(main=empty, second=empty), None
    )
    assert np.array_equal(new, ids) and len(short2) == 0 and not merged
    new, _ = cuts.cut_and_link(frame, track, box, ids, empty)
    assert np.array_equal(new, ids)
    new, _, log = exchange.exchange(rows, box, ids, short, empty, empty)
    assert np.array_equal(new, ids) and not log


def test_exchange_found_by_the_faces():
    frame, track, box = _rows()
    n = len(frame) // 2
    person = np.r_[np.zeros(n, int), np.ones(n, int)]
    ids = track.copy()
    swapped = (frame >= 41) & (frame <= 80)
    ids[swapped] = 3 - ids[swapped]  # the two names are exchanged over forty frames
    a = _table(np.arange(len(frame)), person, 0)
    b = _table(np.arange(len(frame)), person, 1)
    rows = Rows(frame, track, cuts.contact(frame, box))
    new, _, log = exchange.exchange(rows, box, ids, np.zeros((0, 7)), a, b)
    assert len(log) == 1
    faces = [set(new[person == k].tolist()) for k in (0, 1)]
    assert len(faces[0]) == 1 and len(faces[1]) == 1 and faces[0] != faces[1]
