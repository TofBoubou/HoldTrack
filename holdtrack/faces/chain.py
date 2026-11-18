"""The face steps, in order, on one video.

    whole identities, pieces supported by a pass, merges         (pieces)
    the face and a reader                                        (reader)
    cut a name where its face changes, then link                 (cuts)
    pieces proposed by both models, candidates rejudged          (pieces)
    exchanges without a cut, validated by both models, checked   (exchange, identity.check)

The last three are repeated until a turn changes nothing, three turns at most. The input is the output of
the identity chain: the identities of the rows of the tracks and the short-track rows. Nothing here reads an
annotation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..identity import check
from ..identity.geometry import xyxy
from . import cuts, pieces, reader
from .exchange import exchange
from .pieces import FaceTable, Rows, Settings

TURNS = 3


@dataclass
class Tables:
    """Face tables of one video: the two models, and each without the faces shared by two boxes."""

    arcface: FaceTable
    adaface: FaceTable
    arcface_clean: FaceTable
    adaface_clean: FaceTable


def _settings(t: Tables, **more):
    return Settings(
        main=t.adaface,
        second=t.arcface,
        main_pieces=t.adaface_clean,
        second_pieces=t.arcface_clean,
        merge_threshold=0.5,
        second_merge_threshold=0.7,
        fallback=True,
        no_piece_from_main_alone=True,
        **more,
    )


def run(
    video, identities, short, final_boxes, tables: Tables, eye_centers, excluded_pieces=(), log=None
):
    """Identities and short-track rows after the face steps.

    `video` is a `holdtrack.identity.video.Video`; `identities` has one entry per row of its tracks; `short` holds the
    short-track rows (frame, identity, x1, y1, x2, y2, track); `final_boxes` are the corrected boxes of the rows
    (x1, y1, x2, y2); `eye_centers` the centre of the eyes of every row, from the 133 keypoints."""
    log = {} if log is None else log
    frame = video.frame
    track = video.base[:, 1].astype(np.int64)
    box = xyxy(video.base)
    rows = Rows(frame, track, cuts.contact(frame, box))
    ids = np.asarray(identities).astype(np.int64).ravel()
    short = np.asarray(short, float)

    # whole identities, pieces, merges
    ids1, short, log["pieces"], _, log["merges"] = pieces.run(
        rows, ids, short, _settings(tables), reader.supported_by_a_pass(video, ids)
    )
    # the face and a reader
    ids2 = reader.keep_clear_of_short(
        frame, ids1, reader.face_and_reader(video, ids1, tables.arcface), short
    )
    current = ids2
    removed = []
    log["turns"] = []
    for turn in range(TURNS):
        before = current.copy()
        # cut and link; from the second turn, a tie goes to the row that already carried the name
        current, cut_log = cuts.cut_and_link(
            frame, track, box, current, tables.adaface_clean, holder_first=turn > 0
        )
        # pieces proposed by both models, rejudged after each operation
        s = _settings(
            tables,
            no_whole=True,
            no_merge=True,
            rejudge=True,
            piece_gallery_contact=1.01,
            excluded=set(excluded_pieces),
        )
        base = current
        current, short4, piece_log, _, _ = pieces.run(rows, base, short, s, "both")
        # exchanges without a cut, then their check
        new, short5, found = exchange(
            rows,
            final_boxes,
            current,
            short4,
            tables.adaface_clean,
            tables.arcface_clean,
            1.01,
            removed,
        )
        records = [
            dict(
                I=c["I"],
                K=c["K"],
                t_in=c["t_in"],
                t_out=c["t_out"],
                valid=(c["sides"]["I"]["valid"], c["sides"]["K"]["valid"]),
            )
            for c in found
        ]
        bad = (
            check.exchanges(video, current, records, tables.arcface.index, eye_centers)
            if records
            else []
        )
        if bad:
            removed = removed + [
                (int(o["I"]), int(o["K"]), int(o["t_in"]), int(o["t_out"])) for o in bad
            ]
            new, short5, found = exchange(
                rows,
                final_boxes,
                current,
                short4,
                tables.adaface_clean,
                tables.arcface_clean,
                1.01,
                removed,
            )
        log["turns"].append(
            dict(cuts=cut_log, pieces=piece_log, exchanges=found, removed=list(removed))
        )
        changed = (new != before).any() or not np.array_equal(short5, short)
        current, short = new, short5
        if not changed:
            break
    return current, short


def prepare_tables(folder, models, device="cuda", fps=20.0):
    """Write the face tables of a video folder (tracks.txt, frames/, pose133.npz) in <folder>/faces.
    `models` is the folder written by fetch_face_models.py (arcface/glintr100.onnx, adaface/)."""
    import os

    from . import second_eye, tables

    faces = os.path.join(folder, "faces")
    if not os.path.exists(os.path.join(faces, "aligned.npz")):
        tables.align(folder)
    if not os.path.exists(os.path.join(faces, "arcface.npz")):
        tables.embed_arcface(
            folder, tables.load_arcface(os.path.join(models, "arcface", "glintr100.onnx"), device)
        )
    if not os.path.exists(os.path.join(faces, "adaface_clean.npz")):
        second_eye.write_tables(
            folder,
            adaface=second_eye.load_adaface(os.path.join(models, "adaface"), device),
            device=device,
            fps=fps,
        )


def run_folder(folder, identities, short, final_boxes, fps=20.0, log=None):
    """`run` on a video folder whose face tables are written (`prepare_tables`)."""
    import os

    from ..identity.video import Video

    faces = os.path.join(folder, "faces")
    tables = Tables(
        *(
            FaceTable.load(os.path.join(faces, name + ".npz"))
            for name in ("arcface", "adaface", "arcface_clean", "adaface_clean")
        )
    )
    kp = np.load(os.path.join(folder, "pose133.npz"))["kp"].astype(np.float64)
    eyes = (kp[:, 59:65, :2].mean(1) + kp[:, 65:71, :2].mean(1)) / 2
    return run(Video(folder, fps), identities, short, final_boxes, tables, eyes, log=log)


if __name__ == "__main__":
    import sys

    prepare_tables(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]))
