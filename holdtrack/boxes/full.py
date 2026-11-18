"""Finish the complete system on a set of videos: corrected boxes, identities, final tracks.

Each video folder holds, besides what ``pipeline`` reads (frames/, tracks.txt, masks/,
reads/), the files of the identity chain: fusion.txt (output of the box fusion, before the
track-level steps), bytetrack.txt, passes/{native,exclusive,guarded,vote,backward,mirrored}.txt.

    reference boxes  ->  identity chain  ->  [face steps]  ->  relay in place and its check  ->  full corrector  ->  final.txt

The box stages that need a GPU (reads, pose17, pose133, mesh) are run before, with
``pipeline``; this module runs the CPU stages around them and the identity chain.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import assemble, pipeline
from . import relay as relay_module
from . import short_tracks as short_module
from . import upper_body as upper_module
from . import visibility as visibility_module


def identity_inputs(root: Path) -> None:
    """Write, in each video folder, the three tables that the identity chain reads.
    Needs the reference boxes (``pipeline`` stage ``reference``) and the 17 keypoints."""
    work = root / "_boxes"
    for v in pipeline.videos(root):
        d = root / v
        a = np.load(work / "tables_reference" / f"{v}.npz")
        masks = np.load(work / "tables_reference" / f"{v}_masks.npz")["masks"]
        boxes = pipeline.corners(
            np.loadtxt(work / "reference" / f"{v}.txt", delimiter=",", ndmin=2)
        )
        np.save(
            d / "visibility.npy",
            visibility_module.visibility(boxes, masks, a["hidden"], a["contact"]),
        )
        np.save(
            d / "upper_body.npy",
            upper_module.upper_body(a["frame"], a["pid"], boxes, np.load(d / "pose17.npz")["kp"]),
        )
        np.save(
            d / "short_tracks.npy",
            short_module.short_tracks(
                np.loadtxt(d / "fusion.txt", delimiter=",", ndmin=2), a["frame"], a["ref"]
            ),
        )


def final_tracks(root: Path, check=None, fps: float = 20.0, faces=None) -> dict:
    """Relay in place on the identities of the chain, then the final rows of each video.
    ``check`` is the relay check of the identity chain (``identity.check.relays``). ``faces``,
    when given, is the function of the face steps (``faces.chain.run_folder``): it is applied to
    the identities of the chain before the relay."""
    report = {}
    for v in pipeline.videos(root):
        d = root / v
        q = pipeline.rows(root, v)
        ids = np.load(d / "identities.npy")
        attached = np.load(d / "attached.npy").reshape(-1, 7)
        box_rows = [line for line in (d / "boxes.txt").read_text().split("\n") if line]
        boxes = pipeline.corners(np.loadtxt(d / "boxes.txt", delimiter=",", ndmin=2))
        before, before_attached = ids, attached
        if faces is not None:
            before, before_attached = faces(str(d), ids, attached, boxes, fps=fps)
        _, _, found = relay_module.relay(q[:, 0], before, boxes, before_attached)
        removed = (
            check(d, before, [(c[1], c[2]) for c in found], fps=fps) if (check and found) else []
        )
        new, new_attached, found = relay_module.relay(
            q[:, 0], before, boxes, before_attached, excluded={(int(x), int(y)) for x, y in removed}
        )
        rows = assemble.assemble(box_rows, ids, attached, new, new_attached)
        (d / "final.txt").write_text("\n".join(rows) + "\n")
        report[v] = dict(rows=len(rows), relays=len(found), relays_removed=len(removed))
        if faces is not None:
            report[v]["renamed_by_faces"] = int(
                (np.asarray(before) != np.asarray(ids).ravel()).sum()
            )
    return report
