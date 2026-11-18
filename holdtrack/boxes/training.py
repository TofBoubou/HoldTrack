"""Train the trees of the box corrector on annotated tracks, and apply them.

The corrector moves each edge of a box by a prediction relative to the box size. Its trees
read the features of `holdtrack.boxes.features`, the pose and mesh features, and, when frame
rates are given, the track features of `holdtrack.boxes.track_features`. The target of an
edge is the annotated box that the benchmark matches to the box: its offset from the box,
relative to the box size, clipped at `clip`.

Folders of a set of tracks, one file per video:
  tables/<video>.npz and <video>_masks.npz   table of the tracks; `gt` holds the target boxes
  pose17/<video>.npz, pose133/<video>.npz      keypoints of the two pose estimators
  mesh/<video>.npz                             body-mesh outputs

    python -m holdtrack.boxes.training --train SET --apply SET --out FOLDER \
        --thresholds pose_thresholds.json --mhr70 mhr70.py [--clip 2.0] [--frame-rates FILE]

`--save FILE` writes the fitted trees, and `--trees-file FILE` reads them in place of `--train`.
`--networks A B ...` averages the boxes of the trees with the mean of sequence networks
(`holdtrack.boxes.network_training`), with weight two on the trees.
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import HistGradientBoostingRegressor

from . import features, mesh_features, pose_features
from .corrector import DROPPED, SIGN
from .table import body_views, iou_matrix
from .track_features import track_features

GROUPS = (0, 1, 2)


@dataclass
class TrackSet:
    """Folders of one set of tracks, and the two files that the pose and mesh features need."""

    root: Path
    thresholds: Path
    mhr70: Path
    frame_rates: dict | None = (
        None  # {video: frames per second}; None leaves the track features out
    )

    def videos(self) -> list[str]:
        tables = (self.root / "tables").glob("*.npz")
        return sorted(p.stem for p in tables if not p.stem.endswith("_masks"))

    def table(self, video: str) -> dict:
        with np.load(self.root / "tables" / f"{video}.npz") as a:
            return {k: a[k] for k in a.files}


def annotated_targets(frames, boxes, annotation, extended=False):
    """Return the target box (N, 4) of each box, NaN where it has none, and its person.

    annotation: rows of a MOT annotation file, with every class. In each frame the boxes
    are matched one to one with all annotated boxes at an IoU of at least 0.5, and a box
    matched to an evaluated pedestrian takes its annotated box. With `extended`, a box left
    without a target, that lies for 80% of its area inside a pedestrian that no box covers,
    takes this pedestrian (IoU of at least 0.15, one pedestrian per box).
    """
    frames = np.asarray(frames).astype(int)
    boxes = np.asarray(boxes, float)
    annotation = np.asarray(annotation, float)
    annotated = np.stack(
        [
            annotation[:, 2],
            annotation[:, 3],
            annotation[:, 2] + annotation[:, 4],
            annotation[:, 3] + annotation[:, 5],
        ],
        1,
    )
    annotated_frames = annotation[:, 0].astype(int)
    pedestrian = (annotation[:, 7] == 1) & (annotation[:, 6] == 1)
    targets = np.full((len(frames), 4), np.nan)
    persons = np.full(len(frames), -1, np.int64)
    other = np.zeros(len(frames), bool)
    for frame in np.unique(frames):
        rows = np.where(frames == frame)[0]
        candidates = np.where(annotated_frames == frame)[0]
        if not len(candidates):
            continue
        iou = iou_matrix(boxes[rows], annotated[candidates])
        first, second = linear_sum_assignment(-iou)
        for x, y in zip(first, second):
            if iou[x, y] >= 0.5:
                if pedestrian[candidates[y]]:
                    targets[rows[x]] = annotated[candidates[y]]
                    persons[rows[x]] = int(annotation[candidates[y], 1])
                else:
                    other[rows[x]] = True
        if not extended:
            continue
        taken = {int(second[k]) for k in range(len(first)) if iou[first[k], second[k]] >= 0.5}
        free = [y for y in range(len(candidates)) if y not in taken and pedestrian[candidates[y]]]
        alone = [
            x
            for x in range(len(rows))
            if not np.isfinite(targets[rows[x], 0]) and not other[rows[x]]
        ]
        if not free or not alone:
            continue
        b = boxes[rows][alone]
        g = annotated[candidates][free]
        inter = np.clip(
            np.minimum(b[:, None, 2], g[None, :, 2]) - np.maximum(b[:, None, 0], g[None, :, 0]),
            0,
            None,
        ) * np.clip(
            np.minimum(b[:, None, 3], g[None, :, 3]) - np.maximum(b[:, None, 1], g[None, :, 1]),
            0,
            None,
        )
        inside = inter / np.maximum((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]), 1e-9)[:, None]
        score = np.where(inside >= 0.8, iou[np.ix_(alone, free)], 0.0)
        first, second = linear_sum_assignment(-score)
        for x, y in zip(first, second):
            if score[x, y] >= 0.15:
                targets[rows[alone[x]]] = g[y]
                persons[rows[alone[x]]] = int(annotation[candidates[free[y]], 1])
    return targets, persons


def edge_features(tracks: TrackSet, video: str, table: dict, keep: list) -> dict:
    """Return {group: (features as float32, rows, edges, sizes)} for the boxes of the table."""
    limits = json.loads(Path(tracks.thresholds).read_text())
    t1, t2 = float(limits["T1"]), float(limits["T2"])
    reference = table["ref"].astype(float)
    base = features.video_features(str(tracks.root / "tables"), video, {**table, "ref": reference})
    if not keep:
        keep.extend(
            i for i, n in enumerate(features.NAMES) if not any(n.startswith(r) for r in DROPPED)
        )
    along_track = None
    if tracks.frame_rates is not None:
        along_track = track_features(
            table["frame"], table["pid"], reference, tracks.frame_rates[video]
        )
    out = {}
    for g in GROUPS:
        rows, edges, sizes = base[g][1], base[g][2], base[g][3]
        parts = [
            base[g][0][:, keep].astype(np.float32),
            pose_features.pose_features(
                video,
                table,
                reference,
                rows,
                edges,
                t1,
                t2,
                str(tracks.root / "pose17"),
                str(tracks.root / "pose133"),
            ),
            mesh_features.mesh_features(
                video, table, reference, rows, edges, str(tracks.root / "mesh"), str(tracks.mhr70)
            ),
            mesh_features.mesh_time_features(
                video, table, reference, rows, edges, str(tracks.root / "mesh")
            ),
        ]
        if along_track is not None:
            parts.append(along_track[rows])
        out[g] = (np.asarray(np.concatenate(parts, 1), np.float32), rows, edges, sizes)
    return out


def edge_targets(table: dict, rows, edges, sizes):
    """Offset of the target edge from the edge of the box, relative to the box size."""
    reference = table["ref"].astype(float)
    return (table["gt"][rows, edges] - reference[rows, edges]) * SIGN[edges] / sizes


def new_model(trees=800):
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        max_iter=trees,
        learning_rate=0.05,
        max_leaf_nodes=63,
        min_samples_leaf=100,
        l2_regularization=1.0,
        random_state=0,
    )


def fit_trees(tracks: TrackSet, clip=0.3, trees=800, videos=None):
    """Fit one model per group of edges on the boxes that have a target.

    Returns {group: (model, columns without any value)} and the kept feature columns. The
    features of each video are kept on disk while the others are computed.
    """
    videos = tracks.videos() if videos is None else videos
    keep: list = []
    models = {}
    with tempfile.TemporaryDirectory() as folder:
        stored = {}
        for video in videos:
            table = tracks.table(video)
            built = edge_features(tracks, video, table, keep)
            for g in GROUPS:
                x, rows, edges, sizes = built[g]
                path = f"{folder}/{len(stored)}.npy"
                np.save(path, x)
                stored[video, g] = (path, edge_targets(table, rows, edges, sizes))
            del built
            gc.collect()
        for g in GROUPS:
            used = {v: np.flatnonzero(np.isfinite(stored[v, g][1])) for v in videos}
            count = sum(len(rows) for rows in used.values())
            width = np.load(stored[videos[0], g][0], mmap_mode="r").shape[1]
            x = np.empty((count, width), np.float64)
            y = np.empty(count)
            start = 0
            for video in videos:
                rows = used[video]
                if len(rows):
                    x[start : start + len(rows)] = np.load(stored[video, g][0], mmap_mode="r")[rows]
                    y[start : start + len(rows)] = stored[video, g][1][rows]
                    start += len(rows)
            empty = ~np.isfinite(x).any(0)
            x[:, empty] = 0
            models[g] = (new_model(trees).fit(x, np.clip(y, -clip, clip)), empty)
            del x, y
            gc.collect()
    return models, keep


def apply_trees(models, built, reference):
    """Return the boxes moved by the models; a box that degenerates keeps its reference."""
    reference = np.asarray(reference, float)
    new = reference.copy()
    for g in GROUPS:
        x, rows, edges, sizes = built[g]
        model, empty = models[g]
        x = np.array(x)
        x[:, empty] = 0
        new[rows, edges] = reference[rows, edges] + SIGN[edges] * model.predict(x) * sizes
    bad = (new[:, 2] <= new[:, 0] + 1) | (new[:, 3] <= new[:, 1] + 1)
    new[bad] = reference[bad]
    return new


def network_tables(tracks: TrackSet, folder: Path) -> None:
    """Write the tables that the sequence networks read: the views of each table, then the
    three views taken from the body (mesh box, boxes of the 17 and of the 133 keypoints)."""
    folder.mkdir(parents=True, exist_ok=True)
    for video in tracks.videos():
        table = tracks.table(video)
        with np.load(tracks.root / "mesh" / f"{video}.npz") as a:
            mesh_box = a["mesh_box"]
        with np.load(tracks.root / "pose17" / f"{video}.npz") as a:
            points17 = a["kp"]
        with np.load(tracks.root / "pose133" / f"{video}.npz") as a:
            points133 = a["kp"]
        body = body_views(table["ref"], mesh_box, points17, points133)
        np.savez(
            folder / f"{video}.npz",
            frame=table["frame"],
            pid=table["pid"],
            ref=table["ref"],
            views=np.concatenate([table["views"], body[0]]),
            views_iou=np.concatenate([table["views_iou"], body[1]]),
            views_score=np.concatenate([table["views_score"], body[2]]),
        )
        shutil.copyfile(
            tracks.root / "tables" / f"{video}_masks.npz", folder / f"{video}_masks.npz"
        )


def network_offsets(tracks: TrackSet, weights) -> dict:
    """Mean prediction of the sequence networks for the rows of each video of the set."""
    from . import networks

    with tempfile.TemporaryDirectory() as folder:
        network_tables(tracks, Path(folder))
        mean = sum(networks.predict(folder, w).astype(float) for w in weights) / len(weights)
    out = {}
    start = 0
    for video in tracks.videos():
        n = len(tracks.table(video)["ref"])
        out[video] = mean[start : start + n]
        start += n
    return out


def average(reference, trees_boxes, offsets):
    """Boxes of the first stage: the offsets of the trees and of the networks, weights 2 and 1."""
    reference = np.asarray(reference, float)
    size = np.stack([reference[:, 2] - reference[:, 0], reference[:, 3] - reference[:, 1]] * 2, 1)
    mixed = (2 * (trees_boxes - reference) * SIGN / size + offsets) / 3
    out = reference + SIGN[None] * mixed * size
    bad = (out[:, 2] <= out[:, 0] + 1) | (out[:, 3] <= out[:, 1] + 1)
    out[bad] = reference[bad]
    return out


def correct(models, keep, tracks: TrackSet, output, weights=None) -> dict:
    """Write one corrected MOT file per video of the set and return the number of rows."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    offsets = network_offsets(tracks, weights) if weights else None
    rows = {}
    for video in tracks.videos():
        table = tracks.table(video)
        boxes = apply_trees(models, edge_features(tracks, video, table, keep), table["ref"])
        if offsets is not None:
            boxes = average(table["ref"], boxes, offsets[video])
        lines = [
            f"{frame:d},{identity:d},{b[0]:.8f},{b[1]:.8f},{b[2] - b[0]:.8f},{b[3] - b[1]:.8f},"
            "1,-1,-1,-1\n"
            for frame, identity, b in zip(table["frame"], table["pid"], boxes)
        ]
        (output / f"{video}.txt").write_text("".join(lines))
        rows[video] = len(lines)
        gc.collect()
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("apply", "out", "thresholds", "mhr70"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--train", help="annotated set of tracks; not needed with --trees-file")
    parser.add_argument("--save", help="write the fitted trees to this file")
    parser.add_argument("--trees-file", help="read fitted trees instead of training them")
    parser.add_argument("--networks", nargs="+", help="weights of the sequence networks, averaged")
    parser.add_argument("--clip", type=float, default=0.3)
    parser.add_argument("--trees", type=int, default=800)
    parser.add_argument("--frame-rates", help="JSON file {video: frames per second} of both sets")
    a = parser.parse_args()
    rates = json.loads(Path(a.frame_rates).read_text()) if a.frame_rates else None
    target = TrackSet(Path(a.apply), Path(a.thresholds), Path(a.mhr70), rates)
    if a.trees_file:
        with open(a.trees_file, "rb") as stream:
            stored = pickle.load(stream)
        fitted, columns = stored["trees"], stored["columns"]
    else:
        if not a.train:
            parser.error("--train or --trees-file is required")
        training = TrackSet(Path(a.train), Path(a.thresholds), Path(a.mhr70), rates)
        fitted, columns = fit_trees(training, clip=a.clip, trees=a.trees)
    if a.save:
        with open(a.save, "wb") as stream:
            pickle.dump({"trees": fitted, "columns": columns}, stream)
    written = correct(fitted, columns, target, a.out, a.networks)
    print(len(written), "videos,", sum(written.values()), "rows")
