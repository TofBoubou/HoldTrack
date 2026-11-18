"""Apply the trained box corrector to the tracks of a video set.

Inputs, one file per video in each folder of the layout:
  tables/<video>.npz and <video>_masks.npz   rows of the tracks with the detector boxes and the tracker masks
  pose17/<video>.npz, pose133/<video>.npz      keypoints of the two pose estimators
  mesh/<video>.npz                             body-mesh outputs
  tracks/<video>.txt                           MOT rows of the tracks, in the order of the table
and the trained models: the two stages of trees, the thresholds of the pose features, and
the predictions of the two sequence networks on these tables.

Stage 1 moves each edge of each box by the prediction of the trees; the result is averaged
with the two networks, with weight two on the trees; stage 2 corrects this average with a
second set of trees. Rows, identities, order and every field other than the box are kept.
No annotation is read.
"""

from __future__ import annotations

import gc
import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import features, mesh_features, pose_features

SIGN = np.array([-1, -1, 1, 1.0])
DROPPED = (
    "neighbour",
    "hidden",
    "contact",
    "people",
    "width",
    "height",
    "aspect",
    "bottom",
    "border_distance",
)


@dataclass
class Layout:
    tables: Path
    tracks: Path
    trees: Path
    networks: tuple[Path, Path]
    # The four fields below are None for the corrector without pose and mesh features,
    # whose boxes are the reference of the identity chain.
    pose17: Path | None = None
    pose133: Path | None = None
    mesh: Path | None = None
    thresholds: Path | None = None
    mhr70: Path | None = None  # keypoint names of the body-mesh model, from its public repository


def correct(layout: Layout, output: str | Path) -> dict:
    """Write one corrected MOT file per video and return the manifest."""
    full = layout.mesh is not None
    if full:
        limits = json.loads(Path(layout.thresholds).read_text())
        t1, t2 = float(limits["T1"]), float(limits["T2"])
    videos = sorted(
        p.stem for p in Path(layout.tables).glob("*.npz") if not p.stem.endswith("_masks")
    )
    tables = {}
    for v in videos:
        with np.load(Path(layout.tables) / f"{v}.npz") as a:
            tables[v] = {k: a[k] for k in a.files}
    with open(layout.trees, "rb") as f:
        models = pickle.load(f)
    keep: list = [None]

    def build(v, ref):
        f = features.video_features(str(layout.tables), v, {**tables[v], "ref": ref})
        if keep[0] is None:
            keep[0] = [
                i for i, n in enumerate(features.NAMES) if not any(n.startswith(r) for r in DROPPED)
            ]
        out = {}
        for g in (0, 1, 2):
            rows, edges = f[g][1], f[g][2]
            if not full:
                out[g] = (f[g][0][:, keep[0]], rows, edges, f[g][3])
                continue
            out[g] = (
                np.concatenate(
                    [
                        f[g][0][:, keep[0]].astype(np.float32),
                        pose_features.pose_features(
                            v,
                            tables[v],
                            ref,
                            rows,
                            edges,
                            t1,
                            t2,
                            str(layout.pose17),
                            str(layout.pose133),
                        ),
                        mesh_features.mesh_features(
                            v, tables[v], ref, rows, edges, str(layout.mesh), str(layout.mhr70)
                        ),
                        mesh_features.mesh_time_features(
                            v, tables[v], ref, rows, edges, str(layout.mesh)
                        ),
                    ],
                    1,
                ),
                rows,
                edges,
                f[g][3],
            )
        return out

    def stage(v, trees, ref):
        f = build(v, ref)
        new = ref.copy()
        for g in (0, 1, 2):
            x, rows, edges, sizes = f[g]
            new[rows, edges] = ref[rows, edges] + SIGN[edges] * trees[g].predict(x) * sizes
        bad = (new[:, 2] <= new[:, 0] + 1) | (new[:, 3] <= new[:, 1] + 1)
        new[bad] = ref[bad]
        return new

    def average(ref, trees_boxes, networks):
        size = np.stack([ref[:, 2] - ref[:, 0], ref[:, 3] - ref[:, 1]] * 2, 1)
        m = (2 * (trees_boxes - ref) * SIGN / size + networks) / 3
        out = ref + SIGN[None] * m * size
        bad = (out[:, 2] <= out[:, 0] + 1) | (out[:, 3] <= out[:, 1] + 1)
        out[bad] = ref[bad]
        return out

    networks = {v: 0 for v in videos}
    for path in layout.networks:
        x = np.load(path)["pred"]
        o = 0
        for v in videos:
            n = len(tables[v]["ref"])
            networks[v] = networks[v] + x[o : o + n].astype(float) / 2
            o += n
        if o != len(x):
            raise ValueError(f"{path}: the predictions do not match the tables")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)

    def digest(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()

    manifest = {
        "trees": digest(layout.trees),
        "networks": [digest(p) for p in layout.networks],
        "videos": {},
    }
    for v in videos:
        a = tables[v]
        ref = a["ref"]
        new = stage(v, models["etage2"], average(ref, stage(v, models["etage1"], ref), networks[v]))
        gc.collect()
        source = Path(layout.tracks) / f"{v}.txt"
        lines = source.read_text().split("\n")
        body = [line for line in lines if line]
        if len(body) != len(new):
            raise ValueError(f"{v}: the tracks and the table have different numbers of rows")
        out = []
        for line, b, frame, identity in zip(body, new, a["frame"], a["pid"]):
            f = line.split(",")
            if int(f[0]) != frame or int(f[1]) != identity:
                raise ValueError(f"{v}: the tracks and the table are not in the same order")
            out.append(
                ",".join(
                    f[:2]
                    + [f"{b[0]:.8f}", f"{b[1]:.8f}", "%.8f" % (b[2] - b[0]), "%.8f" % (b[3] - b[1])]
                    + f[6:]
                )
            )
        target = output / f"{v}.txt"
        target.write_text("\n".join(out) + ("\n" if lines[-1] == "" else ""))
        manifest["videos"][v] = {"rows": len(body), "sha256": digest(target)}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("tables", "tracks", "trees", "out"):
        p.add_argument("--" + name, required=True)
    for name in ("pose17", "pose133", "mesh", "thresholds", "mhr70"):
        p.add_argument("--" + name)
    p.add_argument("--networks", nargs=2, required=True)
    a = p.parse_args()

    def opt(x):
        return Path(x) if x else None

    m = correct(
        Layout(
            Path(a.tables),
            Path(a.tracks),
            Path(a.trees),
            (Path(a.networks[0]), Path(a.networks[1])),
            opt(a.pose17),
            opt(a.pose133),
            opt(a.mesh),
            opt(a.thresholds),
            opt(a.mhr70),
        ),
        a.out,
    )
    print(len(m["videos"]), "videos,", sum(x["rows"] for x in m["videos"].values()), "rows")
