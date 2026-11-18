"""Corrected boxes of a set of videos, from the pixels, in stages.

The set is a folder with one folder per video:
    frames/                      images, in order
    tracks.txt                   MOT rows of the tracks of the base system
    masks/native.txt, exclusive.txt, guarded.txt
                                 MOT rows of the three forward tracker passes (boxes of the masks)
    reads/yolo.npz, yolo_mirrored.npz, dfine.npz
                                 detections of the two other detectors (frames from 0, boxes, scores)

Stages, in order (each one is a command, so that a stage can run in its own environment):
    reads       the person detector reads every frame: whole, mirrored, at 1280, by bands, on crops     (GPU)
    reference   table of the detector views, sequence networks, reference corrector                     (CPU)
    pose17      17 keypoints around the reference boxes                                                 (GPU)
    pose133     133 keypoints around the reference boxes                                                (GPU)
    mesh        body mesh prompted with the reference boxes                                             (GPU)
    full        table with the body views, sequence networks, full corrector                            (CPU)
The reference boxes go to the identity chain; the boxes of the last stage are the final boxes.

    python pipeline.py <stage> --set FOLDER --models models.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent
PASSES = ("native", "exclusive", "guarded")


def videos(root: Path) -> list[str]:
    return sorted(p.name for p in root.iterdir() if (p / "tracks.txt").is_file())


def images(root: Path, v: str) -> list[Path]:
    folder = root / v / "frames"
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def rows(root: Path, v: str) -> np.ndarray:
    return np.loadtxt(root / v / "tracks.txt", delimiter=",", ndmin=2)


def corners(q: np.ndarray) -> np.ndarray:
    return np.stack([q[:, 2], q[:, 3], q[:, 2] + q[:, 4], q[:, 3] + q[:, 5]], 1)


def stage_reads(root: Path, models: dict) -> None:
    """A read that is already in the folder of the video is kept."""
    from . import reads

    reader = None
    for v in videos(root):
        out = root / v / "reads"
        out.mkdir(exist_ok=True)
        full = [
            (n, r, m)
            for n, r, m in (
                ("rf", None, False),
                ("rf_mirrored", None, True),
                ("rf_1280", 1280, False),
            )
            if not (out / f"{n}.npz").exists()
        ]
        bands = [
            (n, f, r)
            for n, f, r in (("bands3", reads.BANDS_3, 1120), ("bands5", reads.BANDS_5, 1280))
            if not (out / f"{n}_{len(f) - 1}.npz").exists()
        ]
        crops = [
            (n, g, m)
            for n, g, m in (("crop_tight", 1.25, False), ("crop_wide", 1.7, True))
            if not (out / f"{n}.npz").exists()
        ]
        if not (full or bands or crops):
            continue
        if reader is None:
            reader = reads.Reader(Path(models["rf_2xl"]), models["rf_2xl_specification"])
        t = time.time()
        im, q = images(root, v), rows(root, v)
        for name, resolution, mirrored in full:
            f, b, s = reader.full(im, resolution, mirrored)
            np.savez_compressed(
                out / f"{name}.npz", frames=f.astype(np.int64) - 1, boxes=b, scores=s
            )
        for name, fractions, resolution in bands:
            for k, (left, right, f, b, s) in enumerate(reader.bands(im, fractions, resolution)):
                np.savez_compressed(
                    out / f"{name}_{k}.npz",
                    left=left,
                    right=right,
                    frames=f.astype(np.int64) - 1,
                    boxes=b,
                    scores=s,
                )
        base = corners(q).astype(np.float32)
        for name, margin, mirrored in crops:
            box, iou, score = reader.crops(im, q[:, 0], base, margin, mirrored)
            np.savez_compressed(out / f"{name}.npz", box=box, iou=iou, score=score)
        log.info(
            "%s: %d frames, %d rows, reads in %.0f s; outputs skipped: %s",
            v,
            len(im),
            len(q),
            time.time() - t,
            reader.skipped,
        )


def _bank(path: Path, floor: float):
    a = np.load(path)
    return a["frames"].astype(int) + 1, a["boxes"].astype(float), a["scores"].astype(float), floor


def _tables(root: Path, work: Path, body: bool) -> None:
    from . import table as T

    for v in videos(root):
        d, q = root / v, rows(root, v)
        r = d / "reads"
        full = [
            _bank(r / "rf.npz", 0.3),
            _bank(r / "rf_mirrored.npz", 0.3),
            _bank(r / "yolo.npz", 0.3),
            _bank(r / "yolo_mirrored.npz", 0.3),
            _bank(r / "dfine.npz", 0.5),
            _bank(r / "rf_1280.npz", 0.3),
        ]
        bands = []
        for name, count in (("bands3", 3), ("bands5", 5)):
            view = []
            for k in range(count):
                a = np.load(r / f"{name}_{k}.npz")
                view.append(
                    (
                        float(a["left"]),
                        float(a["right"]),
                        a["frames"].astype(int) + 1,
                        a["boxes"].astype(float),
                        a["scores"].astype(float),
                    )
                )
            bands.append(view)
        crops = []
        for name in ("crop_tight", "crop_wide"):
            a = np.load(r / f"{name}.npz")
            crops.append((a["box"].astype(float), a["iou"].astype(float), a["score"].astype(float)))
        masks = [np.loadtxt(d / "masks" / f"{p}.txt", delimiter=",", ndmin=2) for p in PASSES]
        if body:
            mesh = np.load(d / "mesh.npz")
            mesh_box = np.full((len(q), 4), np.nan, np.float32)
            mesh_box[mesh["row"]] = mesh["mesh_box"]
            table, mask_table = T.build_table(
                q,
                full,
                bands,
                crops,
                masks,
                mesh_box,
                np.load(d / "pose17.npz")["kp"],
                np.load(d / "pose133.npz")["kp"],
            )
            parts = (("tables_full", 13), ("tables_ten", 10))
        else:
            table, mask_table = T.build_table(q, full, bands, crops, masks)
            parts = (("tables_reference", 9),)
        for name, count in parts:
            (work / name).mkdir(parents=True, exist_ok=True)
            part = dict(table)
            part.update(
                views=table["views"][:count],
                views_iou=table["views_iou"][:count],
                views_score=table["views_score"][:count],
            )
            np.savez(work / name / f"{v}.npz", **part)
            np.savez(work / name / f"{v}_masks.npz", **mask_table)
        (work / "tracks").mkdir(exist_ok=True)
        link = work / "tracks" / f"{v}.txt"
        if not link.exists():
            link.symlink_to((d / "tracks.txt").resolve())


def _networks(work: Path, tables: str, weights: list[str], views: int) -> tuple[Path, Path]:
    from . import networks

    (work / "networks").mkdir(exist_ok=True)
    out = []
    for k, w in enumerate(weights):
        path = work / "networks" / f"{tables}_{k}.npz"
        np.savez_compressed(path, pred=networks.predict(work / tables, w, views))
        out.append(path)
    return out[0], out[1]


def stage_reference(root: Path, models: dict) -> None:
    from . import corrector as K

    work = root / "_boxes"
    work.mkdir(exist_ok=True)
    _tables(root, work, body=False)
    nets = _networks(work, "tables_reference", models["networks_reference"], 9)
    K.correct(
        K.Layout(work / "tables_reference", work / "tracks", Path(models["trees_reference"]), nets),
        work / "reference",
    )
    for v in videos(root):
        q = np.loadtxt(work / "reference" / f"{v}.txt", delimiter=",", ndmin=2)
        np.save(root / v / "reference_boxes.npy", corners(q).astype(np.float32))
    log.info("reference boxes written: %d videos", len(videos(root)))


def _around_reference(root: Path, name: str, compute) -> None:
    for v in videos(root):
        out = root / v / f"{name}.npz"
        if out.exists():
            continue
        t = time.time()
        q = rows(root, v)
        result = compute(
            images(root, v), q[:, 0].astype(int), np.load(root / v / "reference_boxes.npy")
        )
        if isinstance(result, dict):  # mesh: arrays in the order of its row index
            np.savez_compressed(out, pid=q[result["row"], 1].astype(np.int64), **result)
        else:
            np.savez_compressed(
                out, frame=q[:, 0].astype(int), pid=q[:, 1].astype(np.int64), kp=result
            )
        log.info("%s: %s of %d rows in %.0f s", v, name, len(q), time.time() - t)


def stage_pose17(root: Path, models: dict) -> None:
    from . import pose

    _around_reference(
        root,
        "pose17",
        lambda im, f, b: pose.keypoints17(
            im, f, b, models.get("pose17", "usyd-community/vitpose-plus-huge")
        ),
    )


def stage_pose133(root: Path, models: dict) -> None:
    from . import pose

    _around_reference(
        root, "pose133", lambda im, f, b: pose.keypoints133(im, f, b, Path(models["pose133"]))
    )


def stage_mesh(root: Path, models: dict) -> None:
    from . import mesh

    _around_reference(
        root,
        "mesh",
        lambda im, f, b: mesh.meshes(
            im, f, b, Path(models["mesh_code"]), Path(models["mesh_weights"])
        ),
    )


def stage_full(root: Path, models: dict) -> None:
    from . import corrector as K

    work = root / "_boxes"
    _tables(root, work, body=True)
    for name in ("pose17", "pose133", "mesh"):
        (work / name).mkdir(exist_ok=True)
        for v in videos(root):
            link = work / name / f"{v}.npz"
            if not link.exists():
                link.symlink_to((root / v / f"{name}.npz").resolve())
    nets = _networks(work, "tables_full", models["networks_full"], 13)
    K.correct(
        K.Layout(
            work / "tables_ten",
            work / "tracks",
            Path(models["trees_full"]),
            nets,
            work / "pose17",
            work / "pose133",
            work / "mesh",
            Path(models["thresholds"]),
            Path(models["mhr70"]),
        ),
        work / "full",
    )
    for v in videos(root):
        target = root / v / "boxes.txt"
        if not target.exists():
            target.write_bytes((work / "full" / f"{v}.txt").read_bytes())
    log.info("final boxes written: %d videos", len(videos(root)))


STAGES = dict(
    reads=stage_reads,
    reference=stage_reference,
    pose17=stage_pose17,
    pose133=stage_pose133,
    mesh=stage_mesh,
    full=stage_full,
)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("stage", choices=list(STAGES))
    p.add_argument("--set", required=True, dest="root")
    p.add_argument("--models", required=True)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    t0 = time.time()
    STAGES[a.stage](Path(a.root), json.loads(Path(a.models).read_text()))
    log.info("[%s: %.0f s]", a.stage, time.time() - t0)
