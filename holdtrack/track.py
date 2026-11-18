"""Track the persons of a video and render them, one command.

    holdtrack track clip.mp4

writes, in ``clip_holdtrack/``: ``tracks.txt`` (MOT rows: frame, identity, left, top, width,
height, ...), ``masks/`` (one file per frame) and ``tracks.mp4``, the video with one color per
identity. The default is the light system: one detector, one tracker pass. ``--full`` runs the
complete system without faces: five detector views, five tracker passes, the corrected boxes
and the identity chain. ``--faces`` adds the face steps to ``--full``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

# Tracks shorter than this are dropped: 55 frames at 20 frames per second. A clip that is itself
# shorter keeps every track.
MIN_TRACK_SECONDS = 2.75


def track(
    video: str | Path,
    out: str | Path | None = None,
    *,
    full: bool = False,
    faces: str | Path | None = None,
    weights_dir: str | Path = "weights",
    device: str = "cuda",
    fps: float | None = None,
    video_format: str = "mp4",
    detector_python: str | Path | None = None,
    dfine_source: str | Path | None = None,
    models: str | Path | None = None,
    mesh_source: str | Path | None = None,
    boxes_python: str | Path | None = None,
    pose_python: str | Path | None = None,
    mesh_python: str | Path | None = None,
    faces_python: str | Path | None = None,
) -> dict:
    """Returns the paths written: tracks, masks, video."""
    from holdtrack.config import preset
    from holdtrack.inputs.video import prepare_video
    from holdtrack.pipeline import run
    from holdtrack.render import render

    if video_format not in {"mp4", "webp", "gif"}:
        raise ValueError("Choose mp4, webp or gif")
    if faces and not full:
        raise ValueError("The face steps run on the complete system: add --full")
    if full:
        models = _models_file(models, Path(weights_dir), mesh_source)
    source = Path(video).expanduser().resolve()
    target = Path(out) if out else source.with_name(source.stem + "_holdtrack")
    target.mkdir(parents=True, exist_ok=True)
    work = target / "work"
    decoded = work / "video"
    if source.is_dir():
        sequence = prepare_video(source, decoded, fps=fps)
    elif (decoded / "fps.txt").exists():
        rate = fps or float((decoded / "fps.txt").read_text())
        sequence = prepare_video(decoded / "frames", decoded, fps=rate)
    elif decoded.exists():
        raise FileExistsError(f"{decoded} holds an unfinished decoding; remove it or choose --out")
    else:
        sequence = prepare_video(source, decoded, fps=fps)
        (decoded / "fps.txt").write_text(repr(sequence.fps) + "\n")
    frames = sequence.paths[0].parent
    configuration = preset("full" if full else "light")
    shortest = round(MIN_TRACK_SECONDS * sequence.fps)
    if len(sequence.paths) <= shortest:
        shortest = 0
    if shortest != configuration.steps.min_track_length:
        steps = replace(configuration.steps, min_track_length=shortest)
        configuration = replace(configuration, steps=steps)
    tracks = run(
        frames,
        work / "base.txt" if full else target / "tracks.txt",
        config=configuration,
        work_dir=work / "base",
        weights_dir=weights_dir,
        detector_python=detector_python,
        dfine_source=dfine_source,
        device=device,
        masks=target / "masks",
    )
    if full:
        arguments = dict(
            frames=str(frames),
            base=str(tracks),
            work=str(work),
            out=str(target / "tracks.txt"),
            fps=sequence.fps,
            models=str(models),
            device=device,
            faces=str(Path(faces).expanduser()) if faces else None,
            pythons={
                name: str(python)
                for name, python in (
                    ("pose133", pose_python),
                    ("mesh", mesh_python),
                    ("faces", faces_python),
                )
                if python
            },
        )
        tracks = target / "tracks.txt"
        if boxes_python:
            import json
            import subprocess

            command = [str(boxes_python), "-m", "holdtrack.track", json.dumps(arguments)]
            subprocess.run(command, check=True, env=_environment())
        else:
            _finish_full(**arguments)
    rendered = target / ("tracks." + video_format)
    counts = None
    if not rendered.exists():
        counts = render(frames, tracks, rendered, masks=target / "masks", fps=sequence.fps)
    return dict(
        tracks=tracks, masks=target / "masks", video=rendered, fps=sequence.fps, drawn=counts
    )


BOX_STAGES = ("reads", "reference", "pose17", "pose133", "mesh")
# Files of the complete system in the weights folder, when no models.json is given.
FULL_FILES = dict(
    rf_2xl="rf_2xl.pth",
    networks_reference=["corrector/reference_network_0.pt", "corrector/reference_network_1.pt"],
    trees_reference="corrector/reference_trees.pkl",
    networks_full=["corrector/network_0.pt", "corrector/network_1.pt"],
    trees_full="corrector/trees.pkl",
    thresholds="corrector/pose_thresholds.json",
    pose133="pose133.onnx",
    mesh_weights="sam3d_body",
    appearance="identity/appearance.pt",
)


def _models_file(models, weights: Path, mesh_source) -> Path:
    """The models of the complete system: a models.json, or the files of the weights folder under
    their default names, with the source of SAM 3D Body."""
    import json

    from holdtrack.weights import manifest

    for given in (models, weights / "models.json"):
        if given and Path(given).is_file():
            return Path(given)
    if models:
        raise FileNotFoundError(f"{models} is missing; see docs/FULL.md")
    found = {
        key: [str((weights / n).resolve()) for n in name]
        if isinstance(name, list)
        else str((weights / name).resolve())
        for key, name in FULL_FILES.items()
    }
    missing = [
        n
        for name in FULL_FILES.values()
        for n in (name if isinstance(name, list) else [name])
        if not (weights / n).exists()
    ]
    if missing or not mesh_source:
        need = ", ".join(missing) if missing else "--mesh-source (the source of SAM 3D Body)"
        raise FileNotFoundError(
            f"The full system needs a models.json or, in {weights}: {need}; see docs/FULL.md"
        )
    source = Path(mesh_source).resolve()
    found.update(
        rf_2xl_specification=manifest()["rf_2xl"],
        pose17="usyd-community/vitpose-plus-huge",
        mesh_code=str(source),
        mhr70=str(source / "sam_3d_body" / "metadata" / "mhr70.py"),
    )
    written = weights / "models.json"
    written.write_text(json.dumps(found, indent=1) + "\n")
    return written


PASS_NAMES = (("BASE", "native"), ("EXCL", "exclusive"), ("PROT", "guarded"))


def _link(source: str | Path, target: Path) -> None:
    if not target.exists():
        target.symlink_to(Path(source).resolve())


def _set_folder(frames: Path, base: Path, work: Path) -> Path:
    """The folder read by the box stages and the identity chain, from the work of the base."""
    import json

    stages = json.loads((work / "base" / "stages.json").read_text())
    passes = work / "base" / "passes"
    folder = work / "set" / "video"
    for name in ("masks", "reads", "passes"):
        (folder / name).mkdir(parents=True, exist_ok=True)
    _link(frames, folder / "frames")
    _link(base, folder / "tracks.txt")
    _link(stages["fusion"], folder / "fusion.txt")
    _link(stages["associations"], folder / "bytetrack.txt")
    _link(stages["vote"], folder / "passes" / "vote.txt")
    for role, name in PASS_NAMES:
        _link(passes / role / "sam.txt", folder / "masks" / (name + ".txt"))
        _link(passes / role / "mot.txt", folder / "passes" / (name + ".txt"))
    _link(passes / "reverse_BASE" / "mot.txt", folder / "passes" / "backward.txt")
    _link(passes / "mirror_EXCL" / "mot.txt", folder / "passes" / "mirrored.txt")
    _link(stages["detections"], folder / "reads" / "rf.npz")
    for view, name in (("yolo", "yolo"), ("mirror", "yolo_mirrored"), ("dfine", "dfine")):
        _link(stages["views"][view], folder / "reads" / (name + ".npz"))
    return folder


def _environment() -> dict:
    """The stages run by another interpreter see this package and nothing else of ours."""
    import os

    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
    return environment


def _finish_full(
    frames: str | Path,
    base: str | Path,
    work: str | Path,
    out: str | Path,
    fps: float,
    models: str | Path,
    device: str,
    pythons: dict,
    faces: str | None = None,
) -> Path:
    """Corrected boxes, identity chain and final tracks, on top of the base tracks."""
    import json
    import subprocess
    import sys

    from holdtrack import identity
    from holdtrack.boxes import full as finish

    frames, base, work, out, models = (Path(p) for p in (frames, base, work, out, models))
    if out.exists():
        return out
    folder = _set_folder(frames, base, work)
    root = folder.parent
    environment = _environment()

    def stage(name: str) -> None:
        marker = root / ("_done_" + name)
        if marker.exists():
            return
        python = str(pythons.get(name) or sys.executable)
        command = [python, "-m", "holdtrack.boxes.pipeline", name]
        subprocess.run(
            [*command, "--set", str(root), "--models", str(models)], check=True, env=environment
        )
        marker.write_text("done\n")

    for name in BOX_STAGES:
        stage(name)
    finish.identity_inputs(root)
    stages = work / "appearance"
    if not (stages / "features" / "video.npz").exists():
        identity.chain.make_crops(str(folder), fps)
        network = json.loads(models.read_text())["appearance"]
        identity.appearance.run(str(folder), str(stages), network, device, fps)
    if not (folder / "identities.npy").exists():
        identity.run(str(folder), str(stages), fps=fps)
    stage("full")
    face_steps = None
    if faces:
        from holdtrack.faces import chain as face_chain

        if pythons.get("faces"):
            command = [str(pythons["faces"]), "-m", "holdtrack.faces.chain"]
            subprocess.run(
                [*command, str(folder), faces, device, repr(fps)], check=True, env=environment
            )
        else:
            face_chain.prepare_tables(str(folder), faces, device, fps)
        face_steps = face_chain.run_folder
    finish.final_tracks(root, check=identity.check.relays, fps=fps, faces=face_steps)
    temporary = out.with_name(out.name + ".partial")
    temporary.write_bytes((folder / "final.txt").read_bytes())
    temporary.replace(out)
    return out


if __name__ == "__main__":
    import json
    import sys

    _finish_full(**json.loads(sys.argv[1]))
