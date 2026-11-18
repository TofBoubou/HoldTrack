"""Check the `track` command on synthetic inputs."""

import json
from dataclasses import replace

import numpy as np
from PIL import Image

from holdtrack import pipeline, render, track
from holdtrack.config import preset


def test_track_runs_the_light_system_and_renders(tmp_path, monkeypatch):
    frames = tmp_path / "clip"
    frames.mkdir()
    for index in range(1, 61):
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(frames / f"{index:08d}.png")
    seen = {}

    def run(source, out, **options):
        seen["run"] = (source, options)
        out.write_text("1,1,0,0,4,4,1,-1,-1,-1\n")
        options["masks"].mkdir()
        return out

    def draw(source, tracks, output, **options):
        seen["render"] = (source, tracks, options)
        output.write_bytes(b"video")
        return dict(masks=1, frames=0)

    monkeypatch.setattr(pipeline, "run", run)
    monkeypatch.setattr(render, "render", draw)
    written = track.track(frames, tmp_path / "out", fps=20.0)
    source, options = seen["run"]
    assert source == frames and options["config"] == preset("light")
    assert options["masks"] == tmp_path / "out" / "masks"
    assert seen["render"][2]["fps"] == 20.0
    assert written["tracks"].read_text().startswith("1,1,")
    assert written["video"].name == "tracks.mp4" and written["video"].exists()


def test_track_full_needs_its_models(tmp_path, monkeypatch):
    frames = tmp_path / "clip"
    frames.mkdir()
    for index in range(1, 61):
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(frames / f"{index:08d}.png")
    seen = {}

    def run(source, out, **options):
        seen["config"] = options["config"]
        return out

    monkeypatch.setattr(pipeline, "run", run)
    try:
        track.track(frames, tmp_path / "out", full=True, weights_dir=tmp_path / "weights")
    except FileNotFoundError as error:
        assert "models.json" in str(error) and not seen
    else:
        raise AssertionError("the full system cannot run without its models")
    weights = tmp_path / "weights"
    for name in track.FULL_FILES.values():
        for file in name if isinstance(name, list) else [name]:
            (weights / file).parent.mkdir(parents=True, exist_ok=True)
            (weights / file).write_text("x")
    try:
        track.track(frames, tmp_path / "out", full=True, weights_dir=weights)
    except FileNotFoundError as error:
        assert "--mesh-source" in str(error) and not seen
    else:
        raise AssertionError("the mesh stage needs the source of SAM 3D Body")
    mesh = tmp_path / "sam-3d-body"
    mesh.mkdir()
    monkeypatch.setattr(track, "_finish_full", lambda **a: a["out"])
    monkeypatch.setattr(render, "render", lambda *a, **k: a[2].write_bytes(b"video"))
    track.track(frames, tmp_path / "out", full=True, weights_dir=weights, mesh_source=mesh)
    assert seen["config"] == preset("full") and seen["config"].seed_detector == "rf_large"
    written = json.loads((weights / "models.json").read_text())
    assert (
        written["mesh_code"] == str(mesh) and written["rf_2xl_specification"]["resolution"] == 1120
    )
    assert (
        written["trees_full"].endswith("corrector/trees.pkl") and len(written["networks_full"]) == 2
    )


def test_track_keeps_every_track_of_a_clip_shorter_than_the_shortest_track(tmp_path, monkeypatch):
    seen = {}

    def run(source, out, **options):
        seen.setdefault("steps", []).append(options["config"].steps)
        out.write_text("1,1,0,0,4,4,1,-1,-1,-1\n")
        options["masks"].mkdir()
        return out

    monkeypatch.setattr(pipeline, "run", run)
    monkeypatch.setattr(render, "render", lambda *a, **k: a[2].write_bytes(b"video"))
    for name, count, fps in (("short", 2, 20.0), ("exact", 55, 20.0), ("fast", 60, 30.0)):
        frames = tmp_path / name
        frames.mkdir()
        for index in range(1, count + 1):
            Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(frames / f"{index:08d}.png")
        track.track(frames, tmp_path / (name + "_out"), fps=fps)
    assert [steps.min_track_length for steps in seen["steps"]] == [0, 0, 0]
    frames = tmp_path / "long"
    frames.mkdir()
    for index in range(1, 91):
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(frames / f"{index:08d}.png")
    track.track(frames, tmp_path / "long_out", fps=30.0)
    assert seen["steps"][-1].min_track_length == 82
    assert replace(seen["steps"][-1], min_track_length=55) == preset("light").steps
