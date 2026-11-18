"""Run, render, prepare models and evaluate HoldTrack."""

import argparse
import json
from pathlib import Path

from holdtrack.config import DEFAULT_PRESET


def _run(options: argparse.Namespace) -> None:
    from holdtrack.pipeline import run

    run(
        options.frames,
        options.out,
        config=options.config,
        work_dir=options.work_dir,
        weights_dir=options.weights_dir,
        cached_inputs=options.cached_inputs,
        detector_python=options.detector_python,
        dfine_source=options.dfine_source,
        device=options.device,
    )


def _track(options: argparse.Namespace) -> None:
    from holdtrack.track import track

    written = track(
        options.video,
        options.out,
        full=options.full,
        faces=options.faces,
        weights_dir=options.weights_dir,
        device=options.device,
        fps=options.fps,
        video_format=options.format,
        detector_python=options.detector_python,
        dfine_source=options.dfine_source,
        models=options.models,
        mesh_source=options.mesh_source,
        boxes_python=options.boxes_python,
        pose_python=options.pose_python,
        mesh_python=options.mesh_python,
        faces_python=options.faces_python,
    )
    print(written["tracks"])
    print(written["video"])


def _demo(options: argparse.Namespace) -> None:
    from holdtrack.demo import render

    render(options.video, options.masks, options.out, fps=options.fps, palette=options.palette)


def _download(options: argparse.Namespace) -> None:
    from holdtrack.weights import prepare_weight

    print(
        prepare_weight(
            options.name, options.directory, source=options.source, download=options.source is None
        )
    )


def _evaluate(options: argparse.Namespace) -> None:
    from holdtrack.evaluation import evaluate_tracks

    result = evaluate_tracks(options.tracks, options.gt, num_frames=options.num_frames)
    _write_result(
        dict(units="percent", metrics={name: 100 * value for name, value in result.items()}),
        options.out,
    )


def _write_result(value: dict, output: str | None) -> None:
    text = json.dumps(value, indent=2) + "\n"
    if output:
        Path(output).write_text(text)
    else:
        print(text, end="")


def main(argv: list[str] | None = None) -> None:
    """Expose one installed command for each supported user task."""
    parser = argparse.ArgumentParser(prog="holdtrack", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    simple = commands.add_parser("track", help="Track the persons of a video and render them")
    simple.add_argument("video", help="video file or folder of frames")
    simple.add_argument("--out", help="output folder (default: <video>_holdtrack)")
    simple.add_argument("--full", action="store_true", help="the complete system, without faces")
    simple.add_argument(
        "--faces",
        nargs="?",
        const="~/.cache/holdtrack/faces",
        help="add the face steps to --full; folder of the face models (fetch_face_models.py)",
    )
    simple.add_argument("--weights-dir", default="weights")
    simple.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    simple.add_argument("--fps", type=float)
    simple.add_argument("--format", choices=["mp4", "webp", "gif"], default="mp4")
    simple.add_argument("--detector-python")
    simple.add_argument("--dfine-source")
    simple.add_argument("--models", help="models of the full system (default: weights/models.json)")
    simple.add_argument("--mesh-source", help="source of SAM 3D Body, for --full")
    simple.add_argument("--boxes-python", help="interpreter of the box and identity stages")
    simple.add_argument("--pose-python", help="interpreter of the 133-keypoint stage")
    simple.add_argument("--mesh-python", help="interpreter of the body-mesh stage")
    simple.add_argument("--faces-python", help="interpreter of the face tables")
    simple.set_defaults(handler=_track)
    tracking = commands.add_parser(
        "run",
        help="Run the base system: detectors, tracker passes, vote, fusion, track-level steps",
    )
    inputs = tracking.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--frames")
    inputs.add_argument("--cached-inputs")
    tracking.add_argument("--out", required=True)
    tracking.add_argument(
        "--config", choices=["full", "large", "2xl", "light"], default=DEFAULT_PRESET
    )
    tracking.add_argument("--work-dir")
    tracking.add_argument("--weights-dir", default="weights")
    tracking.add_argument("--detector-python")
    tracking.add_argument("--dfine-source")
    tracking.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    tracking.set_defaults(handler=_run)
    demo = commands.add_parser("demo", help="Overlay masks and identities on video")
    demo.add_argument("--video", required=True)
    demo.add_argument("--masks", required=True)
    demo.add_argument("--out", required=True)
    demo.add_argument("--palette")
    demo.add_argument("--fps", type=float)
    demo.set_defaults(handler=_demo)
    download = commands.add_parser(
        "download", help="Verify a model file and import it into the weights folder"
    )
    download.add_argument("--name", default="sam3")
    download.add_argument("--directory", default="weights")
    download.add_argument("--source", help="Verify and import an existing local file")
    download.set_defaults(handler=_download)
    evaluate = commands.add_parser(
        "evaluate", help="Evaluate MOT tracks against DanceTrack ground truth"
    )
    evaluate.add_argument("--tracks", required=True)
    evaluate.add_argument("--gt", required=True)
    evaluate.add_argument("--num-frames", type=int)
    evaluate.add_argument("--out")
    evaluate.set_defaults(handler=_evaluate)
    options = parser.parse_args(argv)
    options.handler(options)
