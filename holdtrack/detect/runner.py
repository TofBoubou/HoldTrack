"""Isolate detector runtimes and cache their complete outputs."""

import json
import subprocess
from pathlib import Path

from holdtrack.cache import StageCache, atomic_json, file_hash
from holdtrack.weights import manifest


def detect_sequence(
    name: str,
    images: list[Path],
    checkpoint: Path,
    cache: StageCache,
    *,
    python: Path,
    source: Path | None = None,
    device: str = "cuda",
) -> Path:
    """Use the current interpreter by default or an explicitly selected detector environment."""
    configuration = manifest()["yolo" if name == "mirror" else name]
    inputs = dict(
        detector=name,
        implementation={
            path.name: file_hash(path) for path in sorted(Path(__file__).parent.glob("*.py"))
        },
        images=[dict(path=str(path), sha256=file_hash(path)) for path in images],
        checkpoint_sha256=file_hash(checkpoint),
        configuration=configuration,
        device=device,
    )

    def produce(output: Path) -> None:
        request = output.parent / "detector.json"
        atomic_json(
            request,
            dict(
                inputs,
                checkpoint=str(checkpoint),
                output=str(output),
                source=str(source) if source else None,
            ),
        )
        subprocess.run(
            [
                str(python),
                "-I",
                "-c",
                "import runpy\nimport sys\nsys.path.insert(0, sys.argv[1])\nsys.argv = ['holdtrack.detect', sys.argv[2]]\nrunpy.run_module('holdtrack.detect', run_name='__main__')\n",
                str(Path(__file__).resolve().parents[2]),
                str(request),
            ],
            check=True,
        )

    return cache.stage("detect_" + name, inputs, "detections.npz", produce)


def worker(request: str | Path) -> None:
    """Execute one native detector entry without importing tracking or evaluation code."""
    payload = json.loads(Path(request).read_text())
    images = [Path(reference["path"]) for reference in payload["images"]]
    for path, reference in zip(images, payload["images"]):
        if file_hash(path) != reference["sha256"]:
            raise ValueError("A detector input image changed")
    checkpoint = Path(payload["checkpoint"])
    if file_hash(checkpoint) != payload["checkpoint_sha256"]:
        raise ValueError("The detector weights changed")
    name = payload["detector"]
    if name.startswith("rf_"):
        from holdtrack.detect.rfdetr import predict

        result = predict(images, checkpoint, payload["configuration"], payload["device"])
    elif name in {"yolo", "mirror"}:
        from holdtrack.detect.yolo import predict

        result = predict(images, checkpoint, mirrored=name == "mirror", device=payload["device"])
    elif name == "dfine":
        from holdtrack.detect.dfine import predict

        if not payload["source"]:
            raise ValueError(
                "D-FINE requires --dfine-source pointing to its official source checkout"
            )
        result = predict(images, checkpoint, Path(payload["source"]), payload["device"])
    else:
        raise ValueError("Unknown detector")
    result.save(payload["output"])
