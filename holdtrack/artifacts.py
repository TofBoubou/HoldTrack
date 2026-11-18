"""Read and verify reusable inputs for the pipeline."""

import hashlib
import json
from pathlib import Path


def read_json(path: str | Path):
    return json.loads(Path(path).read_text())


def checked(ref: dict[str, str], base: str | Path) -> Path:
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
        raise ValueError("Expected path/sha256 reference")
    path = Path(ref["path"])
    if not path.is_absolute():
        path = Path(base) / path
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != ref["sha256"]:
        raise ValueError("Input SHA mismatch: " + str(path))
    return path


def write_new(path: str | Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
