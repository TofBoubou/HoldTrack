"""Locate and verify public model artifacts."""

import hashlib
import json
import shutil
import urllib.request
from pathlib import Path
from typing import TypedDict, cast


class WeightInfo(TypedDict, total=False):
    filename: str
    sha256: str
    url: str | None
    bundled: bool
    architecture: str
    resolution: int
    classes: int
    model_config: dict
    format: str


def manifest(path: str | Path | None = None) -> dict[str, WeightInfo]:
    source = Path(path) if path else Path(__file__).resolve().parent / "resources/weights.json"
    return cast(dict[str, WeightInfo], json.loads(source.read_text())["weights"])


def verify(path: str | Path, digest: str) -> Path:
    source = Path(path)
    with source.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
            raise ValueError(f"Weight checksum differs: {source.name}")
    return source


def prepare_weight(
    name: str, directory: str | Path, *, source: str | Path | None = None, download: bool = False
) -> Path:
    entry = manifest()[name]
    if entry.get("bundled"):
        return verify(Path(__file__).parent / "resources" / entry["filename"], entry["sha256"])
    target = Path(directory) / entry["filename"]
    if target.exists():
        return verify(target, entry["sha256"])
    if source is None and (not download):
        raise FileNotFoundError(
            f"{entry['filename']} is missing: import it with holdtrack download --name {name} --source FILE"
        )
    if source is None and (not entry["url"]):
        raise ValueError("No download URL for this model: give the file with --source")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    try:
        if source is not None:
            verify(source, entry["sha256"])
            with Path(source).open("rb") as src, temporary.open("xb") as dst:
                shutil.copyfileobj(src, dst, 2**20)
        else:
            with (
                urllib.request.urlopen(entry["url"], timeout=60) as src,
                temporary.open("xb") as dst,
            ):
                shutil.copyfileobj(src, dst, 2**20)
        verify(temporary, entry["sha256"])
        temporary.replace(target)
        return target
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
