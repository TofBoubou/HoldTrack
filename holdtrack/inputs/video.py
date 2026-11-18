"""Enumerate images or decode a user video without changing its image clock."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from holdtrack.cache import file_hash

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class VideoFrames:
    paths: tuple[Path, ...]
    width: int
    height: int
    fps: float
    decoded: bool

    def references(self) -> list[dict]:
        return [dict(path=str(path.absolute()), sha256=file_hash(path)) for path in self.paths]


def _natural_key(path: Path) -> tuple:
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part.casefold())
        for part in re.split(r"(\d+)", path.name)
    ), path.name


def _rate(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Provide a finite, positive frame rate")
    return value


def prepare_video(
    source: str | Path, directory: str | Path, *, fps: float | None = None
) -> VideoFrames:
    """Keep folder pixels intact; decode a video once to lossless, ordered PNGs.

    Folder ordering is natural filename order (frame2 before frame10). All
    images must have the same dimensions. Folder FPS defaults to20; a video
    keeps its declared rate unless the caller supplies FPS. No resampling.
    """
    source = Path(source).expanduser().resolve()
    decoded = not source.is_dir()
    if not decoded:
        paths = tuple(
            sorted(
                (p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
                key=_natural_key,
            )
        )
        rate = _rate(20 if fps is None else fps)
    else:
        if not source.is_file():
            raise FileNotFoundError(source)
        try:
            import cv2
        except ImportError as error:
            raise ImportError("Video decoding requires OpenCV; install holdtrack[sam]") from error
        frames = Path(directory) / "frames"
        frames.mkdir(parents=True, exist_ok=False)
        reader = cv2.VideoCapture(str(source))
        if not reader.isOpened():
            reader.release()
            raise ValueError(f"Cannot decode video: {source}")
        paths_list = []
        try:
            rate = _rate(reader.get(cv2.CAP_PROP_FPS) if fps is None else fps)
            declared_count = reader.get(cv2.CAP_PROP_FRAME_COUNT)
            while True:
                ok, bgr = reader.read()
                if not ok:
                    break
                path = frames / f"{len(paths_list) + 1:08d}.png"
                Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).save(path)
                paths_list.append(path)
            if math.isfinite(declared_count) and declared_count > 0:
                if len(paths_list) != round(declared_count):
                    raise ValueError("Video decoding ended before its declared frame count")
        finally:
            reader.release()
        paths = tuple(paths_list)
    if not paths:
        raise ValueError("The input contains no decodable images")
    dimensions = None
    for path in paths:
        with Image.open(path) as image:
            if dimensions is None:
                dimensions = image.size
            elif image.size != dimensions:
                raise ValueError(f"Frame dimensions changed: {path.name}")
            image.verify()
    width, height = dimensions
    return VideoFrames(paths, width, height, rate, decoded)
