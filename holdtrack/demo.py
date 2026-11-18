"""Render identity-colored masks and labels."""

import json
from collections.abc import Iterable
from functools import lru_cache
from importlib.resources import files
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from holdtrack.masks import decode

DEFAULT_PALETTE = (
    "0072B2",
    "E69F00",
    "009E73",
    "CC79A7",
    "D55E00",
    "56B4E9",
    "F0E442",
)
FILL_ALPHA = 0.38
OUTLINE_PIXELS = 3
OUTLINE_FACTOR = 0.8


def color(identity: int, palette: list | dict | None = None) -> tuple[int, int, int]:
    value = None
    if isinstance(palette, dict):
        value = palette.get(str(identity))
    elif palette:
        value = palette[(identity - 1) % len(palette)]
    if value is None:
        value = DEFAULT_PALETTE[(identity - 1) % len(DEFAULT_PALETTE)]
    if isinstance(value, str):
        value = value.lstrip("#")
        if len(value) != 6:
            raise ValueError("Palette hex colors must contain six digits")
        return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))
    if len(value) != 3 or any(not 0 <= channel <= 255 for channel in value):
        raise ValueError("Palette colors must contain three byte values")
    return tuple(map(int, value))


@lru_cache(maxsize=1)
def _label_font() -> ImageFont.FreeTypeFont:
    resource = files("holdtrack").joinpath("resources/fonts/lmroman10-bold.otf")
    return ImageFont.truetype(BytesIO(resource.read_bytes()), 22)


def _label(image: np.ndarray, text: str, left: int, top: int, rgb: np.ndarray) -> None:
    font = _label_font()
    bounds = font.getbbox(text)
    padding = 6
    width = bounds[2] - bounds[0] + 2 * padding
    height = bounds[3] - bounds[1] + 2 * padding
    left = max(0, min(left, image.shape[1] - width))
    top = max(0, min(top - height - 4, image.shape[0] - height))
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    fill = tuple(map(int, rgb))
    draw.rectangle(
        (left, top, left + width - 1, top + height - 1),
        fill="white",
        outline=fill,
        width=2,
    )
    draw.text((left + padding - bounds[0], top + padding - bounds[1]), text, font=font, fill=fill)
    image[:] = np.asarray(canvas)


def overlay(
    image: np.ndarray, objects, *, palette: list | dict | None = None, alpha: float = FILL_ALPHA
) -> np.ndarray:
    import cv2

    if not 0 <= alpha <= 1:
        raise ValueError("Mask opacity must lie in [0,1]")
    masks = [decode(item["mask"]) for item in objects]
    if any(mask.shape != image.shape[:2] for mask in masks):
        raise ValueError("Mask and RGB dimensions differ")
    result = image.copy()
    labels = []
    for item, mask in zip(objects, masks):
        identity = item.get("mot_id")
        if identity is None:
            identity = item["object_id"]
        rgb = np.asarray(color(int(identity), palette), np.float32)
        result[mask] = np.rint((1 - alpha) * image[mask] + alpha * rgb).astype(np.uint8)
        if not mask.any():
            continue
        eroded = (
            cv2.erode(
                mask.astype(np.uint8),
                np.ones((2 * OUTLINE_PIXELS + 1, 2 * OUTLINE_PIXELS + 1), np.uint8),
                borderType=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            > 0
        )
        result[mask & ~eroded] = np.rint(rgb * OUTLINE_FACTOR).astype(np.uint8)
        vertical, horizontal = np.where(mask)
        labels.append((str(identity), int(horizontal.min()), int(vertical.min()), rgb))
    for label in labels:
        _label(result, *label)
    return result


def render(
    video: str | Path,
    masks: str | Path,
    output: str | Path,
    *,
    fps: float | None = None,
    palette: str | Path | None = None,
) -> Path:
    import cv2

    source, directory, destination = (Path(video), Path(masks), Path(output))
    if destination.exists():
        raise FileExistsError(destination)
    colors = json.loads(Path(palette).read_text()) if palette else None
    capture = None
    if source.is_dir():
        images = sorted(
            p for p in source.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        rate = fps or 20.0

        def frames() -> Iterable[np.ndarray]:
            for path in images:
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError(f"Cannot decode {path.name}")
                yield cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise ValueError("Cannot open the source video")
        rate = fps or capture.get(cv2.CAP_PROP_FPS)

        def frames() -> Iterable[np.ndarray]:
            while True:
                ok, image = capture.read()
                if not ok:
                    return
                yield cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("A positive frame rate is required")
    writer = None
    try:
        for index, image in enumerate(frames(), 1):
            if writer is None:
                destination.parent.mkdir(parents=True, exist_ok=True)
                height, width = image.shape[:2]
                writer = cv2.VideoWriter(
                    str(destination), cv2.VideoWriter_fourcc(*"mp4v"), rate, (width, height)
                )
                if not writer.isOpened():
                    raise ValueError("An MP4 encoder is unavailable")
            frame_masks = directory / f"{index:08d}.json"
            objects = json.loads(frame_masks.read_text())["objects"] if frame_masks.exists() else []
            image = overlay(image, objects, palette=colors)
            writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if writer is None:
            raise ValueError("No input frames were decoded")
        return destination
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
