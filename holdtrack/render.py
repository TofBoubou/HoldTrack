"""Render tracks as identity-colored masks, alone or side by side with other tracks.

Inputs are a video file or a folder of frames, a MOT file of tracks (frame, identity,
left, top, width, height, ...), and optionally a folder of per-frame mask files in the
format of ``holdtrack.masks`` (``<frame:08d>.json`` with ``objects``, each holding a
column-major run-length ``mask``). Each mask takes the color of the track whose box it
overlaps best in its frame; a track without a mask in a frame is drawn as a thin frame
of its color, and a mask without a track is not drawn. Nothing else is inferred.

The output format follows the extension: ``.webp`` and ``.gif`` (animated) or ``.mp4``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PALETTE = (
    (230, 159, 0),
    (86, 180, 233),
    (240, 228, 66),
    (0, 90, 190),
    (70, 70, 70),
    (140, 86, 45),
    (120, 90, 220),
    (0, 158, 115),
    (204, 121, 167),
    (213, 94, 0),
    (160, 170, 40),
    (0, 150, 170),
    (165, 42, 42),
    (250, 150, 190),
    (100, 200, 120),
    (40, 40, 120),
    (200, 120, 60),
    (150, 150, 150),
    (90, 30, 110),
    (190, 210, 90),
)
FILL_ALPHA = 0.55
MIN_OVERLAP = 0.3


def decode(rle: dict) -> np.ndarray:
    height, width = rle["size"]
    counts = np.asarray(rle["counts"], np.int64)
    if (counts < 0).any() or int(counts.sum()) != height * width:
        raise ValueError("Invalid run lengths")
    values = np.zeros(len(counts), bool)
    values[1::2] = True
    return np.repeat(values, counts).reshape((height, width), order="F")


def read_tracks(path: str | Path) -> dict[int, list[tuple[int, float, float, float, float]]]:
    """MOT rows by frame, as (identity, left, top, right, bottom)."""
    rows: dict[int, list] = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        f = line.split(",")
        left, top, width, height = map(float, f[2:6])
        rows[int(float(f[0]))].append((int(float(f[1])), left, top, left + width, top + height))
    return rows


def read_masks(directory: str | Path, frame: int) -> list[np.ndarray]:
    path = Path(directory) / f"{frame:08d}.json"
    if not path.exists():
        return []
    return [decode(item["mask"]) for item in json.loads(path.read_text())["objects"]]


def _box(mask: np.ndarray):
    vertical, horizontal = np.where(mask)
    if len(horizontal) == 0:
        return None
    return (horizontal.min(), vertical.min(), horizontal.max() + 1, vertical.max() + 1)


def assign(masks: list[np.ndarray], rows: list, minimum: float = MIN_OVERLAP) -> dict[int, int]:
    """One mask per track row: the pairs are taken by decreasing overlap.

    The overlap of a mask and a row is the share of the mask pixels inside the box of
    the row, times the IoU of the box of the mask with the box of the row: a box that
    extends behind an occluder still takes the mask of its person, and a large box does
    not take the mask of a neighbor it merely contains.
    """
    pairs = []
    for m, mask in enumerate(masks):
        b = _box(mask)
        if b is None:
            continue
        area = int(mask.sum())
        for r, (_, x1, y1, x2, y2) in enumerate(rows):
            ix1, iy1 = max(b[0], int(np.floor(x1))), max(b[1], int(np.floor(y1)))
            ix2, iy2 = min(b[2], int(np.ceil(x2))), min(b[3], int(np.ceil(y2)))
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inside = int(mask[iy1:iy2, ix1:ix2].sum()) / area
            inter = (ix2 - ix1) * (iy2 - iy1)
            union = (b[2] - b[0]) * (b[3] - b[1]) + (x2 - x1) * (y2 - y1) - inter
            score = inside * inter / union
            if score >= minimum:
                pairs.append((score, m, r))
    taken_masks: set[int] = set()
    out: dict[int, int] = {}
    for _, m, r in sorted(pairs, reverse=True):
        if m in taken_masks or r in out:
            continue
        taken_masks.add(m)
        out[r] = m
    return out


def colors(track_sets: list[dict], fixed: dict[int, tuple[int, int, int]] | None = None) -> dict:
    """One color per identity, shared by all panels, in order of first appearance."""
    fixed = dict(fixed or {})
    first: dict[int, int] = {}
    for tracks in track_sets:
        for frame in sorted(tracks):
            for row in tracks[frame]:
                first.setdefault(row[0], frame)
    free = [c for c in PALETTE if c not in set(fixed.values())]
    out = dict(fixed)
    for n, identity in enumerate(
        i for i in sorted(first, key=lambda i: (first[i], i)) if i not in out
    ):
        out[identity] = free[n % len(free)]
    return out


def frames(source: str | Path) -> Iterator[tuple[int, np.ndarray]]:
    """Frames numbered from 1, as RGB arrays."""
    import cv2

    source = Path(source)
    if source.is_dir():
        paths = sorted(p for p in source.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        for index, path in enumerate(paths, 1):
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Cannot decode {path.name}")
            yield index, cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError("Cannot open the source video")
    try:
        index = 0
        while True:
            ok, image = capture.read()
            if not ok:
                return
            index += 1
            yield index, cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()


def paint(
    image: np.ndarray, rows: list, masks: list[np.ndarray], palette: dict, counts: list
) -> np.ndarray:
    import cv2

    out = image.copy()
    matched = assign(masks, rows) if masks else {}
    for r in sorted(matched, key=lambda r: -int(masks[matched[r]].sum())):  # small masks on top
        mask = masks[matched[r]]
        rgb = np.asarray(palette[rows[r][0]], np.float32)
        out[mask] = np.rint((1 - FILL_ALPHA) * out[mask] + FILL_ALPHA * rgb).astype(np.uint8)
        edges, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, edges, -1, tuple(int(0.75 * c) for c in rgb), 2, lineType=cv2.LINE_AA)
    for r, (identity, x1, y1, x2, y2) in enumerate(rows):
        if r in matched:
            continue
        thick = 2 if masks else 4  # without masks the frames carry the identities
        cv2.rectangle(
            out,
            (round(x1), round(y1)),
            (round(x2), round(y2)),
            palette[identity],
            thick,
            cv2.LINE_AA,
        )
    counts[0] += len(matched)
    counts[1] += len(rows) - len(matched)
    return out


def render(
    source: str | Path,
    tracks: str | Path,
    output: str | Path,
    *,
    masks: str | Path | None = None,
    compare: str | Path | None = None,
    titles: tuple[str, str] | None = None,
    start: int = 1,
    end: int | None = None,
    step: int = 1,
    width: int = 960,
    crop: bool = False,
    fps: float = 20.0,
    fixed_colors: dict[int, tuple[int, int, int]] | None = None,
) -> dict:
    """Write the animation and return the counts of rows drawn with a mask and as a frame.

    ``compare`` is a second MOT file drawn on the left, with the same masks and the same
    color for the same identity. ``crop`` keeps the smallest fixed rectangle, with a
    margin, that holds every box of every panel over the whole range.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    panels = [read_tracks(p) for p in ([compare] if compare else []) + [tracks]]
    palette = colors(panels, fixed_colors)
    region = None
    if crop:
        boxes = np.array(
            [
                row[1:]
                for t in panels
                for f, rows in t.items()
                if f >= start and (end is None or f <= end)
                for row in rows
            ]
        )
        if len(boxes):
            w, h = boxes[:, 2].max() - boxes[:, 0].min(), boxes[:, 3].max() - boxes[:, 1].min()
            region = (
                boxes[:, 0].min() - 0.06 * w,
                boxes[:, 1].min() - 0.10 * h,
                boxes[:, 2].max() + 0.06 * w,
                boxes[:, 3].max() + 0.04 * h,
            )
    gap = 8
    panel_width = (width - gap * (len(panels) - 1)) // len(panels)
    font = ImageFont.load_default(size=15) if titles else None
    counts = [[0, 0] for _ in panels]
    images = []
    for index, image in frames(source):
        if index < start or (index - start) % step:
            continue
        if end is not None and index > end:
            break
        frame_masks = read_masks(masks, index) if masks else []
        h_img, w_img = image.shape[:2]
        x1, y1, x2, y2 = (
            (0, 0, w_img, h_img)
            if region is None
            else (
                int(max(0, region[0])),
                int(max(0, region[1])),
                int(min(w_img, region[2])),
                int(min(h_img, region[3])),
            )
        )
        panel_height = round(panel_width * (y2 - y1) / (x2 - x1))
        canvas = Image.new(
            "RGB", (panel_width * len(panels) + gap * (len(panels) - 1), panel_height), "white"
        )
        for n, panel in enumerate(panels):
            drawn = paint(image, panel.get(index, []), frame_masks, palette, counts[n])
            tile = Image.fromarray(drawn[y1:y2, x1:x2]).resize(
                (panel_width, panel_height), Image.LANCZOS
            )
            if titles:
                draw = ImageDraw.Draw(tile)
                draw.text(
                    (8, 6), titles[n], font=font, fill="white", stroke_width=2, stroke_fill="black"
                )
            canvas.paste(tile, (n * (panel_width + gap), 0))
        images.append(canvas)
    if not images:
        raise ValueError("No frame in the requested range")
    rate = fps / step
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix == ".webp":
        images[0].save(
            output,
            save_all=True,
            append_images=images[1:],
            duration=round(1000 / rate),
            loop=0,
            quality=72,
            method=6,
        )
    elif suffix == ".gif":
        images[0].save(
            output,
            save_all=True,
            append_images=images[1:],
            duration=round(1000 / rate),
            loop=0,
            optimize=True,
        )
    elif suffix == ".mp4":
        import cv2

        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), rate, images[0].size)
        if not writer.isOpened():
            raise ValueError("An MP4 encoder is unavailable")
        for canvas in images:
            writer.write(cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR))
        writer.release()
    else:
        raise ValueError("The output must end in .webp, .gif or .mp4")
    return {
        "frames": len(images),
        "size": list(images[0].size),
        "panels": [{"with_mask": a, "as_frame": b} for a, b in counts],
    }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--video", required=True, help="video file or folder of frames")
    p.add_argument("--tracks", required=True, help="MOT file")
    p.add_argument("--out", required=True, help=".webp, .gif or .mp4")
    p.add_argument("--masks", help="folder of per-frame mask files")
    p.add_argument("--compare", help="second MOT file, drawn on the left")
    p.add_argument("--titles", nargs=2)
    p.add_argument("--start", type=int, default=1)
    p.add_argument("--end", type=int)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--crop", action="store_true")
    p.add_argument("--fps", type=float, default=20.0)
    a = p.parse_args()
    print(
        json.dumps(
            render(
                a.video,
                a.tracks,
                a.out,
                masks=a.masks,
                compare=a.compare,
                titles=tuple(a.titles) if a.titles else None,
                start=a.start,
                end=a.end,
                step=a.step,
                width=a.width,
                crop=a.crop,
                fps=a.fps,
            )
        )
    )
