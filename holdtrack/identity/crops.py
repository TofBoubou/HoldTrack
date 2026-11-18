"""Crops of the boxes: 256 x 128 JPEG, the box widened by 8%, aspect kept, grey around."""

import io
import os

import numpy as np
from PIL import Image

from .geometry import xyxy
from .segments import clean_segments

HEIGHT = 256
WIDTH = 128
QUALITY = 90
GREY = (124, 116, 104)
CROPS_PER_S = 5.0


_frames = {}


def _frame(folder, f):
    """Path of frame f (from 1): the images of frames/, in the order of their names."""
    if folder not in _frames:
        d = os.path.join(folder, "frames")
        _frames[folder] = sorted(
            os.path.join(d, n)
            for n in os.listdir(d)
            if os.path.splitext(n)[1].lower() in (".jpg", ".jpeg", ".png")
        )
    return _frames[folder][int(f) - 1]


def _crop(image, box, downscale):
    x0, y0, x1, y1 = box
    w = x1 - x0
    h = y1 - y0
    if w < 8 or h < 16:
        return None
    x0 -= 0.08 * w
    x1 += 0.08 * w
    y0 -= 0.08 * h
    y1 += 0.08 * h
    w = x1 - x0
    h = y1 - y0
    sc = min(WIDTH / w, HEIGHT / h)
    nw, nh = max(1, int(round(w * sc))), max(1, int(round(h * sc)))
    c = image.crop((int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1)))).resize(
        (nw, nh), downscale if sc < 1 else Image.BILINEAR
    )
    t = Image.new("RGB", (WIDTH, HEIGHT), GREY)
    t.paste(c, ((WIDTH - nw) // 2, (HEIGHT - nh) // 2))
    b = io.BytesIO()
    t.save(b, "JPEG", quality=QUALITY)
    return np.frombuffer(b.getvalue(), np.uint8)


def _pack(jpegs, meta):
    o = np.empty(len(jpegs), object)
    for k, x in enumerate(jpegs):
        o[k] = x
    return o, np.array(meta, np.float64).reshape(len(meta), 10)


def track_crops(video, downscale=Image.BICUBIC):
    """Crops of the boxes of tracks.txt, five frames per second. Returns the JPEGs and one row per crop: frame,
    identity, x, y, width, height, overlap, clean segment (IoU 0.5), clean segment (IoU 0.3), row of the box.
    downscale: filter used when a box is larger than the crop."""
    p = video.rows()
    p = np.concatenate([p, np.ones((len(p), 1)), -np.ones((len(p), 3))], 1)
    box = xyxy(p)
    pf = p[:, 0].astype(int)
    pid = p[:, 1].astype(int)
    s5, _, overlap = clean_segments(p, 0.5, video.fps)
    s3, _, _ = clean_segments(p, 0.3, video.fps)
    step = max(1, int(round(video.fps / CROPS_PER_S)))
    jpegs = []
    meta = []
    for f in np.unique(pf):
        if f % step != 1 % step:
            continue
        image = Image.open(_frame(video.folder, f)).convert("RGB")
        for i in np.where(pf == f)[0]:
            j = _crop(image, box[i], downscale)
            if j is not None:
                jpegs.append(j)
                meta.append(
                    (
                        int(f),
                        int(pid[i]),
                        *[float(z) for z in p[i, 2:6]],
                        float(overlap[i]),
                        int(s5[i]),
                        int(s3[i]),
                        int(i),
                    )
                )
    return _pack(jpegs, meta)


def short_track_crops(video, short_tracks, downscale=Image.BICUBIC):
    """Crops of every box of the short tracks (frame, track, x1, y1, x2, y2). Column 9 is the row in short_tracks."""
    c = short_rows(short_tracks)
    box = xyxy(c)
    cf = c[:, 0].astype(int)
    jpegs = []
    meta = []
    for f in np.unique(cf):
        image = Image.open(_frame(video.folder, f)).convert("RGB")
        for i in np.where(cf == f)[0]:
            j = _crop(image, box[i], downscale)
            if j is not None:
                jpegs.append(j)
                meta.append(
                    (int(f), int(c[i, 1]), *[float(z) for z in c[i, 2:6]], 0.0, 0, 0, int(i))
                )
    return _pack(jpegs, meta)


def short_rows(short_tracks):
    """frame, track, x, y, width, height from frame, track, x1, y1, x2, y2."""
    c = np.asarray(short_tracks, float).reshape(-1, 6)
    return (
        np.stack([c[:, 0], c[:, 1], c[:, 2], c[:, 3], c[:, 4] - c[:, 2], c[:, 5] - c[:, 3]], 1)
        if len(c)
        else np.zeros((0, 6))
    )
