"""Encode output and memory masks for the qualitative examples."""

import json
from pathlib import Path

import numpy as np


def encode(mask: np.ndarray):
    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2:
        raise ValueError("A mask must have height and width dimensions")
    values = array.reshape(-1, order="F")
    changes = np.flatnonzero(values[1:] != values[:-1]) + 1
    counts = np.diff(np.r_[0, changes, len(values)]).tolist()
    if len(values) and values[0]:
        counts.insert(0, 0)
    return {"size": list(array.shape), "counts": counts}


def decode(rle) -> np.ndarray:
    height, width = rle["size"]
    counts = rle["counts"]
    if (
        any(type(value) is not int or value < 0 for value in counts)
        or sum(counts) != height * width
    ):
        raise ValueError("Invalid COCO run lengths")
    pixels = np.repeat(np.arange(len(counts)) % 2, counts).astype(bool)
    return pixels.reshape((height, width), order="F")


class MaskWriter:
    def __init__(self, directory: str | Path, *, start: int = 0, end: int | None = None) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.start = start
        self.end = end

    def frame(
        self, frame: int, masks: dict[int, np.ndarray], scores: dict[int, float], rows: list
    ) -> None:
        if frame < self.start or (self.end is not None and frame > self.end):
            return
        visible = {int(row[1]) for row in rows}
        records = [
            {
                "object_id": identity,
                "mask": encode(mask),
                "object_score": scores.get(identity),
                "score_type": "logit",
                "emitted": identity in visible,
            }
            for identity, mask in masks.items()
        ]
        (self.directory / f"{frame + 1:08d}.json").write_text(
            json.dumps({"frame": frame + 1, "objects": records}) + "\n"
        )

    def memory(self, frame: int, identity: int, logits) -> None:
        if frame < self.start or (self.end is not None and frame > self.end):
            return
        array = logits.detach().float().cpu().numpy()[0, 0] > 0
        folder = self.directory / "memory"
        folder.mkdir(exist_ok=True)
        (folder / f"{frame + 1:08d}_{identity}.json").write_text(
            json.dumps(
                {
                    "frame": frame + 1,
                    "object_id": identity,
                    "memory_mask": encode(array),
                    "resolution": "encoder input",
                }
            )
            + "\n"
        )
