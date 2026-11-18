"""Store detector boxes, scores and frame indices in their native order."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DetectionBank:
    frames: np.ndarray
    boxes: np.ndarray
    scores: np.ndarray

    def __post_init__(self) -> None:
        if (
            self.frames.ndim != 1
            or self.boxes.shape != (len(self.frames), 4)
            or self.scores.shape != self.frames.shape
        ):
            raise ValueError("Detection frames, boxes and scores have incompatible shapes")
        if not all(np.isfinite(values).all() for values in (self.frames, self.boxes, self.scores)):
            raise ValueError("Detection values must be finite")
        if not np.equal(self.frames, self.frames.astype(np.int64)).all() or (self.frames < 0).any():
            raise ValueError("Detection frame indices must be nonnegative integers")
        if ((self.scores < 0) | (self.scores > 1)).any():
            raise ValueError("Detection scores must lie in [0, 1]")

    def arrays(self) -> dict[str, np.ndarray]:
        return dict(frames=self.frames, boxes=self.boxes, scores=self.scores)

    def eligible(self, score_min: float = 0.01) -> tuple["DetectionBank", np.ndarray]:
        indices = np.flatnonzero(
            (self.scores >= score_min) & (self.boxes[:, 2:] > self.boxes[:, :2]).all(1)
        )
        return (
            DetectionBank(self.frames[indices], self.boxes[indices], self.scores[indices]),
            indices,
        )

    def save(self, path: str | Path) -> None:
        with Path(path).open("xb") as stream:
            np.savez_compressed(stream, **self.arrays())

    @classmethod
    def load(cls, path: str | Path) -> "DetectionBank":
        with np.load(path, allow_pickle=False) as archive:
            return cls(*(archive[key].copy() for key in ("frames", "boxes", "scores")))
