"""Associate detector boxes with the original ByteTrack implementation."""

from __future__ import annotations

import hashlib
import importlib.metadata
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

REFERENCE_BACKEND_CONFIG = dict(
    track_high_thresh=0.3,
    track_low_thresh=0.01,
    new_track_thresh=0.3,
    track_buffer=30,
    match_thresh=0.95,
    fuse_score=True,
    emit_unconfirmed=False,
    emit_orphans=False,
)


@dataclass
class Detection:
    frame_idx: int
    bbox_xyxy: np.ndarray
    conf: float
    emb: torch.Tensor | None = None
    source_id: int = -1
    occluded: bool = False


def iou(box1: np.ndarray, box2: np.ndarray) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


@dataclass
class Tracklet:
    tid: int
    start_f: int
    end_f: int
    first_bbox: np.ndarray
    last_bbox: np.ndarray
    first_pos: np.ndarray
    last_pos: np.ndarray
    embs: list[torch.Tensor]
    emb_frames: list[int] = field(default_factory=list)
    emb_clean: list = field(default_factory=list)
    gait_emb: np.ndarray | None = None
    mu: torch.Tensor | None = None
    sigma: float = 0.0
    mean_conf: float = 0.0
    mean_area: float = 0.0
    quality: float = 0.0
    history: dict[int, np.ndarray] = field(default_factory=dict)
    color_hist: np.ndarray | None = None
    conf_history: dict[int, float] = field(default_factory=dict)
    diag_journal: list = field(default_factory=list)
    match_journal: list = field(default_factory=list)
    low_obs: bool = False
    crops: list = field(default_factory=list)
    atomicity: object | None = None


def compute_mu_sigma(embs: list[torch.Tensor]) -> tuple[torch.Tensor | None, float]:
    if len(embs) == 0:
        return (None, 0.0)
    if len(embs) == 1:
        return (embs[0].clone(), 0.0)
    emb_stack = torch.stack(embs)
    mu = F.normalize(emb_stack.mean(dim=0), dim=0)
    distances = []
    for emb in embs:
        data = 1.0 - torch.dot(emb, mu).item()
        distances.append(data)
    sigma = np.mean(distances)
    return (mu, sigma)


def get_bbox_center(bbox: np.ndarray) -> np.ndarray:
    return np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)


class _DetectionBatch:
    def __init__(
        self, boxes: np.ndarray, confidences: np.ndarray, classes: np.ndarray | None = None
    ) -> None:
        self.xyxy = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(confidences, dtype=np.float32).reshape(-1)
        self.cls = (
            np.zeros(len(self.conf), dtype=np.float32)
            if classes is None
            else np.asarray(classes, dtype=np.float32).reshape(-1)
        )

    @property
    def xywh(self) -> np.ndarray:
        boxes = self.xyxy
        result = np.empty_like(boxes)
        result[:, 0] = (boxes[:, 0] + boxes[:, 2]) * 0.5
        result[:, 1] = (boxes[:, 1] + boxes[:, 3]) * 0.5
        result[:, 2] = boxes[:, 2] - boxes[:, 0]
        result[:, 3] = boxes[:, 3] - boxes[:, 1]
        return result

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, item) -> _DetectionBatch:
        indices = np.arange(len(self))[item]
        indices = np.atleast_1d(indices)
        return _DetectionBatch(self.xyxy[indices], self.conf[indices], self.cls[indices])


@dataclass(frozen=True)
class ByteTrackConfig:
    track_high_thresh: float = 0.1
    track_low_thresh: float = 0.01
    new_track_thresh: float = 0.1
    track_buffer: int = 30
    match_thresh: float = 0.8
    fuse_score: bool = True
    emit_unconfirmed: bool = True
    emit_orphans: bool = False


@dataclass
class _ObservationChain:
    tid: int
    history: dict[int, np.ndarray] = field(default_factory=dict)
    conf_history: dict[int, float] = field(default_factory=dict)
    embs: list[torch.Tensor] = field(default_factory=list)
    emb_frames: list[int] = field(default_factory=list)
    emb_clean: list[bool] = field(default_factory=list)
    confirmed_once: bool = False

    def append(self, detection: Detection, frame: int, *, clean: bool) -> None:
        if frame in self.history:
            return
        self.history[int(frame)] = detection.bbox_xyxy.astype(np.float32, copy=True)
        self.conf_history[int(frame)] = float(detection.conf)
        if detection.emb is not None:
            self.embs.append(detection.emb.clone())
            self.emb_frames.append(int(frame))
            self.emb_clean.append(bool(clean))


def _to_tracklet(chain: _ObservationChain) -> Tracklet:
    frames = sorted(chain.history)
    history = {frame: chain.history[frame].copy() for frame in frames}
    first_bbox, last_bbox = (history[frames[0]], history[frames[-1]])
    mu, sigma = compute_mu_sigma(chain.embs)
    confidences = list(chain.conf_history.values())
    mean_conf = float(np.mean(confidences)) if confidences else 0.0
    areas = [
        max(float(box[2] - box[0]), 0.0) * max(float(box[3] - box[1]), 0.0)
        for box in history.values()
    ]
    return Tracklet(
        tid=int(chain.tid),
        start_f=int(frames[0]),
        end_f=int(frames[-1]),
        first_bbox=first_bbox.copy(),
        last_bbox=last_bbox.copy(),
        first_pos=get_bbox_center(first_bbox),
        last_pos=get_bbox_center(last_bbox),
        embs=list(chain.embs),
        emb_frames=list(chain.emb_frames),
        emb_clean=list(chain.emb_clean),
        mu=mu,
        sigma=float(sigma),
        mean_conf=mean_conf,
        mean_area=float(np.mean(areas)),
        quality=mean_conf * len(chain.embs),
        history=history,
        conf_history=dict(chain.conf_history),
    )


class ByteTrackEvidenceTracker:
    def __init__(
        self, fps: float, width: int, height: int, config: ByteTrackConfig | None = None
    ) -> None:
        from ultralytics.trackers.byte_tracker import BYTETracker

        self.fps = float(fps)
        self.W, self.H = (int(width), int(height))
        self.config = config or ByteTrackConfig()
        args = SimpleNamespace(
            tracker_type="bytetrack",
            track_high_thresh=float(self.config.track_high_thresh),
            track_low_thresh=float(self.config.track_low_thresh),
            new_track_thresh=float(self.config.new_track_thresh),
            track_buffer=int(self.config.track_buffer),
            match_thresh=float(self.config.match_thresh),
            fuse_score=bool(self.config.fuse_score),
        )
        self._tracker = BYTETracker(args)
        self._chains: dict[int, _ObservationChain] = {}
        self._orphan_chains: list[_ObservationChain] = []
        self._next_orphan_tid = 1000000000
        self.tracklets: list[Tracklet] = []
        self.hypothesis_archive: list[Tracklet] = []
        self.frame_assignments: dict[int, list[tuple[int, np.ndarray]]] = {}
        self.stats = {"detections": 0, "assigned": 0, "orphans": 0}

    @staticmethod
    def _clean_flags(detections: list[Detection]) -> list[bool]:
        clean = [True] * len(detections)
        for left in range(len(detections)):
            for right in range(left + 1, len(detections)):
                if iou(detections[left].bbox_xyxy, detections[right].bbox_xyxy) >= 0.3:
                    clean[left] = clean[right] = False
        return clean

    def step(self, detections: list[Detection], frame_idx: int) -> None:
        frame_idx = int(frame_idx)
        self.stats["detections"] += len(detections)
        boxes = np.asarray([det.bbox_xyxy for det in detections], dtype=np.float32).reshape(-1, 4)
        confidences = np.asarray([det.conf for det in detections], dtype=np.float32)
        batch = _DetectionBatch(boxes, confidences)
        self._tracker.update(batch)
        clean = self._clean_flags(detections)
        used: set[int] = set()
        assignments: list[tuple[int, np.ndarray]] = []
        for track in self._tracker.tracked_stracks:
            if int(track.frame_id) != int(self._tracker.frame_id):
                continue
            index = int(track.idx)
            if index < 0 or index >= len(detections) or index in used:
                continue
            tid = int(track.track_id)
            chain = self._chains.setdefault(tid, _ObservationChain(tid))
            chain.confirmed_once = chain.confirmed_once or bool(track.is_activated)
            chain.append(detections[index], frame_idx, clean=clean[index])
            used.add(index)
            assignments.append((tid, detections[index].bbox_xyxy.copy()))
            self.stats["assigned"] += 1
        for index, detection in enumerate(detections):
            if index in used:
                continue
            self.stats["orphans"] += 1
            if not self.config.emit_orphans:
                continue
            chain = _ObservationChain(self._next_orphan_tid)
            self._next_orphan_tid += 1
            chain.append(detection, frame_idx, clean=clean[index])
            self._orphan_chains.append(chain)
            assignments.append((chain.tid, detection.bbox_xyxy.copy()))
        self.frame_assignments[frame_idx] = assignments

    def finalize(self) -> list[Tracklet]:
        if self.tracklets:
            return self.tracklets
        chains = [
            chain
            for chain in self._chains.values()
            if chain.history and (chain.confirmed_once or self.config.emit_unconfirmed)
        ]
        chains.extend(self._orphan_chains)
        self.tracklets = [_to_tracklet(chain) for chain in chains]
        self.hypothesis_archive = list(self.tracklets)
        emitted_tids = {tracklet.tid for tracklet in self.tracklets}
        self.frame_assignments = {
            frame: [(tid, box) for tid, box in rows if tid in emitted_tids]
            for frame, rows in self.frame_assignments.items()
        }
        return self.tracklets


def associate_predictions(
    frames: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    *,
    frame_count: int,
    width: int,
    height: int,
) -> tuple[list[tuple[int, int, float, float, float, float]], dict[str, int]]:
    config = ByteTrackConfig(**REFERENCE_BACKEND_CONFIG)
    tracker = ByteTrackEvidenceTracker(30.0, width, height, config=config)
    order = np.argsort(frames, kind="stable")
    cursor = 0
    for frame_index in range(frame_count):
        start = cursor
        while cursor < len(order) and int(frames[order[cursor]]) == frame_index:
            cursor += 1
        selected = order[start:cursor]
        detections = [
            Detection(
                frame_idx=frame_index,
                bbox_xyxy=boxes[index].copy(),
                conf=float(scores[index]),
                source_id=int(index),
            )
            for index in selected
        ]
        tracker.step(detections, frame_index)
    if cursor != len(order):
        raise ValueError("prediction rows fall outside the enumerated frame clock")
    tracker.finalize()
    rows = sorted(
        (
            (int(frame), int(identity), *map(float, box))
            for frame, assignments in tracker.frame_assignments.items()
            for identity, box in assignments
        )
    )
    if len({(row[0], row[1]) for row in rows}) != len(rows):
        raise ValueError("ByteTrack emitted duplicate identity/frame rows")
    return (
        rows,
        {
            "input_detections": int(tracker.stats["detections"]),
            "assigned_observations": int(tracker.stats["assigned"]),
            "orphan_detections": int(tracker.stats["orphans"]),
            "emitted_tracklets": int(len(tracker.tracklets)),
            "output_rows": int(len(rows)),
        },
    )


def mot_text(prediction: Mapping[str, np.ndarray]) -> str:
    lines: list[str] = []
    for frame, identity, box, confidence in zip(
        prediction["frame"],
        prediction["identity"],
        prediction["boxes"],
        prediction["confidence"],
        strict=True,
    ):
        x1, y1, x2, y2 = map(float, box)
        lines.append(
            f"{int(frame) + 1},{int(identity)},{x1:.8f},{y1:.8f},{x2 - x1:.8f},{y2 - y1:.8f},{float(confidence):.8f},-1,-1,-1\n"
        )
    return "".join(lines)


def verify_native_backend():
    if importlib.metadata.version("ultralytics") != "8.4.64":
        raise ValueError("Exact Ultralytics 8.4.64 association runtime required")
    root = Path(importlib.metadata.distribution("ultralytics").locate_file("ultralytics"))
    path = root / "trackers/byte_tracker.py"
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = "00c7c1ff8b1076fc89347071519ae0b06a30bfce00dded94826620e068b1928f"
    if actual != expected:
        raise ValueError("Native ByteTrack source SHA differs")
    return dict(version="8.4.64", native_implementation_SHA256=actual)


def replay_npz(bank, sha256, *, frame_count, width, height, output):
    from holdtrack.artifacts import write_new

    verify_native_backend()
    bank = Path(bank)
    if hashlib.file_digest(bank.open("rb"), "sha256").hexdigest() != sha256:
        raise ValueError("Raw bank SHA differs")
    with np.load(bank, allow_pickle=False) as z:
        keys = set(z.files)
        fkey, skey = (
            ("frame", "score") if {"frame", "boxes", "score"} <= keys else ("frames", "scores")
        )
        frames, boxes, scores = (z[key].copy() for key in [fkey, "boxes", skey])
    if frames.ndim != 1 or boxes.shape != (len(frames), 4) or scores.shape != frames.shape:
        raise ValueError("Malformed raw bank")
    if (
        not np.isfinite(boxes).all()
        or not np.isfinite(scores).all()
        or (boxes[:, 2:] <= boxes[:, :2]).any()
    ):
        raise ValueError("Invalid raw extents/scores")
    if (
        ((scores < 0) | (scores > 1)).any()
        or not np.equal(frames, frames.astype(np.int64)).all()
        or (frames < 0).any()
    ):
        raise ValueError("Invalid source clock/scores")
    rows, audit = associate_predictions(
        frames, boxes, scores, frame_count=frame_count, width=width, height=height
    )
    pred = dict(
        frame=np.asarray([observation[0] for observation in rows], dtype=np.int32),
        identity=np.asarray([observation[1] for observation in rows], dtype=np.int64),
        boxes=np.asarray([observation[2:6] for observation in rows], dtype=np.float32).reshape(
            -1, 4
        ),
        confidence=np.ones(len(rows), dtype=np.float32),
    )
    payload = mot_text(pred).encode()
    write_new(output, payload)
    return dict(audit, SHA256=hashlib.sha256(payload).hexdigest())
