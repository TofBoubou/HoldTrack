"""Apply LiAM-SAM births, confusion handling and references."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from holdtrack.sam.kalman import KalmanFilter

DETECTION_FLOOR = 0.3
MATCH_IOU_GATE = 0.3
BIRTH_SCORE = 0.6
BIRTH_REJECT_COVERAGE = 0.55
NEGATIVE_PROMPT_COVERAGE = 0.35
DUPLICATE_MASK_COVERAGE = 0.03
ACM_INTERVAL = 10
ACM_CAPACITY = 6
ACM_MAX_BOX_IOU = 0.5
SAM2_NUM_MASKMEM = 7
SAM2_MEMORY_STRIDE = 1
SAM2_MAX_COND_FRAMES = 6
MGG_MASK_IOU = 0.8
MGG_SCORE_HISTORY = 8
MGG_SCORE_GAP = 4.0
MGG_SCORE_DROP = 4.0
MGG_SCORE_CUTOFF = 2.0
MGG_TRAJECTORY_WINDOW = 8
MGG_MAHALANOBIS2_GATE = 6.0
MGG_RETIRE_AFTER = 2
MAX_UNSEEN_FRAMES = 25
CTI_GRID_SIZE = 256
MASK_LOGIT_THRESHOLD = 0.0
MAHALANOBIS_EPSILON = 1e-09
_KALMAN_CLASS = None


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def gated_hungarian_matches(
    similarities: np.ndarray, minimum_iou: float = MATCH_IOU_GATE
) -> tuple[tuple[int, int], ...]:
    matrix = np.asarray(similarities, dtype=np.float64)
    if (
        matrix.ndim != 2
        or not np.isfinite(matrix).all()
        or np.any(matrix < 0)
        or np.any(matrix > 1)
    ):
        raise ValueError("IoU matrix must be finite, two-dimensional and in [0,1]")
    if not 0 <= minimum_iou <= 1:
        raise ValueError("invalid IoU gate")
    rows, columns = matrix.shape
    if rows == 0 or columns == 0:
        return ()
    from scipy.optimize import linear_sum_assignment

    assigned_rows, assigned_columns = linear_sum_assignment(1.0 - matrix)
    return tuple(
        sorted(
            (
                (int(observation), int(c))
                for observation, c in zip(assigned_rows, assigned_columns, strict=True)
                if matrix[observation, c] >= minimum_iou
            )
        )
    )


def _mask_on_grid(mask: np.ndarray) -> np.ndarray:
    source = np.asarray(mask, dtype=bool)
    if source.ndim != 2 or source.shape[0] <= 0 or source.shape[1] <= 0:
        raise ValueError("CTI masks must be nonempty 2D arrays")
    h, w = source.shape
    ys = np.minimum(((np.arange(CTI_GRID_SIZE) + 0.5) * h / CTI_GRID_SIZE).astype(np.int64), h - 1)
    xs = np.minimum(((np.arange(CTI_GRID_SIZE) + 0.5) * w / CTI_GRID_SIZE).astype(np.int64), w - 1)
    return source[np.ix_(ys, xs)]


def _grid_box(box: Sequence[float], height: int, width: int) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = map(float, box)
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        raise ValueError("non-finite CTI box")
    sx, sy = (CTI_GRID_SIZE / width, CTI_GRID_SIZE / height)
    left = min(max(math.floor(x1 * sx), 0), CTI_GRID_SIZE)
    top = min(max(math.floor(y1 * sy), 0), CTI_GRID_SIZE)
    right = min(max(math.ceil(x2 * sx), 0), CTI_GRID_SIZE)
    bottom = min(max(math.ceil(y2 * sy), 0), CTI_GRID_SIZE)
    return (left, top, right, bottom) if right > left and bottom > top else None


def tracked_coverage_fraction(
    detection_box: Sequence[float], masks_by_obj_id: Mapping[int, np.ndarray]
) -> float:
    if not masks_by_obj_id:
        return 0.0
    first = np.asarray(next(iter(masks_by_obj_id.values())), dtype=bool)
    if first.ndim != 2:
        raise ValueError("active masks must be two-dimensional")
    height, width = first.shape
    crop = _grid_box(detection_box, height, width)
    if crop is None:
        return 0.0
    left, top, right, bottom = crop
    union = np.zeros((CTI_GRID_SIZE, CTI_GRID_SIZE), dtype=bool)
    for obj_id, mask in sorted(masks_by_obj_id.items()):
        if np.asarray(mask).shape != (height, width):
            raise ValueError(f"mask dimensions differ for obj_id={obj_id}")
        union |= _mask_on_grid(mask)
    crop_mask = union[top:bottom, left:right]
    return float(np.count_nonzero(crop_mask) / crop_mask.size)


def contrastive_negative_points(
    detection_box: Sequence[float], masks_by_obj_id: Mapping[int, np.ndarray]
) -> tuple[tuple[float, float], ...]:
    if not masks_by_obj_id:
        return ()
    from scipy import ndimage

    first = np.asarray(next(iter(masks_by_obj_id.values())), dtype=bool)
    if first.ndim != 2:
        raise ValueError("active masks must be two-dimensional")
    height, width = first.shape
    crop = _grid_box(detection_box, height, width)
    if crop is None:
        return ()
    left, top, right, bottom = crop
    structure = np.ones((3, 3), dtype=np.uint8)
    points: list[tuple[float, float]] = []
    for obj_id, raw_mask in sorted(masks_by_obj_id.items()):
        source = np.asarray(raw_mask, dtype=bool)
        if source.shape != (height, width):
            raise ValueError(f"mask dimensions differ for obj_id={obj_id}")
        local = _mask_on_grid(source)[top:bottom, left:right]
        labels, count = ndimage.label(local, structure=structure)
        if count == 0:
            continue
        sizes = np.bincount(labels.ravel(), minlength=count + 1)[1:]
        selected_label = int(np.flatnonzero(sizes == sizes.max())[0]) + 1
        ys, xs = np.nonzero(labels == selected_label)
        cy, cx = (float(ys.mean()), float(xs.mean()))
        nearest = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2))
        gy, gx = (top + int(ys[nearest]), left + int(xs[nearest]))
        points.append(((gx + 0.5) * width / CTI_GRID_SIZE, (gy + 0.5) * height / CTI_GRID_SIZE))
    return tuple(points)


def duplicate_mask_fraction(
    candidate_mask: np.ndarray, active_masks_by_obj_id: Mapping[int, np.ndarray]
) -> float:
    candidate = np.asarray(candidate_mask, dtype=bool)
    if candidate.ndim != 2:
        raise ValueError("candidate mask must be two-dimensional")
    candidate_grid = _mask_on_grid(candidate)
    area = int(candidate_grid.sum())
    if area == 0:
        return 1.0
    occupied = np.zeros_like(candidate_grid)
    for obj_id, mask in sorted(active_masks_by_obj_id.items()):
        if np.asarray(mask).shape != candidate.shape:
            raise ValueError(f"mask dimensions differ for obj_id={obj_id}")
        occupied |= _mask_on_grid(mask)
    return float(np.count_nonzero(candidate_grid & occupied) / area)


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    a, b = (np.asarray(first, dtype=bool), np.asarray(second, dtype=bool))
    if a.ndim != 2 or a.shape != b.shape:
        raise ValueError("mask IoU requires same-shaped 2D arrays")
    union = int(np.count_nonzero(a | b))
    return float(np.count_nonzero(a & b) / union) if union else 0.0


def mo_overlap_pairs(
    masks_by_obj_id: Mapping[int, np.ndarray],
) -> tuple[tuple[int, int, float], ...]:
    if not isinstance(masks_by_obj_id, Mapping):
        raise TypeError("Mo overlap input must map stable obj_id values to raw masks")
    object_ids = sorted(map(int, masks_by_obj_id))
    if len(object_ids) != len(set(object_ids)):
        raise ValueError("Mo overlap input contains duplicate normalized obj_id values")
    pairs: list[tuple[int, int, float]] = []
    for left, first_id in enumerate(object_ids):
        for second_id in object_ids[left + 1 :]:
            overlap = mask_iou(masks_by_obj_id[first_id], masks_by_obj_id[second_id])
            if overlap > MGG_MASK_IOU:
                pairs.append((first_id, second_id, overlap))
    return tuple(pairs)


def acm_select_reference_frames(frames, *, capacity: int = ACM_CAPACITY) -> tuple[int, ...]:
    if type(capacity) is not int or not 0 <= capacity <= ACM_CAPACITY:
        raise ValueError("ACM capacity must be an integer in [0,6]")
    ordered = list(map(int, frames.keys() if isinstance(frames, Mapping) else frames))
    if len(ordered) != len(set(ordered)):
        raise ValueError("duplicate ACM frame reference")
    if ordered != sorted(ordered):
        raise ValueError("ACM references must be supplied in chronological FIFO order")
    return tuple(ordered[-capacity:]) if capacity else ()


def promote_reference_fifo(
    outputs, frame_idx: int, capacity: int = ACM_CAPACITY
) -> tuple[int, ...]:
    if type(capacity) is not int or not 1 <= capacity <= ACM_CAPACITY:
        raise ValueError("ACM capacity must be in [1,6]")
    if not isinstance(outputs, Mapping):
        raise TypeError("SAM2 output stores must be a mapping")
    cond = outputs.get("cond_frame_outputs")
    recent = outputs.get("non_cond_frame_outputs")
    if not isinstance(cond, dict) or not isinstance(recent, dict):
        raise TypeError("SAM2 conditioning and recent stores must be mutable dictionaries")
    frame = int(frame_idx)
    in_cond, in_recent = (frame in cond, frame in recent)
    if in_cond == in_recent:
        raise RuntimeError("current ACM output must exist in exactly one of cond/non-cond stores")
    output = cond[frame] if in_cond else recent[frame]
    required = {
        "pred_masks",
        "object_score_logits",
        "obj_ptr",
        "maskmem_features",
        "maskmem_pos_enc",
    }
    if not isinstance(output, Mapping) or set(output) != required:
        raise RuntimeError("ACM requires the pinned five-field SAM2 compact output schema")
    if any(
        output.get(name) is None
        for name in (
            "pred_masks",
            "object_score_logits",
            "obj_ptr",
            "maskmem_features",
            "maskmem_pos_enc",
        )
    ):
        raise RuntimeError("ACM cannot promote an incomplete or unencoded SAM2 output")
    if not bool((output["pred_masks"] > 0).any()):
        raise RuntimeError("ACM cannot promote an empty observation")
    if not in_cond:
        cond[frame] = recent.pop(frame)
    evicted: list[int] = []
    while len(cond) > capacity:
        oldest = min(map(int, cond))
        recent[oldest] = cond.pop(oldest)
        evicted.append(oldest)
    return tuple(evicted)


def acm_candidate_eligible(
    *,
    source_frame_number: int,
    observation_present: bool,
    observation_nonempty: bool,
    max_other_active_box_iou: float,
) -> bool:
    return bool(
        int(source_frame_number) > 0
        and int(source_frame_number) % ACM_INTERVAL == 0
        and observation_present
        and observation_nonempty
        and (float(max_other_active_box_iou) < ACM_MAX_BOX_IOU)
    )


def cti_precheck(score: float, coverage: float) -> tuple[bool, bool]:
    score_value, coverage_value = (float(score), float(coverage))
    if not (
        math.isfinite(score_value)
        and math.isfinite(coverage_value)
        and (0.0 <= score_value <= 1.0)
        and (0.0 <= coverage_value <= 1.0)
    ):
        raise ValueError("CTI score and candidate-box coverage must be finite in [0,1]")
    eligible = score_value >= BIRTH_SCORE and coverage_value <= BIRTH_REJECT_COVERAGE
    return (eligible, bool(eligible and coverage_value > NEGATIVE_PROMPT_COVERAGE))


@dataclass
class ConstantVelocityKalman:
    model: KalmanFilter
    mean: np.ndarray
    covariance: np.ndarray

    @classmethod
    def from_box(cls, box: Sequence[float]) -> ConstantVelocityKalman:
        model = KalmanFilter()
        mean, covariance = model.initiate(model.xyxy_to_xyah(list(map(float, box))))
        return cls(
            model, np.asarray(mean, dtype=np.float64), np.asarray(covariance, dtype=np.float64)
        )

    def predict(self) -> None:
        self.mean, self.covariance = self.model.predict(self.mean, self.covariance)

    def update(self, box: Sequence[float]) -> None:
        self.mean, self.covariance = self.model.update(
            self.mean, self.covariance, self.model.xyxy_to_xyah(list(map(float, box)))
        )

    def project(self) -> tuple[np.ndarray, np.ndarray]:
        mean, covariance = self.model.project(self.mean, self.covariance)
        return (np.asarray(mean, dtype=np.float64), np.asarray(covariance, dtype=np.float64))


def _center_d2(
    first: tuple[np.ndarray, np.ndarray], second: tuple[np.ndarray, np.ndarray]
) -> float:
    delta = np.asarray(first[0][:2] - second[0][:2], dtype=np.float64)
    covariance = np.asarray(first[1][:2, :2] + second[1][:2, :2], dtype=np.float64)
    covariance = covariance + np.eye(2, dtype=np.float64) * MAHALANOBIS_EPSILON
    return float(delta @ np.linalg.solve(covariance, delta))


@dataclass
class LiAMTrackState:
    obj_id: int
    birth_frame: int
    score_history: deque[float] = field(default_factory=lambda: deque(maxlen=MGG_SCORE_HISTORY))
    kalman: ConstantVelocityKalman | None = None
    trajectory_history: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    acm_reference_frames: list[int] = field(default_factory=list)
    acm_promoted_from_noncond: set[int] = field(default_factory=set)
    last_seen_frame: int | None = None
    last_confuser_obj_id: int | None = None
    last_confusion_frame: int | None = None
    consecutive_confusions: int = 0

    def __post_init__(self) -> None:
        if type(self.obj_id) is not int or self.obj_id < 1:
            raise ValueError("stable SAM2 obj_id must be a positive integer")
        if type(self.birth_frame) is not int or self.birth_frame < 0:
            raise ValueError("birth frame must be zero-based and nonnegative")

    def age_at(self, zero_based_frame: int) -> int:
        return int(zero_based_frame) - self.birth_frame + 1

    def push_score(self, observed_raw_logit: float) -> None:
        score = float(observed_raw_logit)
        if not math.isfinite(score):
            raise ValueError("raw object_score_logits must be finite")
        self.score_history.append(score)

    def score_drop(self, current_score: float) -> float:
        if not self.score_history:
            return 0.0
        return float(
            np.mean(np.asarray(self.score_history, dtype=np.float64)) - float(current_score)
        )

    def update_motion(self, frame: int, box: Sequence[float] | None) -> None:
        if (
            type(frame) is not int
            or frame < 0
            or (self.trajectory_history and frame <= max(self.trajectory_history))
        ):
            raise ValueError("Kalman motion updates must be strictly forward by source frame")
        if self.kalman is None:
            if box is None:
                return
            self.kalman = ConstantVelocityKalman.from_box(box)
        else:
            self.kalman.predict()
            if box is not None:
                self.kalman.update(box)
        mean, covariance = self.kalman.project()
        self.trajectory_history[frame] = (mean[:2].copy(), covariance[:2, :2].copy())
        while len(self.trajectory_history) > MGG_TRAJECTORY_WINDOW:
            del self.trajectory_history[min(self.trajectory_history)]
        if box is not None:
            self.last_seen_frame = frame

    def append_acm_reference(self, frame: int) -> tuple[int, ...]:
        if type(frame) is not int or frame < 0:
            raise ValueError("ACM frame must be a zero-based source frame")
        if self.acm_reference_frames and frame <= self.acm_reference_frames[-1]:
            if frame == self.acm_reference_frames[-1]:
                return ()
            raise ValueError("ACM references must append in chronological FIFO order")
        self.acm_reference_frames.append(frame)
        keep = acm_select_reference_frames(self.acm_reference_frames)
        evicted = tuple(frame_id for frame_id in self.acm_reference_frames if frame_id not in keep)
        self.acm_reference_frames[:] = keep
        for frame_id in evicted:
            self.acm_promoted_from_noncond.discard(frame_id)
        return evicted


def trajectory_window_squared_mahalanobis(
    first: LiAMTrackState, second: LiAMTrackState, *, window: int = MGG_TRAJECTORY_WINDOW
) -> tuple[float | None, int]:
    if type(window) is not int or window != MGG_TRAJECTORY_WINDOW:
        raise ValueError("LiAM distinctness uses exactly the fixed eight-frame window")
    common = sorted(set(first.trajectory_history) & set(second.trajectory_history))[-window:]
    if len(common) < window:
        return (None, len(common))
    distances = [
        _center_d2(first.trajectory_history[frame], second.trajectory_history[frame])
        for frame in common
    ]
    return (max(distances), len(common))


def _age_loser(first: LiAMTrackState, second: LiAMTrackState) -> tuple[int, str]:
    if first.birth_frame == second.birth_frame:
        return (max(first.obj_id, second.obj_id), "same_age_higher_obj_id_retires")
    return (
        first.obj_id if first.birth_frame > second.birth_frame else second.obj_id,
        "younger_birth_retires",
    )


def choose_mgg_loser(
    first: LiAMTrackState,
    second: LiAMTrackState,
    *,
    mahalanobis2: float | None,
    first_current_score: float,
    second_current_score: float,
) -> tuple[int, str]:
    s1, s2 = (float(first_current_score), float(second_current_score))
    d1, d2 = (first.score_drop(s1), second.score_drop(s2))
    suspected1 = s1 < MGG_SCORE_CUTOFF and d1 >= MGG_SCORE_DROP
    suspected2 = s2 < MGG_SCORE_CUTOFF and d2 >= MGG_SCORE_DROP
    if suspected1 != suspected2:
        return (
            first.obj_id if suspected1 else second.obj_id,
            "single_mo_mean8_local_suspected_loss",
        )
    if suspected1 and suspected2:
        if d1 != d2:
            return (
                first.obj_id if d1 > d2 else second.obj_id,
                "both_suspected_larger_mean8_drop_retires",
            )
        loser, tie = _age_loser(first, second)
        return (loser, f"both_suspected_equal_drop_{tie}")
    if mahalanobis2 is None:
        loser, tie = _age_loser(first, second)
        return (loser, f"insufficient_eight_common_kalman_frames_{tie}")
    if float(mahalanobis2) <= MGG_MAHALANOBIS2_GATE:
        loser, tie = _age_loser(first, second)
        return (loser, f"close_max8_projected_center_trajectories_{tie}")
    if d1 == d2:
        k, _other = (first, second) if first.obj_id < second.obj_id else (second, first)
        score_k, score_other = (s1, s2) if k is first else (s2, s1)
        drop_k = d1
        tie_note = "tie_drop_k_by_smaller_obj_id"
    elif d1 > d2:
        k, _other, score_k, score_other, drop_k = (first, second, s1, s2, d1)
        tie_note = ""
    else:
        k, _other, score_k, score_other, drop_k = (second, first, s2, s1, d2)
        tie_note = ""
    if drop_k >= MGG_SCORE_DROP and (
        score_other - score_k >= MGG_SCORE_GAP or score_k < MGG_SCORE_CUTOFF
    ):
        return (
            k.obj_id,
            "distinct_trajectories_highest_mean8_drop_rule" + (f"_{tie_note}" if tie_note else ""),
        )
    loser, tie = _age_loser(first, second)
    return (loser, f"distinct_trajectories_inconclusive_{tie}")
