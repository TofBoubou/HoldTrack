"""Compute and combine tracking sufficient statistics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1
SCOPE = "TrackEval 1.0.dev1 exact HOTA Identity sufficient statistics"


def _trackeval_types() -> tuple[type, type, type]:
    for name, value in {"float": float, "int": int, "bool": bool}.items():
        if not hasattr(np, name):
            setattr(np, name, value)
    from trackeval.datasets.mot_challenge_2d_box import MotChallenge2DBox
    from trackeval.metrics.hota import HOTA
    from trackeval.metrics.identity import Identity

    return (MotChallenge2DBox, HOTA, Identity)


def _xywh(row) -> list[float]:
    x1, y1, x2, y2 = map(float, row[2:6])
    return [round(x1, 4), round(y1, 4), round(x2 - x1, 4), round(y2 - y1, 4)]


def _group_rows(rows: Sequence[tuple]) -> dict[int, list[tuple]]:
    grouped: dict[int, list[tuple]] = defaultdict(list)
    seen = set()
    for row in sorted(rows):
        if len(row) != 6:
            raise ValueError("TrackEval sufficient row must have six fields")
        frame, identity = (int(row[0]), int(row[1]))
        if frame < 0 or identity < 0:
            raise ValueError("TrackEval sufficient row has negative key")
        key = (frame, identity)
        if key in seen:
            raise ValueError(f"TrackEval sufficient duplicate identity-frame: {key}")
        seen.add(key)
        grouped[frame].append(tuple(row))
    return grouped


def _metric_data(gt_rows: Sequence[tuple], pred_rows: Sequence[tuple]) -> dict:
    dataset_type, _, _ = _trackeval_types()
    gt_by_frame = _group_rows(gt_rows)
    pred_by_frame = _group_rows(pred_rows)
    if not gt_rows:
        raise ValueError("TrackEval sufficient sequence has no GT")
    timesteps = max((*gt_by_frame, *pred_by_frame), default=0) + 1
    gt_values = sorted({int(row[1]) for row in gt_rows})
    pred_values = sorted({int(row[1]) for row in pred_rows})
    gt_map = {value: index for index, value in enumerate(gt_values)}
    pred_map = {value: index for index, value in enumerate(pred_values)}
    gt_ids = []
    tracker_ids = []
    gt_dets = []
    tracker_dets = []
    similarities = []
    for frame in range(timesteps):
        gt_frame = gt_by_frame.get(frame, [])
        pred_frame = pred_by_frame.get(frame, [])
        gt_ids.append(np.asarray([gt_map[int(row[1])] for row in gt_frame], dtype=int))
        tracker_ids.append(np.asarray([pred_map[int(row[1])] for row in pred_frame], dtype=int))
        gt_boxes = np.asarray([_xywh(row) for row in gt_frame], dtype=float)
        pred_boxes = np.asarray([_xywh(row) for row in pred_frame], dtype=float)
        if not gt_frame:
            gt_boxes = np.empty((0, 4), dtype=float)
        if not pred_frame:
            pred_boxes = np.empty((0, 4), dtype=float)
        gt_dets.append(gt_boxes)
        tracker_dets.append(pred_boxes)
        similarities.append(
            dataset_type._calculate_box_ious(gt_boxes, pred_boxes, box_format="xywh", do_ioa=False)
        )
    return {
        "num_timesteps": timesteps,
        "num_gt_ids": len(gt_values),
        "num_tracker_ids": len(pred_values),
        "num_gt_dets": len(gt_rows),
        "num_tracker_dets": len(pred_rows),
        "gt_ids": gt_ids,
        "tracker_ids": tracker_ids,
        "gt_dets": gt_dets,
        "tracker_dets": tracker_dets,
        "similarity_scores": similarities,
        "seq": "content",
    }


def sequence_sufficient_statistics(gt_rows: Sequence[tuple], pred_rows: Sequence[tuple]) -> dict:
    _, hota_type, identity_type = _trackeval_types()
    data = _metric_data(gt_rows, pred_rows)
    hota_metric = hota_type({"THRESHOLD": 0.5, "PRINT_CONFIG": False})
    identity_metric = identity_type({"THRESHOLD": 0.5, "PRINT_CONFIG": False})
    hota = hota_metric.eval_sequence(data)
    identity = identity_metric.eval_sequence(data)
    tp = np.asarray(hota["HOTA_TP"], dtype=float)
    return {
        "schema": SCHEMA_VERSION,
        "scope": SCOPE,
        "alpha_labels": [float(value) for value in hota_metric.array_labels],
        "hota": {
            "HOTA_TP": tp.tolist(),
            "HOTA_FN": np.asarray(hota["HOTA_FN"], dtype=float).tolist(),
            "HOTA_FP": np.asarray(hota["HOTA_FP"], dtype=float).tolist(),
            "AssA_x_TP": (np.asarray(hota["AssA"]) * tp).tolist(),
            "AssRe_x_TP": (np.asarray(hota["AssRe"]) * tp).tolist(),
            "AssPr_x_TP": (np.asarray(hota["AssPr"]) * tp).tolist(),
            "LocA_x_TP": (np.asarray(hota["LocA"]) * tp).tolist(),
        },
        "identity": {key: int(identity[key]) for key in ("IDTP", "IDFN", "IDFP")},
    }


def _validate(statistic) -> int:
    if statistic.get("schema") != SCHEMA_VERSION or statistic.get("scope") != SCOPE:
        raise ValueError("invalid TrackEval sufficient statistic")
    labels = statistic.get("alpha_labels")
    hota = statistic.get("hota")
    identity = statistic.get("identity")
    if (
        not isinstance(labels, list)
        or not labels
        or (not isinstance(hota, Mapping))
        or (not isinstance(identity, Mapping))
    ):
        raise ValueError("malformed TrackEval sufficient statistic")
    length = len(labels)
    fields = ("HOTA_TP", "HOTA_FN", "HOTA_FP", "AssA_x_TP", "AssRe_x_TP", "AssPr_x_TP", "LocA_x_TP")
    if any(not isinstance(hota.get(field), list) or len(hota[field]) != length for field in fields):
        raise ValueError("TrackEval sufficient HOTA shape drift")
    if set(identity) != {"IDTP", "IDFN", "IDFP"}:
        raise ValueError("TrackEval sufficient Identity shape drift")
    return length


def combine_sufficient_statistics(statistics):
    if not statistics:
        raise ValueError("TrackEval sufficient combination is empty")
    length = _validate(statistics[0])
    labels = list(statistics[0]["alpha_labels"])
    for statistic in statistics[1:]:
        if _validate(statistic) != length or list(statistic["alpha_labels"]) != labels:
            raise ValueError("TrackEval sufficient alpha grid drift")
    hota_fields = (
        "HOTA_TP",
        "HOTA_FN",
        "HOTA_FP",
        "AssA_x_TP",
        "AssRe_x_TP",
        "AssPr_x_TP",
        "LocA_x_TP",
    )
    summed = {
        field: sum(
            (np.asarray(statistic["hota"][field], dtype=float) for statistic in statistics),
            np.zeros(length),
        )
        for field in hota_fields
    }
    tp = summed["HOTA_TP"]
    ass_a = summed["AssA_x_TP"] / np.maximum(1.0, tp)
    ass_re = summed["AssRe_x_TP"] / np.maximum(1.0, tp)
    ass_pr = summed["AssPr_x_TP"] / np.maximum(1.0, tp)
    det_a = tp / np.maximum(1.0, tp + summed["HOTA_FN"] + summed["HOTA_FP"])
    hota = np.sqrt(det_a * ass_a)
    identity = {
        field: sum(int(statistic["identity"][field]) for statistic in statistics)
        for field in ("IDTP", "IDFN", "IDFP")
    }
    idtp = identity["IDTP"]
    idfn = identity["IDFN"]
    idfp = identity["IDFP"]
    return {
        "HOTA": float(np.mean(hota)),
        "AssA": float(np.mean(ass_a)),
        "AssRe": float(np.mean(ass_re)),
        "AssPr": float(np.mean(ass_pr)),
        "DetA": float(np.mean(det_a)),
        "IDF1": float(idtp / max(1.0, idtp + 0.5 * idfp + 0.5 * idfn)),
        "IDR": float(idtp / max(1.0, idtp + idfn)),
        "IDP": float(idtp / max(1.0, idtp + idfp)),
        "HOTA_alpha": [float(value) for value in hota],
        "AssA_alpha": [float(value) for value in ass_a],
        "DetA_alpha": [float(value) for value in det_a],
    }


def _read_mot(path: str | Path, *, ground_truth: bool) -> list[tuple]:
    rows = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        columns = line.strip().split(",")
        if len(columns) < (8 if ground_truth else 6):
            raise ValueError(f"MOT row {line_number} has too few columns")
        if ground_truth and (float(columns[6]) <= 0 or int(float(columns[7])) != 1):
            continue
        frame, identity = float(columns[0]), float(columns[1])
        left, top, width, height = map(float, columns[2:6])
        if not np.isfinite([frame, identity, left, top, width, height]).all():
            raise ValueError(f"MOT row {line_number} is nonfinite")
        if not frame.is_integer() or not identity.is_integer() or frame < 1 or identity < 0:
            raise ValueError(f"MOT row {line_number} has an invalid frame or identity")
        if width <= 0 or height <= 0:
            raise ValueError(f"MOT row {line_number} has a nonpositive box")
        rows.append((int(frame) - 1, int(identity), left, top, left + width, top + height))
    return rows


def evaluate_tracks(
    tracks: str | Path, ground_truth: str | Path, *, num_frames: int | None = None
) -> dict[str, float]:
    """Evaluate DanceTrack MOT files with native TrackEval metrics, returned as ratios."""
    _, hota_type, identity_type = _trackeval_types()
    from trackeval.metrics.clear import CLEAR

    data = _metric_data(
        _read_mot(ground_truth, ground_truth=True), _read_mot(tracks, ground_truth=False)
    )
    if num_frames is not None:
        if num_frames < data["num_timesteps"]:
            raise ValueError("A track or annotation lies beyond the supplied frame count")
        for _ in range(data["num_timesteps"], num_frames):
            data["gt_ids"].append(np.empty(0, dtype=int))
            data["tracker_ids"].append(np.empty(0, dtype=int))
            data["gt_dets"].append(np.empty((0, 4)))
            data["tracker_dets"].append(np.empty((0, 4)))
            data["similarity_scores"].append(np.empty((0, 0)))
        data["num_timesteps"] = num_frames
    configuration = {"THRESHOLD": 0.5, "PRINT_CONFIG": False}
    hota = hota_type(configuration).eval_sequence(data)
    identity = identity_type(configuration).eval_sequence(data)
    clear = CLEAR(configuration).eval_sequence(data)
    return dict(
        HOTA=float(np.mean(hota["HOTA"])),
        DetA=float(np.mean(hota["DetA"])),
        AssA=float(np.mean(hota["AssA"])),
        MOTA=float(clear["MOTA"]),
        IDF1=float(identity["IDF1"]),
    )
