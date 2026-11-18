"""Track RGB frames with detector prompts and protected writes."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np

from holdtrack.sam import backend, liam
from holdtrack.sam.transactions import (
    _advance_mo_confusion_counter,
    _birth_prompt,
    _current_output,
    _promote_acm_reference,
    _reencode_no_object,
    _validate_acm_bank_alignment,
    _validate_liam_state_layout,
)

NO_OBJ_SCORE = -1024.0
if TYPE_CHECKING:
    from sam3.model.sam3_tracking_predictor import Sam3TrackerPredictor


def _run_sequence(
    predictor: Sam3TrackerPredictor,
    state: dict,
    detections: dict,
    *,
    video: str,
    width: int,
    height: int,
    frame_count: int,
    voice: str,
    torch_module=None,
    on_frame=None,
    on_memory=None,
    stop_after: int | None = None,
) -> tuple[list[tuple], dict]:
    if voice not in ("BASE", "EXCL", "PROT"):
        raise ValueError("Unknown fixed voice")
    if torch_module is None:
        import torch as torch_module
    torch = torch_module
    if (
        frame_count < 1
        or state["num_frames"] != frame_count
        or (state["video_height"], state["video_width"]) != (height, width)
    ):
        raise ValueError("Explicit sequence/state dimensions differ")
    _validate_liam_state_layout(state)
    backend._validate_state_residency(state, offload_state_to_cpu=True)
    retention_window = backend._state_noncond_retention_window(predictor, frame_count)
    active_tracks = {}
    all_rows: list[tuple[int, int, float, float, float, float]] = []
    next_obj_id = 1
    base = SimpleNamespace(
        propagate_one_frame=backend.propagate_one_frame,
        mask_box=backend.mask_box,
        _prune_expired_noncond_outputs=backend._prune_expired_noncond_outputs,
    )
    manager = SimpleNamespace(sam2_base=base, _score_from_state=backend._score_from_state)
    intervention = None
    if voice != "BASE":
        from holdtrack.sam import memory

        cls = memory.MemoryPolicy
        intervention = cls(
            voice,
            on_write=(
                lambda frame, index, logits: on_memory(
                    frame, int(state["obj_idx_to_id"][index]), logits
                )
            )
            if on_memory
            else None,
        )
        base.propagate_one_frame = intervention.wrap(base.propagate_one_frame)
    compute_device = torch.device(state["device"]).type
    if compute_device not in ("cpu", "cuda"):
        raise ValueError("Unsupported explicit compute device")
    context = (
        torch.autocast("cuda", dtype=torch.bfloat16) if compute_device == "cuda" else nullcontext()
    )
    with torch.inference_mode(), context:
        for frame in range(frame_count if stop_after is None else stop_after):
            raw_logits: dict[int, np.ndarray] = {}
            masks: dict[int, np.ndarray] = {}
            scores: dict[int, float] = {}
            observed_scores: dict[int, float] = {}
            boxes: dict[int, tuple[float, float, float, float]] = {}
            if state["obj_ids"]:
                object_ids, mask_logits = manager.sam2_base.propagate_one_frame(
                    predictor, state, frame
                )
                logits_np = mask_logits.detach().float().cpu().numpy()
                if logits_np.ndim != 4 or logits_np.shape[0] != len(object_ids):
                    raise RuntimeError("LiAM SAM2 propagation object/mask shape mismatch")
                for index, obj_id in enumerate(object_ids):
                    raw_logits[obj_id] = np.asarray(logits_np[index, 0], dtype=np.float32)
                    scores[obj_id] = float(manager._score_from_state(state, obj_id, frame))
                    masks[obj_id] = raw_logits[obj_id] > liam.MASK_LOGIT_THRESHOLD
                    observed_scores[obj_id] = scores[obj_id]
                    box = manager.sam2_base.mask_box(raw_logits[obj_id])
                    if box is not None:
                        boxes[obj_id] = tuple(map(float, box))
                if set(object_ids) != set(active_tracks):
                    raise RuntimeError(
                        f"LiAM external obj_id roster drift at {video}:{frame}: SAM2={sorted(object_ids)} adapter={sorted(active_tracks)}"
                    )
            detector_rows = [
                row
                for row in detections.get(frame, ())
                if float(row.confidence) >= liam.DETECTION_FLOOR
            ]
            detector_rows.sort(key=lambda row: (int(row.index), -float(row.confidence)))
            track_ids = sorted(boxes)
            ious = np.zeros((len(track_ids), len(detector_rows)), dtype=np.float64)
            for row_idx, obj_id in enumerate(track_ids):
                for column, detection in enumerate(detector_rows):
                    ious[row_idx, column] = liam.box_iou(boxes[obj_id], detection.box)
            matches = liam.gated_hungarian_matches(ious)
            matched_detection_indices = {column for _, column in matches}
            birth_events = []
            existing_masks = {obj_id: mask for obj_id, mask in masks.items() if mask.any()}
            unmatched = [
                (column, detection)
                for column, detection in enumerate(detector_rows)
                if column not in matched_detection_indices
            ]
            unmatched.sort(key=lambda pair: (-float(pair[1].confidence), int(pair[1].index)))
            for _column, detection in unmatched:
                coverage = liam.tracked_coverage_fraction(detection.box, existing_masks)
                cti_eligible, cti_use_negative_points = liam.cti_precheck(
                    float(detection.confidence), coverage
                )
                event = {
                    "frame": frame,
                    "detection_index": int(detection.index),
                    "detector_score": float(detection.confidence),
                    "tracked_coverage": coverage,
                    "negative_points": [],
                    "status": None,
                }
                if float(detection.confidence) < liam.BIRTH_SCORE:
                    event["status"] = "rejected_birth_score_below_0.60"
                    birth_events.append(event)
                    continue
                if coverage > liam.BIRTH_REJECT_COVERAGE:
                    event["status"] = "rejected_tracked_coverage_above_0.55"
                    birth_events.append(event)
                    continue
                if not cti_eligible:
                    raise RuntimeError(
                        "CTI precheck disagrees with the fixed score/coverage rejection branches"
                    )
                negative_points: tuple[tuple[float, float], ...] = ()
                if cti_use_negative_points:
                    negative_points = liam.contrastive_negative_points(
                        detection.box, existing_masks
                    )
                next_obj_id = max(next_obj_id, max(state["obj_ids"], default=0) + 1)
                before_ids = tuple(map(int, state["obj_ids"]))
                birth_logits, birth_score = _birth_prompt(
                    predictor,
                    state,
                    frame,
                    next_obj_id,
                    detection.box,
                    negative_points,
                    torch,
                    manager,
                )
                candidate_mask = birth_logits > liam.MASK_LOGIT_THRESHOLD
                duplicate_fraction = liam.duplicate_mask_fraction(candidate_mask, existing_masks)
                event.update(
                    {
                        "negative_points": [list(point) for point in negative_points],
                        "sam2_birth_score_logit": birth_score,
                        "duplicate_mask_fraction": duplicate_fraction,
                        "obj_id": next_obj_id,
                    }
                )
                if duplicate_fraction > liam.DUPLICATE_MASK_COVERAGE:
                    predictor.remove_object(state, next_obj_id, strict=True, need_output=False)
                    if next_obj_id in state["obj_id_to_idx"] or tuple(
                        obj for obj in state["obj_ids"] if obj not in before_ids
                    ):
                        raise RuntimeError(
                            "rejected CTI duplicate was not removed from official SAM2 state"
                        )
                    event["status"] = "rejected_initialized_mask_duplicate_above_0.03"
                    next_obj_id += 1
                    birth_events.append(event)
                    continue
                track = liam.LiAMTrackState(obj_id=next_obj_id, birth_frame=frame)
                track.append_acm_reference(frame)
                active_tracks[next_obj_id] = track
                raw_logits[next_obj_id] = birth_logits
                masks[next_obj_id] = candidate_mask
                scores[next_obj_id] = birth_score
                observed_scores[next_obj_id] = birth_score
                box = manager.sam2_base.mask_box(birth_logits)
                if box is not None:
                    boxes[next_obj_id] = tuple(map(float, box))
                existing_masks[next_obj_id] = candidate_mask
                event["status"] = "accepted"
                birth_events.append(event)
                next_obj_id += 1
            if birth_events and any(event.get("status") == "accepted" for event in birth_events):
                predictor.propagate_in_video_preflight(state)
            for obj_id, track in tuple(active_tracks.items()):
                track.update_motion(frame, boxes.get(obj_id))
            acm_candidates: dict[int, float] = {}
            if (frame + 1) % liam.ACM_INTERVAL == 0:
                for obj_id in sorted(active_tracks):
                    current_box = boxes.get(obj_id)
                    observation_nonempty = bool(
                        masks.get(obj_id) is not None and masks[obj_id].any()
                    )
                    if current_box is None or not observation_nonempty:
                        continue
                    max_peer_iou = max(
                        (
                            liam.box_iou(current_box, boxes[other_id])
                            for other_id in boxes
                            if other_id != obj_id
                        ),
                        default=0.0,
                    )
                    if liam.acm_candidate_eligible(
                        source_frame_number=frame + 1,
                        observation_present=obj_id in state["obj_id_to_idx"],
                        observation_nonempty=observation_nonempty,
                        max_other_active_box_iou=max_peer_iou,
                    ):
                        acm_candidates[obj_id] = float(max_peer_iou)
            ids_for_mo = sorted(
                (obj_id for obj_id, mask in masks.items() if mask.any() and obj_id in active_tracks)
            )
            overlaps = list(liam.mo_overlap_pairs({obj_id: masks[obj_id] for obj_id in ids_for_mo}))
            suppressed: set[int] = set()
            retired: set[int] = set()
            suppressed_against: dict[int, int] = {}
            for first_id, second_id, overlap in overlaps:
                if (
                    first_id in suppressed
                    or second_id in suppressed
                    or first_id in retired
                    or (second_id in retired)
                ):
                    continue
                distance, trajectory_support = liam.trajectory_window_squared_mahalanobis(
                    active_tracks[first_id], active_tracks[second_id]
                )
                loser, rule = liam.choose_mgg_loser(
                    active_tracks[first_id],
                    active_tracks[second_id],
                    mahalanobis2=distance,
                    first_current_score=observed_scores[first_id],
                    second_current_score=observed_scores[second_id],
                )
                partner = second_id if loser == first_id else first_id
                track = active_tracks[loser]
                consecutive = _advance_mo_confusion_counter(track, frame, partner)
                suppressed_against[loser] = partner
                suppressed.add(loser)
                if track.birth_frame == frame:
                    predictor.remove_object(state, loser, strict=True, need_output=False)
                    active_tracks.pop(loser)
                    retired.add(loser)
                else:
                    _reencode_no_object(predictor, state, loser, frame, torch, NO_OBJ_SCORE, track)
                    masks[loser] = np.zeros((height, width), dtype=bool)
                    scores[loser] = float(NO_OBJ_SCORE)
                    raw_logits[loser] = np.full(
                        (height, width), float(NO_OBJ_SCORE), dtype=np.float32
                    )
                if consecutive >= 2:
                    if loser not in retired:
                        predictor.remove_object(state, loser, strict=True, need_output=False)
                        active_tracks.pop(loser)
                        retired.add(loser)
            for obj_id, track in active_tracks.items():
                if obj_id not in suppressed_against:
                    track.last_confuser_obj_id = None
                    track.last_confusion_frame = None
                    track.consecutive_confusions = 0
            for obj_id, track in tuple(active_tracks.items()):
                if (
                    track.last_seen_frame is not None
                    and frame - track.last_seen_frame >= liam.MAX_UNSEEN_FRAMES
                ):
                    predictor.remove_object(state, obj_id, strict=True, need_output=False)
                    active_tracks.pop(obj_id)
                    retired.add(obj_id)
            for obj_id, max_peer_iou in sorted(acm_candidates.items()):
                if obj_id in suppressed or obj_id in retired or obj_id not in active_tracks:
                    continue
                output_container, _output_store, _output = _current_output(state, obj_id, frame)
                if output_container != "output_dict_per_obj":
                    raise RuntimeError("ACM observation was not consolidated before promotion")
                acm = _promote_acm_reference(
                    predictor, state, active_tracks[obj_id], frame, liam, retention_window
                )
                acm.update(
                    {
                        "observation_present": True,
                        "observation_nonempty": True,
                        "sam2_object_score_logit_observed": observed_scores.get(obj_id),
                        "max_other_active_box_iou": max_peer_iou,
                    }
                )
            frame_rows: list[tuple[int, int, float, float, float, float]] = []
            for obj_id, track in tuple(active_tracks.items()):
                if obj_id not in state["obj_id_to_idx"]:
                    raise RuntimeError(f"adapter retained retired obj_id {obj_id}")
                if obj_id not in boxes or obj_id in suppressed:
                    box = None
                else:
                    box = boxes[obj_id]
                if obj_id in observed_scores:
                    track.push_score(observed_scores[obj_id])
                if box is not None:
                    frame_rows.append((frame, obj_id, *map(float, box)))
            all_rows.extend(frame_rows)
            if on_frame is not None:
                on_frame(frame, masks, observed_scores, frame_rows)
            manager.sam2_base._prune_expired_noncond_outputs(state, frame, retention_window)
            _validate_acm_bank_alignment(state, active_tracks, liam.ACM_CAPACITY)
    rows = backend._validate_track_rows(all_rows)
    return (rows, {})
