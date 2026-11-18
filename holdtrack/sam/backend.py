"""Propagate native masks and maintain compact object state."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

SAM2_MASK_LOGIT_THRESHOLD = 0.0


def mot_bytes(rows):
    checked = _validate_track_rows(rows)
    return "".join(
        (
            f"{frame + 1},{tid},{x1:.4f},{y1:.4f},{x2 - x1:.4f},{y2 - y1:.4f},1.000000,-1,-1,-1\n"
            for frame, tid, x1, y1, x2, y2 in checked
        )
    ).encode()


def propagate_one_frame(predictor, state, frame: int):
    generated = iter(
        predictor.propagate_in_video(
            state, start_frame_idx=frame, max_frame_num_to_track=0, reverse=False
        )
    )
    try:
        result = next(generated)
    except StopIteration as error:
        raise RuntimeError(f"SAM2 emitted no current-frame result at {frame}") from error
    try:
        next(generated)
    except StopIteration:
        pass
    else:
        raise RuntimeError(f"SAM2 emitted beyond the current frame {frame}")
    actual_frame, object_ids, masks = result
    if int(actual_frame) != frame:
        raise RuntimeError(f"SAM2 frame index drift: {actual_frame} != {frame}")
    return ([int(value) for value in object_ids], masks)


def mask_box(
    mask_logits: np.ndarray, *, threshold: float = SAM2_MASK_LOGIT_THRESHOLD
) -> tuple[float, float, float, float] | None:
    mask = np.asarray(mask_logits).squeeze()
    if mask.ndim != 2:
        raise ValueError(f"SAM2 mask must be HxW logits, got {mask.shape}")
    ys, xs = np.nonzero(mask > threshold)
    if len(xs) == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def _validate_track_rows(
    rows: Sequence[Sequence[float]],
) -> list[tuple[int, int, float, float, float, float]]:
    checked: list[tuple[int, int, float, float, float, float]] = []
    seen: set[tuple[int, int]] = set()
    for raw in rows:
        if len(raw) != 6:
            raise ValueError("MOT source row must be frame, tid, x1, y1, x2, y2")
        frame, tid = (int(raw[0]), int(raw[1]))
        x1, y1, x2, y2 = map(float, raw[2:])
        if frame < 0 or tid < 1 or (not np.isfinite((x1, y1, x2, y2)).all()):
            raise ValueError("invalid GT-free tracker row")
        if x2 <= x1 or y2 <= y1:
            raise ValueError("non-positive tracker box")
        if (frame, tid) in seen:
            raise ValueError(f"duplicate frame/track row: {(frame, tid)}")
        seen.add((frame, tid))
        checked.append((frame, tid, x1, y1, x2, y2))
    return sorted(checked)


def _state_noncond_retention_window(predictor, frame_count: int) -> int:
    try:
        num_maskmem = int(predictor.num_maskmem)
        stride = int(predictor.memory_temporal_stride_for_eval)
        max_ptrs = (
            int(predictor.max_obj_ptrs_in_encoder) if bool(predictor.use_obj_ptrs_in_encoder) else 1
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise RuntimeError("cannot prove SAM2 non-conditioning cache bound") from error
    if num_maskmem < 1 or stride < 1 or max_ptrs < 1 or (frame_count < 1):
        raise RuntimeError("invalid SAM2 memory-window configuration")
    return min(frame_count, max(max_ptrs, num_maskmem * stride + 2))


def _prune_expired_noncond_outputs(state, current_frame: int, retention_window: int) -> int:
    cutoff = current_frame - retention_window
    removed = 0
    for per_object in state["output_dict_per_obj"].values():
        outputs = per_object["non_cond_frame_outputs"]
        expired = [frame for frame in outputs if int(frame) < cutoff]
        for frame in expired:
            del outputs[frame]
            removed += 1
    return removed


def _score_from_state(state, track_id: int, frame: int) -> float:
    obj_index = state["obj_id_to_idx"].get(track_id)
    if obj_index is None:
        raise RuntimeError(f"SAM2 object score requested for absent id {track_id}")
    for container_name in ("temp_output_dict_per_obj", "output_dict_per_obj"):
        per_object = state[container_name].get(obj_index, {})
        for field in ("cond_frame_outputs", "non_cond_frame_outputs"):
            output = per_object.get(field, {}).get(frame)
            if output is not None and "object_score_logits" in output:
                values = output["object_score_logits"].detach().float().reshape(-1)
                if values.numel() != 1:
                    raise RuntimeError("SAM2 object-score logit must be scalar per object")
                score = float(values[0].cpu())
                if not np.isfinite(score):
                    raise RuntimeError("SAM2 object-score logit is not finite")
                return score
    raise RuntimeError(f"SAM2 did not retain object score for {track_id} at frame {frame}")


def _validate_state_residency(state, *, offload_state_to_cpu: bool) -> dict[str, bool | str]:
    reported_offload = state.get("offload_state_to_cpu")
    storage_device = state.get("storage_device")
    compute_device = state.get("device")
    if reported_offload is None or bool(reported_offload) != bool(offload_state_to_cpu):
        raise RuntimeError("SAM2 state does not report the requested offload mode")
    if storage_device is None or compute_device is None:
        raise RuntimeError("SAM2 state residency cannot be verified from predictor metadata")
    storage_device = str(storage_device)
    compute_device = str(compute_device)
    expected_device = "cpu" if offload_state_to_cpu else compute_device
    if storage_device != expected_device:
        raise RuntimeError(
            f"SAM2 state storage device differs from the requested mode: {storage_device} != {expected_device}"
        )
    return {
        "offload_video_to_cpu": True,
        "offload_state_to_cpu": bool(offload_state_to_cpu),
        "storage_device": storage_device,
        "compute_device": compute_device,
    }
