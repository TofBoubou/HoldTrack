"""Commit births, retirements and reference updates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

_LIAM_COMPACT_OUTPUT_FIELDS = {
    "pred_masks",
    "object_score_logits",
    "obj_ptr",
    "maskmem_features",
    "maskmem_pos_enc",
}


def _current_output(state, obj_id: int, frame: int):
    obj_index = state.get("obj_id_to_idx", {}).get(int(obj_id))
    if obj_index is None or state.get("obj_idx_to_id", {}).get(obj_index) != int(obj_id):
        raise RuntimeError(f"SAM2 external obj_id mapping is not bijective for {obj_id}")
    found = []
    for container_name in ("output_dict_per_obj", "temp_output_dict_per_obj"):
        per_object = state.get(container_name, {}).get(obj_index, {})
        for store_name in ("cond_frame_outputs", "non_cond_frame_outputs"):
            output = per_object.get(store_name, {}).get(frame)
            if output is not None:
                found.append((container_name, store_name, output))
    if len(found) != 1:
        raise RuntimeError(
            f"expected exactly one current SAM2 output for obj_id={obj_id}, frame={frame}; found={len(found)}"
        )
    return found[0]


def _validate_acm_bank_alignment(state, tracks, capacity: int) -> dict[int, int]:
    counts: dict[int, int] = {}
    if set(map(int, state.get("obj_ids", ()))) != set(map(int, tracks)):
        raise RuntimeError("SAM2 object IDs and LiAM track states differ at ACM validation")
    for obj_id in sorted(map(int, state["obj_ids"])):
        obj_index = state["obj_id_to_idx"].get(obj_id)
        if obj_index is None or state["obj_idx_to_id"].get(obj_index) != obj_id:
            raise RuntimeError(
                f"SAM2 object-index mapping drift at ACM validation for obj_id={obj_id}"
            )
        cond = state["output_dict_per_obj"][obj_index]["cond_frame_outputs"]
        frames = tuple(map(int, tracks[obj_id].acm_reference_frames))
        if len(frames) > capacity or tuple(sorted(cond)) != frames:
            raise RuntimeError(
                f"SAM2 conditioning entries are not exactly ACM_FIFO{capacity} for obj_id={obj_id}: cond={sorted(cond)}, acm={list(frames)}"
            )
        if any(
            cond[frame].get("maskmem_features") is None
            or cond[frame].get("maskmem_pos_enc") is None
            or cond[frame].get("obj_ptr") is None
            for frame in frames
        ):
            raise RuntimeError(f"ACM contains incomplete encoded output for obj_id={obj_id}")
        counts[obj_id] = len(frames)
    return counts


def _validate_liam_state_layout(state):
    required = {
        "output_dict_per_obj",
        "temp_output_dict_per_obj",
        "point_inputs_per_obj",
        "mask_inputs_per_obj",
        "frames_tracked_per_obj",
        "cached_features",
        "constants",
    }
    missing = sorted(required - set(state))
    if missing:
        raise RuntimeError(f"pinned SAM2 inference state lacks LiAM cache stores: {missing}")
    persistent_combined = {"output_dict", "consolidated_output_dict", "consolidated_masks"} & set(
        state
    )
    if persistent_combined:
        raise RuntimeError(
            f"unadapted persistent SAM2 combined-output cache present: {sorted(persistent_combined)}"
        )
    return {
        "sam2_commit": "2b90b9f5ceec907a1c18123530e92e794ad901a4",
        "persistent_per_object_outputs": ["output_dict_per_obj", "temp_output_dict_per_obj"],
        "persistent_spatial_pointer_fields": sorted(_LIAM_COMPACT_OUTPUT_FIELDS),
        "persistent_combined_output_cache": False,
        "combined_output_behavior": "_consolidate_temp_output_across_obj returns an ephemeral tensor bundle; not stored in inference_state",
        "shared_immutable_caches_preserved": ["cached_features", "constants"],
        "common_frame_metadata_preserved": "frames_tracked_per_obj",
    }


def _commit_liam_object_cache_transaction(
    state,
    obj_index: int,
    *,
    main_outputs,
    temporary_outputs,
    point_inputs,
    mask_inputs,
    track=None,
    acm_reference_frames: Sequence[int] | None = None,
    acm_promoted_from_noncond: set[int] | None = None,
) -> None:
    if not isinstance(state, dict):
        raise RuntimeError(
            "LiAM cache transactions require the pinned mutable SAM2 state dictionary"
        )
    cache_names = (
        "output_dict_per_obj",
        "temp_output_dict_per_obj",
        "point_inputs_per_obj",
        "mask_inputs_per_obj",
    )
    cache_maps = {}
    old_objects = {}
    for name in cache_names:
        cache_map = state.get(name)
        if not isinstance(cache_map, dict) or obj_index not in cache_map:
            raise RuntimeError(f"LiAM transaction cannot resolve {name}[{obj_index}]")
        cache_maps[name] = cache_map
        old_objects[name] = cache_map[obj_index]
    old_track_refs = list(track.acm_reference_frames) if track is not None else None
    old_promoted = set(track.acm_promoted_from_noncond) if track is not None else None
    staged_main = dict(old_objects["output_dict_per_obj"])
    staged_main.update(
        {
            name: dict(main_outputs[name])
            for name in ("cond_frame_outputs", "non_cond_frame_outputs")
        }
    )
    staged_temp = dict(old_objects["temp_output_dict_per_obj"])
    staged_temp.update(
        {
            name: dict(temporary_outputs[name])
            for name in ("cond_frame_outputs", "non_cond_frame_outputs")
        }
    )
    staged_objects = {
        "output_dict_per_obj": staged_main,
        "temp_output_dict_per_obj": staged_temp,
        "point_inputs_per_obj": dict(point_inputs),
        "mask_inputs_per_obj": dict(mask_inputs),
    }
    if track is not None:
        if acm_reference_frames is None or acm_promoted_from_noncond is None:
            raise ValueError("track transaction requires staged ACM reference bookkeeping")
    try:
        for name in cache_names:
            cache_maps[name][obj_index] = staged_objects[name]
        if track is not None:
            track.acm_reference_frames = list(map(int, acm_reference_frames or ()))
            track.acm_promoted_from_noncond = set(map(int, acm_promoted_from_noncond or set()))
    except BaseException:
        for name in cache_names:
            cache_maps[name][obj_index] = old_objects[name]
        if track is not None:
            track.acm_reference_frames = old_track_refs
            track.acm_promoted_from_noncond = old_promoted
        raise


def _validate_compact_liam_output(output) -> None:
    if not isinstance(output, Mapping) or set(output) != _LIAM_COMPACT_OUTPUT_FIELDS:
        raise RuntimeError("SAM 2 output differs from the five expected fields")
    for key in _LIAM_COMPACT_OUTPUT_FIELDS:
        if output[key] is None:
            raise RuntimeError(f"SAM2 compact output has an empty {key} field")
    masks = output["pred_masks"]
    scores = output["object_score_logits"]
    pointer = output["obj_ptr"]
    if masks.ndim != 4 or tuple(masks.shape[:2]) != (1, 1):
        raise RuntimeError("SAM2 per-object mask must have shape [1,1,H,W]")
    if tuple(scores.shape) != (1, 1):
        raise RuntimeError("SAM2 per-object object_score_logits must have shape [1,1]")
    if pointer.ndim != 2 or int(pointer.shape[0]) != 1:
        raise RuntimeError("SAM2 per-object obj_ptr must have shape [1,C]")


def _promote_acm_reference(predictor, state, track, frame: int, liam, retention_window: int):
    obj_index = state["obj_id_to_idx"].get(track.obj_id)
    if obj_index is None or state["obj_idx_to_id"].get(obj_index) != track.obj_id:
        raise RuntimeError(f"ACM promotion lost external obj_id mapping: {track.obj_id}")
    main = state["output_dict_per_obj"][obj_index]
    temp = state["temp_output_dict_per_obj"][obj_index]
    cond = main["cond_frame_outputs"]
    noncond = main["non_cond_frame_outputs"]
    old_refs = tuple(map(int, track.acm_reference_frames))
    if old_refs != tuple(sorted(set(old_refs))) or tuple(sorted(map(int, cond))) != old_refs:
        raise RuntimeError(
            f"pre-promotion SAM2 conditioning and LiAM ACM disagree for obj_id={track.obj_id}"
        )
    if frame in old_refs:
        if frame not in cond:
            raise RuntimeError(
                "existing birth/reference frame is absent from SAM2 conditioning storage"
            )
        return {
            "frame": frame,
            "obj_id": track.obj_id,
            "already_conditioning": True,
            "retained_frames": list(old_refs),
            "evicted_frames": [],
            "promotion_changes_only_category": True,
            "duplicate_birth_promotion_noop": True,
            "truth_or_outcome_used": False,
        }
    if frame in cond or frame not in noncond:
        raise RuntimeError(
            f"ACM candidate is not a unique recent SAM2 output for obj_id={track.obj_id} frame={frame}"
        )
    output = noncond[frame]
    _validate_compact_liam_output(output)
    original_ids = {name: id(output[name]) for name in _LIAM_COMPACT_OUTPUT_FIELDS}
    staged_cond, staged_noncond = (dict(cond), dict(noncond))
    staged_outputs = {"cond_frame_outputs": staged_cond, "non_cond_frame_outputs": staged_noncond}
    evicted = liam.promote_reference_fifo(staged_outputs, frame, capacity=liam.ACM_CAPACITY)
    restored_short_term: list[int] = []
    removed_old: list[int] = []
    for old_frame in evicted:
        if 0 <= frame - old_frame <= retention_window:
            restored_short_term.append(int(old_frame))
        else:
            staged_noncond.pop(old_frame, None)
            removed_old.append(int(old_frame))
    retained = tuple(sorted(map(int, staged_cond)))
    if retained != liam.acm_select_reference_frames(
        (*old_refs, int(frame)), capacity=liam.ACM_CAPACITY
    ):
        raise RuntimeError("staged conditioning bank differs from declared FIFO6 selection")
    if len(staged_cond) > liam.ACM_CAPACITY:
        raise RuntimeError("staged LiAM ACM_FIFO6 conditioning bank exceeds its cap")
    staged_temp = {
        name: dict(temp[name]) for name in ("cond_frame_outputs", "non_cond_frame_outputs")
    }
    staged_points = dict(state["point_inputs_per_obj"][obj_index])
    staged_masks = dict(state["mask_inputs_per_obj"][obj_index])
    for old_frame in evicted:
        staged_temp["cond_frame_outputs"].pop(old_frame, None)
        staged_temp["non_cond_frame_outputs"].pop(old_frame, None)
        staged_points.pop(old_frame, None)
        staged_masks.pop(old_frame, None)
    promoted_from_noncond = (set(track.acm_promoted_from_noncond) | {int(frame)}) & set(retained)
    for old_frame in set(old_refs) & set(retained):
        if staged_cond[old_frame] is not cond[old_frame]:
            raise RuntimeError("ACM promotion replaced a prior reliable reference")
    if any(id(output[name]) != original_ids[name] for name in _LIAM_COMPACT_OUTPUT_FIELDS):
        raise RuntimeError(
            "ACM promotion mutated an encoded tensor instead of changing only its category"
        )
    if staged_cond.get(frame) is not output or frame in staged_noncond:
        raise RuntimeError("staged ACM promotion did not change only the current frame category")
    _commit_liam_object_cache_transaction(
        state,
        obj_index,
        main_outputs=staged_outputs,
        temporary_outputs=staged_temp,
        point_inputs=staged_points,
        mask_inputs=staged_masks,
        track=track,
        acm_reference_frames=retained,
        acm_promoted_from_noncond=promoted_from_noncond,
    )
    return {
        "frame": frame,
        "obj_id": track.obj_id,
        "already_conditioning": False,
        "retained_frames": list(retained),
        "evicted_frames": list(evicted),
        "restored_short_term_frames": restored_short_term,
        "removed_outside_short_term_window": removed_old,
        "promotion_changes_only_category": True,
        "all_persistent_object_caches_committed_together": True,
        "acm_temporal_encoding": "native cond t_pos=0; original maskmem_pos_enc and obj_ptr retained",
        "truth_or_outcome_used": False,
    }


def _make_no_object_output(predictor, state, frame: int, observed_out, torch, no_obj_score: float):
    _validate_compact_liam_output(observed_out)
    if frame < 0 or bool(getattr(predictor, "training", True)):
        raise RuntimeError(
            "no-object memory encoding requires a nonnegative frame and an eval predictor"
        )
    if not all(
        bool(getattr(predictor, name, False))
        for name in ("pred_obj_scores", "use_obj_ptrs_in_encoder", "fixed_no_obj_ptr")
    ) or bool(getattr(predictor, "soft_no_obj_ptr", False)):
        raise RuntimeError("SAM2 predictor lacks the pinned hard/fixed no-object configuration")
    if getattr(predictor, "no_obj_embed_spatial", None) is None:
        raise RuntimeError("SAM2 predictor lacks the spatial no-object embedding")
    image_size = int(predictor.image_size)
    device = state["device"]
    absent_score = torch.full_like(
        observed_out["object_score_logits"], float(no_obj_score), device=device
    )
    high_res_absence = torch.full(
        (1, 1, image_size, image_size), float(no_obj_score), dtype=torch.float32, device=device
    )
    features, position = predictor._run_memory_encoder(
        inference_state=state,
        frame_idx=frame,
        batch_size=1,
        high_res_masks=high_res_absence,
        object_score_logits=absent_score,
        is_mask_from_pts=False,
    )
    if features is None or position is None:
        raise RuntimeError("official SAM2 memory encoder returned an incomplete absence record")
    old_pointer = observed_out["obj_ptr"]
    pointer = predictor.no_obj_ptr.to(device=old_pointer.device, dtype=old_pointer.dtype)
    if tuple(pointer.shape) != tuple(old_pointer.shape):
        raise RuntimeError("official SAM2 no-object pointer shape differs from the current output")
    new_output = {
        "pred_masks": torch.full_like(observed_out["pred_masks"], float(no_obj_score)),
        "object_score_logits": absent_score,
        "obj_ptr": pointer.clone(),
        "maskmem_features": features,
        "maskmem_pos_enc": position,
    }
    _validate_compact_liam_output(new_output)
    return new_output


def _reencode_no_object(
    predictor, state, obj_id: int, frame: int, torch, no_obj_score: float, track
):
    container_name, store_name, output = _current_output(state, obj_id, frame)
    if container_name != "output_dict_per_obj":
        raise RuntimeError("no-object memory rewrite requires a consolidated official SAM2 output")
    obj_index = state["obj_id_to_idx"][obj_id]
    _validate_compact_liam_output(output)
    observed_tensor_ids = {name: id(output[name]) for name in _LIAM_COMPACT_OUTPUT_FIELDS}
    replacement = _make_no_object_output(predictor, state, frame, output, torch, no_obj_score)
    if any(id(output[name]) != observed_tensor_ids[name] for name in _LIAM_COMPACT_OUTPUT_FIELDS):
        raise RuntimeError("no-object encoding modified the observed pre-intervention output")
    if tuple(replacement["pred_masks"].shape[-2:]) != tuple(output["pred_masks"].shape[-2:]):
        raise RuntimeError("no-object re-encoding changed SAM2 output tensor geometry")
    if int(state["video_height"]) <= 0 or int(state["video_width"]) <= 0:
        raise RuntimeError("SAM2 video geometry is invalid")
    main = state["output_dict_per_obj"][obj_index]
    temp = state["temp_output_dict_per_obj"][obj_index]
    staged_cond = dict(main["cond_frame_outputs"])
    staged_noncond = dict(main["non_cond_frame_outputs"])
    staged_cond.pop(frame, None)
    staged_noncond.pop(frame, None)
    staged_noncond[frame] = replacement
    staged_temp = {
        name: dict(temp[name]) for name in ("cond_frame_outputs", "non_cond_frame_outputs")
    }
    staged_temp["cond_frame_outputs"].pop(frame, None)
    staged_temp["non_cond_frame_outputs"].pop(frame, None)
    staged_points = dict(state["point_inputs_per_obj"][obj_index])
    staged_masks = dict(state["mask_inputs_per_obj"][obj_index])
    staged_points.pop(frame, None)
    staged_masks.pop(frame, None)
    new_refs = tuple(value for value in track.acm_reference_frames if int(value) != int(frame))
    new_promoted = set(track.acm_promoted_from_noncond) - {int(frame)}
    if (
        frame in staged_cond
        or staged_noncond.get(frame) is not replacement
        or frame in staged_temp["cond_frame_outputs"]
        or (frame in staged_temp["non_cond_frame_outputs"])
        or (frame in staged_points)
        or (frame in staged_masks)
        or (frame in new_refs)
    ):
        raise RuntimeError(
            "staged Mo transaction retained stale current-frame cache or ACM membership"
        )
    _commit_liam_object_cache_transaction(
        state,
        obj_index,
        main_outputs={"cond_frame_outputs": staged_cond, "non_cond_frame_outputs": staged_noncond},
        temporary_outputs=staged_temp,
        point_inputs=staged_points,
        mask_inputs=staged_masks,
        track=track,
        acm_reference_frames=new_refs,
        acm_promoted_from_noncond=new_promoted,
    )
    location = "non_cond_frame_outputs"
    return {
        "frame": frame,
        "obj_id": obj_id,
        "storage_after_reencode": location,
        "official_memory_encoder_called": True,
        "official_no_obj_score": float(no_obj_score),
        "official_no_obj_pointer_written": True,
        "atomic_output_replacement": True,
        "all_persistent_caches_replaced_together": True,
        "stale_temp_outputs_invalidated": True,
        "observed_output_preserved_unmodified": True,
        "raw_score_history_uses_observed_score": True,
        "ambiguous_memory_removed": True,
        "ambiguous_acm_reference_revoked": True,
    }


def _birth_prompt(
    predictor,
    state,
    frame: int,
    obj_id: int,
    box: Sequence[float],
    negative_points: Sequence[Sequence[float]],
    torch,
    base,
) -> tuple[np.ndarray, float]:
    points = np.asarray(negative_points, dtype=np.float32).reshape(-1, 2)
    labels = np.zeros((len(points),), dtype=np.int32)
    result = predictor.add_new_points_or_box(
        state,
        frame_idx=frame,
        obj_id=obj_id,
        points=points if len(points) else None,
        labels=labels if len(points) else None,
        box=np.asarray(box, dtype=np.float32),
        clear_old_points=True,
        normalize_coords=True,
    )
    if not isinstance(result, tuple) or len(result) != 3 or int(result[0]) != frame:
        raise RuntimeError("LiAM CTI prompt returned an unexpected SAM2 result")
    object_ids = [int(value) for value in result[1]]
    if len(object_ids) != len(set(object_ids)) or obj_id not in object_ids:
        raise RuntimeError("LiAM CTI prompt did not preserve the requested external obj_id")
    logits = result[2].detach().float().cpu().numpy()
    if logits.ndim != 4 or logits.shape[1] != 1 or logits.shape[0] != len(object_ids):
        raise RuntimeError("LiAM CTI prompt output/object cardinality drift")
    if logits.shape[-2:] != (int(state["video_height"]), int(state["video_width"])):
        raise RuntimeError("LiAM CTI prompt did not return full-resolution masks")
    index = object_ids.index(obj_id)
    mask_logits = np.asarray(logits[index, 0], dtype=np.float32)
    score = float(base._score_from_state(state, obj_id, frame))
    return (mask_logits, score)


def _advance_mo_confusion_counter(track, frame: int, partner_obj_id: int) -> int:
    frame_value, partner = (int(frame), int(partner_obj_id))
    if frame_value < 0 or partner < 1:
        raise ValueError("Mo counter requires a nonnegative frame and positive partner obj_id")
    consecutive = (
        int(track.consecutive_confusions) + 1
        if track.last_confuser_obj_id == partner and track.last_confusion_frame == frame_value - 1
        else 1
    )
    track.last_confuser_obj_id = partner
    track.last_confusion_frame = frame_value
    track.consecutive_confusions = consecutive
    return consecutive
