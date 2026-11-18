"""The three memory updates: native (BASE), exclusive (EXCL) and guarded (PROT)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from holdtrack.config import Voice

if TYPE_CHECKING:
    from sam3.model.sam3_tracking_predictor import Sam3TrackerPredictor


@dataclass(frozen=True)
class MemoryEntry:
    object_index: int
    stores: dict
    store_name: str
    output: dict
    prompted: bool
    absent: bool


class MemoryPolicy:
    def __init__(self, voice: Voice, *, on_write: Callable | None = None) -> None:
        if voice not in {"BASE", "EXCL", "PROT"}:
            raise ValueError("Voice must be BASE, EXCL or PROT")
        self.voice = voice
        self.last_frame: int | None = None
        self.on_write = on_write

    def apply(self, predictor: Sam3TrackerPredictor, state: dict, frame: int) -> None:
        import torch
        import torch.nn.functional as functional

        entries: list[MemoryEntry] = []
        for object_index, stores in state["output_dict_per_obj"].items():
            names = [
                name
                for name in ("non_cond_frame_outputs", "cond_frame_outputs")
                if frame in stores[name]
            ]
            if not names:
                continue
            output = stores[names[0]][frame]
            prompted = (
                frame in state["point_inputs_per_obj"][object_index]
                or frame in state["mask_inputs_per_obj"][object_index]
            )
            entries.append(
                MemoryEntry(
                    object_index,
                    stores,
                    names[0],
                    output,
                    prompted,
                    float(output["pred_masks"].max()) <= -1023.0,
                )
            )
        if not entries:
            return
        masks = [
            functional.interpolate(
                entry.output["pred_masks"].to(state["device"], non_blocking=True).float(),
                size=(int(predictor.image_size), int(predictor.image_size)),
                mode="bilinear",
                align_corners=False,
            )
            for entry in entries
        ]
        protected = set()
        if len(masks) > 1:
            stacked = torch.cat(masks, dim=0)
            exclusive = predictor._apply_non_overlapping_constraints(stacked)
            lost = ((stacked > 0) & (exclusive <= 0)).flatten(1).sum(1)
            area = (stacked > 0).flatten(1).sum(1)
            if self.voice == "PROT":
                for index, entry in enumerate(entries):
                    if int(lost[index]) > 0 and (not entry.prompted) and (not entry.absent):
                        if int(lost[index]) / max(int(area[index]), 1) > 0.5:
                            protected.add(index)
            masks = [exclusive[index : index + 1] for index in range(len(entries))]
        for index, entry in enumerate(entries):
            if index in protected or entry.prompted or entry.absent:
                continue
            features, positions = predictor._run_memory_encoder(
                inference_state=state,
                frame_idx=frame,
                batch_size=1,
                high_res_masks=masks[index],
                object_score_logits=entry.output["object_score_logits"].to(state["device"]),
                is_mask_from_pts=False,
            )
            updated = dict(entry.output)
            updated["maskmem_features"] = features
            updated["maskmem_pos_enc"] = positions
            entry.stores[entry.store_name][frame] = updated
            if self.on_write is not None:
                self.on_write(frame, entry.object_index, masks[index])

    def wrap(self, propagate: Callable) -> Callable:

        def step(predictor, state: dict, frame: int):
            if (
                self.voice != "BASE"
                and self.last_frame is not None
                and (self.last_frame == frame - 1)
            ):
                self.apply(predictor, state, self.last_frame)
            result = propagate(predictor, state, frame)
            self.last_frame = frame
            return result

        return step
