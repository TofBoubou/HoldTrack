"""Refine box geometry using matched detector views."""

from dataclasses import dataclass

import numpy as np

from holdtrack import geometry as box_geometry
from holdtrack.config import RunConfig
from holdtrack.detect.bank import DetectionBank
from holdtrack.fusion.additional import mix as _add_view
from holdtrack.fusion.anchor import geometry as _fuse_anchor
from holdtrack.fusion.constants import (
    ANCHOR_SCORE_MIN,
    DFINE_SCORE_MIN,
    RF_SCORE_MIN,
    VIEW_MATCH_IOU,
    YOLO_SCORE_MIN,
)
from holdtrack.fusion.matching import matched_links
from holdtrack.fusion.rf import fuse as _fuse_yolo_rf
from holdtrack.fusion.yolo import fuse as _fuse_yolo
from holdtrack.inputs.mot import physical
from holdtrack.steps.box_offset import render
from holdtrack.steps.features import mot_bytes


@dataclass(frozen=True)
class FusionViews:
    detections: dict[str, DetectionBank | None]
    principal: DetectionBank | None = None
    associations: bytes = b""

    def arrays(self, name: str) -> dict[str, np.ndarray] | None:
        bank = self.detections.get(name)
        return bank.arrays() if bank is not None else None


def fuse(voted: bytes, views: FusionViews, config: RunConfig) -> bytes:
    """Fuse complementary box views while retaining every voted identity and row."""
    if config.fusion_mode == "none":
        return voted
    auxiliary = {
        name: views.arrays(name) if name in config.auxiliary_detectors else None
        for name in ("yolo", "mirror", "dfine")
    }
    if config.fusion_mode == "yolo_views":
        return _fuse_yolo(voted, auxiliary["mirror"], auxiliary["dfine"])[0]
    if config.fusion_mode == "yolo_rf":
        return _fuse_yolo_rf(
            voted, auxiliary["mirror"], auxiliary["dfine"], views.arrays("rf_large")
        )[0]
    if config.fusion_mode != "anchor":
        raise ValueError("Unknown fusion mode")
    fused = _fuse_anchor(
        voted, dict(yolo=auxiliary["yolo"], flip=auxiliary["mirror"], DFINE=auxiliary["dfine"])
    )[0]
    added = [name for name in ("rf_2xl", "rf_large") if name in config.auxiliary_detectors]
    if not added:
        return fused
    extra = views.arrays(added[0])
    if extra is None or views.principal is None or not views.associations:
        raise ValueError("The extra RF view requires the principal scores and association support")
    bank = views.principal.arrays()
    query, provenance = box_geometry.bind_observations(
        bank, mot_bytes(views.associations), mot_bytes(voted)
    )
    if provenance["ambiguous_score_linked_rows"]:
        raise ValueError("An anchor box has an ambiguous source score")
    rows = physical(voted)
    anchor_boxes = np.asarray([row[2:6] for _, row in rows], float).reshape(-1, 4)
    frames = np.asarray([row[0] for _, row in rows], int)
    options = [
        np.asarray([row[2:6] for _, row in physical(fused)], float).reshape(-1, 4),
        anchor_boxes,
    ]
    for name, floor in (
        ("yolo", YOLO_SCORE_MIN),
        ("mirror", YOLO_SCORE_MIN),
        ("dfine", DFINE_SCORE_MIN),
    ):
        aligned = np.full_like(anchor_boxes, np.nan)
        if auxiliary[name] is not None:
            for index, box in matched_links(
                anchor_boxes, frames, auxiliary[name], floor, box_geometry
            ).items():
                aligned[index] = box
        options.append(aligned)
    proposal, _ = _add_view(
        dict(options=np.stack(options), frames=frames, rf_scores=bank["scores"], query=query),
        dict(frame=extra["frames"], boxes=extra["boxes"], score=extra["scores"]),
        dict(anchor_floor=ANCHOR_SCORE_MIN, new_floor=RF_SCORE_MIN, IoU=VIEW_MATCH_IOU),
        config.extra_view_weight,
    )
    return render(fused, proposal)[0]


__all__ = ["FusionViews", "fuse"]
