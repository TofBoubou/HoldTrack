"""Load the public SAM tracking weights strictly."""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
from typing import TYPE_CHECKING

from holdtrack.sam import runtime

if TYPE_CHECKING:
    from sam3.model.sam3_tracking_predictor import Sam3TrackerPredictor

    from holdtrack.sam.lifecycle import SourceBindings


def checkpoint_subset(target, checkpoint):
    if "model" in checkpoint and isinstance(checkpoint["model"], dict):
        checkpoint = checkpoint["model"]
    selected, used = ({}, set())
    for key, value in target.items():
        source = (
            "detector." + key if key.startswith("backbone.vision_backbone.") else "tracker." + key
        )
        if source not in checkpoint or tuple(checkpoint[source].shape) != tuple(value.shape):
            raise ValueError(f"SAM3 strict checkpoint coverage missing/wrong-shaped: {key}")
        selected[key] = checkpoint[source]
        used.add(source)
    unused_tracker = [k for k in checkpoint if k.startswith("tracker.") and k not in used]
    if unused_tracker:
        raise ValueError(
            "Official tracker tensors not represented by the instantiated architecture"
        )
    return (
        selected,
        dict(
            target_tensor_count=len(target),
            loaded_tensor_count=len(selected),
            unused_tracker_tensor_count=0,
            unused_detector_tensor_count=len(checkpoint) - len(used),
            mapping="tracker.* plus detector.backbone.vision_backbone.*",
        ),
    )


def strict_load(predictor, checkpoint):
    selected, report = checkpoint_subset(predictor.state_dict(), checkpoint)
    result = predictor.load_state_dict(selected, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError("SAM3 strict state_dict loading failed")
    return dict(report, strict=True, missing_keys=[], unexpected_keys=[])


def load_tracker(
    checkpoint: str | Path,
    sha256: str,
    sources: SourceBindings,
    paths: tuple[Path, ...] | list[Path],
    *,
    device: str = "cuda",
    mirror: bool = False,
) -> tuple[Sam3TrackerPredictor, dict]:
    if device not in ["cpu", "cuda"]:
        raise ValueError("Explicit CPU/CUDA device required")
    sources.validate()
    runtime.configure_sources(sources)
    checkpoint = Path(checkpoint)
    with checkpoint.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if len(sha256) != 64 or actual != sha256:
        raise ValueError("Local checkpoint SHA mismatch")
    import torch

    builder = importlib.import_module("sam3.model_builder")
    if Path(builder.__file__).resolve() != (sources.sam3_root / "sam3/model_builder.py").resolve():
        raise ValueError("Wrong official SAM3 installation")
    predictor = builder.build_tracker(
        apply_temporal_disambiguation=False, with_backbone=True, compile_mode=None
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    report = strict_load(predictor, state)
    del state
    predictor = predictor.eval().to(device=device)
    misc = importlib.import_module("sam2.utils.misc")
    if Path(misc.__file__).resolve() != (sources.sam2_predictor.parent / "utils/misc.py").resolve():
        raise ValueError("Wrong pinned fill-holes implementation")
    adaptation = runtime.install(
        predictor,
        paths=paths,
        fill_holes=misc.fill_holes_in_mask_scores,
        torch=torch,
        mirror=mirror,
    )
    options = runtime.validate_options(predictor)
    return (
        predictor,
        dict(strict=report, adaptation=adaptation, options=options, checkpoint_SHA256=sha256),
    )
