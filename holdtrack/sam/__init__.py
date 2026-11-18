"""Run one tracker with a selected memory policy."""

from pathlib import Path

from holdtrack.config import Voice


def run_once(
    frames: str | Path,
    detections: str | Path,
    voice: Voice = "EXCL",
    *,
    checkpoint: str | Path | None = None,
    device: str = "cuda",
    masks: str | Path | None = None,
) -> bytes:
    """Return one SAM trajectory on the supplied detector support in MOT format."""
    from holdtrack.sam._sequence import _run_from_frames

    return _run_from_frames(
        frames, detections, voice, checkpoint=checkpoint, device=device, masks=masks
    )


__all__ = ["run_once"]
