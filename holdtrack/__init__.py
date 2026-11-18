"""Run HoldTrack from frames with resumable stages."""

import importlib


def __getattr__(name):
    """``holdtrack.run`` and ``holdtrack.sam`` are imported when they are first read, so that a
    stage run alone (``python -m holdtrack.boxes.pipeline``) needs only its own requirements."""
    if name == "run":
        return importlib.import_module("holdtrack.pipeline").run
    if name == "sam":
        return importlib.import_module("holdtrack.sam")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__version__ = "0.1.0"
__all__ = ["run", "sam"]
