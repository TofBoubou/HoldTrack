"""Identity chain of HoldTrack: twelve steps that correct the identities of a base tracker, without faces.

    from holdtrack import identity
    identity.run("videos/dancetrack0003", stages="stages")

See chain.py for the inputs and the outputs, appearance.py for the three GPU stages, check.py for the final check
of the late operations, and agree.py for the agreement module alone.
"""

import importlib

_MODULES = (
    "agree",
    "agreement",
    "appearance",
    "chain",
    "check",
    "crops",
    "exchanges",
    "geometry",
    "links",
    "merge",
    "segments",
    "short_tracks",
    "stretches",
    "take_over",
    "unsupported",
    "video",
)


def __getattr__(name):
    if name in _MODULES:
        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def run(folder, stages, fps=20.0, keep_steps=False, threads=2):
    """Run the chain on one video folder; writes identities.npy, attached.npy and operations.json there."""
    return importlib.import_module(".chain", __name__).run(folder, stages, fps, keep_steps, threads)
