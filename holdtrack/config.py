"""Define immutable system and refinement settings."""

from dataclasses import asdict, dataclass
from typing import Literal

Voice = Literal["BASE", "EXCL", "PROT"]
DEFAULT_PRESET = "full"


@dataclass(frozen=True)
class TrackSteps:
    temporal_threshold: float | Literal["KEEP_ALL"] = 0.5
    min_track_length: int = 55
    stitch_max_gap: int = 60
    stitch_max_distance: float = 0.5
    box_offset: tuple[float, float, float, float] = (0.62, -0.12, 0.56, 0.89)
    max_gap: int = 10
    gap_score_min: float = 0.01
    gap_score_max: float = 0.3

    def __post_init__(self) -> None:
        if min(self.min_track_length, self.stitch_max_gap, self.max_gap) < 0:
            raise ValueError("Track lengths and gap limits must be nonnegative")
        if not 0 <= self.gap_score_min < self.gap_score_max <= 1:
            raise ValueError("Gap-filling scores must define an interval in [0, 1]")
        if len(self.box_offset) != 4 or self.stitch_max_distance <= 0:
            raise ValueError("Provide four box offsets and a positive stitching distance")

    @classmethod
    def from_dict(cls, values: dict) -> "TrackSteps":
        settings = dict(values)
        if "box_offset" in settings:
            settings["box_offset"] = tuple(settings["box_offset"])
        return cls(**settings)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RunConfig:
    name: str = "large"
    principal_detector: str = "rf_large"
    voices: tuple[Voice, ...] = ("BASE", "EXCL", "PROT")
    auxiliary_detectors: tuple[str, ...] = ("yolo", "mirror", "dfine", "rf_2xl")
    fusion_mode: Literal["none", "anchor", "yolo_views", "yolo_rf"] = "anchor"
    extra_view_weight: int = 2
    steps: TrackSteps = TrackSteps()
    reliability_head: str = "reliability_large"
    # Detector whose boxes start the tracker passes; None: the principal detector.
    seed_detector: str | None = None

    def __post_init__(self) -> None:
        if not self.voices or len(set(self.voices)) != len(self.voices):
            raise ValueError("Provide distinct memory policies")
        if any(voice not in {"BASE", "EXCL", "PROT"} for voice in self.voices):
            raise ValueError("Unknown memory policy")


def preset(name: str = DEFAULT_PRESET) -> RunConfig:
    """Select a system: full (the base system of the paper), large, 2xl or light."""
    if name == "large":
        return RunConfig(
            steps=TrackSteps(gap_score_min=0.00882734078913927, gap_score_max=0.3026500344276428)
        )
    public_steps = TrackSteps(stitch_max_gap=30, stitch_max_distance=0.7, max_gap=5)
    if name == "full":
        return RunConfig(
            name="full",
            principal_detector="rf_2xl",
            seed_detector="rf_large",
            auxiliary_detectors=("yolo", "mirror", "dfine", "rf_large"),
            reliability_head="reliability_2xl",
            steps=public_steps,
        )
    if name == "2xl":
        return RunConfig(
            name="2xl",
            principal_detector="rf_2xl",
            auxiliary_detectors=("yolo", "mirror", "dfine"),
            reliability_head="reliability_2xl",
            steps=public_steps,
        )
    if name == "light":
        return RunConfig(
            name="light",
            principal_detector="rf_2xl",
            voices=("PROT",),
            auxiliary_detectors=(),
            fusion_mode="none",
            reliability_head="reliability_2xl",
            steps=public_steps,
        )
    raise ValueError("Configuration must be full, large, 2xl or light")
