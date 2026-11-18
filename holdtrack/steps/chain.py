"""Apply reliability, length, stitching, offset and filling in order."""

from dataclasses import dataclass

import numpy as np

from holdtrack import geometry
from holdtrack.config import TrackSteps
from holdtrack.inputs.mot import physical
from holdtrack.steps import reliability
from holdtrack.steps.box_offset import render
from holdtrack.steps.gap_filling import coherent_paths, emit_additions
from holdtrack.steps.short_tracks import retained_observations
from holdtrack.steps.stitching import stitch


@dataclass(frozen=True)
class RefinementInputs:
    features: np.ndarray
    identities: np.ndarray
    source_indices: np.ndarray
    detection_frames: np.ndarray
    detection_boxes: np.ndarray
    detection_scores: np.ndarray


def apply_stitching(source: bytes, settings: TrackSteps) -> bytes:
    observations = physical(source)
    if not settings.stitch_max_gap or not observations:
        return source
    linked, _ = stitch(
        {"sequence": [observation for _, observation in observations]},
        settings.stitch_max_gap,
        settings.stitch_max_distance,
    )
    chunks = []
    for (line, _), linked_observation in zip(observations, linked["sequence"]):
        columns = line.split(b",")
        columns[1] = str(int(linked_observation[1])).encode()
        chunks.append(b",".join(columns))
    return b"".join(chunks)


def apply_box_offset(source: bytes, settings: TrackSteps) -> bytes:
    observations = physical(source)
    offset = np.asarray(settings.box_offset)
    if not observations or not np.any(offset):
        return source
    boxes = np.asarray([observation[2:6] for _, observation in observations])
    corrected = boxes - offset
    invalid = (corrected[:, 2:] <= corrected[:, :2]).any(1) | ~np.isfinite(corrected).all(1)
    corrected[invalid] = boxes[invalid]
    return render(source, corrected)[0]


def apply_gap_filling(
    source: bytes, inputs: RefinementInputs, retained: np.ndarray, settings: TrackSteps
) -> bytes:
    if not settings.max_gap or not source:
        return source
    detections = dict(
        frames=inputs.detection_frames,
        boxes=inputs.detection_boxes - np.asarray(settings.box_offset),
        scores=inputs.detection_scores,
    )
    additions, _, _ = coherent_paths(
        [observation for _, observation in physical(source)],
        detections,
        set(map(int, inputs.source_indices[retained])),
        geometry,
        settings.max_gap,
        settings.gap_score_max,
        score_lower=settings.gap_score_min,
    )
    return emit_additions(source, additions)


def refine(
    source: bytes,
    inputs: RefinementInputs,
    head: dict,
    settings: TrackSteps,
    probabilities: np.ndarray | None = None,
) -> tuple[bytes, dict]:
    """Refine one sequence with the same functions used by the resumable pipeline."""
    probabilities = (
        reliability.predict(head, inputs.features) if probabilities is None else probabilities
    )
    reliable = reliability.keep_mask(probabilities, settings.temporal_threshold)
    retained = retained_observations(inputs.identities, reliable, settings.min_track_length)
    payload = reliability.subset(source, retained)
    payload = apply_stitching(payload, settings)
    payload = apply_box_offset(payload, settings)
    payload = apply_gap_filling(payload, inputs, retained, settings)
    observations = physical(payload)
    if len({(observation[0], observation[1]) for _, observation in observations}) != len(
        observations
    ):
        raise ValueError("Track observations must have unique frame and identity pairs")
    return (
        payload,
        dict(
            deleted=int((~retained).sum()),
            changed=len(set(payload.splitlines()) ^ set(source.splitlines())),
        ),
    )
