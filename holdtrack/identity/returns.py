"""Decide again the identity of a person who comes back after an absence.

A track that disappears and comes back is not always on the same person. Two moves decide
each return again, from the tracks of the video alone.

`cut_returns` cuts. An identity keeps its name across an absence only if it comes back soon
and close to a place that the person could have reached: the place where it was last seen,
or the place given by extending its motion over the absence, forward from before it or
backward from after it. Camera motion is removed.

`link_returns` links. A classifier scores the pairs of tracklets in which the second starts
after the first has ended. Its training pairs are made from the tracks themselves: a
tracklet cut at a random gap gives a true pair, and each side of the gap with another
tracklet gives a false pair. Appearance is compared in a space fitted on the video, in
which the variations inside a tracklet are whitened and the tracklets are separated.

Neither move reads an annotation, and neither changes a box.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression

from holdtrack.steps.camera_motion import transport

REST_OF_TRACKLET = -3
LEFT_SIDE = -1
RIGHT_SIDE = -2
NO_ROWS = np.zeros(0, int)


@dataclass
class ReturnVideo:
    """Tracks of one video and the appearance features of its crops.

    frames, identities: one value per row, frames numbered as in `motion`. boxes: (N, 4) as
    x1, y1, x2, y2. motion: {frame: 3x3 matrix}, see `holdtrack.steps.camera_motion`.
    crop_rows: for each crop, the row it was cut from. features: one vector per crop.
    """

    frames: np.ndarray
    identities: np.ndarray
    boxes: np.ndarray
    fps: float
    width: int
    height: int
    motion: dict
    crop_rows: np.ndarray
    features: np.ndarray


def unit(vectors):
    return vectors / (np.linalg.norm(vectors, axis=-1, keepdims=True) + 1e-9)


def _center(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _velocity(motion, frames, boxes, rows, reference):
    """Velocity of the box center over `rows`, in pixels per frame, seen from `reference`."""
    if len(rows) < 3:
        return np.zeros(2)
    centers = np.array(
        [_center(transport(motion, int(frames[row]), reference, boxes[row])) for row in rows]
    )
    time = frames[rows].astype(float)
    time = time - time.mean()
    return (time[:, None] * (centers - centers.mean(0))).sum(0) / max((time**2).sum(), 1e-9)


def cut_returns(frames, identities, boxes, fps, motion, limit=0.5, longest=2.0, shortest=0.1):
    """Return the identities after the cuts, and the number of cuts.

    An identity is split into segments at every absence longer than `shortest` seconds. Two
    segments stay together when the second starts at most `longest` seconds after the end of
    the first and within `limit` body heights of one of three places: where the first
    ended, where the motion of the first leads over the absence, and where the motion of the
    second leads back. A segment that no earlier segment reaches takes a new identity,
    larger than every identity of the video.
    """
    frames = np.asarray(frames).astype(np.int64)
    identities = np.asarray(identities).astype(np.int64)
    boxes = np.asarray(boxes, float)
    out = identities.copy()
    if not len(identities):
        return out, 0
    next_identity = int(identities.max()) + 1
    cuts = 0
    half_second = max(2, int(round(fps / 2)))
    for identity in np.unique(identities):
        rows = np.where(identities == identity)[0]
        rows = rows[np.argsort(frames[rows], kind="stable")]
        breaks = np.where(np.diff(frames[rows]) - 1 > shortest * fps)[0]
        if not len(breaks):
            continue
        parts = np.split(rows, breaks + 1)
        segments = []
        for part in parts:
            start, end = int(frames[part[0]]), int(frames[part[-1]])
            segments.append(
                {
                    "start": start,
                    "end": end,
                    "first_box": boxes[part[0]],
                    "last_box": boxes[part[-1]],
                    "velocity_start": _velocity(motion, frames, boxes, part[:half_second], start),
                    "velocity_end": _velocity(motion, frames, boxes, part[-half_second:], end),
                }
            )
        root = list(range(len(parts)))

        def find(index):
            while root[index] != index:
                index = root[index]
            return index

        for later in range(1, len(parts)):
            for earlier in range(later):
                first, second = segments[earlier], segments[later]
                gap = second["start"] - first["end"]
                if gap > longest * fps:
                    continue
                carried = transport(motion, first["end"], second["start"], first["last_box"])
                height = max(
                    (carried[3] - carried[1] + second["first_box"][3] - second["first_box"][1]) / 2,
                    1.0,
                )
                left = np.array(_center(carried))
                right = np.array(_center(second["first_box"]))
                turn = motion[second["start"]] @ np.linalg.inv(motion[first["end"]])
                velocity = turn[:2, :2] @ first["velocity_end"]
                distance = (
                    min(
                        np.linalg.norm(left - right),
                        np.linalg.norm(left + velocity * gap - right),
                        np.linalg.norm(right - second["velocity_start"] * gap - left),
                    )
                    / height
                )
                if distance <= limit and find(earlier) != find(later):
                    root[find(later)] = find(earlier)
        new_identity = {}
        for index, part in enumerate(parts):
            group = find(index)
            if group == 0:
                continue
            if group not in new_identity:
                new_identity[group] = next_identity
                next_identity += 1
                cuts += 1
            out[part] = new_identity[group]
    return out, cuts


def discriminant_space(features, labels, dimensions=32):
    """Project the features where the labelled groups are separated.

    The variations inside the groups of at least four crops are whitened, then the features
    are projected on the directions that separate the means of the groups. Crops labelled
    `REST_OF_TRACKLET` belong to no group.
    """
    groups = [g for g in np.unique(labels) if g != REST_OF_TRACKLET and (labels == g).sum() >= 4]
    if len(groups) < 2:
        return unit(features)
    residual = np.concatenate(
        [features[labels == g] - features[labels == g].mean(0) for g in groups]
    )
    scatter = residual.T @ residual / len(residual)
    scatter = 0.5 * scatter + 0.5 * np.trace(scatter) / len(scatter) * np.eye(len(scatter))
    values, vectors = np.linalg.eigh(scatter)
    whitened = features @ ((vectors / np.sqrt(np.maximum(values, 1e-9))) @ vectors.T)
    means = np.array([whitened[labels == g].mean(0) for g in groups])
    means = means - means.mean(0)
    _, _, directions = np.linalg.svd(means, full_matrices=False)
    return unit(whitened @ directions[:dimensions].T)


def _similarity(space, first, second):
    if not len(first) or not len(second):
        return np.nan
    return float(unit(space[first].mean(0)) @ unit(space[second].mean(0)))


def _at_border(box, width, height):
    return bool(box[0] <= 2 or box[1] <= 2 or box[2] >= width - 2 or box[3] >= height - 2)


def describe(video, rows):
    """Ends, boxes and motion of the tracklet made of `rows`."""
    frames, boxes = video.frames, video.boxes
    rows = rows[np.argsort(frames[rows], kind="stable")]
    half_second = max(2, int(round(video.fps / 2)))
    start, end = int(frames[rows[0]]), int(frames[rows[-1]])
    return {
        "start": start,
        "end": end,
        "first_box": boxes[rows[0]],
        "last_box": boxes[rows[-1]],
        "starts_at_border": _at_border(boxes[rows[0]], video.width, video.height),
        "ends_at_border": _at_border(boxes[rows[-1]], video.width, video.height),
        "velocity_start": _velocity(video.motion, frames, boxes, rows[:half_second], start),
        "velocity_end": _velocity(video.motion, frames, boxes, rows[-half_second:], end),
    }


def measures(video, first, second):
    """Seven measures of position, motion and size of a pair of tracklets.

    Distances are in body heights, after the last box of `first` is carried to the first
    frame of `second` with the motion of the camera: distance between the two ends, distance
    to the place where each tracklet leads the other, the smaller of these two, cosine of
    the two velocities, speed of the first in heights per second, and the absolute log ratio
    of the heights.
    """
    fps, motion = video.fps, video.motion
    gap = second["start"] - first["end"]
    carried = transport(motion, first["end"], second["start"], first["last_box"])
    second_box = second["first_box"]
    height = max((carried[3] - carried[1] + second_box[3] - second_box[1]) / 2, 1)
    left = np.array(_center(carried))
    right = np.array(_center(second_box))
    turn = motion[second["start"]] @ np.linalg.inv(motion[first["end"]])
    velocity = turn[:2, :2] @ first["velocity_end"]
    distance = float(np.linalg.norm(left - right) / height)
    forward = float(np.linalg.norm(left + velocity * gap - right) / height)
    backward = float(np.linalg.norm(right - second["velocity_start"] * gap - left) / height)
    cosine = float(
        velocity
        @ second["velocity_start"]
        / (np.linalg.norm(velocity) * np.linalg.norm(second["velocity_start"]) + 1e-9)
    )
    height_ratio = float((second_box[3] - second_box[1]) / max(carried[3] - carried[1], 1))
    return [
        min(distance, 5),
        min(forward, 5),
        min(backward, 5),
        min(forward, backward, 5),
        cosine,
        min(float(np.linalg.norm(velocity)) * fps / height, 3),
        abs(np.log(max(height_ratio, 1e-3))),
    ]


def _with_appearance(row, similarity):
    """Append the appearance similarity, 0.5 when it is missing, and the missing flag."""
    missing = np.isnan(similarity)
    return row + [0.5 if missing else similarity, float(missing)]


def _contact(frames, boxes):
    """Largest IoU of each box with another box of its frame."""
    out = np.zeros(len(frames))
    for frame in np.unique(frames):
        rows = np.where(frames == frame)[0]
        if len(rows) > 1:
            b = boxes[rows]
            width = np.clip(
                np.minimum(b[:, None, 2], b[None, :, 2]) - np.maximum(b[:, None, 0], b[None, :, 0]),
                0,
                None,
            )
            height = np.clip(
                np.minimum(b[:, None, 3], b[None, :, 3]) - np.maximum(b[:, None, 1], b[None, :, 1]),
                0,
                None,
            )
            area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
            iou = width * height / (area[:, None] + area[None] - width * height + 1e-9)
            np.fill_diagonal(iou, 0)
            out[rows] = iou.max(1)
    return out


def _prepare(video, rng, reach, exclude_border, dimensions):
    """Candidate pairs of a video with their measures, and its made-up training pairs."""
    frames = np.asarray(video.frames).astype(np.int64)
    identities = np.asarray(video.identities).astype(np.int64)
    fps = video.fps
    names = [int(identity) for identity in np.unique(identities)]
    tracklets = {name: describe(video, np.where(identities == name)[0]) for name in names}
    crop_rows = np.asarray(video.crop_rows).astype(int)
    crop_names = identities[crop_rows]
    crop_frames = frames[crop_rows]
    features = np.asarray(video.features).astype(np.float32)
    features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-9)
    centered = features - features.mean(0)
    _, _, components = np.linalg.svd(centered, full_matrices=False)
    reduced = unit(centered @ components[:256].T)
    space = discriminant_space(reduced, crop_names, dimensions)
    last_crops, first_crops = {}, {}
    for name in names:
        crops = np.where(crop_names == name)[0]
        if len(crops):
            last_crops[name] = crops[crop_frames[crops] >= crop_frames[crops].max() - 2 * fps]
            first_crops[name] = crops[crop_frames[crops] <= crop_frames[crops].min() + 2 * fps]
    candidates = [
        (a, c)
        for a in names
        for c in names
        if a != c
        and 0 < tracklets[c]["start"] - tracklets[a]["end"] <= reach * fps
        and not (
            exclude_border and (tracklets[a]["ends_at_border"] or tracklets[c]["starts_at_border"])
        )
    ]
    rows = [
        _with_appearance(
            measures(video, tracklets[a], tracklets[c]),
            _similarity(space, last_crops.get(a, NO_ROWS), first_crops.get(c, NO_ROWS)),
        )
        for a, c in candidates
    ]
    contact = _contact(frames, video.boxes)
    examples, labels = [], []
    for name in names:
        own = np.where(identities == name)[0]
        own = own[np.argsort(frames[own], kind="stable")]
        if frames[own[-1]] - frames[own[0]] < 3 * fps:
            continue
        for _ in range(4):
            gap = rng.uniform(1 / fps, reach) * fps
            inside = own[
                (frames[own] >= frames[own[0]] + 0.2 * fps)
                & (frames[own] <= frames[own[-1]] - 0.2 * fps - gap)
            ]
            if not len(inside):
                continue
            touching = inside[contact[inside] >= 0.3]
            cut = frames[rng.choice(touching if len(touching) else inside)]
            left_length = rng.uniform(0.2, 5) * fps
            right_length = rng.uniform(0.2, 5) * fps
            left_rows = own[(frames[own] <= cut) & (frames[own] >= cut - left_length)]
            right_rows = own[(frames[own] >= cut + gap) & (frames[own] <= cut + gap + right_length)]
            if len(left_rows) < 2 or len(right_rows) < 2:
                continue
            left, right = describe(video, left_rows), describe(video, right_rows)
            left_crops = np.where(np.isin(crop_rows, left_rows))[0]
            right_crops = np.where(np.isin(crop_rows, right_rows))[0]
            sides = crop_names.copy()
            sides[crop_names == name] = REST_OF_TRACKLET
            sides[left_crops] = LEFT_SIDE
            sides[right_crops] = RIGHT_SIDE
            cut_space = discriminant_space(reduced, sides, dimensions)
            if len(left_crops):
                recent = crop_frames[left_crops] >= crop_frames[left_crops].max() - 2 * fps
                left_crops = left_crops[recent]
            if len(right_crops):
                early = crop_frames[right_crops] <= crop_frames[right_crops].min() + 2 * fps
                right_crops = right_crops[early]
            examples.append(
                _with_appearance(
                    measures(video, left, right), _similarity(cut_space, left_crops, right_crops)
                )
            )
            labels.append(1)
            for other in names:
                if other == name:
                    continue
                if 0 < tracklets[other]["start"] - left["end"] <= reach * fps:
                    similarity = _similarity(cut_space, left_crops, first_crops.get(other, NO_ROWS))
                    examples.append(
                        _with_appearance(measures(video, left, tracklets[other]), similarity)
                    )
                    labels.append(0)
                if 0 < right["start"] - tracklets[other]["end"] <= reach * fps:
                    similarity = _similarity(cut_space, last_crops.get(other, NO_ROWS), right_crops)
                    examples.append(
                        _with_appearance(measures(video, tracklets[other], right), similarity)
                    )
                    labels.append(0)
    return {
        "frames": frames,
        "identities": identities,
        "names": names,
        "tracklets": tracklets,
        "candidates": candidates,
        "rows": np.array(rows).reshape(len(candidates), 9),
        "examples": np.array(examples).reshape(len(examples), 9),
        "labels": np.array(labels, int),
    }


def _apply_links(data, scores, threshold):
    """Link the candidate pairs scored at least `threshold`, one to one, and rename."""
    frames, identities, names = data["frames"], data["identities"], data["names"]
    tracklets = data["tracklets"]
    score = dict(zip(data["candidates"], scores.tolist()))
    firsts = sorted({a for a, _ in score})
    seconds = sorted({c for _, c in score})
    table = np.full((len(firsts), len(seconds)), -1.0)
    for (a, c), value in score.items():
        table[firsts.index(a), seconds.index(c)] = value
    rows, columns = linear_sum_assignment(-table) if table.size else ([], [])
    links = [(firsts[i], seconds[j]) for i, j in zip(rows, columns) if table[i, j] >= threshold]
    root = {name: name for name in names}

    def find(name):
        while root[name] != name:
            name = root[name]
        return name

    made = []
    for a, c in sorted(links, key=lambda link: tracklets[link[1]]["start"]):
        group_a, group_c = find(a), find(c)
        if group_a == group_c:
            continue
        frames_a = set(
            frames[np.isin(identities, [n for n in names if find(n) == group_a])].tolist()
        )
        frames_c = set(
            frames[np.isin(identities, [n for n in names if find(n) == group_c])].tolist()
        )
        if frames_a & frames_c:
            continue
        root[group_c] = group_a
        made.append((int(a), int(c)))
    return np.array([find(name) for name in identities.tolist()]), made


def link_returns(videos, threshold=0.5, reach=10.0, exclude_border=True, dimensions=32, seed=0):
    """Return {video: identities after the links} and {video: links made}.

    videos: {name: ReturnVideo}, identities already cut by `cut_returns`. A pair of
    tracklets is a candidate when the second starts within `reach` seconds of the end of the
    first and, with `exclude_border`, when the first does not end and the second does not
    start at the image border. One classifier is fitted on the made-up pairs of all the
    videos, taken in the order of their names; the candidates scored at least `threshold`
    are linked one to one, and a link that would give an identity two boxes in a frame is
    dropped.
    """
    rng = np.random.default_rng(seed)
    order = sorted(videos)
    prepared = {
        name: _prepare(videos[name], rng, reach, exclude_border, dimensions) for name in order
    }
    examples = np.concatenate([prepared[name]["examples"] for name in order])
    labels = np.concatenate([prepared[name]["labels"] for name in order])
    mean = examples.mean(0)
    deviation = examples.std(0) + 1e-9
    model = LogisticRegression(C=1.0, max_iter=2000).fit((examples - mean) / deviation, labels)
    identities, links = {}, {}
    for name in order:
        rows = prepared[name]["rows"]
        scores = model.predict_proba((rows - mean) / deviation)[:, 1] if len(rows) else np.zeros(0)
        identities[name], links[name] = _apply_links(prepared[name], scores, threshold)
    return identities, links
