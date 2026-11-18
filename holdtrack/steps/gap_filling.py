"""Fill short gaps using unused detections."""

from collections import Counter, defaultdict

import numpy as np

from holdtrack.inputs.mot import physical


def grouping(frames):
    out = defaultdict(list)
    for i, frame_index in enumerate(frames):
        out[int(frame_index)].append(i)
    return {frame_index: np.asarray(x, int) for frame_index, x in out.items()}


def requests(rows, max_gap):
    objects = defaultdict(list)
    result = defaultdict(list)
    for row in rows:
        objects[int(row[1])].append(row)
    for pid, sequence in sorted(objects.items()):
        sequence.sort(key=lambda observation: observation[0])
        for left, right in zip(sequence, sequence[1:]):
            gap = int(right[0] - left[0] - 1)
            if not 1 <= gap <= max_gap:
                continue
            for frame_index in range(int(left[0]) + 1, int(right[0])):
                ratio = (frame_index - left[0]) / (right[0] - left[0])
                box = (1 - ratio) * np.array(left[2:6]) + ratio * np.array(right[2:6])
                result[frame_index].append(
                    dict(
                        pid=pid,
                        frame0=frame_index,
                        box=box,
                        left_frame=int(left[0]),
                        right_frame=int(right[0]),
                        gap_length=gap,
                    )
                )
    return result


def coherent_paths(rows, raw, occupied, geo, max_gap=10, score_upper=0.3, *, score_lower=0.01):
    A = raw["boxes"]
    scores = raw["scores"]
    frames = raw["frames"]
    rg = grouping(frames)
    track_boxes = np.array([observation[2:6] for observation in rows])
    bf = np.array([observation[0] for observation in rows], int)
    bg = grouping(bf)
    req = requests(rows, max_gap)
    groups = defaultdict(list)
    counts = Counter()
    for frame_index, observations in req.items():
        for request in observations:
            groups[request["pid"], request["left_frame"], request["right_frame"]].append(request)
    endpoints = {
        (int(observation[0]), int(observation[1])): np.array(observation[2:6])
        for observation in rows
    }
    paths = []
    blind = {
        str(frame_index): [
            dict(
                pid=x["pid"],
                left_frame=x["left_frame"],
                right_frame=x["right_frame"],
                gap_length=x["gap_length"],
                predicted_box=x["box"].tolist(),
            )
            for x in observations
        ]
        for frame_index, observations in req.items()
    }
    for (pid, left, right), observations in sorted(
        groups.items(), key=lambda x: (x[0][1], x[0][2], x[0][0])
    ):
        observations.sort(key=lambda x: x["frame0"])
        layers = [endpoints[left, pid][None]]
        predicted = [endpoints[left, pid]]
        indices = [np.array([-1], int)]
        nomination = [np.array([1.0])]
        counts["gap_episodes"] += 1
        for request in observations:
            frame_index = request["frame0"]
            counts["gap_frame_requests"] += 1
            ri = rg.get(frame_index, np.array([], int))
            ri = ri[
                (scores[ri] >= score_lower)
                & (scores[ri] < score_upper)
                & (A[ri, 2:] > A[ri, :2]).all(1)
            ]
            counts["weak_boxes_before_occupancy"] += len(ri)
            free = np.array([i for i in ri if int(i) not in occupied], int)
            counts["removed_exact_carrier_occupied"] += len(ri) - len(free)
            pi = bg.get(frame_index, np.array([], int))
            other = (
                geo.boxes_iou(A[free], track_boxes[pi]).max(1)
                if len(free) and len(pi)
                else np.zeros(len(free))
            )
            expected = geo.boxes_iou(A[free], request["box"][None])[:, 0]
            counts["nodes_before_usership_gate"] += int((expected >= 0.5).sum())
            allowed = (expected >= 0.5) & (other < 0.7) & (expected > other + 0.1)
            counts["removed_by_relative_neighbor_usership"] += int(
                ((expected >= 0.5) & ~allowed).sum()
            )
            free = geo.canonical(A, free[allowed])
            layers.append(A[free])
            indices.append(free)
            predicted.append(request["box"])
            nomination.append(geo.boxes_iou(A[free], request["box"][None])[:, 0])
        layers.append(endpoints[right, pid][None])
        indices.append(np.array([-1], int))
        predicted.append(endpoints[right, pid])
        nomination.append(np.array([1.0]))
        score = np.array([0.0])
        back = []
        possible = True
        for k in range(1, len(layers)):
            if not len(layers[k]) or not len(layers[k - 1]):
                possible = False
                break
            previous_center = (predicted[k - 1][:2] + predicted[k - 1][2:]) / 2
            center = (predicted[k][:2] + predicted[k][2:]) / 2
            translated = layers[k - 1] + np.tile(center - previous_center, 2)
            edge = geo.boxes_iou(translated, layers[k])
            value = score[:, None] + edge + nomination[k][None]
            value[edge < 0.5] = -np.inf
            parent = np.argmax(value, axis=0)
            score = value[parent, np.arange(value.shape[1])]
            back.append(parent)
            if not np.isfinite(score).any():
                possible = False
                break
        if not possible or not np.isfinite(score[0]):
            counts["abstained_no_complete_owned_path"] += 1
            continue
        chosen = [0]
        for parent in reversed(back):
            chosen.append(int(parent[chosen[-1]]))
        chosen.reverse()
        selected = []
        for k, request in enumerate(observations, 1):
            idx = int(indices[k][chosen[k]])
            frame_index = request["frame0"]
            pi = bg.get(frame_index, np.array([], int))
            contact = (
                float(geo.boxes_iou(A[idx : idx + 1], track_boxes[pi]).max()) if len(pi) else 0.0
            )
            selected.append(
                dict(
                    pid=pid,
                    frame0=frame_index,
                    left_frame=left,
                    right_frame=right,
                    gap_length=right - left - 1,
                    raw_bank_row=idx,
                    raw_score=float(scores[idx]),
                    raw_box=A[idx].tolist(),
                    nomination_IoU=float(nomination[k][chosen[k]]),
                    predicted_contact_with_reference=contact,
                )
            )
        paths.append(selected)
        counts["complete_owned_paths_before_conflicts"] += 1
    byframe = defaultdict(list)
    for k, path in enumerate(paths):
        for record in path:
            byframe[record["frame0"]].append((k, record))
    conflict = set()
    for records in byframe.values():
        for a in range(len(records)):
            for b in range(a + 1, len(records)):
                i, x = records[a]
                j, y = records[b]
                if i == j:
                    continue
                if (
                    x["raw_bank_row"] == y["raw_bank_row"]
                    or geo.boxes_iou(np.array(x["raw_box"])[None], np.array(y["raw_box"])[None])[
                        0, 0
                    ]
                    >= 0.7
                ):
                    conflict.update([i, j])
    counts["whole_paths_abstained_due_to_conflict"] = len(conflict)
    add = [x for i, path in enumerate(paths) if i not in conflict for x in path]
    return (add, counts, blind)


def emit_additions(raw, additions):
    existing, inserted = (defaultdict(list), defaultdict(list))
    for line, row in physical(raw):
        existing[int(row[0])].append(line)
    for a in additions:
        x, y, right, bottom = a["raw_box"]
        inserted[a["frame0"]].append(
            f"{a['frame0'] + 1},{a['pid']},{x:.8f},{y:.8f},{right - x:.8f},{bottom - y:.8f},1,-1,-1,-1\n".encode()
        )
    return b"".join(
        line
        for frame_index in sorted(set(existing) | set(inserted))
        for line in existing[frame_index] + inserted[frame_index]
    )
