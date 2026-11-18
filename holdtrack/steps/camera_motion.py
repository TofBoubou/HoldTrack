"""Fill the gaps of an identity while the camera moves.

The motion of the background between consecutive frames is given as one 2x3 matrix per
frame. A gap between two boxes of an identity is filled by carrying each of the two boxes
to the missing frame with the motion of the background, then interpolating between the two
carried boxes. With a fixed camera this is a linear interpolation.
"""

import numpy as np

IDENTITY = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def cumulative_motion(frames, matrices, counts=None, minimum_points=15, scale_tolerance=0.2):
    """Return {frame: 3x3 matrix from the first frame to this frame}.

    matrices[i] carries the background from the frame before frames[i] to frames[i]. An
    estimate from fewer than `minimum_points` background points, or whose scale is farther
    than `scale_tolerance` from one, is replaced by no motion.
    """
    frames = np.asarray(frames)
    if counts is None:
        counts = np.full(len(frames), 10**6)
    out = {}
    current = np.eye(3)
    for frame, matrix, count in zip(frames, matrices, counts):
        scale = np.sqrt(abs(np.linalg.det(matrix[:, :2])))
        if count < minimum_points or abs(scale - 1) > scale_tolerance:
            matrix = IDENTITY
        current = np.vstack([matrix, [0, 0, 1]]) @ current if frame != frames[0] else np.eye(3)
        out[int(frame)] = current
    return out


def estimate_motion(read_frame, first, last, rows=None, margin=3):
    """Estimate the motion of the background between consecutive frames.

    read_frame(frame) returns the frame as a gray image. rows: (m, 6+) frame, identity, x,
    y, width, height of the tracked persons, whose boxes are left out of the background.
    Corners of the background are followed from each frame to the next by optical flow at
    half resolution, and a similarity (scale, rotation, translation) is fitted to them with
    RANSAC. Returns the frames, one 2x3 matrix per frame that carries the frame before to
    this frame, and the number of points kept; the first frame has no motion.
    """
    import cv2

    frames = np.arange(first, last + 1)
    matrices = np.repeat(IDENTITY[None], len(frames), 0)
    counts = np.zeros(len(frames), int)
    previous = None
    for index, frame in enumerate(frames):
        image = cv2.resize(
            read_frame(int(frame)), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA
        )
        if previous is not None:
            background = np.full(previous.shape, 255, np.uint8)
            if rows is not None:
                for row in rows[rows[:, 0] == frame - 1]:
                    x, y, width, height = row[2:6] / 2
                    top, left = max(0, int(y - margin)), max(0, int(x - margin))
                    background[top : int(y + height + margin), left : int(x + width + margin)] = 0
            corners = cv2.goodFeaturesToTrack(
                previous,
                maxCorners=1000,
                qualityLevel=0.01,
                minDistance=4,
                blockSize=3,
                mask=background,
            )
            if corners is not None and len(corners) >= 8:
                moved, status, _ = cv2.calcOpticalFlowPyrLK(
                    previous, image, corners, None, winSize=(21, 21), maxLevel=3
                )
                followed = status[:, 0] == 1
                if followed.sum() >= 8:
                    matrix, inliers = cv2.estimateAffinePartial2D(
                        corners[followed] * 2,
                        moved[followed] * 2,
                        method=cv2.RANSAC,
                        ransacReprojThreshold=2.0,
                    )
                    if matrix is not None:
                        matrices[index] = matrix
                        counts[index] = int(inliers.sum())
        previous = image
    return frames, matrices, counts


def save_motion(path, frames, matrices, counts):
    """Write a camera-motion file that `load_motion` reads."""
    np.savez(path, frame=np.asarray(frames), M=np.asarray(matrices), n=np.asarray(counts))


def load_motion(path):
    """Read a camera-motion file: arrays `frame`, `M` (n, 2, 3) and optionally `n`."""
    data = np.load(path)
    counts = data["n"] if "n" in data.files else None
    return cumulative_motion(data["frame"], data["M"], counts)


def transport(motion, source, target, box):
    """Carry a box (x1, y1, x2, y2) from frame `source` to frame `target`."""
    matrix = motion[target] @ np.linalg.inv(motion[source])
    scale = np.sqrt(abs(np.linalg.det(matrix[:2, :2])))
    center = matrix @ np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, 1.0])
    width, height = (box[2] - box[0]) * scale, (box[3] - box[1]) * scale
    return np.array(
        [
            center[0] - width / 2,
            center[1] - height / 2,
            center[0] + width / 2,
            center[1] + height / 2,
        ]
    )


def fill_gaps(frames, identities, boxes, max_gap, motion):
    """Return the frames, identities and boxes that fill the gaps of at most `max_gap` frames."""
    new_frames, new_identities, new_boxes = [], [], []
    for identity in np.unique(identities):
        rows = np.where(identities == identity)[0]
        rows = rows[np.argsort(frames[rows], kind="stable")]
        times = frames[rows]
        step = times[1:] - times[:-1]
        for k in np.where((step >= 2) & (step <= max_gap + 1))[0]:
            left, right = boxes[rows[k]], boxes[rows[k + 1]]
            missing = int(times[k + 1] - times[k] - 1)
            for offset in range(1, missing + 1):
                frame = int(times[k] + offset)
                ratio = offset / (missing + 1)
                new_frames.append(frame)
                new_identities.append(int(identity))
                new_boxes.append(
                    (1 - ratio) * transport(motion, int(times[k]), frame, left)
                    + ratio * transport(motion, int(times[k + 1]), frame, right)
                )
    return (
        np.array(new_frames, np.int64),
        np.array(new_identities, np.int64),
        np.array(new_boxes, float).reshape(-1, 4),
    )
