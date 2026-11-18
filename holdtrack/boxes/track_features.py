"""Features of a box read along its track and among the boxes of its frame.

A detector can return the visible part of a person who is partly hidden, where the
annotation keeps the whole body. Three kinds of features tell the corrector how large the
person is: the size of the track in the seconds around the box, the height that the scene
gives at the position of the feet, and the share of the box covered by another box. No
annotation is read.
"""

import numpy as np

COLUMNS = 19


def fit_perspective(bottoms, heights, identities):
    """Fit height = a * bottom + b on the tallest boxes of every identity; return (a, b).

    The tallest boxes of an identity are those at least 0.9 times its 0.9 quantile of
    height. Five rounds reject the boxes farther than 2.5 robust standard deviations.
    """
    bottoms = np.asarray(bottoms, float)
    heights = np.asarray(heights, float)
    identities = np.asarray(identities)
    high = np.zeros(len(heights))
    for identity in np.unique(identities):
        own = identities == identity
        high[own] = np.quantile(heights[own], 0.9)
    tall = heights >= 0.9 * high
    if tall.sum() < 20 or np.ptp(bottoms[tall]) < 1:
        return 0.0, float(np.median(heights[tall])) if tall.any() else 0.0
    a, b = np.polyfit(bottoms[tall], heights[tall], 1)
    for _ in range(5):
        residual = heights[tall] - (a * bottoms[tall] + b)
        spread = 1.4826 * np.median(np.abs(residual - np.median(residual))) + 1e-6
        keep = np.abs(residual) <= 2.5 * spread
        if keep.sum() < 20:
            break
        a, b = np.polyfit(bottoms[tall][keep], heights[tall][keep], 1)
    return float(a), float(b)


def track_features(frames, identities, boxes, fps):
    """Return the features (N, 19) of the boxes (N, 4), given as x1, y1, x2, y2.

    Columns 0 to 9: over the boxes of the identity within 1 s, then within 3 s, the median
    and the maximum of the log height and of the log width minus those of the box, and the
    number of boxes. Columns 10 to 12: within 0.5 s, the median center, bottom and top
    minus those of the box, in box sizes. Columns 13 and 14: the 0.9 quantile of the log
    height and of the log width within 5 s, minus those of the box. Column 15: the log of
    the height that the scene gives at the bottom of the box, minus the log height of the
    box. Columns 16 to 18: the largest share of the box covered by another box of the
    frame, and the bottom and top of that box minus those of the box, in box heights.
    """
    frames = np.asarray(frames).astype(float)
    identities = np.asarray(identities)
    boxes = np.asarray(boxes, float)
    width = np.maximum(boxes[:, 2] - boxes[:, 0], 1.0)
    height = np.maximum(boxes[:, 3] - boxes[:, 1], 1.0)
    log_width, log_height = np.log(width), np.log(height)
    center = (boxes[:, 0] + boxes[:, 2]) / 2
    out = np.full((len(frames), COLUMNS), np.nan, np.float32)
    for identity in np.unique(identities):
        rows = np.where(identities == identity)[0]
        rows = rows[np.argsort(frames[rows])]
        time = frames[rows]
        for block, window in enumerate((1.0, 3.0)):
            low = np.searchsorted(time, time - window * fps, "left")
            high = np.searchsorted(time, time + window * fps, "right")
            for j in range(len(rows)):
                near = rows[low[j] : high[j]]
                out[rows[j], 5 * block : 5 * block + 5] = (
                    np.median(log_height[near]) - log_height[rows[j]],
                    log_height[near].max() - log_height[rows[j]],
                    np.median(log_width[near]) - log_width[rows[j]],
                    log_width[near].max() - log_width[rows[j]],
                    high[j] - low[j],
                )
        low = np.searchsorted(time, time - 0.5 * fps, "left")
        high = np.searchsorted(time, time + 0.5 * fps, "right")
        for j in range(len(rows)):
            near = rows[low[j] : high[j]]
            out[rows[j], 10:13] = (
                (np.median(center[near]) - center[rows[j]]) / width[rows[j]],
                (np.median(boxes[near, 3]) - boxes[rows[j], 3]) / height[rows[j]],
                (np.median(boxes[near, 1]) - boxes[rows[j], 1]) / height[rows[j]],
            )
        low = np.searchsorted(time, time - 5.0 * fps, "left")
        high = np.searchsorted(time, time + 5.0 * fps, "right")
        for j in range(len(rows)):
            near = rows[low[j] : high[j]]
            out[rows[j], 13] = np.quantile(log_height[near], 0.9) - log_height[rows[j]]
            out[rows[j], 14] = np.quantile(log_width[near], 0.9) - log_width[rows[j]]
    slope, offset = fit_perspective(boxes[:, 3], height, identities)
    out[:, 15] = np.log(np.maximum(slope * boxes[:, 3] + offset, 1.0)) - log_height
    for frame in np.unique(frames):
        rows = np.where(frames == frame)[0]
        if len(rows) < 2:
            continue
        b = boxes[rows]
        inter = np.clip(
            np.minimum(b[:, None, 2], b[None, :, 2]) - np.maximum(b[:, None, 0], b[None, :, 0]),
            0,
            None,
        ) * np.clip(
            np.minimum(b[:, None, 3], b[None, :, 3]) - np.maximum(b[:, None, 1], b[None, :, 1]),
            0,
            None,
        )
        covered = inter / (width[rows] * height[rows])[:, None]
        np.fill_diagonal(covered, 0)
        cover = covered.argmax(1)
        out[rows, 16] = covered.max(1)
        out[rows, 17] = (b[cover, 3] - b[:, 3]) / height[rows]
        out[rows, 18] = (b[cover, 1] - b[:, 1]) / height[rows]
    return out
