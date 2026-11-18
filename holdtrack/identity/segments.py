"""Clean segments: runs of boxes of one identity that no other box overlaps."""

import numpy as np

from .geometry import frames as n_frames
from .geometry import overlap_with_others, xyxy

MIN_S = 0.5  # shortest segment
GAP_CLEAN_S = 0.15  # a clean box further than this from the previous one starts a new segment
GAP_OVERLAP_S = (
    0.1  # an overlapped box further than this from the last clean one closes the segment
)


def clean_segments(rows, max_iou=0.5, fps=20.0):
    """rows: frame, identity, x, y, width, height. Returns the segment of each row (-1 if none), the list of
    segments (identity, start, end, n) and the overlap of each row with the other boxes of its frame."""
    frame = rows[:, 0].astype(int)
    identity = rows[:, 1].astype(int)
    overlap = overlap_with_others(frame, xyxy(rows))
    shortest = n_frames(MIN_S, fps)
    gap_clean = n_frames(GAP_CLEAN_S, fps)
    gap_overlap = n_frames(GAP_OVERLAP_S, fps)
    segment = np.full(len(rows), -1)
    found = []

    def close(run, name):
        if len(run) >= shortest:
            segment[run] = len(found)
            found.append(
                dict(
                    identity=int(name),
                    start=int(frame[run[0]]),
                    end=int(frame[run[-1]]),
                    n=len(run),
                )
            )

    for name in np.unique(identity):
        idx = np.where(identity == name)[0]
        idx = idx[np.argsort(frame[idx])]
        run = []
        for i in idx:
            if overlap[i] <= max_iou:
                if run and frame[i] - frame[run[-1]] > gap_clean:
                    close(run, name)
                    run = []
                run.append(i)
            elif run and frame[i] - frame[run[-1]] > gap_overlap:
                close(run, name)
                run = []
        close(run, name)
    return segment, found, overlap
