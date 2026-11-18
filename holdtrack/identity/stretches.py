"""Safe stretches: runs of boxes of one identity on which the person is surely herself."""

import numpy as np

from .geometry import frames as n_frames

MIN_S = 0.5
GAP_SAFE_S = 0.15
GAP_UNSAFE_S = 0.1


def safe_rows(visibility, mode="full"):
    """full: the box does not overflow its mask box (four sides under 5%); full03: also contact under 0.3;
    iou05: contact at most 0.5."""
    ok = np.isfinite(visibility[:, 0])
    worst = np.where(ok, np.nan_to_num(visibility[:, :4], nan=-np.inf).max(1), np.inf)
    full = ok & (worst < 0.05)
    return {
        "full03": full & (visibility[:, 9] < 0.3),
        "full": full,
        "iou05": visibility[:, 9] <= 0.5,
    }[mode]


def safety(video, mode):
    """Safe rows and cuts of a mode. upper: upper body hers, cut at anchor jumps; union: fully visible or upper
    body hers (a jump cuts only where the box is not fully visible)."""
    vis = video.visibility
    if mode in ("full03", "full", "iou05"):
        return safe_rows(vis, mode), None
    t = video.upper_body
    full = safe_rows(vis, "full")
    upper = t[:, 1] > 0.5
    jump = t[:, 2] > 0.5
    if mode == "upper":
        return upper, jump
    if mode == "union":
        return full | upper, jump & ~full
    raise ValueError(mode)


def build(rows, safe, cut=None, require=None, fps=20.0):
    """Stretch of each row (-1 if none) and the list of stretches (identity, start, end, n, rows)."""
    frame = rows[:, 0].astype(int)
    identity = rows[:, 1].astype(int)
    stretch = np.full(len(rows), -1)
    found = []
    shortest = n_frames(MIN_S, fps)
    gap_safe = n_frames(GAP_SAFE_S, fps)
    gap_unsafe = n_frames(GAP_UNSAFE_S, fps)

    def close(run, name):
        if len(run) >= shortest and (require is None or int(require[run].sum()) >= shortest):
            stretch[run] = len(found)
            found.append(
                dict(
                    identity=int(name),
                    start=int(frame[run[0]]),
                    end=int(frame[run[-1]]),
                    n=len(run),
                    rows=np.array(run),
                )
            )

    for name in np.unique(identity):
        idx = np.where(identity == name)[0]
        idx = idx[np.argsort(frame[idx])]
        run = []
        for i in idx:
            if safe[i]:
                if run and (frame[i] - frame[run[-1]] > gap_safe or (cut is not None and cut[i])):
                    close(run, name)
                    run = []
                run.append(i)
            elif run and frame[i] - frame[run[-1]] > gap_unsafe:
                close(run, name)
                run = []
        close(run, name)
    return stretch, found


def stretches(video, rows, mode):
    """fullplus: stretches of the union that hold at least half a second of fully visible boxes."""
    if mode == "fullplus":
        safe, cut = safety(video, "union")
        return build(rows, safe, cut, safe_rows(video.visibility, "full"), video.fps)
    safe, cut = safety(video, mode)
    return build(rows, safe, cut, fps=video.fps)
