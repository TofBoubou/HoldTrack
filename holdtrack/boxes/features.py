"""Features of the box corrector, one row per (track row, box edge), from the detector views and the masks.

Every quantity is normalised: a positive value points outward, and the unit is the box width for the left and
right edges, the box height for the top and bottom edges. The left and right edges share one model (group 0);
the top edge is group 1 and the bottom edge group 2.
"""

from __future__ import annotations

import numpy as np

SIGN = np.array([-1, -1, 1, 1.0])
OPPOSITE = [2, 3, 0, 1]
GROUP = [0, 1, 0, 2]
PERPENDICULAR = [(1, 3), (0, 2), (1, 3), (0, 2)]
SHIFTS = (1, 2, 3, 4, 6, 8, 12)  # frames before and after, same identity: reference box
VIEW_SHIFTS = (1, 2)  # frames before and after, same identity: detector views and mask
NAMES: list[str] | None = None  # names of the columns, set by the first call


def video_features(
    tables: str, video: str, table: dict, shifts=SHIFTS, view_shifts=VIEW_SHIFTS
) -> dict:
    """Features of one video. `table` is its table with the reference boxes of the stage under "ref";
    `tables` is the folder that holds <video>_masks.npz. Returns {group: (X, rows, edges, sizes)}."""
    global NAMES
    a = table
    m = np.load(f"{tables}/{video}_masks.npz")
    ref = a["ref"]
    n_rows = len(ref)
    width_image = float(a["W"])
    height_image = float(a["H"])
    fr = a["frame"]
    pid = a["pid"]
    w = ref[:, 2] - ref[:, 0]
    h = ref[:, 3] - ref[:, 1]
    size = np.stack([w, h, w, h], 1)
    views = a["views"]
    masks = m["masks"]
    # neighbours: for each row and edge, the most relevant other box of the frame
    neighbour = np.full((n_rows, 4, 4), np.nan)
    per_frame = np.zeros(n_rows)
    order = np.argsort(fr, kind="stable")
    _, start = np.unique(fr[order], return_index=True)
    for x, s in enumerate(start):
        ii = order[s : (start[x + 1] if x + 1 < len(start) else len(order))]
        B = ref[ii]
        n = len(ii)
        per_frame[ii] = n
        if n < 2:
            continue
        vov = (
            np.minimum(B[:, None, 3], B[None, :, 3]) - np.maximum(B[:, None, 1], B[None, :, 1])
        ) / h[ii][:, None]
        hov = (
            np.minimum(B[:, None, 2], B[None, :, 2]) - np.maximum(B[:, None, 0], B[None, :, 0])
        ) / w[ii][:, None]
        np.fill_diagonal(vov, -1)
        np.fill_diagonal(hov, -1)
        for j in range(4):
            ov = vov if j in (0, 2) else hov
            lo = B[:, j % 2][None, :]
            hi = B[:, j % 2 + 2][None, :]
            e = B[:, j][:, None]
            inside = (ov > 0.3) & (lo < e) & (e < hi)
            # signed distance (outward) from the edge to the far edge of the neighbour, and to its near edge
            far = (hi - e) if j >= 2 else (e - lo)
            near = (e - lo) if j >= 2 else (hi - e)
            sc = np.where(inside, ov, -1)
            k = sc.argmax(1)
            has = sc.max(1) > 0
            r = np.arange(n)
            neighbour[ii[has], j, 0] = (far[r, k] / size[ii, j])[has]
            neighbour[ii[has], j, 1] = (near[r, k] / size[ii, j])[has]
            neighbour[ii[has], j, 2] = ((B[k, 3] - B[:, 3]) / h[ii])[has]
            neighbour[ii[has], j, 3] = ov[r, k][has]
    # time: the same identity at t - d and t + d
    key = pid.astype(np.int64) * 100000 + fr
    idx = {int(k_): i for i, k_ in enumerate(key)}
    time = np.full((n_rows, len(shifts) * 2, 4), np.nan)
    for q, d in enumerate(shifts):
        for s_, sgn in enumerate((-1, 1)):
            j_ = np.array([idx.get(int(k_ + sgn * d), -1) for k_ in key])
            ok = j_ >= 0
            time[ok, q * 2 + s_] = ref[j_[ok]] - ref[ok]
    # views and mask (median of the three passes) at the neighbouring frames, relative to the current reference
    time_views = np.full((n_rows, len(view_shifts) * 2, views.shape[0] + 1, 4), np.nan)
    with np.errstate(all="ignore"):
        mask_median = np.nanmedian(masks, 0)
    for q, d in enumerate(view_shifts):
        for s_, sgn in enumerate((-1, 1)):
            j_ = np.array([idx.get(int(k_ + sgn * d), -1) for k_ in key])
            ok = j_ >= 0
            for kk in range(views.shape[0]):
                time_views[ok, q * 2 + s_, kk] = views[kk][j_[ok]] - ref[ok]
            time_views[ok, q * 2 + s_, views.shape[0]] = mask_median[j_[ok]] - ref[ok]
    out = {}
    n_views = views.shape[0]
    for g in (0, 1, 2):
        Xs = []
        rows = []
        edges = []
        sizes = []
        for j in range(4):
            if GROUP[j] != g:
                continue
            o = OPPOSITE[j]
            sj = size[:, j]
            dv = (views[:, :, j] - ref[None, :, j]) * SIGN[j] / sj[None]
            do = (views[:, :, o] - ref[None, :, o]) * SIGN[o] / sj[None]
            dm = (masks[:, :, j] - ref[None, :, j]) * SIGN[j] / sj[None]
            dmo = (masks[:, :, o] - ref[None, :, o]) * SIGN[o] / sj[None]
            cols = [dv.T, do.T, a["views_score"].T, a["views_iou"].T, dm.T, dmo.T, m["masks_iou"].T]
            names = (
                [f"view{k}" for k in range(n_views)]
                + [f"view_opposite{k}" for k in range(n_views)]
                + [f"score{k}" for k in range(n_views)]
                + [f"iou{k}" for k in range(n_views)]
                + [f"mask{k}" for k in range(3)]
                + [f"mask_opposite{k}" for k in range(3)]
                + [f"mask_iou{k}" for k in range(3)]
            )
            with np.errstate(all="ignore"):
                cols += [
                    np.nanmedian(dv, 0)[:, None],
                    np.nanstd(dv, 0)[:, None],
                    np.nanmax(dv, 0)[:, None],
                    np.nanmin(dv, 0)[:, None],
                    np.nanmin(dm, 0)[:, None],
                    np.nanmax(dm, 0)[:, None],
                ]
            names += ["view_median", "view_std", "view_max", "view_min", "mask_min", "mask_max"]
            hidden = a["hidden"]
            cols += [
                a["contact"][:, None],
                hidden[:, [j]],
                hidden[:, [o]],
                hidden[:, [PERPENDICULAR[j][0]]],
                hidden[:, [PERPENDICULAR[j][1]]],
                per_frame[:, None],
            ]
            names += [
                "contact",
                "hidden",
                "hidden_opposite",
                "hidden_side1",
                "hidden_side2",
                "people",
            ]
            border = (
                ref[:, j] if j < 2 else ((width_image if j == 2 else height_image) - ref[:, j])
            ) / sj
            cols += [
                (w / width_image)[:, None],
                (h / height_image)[:, None],
                (w / h)[:, None],
                (ref[:, 3] / height_image)[:, None],
                border[:, None],
            ]
            names += ["width", "height", "aspect", "bottom", "border_distance"]
            cols += [neighbour[:, j, :], neighbour[:, o, :]]
            names += [f"neighbour{k}" for k in range(4)] + [
                f"neighbour_opposite{k}" for k in range(4)
            ]
            cols += [time[:, :, j] * SIGN[j] / sj[:, None], time[:, :, o] * SIGN[o] / sj[:, None]]
            names += [f"time{k}" for k in range(time.shape[1])] + [
                f"time_opposite{k}" for k in range(time.shape[1])
            ]
            for q in range(time_views.shape[1]):
                cols += [
                    time_views[:, q, :, j] * SIGN[j] / sj[:, None],
                    time_views[:, q, :, o] * SIGN[o] / sj[:, None],
                ]
                names += [f"time_view{q}_{kk}" for kk in range(time_views.shape[2])] + [
                    f"time_view_opposite{q}_{kk}" for kk in range(time_views.shape[2])
                ]
            X = np.concatenate([c.astype(np.float32) for c in cols], 1)
            assert X.shape[1] == len(names), (X.shape[1], len(names))
            Xs.append(X)
            rows.append(np.arange(n_rows))
            edges.append(np.full(n_rows, j))
            sizes.append(sj)
            NAMES = names
        out[g] = (
            np.concatenate(Xs),
            np.concatenate(rows),
            np.concatenate(edges),
            np.concatenate(sizes),
        )
    return out
