"""Steps 2 and 3. Link the safe stretches of a video again.

A safe stretch is a run of boxes of one identity on which the person is surely herself. Each stretch receives at
most one predecessor among its candidates and each stretch at most one successor (one assignment over the video).
The gain of a link is the log-odds of a classifier that reads the passes, the motion and the appearance of the two
stretches. The chains of stretches become the identities.
"""

import collections

import numpy as np
from scipy.optimize import linear_sum_assignment

from .geometry import REFERENCE_FPS
from .geometry import frames as n_frames
from .stretches import stretches as safe_stretches
from .video import majority

FEATURES = (
    "same",
    "passes_agree",
    "vote",
    "llr",
    "llr_gap_b",
    "llr_gap_a",
    "motion",
    "motion_back",
    "gap",
    "extended_a",
    "extended_b",
    "unanimous",
    "gap_unanimous",
    "gap_overflow",
    "gap_visible",
    "gap_extended",
    "bytetrack",
    "backward",
    "llr2",
    "llr2_gap_b",
    "llr2_gap_a",
)
MONOTONIC = (1, 1, 1, 1, 1, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1)
# candidates: gap at most GAP_MAX_S, distance (in box heights) at most DISTANCE_MAX + DISTANCE_SLOPE per frame at 20 frames per second
GAP_MAX_S = 10.0
DISTANCE_MAX = 1.0
DISTANCE_SLOPE = 0.01
SECOND_GAP_MAX_S = 20.0
SECOND_DISTANCE_MAX = 2.0
SECOND_DISTANCE_SLOPE = 0.02
END_S = 0.5
TIP_S = 0.25
WINDOW_S = 1.5


def whiten(x, group):
    """128 principal components, then whitening by the scatter inside the groups."""
    z = x - x.mean(0)
    _, _, vt = np.linalg.svd(z, full_matrices=False)
    k = min(128, vt.shape[0])
    z = z @ vt[:k].T
    zc = z.copy()
    m = group >= 0
    for q in np.unique(group[m]):
        zc[group == q] -= z[group == q].mean(0)
    sw = zc[m].T @ zc[m] / m.sum()
    ev, evec = np.linalg.eigh(sw + 1e-3 * np.trace(sw) / k * np.eye(k))
    return (z @ evec) / np.sqrt(ev)


def _centres(signature, crop_stretch):
    out = {}
    for k in np.unique(crop_stretch[crop_stretch >= 0]):
        c = signature[crop_stretch == k].mean(0)
        out[int(k)] = c / (np.linalg.norm(c) + 1e-9)
    return out


def _extrapolate(fr, x, t):
    if len(fr) < 3 or fr.max() == fr.min():
        return float(np.median(x))
    a, b = np.polyfit(fr - fr.mean(), x, 1)
    return float(b + a * (t - fr.mean()))


def build_pairs(video, identities, crops_meta, embedding, features, mode="full", second=False):
    """Candidate links (a before, b after) between the safe stretches of the video, with their raw measures.
    embedding, features: one row per crop (appearance learned on the video; appearance learned on other videos).
    Returns a dict: a, b (stretch numbers), raw measures by name, and the stretches."""
    fps = video.fps
    scale = REFERENCE_FPS / fps
    gap_max = n_frames(SECOND_GAP_MAX_S if second else GAP_MAX_S, fps)
    d_max = SECOND_DISTANCE_MAX if second else DISTANCE_MAX
    d_slope = (SECOND_DISTANCE_SLOPE if second else DISTANCE_SLOPE) * scale
    end = n_frames(END_S, fps)
    tip = n_frames(TIP_S, fps)
    window = n_frames(WINDOW_S, fps)
    p = video.rows(identities)
    pf = p[:, 0].astype(int)
    pid = p[:, 1].astype(int)
    vis = video.visibility
    assert len(vis) == len(p)
    stretch, S = safe_stretches(video, p, mode)
    vx = video.forward
    cx = p[:, 2] + p[:, 4] / 2
    yb = p[:, 3] + p[:, 5]
    hh = p[:, 5]
    ww = p[:, 4]
    unanimous = (vx[0] == vx[1]) & (vx[1] == vx[2]) & (vx[0] >= 0)
    extended = np.nan_to_num(np.maximum(vis[:, 0], vis[:, 2]), nan=1.0) > 0.08
    crop_stretch = stretch[crops_meta[:, 9].astype(int)]
    assert len(embedding) == len(crops_meta) == len(features)
    C = _centres(embedding.astype(np.float32), crop_stretch)
    s2 = features.astype(np.float32)
    if (crop_stretch >= 0).sum() > 200:
        s2 = whiten(s2, crop_stretch)
    C2 = _centres(s2, crop_stretch)
    by_identity = collections.defaultdict(list)
    for k, s in enumerate(S):
        by_identity[s["identity"]].append(k)
    for name in by_identity:
        by_identity[name].sort(key=lambda k: S[k]["start"])

    def tip_of(k, last):
        i = S[k]["rows"]
        i = i[-end:] if last else i[:end]
        j = i[-tip:] if last else i[:tip]
        return dict(
            fr=pf[i],
            cx=cx[i],
            yb=yb[i],
            x=float(np.median(cx[j])),
            y=float(np.median(yb[j])),
            h=float(np.median(hh[j])),
            w=float(np.median(ww[j])),
        )

    END = {k: tip_of(k, True) for k in range(len(S))}
    START = {k: tip_of(k, False) for k in range(len(S))}

    def extended_share(name, t0, t1):
        m = (pid == name) & (pf >= t0) & (pf <= t1)
        return float(extended[m].mean()) if m.any() else 0.0

    VM = {k: [majority(vx[j, S[k]["rows"]]) for j in range(5)] for k in range(len(S))}
    back = video.backward
    BM = {k: majority(back[s["rows"]]) for k, s in enumerate(S)} if back is not None else {}
    cols = collections.defaultdict(list)
    A = []
    B = []
    for b, sb in enumerate(S):
        cands = []
        i = by_identity[sb["identity"]].index(b)
        if i > 0:
            cands.append(by_identity[sb["identity"]][i - 1])
        for name, ks in by_identity.items():
            if name == sb["identity"]:
                continue
            before = [k for k in ks if S[k]["end"] < sb["start"]]
            if not before:
                continue
            a = before[-1]
            e = sb["start"] - S[a]["end"]
            if e > gap_max:
                continue
            d = np.hypot(START[b]["x"] - END[a]["x"], START[b]["y"] - END[a]["y"]) / max(
                END[a]["h"], 1.0
            )
            if d <= d_max + d_slope * e:
                cands.append(a)
        for a in cands:
            sa = S[a]
            e = sb["start"] - sa["end"]
            a_ = END[a]
            b_ = START[b]
            h = max(a_["h"], 1.0)
            ev = (
                np.hypot(
                    _extrapolate(a_["fr"], a_["cx"], sb["start"]) - b_["x"],
                    _extrapolate(a_["fr"], a_["yb"], sb["start"]) - b_["y"],
                )
                / h
            )
            evb = (
                np.hypot(
                    _extrapolate(b_["fr"], b_["cx"], sa["end"]) - a_["x"],
                    _extrapolate(b_["fr"], b_["yb"], sa["end"]) - a_["y"],
                )
                / h
            )
            vb = []
            for j in range(5):
                x = vx[j, sb["rows"]]
                x = x[x >= 0]
                vb.append(
                    float((x == VM[a][j]).mean()) if len(x) and VM[a][j] is not None else np.nan
                )
            vinv = np.nan
            if back is not None and BM.get(a) is not None:
                x = back[sb["rows"]]
                x = x[x >= 0]
                if len(x):
                    vinv = float((x == BM[a]).mean())
            mg = (
                ((pid == sa["identity"]) | (pid == sb["identity"]))
                & (pf > sa["end"])
                & (pf < sb["start"])
            )
            mb = mg & (pid == sb["identity"])
            ma = mg & (pid == sa["identity"])
            dv = vis[mg][:, :4] if mg.any() else np.zeros((0, 4))
            A.append(a)
            B.append(b)
            cols["same"].append(float(sa["identity"] == sb["identity"]))
            cols["passes_agree"].append(float(np.nansum(np.array(vb[:3]) >= 0.5)))
            cols["vote"].append(vb[3])
            cols["unanimous_a"].append(float(unanimous[sa["rows"]].mean()))
            cols["unanimous_b"].append(float(unanimous[sb["rows"]].mean()))
            cols["gap"].append(float(e) * scale)
            cols["motion"].append(float(ev))
            cols["motion_back"].append(float(evb))
            cols["extended_a"].append(
                extended_share(sa["identity"], sa["end"] + 1, sa["end"] + window)
            )
            cols["extended_b"].append(
                extended_share(sb["identity"], sb["start"] - window, sb["start"] - 1)
            )
            cols["cos"].append(float(C[a] @ C[b]) if a in C and b in C else np.nan)
            cols["cos2"].append(float(C2[a] @ C2[b]) if a in C2 and b in C2 else np.nan)
            cols["gap_unanimous"].append(
                min([float(unanimous[x].mean()) for x in (ma, mb) if x.any()], default=1.0)
            )
            cols["gap_overflow"].append(
                float(np.nanmax(dv)) if dv.size and np.isfinite(dv).any() else 0.0
            )
            cols["gap_visible"].append(
                float(np.nanmin(np.nan_to_num(vis[mg, 4], nan=0.0))) if mg.any() else 1.0
            )
            cols["gap_extended"].append(float(extended[mg].mean()) if mg.any() else 0.0)
            cols["bytetrack"].append(vb[4])
            cols["backward"].append(vinv)
    raw = {k: np.array(v, np.float32).astype(np.float64) for k, v in cols.items()}
    return dict(
        a=np.array(A, int), b=np.array(B, int), raw=raw, stretches=S, stretch=stretch, rows=p
    )


def _appearance_llr(cos, same, a, b):
    """Log-likelihood ratio of the appearance, calibrated on the video without annotation: logistic regression
    "this pair is a link of the base tracker" on the cosine alone; then the gap to the best rival candidate."""
    from sklearn.linear_model import LogisticRegression

    llr = np.zeros(len(cos))
    m = np.isfinite(cos)
    y = same[m]
    if y.sum() >= 5 and (~y).sum() >= 5:
        lr = LogisticRegression(C=10.0, max_iter=200).fit(cos[m][:, None], y)
        llr[m] = lr.decision_function(cos[m][:, None]) - np.log(y.sum() / (~y).sum())
    gaps = []
    for key in (b, a):
        out = np.zeros(len(cos))
        groups = collections.defaultdict(list)
        for i, k in enumerate(key):
            groups[int(k)].append(i)
        for ix in groups.values():
            if len(ix) < 2:
                continue
            lab = llr[ix]
            for j, i in enumerate(ix):
                out[i] = lab[j] - np.max(np.delete(lab, j))
        gaps.append(out)
    return llr, gaps[0], gaps[1]


def model_features(pairs):
    """Matrix of the features of the classifier, columns in the order of FEATURES."""
    r = pairs["raw"]
    if not len(pairs["a"]):
        return np.zeros((0, len(FEATURES)))

    def g(k):
        return np.nan_to_num(r[k], nan=0.0)

    same = r["same"] == 1
    llr, gb, ga = _appearance_llr(r["cos"], same, pairs["a"], pairs["b"])
    llr2, gb2, ga2 = _appearance_llr(r["cos2"], same, pairs["a"], pairs["b"])
    return np.stack(
        [
            g("same"),
            g("passes_agree"),
            g("vote"),
            llr,
            gb,
            ga,
            np.log1p(g("motion")),
            np.log1p(g("motion_back")),
            np.log1p(g("gap")),
            g("extended_a"),
            g("extended_b"),
            g("unanimous_a") * g("unanimous_b"),
            g("gap_unanimous"),
            g("gap_overflow"),
            g("gap_visible"),
            g("gap_extended"),
            g("bytetrack"),
            np.nan_to_num(r["backward"], nan=0.5),
            llr2,
            gb2,
            ga2,
        ],
        1,
    )


def _log_odds(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class LinearModel:
    """Logistic classifier of the first pass, with its weights."""

    def __init__(self, path):
        m = np.load(path, allow_pickle=True)
        assert tuple(m["features"].tolist()) == FEATURES
        self.constant = float(m["constant"])
        self.weights = m["weights"]
        self.mean = m["mean"]
        self.std = m["std"]

    def score(self, F):
        return _log_odds(
            1 / (1 + np.exp(-(self.constant + ((F - self.mean) / self.std) @ self.weights)))
        )


class TrainedPair:
    """Classifiers of the second pass: a logistic regression and boosted trees fitted on the labelled pairs
    shipped with the package; the score is the mean of their log-odds."""

    def __init__(self, path):
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression

        d = np.load(path, allow_pickle=True)
        assert tuple(d["features"].tolist()) == FEATURES
        F = d["F"]
        y = d["y"]
        self.mean = F.mean(0)
        self.std = F.std(0) + 1e-9
        z = (F - self.mean) / self.std
        self.models = [
            LogisticRegression(C=1.0, max_iter=2000).fit(z, y),
            HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.05,
                max_leaf_nodes=8,
                min_samples_leaf=30,
                l2_regularization=1.0,
                monotonic_cst=list(MONOTONIC),
                random_state=0,
            ).fit(z, y),
        ]

    def score(self, F):
        z = (F - self.mean) / self.std
        if not len(z):  # a clip without any candidate pair
            return np.zeros(0)
        return sum(_log_odds(m.predict_proba(z)[:, 1]) for m in self.models) / 2


def assign(pairs, scores, keep_bonus=0.0, cut=-99.0, new=1.0):
    """New identities from the links. A link gains its score, plus keep_bonus if it is a link of the incoming
    identities; a stretch without predecessor costs `cut` if it had one, `new` if not. Returns the identities and
    the number of links between two identities."""
    S = pairs["stretches"]
    n = len(S)
    p = pairs["rows"]
    pf = p[:, 0].astype(int)
    pid = p[:, 1].astype(int)
    stretch = pairs["stretch"]
    if n == 0:
        return pid.copy(), 0
    same = pairs["raw"]["same"] == 1 if len(pairs["a"]) else np.zeros(0, bool)
    M = np.full((n, 2 * n), -1e6)
    had = np.zeros(n, bool)
    for a, b, sc, me in zip(pairs["a"], pairs["b"], scores, same):
        if np.isfinite(sc):
            had[b] |= bool(me)
            M[b, a] = sc + (keep_bonus if me else 0.0)
    for b in range(n):
        M[b, n + b] = cut if had[b] else new
    r, c = linear_sum_assignment(-M)
    pred = {int(b): int(a) for b, a in zip(r, c) if a < n and M[b, a] > -1e5}
    root = list(range(n))

    def R(x):
        while root[x] != x:
            x = root[x]
        return x

    for b, a in pred.items():
        root[R(b)] = R(a)
    chains = collections.defaultdict(list)
    for k in range(n):
        chains[R(k)].append(k)
    # a box outside the stretches follows the nearest stretch of its identity
    nearest = np.full(len(pid), -1)
    for name in np.unique(pid):
        ks = [k for k in range(n) if S[k]["identity"] == name]
        if not ks:
            continue
        idx = np.where(pid == name)[0]
        start = np.array([S[k]["start"] for k in ks])
        end = np.array([S[k]["end"] for k in ks])
        for i in idx:
            d = np.maximum(0, np.maximum(start - pf[i], pf[i] - end))
            nearest[i] = ks[int(d.argmin())]
    chain_of = np.array([R(k) if k >= 0 else -1 for k in nearest])
    out = pid.copy()
    taken = collections.defaultdict(set)
    for i in np.where(chain_of < 0)[0]:
        taken[int(pid[i])].add(int(pf[i]))
    # a chain takes the incoming identity that weighs most of its boxes, if free on its frames
    for r_ in sorted(chains, key=lambda r_: -int((chain_of == r_).sum())):
        li = np.where(chain_of == r_)[0]
        fr = set(pf[li].tolist())
        name = None
        for lab, _ in collections.Counter(pid[li].tolist()).most_common():
            if not (taken[int(lab)] & fr):
                name = int(lab)
                break
        if name is None:
            name = 4 * 10**6 + r_
        out[li] = name
        taken[name] |= fr
    key = pf.astype(np.int64) * 10**9 + out.astype(np.int64)
    u, cnt = np.unique(key, return_counts=True)
    dup = set(u[cnt > 1].tolist())
    if dup:
        seen = set()
        for i in np.argsort(stretch < 0, kind="stable"):
            if key[i] in dup:
                if key[i] in seen:
                    out[i] = 5 * 10**6 + i
                seen.add(key[i])
    return out, sum(1 for b, a in pred.items() if S[a]["identity"] != S[b]["identity"])


def first_pass(video, identities, crops_meta, embedding, features, model):
    """Step 2: links between fully visible stretches, scored by the logistic classifier."""
    pairs = build_pairs(video, identities, crops_meta, embedding, features, "full")
    return assign(pairs, model.score(model_features(pairs)), 0.0, -99.0, 1.0)


def second_pass(video, identities, crops_meta, embedding, features, model):
    """Proposal of step 3: links between stretches where the upper body is hers, wider candidates."""
    pairs = build_pairs(video, identities, crops_meta, embedding, features, "upper", second=True)
    return assign(pairs, model.score(model_features(pairs)), 0.0, -99.0, 0.0)
