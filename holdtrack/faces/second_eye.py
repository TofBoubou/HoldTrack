"""Second face model and the cleaning of the face tables. No annotation is read.

    embed           embeddings of the aligned faces by a second public face model (AdaFace IR-101)
    shared_faces    boxes of one frame on which the pose found the same face
    without_shared  a face table without the faces shared by two boxes
    without_moment  embeddings without the component common to all the faces of one moment (light, sharpness)

The two models are fetched from their authors by fetch_face_models.py; they are not redistributed.
"""

import numpy as np

SHARED_DISTANCE = 1.0
MIN_EYES = 3.0
STEP_S = 0.5
HALF_WINDOW_S = 1.5
DIRECTIONS = 3


def _unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


def load_adaface(folder, device="cuda"):
    """AdaFace IR-101 (WebFace12M) from the folder written by fetch_face_models.py. Runs the code of its author."""
    import os
    import sys

    from transformers import AutoModel

    cwd = os.getcwd()
    os.chdir(folder)
    sys.path.insert(0, folder)
    try:
        model = AutoModel.from_pretrained(folder, trust_remote_code=True)
    finally:
        os.chdir(cwd)
        sys.path.pop(0)
    return model.eval().to(device)


def embed(faces, model, device="cuda", batch=256):
    """faces: list of aligned faces, 112 x 112, BGR uint8 (as OpenCV decodes them). Returns the embeddings (N, 512)
    float16 (the face and its mirror, each normalised, summed and normalised), the norm of the raw output of the
    face and the cosine between the face and its mirror."""
    import torch

    def run(x):
        with torch.no_grad():
            o = model(x)
        if isinstance(o, (tuple, list)):
            o = o[0]
        if isinstance(o, dict):
            o = next(iter(o.values()))
        return o.float().cpu().numpy()

    e0 = []
    e1 = []
    for i in range(0, len(faces), batch):
        x = np.ascontiguousarray(
            (
                (np.stack(faces[i : i + batch])[:, :, :, ::-1].astype(np.float32) / 255.0 - 0.5)
                / 0.5
            ).transpose(0, 3, 1, 2)
        )
        x = torch.from_numpy(x).to(device)
        e0.append(run(x))
        e1.append(run(torch.flip(x, dims=[3])))
    if not e0:
        return np.zeros((0, 512), np.float16), np.zeros(0, np.float32), np.zeros(0, np.float32)
    a = np.concatenate(e0)
    b = np.concatenate(e1)
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    s = a / np.maximum(na[:, None], 1e-9) + b / np.maximum(nb[:, None], 1e-9)
    s /= np.maximum(np.linalg.norm(s, axis=1, keepdims=True), 1e-9)
    return (
        s.astype(np.float16),
        na.astype(np.float32),
        ((a * b).sum(1) / np.maximum(na * nb, 1e-9)).astype(np.float32),
    )


def eye_centres(keypoints):
    """Centres of the two eyes from whole-body poses of 133 points (points 59 to 64 and 65 to 70)."""
    kp = np.asarray(keypoints, np.float64)
    return kp[:, 59:65, :2].mean(1), kp[:, 65:71, :2].mean(1)


def shared_faces(frame, keypoints, threshold=SHARED_DISTANCE):
    """For each box (one row per box, with its frame and its pose): whether another box of the frame has the same
    face (eye centres closer than `threshold` times the larger eye distance, among boxes whose eyes are at least 3
    pixels apart), the row of that box (-1 if none), the distance, and the eye distance."""
    frame = np.asarray(frame).astype(int)
    g, d = eye_centres(keypoints)
    c = (g + d) / 2
    y = np.hypot(*(g - d).T)
    ok = np.isfinite(y) & (y >= MIN_EYES)
    n = len(frame)
    shared = np.zeros(n, bool)
    other = np.full(n, -1, np.int64)
    dist = np.full(n, np.nan, np.float32)
    o = np.argsort(frame, kind="stable")
    b = np.r_[0, np.where(np.diff(frame[o]) != 0)[0] + 1, n]
    for s, e in zip(b[:-1], b[1:]):
        ix = o[s:e]
        ix = ix[ok[ix]]
        if len(ix) < 2:
            continue
        D = np.hypot(c[ix, None, 0] - c[None, ix, 0], c[ix, None, 1] - c[None, ix, 1]) / np.maximum(
            y[ix, None], y[None, ix]
        )
        np.fill_diagonal(D, np.inf)
        j = D.argmin(1)
        m = D[np.arange(len(ix)), j]
        dist[ix] = m
        other[ix] = np.where(m < threshold, ix[j], -1)
        shared[ix] = m < threshold
    return shared, other, dist, y.astype(np.float32)


def without_shared(table, shared, rows="idx"):
    """The table (a dict of arrays; `rows` names the array of the box of each face) without the faces of the boxes
    that share their face. Arrays of another length are copied as they are."""
    idx = table[rows]
    keep = ~np.asarray(shared, bool)[idx]
    return {
        k: (v[keep] if (np.ndim(v) >= 1 and len(v) == len(idx)) else v) for k, v in table.items()
    }


def without_moment(embedding, frame, identity, directions=DIRECTIONS, fps=20.0):
    """embedding: one row per face; frame, identity: frame and identity of the box of each face. The local mean of
    the faces of all the persons moves with time in a few directions; they are removed. Returns X - ((X - g) P') P
    in float16 (X normalised, g its mean over the video): centred on the video afterwards, it is the projected
    embedding. Videos with fewer than 50 faces are returned normalised, unchanged."""
    X = _unit(np.asarray(embedding).astype(np.float32))
    out = X
    if len(X) >= 50:
        fr = np.asarray(frame).astype(int)
        ids = np.asarray(identity).astype(np.int64)
        g = X.mean(0)
        step = int(round(STEP_S * fps))
        half = int(round(HALF_WINDOW_S * fps))
        o = np.argsort(fr, kind="stable")
        frs = fr[o]
        M = []
        for c in np.arange(int(fr.min()), int(fr.max()) + 1, step):
            lo = np.searchsorted(frs, c - half, "left")
            hi = np.searchsorted(frs, c + half, "right")
            ix = o[lo:hi]
            if len(ix) < 6:
                continue
            u, inv, cnt = np.unique(ids[ix], return_inverse=True, return_counts=True)
            good = np.where(cnt >= 2)[0]
            if len(good) < 3:
                continue
            S = np.zeros((len(u), X.shape[1]), np.float64)
            np.add.at(S, inv, X[ix])
            M.append((S[good] / cnt[good, None]).mean(0))
        if len(M) >= directions + 2:
            M = np.array(M, np.float32)
            P = np.linalg.svd(M - M.mean(0), full_matrices=False)[2][:directions]
            out = X - ((X - g) @ P.T) @ P
    return out.astype(np.float16)


# ---------------------------------------------------------------- tables (one file per video, in <video>/faces/)
ALIGNMENT = ("eye_distance", "residual", "scale", "point_scores", "row_count")
POSE_FILE = "pose133.npz"
POSE_FIELD = "kp"  # whole-body poses, (N, 133, 3), one per row of tracks.txt


def adaface_table(arcface, embedding, own_norm, mirror_cosine):
    """Table of the second model on the faces of the ArcFace table. `norm_single` and `norm` are those of ArcFace,
    copied: the steps keep the same faces for the two models. `own_norm` is the norm of the second model."""
    out = dict(
        rows=arcface["rows"],
        embedding=_unit(np.asarray(embedding).astype(np.float32)).astype(np.float16),
        own_norm=own_norm,
        mirror_cosine=mirror_cosine,
    )
    out.update({k: arcface[k] for k in arcface if k not in ("rows", "embedding")})
    return out


def clean(table, shared):
    """The table without the faces shared by two boxes (`shared`: one boolean per row of tracks.txt)."""
    return without_shared(table, shared, "rows")


def project(table, frame, identity, directions=DIRECTIONS, fps=20.0):
    """The table with the component of the moment removed from its embeddings. frame, identity: columns of tracks.txt."""
    rows = table["rows"]
    out = dict(table)
    out["embedding"] = without_moment(
        table["embedding"], np.asarray(frame)[rows], np.asarray(identity)[rows], directions, fps
    )
    return out


def write_tables(folder, keypoints=None, adaface=None, device="cuda", fps=20.0):
    """Writes in <folder>/faces: shared.npz, then for each model present (arcface.npz, and adaface.npz, made here from
    aligned.npz if `adaface` is a loaded model) the tables <model>_clean.npz and <model>_clean_projected.npz.
    keypoints: (N, 133, 3) whole-body poses of the rows of tracks.txt; read from pose133.npz of the folder if None."""
    import os

    if keypoints is None:
        keypoints = np.load(os.path.join(folder, POSE_FILE))[POSE_FIELD]
    F = os.path.join(folder, "faces")
    t = np.loadtxt(os.path.join(folder, "tracks.txt"), delimiter=",", ndmin=2)
    frame = t[:, 0].astype(int)
    identity = t[:, 1].astype(np.int64)
    shared, other, dist, eyes = shared_faces(frame, keypoints)
    np.savez(
        os.path.join(F, "shared.npz"),
        shared=shared,
        other=other,
        distance=dist,
        eye_distance=eyes,
        row_count=np.int64(len(frame)),
    )
    if adaface is not None:
        import cv2

        z = np.load(os.path.join(F, "aligned.npz"))
        offsets = z["offsets"]
        blob = z["jpeg"]
        faces = [
            cv2.imdecode(blob[offsets[i] : offsets[i + 1]], cv2.IMREAD_COLOR)
            for i in range(len(z["rows"]))
        ]
        e, n, c = embed(faces, adaface, device)
        np.savez(
            os.path.join(F, "adaface.npz"),
            **adaface_table(
                dict(np.load(os.path.join(F, "arcface.npz"), allow_pickle=True)), e, n, c
            ),
        )
    for model in ("arcface", "adaface"):
        f = os.path.join(F, model + ".npz")
        if os.path.exists(f):
            c = clean(dict(np.load(f, allow_pickle=True)), shared)
            np.savez(os.path.join(F, model + "_clean.npz"), **c)
            np.savez(
                os.path.join(F, model + "_clean_projected.npz"),
                **project(c, frame, identity, fps=fps),
            )
