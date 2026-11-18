"""Face tables of a video: aligned faces from the 133 keypoints, then their ArcFace embeddings.

    align(folder)                 -> <folder>/faces/aligned.npz
    embed_arcface(folder, model)  -> <folder>/faces/arcface.npz

`folder` holds tracks.txt, frames/ and pose133.npz (whole-body poses of the rows of tracks.txt). The second
model and the tables without shared faces are written by `second_eye.write_tables`. No annotation is read.
"""

from __future__ import annotations

import os

import numpy as np

MIN_EYES = 6.0  # pixels between the eyes
TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    np.float32,
)  # ArcFace template, 112 x 112: eyes, nose tip, mouth corners


def five_points(k):
    return np.stack(
        [k[59:65, :2].mean(0), k[65:71, :2].mean(0), k[53, :2], k[71, :2], k[77, :2]]
    ).astype(np.float32)


def align(folder, keypoints=None):
    """A 112 x 112 face, aligned by a similarity on five points, for every row whose eyes are at least six pixels
    apart; stored as JPEG (quality 95). Returns the number of faces."""
    import cv2

    t = np.loadtxt(os.path.join(folder, "tracks.txt"), delimiter=",", ndmin=2)
    fr = t[:, 0].astype(int)
    kp = (
        np.load(os.path.join(folder, "pose133.npz"))["kp"] if keypoints is None else keypoints
    ).astype(np.float64)
    assert len(kp) == len(fr)
    eyes = np.hypot(
        kp[:, 59:65, 0].mean(1) - kp[:, 65:71, 0].mean(1),
        kp[:, 59:65, 1].mean(1) - kp[:, 65:71, 1].mean(1),
    )
    sel = np.where(np.isfinite(eyes) & (eyes >= MIN_EYES))[0]
    sel = sel[np.argsort(fr[sel], kind="stable")]
    rows, blobs, residual, scale, scores = [], [], [], [], []
    current = -1
    image = None
    for i in sel:
        f = int(fr[i])
        if f != current:
            image = cv2.imread(os.path.join(folder, "frames", f"{f:08d}.jpg"))
            current = f
        p = five_points(kp[i])
        M, _ = cv2.estimateAffinePartial2D(p, TEMPLATE, method=cv2.LMEDS)
        if M is None or image is None:
            continue
        ok, buf = cv2.imencode(
            ".jpg",
            cv2.warpAffine(image, M, (112, 112), borderValue=0),
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
        if not ok:
            continue
        q = p @ M[:, :2].T + M[:, 2]
        rows.append(i)
        blobs.append(buf.tobytes())
        residual.append(float(np.abs(q - TEMPLATE).sum(1).mean()))
        scale.append(float(np.sqrt(abs(np.linalg.det(M[:, :2])))))
        k = kp[i]
        scores.append([k[59:65, 2].mean(), k[65:71, 2].mean(), k[53, 2], k[71, 2], k[77, 2]])
    os.makedirs(os.path.join(folder, "faces"), exist_ok=True)
    np.savez(
        os.path.join(folder, "faces", "aligned.npz"),
        rows=np.array(rows, np.int32),
        offsets=np.cumsum([0] + [len(b) for b in blobs]).astype(np.int64),
        jpeg=np.frombuffer(b"".join(blobs), np.uint8),
        eye_distance=eyes[rows].astype(np.float32),
        residual=np.array(residual, np.float32),
        scale=np.array(scale, np.float32),
        point_scores=np.array(scores, np.float32).reshape(-1, 5),
        row_count=np.int64(len(fr)),
    )
    return len(rows)


def load_arcface(path, device="cuda"):
    """ArcFace R100 (glintr100.onnx, fetched by fetch_face_models.py) as a function of a batch of BGR faces.
    On a GPU the graph runs with PyTorch (onnx2torch) when it is installed, else with onnxruntime."""

    def tensor(images):
        return np.ascontiguousarray(
            ((np.stack(images)[:, :, :, ::-1].astype(np.float32) - 127.5) / 127.5).transpose(
                0, 3, 1, 2
            )
        )

    if device == "cuda":
        try:
            import onnx2torch
            import torch

            module = onnx2torch.convert(path).cuda().eval()

            def embed_torch(images):
                with torch.no_grad():
                    return module(torch.from_numpy(tensor(images)).cuda()).float().cpu().numpy()

            return embed_torch
        except ImportError:
            pass
    import onnxruntime as ort

    providers = ["CPUExecutionProvider"]
    if device == "cuda":
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()  # CUDA and cuDNN libraries installed with PyTorch
        providers.insert(0, ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}))
    session = ort.InferenceSession(path, providers=providers)
    name = session.get_inputs()[0].name

    def embed(images):
        return session.run(None, {name: tensor(images)})[0]

    return embed


def embed_arcface(folder, model, batch=512):
    """Embedding of every aligned face: the sum of the embeddings of the face and of its mirror image, normalised;
    `norm_single` is the norm of the embedding of the face alone, the measure of its quality."""
    import cv2

    F = os.path.join(folder, "faces")
    a = dict(np.load(os.path.join(F, "aligned.npz")))
    n = len(a["rows"])
    faces = [
        cv2.imdecode(a["jpeg"][a["offsets"][i] : a["offsets"][i + 1]], cv2.IMREAD_COLOR)
        for i in range(n)
    ]

    def full(images):
        m = len(images)
        return model(images + [images[-1]] * (batch - m))[:m] if m < batch else model(images)

    if n:
        e0 = np.concatenate([full(faces[i : i + batch]) for i in range(0, n, batch)])
        e1 = np.concatenate(
            [full([im[:, ::-1] for im in faces[i : i + batch]]) for i in range(0, n, batch)]
        )
    else:
        e0 = e1 = np.zeros((0, 512), np.float32)
    s = e0 + e1
    ns = np.linalg.norm(s, axis=1)
    np.savez(
        os.path.join(F, "arcface.npz"),
        rows=a["rows"],
        embedding=(s / np.maximum(ns, 1e-9)[:, None]).astype(np.float16),
        norm_single=np.linalg.norm(e0, axis=1).astype(np.float32),
        norm=(ns / 2).astype(np.float32),
        **{k: a[k] for k in ("eye_distance", "residual", "scale", "point_scores", "row_count")},
    )
    return n
