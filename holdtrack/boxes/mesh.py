"""Body mesh of each tracked person, from the pixels, by SAM 3D Body (public code and
weights), prompted with the box of the track row. Body mode, every frame.

Returns the arrays that the corrector reads, one entry per row: row index, frame, the 70
keypoints in the frame and in 3D, the box that encloses the projected vertices (not
clipped to the frame), the camera translation, the focal length, shape, scale and
global rotation. No annotation is read.
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np


def meshes(
    images: list[Path],
    frames,
    boxes,
    code: Path,
    weights: Path,
    mode: str = "body",
    batch: int = 16,
) -> dict:
    import cv2
    import torch

    sys.path.insert(0, str(code))
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body

    model, configuration = load_sam_3d_body(
        f"{weights}/model.ckpt",
        device=torch.device("cuda"),
        mhr_path=f"{weights}/assets/mhr_model.pt",
    )
    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=configuration,
        human_detector=None,
        human_segmentor=None,
        fov_estimator=None,
    )
    frames = np.asarray(frames).astype(int)
    boxes = np.asarray(boxes, np.float32)
    order = np.argsort(frames, kind="stable")
    values, start = np.unique(frames[order], return_index=True)
    start = list(start) + [len(frames)]
    rows, store = [], [[] for _ in range(8)]
    for k, frame in enumerate(values):
        index = order[start[k] : start[k + 1]]
        path = str(images[int(frame) - 1])
        height, width = cv2.imread(path).shape[:2]
        outputs, s = [], 0
        while s < len(index):
            part = boxes[index[s : s + batch]]
            try:
                with torch.inference_mode(), contextlib.redirect_stdout(io.StringIO()):
                    result = estimator.process_one_image(path, bboxes=part, inference_type=mode)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if batch == 1:
                    raise
                batch = max(1, batch // 2)
                continue
            if len(result) != len(part):
                raise ValueError("The mesh model did not return one body per box")
            outputs += result
            s += len(part)
        for j, o in zip(index, outputs):
            vertices = (
                np.asarray(o["pred_vertices"], np.float64)
                + np.asarray(o["pred_cam_t"], np.float64)[None]
            )
            focal = float(np.asarray(o["focal_length"]).reshape(-1)[0])
            x = focal * vertices[:, 0] / vertices[:, 2] + width / 2
            y = focal * vertices[:, 1] / vertices[:, 2] + height / 2
            rows.append(j)
            for q, value in enumerate(
                (
                    np.asarray(o["pred_keypoints_2d"], np.float32)[:, :2],
                    np.asarray(o["pred_keypoints_3d"], np.float32),
                    np.array([x.min(), y.min(), x.max(), y.max()], np.float32),
                    np.asarray(o["pred_cam_t"], np.float32).reshape(-1),
                    np.float32(focal),
                    np.asarray(o["shape_params"], np.float32).reshape(-1),
                    np.asarray(o["scale_params"], np.float32).reshape(-1),
                    np.asarray(o["global_rot"], np.float32).reshape(-1),
                )
            ):
                store[q].append(value)
    rows = np.array(rows)
    return dict(
        row=rows,
        frame=frames[rows],
        kp2=np.array(store[0]),
        kp3=np.array(store[1]),
        mesh_box=np.array(store[2]),
        cam_t=np.array(store[3]),
        focal=np.array(store[4]),
        forme=np.array(store[5]),
        echelle=np.array(store[6]),
        rot_globale=np.array(store[7]),
    )
