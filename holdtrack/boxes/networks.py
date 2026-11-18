"""Predictions of the two sequence networks of the box corrector on a folder of tables.

For each box, a network reads the measurements of its track over nine frames (the views of the box, the mask
boxes of the three forward passes, scores, and the motion of the box) and predicts a correction of the four
edges of the central frame. Everything is relative to the current box, signed outward, in units of its width
and height. The predictions are in the order of the sorted videos.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SIGN = np.array([-1, -1, 1, 1.0], np.float32)
HALF_WINDOW = 4
WIDTH = 64
LAYERS = 2
DROPOUT = 0.2
RENAMED = (("inp.", "input."), ("pos", "position"), ("corps.", "body."), ("tete.", "head."))


def load_tables(folder: Path, views: int) -> dict:
    """Inputs of the network for every row of every video of the folder, concatenated."""
    X = []
    REF = []
    SIZE = []
    KEY = []
    names = []
    spans = {}
    offset = 0
    for p in sorted(q for q in folder.glob("*.npz") if not q.name.endswith("_masks.npz")):
        v = p.stem
        a = np.load(p)
        m = np.load(folder / f"{v}_masks.npz")
        ref = a["ref"].astype(np.float32)
        n = len(ref)
        w = ref[:, 2] - ref[:, 0]
        h = ref[:, 3] - ref[:, 1]
        size = np.stack([w, h, w, h], 1)
        cols = []
        for k in range(views):
            d = (a["views"][k] - ref) * SIGN / size
            ok = np.isfinite(d[:, 0])
            cols += [
                np.where(ok[:, None], np.arcsinh(50 * np.clip(d, -1, 1)), 0),
                ok[:, None],
                (a["views_score"][k] * ok)[:, None],
                (a["views_iou"][k] * ok)[:, None],
            ]
        for k in range(3):
            d = (m["masks"][k] - ref) * SIGN / size
            ok = np.isfinite(d[:, 0])
            cols += [
                np.where(ok[:, None], np.arcsinh(50 * np.clip(d, -1, 1)), 0),
                ok[:, None],
                (m["masks_iou"][k] * ok)[:, None],
            ]
        X.append(np.concatenate([c.astype(np.float32) for c in cols], 1))
        REF.append(ref)
        SIZE.append(size.astype(np.float32))
        KEY.append(a["pid"].astype(np.int64) * 100000 + a["frame"])
        spans[v] = (offset, offset + n)
        offset += n
        names.append(v)
    return dict(
        X=np.concatenate(X),
        ref=np.concatenate(REF),
        size=np.concatenate(SIZE),
        key=KEY,
        spans=spans,
        names=names,
    )


def windows(data: dict, half: int) -> np.ndarray:
    """Row of the same identity at each frame of the window around each row (-1 if absent)."""
    W = np.full((len(data["X"]), 2 * half + 1), -1, np.int64)
    for v, key in zip(data["names"], data["key"]):
        o = data["spans"][v][0]
        idx = {int(k): i for i, k in enumerate(key)}
        for c, dk in enumerate(range(-half, half + 1)):
            W[o : o + len(key), c] = [
                (idx[int(k) + dk] + o) if (int(k) + dk) in idx else -1 for k in key
            ]
    return W


def _batch(data, W, rows):
    wb = W[rows].copy()
    ok = wb >= 0
    wi = np.where(ok, wb, rows[:, None])
    x = data["X"][wi]
    motion = (data["ref"][wi] - data["ref"][rows][:, None]) * SIGN / data["size"][rows][:, None]
    motion = np.arcsinh(20 * np.clip(motion, -3, 3))
    log_size = np.log(data["size"][wi][:, :, :2] / data["size"][rows][:, None, :2])
    tokens = np.concatenate([x, motion, log_size, ok[:, :, None].astype(np.float32)], 2)
    tokens[~ok] = 0
    tokens[:, :, -1] = ok
    return tokens, ~ok


def _network(inputs: int):
    import torch
    from torch import nn

    length = 2 * HALF_WINDOW + 1

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = nn.Linear(inputs, WIDTH)
            self.position = nn.Parameter(torch.zeros(1, length, WIDTH))
            nn.init.normal_(self.position, std=0.02)
            self.body = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    WIDTH, 4, 2 * WIDTH, DROPOUT, batch_first=True, norm_first=True
                ),
                LAYERS,
            )
            self.head = nn.Sequential(
                nn.LayerNorm(WIDTH),
                nn.Dropout(DROPOUT),
                nn.Linear(WIDTH, WIDTH),
                nn.GELU(),
                nn.Linear(WIDTH, 4),
            )

        def forward(self, x, padding):
            h = self.input(x) + self.position
            h = self.body(h, src_key_padding_mask=padding)
            return 0.02 * self.head(h[:, length // 2])

    return Network()


def predict(
    tables: str | Path, weights: str | Path, views: int = 13, threads: int = 1
) -> np.ndarray:
    """``views``: 13 for the networks of the full corrector, 9 for those of the reference corrector."""
    import torch

    torch.set_num_threads(threads)
    data = load_tables(Path(tables), views)
    W = windows(data, HALF_WINDOW)
    net = _network(data["X"].shape[1] + 4 + 2 + 1)
    state = {}
    for name, value in torch.load(weights, map_location="cpu").items():
        for old, new in RENAMED:
            if name == old or (old.endswith(".") and name.startswith(old)):
                name = new + name[len(old) :]
                break
        state[name] = value
    net.load_state_dict(state)
    net.eval()
    rows = np.arange(len(data["X"]))
    out = np.zeros((len(rows), 4), np.float32)
    with torch.no_grad():
        for b in range(0, len(rows), 4096):
            tokens, padding = _batch(data, W, rows[b : b + 4096])
            out[b : b + 4096] = net(torch.from_numpy(tokens), torch.from_numpy(padding)).numpy()
    return out


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tables", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--views", type=int, default=13)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    np.savez_compressed(a.out, pred=predict(a.tables, a.weights, a.views))
