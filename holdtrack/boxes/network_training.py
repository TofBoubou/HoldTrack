"""Train one sequence network of the box corrector on a folder of training tables.

    python -m holdtrack.boxes.network_training --tables tables/ --out network_0.pt --seed 0 --clip 1.0

The folder holds the tables of the training videos, as built by `holdtrack.boxes.pipeline`, with the target
boxes in their `gt` field. The network of `holdtrack.boxes.networks` is trained for 120 epochs on the rows
that have a target; the corrector averages two networks, trained with seeds 0 and 1. `--clip` bounds the
target offsets, in box sizes: 1.0 for the networks of the full corrector (13 views), 0.3 for those of the
reference corrector (9 views). The file written is read by `holdtrack.boxes.networks.predict`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from holdtrack.boxes import networks

EPOCHS = 120
BATCH = 512
RATE = 1e-3
DECAY = 0.05
VIEW_DROPOUT = 0.45
MIRROR = [2, 1, 0, 3]


def load_targets(folder: Path, data: dict, clip: float) -> np.ndarray:
    """Target offsets of the four edges of every row, relative to the box; NaN where a row has no target."""
    Y = []
    for v in data["names"]:
        a, b = data["spans"][v]
        gt = np.load(folder / f"{v}.npz")["gt"].astype(np.float32)
        Y.append(np.clip((gt - data["ref"][a:b]) * networks.SIGN / data["size"][a:b], -clip, clip))
    return np.concatenate(Y).astype(np.float32)


def _mirror_columns(views: int) -> list[int]:
    """Columns of the inputs after a left-right mirror: the left and right edges of each box are exchanged."""
    columns = []
    c = 0
    for _ in range(views):
        columns += [c + x for x in MIRROR] + [c + 4, c + 5, c + 6]
        c += 7
    for _ in range(3):
        columns += [c + x for x in MIRROR] + [c + 4, c + 5]
        c += 6
    return columns


def _batch(data, Y, W, rows, mirrored, reversed_, views):
    wb = W[rows].copy()
    wb[reversed_] = wb[reversed_][:, ::-1]
    ok = wb >= 0
    wi = np.where(ok, wb, rows[:, None])
    x = data["X"][wi]
    motion = (
        (data["ref"][wi] - data["ref"][rows][:, None]) * networks.SIGN / data["size"][rows][:, None]
    )
    motion = np.arcsinh(20 * np.clip(motion, -3, 3))
    log_size = np.log(data["size"][wi][:, :, :2] / data["size"][rows][:, None, :2])
    tokens = np.concatenate([x, motion, log_size, ok[:, :, None].astype(np.float32)], 2)
    tokens[~ok] = 0
    tokens[:, :, -1] = ok
    y = Y[wi]
    n = data["X"].shape[1]
    flipped = tokens[mirrored].copy()
    flipped[:, :, :n] = flipped[:, :, _mirror_columns(views)]
    flipped[:, :, n : n + 4] = tokens[mirrored][:, :, n : n + 4][:, :, MIRROR]
    tokens[mirrored] = flipped
    y = y.copy()
    y[mirrored] = y[mirrored][:, :, MIRROR]
    return tokens, ~ok, y[:, networks.HALF_WINDOW]


def train(
    tables: str | Path,
    out: str | Path,
    views: int = 13,
    seed: int = 0,
    clip: float = 1.0,
    epochs: int = EPOCHS,
    threads: int = 1,
    device: str = "auto",
) -> Path:
    """Fit one network on the rows of the tables that have a target and write its weights."""
    import torch
    from torch import nn

    torch.set_num_threads(threads)
    folder = Path(tables)
    data = networks.load_tables(folder, views)
    Y = load_targets(folder, data, clip)
    W = networks.windows(data, networks.HALF_WINDOW)
    rows = np.flatnonzero(np.isfinite(Y[:, 0]))
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    torch.manual_seed(seed)
    net = networks._network(data["X"].shape[1] + 4 + 2 + 1)
    nn.init.zeros_(net.head[-1].weight)
    nn.init.zeros_(net.head[-1].bias)
    net = net.to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=RATE, weight_decay=DECAY)
    steps = epochs * (len(rows) // BATCH)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=RATE, total_steps=steps, pct_start=0.05
    )
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        net.train()
        order = rng.permutation(rows)
        total = 0.0
        count = 0
        for b in range(0, len(rows) - BATCH + 1, BATCH):
            r = order[b : b + BATCH]
            mirrored = rng.random(len(r)) < 0.5
            reversed_ = rng.random(len(r)) < 0.5
            tokens, padding, y = _batch(data, Y, W, r, mirrored, reversed_, views)
            x = torch.from_numpy(tokens).to(device)
            c = 0
            for width in [7] * views + [
                6
            ] * 3:  # a whole view is hidden from a sample, over the window
                hidden = torch.rand(x.shape[0], device=device) < VIEW_DROPOUT
                x[hidden, :, c : c + width] = 0
                c += width
            prediction = net(x, torch.from_numpy(padding).to(device))
            loss = nn.functional.smooth_l1_loss(
                prediction, torch.from_numpy(y).to(device), beta=0.01
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            total += float(loss.detach())
            count += 1
        print(f"epoch {epoch + 1}/{epochs}: loss {total / max(count, 1):.5f}", flush=True)
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in net.state_dict().items()}, target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--tables", required=True, help="tables of the training videos, with targets"
    )
    parser.add_argument("--out", required=True, help="weights file to write")
    parser.add_argument("--views", type=int, default=13)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="auto", help="cpu, cuda, or auto (cuda when available)")
    options = parser.parse_args()
    print("written", train(**vars(options)))


if __name__ == "__main__":
    main()
