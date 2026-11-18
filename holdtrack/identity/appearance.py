"""The three appearance stages of the chain. They run on a GPU.

    classifier  a network trained on the video, one class per identity of the base tracker, each crop predicted
                out of block (the video is cut in blocks of ten seconds; trained on the even ones, read on the odd
                ones, and the reverse);
    embedding   a network trained on the video without identities: two crops of one clean segment are the same
                person, two segments present in one frame are different persons;
    features    a network trained on other videos, applied as it is.

The background of every crop is erased with a person mask before the three stages.

    python -m holdtrack.identity.appearance <video folder> <stages folder> <appearance network .pt>
"""

import io
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image

from .geometry import frames as n_frames

BLOCK_S = 10.0
RESOLUTION = 512
EPOCHS = 25
BATCH = 64
MAX_OVERLAP = 0.5
MIN_CROPS = 40
ITERATIONS = 1200
TEMPERATURE = 0.1
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)
_GREY = (124, 116, 104)
_segmenter = {}


def load_crops(jpegs):
    """uint8 tensor (N, 3, 256, 128) of the crops."""
    x = np.stack([np.asarray(Image.open(io.BytesIO(j.tobytes())).convert("RGB")) for j in jpegs])
    return torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()


@torch.no_grad()
def erase_background(X, meta, device="cuda"):
    """Crops with the background replaced by grey (Mask R-CNN of torchvision; the person that best covers the box),
    and the share of pixels kept. A crop without such a person is left as it is."""
    if device not in _segmenter:
        seg = (
            torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
                weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1
            )
            .to(device)
            .eval()
        )
        seg.transform.min_size = (384,)
        seg.transform.max_size = 768
        _segmenter[device] = seg
    seg = _segmenter[device]
    out = X.clone()
    share = np.ones(len(X), np.float32)
    H, W = X.shape[2:]
    grey = torch.tensor(_GREY, dtype=torch.uint8).view(3, 1, 1)
    for k in range(0, len(X), 48):
        res = seg(list(X[k : k + 48].to(device).float() / 255))
        for j, r in enumerate(res):
            w, h = meta[k + j, 4] * 1.16, meta[k + j, 5] * 1.16
            sc = min(W / w, H / h)
            nw, nh = w * sc, h * sc
            cb = torch.tensor(
                [(W - nw) / 2, (H - nh) / 2, (W + nw) / 2, (H + nh) / 2], device=device
            )
            ok = (r["labels"] == 1) & (r["scores"] > 0.3)
            if not ok.any():
                continue
            b = r["boxes"][ok]
            iw = (torch.minimum(b[:, 2], cb[2]) - torch.maximum(b[:, 0], cb[0])).clamp(min=0)
            ih = (torch.minimum(b[:, 3], cb[3]) - torch.maximum(b[:, 1], cb[1])).clamp(min=0)
            iou = (
                iw
                * ih
                / (
                    (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
                    + (cb[2] - cb[0]) * (cb[3] - cb[1])
                    - iw * ih
                    + 1e-6
                )
            )
            i = int(iou.argmax())
            if iou[i] < 0.3:
                continue
            m = F.max_pool2d((r["masks"][ok][i : i + 1] > 0.5).float(), 5, 1, 2)[0].bool().cpu()
            share[k + j] = float(m.float().mean())
            out[k + j] = torch.where(m, X[k + j], grey.expand(3, H, W))
    return out, share


def _normalise(xb, device):
    mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=device).view(1, 3, 1, 1)
    return ((xb.to(device).float() / 255) - mean) / std


def _augment(x, blur, light, shift):
    """Scale, translation, flip, optional loss of resolution, light, and a rectangle erased in half of the crops."""
    n, _, H, W = x.shape
    device = x.device
    s = 1 + 0.15 * (torch.rand(n, device=device) * 2 - 1)
    tx = 0.08 * (torch.rand(n, device=device) * 2 - 1)
    ty = 0.05 * (torch.rand(n, device=device) * 2 - 1)
    fl = (torch.rand(n, device=device) < 0.5).float() * 2 - 1
    th = torch.zeros(n, 2, 3, device=device)
    th[:, 0, 0] = s * fl
    th[:, 1, 1] = s
    th[:, 0, 2] = tx
    th[:, 1, 2] = ty
    x = F.grid_sample(
        x,
        F.affine_grid(th, x.shape, align_corners=False),
        padding_mode="border",
        align_corners=False,
    )
    if blur:
        f = 0.4 + 0.6 * torch.rand(1).item()
        x = F.interpolate(
            F.interpolate(
                x,
                size=(max(16, int(H * f)), max(8, int(W * f))),
                mode="bilinear",
                align_corners=False,
            ),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
    x = x * (1 + light * (torch.rand(n, 1, 1, 1, device=device) * 2 - 1)) + shift * (
        torch.rand(n, 1, 1, 1, device=device) * 2 - 1
    )
    for i in range(n):
        if torch.rand(1).item() < 0.5:
            h = int(H * (0.1 + 0.25 * torch.rand(1).item()))
            w = int(W * (0.2 + 0.4 * torch.rand(1).item()))
            y0 = torch.randint(0, H - h + 1, (1,)).item()
            x0 = torch.randint(0, W - w + 1, (1,)).item()
            x[i, :, y0 : y0 + h, x0 : x0 + w] = 0
    return x


def classifier(X, meta, device="cuda", fps=20.0, seed=0):
    """Log-probabilities of each crop for each identity, out of block (NaN where no network read the crop), and the
    identities (those with at least 40 crops). X: masked crops; meta: one row per crop (see crops.py)."""
    frame = meta[:, 0].astype(int)
    label = meta[:, 1].astype(int)
    names, count = np.unique(label, return_counts=True)
    classes = names[count >= MIN_CROPS]
    K = len(classes)
    y = np.searchsorted(classes, label)
    known = np.isin(label, classes)
    y[~known] = 0
    block = (frame - 1) // n_frames(BLOCK_S, fps)
    log_probs = np.full((len(X), K), np.nan, np.float32)
    used = known & (meta[:, 6] <= MAX_OVERLAP)
    yt = torch.from_numpy(y)

    def prepare(xb):
        return F.interpolate(
            _normalise(xb, device),
            size=(RESOLUTION, RESOLUTION // 2),
            mode="bilinear",
            align_corners=False,
        )

    def train(Xt, yt_):
        torch.manual_seed(seed)
        m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Linear(2048, K)
        m = m.to(device).to(memory_format=torch.channels_last)
        opt = torch.optim.AdamW(
            [
                {
                    "params": [p for n, p in m.named_parameters() if not n.startswith("fc.")],
                    "lr": 2e-4,
                },
                {"params": m.fc.parameters(), "lr": 2e-3},
            ],
            weight_decay=0.05,
        )
        n = len(Xt)
        steps = EPOCHS * max(1, n // BATCH)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=[2e-4, 2e-3], total_steps=steps, pct_start=0.1
        )
        scaler = torch.amp.GradScaler("cuda")
        it = 0
        m.train()
        while it < steps:
            perm = torch.randperm(n)
            for k in range(0, n - BATCH + 1, BATCH):
                idx = perm[k : k + BATCH]
                xb = _augment(prepare(Xt[idx]), False, 0.15, 0.08).contiguous(
                    memory_format=torch.channels_last
                )
                yb = yt_[idx].to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = F.cross_entropy(m(xb), yb, label_smoothing=0.1)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                it += 1
                if it >= steps:
                    break
        return m

    @torch.no_grad()
    def predict(m, Xt):
        m.eval()
        o = []
        for k in range(0, len(Xt), 256):
            xb = prepare(Xt[k : k + 256]).contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o.append((m(xb) + m(torch.flip(xb, dims=[3]))).float().log_softmax(1).cpu())
        return torch.cat(o).numpy()

    for fold in (0, 1):
        tr = used & (block % 2 == fold)
        te = block % 2 != fold
        if tr.sum() < BATCH or te.sum() == 0:
            continue
        m = train(X[torch.from_numpy(tr)], yt[torch.from_numpy(tr)])
        log_probs[te] = predict(m, X[torch.from_numpy(te)])
        del m
    return log_probs.astype(np.float16), classes


class _Embedder(nn.Module):
    def __init__(self):
        super().__init__()
        m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Identity()
        self.b = m
        self.h = nn.Linear(2048, 256)

    def forward(self, x):
        return F.normalize(self.h(self.b(x)).float(), dim=1)


def _contrast(z, g):
    sim = z @ z.t() / TEMPERATURE
    n = len(g)
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    pos = (g[:, None] == g[None, :]) & ~eye
    lp = sim.masked_fill(eye, -1e9).log_softmax(1)
    k = pos.sum(1).clamp(min=1)
    return -((lp * pos).sum(1) / k)[pos.any(1)].mean()


def embedding(X, meta, device="cuda", seed=0):
    """(N, 256) embedding learned on the video from its clean segments (column 7 of meta)."""
    frame = meta[:, 0].astype(int)
    seg = meta[:, 7].astype(int)
    rng = np.random.default_rng(seed)
    of_segment = {k: np.where(seg == k)[0] for k in np.unique(seg[seg >= 0])}
    active = {f: np.unique(seg[(frame == f) & (seg >= 0)]) for f in np.unique(frame)}
    frames = [f for f, a in active.items() if len(a) >= 2]
    torch.manual_seed(seed)
    net = _Embedder().to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(
        [{"params": net.b.parameters(), "lr": 1e-4}, {"params": net.h.parameters(), "lr": 1e-3}],
        weight_decay=0.05,
    )
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[1e-4, 1e-3], total_steps=ITERATIONS, pct_start=0.1
    )
    scaler = torch.amp.GradScaler("cuda")
    net.train()
    if len(frames) >= 5:
        for _ in range(ITERATIONS):
            idx = []
            group = []
            episode = []
            for e in range(2):
                a = active[frames[rng.integers(len(frames))]]
                if len(a) > 8:
                    a = rng.choice(a, 8, replace=False)
                for k in a:
                    idx += rng.choice(of_segment[k], 4, replace=len(of_segment[k]) < 4).tolist()
                    group += [int(k)] * 4
                    episode += [e] * 4
            gid = torch.tensor(group, device=device)
            ep = torch.tensor(episode, device=device)
            xb = _augment(_normalise(X[torch.tensor(idx)], device), True, 0.3, 0.15).contiguous(
                memory_format=torch.channels_last
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = net(xb)
            loss = sum(_contrast(z[ep == e], gid[ep == e]) for e in range(2)) / 2
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
    net.eval()
    out = []
    with torch.no_grad():
        for k in range(0, len(X), 256):
            xb = _normalise(X[k : k + 256], device).contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(F.normalize(net(xb) + net(torch.flip(xb, dims=[3])), dim=1).cpu())
    return torch.cat(out).numpy().astype(np.float16)


def load_network(path, device="cuda"):
    m = torchvision.models.resnet50(weights=None)
    m.fc = nn.Identity()
    m.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
    return m.to(device).to(memory_format=torch.channels_last).eval()


@torch.no_grad()
def features(X, network, device="cuda"):
    """(N, 2048) features of the network trained on other videos, averaged with the flipped crop, normalised."""
    out = []
    for k in range(0, len(X), 256):
        xb = _normalise(X[k : k + 256], device).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(
                F.normalize((network(xb) + network(torch.flip(xb, dims=[3]))).float(), dim=1).cpu()
            )
    return torch.cat(out).numpy().astype(np.float16)


def run(folder, stages, network_path, device="cuda", fps=20.0):
    """The three stages on one video folder (crops.npz and crops_short.npz made by crops.py); writes the stage files."""
    name = os.path.basename(os.path.normpath(folder))
    torch.backends.cudnn.benchmark = True
    network = load_network(network_path, device)
    for kind in ("classifier", "embedding", "features", "features_short"):
        os.makedirs(os.path.join(stages, kind), exist_ok=True)
    d = np.load(os.path.join(folder, "crops.npz"), allow_pickle=True)
    meta = d["meta"]
    X, share = erase_background(load_crops(d["jpeg"]), meta, device)
    lp, classes = classifier(X, meta, device, fps)
    np.savez(os.path.join(stages, "classifier", name + ".npz"), log_probs=lp, classes=classes)
    np.savez(os.path.join(stages, "embedding", name + ".npz"), embedding=embedding(X, meta, device))
    np.savez(
        os.path.join(stages, "features", name + ".npz"),
        features=features(X, network, device),
        mask_share=share,
    )
    f = os.path.join(folder, "crops_short.npz")
    if os.path.exists(f):
        d = np.load(f, allow_pickle=True)
        if len(d["meta"]):
            Xs, _ = erase_background(load_crops(d["jpeg"]), d["meta"], device)
            np.savez(
                os.path.join(stages, "features_short", name + ".npz"),
                features=features(Xs, network, device),
            )


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3])
