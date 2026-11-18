"""Estimate temporal reliability from ten observation features."""

import numpy as np
from scipy.special import expit

THRESHOLDS = [0.5, 0.75, 0.9, 0.95, 0.99, "KEEP_ALL"]


def predict(model, X):
    X = np.asarray(X, float)
    if X.ndim != 2 or X.shape[1] != 10 or (not np.isfinite(X).all()):
        raise ValueError("All-row prediction ABI")
    b = np.asarray(model["beta"])
    return expit(b[0] + (X - model["mean"]) / model["std"] @ b[1:])


def keep_mask(probability, threshold):
    p = np.asarray(probability, float)
    if p.ndim != 1 or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Bad probability array")
    if threshold == "KEEP_ALL":
        return np.ones(len(p), bool)
    if threshold not in THRESHOLDS[:-1]:
        raise ValueError("Only fixed thresholds")
    return p <= threshold


def subset(source, mask):
    lines = source.splitlines(keepends=True)
    mask = np.asarray(mask)
    if mask.dtype != bool or mask.shape != (len(lines),):
        raise ValueError("Mask universe changed")
    return b"".join((line for line, keep in zip(lines, mask) if keep))
