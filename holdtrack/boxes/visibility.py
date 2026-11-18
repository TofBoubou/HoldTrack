"""Occlusion measures of each track row, from the boxes and the tracker masks alone.

Ten columns per row:
  0-3  overhang of the corrected box beyond the box of the mask (median of the three
       forward passes) at the left, top, right and bottom edges, counted outward, as a
       share of the box width (left, right) or height (top, bottom);
  4    visible share: area of (mask box ∩ corrected box) / area of the corrected box;
  5-8  occlusion flag of each edge (0 free, 1 a neighbor in front covers the edge,
       2 a neighbor behind, 3 image border);
  9    contact: largest IoU with another box of the frame.
Rows without a mask in any pass are NaN in columns 0 to 4.
"""

from __future__ import annotations

import numpy as np


def visibility(
    boxes: np.ndarray, mask_boxes: np.ndarray, edge_flags: np.ndarray, contact: np.ndarray
) -> np.ndarray:
    """boxes: (N, 4) corrected boxes as left, top, right, bottom; mask_boxes: (3, N, 4)."""
    b = np.asarray(boxes, float)
    m = np.nanmedian(np.asarray(mask_boxes, float), 0)
    w = b[:, 2] - b[:, 0]
    h = b[:, 3] - b[:, 1]
    over = np.stack(
        [
            (m[:, 0] - b[:, 0]) / w,
            (m[:, 1] - b[:, 1]) / h,
            (b[:, 2] - m[:, 2]) / w,
            (b[:, 3] - m[:, 3]) / h,
        ],
        1,
    )
    iw = np.clip(np.minimum(b[:, 2], m[:, 2]) - np.maximum(b[:, 0], m[:, 0]), 0, None)
    ih = np.clip(np.minimum(b[:, 3], m[:, 3]) - np.maximum(b[:, 1], m[:, 1]), 0, None)
    visible = iw * ih / (w * h)
    return np.concatenate(
        [over, visible[:, None], np.asarray(edge_flags, float), np.asarray(contact)[:, None]], 1
    ).astype(np.float32)
