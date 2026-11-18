"""Check the training of a sequence network of the box corrector on small synthetic tables."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from holdtrack.boxes import network_training, networks

VIEWS = 3


def write_tables(folder: Path, video: str, rows: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    frame = np.arange(1, rows + 1)
    x = 100 + frame * 0.5
    ref = np.stack([x, np.full(rows, 50.0), x + 40, np.full(rows, 150.0)], 1).astype(np.float32)
    gt = ref + np.array([-2.0, 0, 2, 4], np.float32)
    gt[::7] = np.nan
    views = ref[None] + rng.normal(0, 1, (VIEWS, rows, 4)).astype(np.float32)
    views[1, ::5] = np.nan
    np.savez(
        folder / f"{video}.npz",
        frame=frame,
        pid=np.ones(rows, np.int64),
        ref=ref,
        gt=gt,
        views=views,
        views_score=rng.random((VIEWS, rows)).astype(np.float32),
        views_iou=rng.random((VIEWS, rows)).astype(np.float32),
    )
    masks = ref[None] + rng.normal(0, 1, (3, rows, 4)).astype(np.float32)
    np.savez(
        folder / f"{video}_masks.npz",
        masks=masks,
        masks_iou=rng.random((3, rows)).astype(np.float32),
    )


class NetworkTrainingTests(unittest.TestCase):
    def test_targets_are_relative_signed_outward_and_clipped(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            write_tables(folder, "a", 20, 0)
            data = networks.load_tables(folder, VIEWS)
            Y = network_training.load_targets(folder, data, clip=0.03)
        self.assertTrue(np.isnan(Y[0]).all())
        self.assertTrue(np.allclose(Y[1], [0.03, 0.0, 0.03, 0.03]))

    def test_mirror_exchanges_left_and_right_edges(self) -> None:
        columns = network_training._mirror_columns(2)
        self.assertEqual(columns[:7], [2, 1, 0, 3, 4, 5, 6])
        self.assertEqual(columns[14:20], [16, 15, 14, 17, 18, 19])
        self.assertEqual(sorted(columns), list(range(2 * 7 + 3 * 6)))

    def test_trained_weights_are_read_by_predict(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d) / "tables"
            folder.mkdir()
            write_tables(folder, "a", 400, 0)
            write_tables(folder, "b", 400, 1)
            weights = network_training.train(
                folder, Path(d) / "network.pt", views=VIEWS, epochs=2, device="cpu"
            )
            first = networks.predict(folder, weights, views=VIEWS)
            again = network_training.train(
                folder, Path(d) / "again.pt", views=VIEWS, epochs=2, device="cpu"
            )
            second = networks.predict(folder, again, views=VIEWS)
        self.assertEqual(first.shape, (800, 4))
        self.assertTrue(np.isfinite(first).all())
        self.assertTrue(np.array_equal(first, second))


if __name__ == "__main__":
    unittest.main()
