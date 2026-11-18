"""Check the pieces that train the box corrector: targets, track features, trees."""

import unittest

import numpy as np

from holdtrack.boxes.track_features import COLUMNS, fit_perspective, track_features
from holdtrack.boxes.training import annotated_targets, apply_trees, average, new_model


def annotation_rows(frame, person, box, category=1, flag=1):
    x1, y1, x2, y2 = box
    return [frame, person, x1, y1, x2 - x1, y2 - y1, flag, category, 1.0]


class TargetTests(unittest.TestCase):
    def test_matched_pedestrian_is_the_target(self) -> None:
        annotation = np.array(
            [
                annotation_rows(1, 7, (100, 50, 140, 150)),
                annotation_rows(1, 8, (300, 50, 340, 150), category=7),
            ]
        )
        boxes = np.array([[102.0, 52, 141, 149], [301.0, 50, 339, 150], [600.0, 50, 640, 150]])
        targets, persons = annotated_targets([1, 1, 1], boxes, annotation)
        self.assertTrue(np.allclose(targets[0], [100, 50, 140, 150]))
        self.assertEqual(persons.tolist(), [7, -1, -1])
        self.assertTrue(np.isnan(targets[1:]).all())

    def test_truncated_box_takes_the_uncovered_pedestrian(self) -> None:
        annotation = np.array([annotation_rows(1, 7, (100, 50, 140, 150))])
        boxes = np.array([[102.0, 52, 138, 90]])
        plain, _ = annotated_targets([1], boxes, annotation)
        extended, persons = annotated_targets([1], boxes, annotation, extended=True)
        self.assertTrue(np.isnan(plain).all())
        self.assertTrue(np.allclose(extended[0], [100, 50, 140, 150]))
        self.assertEqual(persons.tolist(), [7])


class TrackFeatureTests(unittest.TestCase):
    def test_perspective_line_is_recovered(self) -> None:
        rng = np.random.default_rng(0)
        bottoms = rng.uniform(300, 900, 400)
        identities = np.repeat(np.arange(40), 10)
        heights = 0.3 * bottoms + 20
        slope, offset = fit_perspective(bottoms, heights, identities)
        self.assertAlmostEqual(slope, 0.3, places=6)
        self.assertAlmostEqual(offset, 20.0, places=4)

    def test_truncated_box_stands_out_from_its_track(self) -> None:
        frames = np.arange(1, 61)
        boxes = np.c_[np.full(60, 100.0), np.full(60, 50.0), np.full(60, 140.0), np.full(60, 150.0)]
        boxes[30, 3] = 100.0
        out = track_features(frames, np.ones(60, int), boxes, fps=30)
        self.assertEqual(out.shape, (60, COLUMNS))
        self.assertAlmostEqual(float(out[30, 0]), float(np.log(100.0) - np.log(50.0)), places=5)
        self.assertAlmostEqual(float(out[0, 0]), 0.0, places=6)
        self.assertTrue(np.isnan(out[0, 16]))

    def test_covered_share_is_read_in_the_frame(self) -> None:
        frames = np.array([1, 1])
        boxes = np.array([[100.0, 50, 140, 150], [120.0, 60, 160, 170]])
        out = track_features(frames, np.array([1, 2]), boxes, fps=30)
        self.assertAlmostEqual(float(out[0, 16]), (20 * 90) / (40 * 100), places=6)
        self.assertAlmostEqual(float(out[0, 17]), 0.2, places=6)


class TreeTests(unittest.TestCase):
    def test_trees_learn_an_offset_and_keep_degenerate_boxes(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(size=(600, 3)).astype(np.float32)
        x[:, 0] = np.where(rng.random(600) > 0.5, 1.0, -1.0)
        y = 0.1 * (x[:, 0] > 0) + 0.01 * rng.normal(size=600)
        model = new_model(trees=200).fit(x.astype(np.float64), y)
        reference = np.tile(np.array([100.0, 50.0, 140.0, 150.0]), (600, 1))
        rows = np.arange(600)
        empty = np.zeros(3, bool)
        sizes = np.full(600, 40.0)
        built = {0: (x, rows, np.full(600, 2), sizes), 1: (x, rows, np.full(600, 1), sizes)}
        built[2] = (x, rows, np.full(600, 3), sizes)
        models = {0: (model, empty), 1: (model, empty), 2: (model, empty)}
        new = apply_trees(models, built, reference)
        moved = new[:, 2] - reference[:, 2]
        self.assertGreater(moved[x[:, 0] > 0].mean(), 3.0)
        self.assertLess(abs(moved[x[:, 0] <= 0].mean()), 0.5)
        self.assertTrue(np.array_equal(new[:, 0], reference[:, 0]))
        self.assertLess(new[x[:, 0] > 0, 1].mean(), reference[0, 1] - 3.0)


class AverageTests(unittest.TestCase):
    def test_trees_weigh_twice_the_networks(self) -> None:
        reference = np.array([[100.0, 50, 140, 150]])
        trees = np.array([[97.0, 50, 140, 150]])
        offsets = np.array([[0.0, 0.0, 0.3, 0.0]])
        mixed = average(reference, trees, offsets)
        self.assertTrue(np.allclose(mixed, [[98.0, 50, 144.0, 150]]))

    def test_degenerate_box_keeps_its_reference(self) -> None:
        reference = np.array([[100.0, 50, 140, 150]])
        trees = np.array([[100.0, 50, 20, 150]])
        mixed = average(reference, trees, np.zeros((1, 4)))
        self.assertTrue(np.array_equal(mixed, reference))
