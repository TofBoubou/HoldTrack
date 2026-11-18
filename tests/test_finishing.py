"""Check the steps that finish the tracks: earlier starts, gaps, smoothing."""

import unittest

import numpy as np

from holdtrack.steps.camera_motion import (
    cumulative_motion,
    estimate_motion,
    fill_gaps,
    load_motion,
    save_motion,
    transport,
)
from holdtrack.steps.earlier_start import confirmed_mask_rows, match, rows_before_birth
from holdtrack.steps.finishing import finish, mot_lines
from holdtrack.steps.smoothing import smooth_edges

STILL = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def still_camera(frame_count):
    frames = np.arange(1, frame_count + 1)
    return cumulative_motion(frames, np.repeat(STILL[None], frame_count, 0))


def walker(first, last, identity=1, x=100.0, step=2.0):
    """Rows frame, identity, x, y, width, height of a person who walks to the right."""
    frames = np.arange(first, last + 1)
    return np.c_[
        frames,
        np.full(len(frames), identity),
        x + step * (frames - first),
        np.full(len(frames), 50.0),
        np.full(len(frames), 40.0),
        np.full(len(frames), 100.0),
    ]


class SmoothingTests(unittest.TestCase):
    def test_straight_motion_is_unchanged(self) -> None:
        rows = walker(1, 30)
        boxes = np.c_[rows[:, 2], rows[:, 3], rows[:, 2] + rows[:, 4], rows[:, 3] + rows[:, 5]]
        out = smooth_edges(rows[:, 0], rows[:, 1], boxes, fps=30)
        self.assertTrue(np.allclose(out, boxes, atol=1e-6))

    def test_jitter_is_reduced(self) -> None:
        rows = walker(1, 60)
        boxes = np.c_[rows[:, 2], rows[:, 3], rows[:, 2] + rows[:, 4], rows[:, 3] + rows[:, 5]]
        noisy = boxes + np.random.default_rng(0).normal(0, 2.0, boxes.shape)
        out = smooth_edges(rows[:, 0], rows[:, 1], noisy, fps=30)
        self.assertLess(np.abs(out - boxes).mean(), 0.6 * np.abs(noisy - boxes).mean())

    def test_short_identity_keeps_its_boxes(self) -> None:
        boxes = np.array([[0.0, 0.0, 10.0, 20.0], [3.0, 0.0, 13.0, 20.0]])
        out = smooth_edges([1, 2], [7, 7], boxes, fps=30)
        self.assertTrue(np.array_equal(out, boxes))


class CameraMotionTests(unittest.TestCase):
    def test_still_camera_leaves_the_box(self) -> None:
        box = np.array([10.0, 20.0, 50.0, 120.0])
        self.assertTrue(np.allclose(transport(still_camera(5), 1, 5, box), box))

    def test_translation_moves_the_box(self) -> None:
        shift = np.array([[1.0, 0.0, 3.0], [0.0, 1.0, -1.0]])
        motion = cumulative_motion(np.arange(1, 4), np.repeat(shift[None], 3, 0))
        moved = transport(motion, 1, 3, np.array([0.0, 0.0, 10.0, 20.0]))
        self.assertTrue(np.allclose(moved, [6.0, -2.0, 16.0, 18.0]))

    def test_unreliable_estimate_is_no_motion(self) -> None:
        shift = np.array([[1.0, 0.0, 3.0], [0.0, 1.0, 0.0]])
        motion = cumulative_motion(
            np.arange(1, 4), np.repeat(shift[None], 3, 0), counts=[50, 5, 50]
        )
        moved = transport(motion, 1, 3, np.array([0.0, 0.0, 10.0, 20.0]))
        self.assertTrue(np.allclose(moved, [3.0, 0.0, 13.0, 20.0]))

    def test_gap_is_interpolated(self) -> None:
        frames = np.array([1, 4, 30])
        identities = np.array([1, 1, 1])
        boxes = np.array([[0.0, 0, 10, 20], [30.0, 0, 40, 20], [60.0, 0, 70, 20]])
        new_frames, new_identities, new_boxes = fill_gaps(
            frames, identities, boxes, 5, still_camera(30)
        )
        self.assertEqual(new_frames.tolist(), [2, 3])
        self.assertEqual(new_identities.tolist(), [1, 1])
        self.assertTrue(np.allclose(new_boxes[:, 0], [10.0, 20.0]))


class EstimateMotionTests(unittest.TestCase):
    def test_translation_of_the_background_is_recovered(self) -> None:
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV is not installed")
        rng = np.random.default_rng(0)
        scene = cv2.GaussianBlur(rng.integers(0, 255, (400, 600)).astype(np.uint8), (0, 0), 3)
        scene = cv2.normalize(scene, None, 0, 255, cv2.NORM_MINMAX)

        def read_frame(frame):
            shift = 4 * (frame - 1)
            return scene[20 : 20 + 240, 30 + shift : 30 + shift + 320]

        rows = np.array([[1.0, 1, 100, 60, 40, 100], [2.0, 1, 100, 60, 40, 100]])
        frames, matrices, counts = estimate_motion(read_frame, 1, 3, rows)
        self.assertEqual(frames.tolist(), [1, 2, 3])
        self.assertTrue(np.allclose(matrices[0], STILL))
        self.assertTrue(np.allclose(matrices[1:, :, :2], np.eye(2), atol=0.01))
        self.assertTrue(np.allclose(matrices[1:, :, 2], [-4.0, 0.0], atol=0.3))
        self.assertTrue((counts[1:] >= 15).all())

    def test_motion_file_roundtrip(self) -> None:
        import tempfile

        shift = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0]])
        with tempfile.TemporaryDirectory() as folder:
            save_motion(
                folder + "/motion.npz", [1, 2, 3], np.repeat(shift[None], 3, 0), [0, 40, 40]
            )
            motion = load_motion(folder + "/motion.npz")
        self.assertTrue(np.allclose(motion[3][:2, 2], [4.0, 0.0]))


class EarlierStartTests(unittest.TestCase):
    def test_match_is_one_to_one(self) -> None:
        first = np.array([[0.0, 0, 10, 10], [20.0, 0, 30, 10]])
        second = np.array([[21.0, 0, 31, 10], [0.0, 1, 10, 11], [100.0, 0, 110, 10]])
        self.assertEqual(match(first, second).tolist(), [1, 0, -1])

    def test_confirmed_masks_start_the_identity_earlier(self) -> None:
        rows = walker(11, 30)
        masks = walker(1, 30, identity=5, x=80.0)
        seen = walker(4, 10, x=86.0)
        detections = (
            seen[:, 0].astype(int),
            np.c_[seen[:, 2], seen[:, 3], seen[:, 2] + seen[:, 4], seen[:, 3] + seen[:, 5]],
            np.full(len(seen), 0.4),
        )
        added = confirmed_mask_rows(rows, [masks], *detections, places=("before",))
        self.assertEqual(added[:, 0].astype(int).tolist(), list(range(4, 11)))
        self.assertTrue((added[:, 1] == 1).all())
        weak = (detections[0], detections[1], np.full(len(seen), 0.1))
        self.assertEqual(len(confirmed_mask_rows(rows, [masks], *weak)), 0)

    def test_final_identity_and_place(self) -> None:
        rows = walker(11, 30)
        masks = walker(1, 30, identity=5, x=80.0)
        seen = walker(4, 10, x=86.0)
        detections = (
            seen[:, 0].astype(int),
            np.c_[seen[:, 2], seen[:, 3], seen[:, 2] + seen[:, 4], seen[:, 3] + seen[:, 5]],
            np.full(len(seen), 0.9),
        )
        final = np.full(len(rows), 8)
        added = confirmed_mask_rows(rows, [masks], *detections, final_identities=final)
        self.assertEqual(len(added), 7)
        self.assertTrue((added[:, 1] == 8).all())
        self.assertEqual(len(confirmed_mask_rows(rows, [masks], *detections, places=("after",))), 0)

    def test_masks_before_the_first_box(self) -> None:
        rows = walker(11, 30)
        masks = walker(1, 30, identity=5, x=80.0)
        added = rows_before_birth(rows, masks, fps=30, seconds=0.2)
        self.assertEqual(sorted(added[:, 0].astype(int).tolist()), [5, 6, 7, 8, 9, 10])
        wide = masks.copy()
        wide[:10, 4] *= 3
        self.assertEqual(len(rows_before_birth(rows, wide, fps=30)), 0)


class FinishTests(unittest.TestCase):
    def test_gaps_are_filled_and_rows_are_sorted(self) -> None:
        rows = np.r_[walker(1, 10), walker(14, 30, x=126.0), walker(1, 30, identity=2, x=600.0)]
        frames, identities, boxes, counts = finish(rows, 30, 30, still_camera(30), window=0)
        self.assertEqual(counts["filled"], 3)
        self.assertEqual(len(frames), 60)
        self.assertTrue(np.array_equal(frames, np.sort(frames)))
        line = mot_lines(frames, identities, boxes)[0]
        self.assertEqual(line, "1,1,100.00,50.00,40.00,100.00,1,-1,-1,-1")

    def test_identities_must_be_unique_in_a_frame(self) -> None:
        rows = np.r_[walker(1, 5), walker(1, 5, identity=2, x=300.0)]
        with self.assertRaises(ValueError):
            finish(rows, 30, 5, still_camera(5), identities=np.ones(10, int))

    def test_camera_motion_is_required(self) -> None:
        with self.assertRaises(ValueError):
            finish(walker(1, 5), 30, 5, still_camera(3))
