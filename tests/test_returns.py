"""Check the two moves that decide a return: cut, then link."""

import unittest

import numpy as np

from holdtrack.identity.returns import ReturnVideo, cut_returns, discriminant_space, link_returns
from holdtrack.steps.camera_motion import cumulative_motion

STILL = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def still_camera(frame_count):
    frames = np.arange(1, frame_count + 1)
    return cumulative_motion(frames, np.repeat(STILL[None], frame_count, 0))


def walk(first, last, x, y=100.0, step=1.0):
    frames = np.arange(first, last + 1)
    left = x + step * (frames - first)
    return frames, np.c_[left, np.full(len(frames), y), left + 40, np.full(len(frames), y + 100)]


class CutTests(unittest.TestCase):
    def test_short_absence_at_the_same_place_is_kept(self) -> None:
        f1, b1 = walk(1, 30, 100.0)
        f2, b2 = walk(50, 80, 149.0)
        frames, boxes = np.r_[f1, f2], np.r_[b1, b2]
        out, cuts = cut_returns(frames, np.ones(len(frames)), boxes, 30, still_camera(80))
        self.assertEqual(cuts, 0)
        self.assertTrue((out == 1).all())

    def test_return_far_away_is_cut(self) -> None:
        f1, b1 = walk(1, 30, 100.0)
        f2, b2 = walk(50, 80, 900.0)
        frames, boxes = np.r_[f1, f2], np.r_[b1, b2]
        out, cuts = cut_returns(frames, np.ones(len(frames)), boxes, 30, still_camera(80))
        self.assertEqual(cuts, 1)
        self.assertTrue((out[:30] == 1).all() and (out[30:] == 2).all())

    def test_long_absence_is_cut(self) -> None:
        f1, b1 = walk(1, 30, 100.0, step=0.0)
        f2, b2 = walk(120, 150, 100.0, step=0.0)
        frames, boxes = np.r_[f1, f2], np.r_[b1, b2]
        _, cuts = cut_returns(frames, np.ones(len(frames)), boxes, 30, still_camera(150))
        self.assertEqual(cuts, 1)


class SpaceTests(unittest.TestCase):
    def test_groups_are_separated(self) -> None:
        rng = np.random.default_rng(0)
        centers = rng.normal(size=(3, 16))
        labels = np.repeat(np.arange(3), 20)
        features = centers[labels] + 0.5 * rng.normal(size=(60, 16)) * np.linspace(3, 0.1, 16)
        space = discriminant_space(features, labels, dimensions=2)
        means = np.array([space[labels == g].mean(0) for g in range(3)])
        inside = np.mean(
            [np.linalg.norm(space[labels == g] - means[g], axis=1).mean() for g in range(3)]
        )
        between = np.mean([np.linalg.norm(means[a] - means[b]) for a in range(3) for b in range(a)])
        self.assertGreater(between, 2 * inside)


class LinkTests(unittest.TestCase):
    def test_tracklets_of_one_person_are_linked(self) -> None:
        rng = np.random.default_rng(1)
        frames, identities, boxes, crop_rows, features = [], [], [], [], []
        people = 8
        centers = rng.normal(size=(people, 24))
        for person in range(people):
            for part, (first, last) in enumerate(((1, 150), (211, 400))):
                f, b = walk(first, last, 50.0 + 1.5 * (first - 1), y=60.0 + 110 * person, step=1.5)
                start = len(np.concatenate(frames)) if frames else 0
                frames.append(f)
                boxes.append(b)
                identities.append(np.full(len(f), 1 + 2 * person + part))
                rows = start + np.arange(0, len(f), 5)
                crop_rows.append(rows)
                features.append(centers[person] + 0.3 * rng.normal(size=(len(rows), 24)))
        video = ReturnVideo(
            np.concatenate(frames),
            np.concatenate(identities),
            np.concatenate(boxes),
            30.0,
            1920,
            1080,
            still_camera(400),
            np.concatenate(crop_rows),
            np.concatenate(features),
        )
        names, links = link_returns({"video": video})
        self.assertEqual(len(names["video"]), len(video.frames))
        self.assertGreaterEqual(len(links["video"]), people // 2)
        for first, second in links["video"]:
            self.assertEqual((first - 1) // 2, (second - 1) // 2)
            self.assertEqual(second, first + 1)
