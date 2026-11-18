"""Test cache integrity and interruption handling."""

import tempfile
import unittest
from pathlib import Path

from holdtrack.cache import StageCache


class StageCacheTests(unittest.TestCase):
    def test_complete_stage_is_reused_and_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = StageCache(directory)
            calls = []

            def produce(path):
                calls.append(path)
                path.write_bytes(b"complete")

            first = cache.stage("boxes", {"thresholds": (0.3, 0.5)}, "boxes.npz", produce)
            self.assertEqual(
                first, cache.stage("boxes", {"thresholds": (0.3, 0.5)}, "boxes.npz", produce)
            )
            self.assertEqual(len(calls), 1)
            first.write_bytes(b"modified")
            with self.assertRaises(ValueError):
                cache.stage("boxes", {"thresholds": (0.3, 0.5)}, "boxes.npz", produce)

    def test_changed_inputs_create_a_new_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = StageCache(directory)
            first = cache.stage(
                "boxes", {"model": "first"}, "boxes.npz", lambda path: path.write_bytes(b"first")
            )
            second = cache.stage(
                "boxes", {"model": "second"}, "boxes.npz", lambda path: path.write_bytes(b"second")
            )
            self.assertNotEqual(first, second)
            self.assertEqual(first.read_bytes(), b"first")

    def test_failed_stage_is_not_committed(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = StageCache(directory)

            def fail(path):
                path.write_bytes(b"partial")
                raise RuntimeError("interrupted")

            with self.assertRaises(RuntimeError):
                cache.stage("trajectory", {}, "tracks.txt", fail)
            self.assertFalse(list(Path(directory).rglob("complete.json")))
            complete = cache.stage(
                "trajectory", {}, "tracks.txt", lambda path: path.write_bytes(b"complete")
            )
            self.assertEqual(complete.read_bytes(), b"complete")

    def test_work_directory_has_one_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            with StageCache(directory).locked():
                with self.assertRaises(RuntimeError):
                    with StageCache(directory).locked():
                        self.fail("A second writer acquired the cache")
