"""Check the evaluation on synthetic MOT tracks."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

from holdtrack.evaluation import evaluate_tracks


@unittest.skipUnless(importlib.util.find_spec("trackeval"), "Install the eval extra")
class NativeEvaluationTests(unittest.TestCase):
    def test_perfect_tracks_and_identity_switch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            truth = root / "truth.mot"
            tracks = root / "tracks.mot"
            truth.write_text("1,1,0,0,10,20,1,1,1\n2,1,1,0,10,20,1,1,1\n")
            tracks.write_text("1,7,0,0,10,20,1,-1,-1,-1\n2,7,1,0,10,20,1,-1,-1,-1\n")
            perfect = evaluate_tracks(tracks, truth, num_frames=4)
            self.assertEqual(set(perfect), {"HOTA", "DetA", "AssA", "MOTA", "IDF1"})
            self.assertTrue(all(value == 1 for value in perfect.values()))
            tracks.write_text("1,7,0,0,10,20,1,-1,-1,-1\n2,8,1,0,10,20,1,-1,-1,-1\n")
            switched = evaluate_tracks(tracks, truth)
            self.assertEqual(switched["DetA"], 1)
            self.assertEqual(switched["MOTA"], 0.5)
            self.assertLess(switched["AssA"], 1)
            self.assertLess(switched["IDF1"], 1)

    def test_explicit_sequence_length_rejects_outside_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            truth = Path(directory) / "truth.mot"
            tracks = Path(directory) / "tracks.mot"
            truth.write_text("1,1,0,0,10,20,1,1,1\n")
            tracks.write_text("2,1,0,0,10,20,1,-1,-1,-1\n")
            with self.assertRaises(ValueError):
                evaluate_tracks(tracks, truth, num_frames=1)
