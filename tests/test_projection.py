"""Check the projection of tracker identities onto detector trajectories."""

import unittest

from holdtrack.sam.projection import project


class ProjectionContracts(unittest.TestCase):
    def test_exact_non_identity_tokens_and_missing_sam_anchor(self):
        bt = b"1,7,1.00000000,2,3,4,.8,-1,-1,-1\r\n2,7,1,2,3,4,.7,-1,-1,-1\r\n"
        sam = b"1,123,1,2,3,4,1,-1,-1,-1\n"
        out, _ = project(bt, sam)
        for a, b in zip(bt.splitlines(), out.splitlines()):
            self.assertEqual(
                a.split(b",")[:1] + a.split(b",")[2:], b.split(b",")[:1] + b.split(b",")[2:]
            )
        self.assertEqual([row.split(b",")[1] for row in out.splitlines()], [b"123", b"123"])

    def test_empty_prediction_does_not_invent_an_anchor(self):
        out, _ = project(b"1,7,1,2,3,4,.8,-1,-1,-1\n", b"")
        self.assertEqual(out, b"1,7,1,2,3,4,.8,-1,-1,-1\n")

    def test_empty_detector_preserves_empty_support(self):
        self.assertEqual(project(b"", b"")[0], b"")
