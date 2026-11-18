"""Check the run-length format of the output masks."""

import unittest

import numpy as np

from holdtrack.masks import decode, encode


class MaskFormatTests(unittest.TestCase):
    def test_column_major_roundtrip(self) -> None:
        for mask in [
            np.zeros((3, 4), bool),
            np.ones((3, 4), bool),
            np.eye(4, dtype=bool),
            np.array([[1, 0], [1, 1]], bool),
        ]:
            self.assertTrue(np.array_equal(mask, decode(encode(mask))))

    def test_counts_have_background_first(self) -> None:
        self.assertEqual(encode(np.ones((2, 3), bool))["counts"], [0, 6])

    def test_invalid_mask_fails(self) -> None:
        with self.assertRaises(ValueError):
            decode({"size": [2, 2], "counts": [3]})
