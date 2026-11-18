"""Check the parsing of track files and the counts used by the evaluation."""

import unittest

import numpy as np

from holdtrack.geometry import boxes_iou, match
from holdtrack.inputs.mot import physical
from holdtrack.steps.reliability import subset


class NumericContracts(unittest.TestCase):
    def test_byte_subset_and_zero_based_parser(self):
        raw = b"1,7,1,2,3,4,.8,-1,-1,-1\r\n2,7,2,3,3,4,.7,-1,-1,-1"
        self.assertEqual(physical(raw)[0][1], (0, 7, 1.0, 2.0, 4.0, 6.0))
        self.assertEqual(subset(raw, np.array([True, False])), raw.splitlines(keepends=True)[0])
        with self.assertRaises(ValueError):
            physical(raw + b"\n" + raw.splitlines()[0])

    def test_eligible_cardinality_before_iou_sum(self):
        a, b = match(np.array([[0.99, 0.7], [0.71, 0.0]]), 0.7)
        self.assertEqual(list(zip(a, b)), [(0, 1), (1, 0)])
        np.testing.assert_array_equal(boxes_iou([[0, 0, 2, 2]], [[0, 0, 2, 2]]), [[1.0]])


if __name__ == "__main__":
    unittest.main()
