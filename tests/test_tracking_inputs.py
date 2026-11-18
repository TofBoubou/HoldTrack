"""Check that detector scores, source indices and boxes reach the tracker unchanged."""

import unittest

from holdtrack.sam._sequence import _candidates
from holdtrack.sam.backend import mot_bytes


class InputTests(unittest.TestCase):
    def test_raw_scores_source_indices_and_boxes_preserved(self):
        rows = [[0, 9, -0.5, 1.0, 8.0, 7.0, 0.01], [0, 1, 0.0, 0.0, 3.0, 5.0, 0.9]]
        result = _candidates(rows, 1)
        self.assertEqual([observation.index for observation in result[0]], [1, 9])
        self.assertEqual(result[0][1].box, (-0.5, 1.0, 8.0, 7.0))
        self.assertEqual(result[0][1].confidence, 0.01)

    def test_duplicate_or_invalid_sources_fail_closed(self):
        good = [0, 1, 0, 0, 2, 2, 0.8]
        for rows in [
            [good, good],
            [[1, 1, 0, 0, 2, 2, 0.8]],
            [[0, 1, 2, 0, 0, 2, 0.8]],
            [[0, 1, 0, 0, 2, 2, float("nan")]],
        ]:
            with self.assertRaises(ValueError):
                _candidates(rows, 1)

    def test_original_raw_mask_MOT_order_and_precision(self):
        self.assertEqual(
            mot_bytes([(2, 2, 0.125, 1.0, 2.5, 5.0)]),
            b"3,2,0.1250,1.0000,2.3750,4.0000,1.000000,-1,-1,-1\n",
        )
        with self.assertRaises(ValueError):
            mot_bytes([(0, 1, 0, 0, 2, 2), (0, 1, 1, 1, 2, 2)])


if __name__ == "__main__":
    unittest.main()
