"""Check identity colors and readable mask rendering."""

import unittest

import numpy as np
import pytest

from holdtrack.demo import _label, _label_font, color, overlay
from holdtrack.masks import encode


class DemoStyleTests(unittest.TestCase):
    def test_palette_order_and_cycle(self):
        self.assertEqual(color(1), (0, 114, 178))
        self.assertEqual(color(2), (230, 159, 0))
        self.assertEqual(color(7), (240, 228, 66))
        self.assertEqual(color(8), color(1))

    def test_fill_unchanged_background_and_three_pixel_contour(self):
        pytest.importorskip("cv2")
        image = np.full((80, 80, 3), 200, np.uint8)
        mask = np.zeros((80, 80), bool)
        mask[20:60, 20:60] = True
        rendered = overlay(image, [dict(object_id=1, mask=encode(mask))])
        np.testing.assert_array_equal(rendered[79, 0], image[79, 0])
        np.testing.assert_array_equal(rendered[40, 40], [124, 167, 192])
        np.testing.assert_array_equal(rendered[40, 20], [0, 91, 142])
        np.testing.assert_array_equal(rendered[40, 20], rendered[40, 22])
        self.assertFalse(np.array_equal(rendered[40, 22], rendered[40, 23]))

    def test_bold_latin_modern_on_square_white_plate(self):
        font = _label_font()
        self.assertEqual(font.getname(), ("Latin Modern Roman", "10 Bold"))
        image = np.full((100, 100, 3), 200, np.uint8)
        _label(image, "12", 20, 70, np.array(color(1)))
        changed = np.any(image != 200, axis=2)
        vertical, horizontal = np.where(changed)
        top, bottom, left, right = (
            vertical.min(),
            vertical.max(),
            horizontal.min(),
            horizontal.max(),
        )
        rgb = np.array(color(1), np.uint8)
        for row, column in ((top, left), (top, right), (bottom, left), (bottom, right)):
            np.testing.assert_array_equal(image[row, column], rgb)
        np.testing.assert_array_equal(image[top + 1, left + 1], rgb)
        np.testing.assert_array_equal(image[top + 2, left + 2], [255, 255, 255])
        self.assertTrue(
            np.any(np.all(image[top + 3 : bottom - 2, left + 3 : right - 2] == rgb, axis=2))
        )
