"""Check the rendering of tracks and masks on a video."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from holdtrack.masks import encode


@unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV is optional outside the SAM extra")
class DemoTests(unittest.TestCase):
    def test_real_MP4_roundtrip(self):
        import cv2

        from holdtrack.demo import render

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frames = root / "frames"
            masks = root / "masks"
            frames.mkdir()
            masks.mkdir()
            mask = np.zeros((64, 96), bool)
            mask[12:44, 20:40] = True
            for number in range(1, 4):
                Image.fromarray(np.full((64, 96, 3), 32, np.uint8)).save(
                    frames / f"{number:08d}.png"
                )
                (masks / f"{number:08d}.json").write_text(
                    json.dumps({"objects": [{"object_id": 7, "mot_id": 7, "mask": encode(mask)}]})
                )
            output = render(frames, masks, root / "demo.mp4", fps=20)
            cap = cv2.VideoCapture(str(output))
            self.assertTrue(cap.isOpened())
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 3)
            ok, image = cap.read()
            self.assertTrue(ok)
            self.assertEqual(image.shape, (64, 96, 3))
            cap.release()
