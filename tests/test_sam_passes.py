"""Check the clock, support and pixel conventions of the SAM passes on synthetic inputs."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from holdtrack.detect.bank import DetectionBank
from holdtrack.inputs.mot import physical
from holdtrack.inputs.video import prepare_video
from holdtrack.masks import decode
from holdtrack.sam.passes import ROLES, SamAssets, prepare_passes, run_prepared_passes
from holdtrack.sam.transforms import (
    reflect_boxes,
    reflect_mot,
    restore_mirrored_support,
    restore_reverse_clock,
    reverse_bank,
)


class TransformTests(unittest.TestCase):
    def test_native_tensor_reflection_is_cached_and_does_not_mutate_the_original(self):
        from holdtrack.sam import runtime

        class Frames:
            def __init__(self, paths, image_size, torch):
                self.video_height, self.video_width = 2, 3
                self.image = np.arange(6).reshape(1, 2, 3)
                self[0]

            def __getitem__(self, _):
                return self.image

        flips = []

        def flip(image, axes):
            flips.append(image.copy())
            return np.flip(image, axis=axes)

        with patch.object(runtime, "LazyFrames", Frames):
            loader = runtime.frame_loader(
                [Path("/frames/0001.png")], SimpleNamespace(flip=flip), mirror=True
            )
            frames, _, _ = loader("/frames", 1008, True)
            first = frames[0]
            self.assertIs(first, frames[0])
            np.testing.assert_array_equal(first, np.arange(6).reshape(1, 2, 3)[:, :, ::-1])
            np.testing.assert_array_equal(frames.image, np.arange(6).reshape(1, 2, 3))
            self.assertEqual(len(flips), 1)

    def test_reverse_keeps_proposal_order_and_original_source_indices(self):
        bank = DetectionBank(
            np.array([0, 0, 2, 1]), np.array([[i, 0, i + 1, 2] for i in range(4)]), np.ones(4)
        )
        reversed_bank, source = reverse_bank(bank, 3)
        self.assertEqual(source.tolist(), [2, 3, 0, 1])
        self.assertEqual(reversed_bank.frames.tolist(), [0, 1, 2, 2])
        self.assertTrue(np.array_equal(reversed_bank.boxes, bank.boxes[source]))
        with self.assertRaises(ValueError):
            reverse_bank(bank, 2)

    def test_mirror_restores_literal_native_support_and_fails_on_wrong_support(self):
        original = b"1,3,-0.12500000,2.000,4.25000000,8.0,1.000,-1,-1,-1\r\n"
        mirrored = reflect_mot(original, 12).replace(b",3,", b",91,", 1)
        self.assertEqual(
            restore_mirrored_support(original, mirrored, 12), original.replace(b",3,", b",91,", 1)
        )
        with self.assertRaises(ValueError):
            restore_mirrored_support(original, mirrored.replace(b"8.0", b"8.1"), 12)
        with self.assertRaises(ValueError):
            restore_mirrored_support(original, mirrored + mirrored, 12)
        boxes = np.array([[-0.25, 2, 10.5, 20]])
        np.testing.assert_array_equal(reflect_boxes(reflect_boxes(boxes, 10), 10), boxes)

    def test_reverse_MOT_clock_roundtrip_keeps_every_nonclock_token(self):
        payload = b"1,7,1,2,3,4,.8,-1,-1,-1\n3,9,-1,0,2,3,.6,-1,-1,-1\n"
        backwards = restore_reverse_clock(payload, 4)
        self.assertEqual([r[0] for _, r in physical(backwards)], [1, 3])
        self.assertEqual(restore_reverse_clock(backwards, 4), payload)
        with self.assertRaises(ValueError):
            restore_reverse_clock(payload, 2)


class VideoTests(unittest.TestCase):
    def test_folder_order_pixels_dimensions_and_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, value in [("frame10.png", 10), ("frame2.png", 2), ("frame1.png", 1)]:
                Image.fromarray(np.full((3, 8, 3), value, np.uint8)).save(root / name)
            result = prepare_video(root, root / "unused", fps=25)
            self.assertEqual(
                [p.name for p in result.paths], ["frame1.png", "frame2.png", "frame10.png"]
            )
            self.assertEqual((result.width, result.height, result.fps), (8, 3, 25))
            self.assertFalse(result.decoded)
            Image.new("RGB", (9, 3)).save(root / "frame11.png")
            with self.assertRaises(ValueError):
                prepare_video(root, root / "unused")

    def test_mp4_decode_is_lossless_after_decode_and_preserves_every_frame(self):
        images = [np.full((3, 8, 3), [i, 50, 90], np.uint8) for i in [1, 2, 3]]

        class Reader:
            def __init__(self, _):
                self.index = 0
                self.closed = False

            def isOpened(self):
                return True

            def get(self, key):
                return 29.97 if key == 1 else 3

            def read(self):
                if self.index == len(images):
                    return False, None
                image = images[self.index]
                self.index += 1
                return True, image

            def release(self):
                self.closed = True

        cv2 = SimpleNamespace(
            VideoCapture=Reader,
            CAP_PROP_FPS=1,
            CAP_PROP_FRAME_COUNT=2,
            COLOR_BGR2RGB=3,
            cvtColor=lambda a, _: a[:, :, ::-1],
        )
        with tempfile.TemporaryDirectory() as directory, patch.dict("sys.modules", cv2=cv2):
            root = Path(directory)
            (root / "clip.mp4").write_bytes(b"decoder fixture")
            result = prepare_video(root / "clip.mp4", root / "decoded")
            self.assertTrue(result.decoded)
            self.assertEqual(result.fps, 29.97)
            for path, bgr in zip(result.paths, images, strict=True):
                np.testing.assert_array_equal(np.asarray(Image.open(path)), bgr[:, :, ::-1])


class PassOrchestrationTests(unittest.TestCase):
    def test_crossed_forward_support_never_changes_the_SAM_birth_bank(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "pixels"
            images.mkdir()
            for frame in range(3):
                Image.new("RGB", (8, 3)).save(images / f"{frame + 1:08d}.png")
            checkpoint = root / "weights"
            checkpoint.write_bytes(b"fixture")
            assets = SamAssets(checkpoint, "0" * 64, device="cpu")
            seed_bank = DetectionBank(np.arange(3), np.array([[1, 0, 3, 3]] * 3), np.full(3, 0.8))
            support_bank = DetectionBank(
                np.arange(3), np.array([[4, 0, 7, 3]] * 3), np.full(3, 0.9)
            )
            seed = b"".join(f"{f},1,1,0,2,3,1,-1,-1,-1\n".encode() for f in range(1, 4))
            support = b"".join(f"{f},77,4,0,3,3,1,-1,-1,-1\n".encode() for f in range(1, 4))
            (root / "seed.txt").write_bytes(seed)
            (root / "support.txt").write_bytes(support)
            seen = []

            def execute(manifest, output, validate_inputs_only=False, **_):
                data = json.loads(Path(manifest).read_text())
                if validate_inputs_only:
                    return dict(model_loaded=False, frame_count=3)
                rows = json.loads(Path(data["raw_detections"]["path"]).read_text())
                seen.append((data["voice"], rows))
                output = Path(output)
                output.mkdir(parents=True)
                # Raw mask boxes follow the seed detector, not the alternate bank.
                native = Path(data["bytetrack"]["path"]).read_bytes()
                (output / "sam.txt").write_bytes(native)
                (output / "mot.txt").write_bytes(native)
                return dict(
                    sam=str(output / "sam.txt"), mot=str(output / "mot.txt"), voice=data["voice"]
                )

            with (
                patch.object(
                    SamAssets,
                    "validate",
                    return_value=(
                        checkpoint,
                        SimpleNamespace(sam3_root=root, sam2_predictor=root / "predictor.py"),
                    ),
                ),
                patch("holdtrack.sam._sequence._run_prepared", execute),
                patch(
                    "holdtrack.sam.passes._associations",
                    side_effect=lambda bank, seq, path: (path.write_bytes(seed), path)[1],
                ),
            ):
                plan = prepare_passes(
                    images,
                    seed_bank,
                    root / "run",
                    assets=assets,
                    roles=ROLES,
                    associations=root / "seed.txt",
                    support_detections=support_bank,
                    support_associations=root / "support.txt",
                )
                outputs = run_prepared_passes(plan, export_masks=False)
            self.assertEqual(len(seen), 5)
            for _, rows in seen[:3]:
                self.assertEqual([r[2:6] for r in rows], [[1.0, 0.0, 3.0, 3.0]] * 3)
            self.assertEqual([r[2:6] for r in seen[-1][1]], [[5.0, 0.0, 7.0, 3.0]] * 3)
            for role in ("reverse_BASE", "mirror_EXCL"):
                self.assertEqual(outputs[role].projected.read_bytes(), seed)
                self.assertFalse(
                    json.loads(outputs[role].receipt.read_text())["crossed_projection"]
                )
            for output in (outputs[role] for role in ("BASE", "EXCL", "PROT")):
                self.assertEqual(output.source_projected.read_bytes(), seed)
                # Every non-ID byte belongs to2XL; projecting the intermediate
                # L support instead of raw masks must not substitute its boxes.
                actual = [line.split(b",") for line in output.projected.read_bytes().splitlines()]
                expected = [line.split(b",") for line in support.splitlines()]
                self.assertEqual([x[:1] + x[2:] for x in actual], [x[:1] + x[2:] for x in expected])
                self.assertTrue(json.loads(output.receipt.read_text())["crossed_projection"])

    def test_five_pass_plan_reverse_support_mirror_masks_and_real_writers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "pixels"
            images.mkdir()
            for frame in range(3):
                Image.fromarray(np.full((3, 8, 3), frame, np.uint8)).save(
                    images / f"{frame + 1:08d}.png"
                )
            checkpoint = root / "weights"
            checkpoint.write_bytes(b"fixture")
            assets = SamAssets(checkpoint, "0" * 64, device="cpu")
            bank = DetectionBank(
                np.array([0, 1, 2]), np.array([[1, 0, 3, 3]] * 3), np.array([0.8] * 3)
            )
            associated_banks = []

            def associate(selected, sequence, destination):
                associated_banks.append(selected)
                destination.write_bytes(
                    b"".join(
                        f"{int(f) + 1},1,1.00000000,0,2.00000000,3,1,-1,-1,-1\n".encode()
                        for f in selected.frames
                    )
                )
                return destination

            def execute(manifest, output, validate_inputs_only=False, on_frame=None):
                data = json.loads(Path(manifest).read_text())
                if validate_inputs_only:
                    return dict(model_loaded=False, frame_count=data["frame_count"])
                output = Path(output)
                output.mkdir(parents=True)
                native = Path(data["bytetrack"]["path"]).read_bytes()
                (output / "sam.txt").write_bytes(native)
                (output / "mot.txt").write_bytes(native)
                for frame in range(data["frame_count"]):
                    mask = np.zeros((3, 8), bool)
                    mask[:, 0] = True
                    on_frame(frame, {1: mask}, {1: 0.9}, [(frame, 1, 0, 0, 1, 3)])
                return dict(
                    sam=str(output / "sam.txt"), mot=str(output / "mot.txt"), voice=data["voice"]
                )

            with (
                patch.object(
                    SamAssets,
                    "validate",
                    return_value=(
                        checkpoint,
                        SimpleNamespace(sam3_root=root, sam2_predictor=root / "predictor.py"),
                    ),
                ),
                patch("holdtrack.sam.passes._associations", associate),
                patch("holdtrack.sam._sequence._run_prepared", execute),
            ):
                plan = prepare_passes(images, bank, root / "run", assets=assets)
                result = run_prepared_passes(plan)
            self.assertEqual(tuple(result), ROLES)
            self.assertEqual(len(associated_banks), 2)
            # Reverse source order and decoded pixel order must both be inverted.
            p = json.loads(plan.read_text())
            m = json.loads(Path(p["manifests"]["reverse_BASE"]["path"]).read_text())
            raw = json.loads(Path(m["raw_detections"]["path"]).read_text())
            self.assertEqual([r[1] for r in raw], [2, 1, 0])
            self.assertEqual(
                [int(np.asarray(Image.open(x["path"]))[0, 0, 0]) for x in m["images"]], [2, 1, 0]
            )
            self.assertEqual(
                result["mirror_EXCL"].projected.read_bytes(), result["BASE"].projected.read_bytes()
            )
            for value in result.values():
                self.assertEqual(len(list(value.masks.glob("*.json"))), 3)
                self.assertTrue(value.receipt.is_file())
            mirror = json.loads((result["mirror_EXCL"].masks / "00000001.json").read_text())
            mask = decode(mirror["objects"][0]["mask"])
            self.assertTrue(mask[:, -1].all())
            self.assertFalse(mask[:, 0].any())
            self.assertTrue((root / "run/receipt.json").is_file())


if __name__ == "__main__":
    unittest.main()
