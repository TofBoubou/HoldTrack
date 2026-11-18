"""Run the native D-FINE-L deploy path at 640 pixels."""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

from holdtrack.detect.bank import DetectionBank


def predict(
    images: list[Path], checkpoint: Path, source: Path, device: str = "cuda"
) -> DetectionBank:
    """Strictly load the L architecture and preserve valid COCO-person boxes."""
    import torch
    from torchvision import transforms

    sys.path.insert(0, str(source.resolve()))
    from src.core import YAMLConfig

    configuration = YAMLConfig(
        str(source / "configs/dfine/objects365/dfine_hgnetv2_l_obj2coco.yml")
    )
    configuration.yaml_cfg["HGNetv2"]["pretrained"] = False
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    configuration.model.load_state_dict(state, strict=True)
    del state
    model = configuration.model.deploy().eval().requires_grad_(False).to(device)
    postprocessor = configuration.postprocessor.deploy()
    transform = transforms.Compose([transforms.Resize((640, 640)), transforms.ToTensor()])
    frame_indices, boxes, scores = ([], [], [])
    with torch.inference_mode():
        for frame, path in enumerate(images):
            with Image.open(path) as original:
                image = original.convert("RGB")
                width, height = image.size
                tensor = transform(image).unsqueeze(0).to(device)
            labels, predicted_boxes, confidence = postprocessor(
                model(tensor), torch.tensor([[width, height]], dtype=torch.int64, device=device)
            )
            labels = labels[0].cpu().numpy()
            predicted_boxes = predicted_boxes[0].float().cpu().numpy()
            confidence = confidence[0].float().cpu().numpy()
            person = labels == 0
            if (
                not np.isfinite(predicted_boxes[person]).all()
                or not np.isfinite(confidence[person]).all()
            ):
                raise ValueError("D-FINE person proposals must be finite")
            if ((confidence[person] < 0) | (confidence[person] > 1)).any():
                raise ValueError("D-FINE person scores must lie in [0, 1]")
            valid = (
                person
                & (confidence >= 0.01)
                & (predicted_boxes[:, 2:] > predicted_boxes[:, :2]).all(1)
            )
            boxes.append(predicted_boxes[valid])
            scores.append(confidence[valid])
            frame_indices.append(np.full(int(valid.sum()), frame, np.int64))
    return DetectionBank(
        np.concatenate(frame_indices), np.concatenate(boxes), np.concatenate(scores)
    )
