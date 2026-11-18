"""Run the two RF-DETR architectures with native postprocessing."""

from pathlib import Path

import numpy as np
from PIL import Image

from holdtrack.detect.bank import DetectionBank


def predict(
    images: list[Path], checkpoint: Path, specification: dict, device: str = "cuda"
) -> DetectionBank:
    """Load a strict person detector and keep its native ordered person proposals."""
    import torch
    from rfdetr import RFDETR2XLarge
    from rfdetr.config import RFDETRLargeDeprecatedConfig, TrainConfig
    from rfdetr.datasets.coco import make_coco_transforms_square_div_64
    from rfdetr.detr import _namespace_from_configs
    from rfdetr.models.lwdetr import build_criterion_and_postprocessors, build_model

    configuration_class = (
        RFDETRLargeDeprecatedConfig
        if specification["architecture"] == "RFDETRLargeDeprecated"
        else RFDETR2XLarge._model_config_class
    )
    configuration = configuration_class(
        **dict(specification["model_config"], device=device, pretrain_weights=None)
    )
    training = TrainConfig(
        dataset_dir=".",
        output_dir=".",
        class_names=["person"],
        multi_scale=False,
        tensorboard=False,
        wandb=False,
    )
    namespace = _namespace_from_configs(configuration, training)
    namespace.pretrained_encoder = False
    model = build_model(namespace)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(state, strict=True)
    del state
    model.eval().requires_grad_(False).to(device)
    postprocessor = build_criterion_and_postprocessors(namespace)[1].eval()
    transform = make_coco_transforms_square_div_64(
        "val",
        configuration.resolution,
        multi_scale=False,
        patch_size=configuration.patch_size,
        num_windows=configuration.num_windows,
    )
    frame_indices, boxes, scores = ([], [], [])
    with torch.inference_mode():
        for frame, path in enumerate(images):
            with Image.open(path) as original:
                image = original.convert("RGB")
                width, height = image.size
                target = dict(
                    boxes=torch.empty(0, 4),
                    labels=torch.empty(0, dtype=torch.long),
                    area=torch.empty(0),
                    iscrowd=torch.empty(0, dtype=torch.long),
                    orig_size=torch.tensor([height, width]),
                    size=torch.tensor([height, width]),
                )
                tensor, _ = transform(image, target)
            if tuple(tensor.shape) != (3, configuration.resolution, configuration.resolution):
                raise ValueError("RF-DETR preprocessing produced an unexpected shape")
            prediction = model(tensor.unsqueeze(0).to(device))
            restored = postprocessor(prediction, torch.tensor([[height, width]], device=device))[0]
            person = restored["labels"] == 0
            boxes.append(restored["boxes"][person].float().cpu().numpy())
            scores.append(restored["scores"][person].float().cpu().numpy())
            frame_indices.append(np.full(int(person.sum()), frame, np.int64))
    return DetectionBank(
        np.concatenate(frame_indices), np.concatenate(boxes), np.concatenate(scores)
    )
