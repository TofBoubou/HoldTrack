"""Stream and normalize native RGB frames."""

from __future__ import annotations

import ast
import types
from pathlib import Path

from holdtrack.sam import lifecycle

SOURCES = BASE = UTILS = TRACKER_UTILS = None


def configure_sources(bindings):
    global SOURCES, BASE, UTILS, TRACKER_UTILS
    bindings.validate()
    if SOURCES is not None and (
        bindings.sam3_root.resolve() != SOURCES.sam3_root.resolve()
        or bindings.sam2_predictor.resolve() != SOURCES.sam2_predictor.resolve()
    ):
        raise RuntimeError("One fixed external source binding per process")
    SOURCES = bindings
    BASE = bindings.sam3_root / "sam3/model/sam3_tracker_base.py"
    UTILS = bindings.sam3_root / "sam3/model/utils/sam2_utils.py"
    TRACKER_UTILS = bindings.sam3_root / "sam3/model/sam3_tracker_utils.py"


def validate_options(predictor):
    expected = dict(
        image_size=1008,
        backbone_stride=14,
        low_res_mask_size=288,
        num_maskmem=7,
        memory_temporal_stride_for_eval=1,
        max_cond_frames_in_attn=6,
        max_obj_ptrs_in_encoder=16,
        use_memory_selection=False,
        keep_first_cond_frame=False,
        non_overlap_masks_for_mem_enc=False,
        non_overlap_masks_for_output=False,
        multimask_output_in_sam=True,
        multimask_output_for_tracking=True,
        multimask_min_pt_num=0,
        multimask_max_pt_num=1,
        fill_hole_area=8,
        hidden_dim=256,
        mem_dim=64,
    )
    actual = {key: getattr(predictor, key) for key in expected}
    if actual != expected:
        raise ValueError(f"SAM3/LiAM effective options differ: {actual}")
    decoder = predictor.sam_mask_decoder
    for key, value in dict(
        num_multimask_outputs=3,
        dynamic_multimask_via_stability=True,
        dynamic_multimask_stability_delta=0.05,
        dynamic_multimask_stability_thresh=0.98,
    ).items():
        if getattr(decoder, key) != value:
            raise ValueError(f"SAM3 decoder option differs: {key}")
    if predictor.training:
        raise ValueError("SAM3 exploration is inference only")
    return dict(
        actual,
        model="SAM3 tracker-only",
        source_commit=SOURCES.commit,
        postprocess="SAM2 pinned fill-holes only, area8; no SAM3 sprinkle removal",
    )


class LazyFrames:
    def __init__(self, paths, image_size, torch) -> None:
        import numpy as np
        from PIL import Image

        if not paths:
            raise ValueError("Empty frame population")
        self.frame_paths = tuple(Path(p) for p in paths)
        self.image_size = int(image_size)
        self.loader = lifecycle.extracted(
            UTILS, None, "_load_img_as_tensor", dict(Image=Image, np=np, torch=torch)
        )
        self.mean = torch.tensor((0.5, 0.5, 0.5), dtype=torch.float32)[:, None, None]
        self.std = self.mean.clone()
        self.cache_index = None
        self.cache_image = None
        self.cache_hits = self.cache_misses = self.peak_resident_bytes = 0
        self.loaded_frame_indices = []
        self.video_height = self.video_width = None
        self[0]

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, index):
        if not isinstance(index, int) or not 0 <= index < len(self):
            raise IndexError(index)
        if self.cache_index == index:
            self.cache_hits += 1
            return self.cache_image
        image, height, width = self.loader(str(self.frame_paths[index]), self.image_size)
        if self.video_height is not None and (height, width) != (
            self.video_height,
            self.video_width,
        ):
            raise ValueError("Frame dimensions changed")
        image -= self.mean
        image /= self.std
        self.video_height, self.video_width = (height, width)
        self.cache_index, self.cache_image = (index, image)
        self.loaded_frame_indices.append(index)
        self.cache_misses += 1
        self.peak_resident_bytes = max(
            self.peak_resident_bytes, image.numel() * image.element_size()
        )
        return image


def frame_loader(paths, torch, *, mirror=False):
    paths = tuple(Path(p) for p in paths)

    class MirroredFrames(LazyFrames):
        def __init__(self, *args):
            self.mirror_index = self.mirror_image = None
            super().__init__(*args)

        def __getitem__(self, index):
            image = super().__getitem__(index)
            if self.mirror_index != index:
                self.mirror_index = index
                self.mirror_image = torch.flip(image, (-1,))
            return self.mirror_image

    def load(
        video_path,
        image_size,
        offload_video_to_cpu,
        async_loading_frames=False,
        compute_device=None,
        **kwargs,
    ):
        if not paths or any(p.parent.resolve() != Path(video_path).resolve() for p in paths):
            raise ValueError("Frame population differs from the requested source")
        if not offload_video_to_cpu or async_loading_frames or kwargs:
            raise ValueError("Only bounded official SAM3 CPU frame streaming is supported")
        images = (MirroredFrames if mirror else LazyFrames)(paths, image_size, torch)
        return (images, images.video_height, images.video_width)

    return load


def install(predictor, *, paths, fill_holes, torch, mirror=False):
    concat = lifecycle.extracted(BASE, None, "concat_points", dict(torch=torch))
    record = lifecycle.install(
        predictor,
        frame_loader=frame_loader(paths, torch, mirror=mirror),
        concat_points=concat,
        fill_holes=fill_holes,
        torch_module=torch,
        sources=SOURCES,
    )
    node = lifecycle.definition(
        SOURCES.sam2_predictor, "SAM2VideoPredictor", "_consolidate_temp_output_across_obj"
    )
    replacements = 0
    for item in ast.walk(node):
        if (
            isinstance(item, ast.Assign)
            and isinstance(item.value, ast.BinOp)
            and (
                ast.dump(item.value)
                == ast.dump(ast.parse("self.image_size // 4", mode="eval").body)
            )
        ):
            item.value = ast.parse("self.low_res_mask_size", mode="eval").body
            replacements += 1
    if replacements != 1:
        raise ValueError("Pinned consolidation shape ABI changed")
    scope = dict(torch=torch, NO_OBJ_SCORE=-1024.0)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
            str(SOURCES.sam2_predictor),
            "exec",
        ),
        scope,
    )
    predictor._consolidate_temp_output_across_obj = types.MethodType(scope[node.name], predictor)
    record.update(
        consolidation_mask_grid="native low_res_mask_size, not image_size//4",
        streaming_normalization="SAM3 RGB mean/std0.5; upstream float64 before .float()",
        input_horizontal_reflection=bool(mirror),
    )
    return record
