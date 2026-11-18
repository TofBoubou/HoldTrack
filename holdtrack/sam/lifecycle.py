"""Connect the native SAM tracking lifecycle."""

from __future__ import annotations

import ast
import hashlib
import types
from collections import OrderedDict
from pathlib import Path


def definition(path, cls, name):
    tree = ast.parse(Path(path).read_text())
    parent = (
        tree
        if cls is None
        else next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    )
    return next(
        n
        for n in parent.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
    )


class SourceBindings:
    def __init__(self, sam3_root, sam2_predictor) -> None:
        self.sam3_root = Path(sam3_root)
        self.sam2_predictor = Path(sam2_predictor)
        self.commit = COMMIT

    def validate(self):
        for relative, digest in SAM3_SOURCES.items():
            if hashlib.sha256((self.sam3_root / relative).read_bytes()).hexdigest() != digest:
                raise ValueError("Official SAM 3 source mismatch: " + relative)
        if hashlib.sha256(self.sam2_predictor.read_bytes()).hexdigest() != SAM2_PREDICTOR_SHA256:
            raise ValueError("Official SAM 2 lifecycle source mismatch")


COMMIT = "7f473cfd983d5eb43040ff608f66bff0d7bdcb53"
SAM3_SOURCES = {
    "README.md": "da3dd047d18ef9a4b33557bf6f8187cf166218dc8aef40f2e18a67cc5249a502",
    "LICENSE": "4dea99bfaa016e21bc860d73f344236bd1e5c4977d1a9a8fd32f822b500ae1be",
    "pyproject.toml": "255f5d8d1db011459878e3de296a6afc84e03a8dda2bfa756a4de304aaff6366",
    "sam3/__init__.py": "ca3821a125f767e6a35188603b51ef2501b1e7c3d1a90546ac567c5d2a80b9fb",
    "sam3/model_builder.py": "60cef72b1f7e4a2162d1c3f1ad2c0451bbbdc10e3484433a9aa2d016fed6b448",
    "sam3/model/sam3_tracking_predictor.py": "145df76a5c045605d4d15360456015adb4f987c4b18e82f4f4876a63e6a32345",
    "sam3/model/sam3_tracker_base.py": "b2b52409c002e1590262375aa794f8ab67e7476f42f8fee41a76de0c14aa62e2",
    "sam3/model/sam3_tracker_utils.py": "dc5fdeba2d4416f273394a9bd4450dff050608db6009a4546ca68adbaa24a640",
    "sam3/model/utils/sam2_utils.py": "78048b7b131250ac14b19c0c6f5fdb21402542b281964af675247e19734e5b2e",
    "sam3/model/memory.py": "6712af0db3bed4e2dffd812884d42db48e6d3572bb1273505745dda8b900e0e3",
    "sam3/model/decoder.py": "a282d5c7303b8b22541e14bf198f26019a2da4b6ee23c165838ddd9a9312044b",
    "sam3/sam/mask_decoder.py": "8ab5dc1fe9443ce31deadd69c3666470af89dc6d6f3dec678bf962c5deec111d",
    "sam3/sam/prompt_encoder.py": "7c4f4b2187d3e0e9590d2a8258702ee4491a67175076e7af2b45cf6e328101f3",
    "sam3/sam/transformer.py": "4885315b97577a5ece6ab53f45318fb629b16c240d96315064bf71a152effc6e",
}
SAM2_PREDICTOR_SHA256 = "912555920a77f72ded07839efa33fcceecc79f3523abbec783f0be46ee2c55c2"
LIFECYCLE = (
    "init_state",
    "_obj_id_to_idx",
    "_obj_idx_to_id",
    "_get_obj_num",
    "add_new_points_or_box",
    "add_new_points",
    "add_new_mask",
    "_get_orig_video_res_output",
    "_consolidate_temp_output_across_obj",
    "propagate_in_video_preflight",
    "propagate_in_video",
    "clear_all_prompts_in_frame",
    "reset_state",
    "_reset_tracking_results",
    "remove_object",
    "_clear_non_cond_mem_around_input",
)


def extracted(path, cls, name, namespace, *, blocking_storage=False):
    node = definition(path, cls, name)
    if blocking_storage:
        changed = 0
        for item in ast.walk(node):
            if (
                isinstance(item, ast.Call)
                and isinstance(item.func, ast.Attribute)
                and (item.func.attr == "to")
                and item.args
                and isinstance(item.args[0], ast.Name)
                and (item.args[0].id == "storage_device")
            ):
                for key in item.keywords:
                    if key.arg == "non_blocking":
                        if not isinstance(key.value, ast.Constant) or key.value.value is not True:
                            raise ValueError("Unexpected SAM3 storage-copy ABI")
                        key.value = ast.Constant(False)
                        changed += 1
        expected = 2 if name == "_run_single_frame_inference" else 1
        if changed != expected:
            raise ValueError("SAM3 storage-copy count changed")
    scope = dict(namespace)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec"
        ),
        scope,
    )
    return scope[name]


def install(predictor, *, sources, frame_loader, concat_points, fill_holes, torch_module=None):
    sources.validate()
    if getattr(predictor, "_holdtrack_sam3_lifecycle_installed", False):
        raise ValueError("Only one SAM3 lifecycle installation")
    if torch_module is None:
        import torch as torch_module
    torch = torch_module
    for name, value in dict(
        num_maskmem=7,
        memory_temporal_stride_for_eval=1,
        max_cond_frames_in_attn=6,
        use_memory_selection=False,
        non_overlap_masks=False,
        non_overlap_masks_for_output=False,
        non_overlap_masks_for_mem_enc=False,
        clear_non_cond_mem_around_input=False,
        add_all_frames_to_correct_as_cond=False,
        fill_hole_area=8,
        pred_obj_scores=True,
        use_obj_ptrs_in_encoder=True,
        fixed_no_obj_ptr=True,
        soft_no_obj_ptr=False,
        binarize_mask_from_pts_for_mem_enc=True,
        directly_add_no_mem_embed=True,
    ).items():
        setattr(predictor, name, value)
    scope = dict(
        torch=torch,
        F=torch.nn.functional,
        OrderedDict=OrderedDict,
        NO_OBJ_SCORE=-1024.0,
        concat_points=concat_points,
        load_video_frames=frame_loader,
        tqdm=lambda x, **kwargs: x,
    )
    for name in LIFECYCLE:
        method = extracted(sources.sam2_predictor, "SAM2VideoPredictor", name, scope)
        setattr(predictor, name, types.MethodType(method, predictor))
    source = sources.sam3_root / "sam3/model/sam3_tracking_predictor.py"
    inference = extracted(
        source, "Sam3TrackerPredictor", "_run_single_frame_inference", scope, blocking_storage=True
    )
    encoder = extracted(
        source, "Sam3TrackerPredictor", "_run_memory_encoder", scope, blocking_storage=True
    )
    pos = extracted(source, "Sam3TrackerPredictor", "_get_maskmem_pos_enc", scope)
    predictor._get_maskmem_pos_enc = types.MethodType(pos, predictor)
    predictor._run_memory_encoder = types.MethodType(encoder, predictor)

    def infer(self, *args, **kwargs):
        import inspect

        bound = inspect.signature(inference).bind(self, *args, **kwargs)
        bound.apply_defaults()
        if bound.arguments["batch_size"] != 1:
            raise ValueError("LiAM SAM3 replacement must decode objects individually")
        with torch.inference_mode():
            compact, logits = inference(*bound.args, **bound.kwargs)
            logits = fill_holes(logits, self.fill_hole_area)
            compact = dict(
                compact,
                pred_masks=logits.to(
                    bound.arguments["inference_state"]["storage_device"], non_blocking=False
                ),
            )
            if set(compact) != {
                "maskmem_features",
                "maskmem_pos_enc",
                "pred_masks",
                "obj_ptr",
                "object_score_logits",
            }:
                raise ValueError("LiAM five-field compact ABI differs")
            return (compact, logits)

    predictor._run_single_frame_inference = types.MethodType(infer, predictor)
    predictor._holdtrack_sam3_lifecycle_installed = True
    return dict(
        kind="SAM3 tracker-only / public SAM2 per-object lifecycle",
        commit=sources.commit,
        image_size=predictor.image_size,
        low_res_mask_size=predictor.low_res_mask_size,
        max_cond=6,
        num_maskmem=7,
        stride=1,
        sam3_memory_selection=False,
        D2H_storage_blocking=True,
        fill_holes_area=8,
        late_births=True,
        batch_size=1,
        no_weights_loaded_by_adapter=True,
    )
