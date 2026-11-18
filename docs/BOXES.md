# Box corrector: inputs and models

For each box of the tracks, the corrector reads the outputs of the models below and returns the corrected box. The weight files are those of `holdtrack/resources/weights.json` and `holdtrack/fetch_models.py`. Its own weights are two stages of gradient-boosted trees and two small sequence networks, trained on the DanceTrack annotations.

## Models that produce the inputs

| Input of the corrector | Model | Source and revision | Weights (SHA-256) | License |
|:--|:--|:--|:--|:--|
| Detector boxes 1 to 7: frame at two resolutions, mirrored frame, two sets of vertical bands, two crops around the person | RF-DETR-2XL, fine-tuned on DanceTrack | `rfdetr` 1.12.0, `rfdetr-plus` 1.1.0 | `rf_2xl.pth` `ccf2f57e34765514917bcea38158ad97b575e7070a4211e2dd91b1c72651d141` | code Apache-2.0; weights PML-1.0 |
| Detector boxes 8 and 9: frame and mirrored frame | YOLO26X, fine-tuned on CrowdHuman and DanceTrack together | `ultralytics` 8.4.64 | `yolo.pth` `ee9cb04bf713a7c399bf7029eac2fce618075f010f64d34517ff0e989318e4bb` | AGPL-3.0 |
| Detector box 10 | D-FINE-L, fine-tuned on DanceTrack | D-FINE repository | `dfine.pth` `616f57e483c9ae66f020eb91d82577bffe3f3eccb81129f7c2a3dd63f3352709` | Apache-2.0 |
| Mask boxes, one per forward pass | SAM 3, three memory update rules, prompted by RF-DETR-L at 1344 | SAM 3 checkpoint, not fine-tuned; `rf_large.pth` | `sam3.pt` `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`; `rf_large.pth` `8903b7d227cbbde936a42ba82cc47acca6d6bff8861248deabd7056f9a1f5f94` | SAM License; Apache-2.0 |
| Keypoints, 17 points | ViTPose+ H | `usyd-community/vitpose-plus-huge`, revision `9f36d7aec1800d23e97f10c2e74393aee92aa53f`; 192 × 256 window enlarged by 1.25, mirrored, bfloat16 | upstream | Apache-2.0 |
| Keypoints, 133 points | RTMW-L | MMPose, ONNX export `end2end.onnx` | `bd033156e5104c4f5d2edfe0453e02661e30a2f3da453ec93c8764d561b83054` | Apache-2.0 |
| Body mesh: 70 projected points, mesh box, depth | SAM 3D Body | public repository; `model.ckpt` | `b5a2f9d305dd02626b967aa2e86021fba07065df66ce7a7e00ffb9664f150abf` | SAM License |

## Models of the corrector

| File | Content | SHA-256 |
|:--|:--|:--|
| `trees.pkl` | two stages of trees, three models per stage (left and right edges, top edge, bottom edge); scikit-learn 1.9.0 `HistGradientBoostingRegressor` | `e348dfd82231960bb9364d310f6f815422c4e8c0bf66b930f069b161feb44a1f` |
| `network_0.pt`, `network_1.pt` | two sequence networks over thirteen views of the box (ten detector boxes, mesh box, two keypoint boxes) | `3e8f4724af1cf0b64b5839cb8ea0f5292631d0c03673f4fe1981a4b88b5004ef`, `1c02c27450a913abc74d14b8227d06ac857f880a409f98fd5a3651cc43cb0a75` |
| `pose_thresholds.json` | two confidence thresholds of the keypoint features | `2783ad50b5ddf4b0c07b035403e963bf7ac105b69b70c0a65b7ed74ba574902b` |
| `reference_trees.pkl` | trees of the reference corrector, which reads the nine detector and mask views | `8c5014e4986bb620404c24d3b2f726f7c49ce2b4e12ed4213ce432cc3aa8c2d5` |
| `reference_network_0.pt`, `reference_network_1.pt` | sequence networks of the reference corrector | `7e08159e070667addbec5f0a8d9f5ba38afb3d93d390559620e636ef8c1d5ba2`, `c7072231894f3a300017c04b6d5ba66a82a38ab221a4663e0a265e0ea055e2ad` |

The corrector itself needs Python 3.12, NumPy 2.5, SciPy 1.18, scikit-learn 1.9.0 and PyTorch 2.10. It runs on CPU and takes 146 seconds for the 35 test videos (286,325 boxes).

## Training the sequence networks

    python -m holdtrack.boxes.network_training --tables tables/ --out network_0.pt --seed 0 --clip 1.0

trains one sequence network on the tables of the training videos, whose `gt` field holds the target boxes. It runs 120 epochs on the rows that have a target, in batches of 512, with AdamW, one learning-rate cycle up to 1e-3 and a weight decay of 0.05. Half of the samples are mirrored, half are read backward in time, and each view is hidden from a sample with probability 0.45. The corrector averages two networks, trained with seeds 0 and 1. `--clip` bounds the target offsets, in box sizes. The bound is 1.0 with the thirteen views of the full corrector, and 0.3 with `--views 9` for the reference corrector.

Training is not bit-reproducible. The networks of the MOT17 submission were fitted with this command on a CPU. Fitted again the same way, the first network of the MOT17 corrector follows the same losses over the first epochs and ends at 0.0311, like the network used for the submission. On the 169,972 boxes of the MOT17 test videos, its edges differ from those of that network by 0.3 pixel (median) and 1.8 pixels (95th percentile). With the weights used for the submission, `holdtrack.boxes.networks.predict` returns their stored predictions exactly.

## Reproducibility

[`research/replays/chain_test35.py`](../research/replays/chain_test35.py) applies the packaged corrector to the stored inputs of the test set. With the released predictions of the two sequence networks, its 35 files are byte-identical to the boxes of the released results. With the two networks run again on CPU, 19 box edges out of 1,145,300 move by more than 0.01 pixel, and after rounding to the pixel one box out of 286,325 differs.
