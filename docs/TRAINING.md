# Training the detectors

This page describes how the detector weights of the DanceTrack system were trained, for RF-DETR-2XL, RF-DETR-L, YOLO26X and D-FINE-L. The weight files and their digests are listed in [BOXES.md](BOXES.md) and in `holdtrack/resources/weights.json`. The detectors of the MOT17 system are described in [MOT17.md](MOT17.md).

## Data

| Set | Videos | Images |
|:--|:-:|:-:|
| DanceTrack training set | 40 | 41,796 |
| DanceTrack validation set | 25 | 25,508 |
| CrowdHuman, training and validation sets, full-body boxes | | 19,370 |

No detector is run on a video it was trained on. The test videos are read by detectors trained on DanceTrack videos outside the test set. The validation videos are read by detectors trained on the 40 training videos. The 32 training videos on which the system is measured are split in four folds of eight videos, and each fold is read by detectors trained on the 32 other training videos:

| Fold | Videos (`dancetrackNNNN`) |
|:-:|:--|
| 1 | 0002 0016 0029 0044 0053 0066 0075 0087 |
| 2 | 0006 0020 0032 0045 0055 0068 0080 0096 |
| 3 | 0008 0023 0033 0049 0057 0069 0082 0098 |
| 4 | 0012 0024 0037 0051 0061 0072 0083 0099 |

## RF-DETR-2XL

Training has two stages with the same settings. Both run at resolution 1120 with one scale, in batches of 16. The optimizer is AdamW with a learning rate of 1e-4 (1.5e-4 for the encoder), a weight decay of 1e-4, a layer decay of 0.8 and a component decay of 0.7. Gradients are clipped at 0.1. The model has 300 queries and 13 groups, and it trains in bfloat16 mixed precision with seed 20261107. The learning rate warms up over the first 5% of the updates, follows a cosine down to 1% at 95%, then stays constant. The weights kept are a moving average with a half-life of 1% of the stage, at its last update. The augmentations are those of RF-DETR. A frame is flipped horizontally with probability 0.5 and, with probability 0.5, resized to 400, 500 or 600 and then cropped to a size between 384 and 600. There is no mosaic and no mixup.

1. CrowdHuman, 10 epochs (12,100 updates), from the public checkpoint `rf-detr-xxlarge.pth`.
2. DanceTrack, 3 epochs, from the weights of stage 1, with a new optimizer and a new moving average:

| Videos read | Training videos | Updates |
|:--|:--|:-:|
| test set | the 40 training and 25 validation videos | 12,618 |
| validation set | the 40 training videos | 7,836 |
| fold 1 of the training set | the 32 other training videos | 5,952 |
| fold 2 | the 32 other training videos | 6,372 |
| fold 3 | the 32 other training videos | 6,297 |
| fold 4 | the 32 other training videos | 6,339 |

The mirrored frame, the frame at 1280, the vertical bands and the crops around each person are read with these same weights.

## RF-DETR-L

The weights that read the test and validation videos are trained in two stages, in batches of 16. The optimizer is AdamW with a constant learning rate of 2e-4 (3.2e-5 for the encoder), a weight decay of 1e-4, a layer decay of 0.8 and a component decay of 0.7. Gradients are clipped at 0.1, the model has 300 queries and 13 groups, and the moving average has a decay of 0.993. Training runs in bfloat16 mixed precision with seed 42 and the augmentations of RF-DETR-2XL, with one scale drawn per batch.

1. CrowdHuman, 10 epochs (12,100 updates), from the public checkpoint `rf-detr-large.pth`, at 1120 with scales 840 to 1400 by steps of 56.
2. The 40 DanceTrack training videos, 3 epochs (7,836 updates), from the weights of stage 1, at 1344 with scales 1064 to 1624 by steps of 56.

The weights that read the four folds of the training set are trained in one stage with `rfdetr` 1.2.1. Each starts from the public checkpoint `rf-detr-large.pth` and runs one epoch on the 32 other training videos (1,984, 2,124, 2,099 and 2,113 updates), at 1120 with scales 840 to 1400. The batch of 16 is accumulated one image at a time. The optimizer is AdamW with a constant learning rate of 1e-4 (1.5e-4 for the encoder) and a weight decay of 1e-4. The moving average has a decay of 0.993, and the seed is 20261027.

## YOLO26X

This detector is trained in one stage from the public checkpoint `yolo26x.pt`, on CrowdHuman and DanceTrack together. Each update reads one image of the CrowdHuman training set (15,000 images, full-body boxes) and one image of the 40 DanceTrack training videos. Training lasts 41,796 updates, one pass over the DanceTrack images. Frames are letterboxed at resolution 1280. The optimizer is AdamW with a learning rate of 1e-5 and a weight decay of 1e-4. The loss gains are 7.5 for boxes, 0.5 for classes and 1.5 for the distribution focal loss, and the one-to-many and one-to-one branches are weighted 0.8 and 0.2. Frames are flipped horizontally on every other update, with no mosaic, no mixup and no color jitter. The ignored regions of CrowdHuman are masked in the background loss, and the seed is 20261004. The weights kept are the moving average at the last update, and they read the frame and the mirrored frame.

These weights read the test and validation videos. Each fold of the training set is read by weights trained the same way on CrowdHuman and the 32 other training videos, with one pass over their images, which takes 31,752, 33,994, 33,593 and 33,812 updates.

## D-FINE-L

This detector is trained in one stage with the D-FINE source at commit `956d1709314c2c6a4df6f34de232054578a7449f`, from the public checkpoint `dfine_l_obj2coco.pth` (Objects365 then COCO), without a CrowdHuman stage. The head keeps its 80 classes, and persons are COCO category 1. Training reads one frame out of four of each video (frames 1, 5, 9, and so on) for 24 epochs. The resolution is 640, with the scales of the D-FINE collate function, from 480 to 800 by steps of 32, and batches hold 8 images. The optimizer is AdamW with a constant learning rate of 2.5e-4 (1.25e-5 for the backbone), without warm-up. The weight decay is 1.25e-4, with none on the normalization layers of the encoder and decoder. Gradients are clipped at 0.1, training runs in mixed precision, the moving average has a decay of 0.9999 and the seed is 20261016. The augmentations are those of D-FINE, a photometric distortion with probability 0.5, a zoom out, an IoU crop with probability 0.8 and a horizontal flip. The weights kept are the moving average after the last epoch.

| Videos read | Training videos | Images | Updates |
|:--|:--|:-:|:-:|
| test and validation sets | the 32 training videos of the four folds | 8,517 | 25,536 |
| fold 1 of the training set | the 32 other training videos | 7,947 | 23,832 |
| fold 2 | the 32 other training videos | 8,507 | 25,512 |
| fold 3 | the 32 other training videos | 8,407 | 25,200 |
| fold 4 | the 32 other training videos | 8,462 | 25,368 |
