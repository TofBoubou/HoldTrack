# HoldTrack on MOT17

MOT17 films pedestrians in streets, from fixed and moving cameras, and annotates a full-body box for each of them through occlusion. HoldTrack runs on it with its trained parts trained again on the MOT17 training videos and its steps selected again. A step is kept where it gains on videos held out from every model it uses.

This page describes the steps that this selection keeps after the box corrector, the modules that hold them and the replay that rebuilds the files of the submission.

## Where identities are lost

On DanceTrack, identities are lost at contact. On the MOT17 training videos, among 189 contacts between two identities on four held-out street videos, the annotation shows no exchange of the two identities. Here identities are lost when a person returns. With each video held out in turn, the annotation labels 46 returns of an identity after an absence of two seconds or more, and 15 of them are on the person it left.

## Steps

| Step | Module | What it reads |
|:--|:--|:--|
| Identity at the return of a person | [`holdtrack.identity.returns`](../holdtrack/identity/returns.py) | tracks, camera motion, appearance features of the crops |
| Earlier start of each track | [`holdtrack.steps.earlier_start`](../holdtrack/steps/earlier_start.py) | tracks, mask boxes of the forward and backward passes, detections |
| Gaps filled under camera motion | [`holdtrack.steps.camera_motion`](../holdtrack/steps/camera_motion.py) | tracks, camera motion |
| Smoothing of the edges | [`holdtrack.steps.smoothing`](../holdtrack/steps/smoothing.py) | tracks |

None of them reads an annotation, and none is trained on one.

**Identity at the return of a person.** Two moves decide each return again. The first, `cut_returns`, cuts. An identity keeps its name across an absence only if it comes back within two seconds and within half a body height of a place that the person could have reached. That place is where it was last seen, or the place given by extending its motion over the absence, forward from before it or backward from after it. Camera motion is removed first. The second move, `link_returns`, links. A classifier scores each pair of tracklets in which the second starts within ten seconds of the end of the first and neither end touches the image border. It reads the distance between the two ends, their motion, their heights and an appearance space fitted on the video, and the pairs scored at least 0.5 are linked one to one. Its training pairs come from the tracks themselves. A tracklet cut at a random gap gives a true pair, and each side of the gap with another tracklet gives a false pair.

```python
from holdtrack.identity.returns import ReturnVideo, cut_returns, link_returns

identities, cuts = cut_returns(frames, identities, boxes, fps, motion)
video = ReturnVideo(frames, identities, boxes, fps, width, height, motion, crop_rows, features)
identities, links = link_returns({"video": video})
```

`boxes` are `x1, y1, x2, y2`. `motion` maps each frame to a 3x3 matrix (`holdtrack.steps.camera_motion.load_motion`). `crop_rows` gives the row of each appearance crop and `features` its feature vector. `link_returns` takes several videos and fits one classifier on the pairs of all of them.

**Earlier start of each track.** An identity starts at its first box, which can come after the person has entered the frame. In the frames before that box, an object of the forward or of the backward pass can already hold the person in a mask. Where a detection with a score of at least 0.3 matches such a mask, and no box of the frame already covers it, the detection is added to the identity (`confirmed_mask_rows`). In the five seconds before the first box, a mask of the forward pass that has the size of that box and covers no other box is added as well (`rows_before_birth`).

**Gaps filled under camera motion.** A gap of at most 1.5 s inside an identity is filled. Each of its two ends is carried to the missing frame with the motion of the background, and the two carried boxes are interpolated.

**Smoothing of the edges.** On MOT17, an annotated edge moves along straight segments. Between consecutive frames, its second difference is at most one pixel for 92% of the annotated edges. Each edge of a box is replaced by the value, at its frame, of a line fitted to that edge along the identity over ±0.2 s, with weights that decrease linearly with the distance in time. On eight annotated DanceTrack videos the same share is 36%, and the system evaluated on DanceTrack does not use this step.

`holdtrack.steps.finishing.finish` runs the last three steps in this order on the tracks of a video:

```python
from holdtrack.steps.camera_motion import load_motion
from holdtrack.steps.finishing import finish, mot_lines

frames, identities, boxes, counts = finish(
    rows, fps, frame_count, load_motion("camera.npz"), identities=identities,
    forward_masks=forward, backward_masks=backward, detections=(det_frames, det_boxes, det_scores),
)
open("tracks.txt", "w").write("\n".join(mot_lines(frames, identities, boxes)) + "\n")
```

## Measured on held-out videos

The table gives HOTA before and after each step, on training videos held out from every model that the step uses. Each video is processed whole, and the detectors, the box corrector and the appearance network that process it are trained without it.

| Step | Videos | Before | After |
|:--|:--|:-:|:-:|
| Box corrector, trees of the first stage | the seven training videos | 68.50 | 70.43 |
| Box corrector, four sequence networks averaged with the trees | the seven training videos | 70.03 | 70.29 |
| Identity at the return of a person | the six street videos | 57.64 | 58.40 |
| Earlier start of each track | the six street videos | 58.40 | 59.54 |
| Smoothing of the edges, ±0.2 s | the seven training videos | 62.97 | 63.72 |

The rows are measured on different versions of the tracks, so they do not chain. The smoothing window is chosen for each video on the six others. It comes out at ±0.2 s for all seven videos, and each of them gains between 0.2 and 2.1 points.

## Parts trained again

RF-DETR-2XL and RF-DETR-L are fine-tuned on the MOT17 training videos. The tracks start from the public YOLOX detector of ByteTrack, which also takes the place of YOLO26X among the detector boxes read by the corrector, and RF-DETR-L takes the place of D-FINE. The box filter of the base system and the box corrector are trained on the MOT17 training videos. The corrector keeps its first stage, the trees and four sequence networks. The appearance network is trained again on the pedestrians of the MOT17 training videos. SAM 3, the pose estimators and the body-mesh model are the same frozen models. Durations given in frames on DanceTrack are converted to seconds, since the MOT17 videos run at 14, 25 and 30 frames per second.

## Training the detectors

The detectors of the DanceTrack system are described in [TRAINING.md](TRAINING.md). Both RF-DETR detectors start from weights trained for 10 epochs on the full-body boxes of CrowdHuman (training and validation sets, 19,370 images). They are then fine-tuned on the seven MOT17 training videos, 5,316 images. The targets are the evaluated pedestrians (class 1, flag 1), and the boxes of classes 2, 7, 8 and 12 are ignored. An annotated box that leaves the image is not clipped, in training or at inference.

| | RF-DETR-2XL | RF-DETR-L |
|:--|:--|:--|
| Resolution | 1120 | 1344, one scale per batch among 1064 to 1624 by steps of 56 |
| Epochs, updates | 2, 664 | 4, 1,328 |
| Batch | 16 | 16 |
| Optimizer | AdamW, weight decay 1e-4 | AdamW, weight decay 1e-4 |
| Learning rate | 1e-4, encoder 1.5e-4 | 2e-4, encoder 3.2e-5 |
| Layer and component decay | 0.8 and 0.7 | 0.8 and 0.7 |
| Schedule | warm-up over the first 5% of the updates, cosine down to 1% at 95%, then constant | constant |
| Weights kept | moving average with a half-life of 1% of the fit, at the last update | moving average with decay 0.993, at the last update |
| Precision | bfloat16 mixed precision | bfloat16 mixed precision |
| Seed | 20261107 | 42 |

For the half validation, RF-DETR-L is trained by the same recipe on frames 1 to n/2 + 1 of each video (2,664 images, 664 updates). For the leave-one-video-out validation, the detectors are trained on the six other videos.

The tracks start from the public YOLOX-X weights of ByteTrack, used as they are. The test set is read with `bytetrack_x_mot17.pth.tar` (SHA-256 `e3945f3523fde1e107708aacd64dab0670c34e371d136e54587cac7a50d3cfba`) and the half validation with `bytetrack_ablation.pth.tar` (`26cb8d2808664e5068a4c812d53becbc948b47fd6eacf2b45db049ab40c48b1a`). The detector takes an input of 800 by 1440, with a score floor of 0.01 and a non-maximum suppression at 0.7.

## Training the box corrector

[`holdtrack.boxes.training`](../holdtrack/boxes/training.py) fits the trees of the corrector on annotated tracks and applies them. The learner and the inputs are those of the DanceTrack corrector ([BOXES.md](BOXES.md)), with one gradient-boosted model per group of edges, 800 trees and an absolute-error loss. Three settings differ, each chosen on the MOT17 training videos:

- The target offsets are clipped at 2.0 box sizes instead of 0.3, since the full-body box of an occluded pedestrian can lie farther than that from the detected box.
- A box that lies for 80% of its area inside the annotated box of a pedestrian that no other box covers takes this pedestrian as its target (`annotated_targets(..., extended=True)`).
- The features add those of [`holdtrack.boxes.track_features`](../holdtrack/boxes/track_features.py), which are the size of the track within 1, 3 and 5 seconds, the box height that the scene gives at the position of the feet, and the share of the box covered by another box.

The corrector is trained on two sets of tracks of each training video, one from detectors that have seen the video and one from detectors that have not. Only the first stage is kept. It holds the trees and four sequence networks with the architecture and the settings of the DanceTrack system (seeds 0 to 3, 120 epochs), trained on the same 14 tables. The corrected box is (2 × trees + mean of the four networks) / 3, with the weights of the DanceTrack system.

```bash
python -m holdtrack.fetch_models --weights-dir weights --mot17
python -m holdtrack.boxes.training --train corrector/train --apply corrector/test --out corrected \
    --networks weights/mot17/network_0.pt weights/mot17/network_1.pt weights/mot17/network_2.pt weights/mot17/network_3.pt \
    --thresholds pose_thresholds.json --mhr70 mhr70.py --clip 2.0 --frame-rates corrector/frame_rates.json
```

Each set holds `tables/`, `pose17/`, `pose133/` and `mesh/`, one file per video, as built by `holdtrack.boxes.pipeline`. The `gt` field of a training table holds the target boxes. `frame_rates.json` gives the frame rate of each video of both sets.

Run on the stored tables of the replay data (`data/mot17/corrector/`) with four threads, this command fits the trees in about six minutes within 5 GB of memory and averages them with the four networks. Its boxes differ from `data/mot17/test/corrected/`, the corrected tracks of the submission, by at most 0.00001 pixel, a rounding of the average. Started from them, the replay writes the 14 files of the submission byte-identical. In place of `--train`, `--trees-file weights/mot17/trees.pkl` reads the released trees and writes the same boxes, and `--save` writes the trees just fitted. Without `--networks`, the files are the boxes of the trees alone, byte-identical to `data/mot17/corrector/test_trees/`. The networks are trained by `holdtrack.boxes.network_training` ([BOXES.md](BOXES.md)).

## Camera motion

`holdtrack.steps.camera_motion.estimate_motion` estimates the motion of the background between consecutive frames. Corners outside the boxes of the tracked persons are followed by optical flow at half resolution, and a similarity is fitted to them with RANSAC.

```python
import cv2
from holdtrack.steps.camera_motion import estimate_motion, save_motion

frames, matrices, counts = estimate_motion(
    lambda frame: cv2.imread(f"img1/{frame:06d}.jpg", cv2.IMREAD_GRAYSCALE), 1, frame_count, rows
)
save_motion("camera.npz", frames, matrices, counts)
```

## Replay

```bash
python research/replays/mot17_test.py data work
```

rebuilds the files of the submission from the stored outputs of the models on the seven test videos, on a CPU, and compares each of them with the digest of the submitted file. The server lists 42 files, the 14 videos under three detector names. The seven training videos are filled with stored public tracks. The script prints `14 of 14 files identical to the submission` and writes the 42 files and their archive in `work/`.

The replay data are in `data/mot17/`. Names are those of the videos (`MOT17-03-FRCNN`).

```
sequences.json                frame rate, number of frames, image size and split of the 14 videos
submission.sha256             digest of the 14 files of the submission
test/
  tracks/<video>.txt          tracks of the base system: frame, identity, x, y, width, height
  corrected/<video>.txt       the same rows after the box corrector
  camera/<video>.npz          camera motion: frame, M (one 2x3 matrix per frame), n (background points)
  crops/<video>.npy           row of each appearance crop
  features/<video>.npz        appearance feature of each crop
  masks_forward/<video>.txt   mask boxes of the forward pass: frame, object, x, y, width, height
  masks_backward/<video>.txt  mask boxes of the backward pass
  detections/<video>.npz      detections of the detector that starts the tracks: frame, boxes, score
train/public/<video>.txt      public tracks of the seven training videos
corrector/
  frame_rates.json            frame rate of each table of both sets
  train/                      tables/, pose17/, pose133/, mesh/: two sets of tracks of each training video
  test/                       the same folders for the seven test videos
  test_trees/<video>.txt      boxes of the trees alone on the seven test videos
```
