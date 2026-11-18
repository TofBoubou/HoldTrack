# Reproduce HoldTrack

One command, `holdtrack track`, runs the system on a video. The [README](../README.md) and [FULL.md](FULL.md) describe it. This page covers the base system alone, the evaluation and the data of the replays.

## Installation

Use Python 3.12 and run `pip install -e ".[sam,detect,eval]"`. Install the official sources of SAM 3 (commit `7f473cfd983d5eb43040ff608f66bff0d7bdcb53`) and SAM 2 (commit `2b90b9f5ceec907a1c18123530e92e794ad901a4`). `holdtrack.sam.lifecycle.SourceBindings` checks their files. D-FINE needs a checkout of its official source. ByteTrack is the implementation of Ultralytics 8.4.64, and RF-DETR uses `rfdetr==1.12.0` and `rfdetr-plus==1.1.0`. `--detector-python PATH` runs the detectors in another interpreter.

## Models

`holdtrack/resources/weights.json` lists SAM 3, the four detector models and the two reliability heads of the package, with their SHA-256. `python -m holdtrack.fetch_models --weights-dir weights` fetches the model files and verifies each of them ([FULL.md](FULL.md)). `holdtrack download --name NAME --source FILE` verifies a file that you already have and imports it into the weights folder.

## The base system

```bash
holdtrack run --frames images/ --out tracks.txt --dfine-source /path/to/D-FINE
holdtrack run --frames images/ --out light.txt --config light
```

| `--config` | system |
| --- | --- |
| `full` (default) | the base system of the paper: passes started on RF-DETR L at 1344 and projected on the tracks of RF-DETR 2XL, with YOLO26X, YOLO26X on the mirrored frame, D-FINE-L and RF-DETR L as views |
| `large` | RF-DETR L at 1344 alone for the tracks, with YOLO26X, its mirror, D-FINE-L and RF-DETR 2XL as views |
| `2xl` | RF-DETR 2XL alone for the tracks, with YOLO26X, its mirror and D-FINE-L as views |
| `light` | the light base system: RF-DETR 2XL, the guarded pass and the track-level steps |

`full`, `large` and `2xl` run the three forward passes, their vote, the fusion of the detector views and the five track-level steps.

The names of the frames must sort in time order. `--work-dir PATH` keeps the stages of a run. Run again, the same command verifies the finished stages and reuses them, and a stage whose inputs changed is computed again. An output file that exists with another content is not overwritten.

`holdtrack run --cached-inputs sequence.json` runs the vote, the fusion and the track-level steps on stored detections and stored passes, listed with their SHA-256 (`holdtrack.pipeline.CachedSequence`).

In Python, `holdtrack.run(frames, out, config="full")` runs the base system, and `holdtrack.sam.run_once(frames, detections, "EXCL")` runs one pass and returns its tracks in MOT format ([SAM_PASSES.md](SAM_PASSES.md)). A detection file is an NPZ with `frames` (zero-based), `boxes` (N by 4, `x1, y1, x2, y2`) and `scores`.

## Evaluation

```bash
holdtrack evaluate --tracks tracks.txt --gt gt/gt.txt --num-frames 1203 --out metrics.json
```

reads a MOT file and the `gt.txt` of a DanceTrack video and computes HOTA, DetA, AssA, MOTA and IDF1 with TrackEval. The values are percentages. `--num-frames` gives the length of the video and rejects rows beyond it.

## Demo

`holdtrack demo --video images/ --masks masks/ --out demo.mp4` overlays masks and identities on the video. With `--palette FILE`, it takes colors from a JSON file. `run_once(..., masks="masks/")` saves the masks of a pass as COCO run lengths.

## Replay data

The scripts of `research/replays/` take a folder of replay data as first argument. It holds the stored outputs of the models on the 35 test videos and on the 57 annotated videos, and the frozen outputs of the system that the package is compared with. Names are those of the videos (`dancetrack0003`).

```
dancetrack/<video>/img1/            frames of DanceTrack (and gt/gt.txt for the 57 annotated videos)
models/corrector/                   trees.pkl, reference_trees.pkl, pose_thresholds.json
models/sam3d_body/mhr70.py          joint list of SAM 3D Body
test35/
  base/<video>/                     tracks.txt and fusion.txt of the base system, bytetrack.txt
  passes/<video>/                   native, exclusive, guarded, vote, backward, mirrored (.txt)
  masks/<video>/                    mask boxes of the three forward passes
  detectors/<view>/<video>          rf_2xl, rf_2xl_mirror, yolo, yolo_mirror, dfine
  reads/<read>/<video>.npz          1280, band_1120_0..2, band_1280_0..4, crops_1, crops_2
  pose17/, pose133/, mesh/          keypoints and body mesh, one file per video
  networks/                         outputs of the sequence networks: reference_0, reference_1, full_0, full_1
  tables/                           frozen tables of the corrector, for the comparison
  stages/<stage>/<video>.npz        appearance stages: classifier, embedding, features, features_short
  faces/<table>/<video>.npz         face tables (arcface, adaface, their _clean and _clean_projected tables, shared, adaface_embeddings): not distributed, see below
  identity_steps/                   videos/, stages/ and work/: identities stored after each step of the chain
  submissions/                      without_faces/ and complete/: the 35 files of each submission
annotated57/
  videos/<video>/                   tracks.txt, bytetrack.txt, passes/, visibility.npy, upper_body.npy, short_tracks.npy
  stages/<stage>/<video>.npz        crops, classifier, embedding, features, crops_short, features_short
  identities/<step>.npz             frozen identities after each step
  attached.npz                      short-track rows attached by the last step
  faces/, pose133/                  as for the test videos
  reference/                        where_lost.json, shared_errors.json: the measurements of the paper
```

```bash
python research/replays/chain_test35.py data work
python research/replays/faces_chain_test35.py data work faces
python research/replays/identity_steps_annotated57.py data steps57
```

The face tables hold embeddings of faces. They are not distributed with the replay data, so the replays of the face steps (`faces_chain_test35.py`, `faces_second_eye.py`) need them computed again from the frames, with the two face models fetched from their authors ([FACES.md](FACES.md)).

The replay of the MOT17 test set, `python research/replays/mot17_test.py data work`, reads `data/mot17/`. Its layout is described in [MOT17.md](MOT17.md).
