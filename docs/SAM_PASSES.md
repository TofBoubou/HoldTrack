# The SAM 3 passes

`holdtrack.sam.passes` runs the tracker passes of HoldTrack on a video (an MP4 file or a folder of images) from the boxes of a detector on the same frames. It does not run a detector.

```python
from pathlib import Path
from holdtrack.sam.passes import SamAssets, run_passes

outputs = run_passes(
    "clip.mp4", "rf_2xl_detections.npz", "work/clip",
    assets=SamAssets(
        checkpoint=Path("weights/sam3.pt"),
        checkpoint_sha256="<SHA-256 of the checkpoint>",
    ),
)
tracks = outputs["PROT"].projected
masks = outputs["PROT"].masks
```

## The five passes

| name | pass of the paper |
| --- | --- |
| `BASE` | forward, native memory update |
| `EXCL` | forward, exclusive update |
| `PROT` | forward, guarded update |
| `reverse_BASE` | backward in time, native update |
| `mirror_EXCL` | mirrored frames, exclusive update |

`roles=` selects the passes, and the light system runs `roles=("PROT",)`. The memory updates are in [`holdtrack/sam/memory.py`](../holdtrack/sam/memory.py).

## Inputs

The detections are an NPZ file with `frames`, `boxes` and `scores` (or `frame`, `boxes`, `score`). Frames are zero-based and boxes are `x1, y1, x2, y2` in pixels of the image. Detections of positive area with a score of at least 0.01 are used, in their order in the file. `associations=` gives the ByteTrack tracks of these detections as a MOT file. Without it, ByteTrack (Ultralytics 8.4.64) is run on them.

A folder of images is read in natural filename order, and a video file is decoded once to PNG images. All frames must have the same size. The frame rate is read from the video file or given by `fps=`. A folder without `fps=` is taken at 20 frames per second.

## Two detectors

The complete system uses two detectors. The passes are started and prompted on the boxes of RF-DETR L at 1344, and the three forward passes are projected on the ByteTrack tracks of RF-DETR 2XL:

```python
outputs = run_passes(
    "clip.mp4", "rf_large_detections.npz", "work/full",
    assets=assets, associations="rf_large_bytetrack.txt",
    support_detections="rf_2xl_detections.npz", support_associations="rf_2xl_bytetrack.txt",
)
```

The backward and mirrored passes stay on the tracks of the first detector. Without `support_detections`, every pass is projected on the tracks of the detector that starts it.

## Outputs

Each pass returns a `PassOutput`:

| field | content |
| --- | --- |
| `raw` | MOT file of the boxes of the masks, with the identities of SAM 3 |
| `projected` | MOT file of the detector tracks, with the identities of the pass |
| `source_projected` | the same on the tracks of the detector that starts the pass |
| `masks` | folder of masks, one JSON file per image, COCO run-length encoding |
| `receipt` | checksums of the outputs, GPU name and elapsed time |

MOT frames are one-based. Boxes and masks are in the coordinates and the frame order of the input video, for the backward and mirrored passes too. An identity of `raw` is not always the identity of the same person in `projected`, so a renderer matches masks and projected boxes by IoU. The outputs of a pass before projection are kept in `<name>/native/`.

The backward pass reverses the frames and the detections, runs ByteTrack on the reversed video and restores the frame numbers. The mirrored pass mirrors the boxes and the ByteTrack tracks and flips the image tensor, without running ByteTrack again.

## Installation

Install `pip install -e ".[sam]"` and the official sources of SAM 3 and SAM 2. At start, the sources and the checkpoint are verified by checksum. `SamAssets.sam3_root` and `SamAssets.sam2_predictor` give the location of the sources.

## Preparing without a GPU

```sh
python -m holdtrack.sam.passes clip.mp4 --detections rf_2xl_detections.npz \
  --checkpoint weights/sam3.pt --out work/clip --prepare-only
```

decodes the video, builds the ByteTrack tracks, verifies every input and writes `work/clip/passes.json`, without loading a model. The passes then run with

```python
from holdtrack.sam.passes import run_prepared_passes
outputs = run_prepared_passes("work/clip/passes.json")
```

Each run takes a new output folder. `receipt.json` is written at its root once every pass and every mask file is complete.
