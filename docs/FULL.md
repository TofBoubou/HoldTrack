# The complete system

    holdtrack track clip.mp4 --full

runs the system of the paper without its face steps, from the pixels to the final tracks, and renders the video. It writes the same three outputs as the light command, `tracks.txt`, `masks/` and `tracks.mp4`.

## What runs

1. **Detectors.** RF-DETR 2XL at 1120, RF-DETR L at 1344, YOLO26X, YOLO26X on the mirrored frame, D-FINE-L.
2. **Tracker passes.** Five SAM 3 passes started on the boxes of RF-DETR L (native, exclusive, guarded, backward and mirrored). The three forward passes are projected on the ByteTrack tracks of RF-DETR 2XL (`holdtrack.sam.passes`).
3. **Base tracks.** Vote of the three forward passes, fusion of the five detector views, track-level steps (`holdtrack.pipeline`, configuration `full`).
4. **Boxes.** The detector reads every frame again (mirrored, at 1280, by bands, on crops). Around the reference boxes, 17 keypoints, 133 keypoints and a body mesh are read. Two correctors then write the reference boxes and the final boxes (`holdtrack.boxes`).
5. **Identities.** Three appearance stages on the video, then the twelve steps of the identity chain (`holdtrack.identity`).
6. **Final tracks.** Relay in place, its check and the assembly, then the rendering.

Every stage writes its output in `<out>/work/`. A second call reuses the stages that are finished.

On one A40, a video of 603 frames with five dancers takes 54 minutes, or 5.4 seconds per frame, model loading included. The detectors, the five passes and the base tracks take 25 minutes, the box reads 16, the poses and the mesh 5, the appearance stages and the identity chain 6, and the rendering 2. The frozen output of the system on this video was computed on other GPUs. Against it, the command produces 3,008 of the 3,009 rows, each with the same identity, and 98.0% of the boxes are within one pixel.

## Models

The weights of stages 1 to 3 are read in `--weights-dir`, as for the light command, and D-FINE needs its official source (`--dfine-source`). Stages 4 and 5 read the following files from the same folder:

| file | model |
| --- | --- |
| `rf_2xl.pth` | RF-DETR 2XL, read again by the box stages |
| `corrector/reference_trees.pkl`, `corrector/reference_network_0.pt`, `corrector/reference_network_1.pt` | reference corrector |
| `corrector/trees.pkl`, `corrector/network_0.pt`, `corrector/network_1.pt`, `corrector/pose_thresholds.json` | full corrector |
| `identity/appearance.pt` | appearance network of the identity chain |
| `pose133.onnx` | RTMW-L, 133 keypoints, from the MMPose release |
| `sam3d_body/` | weights of SAM 3D Body (`facebook/sam-3d-body-dinov3`); its source is given by `--mesh-source` |

One command fetches these files and verifies each of them:

    python -m holdtrack.fetch_models --weights-dir weights --full

SAM 3 and SAM 3D Body are gated on Hugging Face. Accept their license there, then log in (`huggingface-cli login`) or set `HF_TOKEN`. ViTPose++ huge (17 keypoints) is downloaded from Hugging Face at the first run. Sources, revisions and checksums of every model are in [BOXES.md](BOXES.md).

    holdtrack track clip.mp4 --full --dfine-source D-FINE --mesh-source sam-3d-body

Files stored elsewhere are named in a `models.json`, given by `--models` or placed in the weights folder. The command writes this file at its first run, with the paths it found.

## Environments

The stages have requirements that do not install together. Each of four options gives an interpreter to a group of stages. Without them, every stage runs in the current interpreter.

| option | stages | needs |
| --- | --- | --- |
| `--boxes-python` | box reads, correctors, 17 keypoints, appearance, identity chain | `pip install -e ".[boxes,identity]"` |
| `--pose-python` | 133 keypoints | `pip install -e ".[pose]"` |
| `--mesh-python` | body mesh | `pip install -e .` and the requirements of SAM 3D Body |
| `--faces-python` | face tables, with `--faces` | `pip install -e ".[faces]"` |

## Frame rate and short clips

The system was built at 20 frames per second. The identity chain reads its durations in seconds and takes the frame rate of the video. The base drops a track of 2.75 s or less (55 frames at 20 frames per second). A clip that is itself that short keeps every track.

## Faces

`--faces` adds the face steps (`holdtrack.faces`) between the identity chain and the relay, as described in [FACES.md](FACES.md).
