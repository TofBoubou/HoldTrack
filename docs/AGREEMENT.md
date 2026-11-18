# Agreement of a backward and a mirrored pass

This is a training-free identity corrector for mask-based trackers (SAM 2, SAM 3). It runs after the tracker and uses two more passes of the same tracker, one on the reversed video and one on horizontally mirrored frames. It changes an identity only where the two passes agree.

- **Link step.** A track is cut where its boxes stop being clear (the box of the mask no longer covers the box). Two pieces of different identities are joined when both passes give them the same object, and when the links of the input tracks at both ends are supported by neither pass and contradicted by both.
- **Box step.** A box that both passes place in another identity takes it if that identity has no box in the frame. If both passes place the box that carries it in the identity of the first, the two boxes exchange.

Nothing is learned, and the module uses no appearance model, no face model and no annotation.

## Use

```bash
python -m holdtrack.identity.agree --tracks tracks.txt --mask-boxes masks.txt --backward backward.txt --mirrored mirrored.txt --out corrected.txt
```

or, from Python:

```python
from holdtrack.identity import agree
ids = agree.correct(tracks, clear, backward_rows, mirrored_rows, fps=20.0)
```

`tracks` are MOT rows (frame, identity, x, y, width, height). `clear` holds one boolean per box, which `agree.clear_boxes` computes from the boxes of the masks. The two passes are MOT rows in the frame of the original video.

## Inputs

- The tracks, with the box of each person and the box of its mask. A track is cut wherever the mask stops covering the box, which is where trackers exchange identities.
- Two more passes of the tracker over the video, one on the reversed video and one on mirrored frames.

## Measured

HOTA on 57 annotated videos of DanceTrack, with a 95% bootstrap interval over videos:

| Applied to | Gain | Links joined |
|:--|:-:|:-:|
| Our base system (vote of three passes, 83.1 HOTA) | +0.50 (0.26 to 0.77) | 20, all correct |
| One pass of the tracker alone (81.8 HOTA) | +0.93 (0.58 to 1.31) | 27, all correct |

The constants (gaps, lengths, distances) were set on DanceTrack at 20 frames per second. For other videos, `fps` rescales the durations.

## Proof

[`research/replays/identity_agree.py`](../research/replays/identity_agree.py) runs the module on the stored inputs of the system and compares its output with the stored identities. 0 rows differ out of 510,529 on the 57 annotated videos, and 0 out of 286,325 on the 35 test videos.
