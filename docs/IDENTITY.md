# Identity chain

The identity chain corrects the identities of a base tracker on one video in twelve steps, without faces. Each step changes an identity only on a stated proof, and `operations.json` records what each step changed.

| step | module | what it does |
|---|---|---|
| 1 | `merge.py` | merges whole identities that the appearance classifier of the video names as one person |
| 2 | `links.py` | links the fully visible stretches again, with a classifier of pairs of stretches |
| 3 | `links.py`, `exchanges.py` | applies the exchanges at a contact that a second pass of links proposes and two appearance probes confirm |
| 4 | `exchanges.py` | the same, proposed by the backward pass |
| 5 | `exchanges.py` | merges two identities that two appearance models put together |
| 6, 7 | `agree.py` | links, then boxes, on which the backward pass and the mirrored pass agree |
| 8, 9 | `agreement.py` | boxes on which the backward pass and a forward pass agree |
| 10 | `unsupported.py` | renames a box whose identity no pass supports |
| 11 | `take_over.py` | when an identity takes over the body of another, the shorter run takes the name of the other |
| 12 | `short_tracks.py` | attaches the short tracks that the base system removed, when two proofs agree |

## Use

```python
from holdtrack import identity
from holdtrack.identity import appearance

appearance.run("videos/dancetrack0003", "stages", "appearance_network.pt")   # GPU: classifier, embedding, features
identity.run("videos/dancetrack0003", "stages")                              # CPU: the twelve steps
```

The folder of a video holds `tracks.txt`, `visibility.npy`, `upper_body.npy`, `short_tracks.npy`, `bytetrack.txt`, `passes/{native,exclusive,guarded,vote,backward,mirrored}.txt` and `frames/` (see `video.py`). The run writes `identities.npy` (one identity per row of `tracks.txt`), `attached.npy` (boxes of short tracks attached) and `operations.json`.

By default, `crops.track_crops` makes the crops with the bicubic filter when a box is larger than the crop. With `downscale=Image.BILINEAR`, they are byte for byte those of the 57 annotated videos of DanceTrack.

Durations are written in seconds and converted with the frame rate of the video (`fps`, 20 by default). The values were set on DanceTrack, at 20 frames per second.

## Final check

`check.relays` and `check.exchanges` remove a take-over or an interval exchange placed after the chain when the passes contradict it (see `check.py`).

## Data

`data/link_model.npz` holds the weights of the classifier of step 2. `data/second_link_pairs.npz` holds the labelled pairs of stretches on which the two classifiers of step 3 are fitted when the chain starts.
