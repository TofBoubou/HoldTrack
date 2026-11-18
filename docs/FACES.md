# Faces

    holdtrack track clip.mp4 --full --faces

adds the face steps to the complete system. With them, it scores 89.4 HOTA on the DanceTrack test set, against 87.9 without them. The steps read two public face models, ArcFace R100 and AdaFace IR-101. Both are fetched from their authors and verified, and neither is part of this repository.

    python -m holdtrack.faces.fetch_face_models --dest ~/.cache/holdtrack/faces

`--faces` takes the folder of the models (default `~/.cache/holdtrack/faces`). The face tables need the `faces` extra, `pip install -e ".[faces]"`, and `--faces-python` gives them their own interpreter.

## What runs

The steps run after the identity chain, on its identities, and before the relay in place.

| step | module | what it does |
| --- | --- | --- |
| tables | [`tables`](../holdtrack/faces/tables.py), [`second_eye`](../holdtrack/faces/second_eye.py) | a 112 × 112 face aligned on five of the 133 keypoints, for every box whose eyes are six pixels apart; its embedding by the two models; the tables without the faces shared by two boxes |
| whole identities and pieces | [`pieces`](../holdtrack/faces/pieces.py) | an identity, or a piece of it delimited by a gap or a change of track, moves to the identity whose gallery of faces it resembles; a piece needs a pass of the tracker that agrees; identities never present together whose galleries match are merged |
| the face and a reader | [`reader`](../holdtrack/faces/reader.py) | a row takes the identity its face designates when a pass of the tracker holds that identity |
| cuts and links | [`cuts`](../holdtrack/faces/cuts.py) | a name is cut where its face changes, and the parts are linked again by their faces |
| pieces of both models | [`pieces`](../holdtrack/faces/pieces.py) | pieces proposed by both models, the candidates recomputed after each operation |
| exchanges | [`exchange`](../holdtrack/faces/exchange.py), [`identity.check`](../holdtrack/identity/check.py) | two identities are swapped over an interval that the faces find and that both models validate; an exchange is removed when four of the five passes keep the identities in place and half of the faces have their eyes inside the box of the other dancer |

The last three steps are repeated until a turn changes nothing, three turns at most ([`chain`](../holdtrack/faces/chain.py)). Every threshold is fixed. The two that depend on the video (the face and a reader, and the links after a cut) are read on the video itself. No step reads an annotation.

## Tables

The tables of a video are written in `<video>/faces/`:

| file | content |
| --- | --- |
| `aligned.npz` | the aligned faces (JPEG), the row of each face in `tracks.txt`, the distance between the eyes |
| `arcface.npz`, `adaface.npz` | one embedding per face, and the norm of the raw embedding, the measure of its quality |
| `shared.npz` | boxes of one frame on which the pose found the same face |
| `<model>_clean.npz` | the table without the faces shared by two boxes |
| `<model>_clean_projected.npz` | the same, without the component common to all the faces of one moment |

```python
from holdtrack.faces import chain
chain.prepare_tables("videos/dancetrack0003", "~/.cache/holdtrack/faces")   # expanded path
identities, short = chain.run_folder("videos/dancetrack0003", identities, short, boxes)
```

The operations are also functions on arrays: `tables.align`, `second_eye.embed`, `second_eye.shared_faces`, `pieces.run`, `reader.face_and_reader`, `cuts.cut_and_link`, `exchange.exchange`.

## Replay

[`research/replays/faces_chain_test35.py`](../research/replays/faces_chain_test35.py) runs the face steps, the relay, its check and the assembly on the 35 test videos. It starts from the identities of the chain without faces and from the stored face tables. In 99 seconds on a CPU, it writes the 35 files byte-identical to the submission of the complete system.

The steps were also run on a rented A40, from the frames and the 133 keypoints of two test videos, with the two models fetched from their authors. The tables take 90 and 59 seconds and the steps 11 and 97 seconds, and every row has the identity of the submission (28 530 rows).
