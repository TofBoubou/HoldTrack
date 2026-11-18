"""Replay of the steps after the box corrector on the seven videos of the MOT17 test set.

From the stored outputs of the models (tracks of the base system, the same tracks after the
box corrector, appearance features, camera motion, mask boxes of the forward and backward
passes, detections) to the files of the submission: identity at the return of a person,
earlier start of each track, gaps filled under camera motion, smoothing of the edges. The
server lists 42 files, the 14 videos under three detector names; the seven training videos
are filled with stored public tracks. Each file must be byte-identical to the submission.
CPU only; no annotation. The replay data are described in docs/MOT17.md.

    python mot17_test.py <replay data> <work folder>
"""

import hashlib
import json
import os
import sys
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from holdtrack.identity.returns import ReturnVideo, cut_returns, link_returns
from holdtrack.steps.camera_motion import load_motion
from holdtrack.steps.finishing import finish, mot_lines

DATA = os.path.abspath(sys.argv[1]) + "/mot17"
WORK = os.path.abspath(sys.argv[2])
TEST = DATA + "/test"
SEQUENCES = json.load(open(DATA + "/sequences.json"))
VIDEOS = sorted(name for name, info in SEQUENCES.items() if info["split"] == "test")
TRAINING = sorted(name for name, info in SEQUENCES.items() if info["split"] == "train")
SUFFIXES = ("DPM", "FRCNN", "SDP")


def rows_of(path):
    return np.loadtxt(path, delimiter=",", ndmin=2)


def normalized(path, frame_count):
    """Rows of a MOT file, sorted and written with two decimals."""
    rows = rows_of(path)
    frames = rows[:, 0].astype(np.int64)
    identities = rows[:, 1].astype(np.int64)
    boxes = np.stack([rows[:, 2], rows[:, 3], rows[:, 2] + rows[:, 4], rows[:, 3] + rows[:, 5]], 1)
    order = np.lexsort((identities, frames))
    frames, identities, boxes = frames[order], identities[order], boxes[order]
    assert frames.min() >= 1 and frames.max() <= frame_count, path
    assert len(set(zip(frames.tolist(), identities.tolist()))) == len(frames), path
    return mot_lines(frames, identities, boxes)


os.makedirs(WORK + "/identities", exist_ok=True)
os.makedirs(WORK + "/submission", exist_ok=True)

# Identity at the return of a person: cut, then link.
videos = {}
for video in VIDEOS:
    info = SEQUENCES[video]
    rows = rows_of(f"{TEST}/tracks/{video}.txt")
    frames = rows[:, 0].astype(np.int64)
    boxes = np.c_[rows[:, 2:4], rows[:, 2:4] + rows[:, 4:6]]
    motion = load_motion(f"{TEST}/camera/{video}.npz")
    cut, cuts = cut_returns(frames, rows[:, 1], boxes, info["fps"], motion)
    videos[video] = ReturnVideo(
        frames,
        cut,
        boxes,
        info["fps"],
        info["width"],
        info["height"],
        motion,
        np.load(f"{TEST}/crops/{video}.npy"),
        np.load(f"{TEST}/features/{video}.npz")["features"],
    )
    print(f"{video}: {len(rows)} rows, {cuts} cuts", flush=True)
identities, links = link_returns(videos)
for video in VIDEOS:
    np.save(f"{WORK}/identities/{video}.npy", identities[video])
    print(f"{video}: {len(links[video])} links", flush=True)

# Earlier starts, gaps, smoothing, on the tracks after the box corrector.
files = {}
for video in VIDEOS:
    info = SEQUENCES[video]
    bank = np.load(f"{TEST}/detections/{video}.npz")
    frames, names, boxes, counts = finish(
        rows_of(f"{TEST}/corrected/{video}.txt"),
        info["fps"],
        info["frames"],
        videos[video].motion,
        identities=identities[video],
        forward_masks=rows_of(f"{TEST}/masks_forward/{video}.txt"),
        backward_masks=rows_of(f"{TEST}/masks_backward/{video}.txt"),
        detections=(
            bank["frame"].astype(np.int64) + 1,
            bank["boxes"].astype(float),
            bank["score"].astype(float),
        ),
    )
    files[video] = "\n".join(mot_lines(frames, names, boxes)) + "\n"
    print(f"{video}: {len(frames)} rows, {counts}", flush=True)
for video in TRAINING:
    lines = normalized(f"{DATA}/train/public/{video}.txt", SEQUENCES[video]["frames"])
    files[video] = "\n".join(lines) + "\n"

# The 42 files of the submission, and their comparison with the stored digests.
expected = dict(
    line.split()[::-1] for line in open(DATA + "/submission.sha256").read().splitlines() if line
)
names = []
identical = 0
for video, text in sorted(files.items()):
    digest = hashlib.sha256(text.encode()).hexdigest()
    same = expected.get(video + ".txt") == digest
    identical += same
    print(f"{video}: {'identical' if same else 'DIFFERENT'} {digest}", flush=True)
    for suffix in SUFFIXES:
        name = f"{video[:-6]}-{suffix}.txt"
        open(f"{WORK}/submission/{name}", "w").write(text)
        names.append(name)
with zipfile.ZipFile(WORK + "/submission.zip", "w", zipfile.ZIP_DEFLATED) as archive:
    for name in sorted(names):
        entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
        entry.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(entry, open(f"{WORK}/submission/{name}", "rb").read())
print(f"{identical} of {len(files)} files identical to the submission; {len(names)} files zipped")
sys.exit(0 if identical == len(files) else 1)
