"""Replay of the face steps on the 35 videos of the DanceTrack test set, to the final tracks.

Starts from a work folder of `chain_test35.py` (identities of the chain without faces, short-track rows,
corrected boxes) and from the face tables; runs the face steps, the relay in place, its check and the
assembly. The 35 files must be byte-identical to the submission of the complete system. CPU only; no
annotation. The replay data are described in docs/REPRODUCE.md.

    python faces_chain_test35.py <replay data> <work folder of chain_test35.py> <output folder>
"""

import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from holdtrack.boxes import assemble as A
from holdtrack.boxes import relay as RL
from holdtrack.faces import chain
from holdtrack.faces.pieces import FaceTable
from holdtrack.identity import check
from holdtrack.identity.video import Video

DATA, W, OUT = (os.path.abspath(x) for x in sys.argv[1:4])
SET = DATA + "/test35"
FACES = SET + "/faces/{t}/{v}.npz"
POSE = SET + "/pose133/{v}.npz"
TARGET = SET + "/submissions/complete"
os.makedirs(OUT, exist_ok=False)
t0 = time.time()
same = rows_total = kept = removed = changed = 0
videos = sorted(os.path.basename(p) for p in glob.glob(W + "/videos/dancetrack????"))
for v in videos:
    d = f"{W}/videos/{v}"
    video = Video(d)
    ids = np.load(f"{d}/identities.npy")
    att = np.load(f"{d}/attached.npy").reshape(-1, 7)
    q = np.loadtxt(f"{W}/boxes_full/{v}.txt", delimiter=",", ndmin=2)
    boxes = np.stack([q[:, 2], q[:, 3], q[:, 2] + q[:, 4], q[:, 3] + q[:, 5]], 1)
    kp = np.load(POSE.format(v=v))["kp"].astype(np.float64)
    eyes = (kp[:, 59:65, :2].mean(1) + kp[:, 65:71, :2].mean(1)) / 2
    tables = chain.Tables(
        *(
            FaceTable.load(FACES.format(t=t, v=v))
            for t in ("arcface", "adaface", "arcface_clean", "adaface_clean")
        )
    )
    faces_ids, faces_att = chain.run(video, ids, att, boxes, tables, eyes)
    changed += int((faces_ids != np.asarray(ids).ravel()).sum())
    _, _, found = RL.relay(video.frame, faces_ids, boxes, faces_att)
    out = check.relays(d, faces_ids, [(c[1], c[2]) for c in found]) if found else []
    new, new_att, found = RL.relay(
        video.frame, faces_ids, boxes, faces_att, excluded={(int(x), int(y)) for x, y in out}
    )
    kept += len(found)
    removed += len(out)
    rows = A.assemble(
        [line for line in open(f"{W}/boxes_full/{v}.txt").read().split("\n") if line],
        ids,
        att,
        new,
        new_att,
    )
    data = ("\n".join(rows) + "\n").encode()
    open(f"{OUT}/{v}.txt", "wb").write(data)
    rows_total += len(rows)
    ok = data == open(f"{TARGET}/{v}.txt", "rb").read()
    same += int(ok)
    print(v, "identical" if ok else "DIFFERENT", flush=True)
print(
    f"rows renamed by the face steps: {changed}; relays kept: {kept}; removed by the check: {removed}; rows: {rows_total}"
)
print(
    f"FINAL FILES BYTE-IDENTICAL TO THE SUBMISSION OF THE COMPLETE SYSTEM: {same} of {len(videos)}; {time.time() - t0:.0f} s"
)
