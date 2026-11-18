"""Replay of the identity chain on the 57 annotated videos, steps 3 to 12, against the frozen identities.

Steps 1 and 2 are learned out of group on the 57 videos, so their frozen identities are given. No annotation
is read. The replay data are described in docs/REPRODUCE.md.

    python identity_steps_annotated57.py <replay data> <work folder>
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stored

from holdtrack.identity import (
    agreement as ag,
)
from holdtrack.identity import (
    crops,
    take_over,
    unsupported,
)
from holdtrack.identity import (
    exchanges as ex,
)
from holdtrack.identity import (
    short_tracks as st,
)
from holdtrack.identity import (
    video as vid,
)

DATA, WORK = (os.path.abspath(x) for x in sys.argv[1:3])
SET = DATA + "/annotated57"
S = SET + "/stages"
STEPS = (
    "exchanges",
    "exchanges_backward",
    "merge_two",
    "agree_links",
    "agree_boxes",
    "agree_forward",
    "agree_wide",
    "unsupported",
    "take_over",
)
REF = {k: np.load(f"{SET}/identities/{k}.npz") for k in STEPS}
FIRST = np.load(f"{SET}/identities/links_first.npz")
SECOND = np.load(f"{SET}/identities/links_second.npz")
ATTACHED = np.load(f"{SET}/attached.npz")
videos = sorted(os.listdir(SET + "/videos"))
remade = (videos[0], videos[40])  # crops remade from the frames on two videos


def link(a, b):
    assert os.path.exists(a), a
    if not os.path.lexists(b):
        os.symlink(a, b)


dif = {k: 0 for k in STEPS}
n = 0
same = True
rows = 0
dc = 0
nc = 0
t0 = time.time()
for v in videos:
    d = f"{WORK}/{v}"
    os.makedirs(f"{d}/passes", exist_ok=True)
    for k in (
        "tracks.txt",
        "visibility.npy",
        "upper_body.npy",
        "short_tracks.npy",
        "bytetrack.txt",
    ):
        link(f"{SET}/videos/{v}/{k}", f"{d}/{k}")
    for k in ("native", "exclusive", "guarded", "vote", "backward", "mirrored"):
        link(f"{SET}/videos/{v}/passes/{k}.txt", f"{d}/passes/{k}.txt")
    link(f"{DATA}/dancetrack/{v}/img1", f"{d}/frames")
    V_ = vid.Video(d)
    z = np.load(f"{S}/crops/{v}.npz", allow_pickle=True)
    meta = z["meta"]
    emb = stored.load(f"{S}/embedding/{v}.npz")["embedding"]
    ft = stored.load(f"{S}/features/{v}.npz")["features"]
    cl = stored.load(f"{S}/classifier/{v}.npz")
    if v in remade:
        from PIL import Image

        jp, m2 = crops.track_crops(
            V_, downscale=Image.BILINEAR
        )  # the crops of the 57 videos were made with the bilinear filter
        nc += len(jp)
        dc += (
            int(len(jp) != len(z["jpeg"]))
            + sum(int(bytes(a) != bytes(b)) for a, b in zip(jp, z["jpeg"]))
            + int((m2 != meta).sum())
        )
    frame, box, crow, E1, E2_ = ex.probes_input(V_, meta, emb, ft)
    a = FIRST[v].astype(np.int64)
    b = SECOND[v].astype(np.int64)
    out = {}
    ids = a
    if not np.array_equal(a, b):
        ids, _ = ex.confirmed_by_second_pass(a, b, frame, box, crow, E1, E2_)
    out["exchanges"] = ids
    if V_.backward is not None:
        ids, _ = ex.proposed_by_backward_pass(
            ids.astype(np.int64), V_.backward, frame, box, crow, E1, E2_
        )
    out["exchanges_backward"] = ids
    ids, _ = ex.merge_two_models(
        ids.astype(np.int64),
        frame,
        V_.base[:, 1].astype(int),
        emb,
        crow,
        cl["log_probs"],
        cl["classes"],
    )
    out["merge_two"] = ids
    ids = ag.links(V_, ids)
    out["agree_links"] = ids
    ids = ag.boxes(V_, ids)
    out["agree_boxes"] = ids
    ids, _ = ag.boxes_with_forward(V_, ids)
    out["agree_forward"] = ids
    ids, _ = ag.boxes_with_forward(V_, ids, wide=True)
    out["agree_wide"] = ids
    ids, _ = unsupported.rename_unsupported(V_, ids)
    out["unsupported"] = ids
    ids, _ = take_over.take_over(V_, ids)
    out["take_over"] = ids
    for k in STEPS:
        dif[k] += int((out[k] != REF[k][v]).sum())
    short = np.load(f"{d}/short_tracks.npy")
    fs = f"{S}/crops_short/{v}.npz"
    add = np.zeros((0, 7))
    if len(short) and os.path.exists(fs):
        add = st.attach(
            V_,
            ids,
            short,
            meta,
            ft,
            np.load(fs, allow_pickle=True)["meta"],
            stored.load(f"{S}/features_short/{v}.npz")["features"],
        )[0]
    r = ATTACHED[v] if v in ATTACHED.files else np.zeros((0, 7))
    same &= bool(add.shape == r.shape and np.array_equal(add, r))
    rows += len(add)
    n += V_.n
    print(v, sum(dif.values()), same, "%.0f s" % (time.time() - t0), flush=True)
print(
    f"57 annotated videos, {n} rows, steps 3 to 12: rows that differ from the frozen identities after each step: {dif} ; attached rows identical: {same} ({rows}) ; crops remade on two videos: {nc}, differences {dc}"
)
print("VERDICT:", "IDENTICAL" if not any(dif.values()) and same and dc == 0 else "DIFFERENT")
