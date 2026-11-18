"""End-to-end replay of the chain without faces on the 35 videos of the DanceTrack test set.

From the stored outputs of the models (detector reads, tracker passes, poses, meshes, the
tracks of the base system) to the final tracks: table, reference corrector, inputs of the
identity chain, identity chain, relay, its check, full corrector, assembly. The 35 files
must be byte-identical to the submission of the system without faces. CPU only; no annotation.
The replay data are described in docs/REPRODUCE.md.

    python chain_test35.py <replay data> <work folder> [table|reference|inputs|identity|full|final|all]
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stored

DATA = os.path.abspath(sys.argv[1])
SET = DATA + "/test35"
MODELS = DATA + "/models/corrector"
TARGET = SET + "/submissions/without_faces"
W = os.path.abspath(sys.argv[2])
what = sys.argv[3] if len(sys.argv) > 3 else "all"
VIDEOS = sorted(os.listdir(SET + "/base"))
os.environ["HOLDTRACK_MHR70"] = DATA + "/models/sam3d_body/mhr70.py"


def link(src, dst):
    assert os.path.exists(src), src
    if not os.path.lexists(dst):
        os.symlink(src, dst)


def boxes_of(path):
    q = np.loadtxt(path, delimiter=",", ndmin=2)
    return np.stack([q[:, 2], q[:, 3], q[:, 2] + q[:, 4], q[:, 3] + q[:, 5]], 1)


def dfine(path):
    fr, bx, sc = [], [], []
    d = json.load(open(path))
    rows = d if isinstance(d, list) else d.get("predictions", d.get("rows", d))
    for r in rows:
        if isinstance(r, dict):
            f, b, s = (
                r.get("frame", r.get("image_id")),
                r.get("box", r.get("bbox", r.get("xyxy"))),
                r.get("score"),
            )
        else:
            f, b, s = r[0] + 1, r[2:6], r[6]
        fr.append(int(f))
        bx.append([float(x) for x in b])
        sc.append(float(s))
    return np.array(fr), np.array(bx, float).reshape(-1, 4), np.array(sc)


def step_table():
    from holdtrack.boxes import table as T

    def npz(path):
        a = np.load(path)
        return a["frame"].astype(int) + 1, a["boxes"].astype(float), a["score"].astype(float)

    for d in ("tables_full", "tables_ten", "tables_reference", "tracks", "mesh"):
        os.makedirs(f"{W}/{d}", exist_ok=True)
    equal = 0
    for v in VIDEOS:
        q = np.loadtxt(f"{SET}/base/{v}/tracks.txt", delimiter=",", ndmin=2)
        m = np.load(f"{SET}/detectors/rf_2xl_mirror/{v}.npz")
        r = np.load(f"{SET}/reads/1280/{v}.npz")
        full = [
            (*npz(f"{SET}/detectors/rf_2xl/{v}.npz"), 0.3),
            (
                m["image_id"] - np.unique(m["image_id"]).min() + 1,
                m["xyxy"].astype(float),
                m["score"].astype(float),
                0.3,
            ),
            (*npz(f"{SET}/detectors/yolo/{v}.npz"), 0.3),
            (*npz(f"{SET}/detectors/yolo_mirror/{v}.npz"), 0.3),
            (*dfine(f"{SET}/detectors/dfine/{v}.json"), 0.5),
            (r["frame"].astype(int), r["boxes"].astype(float), r["score"].astype(float), 0.3),
        ]
        bands = []
        for count, res in ((3, 1120), (5, 1280)):
            one_view = []
            for k in range(count):
                b = stored.load(f"{SET}/reads/band_{res}_{k}/{v}.npz")
                bd = b["band"].astype(float)
                one_view.append(
                    (
                        float(bd[0, 0]),
                        float(bd[0, 1]),
                        b["frame"].astype(int),
                        b["boxes"].astype(float),
                        b["score"].astype(float),
                    )
                )
            bands.append(one_view)
        crops = []
        for k in (1, 2):
            z = np.load(f"{SET}/reads/crops_{k}/{v}.npz")
            key = {(int(f), int(i)): n for n, (f, i) in enumerate(zip(z["frame"], z["pid"]))}
            idx = np.array([key[(int(f), int(i))] for f, i in zip(q[:, 0], q[:, 1])])
            crops.append(
                (
                    z["box"][idx].astype(float),
                    z["iou"][idx].astype(float),
                    z["score"].astype(float)[idx],
                )
            )
        mesh = stored.load(f"{SET}/mesh/{v}.npz")
        mesh_box = np.full((len(q), 4), np.nan, np.float32)
        mesh_box[mesh["row"]] = mesh["mesh_box"]
        table, masks = T.build_table(
            q,
            full,
            bands,
            crops,
            [
                np.loadtxt(f"{SET}/masks/{v}/{p}.txt", delimiter=",")
                for p in ("native", "exclusive", "guarded")
            ],
            mesh_box,
            np.load(f"{SET}/pose17/{v}.npz")["kp"],
            np.load(f"{SET}/pose133/{v}.npz")["kp"],
        )
        np.savez(f"{W}/tables_full/{v}.npz", **table)
        np.savez(f"{W}/tables_full/{v}_masks.npz", **masks)
        # the trees of the full corrector read the ten detector views; those of the reference corrector, the first nine;
        # the three body views are read by the sequence networks only
        for name, count in (("tables_ten", 10), ("tables_reference", 9)):
            part = dict(table)
            part.update(
                views=table["views"][:count],
                views_iou=table["views_iou"][:count],
                views_score=table["views_score"][:count],
            )
            np.savez(f"{W}/{name}/{v}.npz", **part)
            np.savez(f"{W}/{name}/{v}_masks.npz", **masks)
        link(f"{SET}/base/{v}/tracks.txt", f"{W}/tracks/{v}.txt")
        np.savez(f"{W}/mesh/{v}.npz", **mesh)
        frozen = stored.load(f"{SET}/tables/{v}.npz")
        frozen_masks = stored.load(f"{SET}/tables/{v}_masks.npz")
        equal += int(
            all(np.array_equal(table[k], frozen[k], equal_nan=True) for k in frozen)
            and all(np.array_equal(masks[k], frozen_masks[k], equal_nan=True) for k in frozen_masks)
        )
    print(
        "tables built:",
        len(VIDEOS),
        "videos; equal to the frozen tables, every array:",
        equal,
        flush=True,
    )


def step_corrector(full):
    from holdtrack.boxes import corrector as K

    N = SET + "/networks/"
    P = Path
    if full:
        layout = K.Layout(
            P(f"{W}/tables_ten"),
            P(f"{W}/tracks"),
            P(MODELS + "/trees.pkl"),
            (P(N + "full_0.npz"), P(N + "full_1.npz")),
            P(SET + "/pose17"),
            P(SET + "/pose133"),
            P(f"{W}/mesh"),
            P(MODELS + "/pose_thresholds.json"),
            P(os.environ["HOLDTRACK_MHR70"]),
        )
    else:
        layout = K.Layout(
            P(f"{W}/tables_reference"),
            P(f"{W}/tracks"),
            P(MODELS + "/reference_trees.pkl"),
            (P(N + "reference_0.npz"), P(N + "reference_1.npz")),
            None,
            None,
            None,
            None,
            None,
        )
    m = K.correct(layout, f"{W}/boxes_full" if full else f"{W}/boxes_reference")
    print(
        "corrector,", "full" if full else "reference", ":", len(m["videos"]), "videos", flush=True
    )


def step_inputs():
    from holdtrack.boxes import short_tracks as S
    from holdtrack.boxes import upper_body as U
    from holdtrack.boxes import visibility as V

    for v in VIDEOS:
        d = f"{W}/videos/{v}"
        os.makedirs(f"{d}/passes", exist_ok=True)
        a = np.load(f"{W}/tables_reference/{v}.npz")
        masks = np.load(f"{W}/tables_reference/{v}_masks.npz")["masks"]
        b = boxes_of(f"{W}/boxes_reference/{v}.txt")
        with open(f"{d}/tracks.txt", "w") as o:
            for line in open(f"{SET}/base/{v}/tracks.txt").read().split("\n"):
                if line:
                    o.write(",".join(line.split(",")[:6]) + "\n")
        np.save(f"{d}/visibility.npy", V.visibility(b, masks, a["hidden"], a["contact"]))
        np.save(
            f"{d}/upper_body.npy",
            U.upper_body(a["frame"], a["pid"], b, np.load(f"{SET}/pose17/{v}.npz")["kp"]),
        )
        np.save(
            f"{d}/short_tracks.npy",
            S.short_tracks(
                np.loadtxt(f"{SET}/base/{v}/fusion.txt", delimiter=",", ndmin=2),
                a["frame"],
                a["ref"],
            ),
        )
        for y in ("native", "exclusive", "guarded", "vote", "backward", "mirrored"):
            link(f"{SET}/passes/{v}/{y}.txt", f"{d}/passes/{y}.txt")
        link(f"{SET}/base/{v}/bytetrack.txt", f"{d}/bytetrack.txt")
        link(f"{DATA}/dancetrack/{v}/img1", f"{d}/frames")
    print("inputs of the identity chain written:", len(VIDEOS), "videos", flush=True)


def step_identity():
    from holdtrack import identity

    for v in VIDEOS:
        identity.run(f"{W}/videos/{v}", SET + "/stages", fps=20.0)
    print("identity chain done", flush=True)


def step_final():
    from holdtrack.boxes import assemble as A
    from holdtrack.boxes import relay as RL
    from holdtrack.identity import check

    os.makedirs(f"{W}/final", exist_ok=True)
    same = 0
    rows_total = 0
    kept = removed = 0
    for v in VIDEOS:
        d = f"{W}/videos/{v}"
        a = np.load(f"{W}/tables_full/{v}.npz")
        ids = np.load(f"{d}/identities.npy")
        att = np.load(f"{d}/attached.npy").reshape(-1, 7)
        b = boxes_of(f"{W}/boxes_full/{v}.txt")
        _, _, found = RL.relay(a["frame"], ids, b, att)
        out = check.relays(d, ids, [(c[1], c[2]) for c in found]) if found else []
        new, new_att, found = RL.relay(
            a["frame"], ids, b, att, excluded={(int(x), int(y)) for x, y in out}
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
        open(f"{W}/final/{v}.txt", "wb").write(data)
        rows_total += len(rows)
        same += int(data == open(f"{TARGET}/{v}.txt", "rb").read())
    print(f"relays kept: {kept}; removed by the check: {removed}; rows: {rows_total}")
    print(
        f"FINAL FILES BYTE-IDENTICAL TO THE SUBMISSION WITHOUT FACES: {same} of {len(VIDEOS)}",
        flush=True,
    )


t0 = time.time()
for name, fn in (
    ("table", step_table),
    ("reference", lambda: step_corrector(False)),
    ("inputs", step_inputs),
    ("identity", step_identity),
    ("full", lambda: step_corrector(True)),
    ("final", step_final),
):
    if what in (name, "all"):
        t = time.time()
        fn()
        print(f"  [{name}: {time.time() - t:.0f} s]", flush=True)
print(f"total {time.time() - t0:.0f} s")
