"""Replay of the identity chain on the 35 test videos, step by step, against the frozen identities.

The folder holds the video folders, the appearance stages and the identities stored after each step. No
annotation is read. The replay data are described in docs/REPRODUCE.md.

    python identity_steps_test.py <replay data>/test35/identity_steps <step>
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stored

T = sys.argv[1]
step = sys.argv[2]
V = f"{T}/videos"
W = f"{T}/work"
S = f"{T}/stages"
vids = sorted(os.listdir(V))


def tracks(v):
    return np.loadtxt(f"{V}/{v}/tracks.txt", delimiter=",", ndmin=2)[:, :6]


if step == "merge":
    from holdtrack.identity import merge

    ref = np.load(f"{W}/s01_merge/ids.npz")
    dif = 0
    n = 0
    for v in vids:
        d = stored.load(f"{S}/classifier/{v}.npz")
        meta = np.load(f"{W}/crops/{v}.npz", allow_pickle=True)["meta"]
        new, nb, t = merge.merge(tracks(v), meta, d["log_probs"], d["classes"])
        dif += int((new != ref[v]).sum())
        n += len(new)
    print(f"merge: {dif} row(s) differ out of {n}")
if step in ("links", "second"):
    from holdtrack.identity import chain, links
    from holdtrack.identity import video as vid

    sub = "links" if step == "links" else "second"
    D = stored.load(f"{W}/{sub}/pairs.npz")
    columns = D["names"].tolist()
    SC = np.load(
        f"{W}/links/scores.npz" if step == "links" else f"{W}/second/scores_mean.npz",
        allow_pickle=True,
    )["score"]
    before = np.load(f"{W}/s01_merge/ids.npz" if step == "links" else f"{W}/s02_links/ids.npz")
    ref = np.load(f"{W}/s02_links/ids.npz" if step == "links" else f"{W}/s03_proposal/ids.npz")
    model = (
        links.LinearModel(os.path.join(os.path.dirname(chain.__file__), "data", "link_model.npz"))
        if step == "links"
        else links.TrainedPair(
            os.path.join(os.path.dirname(chain.__file__), "data", "second_link_pairs.npz")
        )
    )
    npairs = 0
    dpairs = 0
    dfeat = 0
    dsc = 0.0
    dif = 0
    n = 0
    for v in vids:
        V_ = vid.Video(f"{V}/{v}")
        meta = np.load(f"{W}/crops/{v}.npz", allow_pickle=True)["meta"]
        emb = stored.load(f"{S}/embedding/{v}.npz")["embedding"]
        ft = stored.load(f"{S}/features/{v}.npz")["features"]
        pairs = links.build_pairs(
            V_,
            before[v],
            meta,
            emb,
            ft,
            "full" if step == "links" else "upper",
            second=(step == "second"),
        )
        m = D["video"] == v
        npairs += int(m.sum())
        if (
            len(pairs["a"]) != m.sum()
            or (pairs["a"] != D["a"][m]).any()
            or (pairs["b"] != D["b"][m]).any()
        ):
            dpairs += 1
            print("pairs differ", v, len(pairs["a"]), int(m.sum()))
            continue
        for k, column in stored.PAIR_COLUMNS.items():
            x = D["X"][m][:, columns.index(column)].astype(np.float64)
            y = pairs["raw"][k]
            dfeat += int(((x != y) & ~(np.isnan(x) & np.isnan(y))).sum())
        sc = model.score(links.model_features(pairs))
        dsc = max(dsc, float(np.abs(sc - SC[m]).max()) if len(sc) else 0.0)
        new, _ = links.assign(pairs, sc, 0.0, -99.0, 1.0 if step == "links" else 0.0)
        dif += int((new != ref[v]).sum())
        n += len(new)
    print(
        f"{step}: {npairs} pairs ; videos whose pairs differ {dpairs} ; raw values that differ {dfeat} ; largest score difference {dsc:.2e} ; {dif} row(s) differ out of {n}"
    )
if step == "exchanges":
    from holdtrack.identity import exchanges as ex
    from holdtrack.identity import video as vid

    A1 = np.load(f"{W}/s02_links/ids.npz")
    B1 = np.load(f"{W}/s03_proposal/ids.npz")
    R3 = np.load(f"{W}/s03_exchanges/ids.npz")
    R4 = np.load(f"{W}/s04_exchanges_backward/ids.npz")
    R5 = np.load(f"{W}/s05_merge_two/ids.npz")
    d3 = d4 = d5 = 0
    n = 0
    for v in vids:
        V_ = vid.Video(f"{V}/{v}")
        meta = np.load(f"{W}/crops/{v}.npz", allow_pickle=True)["meta"]
        emb = stored.load(f"{S}/embedding/{v}.npz")["embedding"]
        ft = stored.load(f"{S}/features/{v}.npz")["features"]
        cl = stored.load(f"{S}/classifier/{v}.npz")
        frame, box, crow, E1, E2 = ex.probes_input(V_, meta, emb, ft)
        a = A1[v].astype(np.int64)
        b = B1[v].astype(np.int64)
        c = a
        if not np.array_equal(a, b):
            c, _ = ex.confirmed_by_second_pass(a, b, frame, box, crow, E1, E2)
        d3 += int((c != R3[v]).sum())
        c4 = R3[v].astype(np.int64)
        if V_.backward is not None:
            c4, _ = ex.proposed_by_backward_pass(
                R3[v].astype(np.int64), V_.backward, frame, box, crow, E1, E2
            )
        d4 += int((c4 != R4[v]).sum())
        c5, _ = ex.merge_two_models(
            R4[v].astype(np.int64),
            frame,
            V_.base[:, 1].astype(int),
            emb,
            crow,
            cl["log_probs"],
            cl["classes"],
        )
        d5 += int((c5 != R5[v]).sum())
        n += len(a)
    print(
        f"exchanges: {d3} row(s) differ ; backward exchanges: {d4} ; merge of two models: {d5} ; out of {n}"
    )
if step == "unsupported":
    from holdtrack.identity import unsupported
    from holdtrack.identity import video as vid

    A1 = np.load(f"{W}/s09_agree_wide/ids.npz")
    R = np.load(f"{W}/s10_unsupported/ids.npz")
    d = 0
    n = 0
    for v in vids:
        new, _ = unsupported.rename_unsupported(vid.Video(f"{V}/{v}"), A1[v])
        d += int((new != R[v]).sum())
        n += len(new)
    print(f"unsupported: {d} row(s) differ out of {n}")
if step == "take_over":
    from holdtrack.identity import take_over
    from holdtrack.identity import video as vid

    A1 = np.load(f"{W}/s10_unsupported/ids.npz")
    R = np.load(f"{W}/s11_take_over/ids.npz")
    d = 0
    n = 0
    for v in vids:
        new, _ = take_over.take_over(vid.Video(f"{V}/{v}"), A1[v])
        d += int((new != R[v]).sum())
        n += len(new)
    print(f"take over: {d} row(s) differ out of {n}")
if step == "short":
    from holdtrack.identity import crops
    from holdtrack.identity import short_tracks as st
    from holdtrack.identity import video as vid

    A1 = np.load(f"{W}/s11_take_over/ids.npz")
    R = np.load(f"{W}/s12_short_tracks/attached.npz")
    same = True
    rows = 0
    dc = 0
    nc = 0
    dm = 0
    for v in vids:
        V_ = vid.Video(f"{V}/{v}")
        z = np.load(f"{W}/crops/{v}.npz", allow_pickle=True)
        ft = stored.load(f"{S}/features/{v}.npz")["features"]
        short = np.load(f"{V}/{v}/short_tracks.npy")
        if (
            v == vids[0] or v == vids[17]
        ):  # the crops are remade from the frames on two videos (all 35 were compared for the first runner)
            jp, meta = crops.track_crops(V_)
            nc += len(jp)
            dc += int(len(jp) != len(z["jpeg"])) + sum(
                int(bytes(a) != bytes(b)) for a, b in zip(jp, z["jpeg"])
            )
            dm += int((meta != z["meta"]).sum())
            if len(short):
                zs = np.load(f"{W}/crops_short/{v}.npz", allow_pickle=True)
                jp, meta = crops.short_track_crops(V_, short)
                nc += len(jp)
                dc += int(len(jp) != len(zs["jpeg"])) + sum(
                    int(bytes(a) != bytes(b)) for a, b in zip(jp, zs["jpeg"])
                )
                dm += int((meta != zs["meta"]).sum())
        fs = f"{W}/crops_short/{v}.npz"
        out = (
            st.attach(
                V_,
                A1[v],
                short,
                z["meta"],
                ft,
                np.load(fs, allow_pickle=True)["meta"],
                stored.load(f"{S}/features_short/{v}.npz")["features"],
            )[0]
            if len(short) and os.path.exists(fs)
            else np.zeros((0, 7))
        )
        ref = R[v] if v in R.files else np.zeros((0, 7))
        same &= bool(out.shape == ref.shape and np.array_equal(out, ref))
        rows += len(out)
    print(
        f"short tracks: attached rows identical: {same} ({rows} rows) ; crops remade on two videos: {nc}, different {dc}, meta values different {dm}"
    )
if step == "agreement":
    from holdtrack.identity import agreement as ag
    from holdtrack.identity import video as vid

    E = {
        k: np.load(f"{W}/{k}/ids.npz")
        for k in (
            "s05_merge_two",
            "s06_agree_links",
            "s07_agree_boxes",
            "s08_agree_forward",
            "s09_agree_wide",
        )
    }
    d = [0, 0, 0, 0]
    n = 0
    for v in vids:
        V_ = vid.Video(f"{V}/{v}")
        n += V_.n
        d[0] += int((ag.links(V_, E["s05_merge_two"][v]) != E["s06_agree_links"][v]).sum())
        d[1] += int((ag.boxes(V_, E["s06_agree_links"][v]) != E["s07_agree_boxes"][v]).sum())
        d[2] += int(
            (
                ag.boxes_with_forward(V_, E["s07_agree_boxes"][v])[0] != E["s08_agree_forward"][v]
            ).sum()
        )
        d[3] += int(
            (
                ag.boxes_with_forward(V_, E["s08_agree_forward"][v], wide=True)[0]
                != E["s09_agree_wide"][v]
            ).sum()
        )
    print(f"agreement: rows that differ, steps 6 to 9: {d} out of {n}")
if step == "chain":
    # end to end: crops remade from the frames (given), every step feeds the next; stage outputs of the freeze passed as arrays
    from holdtrack.identity import chain, crops
    from holdtrack.identity import short_tracks as st
    from holdtrack.identity import video as vid

    NAMES = dict(
        merge="s01_merge",
        links="s02_links",
        exchanges="s03_exchanges",
        exchanges_backward="s04_exchanges_backward",
        merge_two="s05_merge_two",
        agree_links="s06_agree_links",
        agree_boxes="s07_agree_boxes",
        agree_forward="s08_agree_forward",
        agree_wide="s09_agree_wide",
        unsupported="s10_unsupported",
        take_over="s11_take_over",
    )
    REF = {k: np.load(f"{W}/{d}/ids.npz") for k, d in NAMES.items()}
    R = np.load(f"{W}/s12_short_tracks/attached.npz")
    dif = {k: 0 for k in NAMES}
    n = 0
    same = True
    rows = 0
    for v in vids:
        V_ = vid.Video(f"{V}/{v}")
        z = np.load(f"{W}/crops/{v}.npz", allow_pickle=True)
        cl = stored.load(f"{S}/classifier/{v}.npz")
        emb = stored.load(f"{S}/embedding/{v}.npz")["embedding"]
        ft = stored.load(f"{S}/features/{v}.npz")["features"]
        out = chain.identities(V_, z["meta"], cl["log_probs"], cl["classes"], emb, ft)
        n += V_.n
        for k in NAMES:
            dif[k] += int((out[k] != REF[k][v]).sum())
        short = np.load(f"{V}/{v}/short_tracks.npy")
        fs = f"{W}/crops_short/{v}.npz"
        add = (
            st.attach(
                V_,
                out["take_over"],
                short,
                z["meta"],
                ft,
                np.load(fs, allow_pickle=True)["meta"],
                stored.load(f"{S}/features_short/{v}.npz")["features"],
            )[0]
            if len(short) and os.path.exists(fs)
            else np.zeros((0, 7))
        )
        ref = R[v] if v in R.files else np.zeros((0, 7))
        same &= bool(add.shape == ref.shape and np.array_equal(add, ref))
        rows += len(add)
        print(v, {k: dif[k] for k in ("merge", "take_over")}, flush=True)
    print(
        f"chain, end to end, {len(vids)} videos, {n} rows: rows that differ from the frozen export after each step: {dif} ; attached rows identical: {same} ({rows})"
    )
    print("VERDICT:", "IDENTICAL" if not any(dif.values()) and same else "DIFFERENT")
