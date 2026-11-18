"""The identity chain without faces, on one video.

    from holdtrack.identity import chain
    chain.run("videos/dancetrack0003", stages="stages")

The folder of the video is described in video.py, plus `short_tracks.npy` and `frames/`. Three appearance stages
run on a GPU (appearance.py); their outputs are read from `stages`:
    classifier/<video>.npz      log_probs (crops, classes), classes
    embedding/<video>.npz       embedding (crops, 256)
    features/<video>.npz        features (crops, 2048)
    features_short/<video>.npz  features of the crops of the short tracks
The run writes, in the folder of the video:
    identities.npy     identity of each row of tracks.txt
    attached.npy       (K, 7) boxes of short tracks attached: frame, identity, x1, y1, x2, y2, short track
    operations.json    what each step changed
"""

import json
import os

import numpy as np

from . import agreement, crops, exchanges, links, merge, short_tracks, take_over, unsupported
from .video import Video

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
STEPS = (
    "merge",
    "links",
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
_models = {}


def models():
    """Classifier of the links (shipped weights) and classifiers of the second pass (fitted once on the shipped pairs)."""
    if not _models:
        _models["first"] = links.LinearModel(os.path.join(DATA, "link_model.npz"))
        _models["second"] = links.TrainedPair(os.path.join(DATA, "second_link_pairs.npz"))
    return _models["first"], _models["second"]


def identities(video, crops_meta, log_probs, classes, embedding, features, log=None):
    """The eleven identity steps. Returns the identities after each step, by name."""
    first, second = models()
    out = {}
    log = {} if log is None else log
    frame, box, crop_row, E1, E2 = exchanges.probes_input(video, crops_meta, embedding, features)
    ids, n, temperature = merge.merge(video.rows(), crops_meta, log_probs, classes, fps=video.fps)
    out["merge"] = ids
    log["merge"] = dict(merges=int(n), temperature=temperature)
    ids, n = links.first_pass(video, ids, crops_meta, embedding, features, first)
    out["links"] = ids
    log["links"] = dict(links_between_identities=int(n))
    proposal, _ = links.second_pass(video, ids, crops_meta, embedding, features, second)
    if not np.array_equal(ids, proposal):
        ids, done = exchanges.confirmed_by_second_pass(
            ids.astype(np.int64),
            proposal.astype(np.int64),
            frame,
            box,
            crop_row,
            E1,
            E2,
            fps=video.fps,
        )
        log["exchanges"] = done
    out["exchanges"] = ids
    if video.backward is not None:
        ids, done = exchanges.proposed_by_backward_pass(
            ids.astype(np.int64), video.backward, frame, box, crop_row, E1, E2, fps=video.fps
        )
        log["exchanges_backward"] = done
    out["exchanges_backward"] = ids
    ids, done = exchanges.merge_two_models(
        ids.astype(np.int64),
        frame,
        video.base[:, 1].astype(int),
        embedding,
        crop_row,
        log_probs,
        classes,
    )
    out["merge_two"] = ids
    log["merge_two"] = done
    ids = agreement.links(video, ids)
    out["agree_links"] = ids
    ids = agreement.boxes(video, ids)
    out["agree_boxes"] = ids
    ids, c = agreement.boxes_with_forward(video, ids)
    out["agree_forward"] = ids
    log["agree_forward"] = dict(c)
    ids, c = agreement.boxes_with_forward(video, ids, wide=True)
    out["agree_wide"] = ids
    log["agree_wide"] = dict(c)
    ids, c = unsupported.rename_unsupported(video, ids)
    out["unsupported"] = ids
    log["unsupported"] = dict(c)
    ids, c = take_over.take_over(video, ids)
    out["take_over"] = ids
    log["take_over"] = dict(c)
    base = video.base[:, 1].astype(np.int64)
    previous = base
    for k in STEPS:
        log.setdefault(k, {})
        entry = log[k] if isinstance(log[k], dict) else dict(operations=log[k])
        entry["rows_changed"] = int((out[k] != previous).sum())
        log[k] = entry
        previous = out[k]
    ordered = {k: log[k] for k in STEPS}
    log.clear()
    log.update(ordered)
    return out


def _stage(stages, kind, name, key):
    return np.load(os.path.join(stages, kind, name + ".npz"))[key]


def run(folder, stages, fps=20.0, keep_steps=False, threads=2):
    """Run the chain on one video folder and write its outputs there. Returns the identities after each step.
    threads: threads given to the numerical libraries; the chain is light, and more threads do not help."""
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=threads):
        return _run(folder, stages, fps, keep_steps)


def make_crops(folder, fps=20.0):
    """Write crops.npz and crops_short.npz, read by the appearance stages, if they are not there."""
    video = Video(folder, fps)
    f = os.path.join(folder, "crops.npz")
    if not os.path.exists(f):
        jpeg, meta = crops.track_crops(video)
        np.savez(f, jpeg=jpeg, meta=meta)
    short = np.load(os.path.join(folder, "short_tracks.npy"))
    fs = os.path.join(folder, "crops_short.npz")
    if len(short) and not os.path.exists(fs):
        jpeg, sm = crops.short_track_crops(video, short)
        np.savez(fs, jpeg=jpeg, meta=sm)


def _run(folder, stages, fps, keep_steps):
    video = Video(folder, fps)
    name = video.name
    f = os.path.join(folder, "crops.npz")
    if not os.path.exists(f):
        jpeg, meta = crops.track_crops(video)
        np.savez(f, jpeg=jpeg, meta=meta)
    meta = np.load(f, allow_pickle=True)["meta"]
    log = {}
    out = identities(
        video,
        meta,
        _stage(stages, "classifier", name, "log_probs"),
        _stage(stages, "classifier", name, "classes"),
        _stage(stages, "embedding", name, "embedding"),
        _stage(stages, "features", name, "features"),
        log,
    )
    short = np.load(os.path.join(folder, "short_tracks.npy"))
    added = np.zeros((0, 7))
    fs = os.path.join(folder, "crops_short.npz")
    if len(short):
        if not os.path.exists(fs):
            jpeg, sm = crops.short_track_crops(video, short)
            np.savez(fs, jpeg=jpeg, meta=sm)
        sm = np.load(fs, allow_pickle=True)["meta"]
        if len(sm):
            added, c = short_tracks.attach(
                video,
                out["take_over"],
                short,
                meta,
                _stage(stages, "features", name, "features"),
                sm,
                _stage(stages, "features_short", name, "features"),
            )
            log["short_tracks"] = dict(c)
    np.save(os.path.join(folder, "identities.npy"), out["take_over"].astype(np.int64))
    np.save(os.path.join(folder, "attached.npy"), added)
    json.dump(log, open(os.path.join(folder, "operations.json"), "w"), indent=1)
    if keep_steps:
        np.savez(
            os.path.join(folder, "identity_steps.npz"),
            **{k: v.astype(np.int64) for k, v in out.items()},
        )
    return out, added
