"""Names of the columns in the stored files of the replay data, and their names in the package."""

import numpy as np

NAMES = dict(
    vues="views", vues_iou="views_iou", vues_score="views_score", masques="masks", masques_iou="masks_iou",
    cache="hidden", ligne="row", boite_maillage="mesh_box", focale="focal", bande="band", forme="shape",
    rot_globale="global_rotation",
    idx="rows", emb="embedding", norme0="norm_single", norme="norm", yeux="eye_distance", residu="residual",
    echelle="scale", score5="point_scores", n_lignes="row_count", norme_propre="own_norm",
    cos_miroir="mirror_cosine", partage="shared", autre="other", traits="features",
    logp="log_probs", noms="names",
)  # fmt: skip


# columns of the stored pair tables: name in the package -> stored name
PAIR_COLUMNS = dict(
    same="meme", passes_agree="n_accord", vote="v_vote", unanimous_a="una_a", unanimous_b="una_b", gap="ecart",
    motion="err_v", motion_back="err_vb", extended_a="prol_a", extended_b="prol_b", cos="cos", cos2="cos2",
    gap_unanimous="una_gap", gap_overflow="dep_gap", gap_visible="vis_gap", gap_extended="prol_gap",
    bytetrack="v_bt", backward="v_inv",
)  # fmt: skip


def load(path):
    """Arrays of a stored file, under the names of the package."""
    z = np.load(path, allow_pickle=True)
    return {NAMES.get(k, k): z[k] for k in z.files}
