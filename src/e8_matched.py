"""E8 at the true matched-utility parameters of the manuscript's Table II.

The original e8_worstcase.py evaluated the worst-case metrics at the sweep
grid points (grr eps=12, brappor eps=12, pca d'=8), which do not coincide
with the matched-utility operating points of Table II (grr eps~10
interpolated, brappor eps=16, pca d'~7 interpolated).  This rerun evaluates
the identical metrics at the Table II parameters so Tables II and III share
one operating point per mechanism.  Protocol, estimators, and seeds are
unchanged; only the parameter dict and the output filename differ.

Output: results/e8_worstcase_<ds>_matched.csv
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import (load_dataset, attrs_of, build_user_side, cluster_types,
                      mechanism_grouping, mechanism_grr, mechanism_brappor,
                      mechanism_gaussian, mechanism_pca)
from e8_worstcase import (RES, max_leakage, min_entropy_leakage,
                          ml_conditional, bootstrap_ci, il_inf_conditional,
                          advantage_percentiles, vq_labels, onehot_argmax)


def main(dataset_name="ml-1m", M=128, master_seed=7):
    ds = load_dataset(dataset_name)
    emb, svd, idx_i, top = build_user_side(ds, top_items=500, svd_dim=32,
                                           seed=master_seed)
    theta, centers = cluster_types(emb, M, seed=master_seed)
    yattrs = {a: ds.users[a].values for a in attrs_of(ds)}
    scale = float(np.std(emb, axis=0).mean())

    # matched-utility operating points of Table II (manuscript)
    mech = {
        ("grouping", "K=89"): mechanism_grouping(theta, centers, 89,
                                                 master_seed),
        ("grr", "eps=10"): mechanism_grr(theta, M, 10, master_seed),
        ("brappor", "eps=16"): mechanism_brappor(theta, M, 16, master_seed),
        ("gaussian", "sigma=0.37s"): mechanism_gaussian(
            emb, centers, 0.37 * scale, master_seed),
        ("pca", "d'=7"): mechanism_pca(emb, centers, 7, master_seed),
    }

    rows = []
    for (name, param), out in mech.items():
        decode = out[0]
        Z = out[2]
        if name in ("grouping", "grr"):
            z_decode = onehot_argmax(Z)
        else:
            z_decode = decode.astype(int)
        z_vq = vq_labels(np.asarray(Z)) if name in ("brappor", "gaussian",
                                                    "pca") else None
        for attr, y in yattrs.items():
            for view, zv in (("decode", z_decode), ("vq", z_vq)):
                if zv is None:
                    continue
                r = {"mechanism": name, "param": param, "attr": attr,
                     "view": view, "alphabet": int(zv.max()) + 1,
                     "ML": max_leakage(zv, y),
                     "IL_inf": min_entropy_leakage(zv, y)}
                r["ML_ci"] = bootstrap_ci(max_leakage, zv, y)
                r["IL_inf_ci"] = bootstrap_ci(min_entropy_leakage, zv, y)
                r.update(ml_conditional(zv, y, theta))
                r["IL_inf_cond"] = il_inf_conditional(zv, y, theta)
                pct, worst = advantage_percentiles(zv, y)
                r["adv_p10"], r["adv_p50"], r["adv_p90"] = \
                    pct[10], pct[50], pct[90]
                r["adv_worst_user"] = worst
                rows.append(r)
                print({k: (round(v, 4) if isinstance(v, float) else v)
                       for k, v in r.items()}, flush=True)

    os.makedirs(RES, exist_ok=True)
    path = f"{RES}/e8_worstcase_{dataset_name}_matched.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print("saved", path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ml-1m",
         int(sys.argv[2]) if len(sys.argv) > 2 else 128)
