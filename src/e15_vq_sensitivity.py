"""E15: vector-quantization resolution sensitivity (teacher review R3-m3).

The worst-case metrics for continuous/binary-vector releases (Gaussian, PCA,
bRAPPOR) are evaluated on an 89-bin VQ of the released vector, which is a
lower bound on the continuous-channel leakage by the data-processing
inequality.  This script recomputes ML, IL_inf, and the permutation-corrected
conditional ML at VQ resolutions {32, 64, 89, 128, 256} bins for the matched
operating points, so the supplement can report the sensitivity of the
conclusions to the VQ resolution.

Output: results/e15_vq_{ds}.csv
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import (load_dataset, attrs_of, build_user_side, cluster_types,
                      mechanism_grr, mechanism_brappor, mechanism_gaussian,
                      mechanism_pca)
from e8_worstcase import (max_leakage, min_entropy_leakage, ml_conditional,
                          il_inf_conditional, onehot_argmax)

RES = os.path.join(os.path.dirname(__file__), "..", "results")
BINS = (32, 64, 89, 128, 256)


def vq_at(Z, n_bins, seed=0):
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_bins, n_init=4, random_state=seed).fit(Z)
    return km.labels_.astype(int)


def main(dataset_name="ml-1m", M=128, master_seed=7):
    ds = load_dataset(dataset_name)
    emb, svd, idx_i, top = build_user_side(ds, top_items=500, svd_dim=32,
                                           seed=master_seed)
    theta, centers = cluster_types(emb, M, seed=master_seed)
    yattrs = {a: ds.users[a].values for a in attrs_of(ds)}
    scale = float(np.std(emb, axis=0).mean())

    mech = {
        ("bRAPPOR", "eps=16"): mechanism_brappor(theta, M, 16, master_seed),
        ("Gaussian", "sigma=0.37s"): mechanism_gaussian(
            emb, centers, 0.37 * scale, master_seed),
        ("PCA", "d'=7"): mechanism_pca(emb, centers, 7, master_seed),
    }

    rows = []
    for (name, param), out in mech.items():
        Z = np.asarray(out[2])
        for bins in BINS:
            zv = vq_at(Z, bins)
            for attr, y in yattrs.items():
                r = {"mechanism": name, "param": param, "bins": bins,
                     "attr": attr, "alphabet": int(zv.max()) + 1,
                     "ML": max_leakage(zv, y),
                     "IL_inf": min_entropy_leakage(zv, y)}
                r.update(ml_conditional(zv, y, theta))
                r["IL_inf_cond"] = il_inf_conditional(zv, y, theta)
                rows.append(r)
                print(f"{name} bins={bins} {attr}: "
                      f"ML={r['ML']:.3f} "
                      f"MLc={r['ML_cond_corr_mean_nonneg']:.4f} "
                      f"n2s={r['ML_cond_n_over2']} "
                      f"fdr={r['ML_cond_n_fdr05']}", flush=True)

    os.makedirs(RES, exist_ok=True)
    path = f"{RES}/e15_vq_{dataset_name}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print("saved", path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ml-1m",
         int(sys.argv[2]) if len(sys.argv) > 2 else 128)
