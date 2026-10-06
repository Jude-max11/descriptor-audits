"""E4: population heterogeneity and correlated-attribute leakage.

(1) Effective cardinality: empirical H(theta) vs log M on real data.
(2) Per-group exposure: attribute advantage conditional on the released
    group label, reported as 10/50/90 percentiles ACROSS USERS
    (Gil et al. C&S'25 presentation convention).
(3) Correlated attribute: I(theta; A) is nonzero on real data; measure how
    the grouping descriptor's indirect leakage I(A; Z) grows with the
    information budget and compare against the data-processing ceiling
    I(A; Z) <= I(theta; A).
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import (load_dataset, attrs_of, build_user_side, cluster_types,
                      balanced_group_partition, empirical_mi)

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
RES = os.path.join(os.path.dirname(__file__), "..", "results")


def group_advantage_percentiles(theta_g, y, groups):
    """Per-user posterior advantage: for each user's released group k,
    G_k = max_a P(a|Z=k) - max_a P(a).  Return percentiles across users."""
    prior_best = np.bincount(y).max() / len(y)
    adv_by_group = {}
    for k in np.unique(groups):
        m = groups == k
        if m.sum() < 5:
            continue
        adv_by_group[k] = np.bincount(y[m], minlength=2).max() / m.sum() \
            - prior_best
    per_user = np.array([adv_by_group.get(g, 0.0) for g in groups])
    return {p: float(np.percentile(per_user, p)) for p in (10, 50, 90)}, \
        {k: float(v) for k, v in adv_by_group.items()}


def main(dataset_name="ml-1m", M=128, master_seed=7):
    ds = load_dataset(dataset_name)
    emb, svd, idx_i, top = build_user_side(ds, top_items=500, svd_dim=32,
                                           seed=master_seed)
    theta, centers = cluster_types(emb, M, seed=master_seed)
    yattrs = {a: ds.users[a].values for a in attrs_of(ds)}

    # (1) effective cardinality
    counts = np.bincount(theta, minlength=M)
    p = counts / counts.sum()
    H = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    eff_card = float(np.exp(H))

    # (2)+(3) sweep K
    rows = []
    for K in [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 128]:
        user_g, _, gc = balanced_group_partition(theta, centers, K, master_seed)
        row = {"K": K, "alpha": float(np.log(K)),
               "I_thetaZ": empirical_mi(theta, user_g)}
        for attr, y in yattrs.items():
            pct, by_group = group_advantage_percentiles(theta, y, user_g)
            row[f"{attr}_adv_p10"], row[f"{attr}_adv_p50"], row[f"{attr}_adv_p90"] = \
                pct[10], pct[50], pct[90]
            row[f"{attr}_I_AZ"] = empirical_mi(user_g, y)
            row[f"{attr}_I_thetaA"] = empirical_mi(theta, y)
        rows.append(row)
        print(row, flush=True)

    out = {"H_theta": H, "logM": float(np.log(M)),
           "effective_cardinality": eff_card, "M": M}
    os.makedirs(RES, exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{RES}/e4_heterogeneity_{dataset_name}.csv",
                              index=False)
    with open(f"{RES}/e4_meta_{dataset_name}.json", "w") as f:
        import json; json.dump(out, f, indent=2)
    print("saved E4 results", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ml-1m",
         int(sys.argv[2]) if len(sys.argv) > 2 else 128)
