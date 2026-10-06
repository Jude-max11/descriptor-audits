"""E10: attribute leakage of oracle vs learned grouping descriptors.

Review round 2, Point 5: the zero residual leakage of the grouping
descriptor holds by construction for the ORACLE implementation (group label
= deterministic function of the training-period cluster label theta).  A
LEARNED variant estimates the type from fresh local data (half B of each
user's interactions, projected on the public SVD basis, nearest group
center) and may inherit attribute-correlated estimation errors, so
P(Z | theta, A) may differ from P(Z | theta).

This experiment quantifies, per group count K:
  P_group      task-group error of the learned descriptor (fresh half B)
  Lres_oracle  I(A; g(theta) | theta)              -- zero by construction
  Lres_learned I(A; g(theta_hat_B) | theta)        -- permutation-corrected
               per type (mean over types, truncated at zero)
  adv_oracle / adv_learned   trained-attack advantage (best MLP/LR)
  IAZ_oracle / IAZ_learned   total I(A; Z)

Output: results/e10_learned_grouping_{ds}.csv
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import (load_dataset, attrs_of, build_user_side, cluster_types,
                      balanced_group_partition, empirical_mi, run_attack,
                      _plugin_mi)
from e2_task_inference import embed_from_ratings, nn_decode

RES = os.path.join(os.path.dirname(__file__), "..", "results")
MIN_TYPE = 10
N_PERM = 50


def lres_conditional(z, a, theta, seed=0, n_perm=N_PERM):
    """Permutation-corrected I(A;Z|theta): prior-weighted per-type MI with
    within-type attribute-permutation null subtracted."""
    def value(z, a):
        tot, cnt = 0.0, 0
        for t in np.unique(theta):
            m = theta == t
            if m.sum() < MIN_TYPE or len(np.unique(a[m])) < 2:
                continue
            tot += _plugin_mi(z[m], a[m]) * m.sum()
            cnt += m.sum()
        return tot / max(cnt, 1)

    rng = np.random.default_rng(seed)
    raw = value(np.asarray(z), np.asarray(a))
    nulls = []
    for _ in range(n_perm):
        a2 = np.array(a)
        for t in np.unique(theta):
            m = theta == t
            a2[m] = rng.permutation(np.asarray(a)[m])
        nulls.append(value(np.asarray(z), a2))
    return max(raw - float(np.mean(nulls)), 0.0), raw


def best_adv(Z, y):
    best = -1.0
    for kind in ("mlp", "lr"):
        a = run_attack(Z, y, seeds=tuple(range(10)), folds=5,
                       attackers=(kind,))
        d = a.get("adv", (float("nan"),))[0]
        if np.isfinite(d) and d > best:
            best = d
    return best


def main(dataset_name="ml-1m", M=128, master_seed=7):
    ds = load_dataset(dataset_name)
    emb, svd, idx_i, top = build_user_side(ds, top_items=500, svd_dim=32,
                                           seed=master_seed)
    theta, centers = cluster_types(emb, M, seed=master_seed)
    idx_u = {u: i for i, u in enumerate(ds.uids)}
    yattrs = {a: ds.users[a].values for a in attrs_of(ds)}

    r = ds.ratings[ds.ratings["iid"].isin(idx_i)].sample(
        frac=1.0, random_state=master_seed).copy()
    r["rank"] = r.groupby("uid").cumcount()
    r["sz"] = r.groupby("uid")["uid"].transform("size")
    r["A"] = r["rank"] < (0.5 * r["sz"]).astype(int)

    rows = []
    for K in (13, 34, 89):
        _, t2g, gc = balanced_group_partition(theta, centers, K, master_seed)
        theta_g = t2g[theta]
        gcenters = np.stack([gc[k] for k in range(len(gc))])
        embB = embed_from_ratings(r[~r["A"]], ds.uids, idx_i, idx_u,
                                  svd.components_)
        z_learned = nn_decode(embB, gcenters)
        row = {"K": K,
               "P_group_learned": float(np.mean(z_learned != theta_g))}
        for attr, y in yattrs.items():
            lr_o_raw = 0.0                      # oracle: Z=f(theta)
            lr_l, lr_l_raw = lres_conditional(z_learned, y, theta)
            row[f"{attr}_Lres_oracle"] = lr_o_raw
            row[f"{attr}_Lres_learned"] = lr_l
            row[f"{attr}_Lres_learned_raw"] = lr_l_raw
            row[f"{attr}_adv_oracle"] = best_adv(np.eye(K)[theta_g], y)
            row[f"{attr}_adv_learned"] = best_adv(np.eye(K)[z_learned], y)
            row[f"{attr}_IAZ_oracle"] = empirical_mi(theta_g, y)
            row[f"{attr}_IAZ_learned"] = empirical_mi(z_learned, y)
        rows.append(row)
        print(row, flush=True)

    os.makedirs(RES, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        f"{RES}/e10_learned_grouping_{dataset_name}.csv", index=False)
    print("saved e10")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ml-1m",
         int(sys.argv[2]) if len(sys.argv) > 2 else 128)
