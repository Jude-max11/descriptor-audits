"""E6-A (corrected): attacker-depth ablation with class-balanced GPU MLP.

The first pass showed plain MLPs collapse to the majority class at every
depth (adv ~ 0), conflating capacity with the class-imbalance failure.  This
rerun uses balanced cross-entropy so depth measures capacity; depths 1-6.
Also runs the balanced MLP on the grouping K-sweep of both ML datasets to
check whether the headline 'strongest attacker' numbers move.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import load_dataset, attrs_of, build_user_side, cluster_types, \
    balanced_group_partition, run_attack
from gpu_attacker import TorchMLP, balanced_mlp_adv

RES = os.path.join(os.path.dirname(__file__), "..", "results")


def main(dataset_name="ml-1m", M=128, master_seed=7):
    ds = load_dataset(dataset_name)
    emb, svd, idx_i, top = build_user_side(ds, top_items=500, svd_dim=32,
                                           seed=master_seed)
    theta, centers = cluster_types(emb, M, seed=master_seed)
    yattrs = {a: ds.users[a].values for a in attrs_of(ds)}
    seeds = tuple(range(5))

    rows = []
    for K in (21, 89):
        user_g, _, _ = balanced_group_partition(theta, centers, K, master_seed)
        Z = np.eye(K)[user_g].astype(np.float32)
        for attr, y in yattrs.items():
            for r in balanced_mlp_adv(Z, y, depths=(1, 2, 3, 4, 5, 6),
                                      seeds=seeds):
                rows.append({"K": K, "attr": attr, **r})
                print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(
        f"{RES}/e6_ablA_depth_balanced_{dataset_name}.csv", index=False)

    # balanced-MLP check across the grouping K-sweep
    rows2 = []
    for K in [5, 13, 21, 34, 55, 89, 128]:
        if K > M:
            continue
        user_g, _, _ = balanced_group_partition(theta, centers, K, master_seed)
        Z = np.eye(K)[user_g].astype(np.float32)
        r = {"K": K}
        for attr, y in yattrs.items():
            best = -1.0
            for depth in (2, 3):
                advs = []
                from sklearn.model_selection import StratifiedKFold
                for s in seeds[:5]:
                    skf = StratifiedKFold(5, shuffle=True, random_state=s)
                    for tr, te in skf.split(Z, y):
                        clf = TorchMLP(depth=depth, seed=s).fit(Z[tr], y[tr])
                        pred = clf.predict(Z[te])
                        classes = np.unique(y[tr])
                        per = [np.mean(pred[y[te] == c] == c)
                               if (y[te] == c).any() else 0.0
                               for c in classes]
                        advs.append(np.mean(per) - 1.0 / len(classes))
                best = max(best, float(np.mean(advs)))
            r[f"{attr}_adv_balmlp"] = best
        rows2.append(r)
        print(rows2[-1], flush=True)
    pd.DataFrame(rows2).to_csv(
        f"{RES}/e6_balanced_mlp_sweep_{dataset_name}.csv", index=False)
    print("done")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ml-1m",
         int(sys.argv[2]) if len(sys.argv) > 2 else 128)
