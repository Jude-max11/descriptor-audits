"""E6: ablation suite (RAID-style robustness checks).

A. Attacker capacity: MLP with 1-4 hidden layers (last layer = ours, 100
   units each).  Purpose: show the reported attack is not underfitting.
B. Shadow-set size: attacker trained on 20/40/60/80% of the labeled shadow
   pool, evaluated on a fixed held-out split.
C. Task-family granularity: M in {64, 128, 256} on ML-1M; grouping sweep
   re-run per M (utility + leakage + attack).
D. Grouping strategy: bisecting-2-means tree (default) vs consecutive
   grouping (order types along the first PC of centroids, cut into K
   contiguous blocks) -- the manuscript's equally-spaced analogue.

Outputs: results/e6_ablA_depth_{ds}.csv, e6_ablB_shadow_{ds}.csv,
e6_ablC_granularity_{ds}.csv, e6_ablD_grouping_{ds}.csv
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import (load_movielens, build_user_side, cluster_types,
                      balanced_group_partition, empirical_mi, make_attacker,
                      run_attack)
from e1e3_sweep import RatingUtility

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
RES = os.path.join(os.path.dirname(__file__), "..", "results")


def mlp_adv(Z, y, depths=(1, 2, 3, 4), seeds=tuple(range(10)), folds=5):
    """advantage of MLP with `depths` hidden layers (100 units each)."""
    out = []
    for d in depths:
        rec = {"depth": d}
        advs, aucs = [], []
        for s in seeds:
            skf = StratifiedKFold(folds, shuffle=True, random_state=s)
            for tr, te in skf.split(Z, y):
                clf = MLPClassifier(
                    hidden_layer_sizes=(100,) * d, alpha=1.0,
                    learning_rate_init=1e-3, max_iter=500)
                clf.fit(Z[tr], y[tr])
                pred = clf.predict(Z[te])
                classes = np.unique(y[tr])
                per = [np.mean(pred[y[te] == c] == c)
                       if (y[te] == c).any() else 0.0 for c in classes]
                advs.append(np.mean(per) - 1.0 / len(classes))
                if len(classes) == 2:
                    try:
                        aucs.append(roc_auc_score(
                            y[te], clf.predict_proba(Z[te])[:, 1]))
                    except Exception:
                        pass
        rec["adv_mean"] = float(np.mean(advs))
        rec["adv_std"] = float(np.std(advs))
        rec["auc_mean"] = float(np.mean(aucs)) if aucs else np.nan
        out.append(rec)
    return out


def shadow_frac_adv(Z, y, fracs=(0.2, 0.4, 0.6, 0.8), seeds=tuple(range(10)),
                    kinds=("mlp", "lr")):
    """Train on a fraction of a fixed shadow pool, test on a fixed eval set."""
    out = []
    idx = np.arange(len(y))
    # fixed 50% eval pool disjoint from the shadow pool
    tr_pool, ev = train_test_split(idx, test_size=0.5, stratify=y,
                                   random_state=99)
    for frac in fracs:
        advs = []
        for s in seeds:
            pos = np.random.default_rng(s).choice(
                len(tr_pool), int(frac * len(tr_pool)), replace=False)
            tr = tr_pool[pos]
            best = -1.0
            for kind in kinds:
                clf = make_attacker(kind)
                try:
                    clf.fit(Z[tr], y[tr])
                    pred = clf.predict(Z[ev])
                    classes = np.unique(y[tr])
                    per = [np.mean(pred[y[ev] == c] == c)
                           if (y[ev] == c).any() else 0.0 for c in classes]
                    best = max(best, float(np.mean(per)) - 1.0 / len(classes))
                except Exception:
                    pass
            advs.append(best)
        out.append({"frac": frac, "adv_mean": float(np.mean(advs)),
                    "adv_std": float(np.std(advs)),
                    "n_shadow": int(frac * len(tr_pool))})
    return out


def consecutive_partition(theta, centers, K):
    """Order types along PC1 of the centroids, cut into K contiguous blocks
    (the equally-spaced-family analogue of the manuscript)."""
    M = len(centers)
    if K >= M:
        t2g = np.arange(M)
        return t2g[theta], t2g, {m: centers[m] for m in range(M)}
    pc = PCA(n_components=1).fit(centers)
    order = np.argsort(pc.transform(centers).ravel())
    t2g = np.zeros(M, dtype=int)
    blocks = np.array_split(order, K)
    for k, b in enumerate(blocks):
        t2g[b] = k
    gc = {k: centers[b].mean(0) for k, b in enumerate(blocks)}
    return t2g[theta], t2g, gc


def main(dataset_name="ml-1m", M=128, master_seed=7):
    ds = load_movielens(os.path.join(DATA, dataset_name), dataset_name)
    emb, svd, idx_i, top = build_user_side(ds, top_items=500, svd_dim=32,
                                           seed=master_seed)
    theta, centers = cluster_types(emb, M, seed=master_seed)
    util = RatingUtility(ds, idx_i, top, test_frac=0.2, seed=master_seed)
    yattrs = {a: ds.users[a].values for a in ("gender", "age3")}
    mse_oracle = util.mse(*util.models_from_decode(theta))
    os.makedirs(RES, exist_ok=True)

    # ---- A. attacker depth (on a representative grouping descriptor) ----
    rowsA = []
    for K in (21, 89):
        user_g, _, _ = balanced_group_partition(theta, centers, K, master_seed)
        Z = np.eye(K)[user_g]
        for attr, y in yattrs.items():
            for r in mlp_adv(Z, y):
                rowsA.append({"K": K, "attr": attr, **r})
    pd.DataFrame(rowsA).to_csv(f"{RES}/e6_ablA_depth_{dataset_name}.csv",
                               index=False)
    print("A done", flush=True)

    # ---- B. shadow-set size ----
    rowsB = []
    for K in (21, 89):
        user_g, _, _ = balanced_group_partition(theta, centers, K, master_seed)
        Z = np.eye(K)[user_g]
        for attr, y in yattrs.items():
            for r in shadow_frac_adv(Z, y):
                rowsB.append({"K": K, "attr": attr, **r})
    pd.DataFrame(rowsB).to_csv(f"{RES}/e6_ablB_shadow_{dataset_name}.csv",
                               index=False)
    print("B done", flush=True)

    # ---- C. task-family granularity ----
    rowsC = []
    for M2 in (64, 128, 256):
        theta2, centers2 = cluster_types(emb, M2, seed=master_seed)
        util2 = RatingUtility(ds, idx_i, top, test_frac=0.2,
                              seed=master_seed)
        mse_or2 = util2.mse(*util2.models_from_decode(theta2))
        for K in sorted({max(1, M2 // 32), M2 // 8, M2 // 4, M2 // 2, M2}):
            user_g, _, _ = balanced_group_partition(theta2, centers2, K,
                                                    master_seed)
            mus, cv = util2.models_from_decode(user_g)
            excess = util2.mse(mus, cv) - mse_or2
            Z = np.eye(K)[user_g]
            r = {"M": M2, "K": K, "alpha_emp": empirical_mi(theta2, user_g),
                 "excess": excess}
            for attr, y in yattrs.items():
                best = -1.0
                for kind in ("mlp", "lr"):
                    a = run_attack(Z, y, seeds=(0, 1, 2), folds=5,
                                   attackers=(kind,))
                    d = a.get("adv", (np.nan,))[0]
                    if np.isfinite(d):
                        best = max(best, d)
                r[f"{attr}_adv"] = best
            rowsC.append(r)
            print("C:", r, flush=True)
    pd.DataFrame(rowsC).to_csv(f"{RES}/e6_ablC_granularity_{dataset_name}.csv",
                               index=False)

    # ---- D. grouping strategy ----
    rowsD = []
    for K in (5, 13, 21, 34, 55, 89):
        for strat in ("bisecting", "consecutive"):
            if strat == "bisecting":
                user_g, _, _ = balanced_group_partition(theta, centers, K,
                                                        master_seed)
            else:
                user_g, _, _ = consecutive_partition(theta, centers, K)
            mus, cv = util.models_from_decode(user_g)
            excess = util.mse(mus, cv) - mse_oracle
            Z = np.eye(K)[user_g]
            r = {"K": K, "strategy": strat,
                 "alpha_emp": empirical_mi(theta, user_g), "excess": excess}
            for attr, y in yattrs.items():
                best = -1.0
                for kind in ("mlp", "lr"):
                    a = run_attack(Z, y, seeds=(0, 1, 2), folds=5,
                                   attackers=(kind,))
                    d = a.get("adv", (np.nan,))[0]
                    if np.isfinite(d):
                        best = max(best, d)
                r[f"{attr}_adv"] = best
                r[f"{attr}_I_AZ"] = empirical_mi(user_g, y)
            rowsD.append(r)
            print("D:", r, flush=True)
    pd.DataFrame(rowsD).to_csv(f"{RES}/e6_ablD_grouping_{dataset_name}.csv",
                               index=False)
    print("E6 done")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ml-1m",
         int(sys.argv[2]) if len(sys.argv) > 2 else 128)
