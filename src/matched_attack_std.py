"""Per-seed AUC dispersion at the matched-utility operating points.

Table II reports advantage mean+-std but a bare AUC mean, because the sweep
CSV stored only the AUC mean.  run_attack() records AUC per seed/fold
internally; this script reruns the identical protocol (stratified 5-fold
CV x 10 seeds; MLP(100) and class-balanced LR, best-of) at the five matched
parameters of Table II and saves mean+-std for BOTH advantage and AUC.

Output: results/matched_attack_std_<ds>.csv
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import (load_dataset, attrs_of, build_user_side, cluster_types,
                      mechanism_grouping, mechanism_grr, mechanism_brappor,
                      mechanism_gaussian, mechanism_pca, run_attack)

RES = os.path.join(os.path.dirname(__file__), "..", "results")

SETUP = {}


def setup(dataset_name="ml-1m", M=128, master_seed=7):
    ds = load_dataset(dataset_name)
    emb, svd, idx_i, top = build_user_side(ds, top_items=500, svd_dim=32,
                                           seed=master_seed)
    theta, centers = cluster_types(emb, M, seed=master_seed)
    yattrs = {a: ds.users[a].values for a in attrs_of(ds)}
    scale = float(np.std(emb, axis=0).mean())
    mech = {
        ("BTQ", "K=89"): mechanism_grouping(theta, centers, 89, master_seed)[2],
        ("GRR", "eps=10"): mechanism_grr(theta, M, 10, master_seed)[2],
        ("bRAPPOR", "eps=16"): mechanism_brappor(theta, M, 16, master_seed)[2],
        ("Gaussian", "sigma=0.37s"): mechanism_gaussian(
            emb, centers, 0.37 * scale, master_seed)[2],
        ("PCA", "d'=7"): mechanism_pca(emb, centers, 7, master_seed)[2],
    }
    return mech, yattrs


def one(task):
    name, param, Z, attr, y, kind = task
    res = run_attack(np.asarray(Z), y, seeds=tuple(range(10)), folds=5,
                     attackers=(kind,))
    adv_m, adv_s = res["adv"]
    auc = res.get("auc", (float("nan"), float("nan")))
    return {"mechanism": name, "param": param, "attr": attr, "attacker": kind,
            "adv_mean": adv_m, "adv_std": adv_s,
            "auc_mean": auc[0], "auc_std": auc[1]}


def main(dataset_name="ml-1m", M=128, master_seed=7):
    mech, yattrs = setup(dataset_name, M, master_seed)
    tasks = [(n, p, Z, a, y, k)
             for (n, p), Z in mech.items()
             for a, y in yattrs.items()
             for k in ("mlp", "lr")]
    with ProcessPoolExecutor(max_workers=10) as ex:
        rows = list(ex.map(one, tasks))
    df = pd.DataFrame(rows)
    # best-of rule of the manuscript: per (mechanism, param, attr) keep the
    # attacker with the higher mean advantage
    best = (df.sort_values("adv_mean", ascending=False)
              .drop_duplicates(["mechanism", "param", "attr"]))
    os.makedirs(RES, exist_ok=True)
    df.to_csv(f"{RES}/matched_attack_std_{dataset_name}_all.csv", index=False)
    best.to_csv(f"{RES}/matched_attack_std_{dataset_name}.csv", index=False)
    print(best.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ml-1m",
         int(sys.argv[2]) if len(sys.argv) > 2 else 128)
