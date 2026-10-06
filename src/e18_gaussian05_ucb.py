"""E18: permutation UCB for Gaussian L_res / ML_cond at the matched grid
point sigma = 0.5*scale (e14 Part B protocol, 100 permutations).

Output: results/e18_gaussian05_ucb.csv
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import (load_dataset, attrs_of, build_user_side, cluster_types,
                      mechanism_gaussian, empirical_mi)
from e8_worstcase import max_leakage, MIN_TYPE

RES = os.path.join(os.path.dirname(__file__), "..", "results")


def stat_l_res(z, y, theta):
    tot, cnt = 0.0, 0
    for t in np.unique(theta):
        m = theta == t
        if m.sum() >= 5 and len(np.unique(y[m])) > 1:
            tot += empirical_mi(z[m], y[m], n_perm=0) * m.sum()
            cnt += m.sum()
    return tot / max(cnt, 1)


def stat_ml_cond(z, y, theta):
    raws, nulls = [], []
    rng = np.random.default_rng(0)
    for t in np.unique(theta):
        m = theta == t
        if m.sum() < MIN_TYPE or len(np.unique(y[m])) < 2:
            continue
        zt, at = z[m], np.asarray(y)[m]
        null = [max_leakage(zt, rng.permutation(at)) for _ in range(20)]
        raws.append(max_leakage(zt, at))
        nulls.append(float(np.mean(null)))
    return max(float(np.mean(np.array(raws) - np.array(nulls))), 0.0)


def ucb(fn, z, y, theta, n_perm=100, seed=0):
    rng = np.random.default_rng(seed)
    obs = fn(z, y, theta)
    null = []
    for _ in range(n_perm):
        yv = np.array(y)
        for t in np.unique(theta):
            m = theta == t
            yv[m] = rng.permutation(np.asarray(y)[m])
        null.append(fn(z, yv, theta))
    null = np.array(null)
    bias = null.mean()
    return obs, float(max(obs - bias, 0.0)
                      + np.percentile(null - bias, 95))


def main(dataset_name="ml-1m", M=128, master_seed=7):
    ds = load_dataset(dataset_name)
    emb, svd, idx_i, top = build_user_side(ds, top_items=500, svd_dim=32,
                                           seed=master_seed)
    theta, centers = cluster_types(emb, M, seed=master_seed)
    yattrs = {a: ds.users[a].values for a in attrs_of(ds)}
    scale = float(np.std(emb, axis=0).mean())
    decode, _, Z, _ = mechanism_gaussian(emb, centers, 0.5 * scale, master_seed)
    zv = decode.astype(int)
    rows = []
    for attr, y in yattrs.items():
        obs, u = ucb(stat_l_res, zv, y, theta)
        rows.append({"mechanism": "gaussian 0.5s", "attr": attr,
                     "stat": "L_res", "obs": obs, "ucb95": u})
        print(rows[-1], flush=True)
        obs, u = ucb(stat_ml_cond, zv, y, theta)
        rows.append({"mechanism": "gaussian 0.5s", "attr": attr,
                     "stat": "ML_cond", "obs": obs, "ucb95": u})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(f"{RES}/e18_gaussian05_ucb.csv", index=False)


if __name__ == "__main__":
    main()
