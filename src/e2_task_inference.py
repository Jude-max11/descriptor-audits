"""E2: task identification attack, strong vs weak adversary + finite-sample gap.

Strong adversary: server retains the descriptor computed from the client's
training-period data (half A).  Weak adversary: only sees a fresh descriptor
from half B (new local data).  The gap-vs-n curve sweeps the fraction n of
local ratings used to build the descriptor: theory predicts the effective
budget beta_n = min(alpha, H(theta), 1/2 log(1 + n V/sigma^2)), i.e.
P_theta improves with n up to the rate limit and then saturates.
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


def embed_from_ratings(ratings, uids, idx_i, idx_u, svd_components):
    """Project a (possibly partial) rating set onto the public SVD basis."""
    X = np.zeros((len(uids), len(idx_i)))
    sub = ratings[ratings["iid"].isin(idx_i)]
    rows = sub["uid"].map(idx_u).values
    cols = sub["iid"].map(idx_i).values
    X[rows, cols] = sub["rating"].values
    nz = X > 0
    cnt = nz.sum(1, keepdims=True); cnt[cnt == 0] = 1
    mu = X.sum(1, keepdims=True) / cnt
    Xc = np.where(nz, X - mu, 0.0)
    return Xc @ svd_components.T


def nn_decode(E, gcenters):
    d2 = ((E[:, None, :] - gcenters[None, :, :]) ** 2).sum(-1)
    return np.argmin(d2, axis=1)


def main(dataset_name="ml-1m", M=128, master_seed=7):
    ds = load_dataset(dataset_name)
    emb_full, svd, idx_i, top = build_user_side(ds, top_items=500, svd_dim=32,
                                                seed=master_seed)
    theta, centers = cluster_types(emb_full, M, seed=master_seed)
    idx_u = {u: i for i, u in enumerate(ds.uids)}

    r = ds.ratings[ds.ratings["iid"].isin(idx_i)].sample(
        frac=1.0, random_state=master_seed).copy()
    r["rank"] = r.groupby("uid").cumcount()
    r["sz"] = r.groupby("uid")["uid"].transform("size")
    r["A"] = r["rank"] < (0.5 * r["sz"]).astype(int)

    rows = []
    for K in [2, 5, 13, 34, 128]:
        _, t2g, gc = balanced_group_partition(theta, centers, K, master_seed)
        theta_g = t2g[theta]                               # per-user target
        gcenters = np.stack([gc[k] for k in range(len(gc))])
        embA = embed_from_ratings(r[r["A"]], ds.uids, idx_i, idx_u,
                                  svd.components_)
        embB = embed_from_ratings(r[~r["A"]], ds.uids, idx_i, idx_u,
                                  svd.components_)
        decA, decB = nn_decode(embA, gcenters), nn_decode(embB, gcenters)
        rows.append({"K": K, "alpha": float(np.log(K)),
                     "P_theta_strong": float(np.mean(decA != theta_g)),
                     "P_theta_weak": float(np.mean(decB != theta_g)),
                     "MI_A": empirical_mi(theta_g, decA),
                     "MI_B": empirical_mi(theta_g, decB)})
        print(rows[-1], flush=True)

    n_rows = []
    for K in [5, 13, 34]:
        _, t2g, gc = balanced_group_partition(theta, centers, K, master_seed)
        theta_g = t2g[theta]
        gcenters = np.stack([gc[k] for k in range(len(gc))])
        for frac in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
            m = r["rank"] < (frac * r["sz"]).astype(int)
            E = embed_from_ratings(r[m], ds.uids, idx_i, idx_u, svd.components_)
            dec = nn_decode(E, gcenters)
            n_rows.append({"K": K, "frac": frac,
                           "P_theta": float(np.mean(dec != theta_g)),
                           "MI": empirical_mi(theta_g, dec)})
        print(n_rows[-1], flush=True)

    os.makedirs(RES, exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{RES}/e2_task_inference_{dataset_name}.csv",
                              index=False)
    pd.DataFrame(n_rows).to_csv(
        f"{RES}/e2_gap_vs_n_{dataset_name}.csv", index=False)
    print("saved E2 results")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ml-1m",
         int(sys.argv[2]) if len(sys.argv) > 2 else 128)
