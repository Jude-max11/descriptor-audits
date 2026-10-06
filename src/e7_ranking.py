"""E7: ranking-utility evaluation (HR@10 / NDCG@10) under each descriptor.

Standard leave-one-out protocol (Koren et al. 2009): per user, hold out one
positive item (highest-rating latest), pair it with 99 uniformly sampled
unrated items; the served model scores the 100 candidates; report HR@10 and
NDCG@10 averaged over users.  Held-out interactions are EXCLUDED from the
training scores.  Served models: oracle (true type), grouping (K sweep),
GRR / bRAPPOR (eps sweep), Gaussian (sigma), PCA (d'), global population.

This answers the reviewer question "does the excess-MSE ordering survive a
recommender-style ranking metric?" without changing the threat model.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import (load_dataset, attrs_of, build_user_side, cluster_types,
                      balanced_group_partition, mechanism_grr,
                      mechanism_brappor, mechanism_gaussian, mechanism_pca,
                      empirical_mi)

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
N_NEG = 99
K_CUT = 10


def build_eval_set(ratings, idx_i, idx_u, seed=0):
    """Returns (eval_rows, held_keys). eval_rows: (user_pos, item_positions
    [1 positive + 99 negatives], labels). held_keys: {(uid, iid)} excluded
    from training."""
    rng = np.random.default_rng(seed)
    r = ratings[ratings["iid"].isin(idx_i)].copy()
    if "ts" in r.columns:
        r = r.sort_values(["uid", "ts"])
    else:                       # Book-Crossing ships no timestamps
        r = r.sort_values("uid", kind="stable")
    pos = r[r["rating"] >= 4].groupby("uid").tail(1)
    have = r.groupby("uid")["iid"].apply(lambda s: set(s))
    items = np.array(sorted(idx_i.keys()))      # actual item ids
    item_pos = idx_i                            # iid -> C column index
    eval_rows, held_keys = [], set()
    for _, row in pos.iterrows():
        u, iid = row["uid"], row["iid"]
        seen = have[u]
        cand_items, labels = [iid], [1]
        tries = 0
        while len(cand_items) < N_NEG + 1 and tries < 5000:
            cand = items[rng.integers(len(items))]
            tries += 1
            if cand in seen or cand in cand_items:
                continue
            cand_items.append(cand)
            labels.append(0)
        if len(cand_items) < N_NEG + 1:
            continue
        eval_rows.append((idx_u[u],
                          np.array([item_pos[c] for c in cand_items]),
                          np.array(labels)))
        held_keys.add((u, iid))
    return eval_rows, items, held_keys


def rank_metrics(eval_rows, score_fn):
    hr, ndcg = [], []
    for u, cands, labels in eval_rows:
        s = score_fn(u, cands)
        order = np.argsort(-s)
        lab = labels[order][:K_CUT]
        hr.append(float(lab[0]))
        dcg = float(np.sum(lab / np.log2(np.arange(2, len(lab) + 2))))
        ndcg.append(dcg)          # ideal DCG = 1 (single relevant item)
    return float(np.mean(hr)), float(np.mean(ndcg))


def main(dataset_name="ml-1m", M=128, master_seed=7):
    ds = load_dataset(dataset_name)
    emb, svd, idx_i, top = build_user_side(ds, top_items=500, svd_dim=32,
                                           seed=master_seed)
    theta, centers = cluster_types(emb, M, seed=master_seed)
    idx_u = {u: i for i, u in enumerate(ds.uids)}
    n_u = len(ds.uids)

    eval_rows, items, held_keys = build_eval_set(ds.ratings, idx_i, idx_u,
                                                 seed=master_seed)
    # training scores on all interactions EXCEPT the held-out positives
    r = ds.ratings[ds.ratings["iid"].isin(idx_i)]
    key = list(zip(r["uid"].values, r["iid"].values))
    mask = np.array([k not in held_keys for k in key])
    r = r[mask]
    print(f"eval users: {len(eval_rows)}, excluded held-out: {len(held_keys)}",
          flush=True)

    mu_u = np.full(n_u, r["rating"].mean())
    C = np.zeros((n_u, len(idx_i)))
    rows = r["uid"].map(idx_u).values
    cols = r["iid"].map(idx_i).values
    grp = r.groupby("uid")["rating"].mean()
    mu_u[[idx_u[u] for u in grp.index]] = grp.values
    np.add.at(C, (rows, cols), r["rating"].values - mu_u[rows])

    def served_by(decode):
        mus = np.empty(n_u)
        cvs = np.empty((n_u, C.shape[1]))
        for k in np.unique(decode):
            sel = decode == k
            mus[sel] = mu_u[sel].mean()
            cvs[sel] = C[sel].mean(0)
        return mus, cvs

    out = []

    def add(mech, param, decode):
        mus, cvs = served_by(decode)
        hr, nd = rank_metrics(eval_rows, lambda u, c: mus[u] + cvs[u, c])
        out.append({"mech": mech, "param": param, "HR@10": hr, "NDCG@10": nd,
                    "alpha_emp": empirical_mi(theta, decode)})
        print(out[-1], flush=True)

    add("oracle", "-", theta)
    add("global", "-", np.zeros(n_u, dtype=int))
    for K in [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 128]:
        if K > M:
            continue
        ug, _, _ = balanced_group_partition(theta, centers, K, master_seed)
        add("grouping", K, ug)
    for eps in [2, 4, 6, 8, 12]:
        d, _, Z, _ = mechanism_grr(theta, M, eps, master_seed)
        add("grr", eps, d)
        d, _, Z, _ = mechanism_brappor(theta, M, eps, master_seed)
        add("brappor", eps, d)
    scale = float(np.std(emb, axis=0).mean())
    for sig in [0.5, 1, 2, 5, 13]:
        d, _, Z, _ = mechanism_gaussian(emb, centers, sig * scale,
                                        master_seed)
        add("gaussian", sig, d)
    for dp in [2, 4, 8, 16, 32]:
        d, _, Z, _ = mechanism_pca(emb, centers, dp, master_seed)
        add("pca", dp, d)

    os.makedirs(RES, exist_ok=True)
    pd.DataFrame(out).to_csv(f"{RES}/e7_ranking_{dataset_name}.csv",
                             index=False)
    print("saved E7")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ml-1m",
         int(sys.argv[2]) if len(sys.argv) > 2 else 128)
