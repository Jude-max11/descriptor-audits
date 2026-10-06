"""E1 + E3: attribute-inference attacks across descriptor mechanisms.

Sweep: grouping (K), GRR (eps), bRAPPOR (eps), Gaussian (sigma), PCA (d').
Per config report:
  alpha_emp   empirical I(theta; Z_disc) in nats  (common x-axis)
  excess_risk held-out MSE(server model) - MSE(client's own model)
  P_theta     task identification error of the server's decode
  attack      BAcc / advantage / micro-F1 / AUC of trained attacker on Z
              for each protected attribute (gender, age3)
  I_A_given   I(A; Z_disc) and residual I(A; Z_disc | theta)
Protocol: stratified 5-fold CV x seeds, MLP(100) + LR + RF attackers.
"""
from __future__ import annotations

import os
import sys
import math
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import (load_dataset, attrs_of, build_user_side, cluster_types,
                      mechanism_grouping, mechanism_grr, mechanism_brappor,
                      mechanism_gaussian, mechanism_pca, empirical_mi,
                      run_attack, save_json)

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
RES = os.path.join(os.path.dirname(__file__), "..", "results")


# ---------------------------------------------------------------- utility

class RatingUtility:
    """Held-out rating MSE for server-side models (from descriptor decode)
    vs the client's own local model."""

    def __init__(self, ds, idx_i, top, test_frac=0.2, seed=0):
        rng = np.random.default_rng(seed)
        r = ds.ratings.copy()
        r = r[r["iid"].isin(idx_i)]
        r["fold"] = rng.random(len(r)) < test_frac          # True = held-out
        self.train, self.test = r[~r["fold"]], r[r["fold"]]
        self.idx_u = {u: i for i, u in enumerate(ds.uids)}
        self.idx_i = idx_i
        self.n_items = len(top)
        self.global_mu = self.train["rating"].mean()
        n_u = len(ds.uids)
        # client local model: per-user mean + centered offsets (train only)
        tr = self.train
        self.mu_u = np.full(n_u, self.global_mu)
        grp = tr.groupby("uid")["rating"].mean()
        self.mu_u[[self.idx_u[u] for u in grp.index]] = grp.values
        self.C = np.zeros((n_u, self.n_items))
        rows = tr["uid"].map(self.idx_u).values
        cols = tr["iid"].map(self.idx_i).values
        vals = tr["rating"].values - self.mu_u[rows]
        np.add.at(self.C, (rows, cols), vals)               # one rating per (u,i)
        # held-out lists per user
        self.test_rows = self.test["uid"].map(self.idx_u).values
        self.test_items = self.test["iid"].map(self.idx_i).values
        self.test_ratings = self.test["rating"].values

    def mse(self, mus, cvecs):
        """Mean squared error of (per-user) served models on held-out ratings."""
        pred = mus[self.test_rows] + cvecs[self.test_rows, self.test_items]
        return float(np.mean((pred - self.test_ratings) ** 2))

    def models_from_decode(self, decode):
        """Server serves the mean local model of the decoded type/group."""
        n_u = len(self.mu_u)
        mus = np.empty(n_u)
        cvecs = np.empty((n_u, self.n_items))
        for k in np.unique(decode):
            sel = decode == k
            mus[sel] = self.mu_u[sel].mean()
            cvecs[sel] = self.C[sel].mean(0)
        return mus, cvecs


def bayes_task_error(theta, decode):
    """Bayes-optimal task identification error from the empirical joint
    P(theta, Z_disc): P_e = 1 - sum_z P(z) max_theta P(theta|z)."""
    n = len(theta)
    T = int(theta.max()) + 1
    K = int(decode.max()) + 1
    joint = np.zeros((K, T))
    np.add.at(joint, (decode, theta), 1.0)
    pz = joint.sum(1, keepdims=True)
    return float(1.0 - np.sum(np.max(joint, 1) / n))


def run_config(name, param, decode, Z, theta, util, yattrs, seed):
    mus_s, cvecs_s = util.models_from_decode(decode)
    mus_o, cvecs_o = util.models_from_decode(theta)     # oracle: server
    mse_own = util.mse(util.mu_u, util.C)               # knows theta
    mse_oracle = util.mse(mus_o, cvecs_o)
    mse_srv = util.mse(mus_s, cvecs_s)
    p_theta = float(np.mean(decode != theta))
    p_theta_bayes = bayes_task_error(theta, decode)
    alpha_emp = empirical_mi(theta, decode)
    row = {"mechanism": name, "param": param, "alpha_emp": alpha_emp,
           "P_theta": p_theta, "P_theta_bayes": p_theta_bayes,
           "mse_own": mse_own, "mse_server": mse_srv,
           "mse_oracle": mse_oracle,
           "excess_risk": mse_srv - mse_oracle, "seed_global": seed}
    for attr, y in yattrs.items():
        # strongest-attacker evaluation (standard AIA practice): best of
        # MLP(100) and class-balanced logistic regression, 5-fold x 10 seeds
        best_adv, best_auc, best_bacc, best_std = -1.0, -1.0, -1.0, np.nan
        for kind in ("mlp", "lr"):
            atk = run_attack(Z, y, seeds=tuple(range(10)), folds=5,
                             attackers=(kind,))
            adv = atk.get("adv", (np.nan,))[0]
            if np.isfinite(adv) and adv > best_adv:
                best_adv = atk["adv"][0]
                best_std = atk["adv"][1]
                best_auc = atk.get("auc", (np.nan,))[0]
                best_bacc = atk.get("bacc", (np.nan,))[0]
        row[f"{attr}_bacc"] = best_bacc
        row[f"{attr}_adv"] = best_adv
        row[f"{attr}_adv_std"] = best_std
        row[f"{attr}_auc"] = best_auc
        # threshold-free advantage (binary): 2(AUC - 0.5), MIA-standard
        row[f"{attr}_adv_auc"] = (2 * (best_auc - 0.5)
                                  if np.isfinite(best_auc) else np.nan)
        row[f"{attr}_I_A"] = empirical_mi(decode, y)
    # residual leakage I(A; decode | theta): prior-weighted per-type MI
    for attr, y in yattrs.items():
        tot, cnt = 0.0, 0
        for t in np.unique(theta):
            m = theta == t
            if m.sum() >= 5 and len(np.unique(y[m])) > 1:
                tot += empirical_mi(decode[m], y[m], n_perm=0) * m.sum()
                cnt += m.sum()
        row[f"{attr}_L_res"] = tot / max(cnt, 1)
    return row


def main(dataset_name: str, path: str, M: int = 128, master_seed: int = 7):
    np.random.seed(master_seed)
    ds = load_dataset(dataset_name)
    emb, svd, idx_i, top = build_user_side(ds, top_items=500, svd_dim=32,
                                           seed=master_seed)
    theta, centers = cluster_types(emb, M, seed=master_seed)
    util = RatingUtility(ds, idx_i, top, test_frac=0.2, seed=master_seed)
    yattrs = {a: ds.users[a].values for a in attrs_of(ds)}

    # embedding scale for sigma grid
    scale = float(np.std(emb, axis=0).mean())

    configs = []
    for K in [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 128]:
        if K <= M:
            configs.append(("grouping", K))
    for eps in [0.5, 1, 2, 3, 4, 6, 8, 12, 16, 20]:
        configs.append(("grr", eps))
        configs.append(("brappor", eps))
    for sig in [0.5, 1, 2, 3, 5, 8, 13, 21]:
        configs.append(("gaussian", sig * scale))
    for dp in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]:
        configs.append(("pca", dp))

    def build(name, param):
        seed = master_seed
        if name == "grouping":
            decode, gc, Z = mechanism_grouping(theta, centers, param, seed)
        elif name == "grr":
            decode, _, Z, _ = mechanism_grr(theta, M, param, seed)
        elif name == "brappor":
            decode, _, Z, _ = mechanism_brappor(theta, M, param, seed)
        elif name == "gaussian":
            decode, _, Z, _ = mechanism_gaussian(emb, centers, param, seed)
        elif name == "pca":
            decode, _, Z, _ = mechanism_pca(emb, centers, int(param), seed)
        return decode, Z

    from joblib import Parallel, delayed

    def one(name, param):
        decode, Z = build(name, param)
        return run_config(name, param, decode, Z, theta, util, yattrs,
                          master_seed)

    only = sys.argv[3].split(",") if len(sys.argv) > 3 else None
    cfgs = [(n, p) for n, p in configs if not only or n in only]
    rows = Parallel(n_jobs=min(len(cfgs), 40), max_nbytes=None)(
        delayed(one)(n, p) for n, p in cfgs)
    for row in rows:
        print(f"[{dataset_name}] {row['mechanism']} p={row['param']}: "
              f"alpha={row['alpha_emp']:.2f} excess={row['excess_risk']:.4f} "
              f"adv_g={row.get('gender_adv', float('nan')):.3f}", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(RES, exist_ok=True)
    if only:
        df.to_csv(f"{RES}/sweep_{dataset_name}_part_{only[0]}.csv", index=False)
        print("saved part", only)
    else:
        df.to_csv(f"{RES}/sweep_{dataset_name}.csv", index=False)
        save_json({"M": M, "master_seed": master_seed, "emb_scale": scale,
                   "n_users": len(ds.uids), "n_test_ratings": len(util.test)},
                  f"{RES}/meta_{dataset_name}.json")
        print("saved", f"{RES}/sweep_{dataset_name}.csv")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "ml-1m"
    M = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    main(which, None, M=M)
