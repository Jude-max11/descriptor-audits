"""E12: user-level isolated-pipeline replication (teacher review R3-M1).

The original pipeline fits the SVD basis and the K-means type family on the
FULL interaction matrix (all users, all ratings, including ratings later held
out for utility evaluation and the users later used as attack targets).  This
script rebuilds the deployment evaluation under strict user-level isolation:

  - User-level 60/20/20 train/validation/test split (stratified by gender).
  - Within every user, ratings split 80/20 (train-portion / held-out).
  - FIT stage (train users' train-portion ratings ONLY): SVD(32) basis,
    KMeans(M=128) centers, and the per-label served models.
  - ASSIGN stage: every user's embedding is the frozen-basis projection of
    their own train-portion summary; theta = nearest frozen center
    (estimate-then-group for evaluation users).
  - SELECT stage (validation users only): per mechanism, the grid point
    whose validation excess risk is closest to the manuscript's matched
    window (0.006 MSE above the true-type oracle).
  - REPORT stage (test users only): excess risk, trained attackers fit on
    train users and evaluated once on test users (10 attacker seeds, best of
    MLP(100)/balanced LR), AUC, P_theta, I(theta;A), permutation-corrected
    L_res.
  - Isolation audit logged as JSON (split sizes, rating counts, assertions).
  - K-means seed sensitivity: centers refit with seeds {7,8,9}; the BTQ
    matched attack and I(theta;A) are repeated per seed.

Outputs: results/e12_isolated_{ds}.csv, results/e12_audit_{ds}.json,
         results/e12_kmseed_{ds}.csv
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import (load_dataset, attrs_of, balanced_group_partition,
                      mechanism_grr, mechanism_brappor, mechanism_gaussian,
                      mechanism_pca, empirical_mi)

RES = os.path.join(os.path.dirname(__file__), "..", "results")
SPLIT_SEED = 20260919
TOP_ITEMS, SVD_DIM, M_TYPES = 500, 32, 128
TARGET_EXCESS = 0.006

GRIDS = {
    "BTQ": [("grouping", k) for k in (55, 89, 128)],
    "GRR": [("grr", e) for e in (8, 10, 12)],
    "bRAPPOR": [("brappor", e) for e in (12, 16, 20)],
    "Gaussian": [("gaussian", s) for s in (0.25, 0.37, 0.50)],
    "PCA": [("pca", d) for d in (6, 7, 8)],
}
TABLE2_PARAM = {"BTQ": ("grouping", 89), "GRR": ("grr", 10),
                "bRAPPOR": ("brappor", 16), "Gaussian": ("gaussian", 0.37),
                "PCA": ("pca", 7)}


# ---------------------------------------------------------------- fits

def build_train_matrix(ds, idx_u, idx_i, heldout_mask):
    """Users x top-items matrix from NON-held-out ratings only."""
    X = np.zeros((len(ds.uids), len(idx_i)))
    r = ds.ratings[~heldout_mask]
    sub = r[r["iid"].isin(idx_i)]
    X[sub["uid"].map(idx_u).values, sub["iid"].map(idx_i).values] = \
        sub["rating"].values
    nz = X > 0
    cnt = nz.sum(1, keepdims=True)
    cnt[cnt == 0] = 1
    mu = X.sum(1, keepdims=True) / cnt
    return np.where(nz, X - mu, 0.0)


def client_models(ds, idx_u, idx_i, heldout_mask, global_mu):
    """Per-user local models (mu_u, centered offsets) from train portions."""
    r = ds.ratings[~heldout_mask]
    r = r[r["iid"].isin(idx_i)]
    mu_u = np.full(len(ds.uids), global_mu)
    grp = r.groupby("uid")["rating"].mean()
    mu_u[[idx_u[u] for u in grp.index]] = grp.values
    C = np.zeros((len(ds.uids), len(idx_i)))
    rows = r["uid"].map(idx_u).values
    cols = r["iid"].map(idx_i).values
    np.add.at(C, (rows, cols), r["rating"].values - mu_u[rows])
    return mu_u, C


def heldout_by_user(ds, idx_u, idx_i, heldout_mask):
    """Held-out ratings as positional (row_user, col_item, rating)."""
    r = ds.ratings[heldout_mask]
    r = r[r["iid"].isin(idx_i)]
    return (r["uid"].map(idx_u).values, r["iid"].map(idx_i).values,
            r["rating"].values)


def served_models(decode, train_users, mu_u, C, n_labels):
    """Per-label mean client model pooled over TRAIN users, applied to every
    user by their decoded label."""
    gm, gC = mu_u[train_users].mean(), C[train_users].mean(0)
    lab_mu = np.empty(n_labels)
    lab_C = np.empty((n_labels, C.shape[1]))
    dtr = decode[train_users]
    for k in range(n_labels):
        sel = train_users[dtr == k]
        lab_mu[k] = mu_u[sel].mean() if len(sel) else gm
        lab_C[k] = C[sel].mean(0) if len(sel) else gC
    k_ = np.clip(decode, 0, n_labels - 1)
    return lab_mu[k_], lab_C[k_]


def mse_on(mus, vecs, rows, cols, ratings):
    pred = mus[rows] + vecs[rows, cols]
    return float(np.mean((pred - ratings) ** 2))


# ---------------------------------------------------------------- attacks

def _one_fit(kind, s, Z, y, tr, te):
    if kind == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=(100,), alpha=1.0,
                            learning_rate_init=1e-3, max_iter=500,
                            random_state=s, early_stopping=False)
    else:
        clf = LogisticRegression(max_iter=2000, C=1.0,
                                 class_weight="balanced")
    clf.fit(Z[tr], y[tr])
    pred = clf.predict(Z[te])
    classes = np.unique(y[tr])
    bacc = float(np.mean([np.mean(pred[y[te] == c] == c)
                          for c in classes]))
    adv = bacc - 1.0 / len(classes)
    auc = None
    if len(classes) == 2:
        auc = roc_auc_score(
            y[te], clf.predict_proba(Z[te])[:, list(classes).index(1)])
    return kind, s, adv, auc


def attack_test(Z, y, tr, te, seeds=tuple(range(10))):
    """Best-of MLP/LR trained on TRAIN users, evaluated once on TEST users."""
    from joblib import Parallel, delayed
    out = Parallel(n_jobs=20)(
        delayed(_one_fit)(k, s, Z, y, tr, te)
        for k in ("mlp", "lr") for s in seeds)
    per = {k: {"adv_seeds": [], "auc_seeds": []} for k in ("mlp", "lr")}
    for kind, s, adv, auc in out:
        per[kind]["adv_seeds"].append(adv)
        if auc is not None:
            per[kind]["auc_seeds"].append(auc)
    best = max(per, key=lambda k: np.mean(per[k]["adv_seeds"]))
    b = per[best]
    return {"attacker": best,
            "adv_mean": float(np.mean(b["adv_seeds"])),
            "adv_std": float(np.std(b["adv_seeds"])),
            "auc_mean": (float(np.mean(b["auc_seeds"])) if b["auc_seeds"]
                         else float("nan")),
            "auc_std": (float(np.std(b["auc_seeds"])) if b["auc_seeds"]
                        else float("nan"))}


def l_res_corr(decode, y, theta, n_perm=20, seed=0):
    """Prior-weighted per-type MI(decode; A), permutation bias corrected."""
    tot, cnt = 0.0, 0
    for t in np.unique(theta):
        m = theta == t
        if m.sum() >= 5 and len(np.unique(y[m])) > 1:
            tot += empirical_mi(decode[m], y[m], n_perm=n_perm,
                                seed=seed) * m.sum()
            cnt += m.sum()
    return tot / max(cnt, 1)


# ---------------------------------------------------------------- main

def run_split(ds, cluster_seed=SPLIT_SEED, tag="main", mech_filter=None):
    rng = np.random.default_rng(SPLIT_SEED)
    idx = np.arange(len(ds.uids))
    gender = ds.users["gender"].values
    tr, rest = train_test_split(idx, test_size=0.4, stratify=gender,
                                random_state=SPLIT_SEED)
    va, te = train_test_split(rest, test_size=0.5, stratify=gender[rest],
                              random_state=SPLIT_SEED)
    tr, va, te = np.sort(tr), np.sort(va), np.sort(te)

    heldout_mask = rng.random(len(ds.ratings)) < 0.2

    pop = ds.ratings["iid"].value_counts()
    top = pop.index[:TOP_ITEMS]
    idx_u = {u: i for i, u in enumerate(ds.uids)}
    idx_i = {i: j for j, i in enumerate(top)}

    Xc = build_train_matrix(ds, idx_u, idx_i, heldout_mask)
    svd = TruncatedSVD(n_components=SVD_DIM, random_state=7)
    svd.fit(Xc[tr])                                  # TRAIN users only
    emb = svd.transform(Xc)

    km = KMeans(n_clusters=M_TYPES, n_init=10, random_state=cluster_seed)
    km.fit(emb[tr])                                  # TRAIN users only
    theta = km.predict(emb).astype(int)
    centers = km.cluster_centers_

    gmu = float(ds.ratings.loc[~heldout_mask, "rating"].mean())
    mu_u, C = client_models(ds, idx_u, idx_i, heldout_mask, gmu)

    hr, hc, hv = heldout_by_user(ds, idx_u, idx_i, heldout_mask)

    def excess(decode, n_labels):
        mus_o, vecs_o = served_models(theta, tr, mu_u, C, M_TYPES)
        mus_s, vecs_s = served_models(decode, tr, mu_u, C, n_labels)
        out = {}
        for split, users in (("val", va), ("test", te)):
            m = np.isin(hr, users)
            out[split] = (mse_on(mus_s, vecs_s, hr[m], hc[m], hv[m])
                          - mse_on(mus_o, vecs_o, hr[m], hc[m], hv[m]))
        return out

    audit = {
        "tag": tag, "cluster_seed": cluster_seed,
        "n_users": {"train": len(tr), "val": len(va), "test": len(te)},
        "assert_user_disjoint": bool(
            not (set(tr.tolist()) & set(va.tolist())
                 or set(tr.tolist()) & set(te.tolist())
                 or set(va.tolist()) & set(te.tolist()))),
        "n_ratings": {"train_portion": int((~heldout_mask).sum()),
                      "heldout": int(heldout_mask.sum()),
                      "heldout_test_rows": int(np.isin(hr, te).sum()),
                      "heldout_val_rows": int(np.isin(hr, va).sum())},
        "svd_kmeans_fit_scope": "train users, train-portion ratings only",
        "I_thetaA": {},
    }
    yattrs = {a: ds.users[a].values for a in attrs_of(ds)}
    for a, y in yattrs.items():
        audit["I_thetaA"][a] = empirical_mi(theta[te], y[te])

    scale = float(np.std(emb[tr], axis=0).mean())

    def build(mech, param):
        seed = 7
        if mech == "grouping":
            _, t2g, _ = balanced_group_partition(theta[tr], centers, param,
                                                 seed)
            decode = t2g[theta]
            Z = np.eye(int(decode.max()) + 1)[decode]
        elif mech == "grr":
            decode, _, Z, _ = mechanism_grr(theta, M_TYPES, param, seed)
        elif mech == "brappor":
            decode, _, Z, _ = mechanism_brappor(theta, M_TYPES, param, seed)
        elif mech == "gaussian":
            decode, _, Z, _ = mechanism_gaussian(emb, centers, param * scale,
                                                 seed)
        elif mech == "pca":
            decode, _, Z, _ = mechanism_pca(emb, centers, int(param), seed)
        return decode, np.asarray(Z)

    n_labels_of = {"grouping": None, "grr": M_TYPES, "brappor": M_TYPES,
                   "gaussian": M_TYPES, "pca": M_TYPES}

    rows = []
    for label, grid in GRIDS.items():
        if mech_filter and label not in mech_filter:
            continue
        mech = grid[0][0]
        cands = []
        for _, param in grid:
            decode, Z = build(mech, param)
            nl = n_labels_of[mech] or int(decode.max()) + 1
            cands.append((param, excess(decode, nl), decode, Z, nl))
        sel = min(cands, key=lambda c: abs(c[1]["val"] - TARGET_EXCESS))
        t2p = TABLE2_PARAM[label][1]
        for param, ex, decode, Z, nl in cands:
            is_sel, is_t2 = param == sel[0], param == t2p
            if not (is_sel or is_t2):
                continue
            P_theta = float(np.mean(decode[te] != theta[te]))
            alpha_emp = empirical_mi(theta[te], decode[te])
            for a, y in yattrs.items():
                atk = attack_test(Z, y, tr, te)
                rows.append({
                    "mechanism": label, "mech": mech, "param": param,
                    "selected": is_sel, "table2_param": is_t2,
                    "excess_val": ex["val"], "excess_test": ex["test"],
                    "P_theta": P_theta, "alpha_emp": alpha_emp,
                    "attr": a, **atk,
                    "L_res": l_res_corr(decode[te], y[te], theta[te]),
                })
                print(f"[{tag}] {label} p={param} sel={is_sel} t2={is_t2} "
                      f"{a}: excess={ex['test']:.4f} "
                      f"adv={atk['adv_mean']:.3f}", flush=True)
    return rows, audit


def kmseed_sensitivity(ds):
    out = []
    for cseed in (7, 8, 9):
        rows, audit = run_split(ds, cluster_seed=cseed, tag=f"km{cseed}",
                                mech_filter=("BTQ",))
        btq = [r for r in rows if r["selected"] and r["attr"] == "gender"][0]
        out.append({"cluster_seed": cseed,
                    "I_thetaA_gender": audit["I_thetaA"]["gender"],
                    "K_sel": btq["param"], "excess_test": btq["excess_test"],
                    "adv_mean": btq["adv_mean"], "adv_std": btq["adv_std"]})
    return out


def main(dataset_name="ml-1m"):
    ds = load_dataset(dataset_name)
    rows, audit = run_split(ds)
    os.makedirs(RES, exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{RES}/e12_isolated_{dataset_name}.csv",
                              index=False)
    with open(f"{RES}/e12_audit_{dataset_name}.json", "w") as f:
        json.dump(audit, f, indent=2)
    km = kmseed_sensitivity(ds)
    pd.DataFrame(km).to_csv(f"{RES}/e12_kmseed_{dataset_name}.csv",
                            index=False)
    print("audit:", json.dumps(audit, indent=2))
    print("kmseed:", json.dumps(km, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ml-1m")
