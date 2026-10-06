"""E9: adversary auxiliary-information and knowledge sweep (review R5).

Part 1 -- auxiliary fraction: attacker trained on f in {1%,...,100%} of a
fixed shadow pool (disjoint 50% eval pool), best of MLP(100)/balanced LR,
10 seeds, for grouping K=89, Gaussian sigma=0.37s, PCA d'=8 (matched
operating points).  Traces the white-box-to-near-black-box path: small f is
the realistic gray-box attacker, f=1 approximates the white-box ceiling.

Part 2 -- task side information: for embedding-level releases the attacker
may additionally know the server's decoded type theta_hat (authorized task
information at deployment).  We compare attackers observing Z only,
[Z, onehot(theta_hat)], and onehot(theta_hat) only: the gap between the
Z-only and theta_hat-only advantages is the within-type (residual) signal
actually exploited by the trained attack.

Outputs: results/e9_auxsweep_{ds}.csv, results/e9_taskside_{ds}.csv
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import (load_dataset, attrs_of, build_user_side, cluster_types,
                      mechanism_grouping, mechanism_gaussian, mechanism_pca,
                      make_attacker)

RES = os.path.join(os.path.dirname(__file__), "..", "results")
FRACS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)


def _adv_one(Z, y, tr, ev, kind):
    clf = make_attacker(kind)
    try:
        clf.fit(Z[tr], y[tr])
        pred = clf.predict(Z[ev])
        classes = np.unique(y[tr])
        per = [np.mean(pred[y[ev] == c] == c) if (y[ev] == c).any() else 0.0
               for c in classes]
        return float(np.mean(per)) - 1.0 / len(classes)
    except Exception:
        return -1.0


def aux_sweep(Z, y, fracs=FRACS, seeds=tuple(range(10)),
              kinds=("mlp", "lr"), n_jobs=8):
    idx = np.arange(len(y))
    tr_pool, ev = train_test_split(idx, test_size=0.5, stratify=y,
                                   random_state=99)
    jobs = []
    for frac in fracs:
        for s in seeds:
            pos = np.random.default_rng(s).choice(
                len(tr_pool), max(int(frac * len(tr_pool)), 2),
                replace=False)
            tr = tr_pool[pos]
            for kind in kinds:
                jobs.append((frac, s, kind, tr))
    out = Parallel(n_jobs=n_jobs)(
        delayed(lambda t: (t[0], t[1], t[2], _adv_one(Z, y, t[3], ev, t[2])))(
            t) for t in jobs)
    rows = []
    for frac in fracs:
        vals = [v for f, s, k, v in out if f == frac]
        rows.append({"frac": frac, "adv_mean": float(np.mean(vals)),
                     "adv_std": float(np.std(vals)),
                     "n_shadow": max(int(frac * len(tr_pool)), 2)})
    return rows


def task_side(Z, decode, y, seeds=tuple(range(10)), folds=5, n_jobs=8):
    """Attackers observing Z, [Z, onehot(decode)], or onehot(decode)."""
    from sklearn.model_selection import StratifiedKFold
    M = int(decode.max()) + 1
    feats = {"Z": Z,
             "Z+thetahat": np.hstack([Z, np.eye(M)[decode]]),
             "thetahat": np.eye(M)[decode]}
    jobs = []
    for name, X in feats.items():
        for s in seeds:
            skf = StratifiedKFold(folds, shuffle=True, random_state=s)
            for tr, te in skf.split(X, y):
                for kind in ("mlp", "lr"):
                    jobs.append((name, X, kind, tr, te))
    out = Parallel(n_jobs=n_jobs)(
        delayed(lambda t: (t[0], _adv_task(t[1], y, t[3], t[4], t[2])))(
            t) for t in jobs)
    rows = []
    for name in feats:
        vals = [v for n, v in out if n == name]
        rows.append({"features": name, "adv_mean": float(np.mean(vals)),
                     "adv_std": float(np.std(vals))})
    return rows


def _adv_task(X, y, tr, te, kind):
    clf = make_attacker(kind)
    try:
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        classes = np.unique(y[tr])
        per = [np.mean(pred[y[te] == c] == c) if (y[te] == c).any() else 0.0
               for c in classes]
        return float(np.mean(per)) - 1.0 / len(classes)
    except Exception:
        return -1.0


def main(dataset_name="ml-1m", M=128, master_seed=7):
    ds = load_dataset(dataset_name)
    emb, svd, idx_i, top = build_user_side(ds, top_items=500, svd_dim=32,
                                           seed=master_seed)
    theta, centers = cluster_types(emb, M, seed=master_seed)
    yattrs = {a: ds.users[a].values for a in attrs_of(ds)}
    scale = float(np.std(emb, axis=0).mean())

    mech = {
        "grouping K=89": mechanism_grouping(theta, centers, 89, master_seed),
        "gaussian 0.37s": mechanism_gaussian(emb, centers, 0.37 * scale,
                                             master_seed),
        "pca d'=8": mechanism_pca(emb, centers, 8, master_seed),
    }

    # Part 1: auxiliary fraction sweep
    rows = []
    for name, out in mech.items():
        Z = out[2]
        for attr, y in yattrs.items():
            for r in aux_sweep(np.asarray(Z), y):
                rows.append({"mechanism": name, "attr": attr, **r})
            print("aux:", name, attr, flush=True)
    pd.DataFrame(rows).to_csv(f"{RES}/e9_auxsweep_{dataset_name}.csv",
                              index=False)
    print("saved auxsweep")

    # Part 2: task side information (embedding-level releases)
    rows2 = []
    for name in ("gaussian 0.37s", "pca d'=8"):
        decode = mech[name][0]
        Z = np.asarray(mech[name][2])
        for attr, y in yattrs.items():
            for r in task_side(Z, decode, y):
                rows2.append({"mechanism": name, "attr": attr, **r})
            print("taskside:", name, attr, flush=True)
    pd.DataFrame(rows2).to_csv(f"{RES}/e9_taskside_{dataset_name}.csv",
                               index=False)
    print("saved taskside")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ml-1m",
         int(sys.argv[2]) if len(sys.argv) > 2 else 128)
