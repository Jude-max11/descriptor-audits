"""E13: unified estimand table at the matched operating points (R1-M1/R2-M3).

For each of the five mechanisms at the Table II matched parameters, this
script places on ONE footing, per protected attribute:

  adv(Z)        trained attack on the released descriptor only
  adv(Z,that)   trained attack on [Z, onehot(decoded type)]
  adv(that)     trained attack on onehot(decoded type) alone  (authorized
                task information)
  residual      adv(Z,that) - adv(that): the within-type signal the trained
                attack actually exploits
  I(A;Z)        empirical MI of the decoded-type view (nats)
  L_res         prior-weighted per-type residual MI (plugin, as in E1/E3)

Protocol: stratified 5-fold CV x 10 seeds, best of MLP(100)/balanced LR per
feature set (identical to the manuscript's trained-attack protocol).

Output: results/e13_estimand_{ds}.csv
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import (load_dataset, attrs_of, build_user_side, cluster_types,
                      mechanism_grouping, mechanism_grr, mechanism_brappor,
                      mechanism_gaussian, mechanism_pca, empirical_mi,
                      make_attacker)
from e8_worstcase import onehot_argmax

RES = os.path.join(os.path.dirname(__file__), "..", "results")


def seed_adv(kind, s, X, y, folds=5):
    """Mean CV advantage of attacker `kind` under seed `s` (5 folds)."""
    skf = StratifiedKFold(folds, shuffle=True, random_state=s)
    vals = []
    for tr, te in skf.split(X, y):
        clf = make_attacker(kind)
        try:
            clf.fit(X[tr], y[tr])
            pred = clf.predict(X[te])
        except Exception:
            continue
        classes = np.unique(y[tr])
        per = [np.mean(pred[y[te] == c] == c)
               if (y[te] == c).any() else 0.0 for c in classes]
        vals.append(float(np.mean(per)) - 1.0 / len(classes))
    return float(np.mean(vals)) if vals else float("nan")


def adv_cv(X, y, seeds=tuple(range(10)), folds=5):
    """Mean advantage of the best attacker kind under CV (manuscript rule)."""
    out = Parallel(n_jobs=20)(
        delayed(seed_adv)(k, s, X, y, folds)
        for k in ("mlp", "lr") for s in seeds)
    mlp = np.nanmean([v for k, v in zip(("mlp",) * 10 + ("lr",) * 10, out)
                      if k == "mlp"])
    lr = np.nanmean([v for k, v in zip(("mlp",) * 10 + ("lr",) * 10, out)
                     if k == "lr"])
    kind = "mlp" if mlp >= lr else "lr"
    return (mlp if kind == "mlp" else lr), kind


def main(dataset_name="ml-1m", M=128, master_seed=7):
    ds = load_dataset(dataset_name)
    emb, svd, idx_i, top = build_user_side(ds, top_items=500, svd_dim=32,
                                           seed=master_seed)
    theta, centers = cluster_types(emb, M, seed=master_seed)
    yattrs = {a: ds.users[a].values for a in attrs_of(ds)}
    scale = float(np.std(emb, axis=0).mean())

    mech = {
        ("BTQ", "K=89"): mechanism_grouping(theta, centers, 89, master_seed),
        ("GRR", "eps=10"): mechanism_grr(theta, M, 10, master_seed),
        ("bRAPPOR", "eps=16"): mechanism_brappor(theta, M, 16, master_seed),
        ("Gaussian", "sigma=0.37s"): mechanism_gaussian(
            emb, centers, 0.37 * scale, master_seed),
        ("PCA", "d'=7"): mechanism_pca(emb, centers, 7, master_seed),
        ("PCA", "d'=8"): mechanism_pca(emb, centers, 8, master_seed),
    }

    rows = []
    for (name, param), out in mech.items():
        decode, Z = out[0], np.asarray(out[2])
        if name in ("BTQ", "GRR"):
            that = onehot_argmax(Z)
        else:
            that = decode.astype(int)
        nL = int(that.max()) + 1
        onehot = np.eye(nL)[that]
        feats = {"Z": Z, "Z+thetahat": np.hstack([Z, onehot]),
                 "thetahat": onehot}
        for attr, y in yattrs.items():
            names = list(feats.keys())
            out = Parallel(n_jobs=30)(
                delayed(seed_adv)(kind, s, feats[fn], y)
                for fn in names for kind in ("mlp", "lr")
                for s in range(10))
            i = 0
            per_feat = {}
            for fn in names:
                per = {}
                for kind in ("mlp", "lr"):
                    per[kind] = out[i:i + 10]
                    i += 10
                kind = max(per, key=lambda k: np.nanmean(per[k]))
                per_feat[fn] = (float(np.nanmean(per[kind])), kind)
            r = {"mechanism": name, "param": param, "attr": attr,
                 "I_A_decode": empirical_mi(that, y)}
            tot, cnt = 0.0, 0
            for t in np.unique(theta):
                m = theta == t
                if m.sum() >= 5 and len(np.unique(y[m])) > 1:
                    tot += empirical_mi(that[m], y[m], n_perm=0) * m.sum()
                    cnt += m.sum()
            r["L_res"] = tot / max(cnt, 1)
            for fn, (v, kind) in per_feat.items():
                r[f"adv_{fn}"] = v
                r[f"attacker_{fn}"] = kind
            r["residual_adv"] = r["adv_Z+thetahat"] - r["adv_thetahat"]
            rows.append(r)
            print({k: (round(v, 4) if isinstance(v, float) else v)
                   for k, v in r.items()}, flush=True)

    os.makedirs(RES, exist_ok=True)
    path = f"{RES}/e13_estimand_{dataset_name}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print("saved", path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ml-1m",
         int(sys.argv[2]) if len(sys.argv) > 2 else 128)
