"""E14: statistical reinforcement for the matched comparison (R1-M3/R3-M3).

Part A -- paired attack CIs.  Reruns the manuscript's matched-utility attack
protocol (stratified 5-fold CV x 10 seeds, best of MLP/LR) storing PER-SEED
mean advantages, so mechanism differences can be tested paired by seed.
Reports mean difference, 95% bootstrap CI (10k resamples), and a two-sided
paired t-test p-value for the key contrasts (type-level vs embedding-level).

Part B -- upper confidence bounds for the zero claims.  For each matched
mechanism and attribute: L_res (prior-weighted per-type plugin MI) and the
permutation-corrected conditional ML aggregation are recomputed together
with the permutation-null distribution of the SAME statistic (attribute
shuffled within type, 200 permutations).  Reported UCB is the bias-corrected
one-sided 95% bound: max(0, obs - null_mean) + q95(null - null_mean).

Part C -- excess-risk bootstrap CI at the matched points (rating-level
bootstrap, 500 resamples) under the original utility protocol.

Outputs: results/e14_paired_{ds}.csv, results/e14_ucb_{ds}.csv,
         results/e14_excess_ci_{ds}.csv
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import (load_dataset, attrs_of, build_user_side, cluster_types,
                      mechanism_grouping, mechanism_grr, mechanism_brappor,
                      mechanism_gaussian, mechanism_pca, make_attacker,
                      empirical_mi)
from e8_worstcase import max_leakage, MIN_TYPE, onehot_argmax
from e1e3_sweep import RatingUtility

RES = os.path.join(os.path.dirname(__file__), "..", "results")


# ---------------------------------------------------------------- Part A

def _seed_mean(kind, s, Z, y, folds=5):
    """Mean CV advantage of attacker `kind` under seed `s`."""
    skf = StratifiedKFold(folds, shuffle=True, random_state=s)
    vals = []
    for tr, te in skf.split(Z, y):
        clf = make_attacker(kind)
        try:
            clf.fit(Z[tr], y[tr])
            pred = clf.predict(Z[te])
        except Exception:
            continue
        classes = np.unique(y[tr])
        per = [np.mean(pred[y[te] == c] == c)
               if (y[te] == c).any() else 0.0 for c in classes]
        vals.append(float(np.mean(per)) - 1.0 / len(classes))
    return float(np.mean(vals)) if vals else float("nan")


def per_seed_adv(Z, y, seeds=tuple(range(10)), folds=5):
    """Per-seed mean advantage for the best attacker kind (paired by seed)."""
    from joblib import Parallel, delayed
    out = Parallel(n_jobs=20)(
        delayed(_seed_mean)(k, s, Z, y, folds)
        for k in ("mlp", "lr") for s in seeds)
    mlp, lr = out[:10], out[10:]
    kind = "mlp" if np.nanmean(mlp) >= np.nanmean(lr) else "lr"
    return (mlp if kind == "mlp" else lr), kind


def paired_bootstrap(d, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(d)
    means = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    t = stats.ttest_1samp(d, 0.0)
    return float(d.mean()), float(lo), float(hi), float(t.pvalue)


# ---------------------------------------------------------------- Part B

def stat_l_res(z, y, theta):
    tot, cnt = 0.0, 0
    for t in np.unique(theta):
        m = theta == t
        if m.sum() >= 5 and len(np.unique(y[m])) > 1:
            tot += empirical_mi(z[m], y[m], n_perm=0) * m.sum()
            cnt += m.sum()
    return tot / max(cnt, 1)


def stat_ml_cond(z, y, theta):
    """Permutation-corrected mean conditional ML (truncated at zero)."""
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


# ---------------------------------------------------------------- Part C
# (excess-risk bootstrap implemented inline in main)


def main(dataset_name="ml-1m", M=128, master_seed=7):
    ds = load_dataset(dataset_name)
    emb, svd, idx_i, top = build_user_side(ds, top_items=500, svd_dim=32,
                                           seed=master_seed)
    theta, centers = cluster_types(emb, M, seed=master_seed)
    yattrs = {a: ds.users[a].values for a in attrs_of(ds)}
    scale = float(np.std(emb, axis=0).mean())
    util = RatingUtility(ds, idx_i, top, test_frac=0.2, seed=master_seed)

    mech = {
        ("BTQ", "K=89"): mechanism_grouping(theta, centers, 89, master_seed),
        ("GRR", "eps=10"): mechanism_grr(theta, M, 10, master_seed),
        ("bRAPPOR", "eps=16"): mechanism_brappor(theta, M, 16, master_seed),
        ("Gaussian", "sigma=0.37s"): mechanism_gaussian(
            emb, centers, 0.37 * scale, master_seed),
        ("PCA", "d'=7"): mechanism_pca(emb, centers, 7, master_seed),
    }

    # ---------------- Part A: paired per-seed advantages ----------------
    per_seed = {}
    for (name, param), out in mech.items():
        Z = np.asarray(out[2])
        for attr in ("gender", "age3"):
            if attr not in yattrs:
                continue
            vals, kind = per_seed_adv(Z, yattrs[attr])
            per_seed[(name, attr)] = vals
            print(f"A: {name} {attr} ({kind}): "
                  f"{np.mean(vals):.4f}+-{np.std(vals):.4f}", flush=True)

    rows = []
    contrasts = [("BTQ", "Gaussian"), ("BTQ", "PCA"), ("GRR", "Gaussian"),
                 ("bRAPPOR", "Gaussian"), ("BTQ", "bRAPPOR"),
                 ("Gaussian", "PCA")]
    for attr in ("gender", "age3"):
        if attr not in yattrs:
            continue
        for a, b in contrasts:
            d = np.array(per_seed[(a, attr)]) - np.array(per_seed[(b, attr)])
            m, lo, hi, p = paired_bootstrap(d)
            rows.append({"attr": attr, "contrast": f"{a}-{b}",
                         "mean_diff": m, "ci_lo": lo, "ci_hi": hi,
                         "p_paired_t": p})
    pd.DataFrame(rows).to_csv(f"{RES}/e14_paired_{dataset_name}.csv",
                              index=False)

    # ---------------- Part B: UCBs for zero claims ----------------------
    rows = []
    for (name, param), out in mech.items():
        decode = out[0]
        Z = out[2]
        z_view = onehot_argmax(np.asarray(Z)) if name in ("BTQ", "GRR") \
            else decode.astype(int)
        for attr in ("gender", "age3"):
            if attr not in yattrs:
                continue
            y = yattrs[attr]
            for sname, fn in (("L_res", stat_l_res),
                              ("ML_cond", stat_ml_cond)):
                obs, ub = ucb(fn, z_view, y, theta)
                rows.append({"mechanism": name, "attr": attr,
                             "statistic": sname, "observed": obs,
                             "ucb95": ub})
                print(f"B: {name} {attr} {sname}: obs={obs:.4f} "
                      f"ucb95={ub:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(f"{RES}/e14_ucb_{dataset_name}.csv", index=False)

    # ---------------- Part C: excess bootstrap CI -----------------------
    rng = np.random.default_rng(0)
    rows = []
    tr_rows, tr_items, tr_rat = util.test_rows, util.test_items, \
        util.test_ratings
    for (name, param), out in mech.items():
        decode = out[0]
        mus_s, cvecs_s = util.models_from_decode(decode)
        mus_o, cvecs_o = util.models_from_decode(theta)
        errs_s = ((mus_s[tr_rows] + cvecs_s[tr_rows, tr_items]) - tr_rat) ** 2
        errs_o = ((mus_o[tr_rows] + cvecs_o[tr_rows, tr_items]) - tr_rat) ** 2
        boots = []
        for _ in range(500):
            idx = rng.integers(0, len(errs_s), len(errs_s))
            boots.append(float(errs_s[idx].mean() - errs_o[idx].mean()))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        point = float(errs_s.mean() - errs_o.mean())
        rows.append({"mechanism": name, "param": param, "excess": point,
                     "ci_lo": float(lo), "ci_hi": float(hi)})
        print(f"C: {name}: excess={point:.5f} [{lo:.5f},{hi:.5f}]",
              flush=True)
    pd.DataFrame(rows).to_csv(f"{RES}/e14_excess_ci_{dataset_name}.csv",
                              index=False)
    print("saved e14 outputs")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ml-1m",
         int(sys.argv[2]) if len(sys.argv) > 2 else 128)
