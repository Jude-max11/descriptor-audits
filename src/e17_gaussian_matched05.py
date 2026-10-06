"""E17: correction rerun for the Gaussian matched operating point.

The matched-protocol scripts (matched_attack_std, e8_matched, e9, e11,
e13, e14) used sigma = 0.37*scale = 0.2726, an OFF-GRID point obtained by
misreading the sweep grid's first entry (absolute sigma 0.3684, which is
0.5*scale) as "0.37s".  The in-window matched point (excess ~0.0057) is
the grid point sigma = 0.5*scale.  This script recomputes the Gaussian
chain at 0.5*scale, plus BTQ K=89 for the paired comparison, using the
identical protocols of the corresponding experiments:

  A  matched table cells: best-of MLP(100)/balanced LR, 5-fold x 10 seeds,
     advantage/AUC mean+-std, per-seed vectors (Table I row)
  B  estimand decomposition (e13 protocol): adv(Z), adv(Z+theta_hat),
     adv(theta_hat), residual, I(A;Z) decode view, L_res
  C  paired-by-seed bootstrap BTQ-Gaussian (e14 protocol, 10k resamples)
  D  auxiliary-fraction sweep (e9 protocol, disjoint half pool)
  E  repeated release (e11 protocol, LR x 5 seeds, independent/memoized)
  F  worst-case metrics (e8 protocol, decode + VQ views, 50 perms)

Outputs: results/e17_*.csv
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import (load_dataset, attrs_of, build_user_side, cluster_types,
                      mechanism_grouping, mechanism_gaussian,
                      empirical_mi, make_attacker, run_attack)
from e8_worstcase import (max_leakage, min_entropy_leakage, ml_conditional,
                          bootstrap_ci, il_inf_conditional,
                          advantage_percentiles, vq_labels)

RES = os.path.join(os.path.dirname(__file__), "..", "results")
G05 = ("Gaussian", "sigma=0.5s")


def setup(dataset_name="ml-1m", M=128, master_seed=7):
    ds = load_dataset(dataset_name)
    emb, svd, idx_i, top = build_user_side(ds, top_items=500, svd_dim=32,
                                           seed=master_seed)
    theta, centers = cluster_types(emb, M, seed=master_seed)
    yattrs = {a: ds.users[a].values for a in attrs_of(ds)}
    scale = float(np.std(emb, axis=0).mean())
    print(f"scale s = {scale:.6f}; matched sigma = {0.5*scale:.6f}", flush=True)
    mech = {
        ("BTQ", "K=89"): mechanism_grouping(theta, centers, 89, master_seed),
        G05: mechanism_gaussian(emb, centers, 0.5 * scale, master_seed),
    }
    return ds, emb, theta, centers, yattrs, scale, mech


def _seed_means(X, y, seeds=tuple(range(10)), folds=5):
    """Per-seed mean CV advantage for both attacker kinds (e14 pattern)."""
    out = {}
    for kind in ("mlp", "lr"):
        vals = []
        for s in seeds:
            skf = StratifiedKFold(folds, shuffle=True, random_state=s)
            per = []
            for tr, te in skf.split(X, y):
                clf = make_attacker(kind)
                try:
                    clf.fit(X[tr], y[tr])
                    pred = clf.predict(X[te])
                except Exception:
                    continue
                classes = np.unique(y[tr])
                per.append(float(np.mean(
                    [np.mean(pred[y[te] == c] == c)
                     if (y[te] == c).any() else 0.0 for c in classes]))
                    - 1.0 / len(classes))
            vals.append(float(np.mean(per)) if per else float("nan"))
        out[kind] = vals
    return out


def part_a_b(ds, emb, theta, centers, yattrs, mech):
    """Table I cells + estimand decomposition + paired bootstrap inputs."""
    rows, per_seed = [], {}
    for (name, param), out in mech.items():
        decode, Z = out[0], np.asarray(out[2])
        that = (np.argmax(Z, axis=1) if name == "BTQ"
                else decode.astype(int))
        nL = int(that.max()) + 1
        onehot = np.eye(nL)[that]
        feats = {"Z": Z, "Z+thetahat": np.hstack([Z, onehot]),
                 "thetahat": onehot}
        for attr, y in yattrs.items():
            # A: table cells via run_attack (adv/auc mean+std, best-of)
            best = None
            for kind in ("mlp", "lr"):
                res = run_attack(Z, y, seeds=tuple(range(10)), folds=5,
                                 attackers=(kind,))
                if best is None or res["adv"][0] > best[1]["adv"][0]:
                    best = (kind, res)
            kind, res = best
            # B: estimand per feature set
            est = {}
            for fn, X in feats.items():
                sm = _seed_means(X, y)
                kbest = max(sm, key=lambda k: np.nanmean(sm[k]))
                est[fn] = (float(np.nanmean(sm[kbest])), kbest)
            i_a = empirical_mi(that, y)
            tot = cnt = 0
            for t in np.unique(theta):
                m = theta == t
                if m.sum() >= 5 and len(np.unique(y[m])) > 1:
                    tot += empirical_mi(that[m], y[m], n_perm=0) * m.sum()
                    cnt += m.sum()
            rows.append({
                "mechanism": name, "param": param, "attr": attr,
                "adv_mean": res["adv"][0], "adv_std": res["adv"][1],
                "auc_mean": res.get("auc", (np.nan, np.nan))[0],
                "auc_std": res.get("auc", (np.nan, np.nan))[1],
                "attacker": kind,
                "adv_Z_thetahat": est["Z+thetahat"][0],
                "adv_thetahat": est["thetahat"][0],
                "residual_adv": est["Z+thetahat"][0] - est["thetahat"][0],
                "I_A_decode": i_a, "L_res": tot / max(cnt, 1)})
            # C: per-seed best-kind vector for pairing
            if attr == "gender":
                smz = _seed_means(Z, y)
                kbest = max(smz, key=lambda k: np.nanmean(sm[k]))
                per_seed[name] = np.array(smz[kbest], dtype=float)
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(f"{RES}/e17_gaussian05_matched.csv", index=False)
    return per_seed


def part_c(per_seed, n_boot=10000, seed=0):
    rows = []
    d = np.asarray(per_seed["BTQ"]) - np.asarray(per_seed[G05[0]])
    rng = np.random.default_rng(seed)
    means = [d[rng.integers(0, len(d), len(d))].mean()
             for _ in range(n_boot)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    t = stats.ttest_1samp(d, 0.0)
    rows.append({"contrast": "BTQ-Gaussian@0.5s", "mean_diff": d.mean(),
                 "ci_lo": lo, "ci_hi": hi, "p_paired_t": t.pvalue,
                 "btq_seeds": per_seed["BTQ"].round(4).tolist(),
                 "gauss_seeds": per_seed[G05[0]].round(4).tolist()})
    pd.DataFrame(rows).to_csv(f"{RES}/e17_gaussian05_paired.csv", index=False)
    print(rows[0], flush=True)


def part_d(emb, centers, yattrs, scale, master_seed=7):
    """e9-style auxiliary-fraction sweep for Gaussian@0.5s."""
    from sklearn.model_selection import train_test_split
    Z = np.asarray(mechanism_gaussian(emb, centers, 0.5 * scale,
                                      master_seed)[2])
    rows = []
    for attr, y in yattrs.items():
        idx = np.arange(len(y))
        tr_pool, ev = train_test_split(idx, test_size=0.5, stratify=y,
                                       random_state=99)
        vals = {}
        for frac in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0):
            advs = {"mlp": [], "lr": []}
            for s in range(10):
                pos = np.random.default_rng(s).choice(
                    len(tr_pool), max(int(frac * len(tr_pool)), 2),
                    replace=False)
                tr = tr_pool[pos]
                for kind in ("mlp", "lr"):
                    clf = make_attacker(kind)
                    clf.fit(Z[tr], y[tr])
                    pred = clf.predict(Z[ev])
                    classes = np.unique(y[tr])
                    advs[kind].append(float(np.mean(
                        [np.mean(pred[y[ev] == c] == c)
                         if (y[ev] == c).any() else 0.0
                         for c in classes])) - 1.0 / len(classes))
            kind = max(advs, key=lambda k: np.mean(advs[k]))
            rows.append({"mechanism": "Gaussian 0.5s", "attr": attr,
                         "frac": frac, "adv_mean": float(np.mean(advs[kind])),
                         "adv_std": float(np.std(advs[kind])),
                         "n_shadow": int(max(frac * len(tr_pool), 2)),
                         "attacker": kind})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(f"{RES}/e17_gaussian05_auxsweep.csv",
                              index=False)


def part_e(emb, centers, yattrs, scale, master_seed=7):
    """e11-style repeated release for Gaussian@0.5s."""
    d1, _, z1, _ = mechanism_gaussian(emb, centers, 0.5 * scale, master_seed)
    d2, _, z2, _ = mechanism_gaussian(emb, centers, 0.5 * scale,
                                      master_seed + 1000)
    configs = {"independent": (z1, z2, d1, d2), "memoized": (z1, z1, d1, d1)}
    rows = []
    for regime, (Z1, Z2, dd1, dd2) in configs.items():
        pair = np.asarray(dd1) * 128 + np.asarray(dd2)
        for attr, y in yattrs.items():
            for k, feats in ((1, np.asarray(Z1)),
                             (2, np.hstack([Z1, Z2]))):
                res = run_attack(feats, y, seeds=(0, 1, 2, 3, 4), folds=5,
                                 attackers=("lr",))
                lab = np.asarray(dd1) if k == 1 else pair
                rows.append({"mechanism": "gaussian 0.5s", "regime": regime,
                             "k": k, "attr": attr,
                             "adv_mean": res["adv"][0],
                             "adv_std": res["adv"][1],
                             "I_A_corr": empirical_mi(lab, y, n_perm=20,
                                                      seed=master_seed)})
    pd.DataFrame(rows).to_csv(f"{RES}/e17_gaussian05_composition.csv",
                              index=False)
    print(pd.DataFrame(rows).pivot_table(
        index=["regime", "attr"], columns="k", values="adv_mean")
        .to_string(), flush=True)


def part_f(emb, theta, centers, yattrs, scale, master_seed=7):
    """e8-style worst-case metrics for Gaussian@0.5s (decode + VQ)."""
    decode, _, Z, _ = mechanism_gaussian(emb, centers, 0.5 * scale, master_seed)
    z_decode = decode.astype(int)
    z_vq = vq_labels(np.asarray(Z))
    rows = []
    for attr, y in yattrs.items():
        for view, zv in (("decode", z_decode), ("vq", z_vq)):
            r = {"mechanism": "gaussian 0.5s", "attr": attr, "view": view,
                 "alphabet": int(zv.max()) + 1,
                 "ML": max_leakage(zv, y),
                 "IL_inf": min_entropy_leakage(zv, y)}
            r["ML_ci"] = bootstrap_ci(max_leakage, zv, y)
            r["IL_inf_ci"] = bootstrap_ci(min_entropy_leakage, zv, y)
            r.update(ml_conditional(zv, y, theta))
            r["IL_inf_cond"] = il_inf_conditional(zv, y, theta)
            pct, worst = advantage_percentiles(zv, y)
            r["adv_p10"], r["adv_p50"], r["adv_p90"] = \
                pct[10], pct[50], pct[90]
            r["adv_worst_user"] = worst
            rows.append(r)
            print({k: (round(v, 4) if isinstance(v, float) else v)
                   for k, v in r.items()}, flush=True)
    pd.DataFrame(rows).to_csv(f"{RES}/e17_gaussian05_worstcase.csv",
                              index=False)


if __name__ == "__main__":
    ds, emb, theta, centers, yattrs, scale, mech = setup()
    per_seed = part_a_b(ds, emb, theta, centers, yattrs, mech)
    part_c(per_seed)
    part_d(emb, centers, yattrs, scale)
    part_e(emb, centers, yattrs, scale)
    part_f(emb, theta, centers, yattrs, scale)
    print("e17 done", flush=True)
