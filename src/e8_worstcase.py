"""E8: worst-case leakage metrics for the matched operating points.

Complements the average-case reporting (I(A;Z|theta), guessing advantage)
with worst-case quantities requested in review:

  ML(A;Z)      maximal leakage (Issa-Kamath-Wagner closed form for finite
               alphabets): log sum_z max_a P(z|a)
  IL_inf       min-entropy leakage (Smith 2009):
               log[ sum_z max_a P(a,z) / max_a P(a) ]
  ML_cond      per-type conditional maximal leakage; because per-type
               samples are small (alphabet >> n), reported as the
               permutation-corrected mean and the standardized excess
               (raw - null_mean)/null_std across types
  IL_inf_cond  conditional min-entropy leakage via aggregated vulnerability,
               permutation-corrected within type
  adv p10/50/90  per-user posterior advantage percentiles on the same
               alphabet (promoted from E4 to a formal security output)

Type-level mechanisms (grouping, GRR, bRAPPOR) release a channel that does
not depend on A given theta, so their conditional worst-case leakage is
exactly zero by construction; the standardized-excess reporting verifies
this empirically at the available per-type sample sizes.  Continuous
mechanisms (Gaussian, PCA) are evaluated on two finite alphabets:
(i) the decoded type label (white-box observable) and (ii) a vector
quantization of the released vector at 89 bins (captures within-type
signal; a lower bound on the continuous-channel leakage by the
data-processing inequality).
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import (load_dataset, attrs_of, build_user_side, cluster_types,
                      mechanism_grouping, mechanism_grr, mechanism_brappor,
                      mechanism_gaussian, mechanism_pca)

RES = os.path.join(os.path.dirname(__file__), "..", "results")
VQ_BINS = 89
MIN_TYPE = 10          # types with fewer users are skipped in conditional ML


def max_leakage(z, a):
    """ML(A;Z) = log sum_z max_a P(z|a), plugin estimate (nats)."""
    z = np.asarray(z)
    a = np.asarray(a)
    zu, zi = np.unique(z, return_inverse=True)
    au, ai = np.unique(a, return_inverse=True)
    za = np.zeros((len(zu), len(au)))
    np.add.at(za, (zi, ai), 1.0)
    cond = za / za.sum(0, keepdims=True)          # rows=z, cols=a -> P(z|a)
    return float(np.log(max(cond.max(1).sum(), 1e-12)))


def min_entropy_leakage(z, a):
    """IL_inf(A;Z) = log[ sum_z max_a P(a,z) / max_a P(a) ] (nats)."""
    za = np.zeros((int(z.max()) + 1, int(a.max()) + 1))
    np.add.at(za, (z, a), 1.0)
    pza = za / za.sum()
    vuln = float(pza.max(1).sum())
    prior_vuln = float((pza.sum(0)).max())
    return float(np.log(vuln / prior_vuln))


def _shuffle_within(a, theta, rng):
    out = np.array(a)
    for t in np.unique(theta):
        m = theta == t
        out[m] = rng.permutation(np.asarray(a)[m])
    return out


def ml_conditional(z, a, theta, seed=0, n_perm=50):
    """Per-type conditional ML with permutation null.

    The plugin ML at alphabet >> per-type n is strongly upward biased, so we
    report (a) the raw mean and sup across types, (b) the permutation-
    corrected mean (truncated at zero for the residual reading), and (c) the
    number of types whose |standardized excess| exceeds 2, together with the
    Benjamini-Hochberg FDR count at q=0.05.  A channel that is
    A-independent given theta has true conditional ML exactly zero and
    should show |z| within noise and no FDR rejections.
    """
    rng = np.random.default_rng(seed)
    raws, nulls, stds = [], [], []
    for t in np.unique(theta):
        m = theta == t
        if m.sum() < MIN_TYPE or len(np.unique(a[m])) < 2:
            continue
        zt, at = z[m], a[m]
        raw = max_leakage(zt, at)
        null = [max_leakage(zt, rng.permutation(at)) for _ in range(n_perm)]
        raws.append(raw)
        nulls.append(float(np.mean(null)))
        stds.append(float(np.std(null)) + 1e-9)
    raws, nulls, stds = map(np.array, (raws, nulls, stds))
    excess = (raws - nulls) / stds
    # Benjamini-Hochberg at q=0.05 on two-sided |z| p-values (normal approx)
    from scipy.stats import norm
    p = 2 * (1 - norm.cdf(np.abs(excess)))
    order = np.argsort(p)
    ranked = p[order]
    mtests = len(p)
    thresh = np.arange(1, mtests + 1) * 0.05 / mtests
    rej = 0
    for i in range(mtests - 1, -1, -1):
        if ranked[i] <= thresh[i]:
            rej = i + 1
            break
    return {"ML_cond_raw_mean": float(np.mean(raws)),
            "ML_cond_raw_sup": float(np.max(raws)),
            "ML_cond_corr_mean": float(np.mean(raws - nulls)),
            "ML_cond_corr_mean_nonneg": float(max(np.mean(raws - nulls), 0.0)),
            "ML_cond_zmax": float(np.max(np.abs(excess))),
            "ML_cond_n_over2": int((np.abs(excess) > 2).sum()),
            "ML_cond_n_fdr05": int(rej),
            "ML_cond_n_types": int(len(raws))}


def bootstrap_ci(fn, z, a, n_boot=200, seed=0):
    """Percentile bootstrap CI (2.5/97.5) over users for a statistic fn."""
    rng = np.random.default_rng(seed)
    vals = []
    n = len(z)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(fn(z[idx], a[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def il_inf_conditional(z, a, theta, seed=0, n_perm=50):
    """log[ sum_theta p(theta) sum_z max_a P(a,z|theta) / prior vuln ],
    permutation-corrected (null: attribute shuffled within type)."""
    def value(z, a, theta):
        num, den, tot = 0.0, 0.0, len(theta)
        for t in np.unique(theta):
            m = theta == t
            if m.sum() < MIN_TYPE or len(np.unique(a[m])) < 2:
                continue
            za = np.zeros((int(z[m].max()) + 1, int(a[m].max()) + 1))
            np.add.at(za, (z[m], a[m]), 1.0)
            pza = za / m.sum()
            num += (m.sum() / tot) * float(pza.max(1).sum())
            den += (m.sum() / tot) * float(pza.sum(0).max())
        return float(np.log(num / den)) if num > 0 else 0.0

    rng = np.random.default_rng(seed)
    raw = value(z, a, theta)
    null = float(np.mean([
        value(z, _shuffle_within(a, theta, rng), theta)
        for _ in range(n_perm)]))
    return raw - null


def advantage_percentiles(z, a):
    """Per-user posterior advantage percentiles (10/50/90) on alphabet z."""
    za = np.zeros((int(z.max()) + 1, int(a.max()) + 1))
    np.add.at(za, (z, a), 1.0)
    cnt = za.sum(1)
    post = np.zeros_like(za)
    ok = cnt > 0
    post[ok] = za[ok] / cnt[ok, None]                   # P(a|z)
    prior_best = float(za.sum(0).max() / len(z))
    adv_z = np.where(ok, post.max(1) - prior_best, 0.0)
    per_user = adv_z[z]
    return {p: float(np.percentile(per_user, p)) for p in (10, 50, 90)}, \
        float(adv_z[ok].max())


def vq_labels(Z, n_bins=VQ_BINS, seed=0):
    """Vector-quantize released vectors to n_bins labels (k-means++)."""
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_bins, n_init=4, random_state=seed).fit(Z)
    return km.labels_.astype(int)


def onehot_argmax(Z):
    return Z.argmax(1).astype(int)


def main(dataset_name="ml-1m", M=128, master_seed=7):
    ds = load_dataset(dataset_name)
    emb, svd, idx_i, top = build_user_side(ds, top_items=500, svd_dim=32,
                                           seed=master_seed)
    theta, centers = cluster_types(emb, M, seed=master_seed)
    yattrs = {a: ds.users[a].values for a in attrs_of(ds)}
    scale = float(np.std(emb, axis=0).mean())

    # matched operating points of the manuscript's matched-utility table
    mech = {
        ("grouping", "K=89"): mechanism_grouping(theta, centers, 89, master_seed),
        ("grr", "eps=12"): mechanism_grr(theta, M, 12, master_seed),
        ("brappor", "eps=12"): mechanism_brappor(theta, M, 12, master_seed),
        ("gaussian", "sigma=0.37s"): mechanism_gaussian(
            emb, centers, 0.37 * scale, master_seed),
        ("pca", "d'=8"): mechanism_pca(emb, centers, 8, master_seed),
    }

    rows = []
    for (name, param), out in mech.items():
        decode = out[0]
        Z = out[2]
        if name in ("grouping", "grr"):
            z_decode = onehot_argmax(Z)          # released label (== decode)
        else:
            z_decode = decode.astype(int)        # white-box decoded type
        z_vq = vq_labels(np.asarray(Z)) if name in ("brappor", "gaussian",
                                                    "pca") else None
        for attr, y in yattrs.items():
            for view, zv in (("decode", z_decode), ("vq", z_vq)):
                if zv is None:
                    continue
                r = {"mechanism": name, "param": param, "attr": attr,
                     "view": view, "alphabet": int(zv.max()) + 1,
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

    os.makedirs(RES, exist_ok=True)
    path = f"{RES}/e8_worstcase_{dataset_name}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print("saved", path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ml-1m",
         int(sys.argv[2]) if len(sys.argv) > 2 else 128)
