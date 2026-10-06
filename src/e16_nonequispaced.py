"""E16: constant-factor matching beyond the equally spaced family (R2-M2).

The manuscript proves matching converses/achievability (BTQ within constants
of the Fano/entropy-distortion converse) for the equally spaced linear
regression family under a uniform prior.  This script checks, numerically and
exactly (no sampling), how the constant behaves when the family assumptions
are relaxed:

  scenarios
    equispaced-uniform : baseline (positions w_m = m, uniform prior)
    lognormal-<s>       : spacings are lognormal(s), normalized (3 seeds)
    twoblock            : half the spacings delta, half 3*delta
    zipf                : equispaced positions, Zipf(0.8) prior
    zipf+lognormal      : both perturbations combined

  constructions at group count K in {8, 16, 32}, M = 128 types
    BTQ       : balanced-mass contiguous partition (equal prior mass per
                group, boundaries at prior quantiles) -- the manuscript's
                balanced type quantization generalized beyond equal spacing
    DP-opt    : variance-optimal contiguous partition (dynamic programming)
                -- the best contiguous K-level release
    converse  : packing-style worst-case lower bound
                ((ceil(M/K)-1) * delta_min / 2)^2, where delta_min is the
                minimum spacing -- the quantity the Fano/packing converse
                controls

  risk of a K-level release: sum_g P(g) * E[(w - mean_g)^2 | g] under the
  scenario prior (exact arithmetic).

  Readout: the ratio risk_BTQ / risk_DPopt stays bounded by a small constant
  (it is 1 in the equispaced-uniform baseline) and risk_BTQ / converse
  remains within a constant of the baseline value across scenarios.

Output: results/e16_nonequispaced.csv
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

RES = os.path.join(os.path.dirname(__file__), "..", "results")
M = 128
KS = (8, 16, 32)


def positions(scenario, seed=0):
    rng = np.random.default_rng(seed)
    if scenario == "equispaced-uniform":
        spac = np.ones(M - 1)
        prior = np.ones(M) / M
    elif scenario.startswith("lognormal"):
        s = float(scenario.split("-")[1])
        spac = np.exp(s * rng.standard_normal(M - 1))
        prior = np.ones(M) / M
    elif scenario == "twoblock":
        spac = np.concatenate([np.ones(M // 2), 3 * np.ones(M // 2 - 1)])
        prior = np.ones(M) / M
    elif scenario == "zipf":
        spac = np.ones(M - 1)
        prior = 1.0 / np.arange(1, M + 1) ** 0.8
        prior /= prior.sum()
    elif scenario == "zipf+lognormal":
        spac = np.exp(0.5 * rng.standard_normal(M - 1))
        prior = 1.0 / np.arange(1, M + 1) ** 0.8
        prior /= prior.sum()
    else:
        raise ValueError(scenario)
    w = np.concatenate([[0.0], np.cumsum(spac)])
    return w, prior


def risk(w, prior, bounds):
    """Exact prior-averaged risk of a contiguous partition."""
    total = 0.0
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        p = prior[lo:hi].sum()
        if p <= 0:
            continue
        mu = (w[lo:hi] * prior[lo:hi]).sum() / p
        total += ((w[lo:hi] - mu) ** 2 * prior[lo:hi]).sum()
    return total


def risk_wc(w, prior, bounds):
    """Worst-case (sup over types) risk of a contiguous partition, the
    estimand of the manuscript's Theorems 8/13: sup_theta (mean_g - w)^2."""
    worst = 0.0
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        p = prior[lo:hi].sum()
        mu = (w[lo:hi] * prior[lo:hi]).sum() / p if p > 0 else w[lo:hi].mean()
        worst = max(worst, float(np.max((w[lo:hi] - mu) ** 2)))
    return worst


def btq_bounds(prior, K):
    """Balanced-mass contiguous partition: boundaries at prior quantiles."""
    csum = np.concatenate([[0.0], np.cumsum(prior)])
    bounds = [0]
    for k in range(1, K):
        target = k / K
        b = int(np.searchsorted(csum, target - 1e-12)) + 1
        b = max(b, bounds[-1] + 1)
        bounds.append(min(b, M - (K - k)))
    bounds.append(M)
    return bounds


def dp_bounds(w, prior, K):
    """Variance-optimal contiguous partition via DP (O(M^2 K))."""
    c1 = np.concatenate([[0.0], np.cumsum(w * prior)])
    c2 = np.concatenate([[0.0], np.cumsum(w * w * prior)])
    INF = np.inf
    cost = np.full((M + 1, K + 1), INF)
    arg = np.zeros((M + 1, K + 1), dtype=int)
    cost[0, 0] = 0.0
    for k in range(1, K + 1):
        for i in range(k, M + 1):
            for j in range(k - 1, i):
                if cost[j, k - 1] == INF:
                    continue
                p = c1[i] - c1[j]
                if p <= 0:
                    c = 0.0
                else:
                    c = ((c2[i] - c2[j]) - p * p / (prior[j:i].sum()))
                val = cost[j, k - 1] + c
                if val < cost[i, k]:
                    cost[i, k] = val
                    arg[i, k] = j
    bounds = [M]
    i, k = M, K
    while k > 0:
        j = arg[i, k]
        bounds.append(j)
        i, k = j, k - 1
    return sorted(bounds)


def converse(w, K):
    dmin = float(np.diff(w).min())
    gap = int(np.ceil(M / K)) - 1
    return (gap * dmin / 2.0) ** 2


def main():
    rows = []
    scenarios = (["equispaced-uniform", "lognormal-0.5", "lognormal-1.0",
                  "twoblock", "zipf", "zipf+lognormal"])
    for sc in scenarios:
        seeds = (0, 1, 2) if sc.startswith("lognormal") or sc == "zipf+lognormal" else (0,)
        for sd in seeds:
            w, prior = positions(sc, sd)
            for K in KS:
                b_btq = btq_bounds(prior, K)
                b_dp = dp_bounds(w, prior, K)
                r_btq = risk(w, prior, b_btq)
                r_dp = risk(w, prior, b_dp)
                wc_btq = risk_wc(w, prior, b_btq)
                wc_dp = risk_wc(w, prior, b_dp)
                cv = converse(w, K)
                rows.append({
                    "scenario": sc, "seed": sd, "M": M, "K": K,
                    "risk_btq": r_btq, "risk_dp": r_dp, "converse": cv,
                    "risk_btq_wc": wc_btq, "risk_dp_wc": wc_dp,
                    "ratio_btq_dp": r_btq / r_dp if r_dp > 0 else np.inf,
                    "ratio_btq_wc": (wc_btq / wc_dp) if wc_dp > 0 else np.inf,
                    "ratio_btq_converse": (r_btq / cv) if cv > 0 else np.inf,
                    "ratio_wc_converse": (wc_btq / cv) if cv > 0 else np.inf,
                    "spacing_ratio": float(np.diff(w).max()
                                           / np.diff(w).min()),
                    "prior_eff": float(np.exp(-np.sum(prior
                                                      * np.log(prior)))),
                })
                print(f"{sc} s{sd} K={K}: btq={r_btq:.4f} dp={r_dp:.4f} "
                      f"cv={cv:.4f} ratio={r_btq / max(r_dp, 1e-12):.2f} "
                      f"wc: {wc_btq:.3f}/{wc_dp:.3f}", flush=True)
    os.makedirs(RES, exist_ok=True)
    path = f"{RES}/e16_nonequispaced.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print("saved", path)


if __name__ == "__main__":
    main()
