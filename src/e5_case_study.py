"""E5: end-to-end design-rule case study on real data.

The manuscript's rule (integer-safe, revision 2): target worst-case excess
risk eps gives q = floor(sqrt(eps)/Delta) + 1, K_req = ceil(M / q),
alpha_req = log K_req.

Rationale: the grouping bound is Delta^2 (ceil(M/K) - 1)^2, so the target
eps requires ceil(M/K) - 1 <= sqrt(eps)/Delta.  Because of the inner
ceiling this needs the FLOOR of sqrt(eps)/Delta in the denominator: with
K = ceil(M / (1 + sqrt(eps)/Delta)) the counterexample M=128, eps=0.01,
Delta=1 gives K=117 yet ceil(128/117)=2 and bound (2-1)^2 = 1 >> eps
(the rounding was lost when compressing the journal rule; restored after
review round 2, matching the conference-version fix).

On real data we estimate Delta from the measured excess risk at a reference
K0 (Delta^2 = excess(K0) / (ceil(M/K0)-1)^2), predict excess risk at other
K, and check (a) prediction quality and (b) CONSERVATIVENESS (predicted >=
measured), then report the deployed leakage at the certified operating
point.
"""
from __future__ import annotations

import os
import sys
import math
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import (load_dataset, attrs_of, build_user_side, cluster_types,
                      balanced_group_partition, empirical_mi, run_attack)
from e1e3_sweep import RatingUtility

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
RES = os.path.join(os.path.dirname(__file__), "..", "results")


def theory_excess(M, K, Delta):
    return Delta ** 2 * (math.ceil(M / K) - 1) ** 2


def main(dataset_name="ml-1m", M=128, master_seed=7):
    ds = load_dataset(dataset_name)
    emb, svd, idx_i, top = build_user_side(ds, top_items=500, svd_dim=32,
                                           seed=master_seed)
    theta, centers = cluster_types(emb, M, seed=master_seed)
    util = RatingUtility(ds, idx_i, top, test_frac=0.2, seed=master_seed)
    yattrs = {a: ds.users[a].values for a in attrs_of(ds)}
    mse_oracle = util.mse(*util.models_from_decode(theta))   # server w/ theta

    def excess(K):
        user_g, _, _ = balanced_group_partition(theta, centers, K, master_seed)
        mus, cvecs = util.models_from_decode(user_g)
        return util.mse(mus, cvecs) - mse_oracle

    # effective cardinality (prior-sensitive rule from the manuscript:
    # replace log M by H(theta) for nonuniform populations)
    counts = np.bincount(theta, minlength=M)
    p = counts / counts.sum()
    H = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    M_eff = float(np.exp(H))

    # calibrate Delta at reference K0 = M/4 (mid-range, stable estimate)
    K0 = M // 4
    e_K0 = excess(K0)
    Delta2 = e_K0 / (math.ceil(M / K0) - 1) ** 2
    Delta = math.sqrt(Delta2)
    e_base = excess(1)                      # population spread (K=1)

    rows = []
    for K in [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 128]:
        if K > M:
            continue
        e_meas = excess(K)
        rows.append({"K": K, "alpha": math.log(K), "excess_measured": e_meas,
                     "excess_predicted": theory_excess(M, K, Delta),
                     "excess_predicted_eff": theory_excess(M_eff, K, Delta)})
    cal = pd.DataFrame(rows)
    cal["conservative"] = cal["excess_predicted"] >= cal["excess_measured"]

    # certified operating points: relative utility targets
    cases = []
    for eps_rel in [0.5, 0.25, 0.1, 0.05]:
        eps = eps_rel * e_base
        for variant, Mstar in (("M", M), ("M_eff", M_eff)):
            # integer-safe rule: q = floor(sqrt(eps)/Delta) + 1 (see module
            # docstring); guarantees ceil(M/K_req) <= q hence
            # excess <= Delta^2 (q-1)^2 <= eps
            q = math.floor(math.sqrt(eps) / Delta) + 1
            K_req = math.ceil(Mstar / q)
            K_req = max(1, min(M, K_req))
            e_meas = excess(K_req)
            user_g, _, _ = balanced_group_partition(theta, centers, K_req,
                                                    master_seed)
            Z = np.eye(K_req)[user_g]
            case = {"target_eps_rel": eps_rel, "target_eps": eps,
                    "variant": variant, "Mstar": Mstar, "Delta": Delta,
                    "K_req": K_req, "alpha_req": math.log(K_req),
                    "excess_measured": e_meas,
                    "certified": bool(e_meas <= eps)}
            for attr, y in yattrs.items():
                best, auc = -1.0, float("nan")
                for kind in ("mlp", "lr"):
                    a = run_attack(Z, y, seeds=tuple(range(10)), folds=5,
                                   attackers=(kind,))
                    d = a.get("adv", (float("nan"),))[0]
                    if np.isfinite(d) and d > best:
                        best, auc = d, a.get("auc", (float("nan"),))[0]
                case[f"{attr}_adv"] = best
                case[f"{attr}_auc"] = auc
                case[f"{attr}_I_AZ"] = empirical_mi(user_g, y)
            cases.append(case)
        print(cases[-1], flush=True)

    os.makedirs(RES, exist_ok=True)
    cal.to_csv(f"{RES}/e5_calibration_{dataset_name}.csv", index=False)
    pd.DataFrame(cases).to_csv(f"{RES}/e5_case_study_{dataset_name}.csv",
                               index=False)
    print("saved E5 results")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ml-1m",
         int(sys.argv[2]) if len(sys.argv) > 2 else 128)
