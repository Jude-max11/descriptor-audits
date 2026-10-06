"""E11: attribute leakage under repeated release (composition experiment).

Proposition 2 (attribute-side composition) states: (i) per-release bounds
do not compose (one-time-pad-style counterexample), (ii) conditionally
i.i.d. randomized releases accumulate linearly, (iii) theta-measurable
descriptors are repetition-stable.  All deployment experiments in the
manuscript release the descriptor once; this experiment closes that gap.

Design (ml-1m, M=128, master seed 7, five-fold CV x 5 attack seeds, class-balanced
LR -- the stronger attacker on one-hot type-level features), at LOW-RATE
operating points where single-release leakage sits below the type--attribute
ceiling (the near-full-rate matched points are already ceiling-capped):
  mechanism x release regime, two releases Z1, Z2:
    btq K=13        repeat      Z2 = Z1 (theta-measurable by construction)
    grr eps=2       independent fresh randomized-response draw per release
                    memoized    the same draw reused (memoization)
    brappor eps=4   independent fresh permanent RR per release (no memoize)
                    memoized    the same permanent bit vector reused
    gaussian .37s   independent fresh Gaussian noise per release
                    memoized    the same noise draw reused
  For k in {1,2}:
    adv      trained gray-box attack advantage on [Z1] vs [Z1 Z2]
    I_A      permutation-corrected I(A; decoded label(s)), decode view of
             each mechanism (released label / LLR decode / nearest center)
    L_res    permutation-corrected conditional ML mean within type (e8
             machinery) on the (pair-)decoded labels

Expected: independent regimes grow from k=1 to k=2 (ii); memoized and
type-level deterministic regimes stay flat ((iii); the deployment meaning
of RAPPOR memoization).  The (i) counterexample remains analytic.

Output: results/e11_composition_<ds>.csv
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import (load_dataset, attrs_of, build_user_side, cluster_types,
                      mechanism_grouping, mechanism_grr, mechanism_brappor,
                      mechanism_gaussian, empirical_mi, run_attack)
from e8_worstcase import ml_conditional

RES = os.path.join(os.path.dirname(__file__), "..", "results")
ATTACK_SEEDS = (0, 1, 2, 3, 4)


def build_configs(emb, theta, centers, M, master_seed):
    """{label: regime -> (Z1, Z2, decode1, decode2)} with native decoders."""
    scale = float(np.std(emb, axis=0).mean())
    out = {}
    # LOW-RATE tier: single-release leakage sits below the type--attribute
    # ceiling, so accumulation is observable (at the near-full-rate matched
    # points the type-level channels are already ceiling-capped)
    lab, _, Z = mechanism_grouping(theta, centers, 13, master_seed)
    out[("btq K=13", "repeat")] = (Z, Z, lab, lab)
    for mech_name, fn, param in (("grr eps=2", mechanism_grr, 2),
                                 ("brappor eps=4", mechanism_brappor, 4)):
        d1, _, z1, _ = fn(theta, M, param, master_seed)
        d2, _, z2, _ = fn(theta, M, param, master_seed + 1000)
        out[(mech_name, "independent")] = (z1, z2, d1, d2)
        out[(mech_name, "memoized")] = (z1, z1, d1, d1)
    # embedding-level Gaussian: independent vs shared noise
    d1, _, z1, _ = mechanism_gaussian(emb, centers, 0.37 * scale, master_seed)
    d2, _, z2, _ = mechanism_gaussian(emb, centers, 0.37 * scale,
                                      master_seed + 1000)
    out[("gaussian 0.37s", "independent")] = (z1, z2, d1, d2)
    out[("gaussian 0.37s", "memoized")] = (z1, z1, d1, d1)
    return out


def one_attack(task):
    label, regime, feats, k, attr, y = task
    res = run_attack(feats, y, seeds=ATTACK_SEEDS, folds=5,
                     attackers=("lr",))
    adv_m, adv_s = res["adv"]
    return {"mechanism": label, "regime": regime, "k": k, "attr": attr,
            "adv_mean": adv_m, "adv_std": adv_s}


def main(dataset_name="ml-1m", M=128, master_seed=7):
    ds = load_dataset(dataset_name)
    emb, svd, idx_i, top = build_user_side(ds, top_items=500, svd_dim=32,
                                           seed=master_seed)
    theta, centers = cluster_types(emb, M, seed=master_seed)
    yattrs = {a: ds.users[a].values for a in attrs_of(ds)}
    configs = build_configs(emb, theta, centers, M, master_seed)

    tasks, info_rows = [], []
    for (label, regime), (Z1, Z2, d1, d2) in configs.items():
        pair = np.asarray(d1) * M + np.asarray(d2)   # joint decode cell
        for attr, y in yattrs.items():
            for k, lab in ((1, np.asarray(d1)), (2, pair)):
                info_rows.append({
                    "mechanism": label, "regime": regime, "k": k,
                    "attr": attr,
                    "I_A_corr": empirical_mi(lab, y, n_perm=20,
                                             seed=master_seed),
                    "L_res": ml_conditional(
                        lab, y, theta)["ML_cond_corr_mean_nonneg"],
                })
            for k, feats in ((1, np.asarray(Z1)),
                             (2, np.hstack([Z1, Z2]))):
                tasks.append((label, regime, feats, k, attr, y))

    with ProcessPoolExecutor(max_workers=10) as ex:
        atk_rows = list(ex.map(one_attack, tasks))

    df = pd.merge(pd.DataFrame(info_rows), pd.DataFrame(atk_rows),
                  on=["mechanism", "regime", "k", "attr"])
    os.makedirs(RES, exist_ok=True)
    path = f"{RES}/e11_composition_{dataset_name}.csv"
    df.to_csv(path, index=False)
    print("saved", path)
    print(df.pivot_table(index=["mechanism", "regime", "attr"],
                         columns="k", values=["adv_mean", "I_A_corr"])
          .to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ml-1m",
         int(sys.argv[2]) if len(sys.argv) > 2 else 128)
