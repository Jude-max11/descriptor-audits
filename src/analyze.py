"""Analysis and figures for the C&S supplementary experiments.

Reads results/*.csv and produces (into figures/ and results/):
  fig_frontier_<ds>.pdf/png : utility (excess risk) vs attribute-attack
                              advantage per mechanism  [E1/E3 headline]
  fig_info_attack_<ds>.pdf : attack advantage vs released information (nats)
  fig_e2_<ds>.pdf          : strong/weak P_theta vs alpha + gap-vs-n
  fig_e4_<ds>.pdf          : percentile exposure + correlated leakage
  summary_report.md        : consolidated tables + narrative numbers
"""
from __future__ import annotations

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")

MECH_STYLE = {
    "grouping": dict(color="#1a6faf", marker="o", label="BTQ (this work)"),
    "grr": dict(color="#d62728", marker="s", label="GRR (CCS'14)"),
    "brappor": dict(color="#9467bd", marker="D", label="bRAPPOR (CCS'14)"),
    "gaussian": dict(color="#ff7f0e", marker="^", label="Gaussian mech."),
    "pca": dict(color="#2ca02c", marker="v", label="PCA projection"),
}


def _sort_pos(ax):
    ax.grid(alpha=0.3)
    ax.x_label = None


def fig_frontier(df, ds):
    attrs = [a for a in ("gender", "age3") if f"{a}_adv" in df.columns]
    titles = {"gender": "gender", "age3": "age (3-class)"}
    fig, axes = plt.subplots(1, max(len(attrs), 1), figsize=(5 * len(attrs), 4))
    axes = np.atleast_1d(axes)
    for ax, attr, lab in zip(axes, attrs, ["(a)", "(b)"]):
        for mech, st in MECH_STYLE.items():
            sub = df[df["mechanism"] == mech].sort_values("excess_risk")
            if not len(sub):
                continue
            ax.plot(sub["excess_risk"], sub[f"{attr}_adv"],
                    **{k: v for k, v in st.items() if k != "label"},
                    label=st["label"])
        ax.set_xlabel("excess risk vs oracle")
        ax.set_ylabel("attack advantage (BAcc - random)")
        ax.set_title(f"{lab} attribute inference vs utility: {titles[attr]}",
                     fontsize=10)
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle(f"Utility-leakage frontier, {ds}")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig_frontier_{ds}.png", dpi=200)
    fig.savefig(f"{FIG}/fig_frontier_{ds}.pdf")
    plt.close(fig)


def fig_info_attack(df, ds, ceil_g, ceil_a):
    attrs = [a for a in ("gender", "age3") if f"{a}_I_A" in df.columns]
    ceils = dict(gender=ceil_g, age3=ceil_a)
    fig, axes = plt.subplots(1, max(len(attrs), 1), figsize=(5 * len(attrs), 4))
    axes = np.atleast_1d(axes)
    for mech, st in MECH_STYLE.items():
        sub = df[df["mechanism"] == mech].sort_values("alpha_emp")
        if not len(sub):
            continue
        for ax, attr in zip(axes, attrs):
            ax.plot(sub["alpha_emp"], sub[f"{attr}_I_A"],
                    **{k: v for k, v in st.items() if k != "label"},
                    label=st["label"])
    for ax, attr, lab in zip(axes, attrs, ["(a)", "(b)"]):
        ceil = ceils[attr]
        ax.set_xlabel("released task information I(theta;Z) (nats)")
        ax.set_ylabel(f"I(A;Z) (nats)")
        if np.isfinite(ceil):
            ax.axhline(ceil, ls="--", c="k", lw=1,
                       label="I(A;theta) ceiling (type-level Z)")
        ax.set_title(f"{lab} leakage vs budget: {attr}", fontsize=10)
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle(f"Information-budget view, {ds}")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig_info_attack_{ds}.png", dpi=200)
    fig.savefig(f"{FIG}/fig_info_attack_{ds}.pdf")
    plt.close(fig)


def fig_e2(ds, M=128):
    t = pd.read_csv(f"{RES}/e2_task_inference_{ds}.csv")
    g = pd.read_csv(f"{RES}/e2_gap_vs_n_{ds}.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(t["alpha"], t["P_theta_strong"], "o-", label="strong (train-time Z)")
    axes[0].plot(t["alpha"], t["P_theta_weak"], "s--", label="weak (fresh Z)")
    # Fano-style converse overlay: P_e >= (H - I - 1)/log M
    H = float(np.log(M))
    alphas = np.linspace(0, np.log(128), 100)
    fano = np.maximum(0.0, (H - alphas - 1) / np.log(128))
    axes[0].plot(alphas, fano, "k:", label="Fano converse (uniform prior)")
    axes[0].set_xlabel("alpha = log K (nats)"); axes[0].set_ylabel("P(task id error)")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[0].set_title("task identification: strong vs weak")
    for K, sub in g.groupby("K"):
        axes[1].plot(sub["frac"], sub["P_theta"], "o-", label=f"K={K}")
    axes[1].set_xlabel("fraction n of local ratings used")
    axes[1].set_ylabel("P(task id error)")
    axes[1].set_title("finite-sample gap: saturation at rate limit")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_e2_{ds}.png", dpi=200)
    fig.savefig(f"{FIG}/fig_e2_{ds}.pdf"); plt.close(fig)


def fig_e4(ds):
    d = pd.read_csv(f"{RES}/e4_heterogeneity_{ds}.csv")
    attr = "gender" if "gender_adv_p10" in d else "age3"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(d["alpha"], d[f"{attr}_adv_p10"], "v--", label="p10")
    axes[0].plot(d["alpha"], d[f"{attr}_adv_p50"], "o-", label="p50")
    axes[0].plot(d["alpha"], d[f"{attr}_adv_p90"], "^--", label="p90")
    axes[0].set_xlabel("alpha = log K"); axes[0].set_ylabel(f"per-user advantage ({attr})")
    axes[0].set_title("exposure percentiles across users"); axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    if "gender_I_AZ" in d:
        axes[1].plot(d["alpha"], d["gender_I_AZ"], "o-", label="I(gender;Z)")
        axes[1].axhline(d["gender_I_thetaA"].iloc[0], ls="--", c="k", lw=1,
                        label="I(gender;theta) ceiling")
    axes[1].plot(d["alpha"], d["age3_I_AZ"], "s-", label="I(age;Z)")
    axes[1].axhline(d["age3_I_thetaA"].iloc[0], ls=":", c="r", lw=1,
                    label="I(age;theta) ceiling")
    axes[1].set_xlabel("alpha = log K"); axes[1].set_ylabel("indirect leakage (nats)")
    axes[1].set_title("correlated-attribute leakage vs budget")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_e4_{ds}.png", dpi=200)
    fig.savefig(f"{FIG}/fig_e4_{ds}.pdf"); plt.close(fig)


def fig_e6(ds):
    """Ablation figures: attacker depth, shadow size, granularity."""
    have = any(os.path.exists(f"{RES}/e6_abl{sec}_{ds}.csv")
               for sec in ("A_depth", "A_depth_balanced", "B_shadow",
                           "C_granularity"))
    if not have:
        return
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    p = f"{RES}/e6_ablA_depth_balanced_{ds}.csv"
    if not os.path.exists(p):
        p = f"{RES}/e6_ablA_depth_{ds}.csv"
    if os.path.exists(p):
        d = pd.read_csv(p)
        for (K, attr), sub in d.groupby(["K", "attr"]):
            axes[0, 0].errorbar(sub["depth"], sub["adv_mean"],
                                 yerr=sub["adv_std"], marker="o",
                                 label=f"K={K},{attr}")
        axes[0, 0].set_xlabel("MLP hidden layers")
        axes[0, 0].set_ylabel("advantage")
        axes[0, 0].set_title("(a) attacker capacity (balanced MLP)")
        axes[0, 0].legend(fontsize=7); axes[0, 0].grid(alpha=0.3)
    p = f"{RES}/e6_ablB_shadow_{ds}.csv"
    if os.path.exists(p):
        d = pd.read_csv(p)
        for (K, attr), sub in d.groupby(["K", "attr"]):
            axes[0, 1].errorbar(sub["frac"], sub["adv_mean"],
                                 yerr=sub["adv_std"], marker="s",
                                 label=f"K={K},{attr}")
        axes[0, 1].set_xlabel("shadow-set fraction")
        axes[0, 1].set_ylabel("advantage")
        axes[0, 1].set_title("(b) shadow data availability")
        axes[0, 1].legend(fontsize=7); axes[0, 1].grid(alpha=0.3)
    p = f"{RES}/e6_ablC_granularity_{ds}.csv"
    if os.path.exists(p):
        d = pd.read_csv(p)
        for Mv, sub in d.groupby("M"):
            sub = sub.sort_values("alpha_emp")
            axes[1, 0].plot(sub["alpha_emp"], sub["excess"], "o-",
                            label=f"M={Mv}")
            axes[1, 1].plot(sub["alpha_emp"], sub["gender_adv"], "s--",
                            label=f"M={Mv}")
        axes[1, 0].set_xlabel("released info (nats)")
        axes[1, 0].set_ylabel("excess risk"); axes[1, 0].set_yscale("log")
        axes[1, 0].set_title("(c) granularity: utility")
        axes[1, 0].legend(fontsize=7); axes[1, 0].grid(alpha=0.3)
        axes[1, 1].set_xlabel("released info (nats)")
        axes[1, 1].set_ylabel("gender advantage")
        axes[1, 1].set_title("(d) granularity: leakage")
        axes[1, 1].legend(fontsize=7); axes[1, 1].grid(alpha=0.3)
    fig.suptitle(f"Ablations, {ds}")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig_e6_ablations_{ds}.png", dpi=200)
    fig.savefig(f"{FIG}/fig_e6_ablations_{ds}.pdf")
    plt.close(fig)


def fig_e7(ds):
    p = f"{RES}/e7_ranking_{ds}.csv"
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for mech, st in MECH_STYLE.items():
        sub = d[d["mech"] == mech].sort_values("alpha_emp")
        if not len(sub):
            continue
        axes[0].plot(sub["alpha_emp"], sub["HR@10"],
                     **{k: v for k, v in st.items() if k != "label"},
                     label=st["label"])
        axes[1].plot(sub["alpha_emp"], sub["NDCG@10"],
                     **{k: v for k, v in st.items() if k != "label"},
                     label=st["label"])
    orc = d[d["mech"] == "oracle"]
    if len(orc):
        axes[0].axhline(orc["HR@10"].iloc[0], ls="--", c="k", lw=1,
                        label="oracle (theta)")
        axes[1].axhline(orc["NDCG@10"].iloc[0], ls="--", c="k", lw=1,
                        label="oracle (theta)")
    for ax, m_ in zip(axes, ["HR@10", "NDCG@10"]):
        ax.set_xlabel("released task information (nats)")
        ax.set_ylabel(m_)
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    axes[0].set_title("(a) hit rate @10 vs budget", fontsize=10)
    axes[1].set_title("(b) NDCG @10 vs budget", fontsize=10)
    fig.suptitle(f"Ranking utility, {ds}")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig_e7_ranking_{ds}.png", dpi=200)
    fig.savefig(f"{FIG}/fig_e7_ranking_{ds}.pdf")
    plt.close(fig)


def fig_e6_grouping(ds):
    p = f"{RES}/e6_ablD_grouping_{ds}.csv"
    if not os.path.exists(p):
        return  # D ablation ran on the primary dataset only
    d = pd.read_csv(p)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for strat, sub in d.groupby("strategy"):
        sub = sub.sort_values("K")
        axes[0].plot(sub["K"], sub["excess"], "o-", label=strat)
        axes[1].plot(sub["K"], sub["gender_adv"], "s--", label=strat)
    axes[0].set_xlabel("K"); axes[0].set_ylabel("excess risk")
    axes[0].set_title("grouping strategy: utility")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("K"); axes[1].set_ylabel("gender advantage")
    axes[1].set_title("grouping strategy: leakage")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig_e6_grouping_{ds}.png", dpi=200)
    fig.savefig(f"{FIG}/fig_e6_grouping_{ds}.pdf")
    plt.close(fig)


def sel_cols(df):
    return set(df.columns)


def main():
    os.makedirs(FIG, exist_ok=True)
    report = ["# C&S supplementary experiments: results summary", ""]
    for ds in ["ml-1m", "ml-100k", "book_crossing"]:
        sw = f"{RES}/sweep_{ds}.csv"
        if not os.path.exists(sw):
            continue
        df = pd.read_csv(sw)
        df = df[df["mechanism"] != "grouping_dud"]
        e4 = pd.read_csv(f"{RES}/e4_heterogeneity_{ds}.csv")
        ceil_g = e4["gender_I_thetaA"].iloc[0] if "gender_I_thetaA" in e4 else float("nan")
        ceil_a = e4["age3_I_thetaA"].iloc[0]
        fig_frontier(df, ds)
        fig_info_attack(df, ds, ceil_g, ceil_a)
        M_ds = 128
        try:
            M_ds = json.load(open(f"{RES}/meta_{ds}.json"))["M"]
        except Exception:
            pass
        fig_e2(ds, M_ds)
        fig_e4(ds)
        fig_e6(ds)
        fig_e6_grouping(ds)
        fig_e7(ds)

        # E7: ranking-utility table (matched NDCG comparison)
        p7 = f"{RES}/e7_ranking_{ds}.csv"
        if os.path.exists(p7):
            e7 = pd.read_csv(p7)
            report += [f"### E7 ranking utility ({ds})", "",
                       e7.to_markdown(index=False, floatfmt=".3f"), ""]

        # ablation tables
        for name in [f"e6_ablA_depth_{ds}.csv", f"e6_ablB_shadow_{ds}.csv",
                     f"e6_ablC_granularity_{ds}.csv",
                     f"e6_ablD_grouping_{ds}.csv"]:
            p = f"{RES}/{name}"
            if os.path.exists(p):
                report += [f"### {name}", "",
                           pd.read_csv(p).to_markdown(index=False,
                                                      floatfmt=".3f"), ""]

        # matched-utility comparison at half the population spread
        base = df[(df["mechanism"] == "grouping") &
                  (df["param"] == 1)]["excess_risk"].iloc[0]
        target = 0.5 * base
        report += [f"## {ds}", "",
                   f"### Frontier at matched utility (excess risk ~ {target:.3f})",
                   ""]
        sel = []
        for mech in df["mechanism"].unique():
            sub = df[df["mechanism"] == mech].copy()
            sub["d"] = (sub["excess_risk"] - target).abs()
            sel.append(sub.nsmallest(1, "d").iloc[0])
        attr_cols = [c for c in ["gender_adv", "gender_auc", "age3_adv",
                                 "gender_I_A"] if c in sel_cols(df)]
        tab = pd.DataFrame(sel)[
            ["mechanism", "param", "alpha_emp", "excess_risk",
             "P_theta_bayes"] + attr_cols]
        report += [tab.to_markdown(index=False, floatfmt=".3f"), ""]
        meta_f = f"{RES}/meta_{ds}.json"
        if os.path.exists(meta_f):
            report += ["meta: " + open(meta_f).read(), ""]
        for name in [f"e2_task_inference_{ds}.csv", f"e2_gap_vs_n_{ds}.csv",
                     f"e5_case_study_{ds}.csv", f"e5_calibration_{ds}.csv"]:
            p = f"{RES}/{name}"
            if os.path.exists(p):
                report += [f"### {name}", "",
                           pd.read_csv(p).to_markdown(index=False,
                                                      floatfmt=".3f"), ""]
        if os.path.exists(f"{RES}/e4_meta_{ds}.json"):
            report += ["### E4 meta", "", "```json",
                       open(f"{RES}/e4_meta_{ds}.json").read(), "```", ""]
    with open(f"{RES}/summary_report.md", "w") as f:
        f.write("\n".join(report))
    print("analysis done")


if __name__ == "__main__":
    main()
