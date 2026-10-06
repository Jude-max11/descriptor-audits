#!/usr/bin/env python3
"""Reproduce the numerical illustrations for rate-constrained personalization."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np


def entropy(p: np.ndarray) -> float:
    q = p[p > 0]
    return float(-np.sum(q * np.log(q)))


def make_prior(spec: dict, m: int) -> np.ndarray:
    kind = spec["kind"]
    if kind == "uniform":
        p = np.ones(m)
    elif kind == "zipf":
        p = np.arange(1, m + 1, dtype=float) ** (-float(spec["exponent"]))
    elif kind == "head_tail":
        if m < 2:
            raise ValueError("head_tail requires M >= 2")
        head = float(spec["head_mass"])
        if not 0 < head < 1:
            raise ValueError("head_mass must lie in (0,1)")
        p = np.full(m, (1.0 - head) / (m - 1))
        p[0] = head
    else:
        raise ValueError(f"Unknown prior kind: {kind}")
    return p / p.sum()


def lower_bound(h: float, alpha: np.ndarray | float, delta: float) -> np.ndarray:
    a = np.asarray(alpha, dtype=float)
    inside = np.exp(2.0 * (h - a)) / (2.0 * math.pi * math.e) - 1.0 / 12.0
    return 4.0 * delta**2 * np.maximum(inside, 0.0)


def groups_for_rate(m: int, alpha: float) -> int:
    log_m = math.log(m)
    if alpha >= log_m - 1e-12:
        return m
    return min(m, max(1, int(math.floor(math.exp(alpha) + 1e-12))))


def consecutive_groups(m: int, k: int) -> list[np.ndarray]:
    return [x for x in np.array_split(np.arange(1, m + 1, dtype=float), k) if len(x)]


def grouping_risks(p: np.ndarray, delta: float, k: int) -> tuple[float, float]:
    risks = np.zeros(len(p), dtype=float)
    for idx in np.array_split(np.arange(len(p)), k):
        # The theorem uses the unweighted group mean and controls every type.
        representative = 2.0 * delta * float(np.mean(idx + 1))
        risks[idx] = (representative - 2.0 * delta * (idx + 1)) ** 2
    return float(np.dot(p, risks)), float(np.max(risks))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rate_sweep(cfg: dict, out: Path) -> list[dict]:
    m = int(cfg["primary_M"])
    delta = float(cfg["primary_delta"])
    alphas = np.linspace(0.0, math.log(m) + float(cfg["alpha_margin"]), int(cfg["alpha_points"]))
    p = np.ones(m) / m
    h = math.log(m)
    rows = []
    for a in alphas:
        k = groups_for_rate(m, float(a))
        avg, worst = grouping_risks(p, delta, k)
        rows.append({"alpha": float(a), "K": k, "lower_average": float(lower_bound(h, a, delta)),
                     "grouping_average": avg, "grouping_worst": worst})
    write_csv(out / "rate_sweep.csv", list(rows[0]), rows)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.semilogy(alphas, np.maximum([r["lower_average"] for r in rows], 1e-10), label="entropy converse", lw=2)
    ax.semilogy(alphas, np.maximum([r["grouping_average"] for r in rows], 1e-10), label="BTQ: average", lw=2)
    ax.semilogy(alphas, np.maximum([r["grouping_worst"] for r in rows], 1e-10), label="BTQ: worst case", lw=1.6, ls="--")
    ax.axvline(math.log(m), color="0.5", lw=1, ls=":", label=r"$\log M$")
    ax.set(xlabel=r"descriptor budget $\alpha$ (nats)", ylabel="squared excess risk",
           title=fr"Uniform types: $M={m}$, $\Delta={delta:g}$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "rate_sweep.pdf")
    fig.savefig(out / "rate_sweep.png", dpi=220)
    plt.close(fig)
    return rows


def prior_sweep(cfg: dict, out: Path) -> list[dict]:
    m = int(cfg["primary_M"])
    delta = float(cfg["primary_delta"])
    alphas = np.linspace(0.0, math.log(m) + float(cfg["alpha_margin"]), int(cfg["alpha_points"]))
    rows = []
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for spec in cfg["priors"]:
        p = make_prior(spec, m)
        h = entropy(p)
        values = lower_bound(h, alphas, delta)
        ax.semilogy(alphas, np.maximum(values, 1e-10), lw=2, label=fr"{spec['name']}: $e^H={math.exp(h):.1f}$")
        for a, value in zip(alphas, values):
            rows.append({"prior": spec["name"], "entropy": h, "effective_types": math.exp(h),
                         "alpha": float(a), "lower_average": float(value)})
    write_csv(out / "prior_sweep.csv", list(rows[0]), rows)
    ax.set(xlabel=r"descriptor budget $\alpha$ (nats)", ylabel="average-risk lower bound",
           title=fr"Effect of type concentration: $M={m}$, $\Delta={delta:g}$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "prior_sweep.pdf")
    fig.savefig(out / "prior_sweep.png", dpi=220)
    plt.close(fig)
    return rows


def scaling_sweep(cfg: dict, out: Path) -> list[dict]:
    rows = []
    for m in map(int, cfg["M_values"]):
        for delta in map(float, cfg["delta_values"]):
            for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
                alpha = fraction * math.log(m)
                k = groups_for_rate(m, alpha)
                p = np.ones(m) / m
                avg, worst = grouping_risks(p, delta, k)
                rows.append({"M": m, "delta": delta, "alpha_fraction": fraction, "alpha": alpha, "K": k,
                             "lower_average": float(lower_bound(math.log(m), alpha, delta)),
                             "grouping_average": avg, "grouping_worst": worst})
    write_csv(out / "scaling_table.csv", list(rows[0]), rows)
    return rows


def finite_sample_sweep(cfg: dict, out: Path) -> list[dict]:
    spec = cfg["finite_sample"]
    m = int(spec["M"])
    delta = float(spec["delta"])
    sigma = float(spec["sigma"])
    h = math.log(m)
    variance_mu = delta**2 * (m**2 - 1.0) / 3.0
    alphas = np.linspace(0.0, h + float(cfg["alpha_margin"]), int(cfg["alpha_points"]))
    rows = []
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    oracle = lower_bound(h, alphas, delta)
    ax.semilogy(alphas, np.maximum(oracle, 1e-12), lw=2.2, color="black", label="distribution oracle")
    for n in map(int, spec["sample_sizes"]):
        raw_sample_cap = 0.5 * math.log(1.0 + n * variance_mu / sigma**2)
        sample_cap = min(h, raw_sample_cap)
        beta = np.minimum(np.minimum(alphas, sample_cap), h)
        values = lower_bound(h, beta, delta)
        line, = ax.semilogy(alphas, np.maximum(values, 1e-12), lw=1.8,
                           label=fr"lower, $n={n}$ (cap {sample_cap:.2f})")
        groups = np.minimum(m, np.maximum(1, np.floor(np.exp(alphas)))).astype(int)
        group_radius = np.maximum(0.0, np.ceil(m / groups) - 1.0)
        error_probability = min(1.0, 2.0 * (1.0 + delta**2 / sigma**2)**(-n / 2.0))
        upper_tail = (delta**2 * group_radius**2
                      + 4.0 * delta**2 * (m - 1)**2 * error_probability)
        if n > 2:
            upper_ols = 2.0 * delta**2 * group_radius**2 + 8.0 * sigma**2 / (n - 2)
            upper = np.minimum(upper_ols, upper_tail)
        else:
            upper = upper_tail
        upper = np.minimum(upper, delta**2 * (m - 1)**2)
        ax.semilogy(alphas, upper, ls="--", lw=1.35, color=line.get_color(),
                    label=fr"achievable, $n={n}$")
        for a, b, value, achievable in zip(alphas, beta, values, upper):
            rows.append({"M": m, "delta": delta, "sigma": sigma, "n": n,
                         "alpha": float(a), "sample_information_cap": sample_cap,
                         "gaussian_sample_cap": raw_sample_cap,
                         "effective_budget": float(b), "lower_average": float(value),
                         "achievable_worst": float(achievable)})
    write_csv(out / "finite_sample_sweep.csv", list(rows[0]), rows)
    ax.set(xlabel=r"descriptor budget $\alpha$ (nats)", ylabel="excess-risk bound",
           title=fr"Finite samples: $M={m}$, $\Delta={delta:g}$, $\sigma={sigma:g}$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "finite_sample_sweep.pdf")
    fig.savefig(out / "finite_sample_sweep.png", dpi=220)
    plt.close(fig)
    return rows


def finite_bounds(m: int, delta: float, sigma: float, n: int, alpha: float) -> tuple[float, float]:
    """Return the finite-sample average-risk converse and worst-risk guarantee."""
    h = math.log(m)
    variance_mu = delta**2 * (m**2 - 1.0) / 3.0
    sample_cap = min(h, 0.5 * math.log(1.0 + n * variance_mu / sigma**2))
    beta = min(alpha, sample_cap)
    lower = float(lower_bound(h, beta, delta))
    k = groups_for_rate(m, alpha)
    radius = max(0, math.ceil(m / k) - 1)
    error_probability = min(1.0, 2.0 * (1.0 + delta**2 / sigma**2)**(-n / 2.0))
    tail = delta**2 * radius**2 + 4.0 * delta**2 * (m - 1)**2 * error_probability
    candidates = [tail, delta**2 * (m - 1)**2]
    if n > 2:
        candidates.append(2.0 * delta**2 * radius**2 + 8.0 * sigma**2 / (n - 2))
    return lower, min(candidates)


def simulate_estimate_then_group(rng: np.random.Generator, m: int, delta: float,
                                 sigma: float, n: int, alpha: float,
                                 trials: int) -> tuple[float, float, float]:
    """Exact-distribution Monte Carlo for OLS followed by nearest-type grouping."""
    theta = rng.integers(0, m, size=trials)
    q = rng.chisquare(n, size=trials)
    ols_error = sigma * rng.standard_normal(trials) / np.sqrt(q)
    true_mu = 2.0 * delta * (theta + 1)
    estimated = np.rint((true_mu + ols_error) / (2.0 * delta)).astype(int) - 1
    estimated = np.clip(estimated, 0, m - 1)
    k = groups_for_rate(m, alpha)
    representative = np.empty(m, dtype=float)
    for idx in np.array_split(np.arange(m), k):
        representative[idx] = 2.0 * delta * float(np.mean(idx + 1))
    loss = (representative[estimated] - true_mu)**2
    average = float(np.mean(loss))
    standard_error = float(np.std(loss, ddof=1) / math.sqrt(trials))
    counts = np.bincount(theta, minlength=m)
    totals = np.bincount(theta, weights=loss, minlength=m)
    conditional = np.divide(totals, counts, out=np.zeros(m), where=counts > 0)
    return average, float(np.max(conditional)), standard_error


def monte_carlo_rate_gap(cfg: dict, out: Path) -> list[dict]:
    spec = cfg["monte_carlo_finite"]
    rng = np.random.default_rng(int(cfg["seed"]) + 1)
    m, delta, sigma = int(spec["M"]), float(spec["delta"]), float(spec["sigma"])
    trials = int(spec["trials"])
    alphas = np.linspace(0.0, math.log(m), int(spec["alpha_points"]))
    rows = []
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    for n in map(int, spec["sample_sizes"]):
        empirical = []
        lowers, uppers = [], []
        for alpha in alphas:
            avg, worst, se = simulate_estimate_then_group(
                rng, m, delta, sigma, n, float(alpha), trials)
            lower, upper = finite_bounds(m, delta, sigma, n, float(alpha))
            empirical.append(avg); lowers.append(lower); uppers.append(upper)
            rows.append({"M": m, "delta": delta, "sigma": sigma, "n": n,
                         "alpha": float(alpha), "trials": trials,
                         "lower_average": lower, "empirical_average": avg,
                         "empirical_worst": worst, "standard_error": se,
                         "upper_worst": upper})
        line, = ax.semilogy(alphas, np.maximum(empirical, 1e-12), lw=2,
                           label=fr"Monte Carlo, $n={n}$")
        ax.semilogy(alphas, np.maximum(lowers, 1e-12), color=line.get_color(),
                    lw=1.0, ls=":")
        ax.semilogy(alphas, np.maximum(uppers, 1e-12), color=line.get_color(),
                    lw=1.2, ls="--")
    write_csv(out / "monte_carlo_rate_gap.csv", list(rows[0]), rows)
    ax.set(xlabel=r"descriptor budget $\alpha$ (nats)", ylabel="squared excess risk",
           title=fr"Estimate--then--group: $M={m}$, $\Delta={delta:g}$, $\sigma={sigma:g}$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.text(0.02, 0.02, "dotted: converse; solid: empirical average; dashed: worst-risk guarantee",
            transform=ax.transAxes, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "monte_carlo_rate_gap.pdf")
    fig.savefig(out / "monte_carlo_rate_gap.png", dpi=220)
    plt.close(fig)
    return rows


def sensitivity_monte_carlo(cfg: dict, out: Path) -> list[dict]:
    spec = cfg["sensitivity"]
    rng = np.random.default_rng(int(cfg["seed"]) + 2)
    base = dict(spec["baseline"])
    trials = int(spec["trials"])
    fraction = float(spec["alpha_fraction"])
    keys = [("M", "population size $M$"), ("delta", r"separation $\Delta$"),
            ("sigma", r"noise $\sigma$"), ("n", "sample size $n$")]
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.2))
    for ax, (key, label) in zip(axes.flat, keys):
        xs, empirical, lower_values, upper_values = [], [], [], []
        for value in spec["values"][key]:
            params = dict(base); params[key] = value
            m, n = int(params["M"]), int(params["n"])
            delta, sigma = float(params["delta"]), float(params["sigma"])
            alpha = fraction * math.log(m)
            avg, worst, se = simulate_estimate_then_group(
                rng, m, delta, sigma, n, alpha, trials)
            lower, upper = finite_bounds(m, delta, sigma, n, alpha)
            xs.append(float(value)); empirical.append(avg)
            lower_values.append(lower); upper_values.append(upper)
            rows.append({"varied": key, "value": value, "M": m, "delta": delta,
                         "sigma": sigma, "n": n, "alpha": alpha, "trials": trials,
                         "lower_average": lower, "empirical_average": avg,
                         "empirical_worst": worst, "standard_error": se,
                         "upper_worst": upper})
        ax.semilogy(xs, np.maximum(lower_values, 1e-12), marker="v", ls=":", label="converse")
        ax.semilogy(xs, np.maximum(empirical, 1e-12), marker="o", lw=2, label="Monte Carlo")
        ax.semilogy(xs, np.maximum(upper_values, 1e-12), marker="^", ls="--", label="guarantee")
        ax.set_xlabel(label); ax.grid(True, which="both", alpha=0.25)
        if key in {"M", "n"}: ax.set_xscale("log")
    axes[0, 0].set_ylabel("squared excess risk")
    axes[1, 0].set_ylabel("squared excess risk")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(fr"Sensitivity at $\alpha={fraction:g}\log M$; other parameters at baseline")
    fig.tight_layout()
    fig.savefig(out / "sensitivity_monte_carlo.pdf")
    fig.savefig(out / "sensitivity_monte_carlo.png", dpi=220)
    plt.close(fig)
    write_csv(out / "sensitivity_monte_carlo.csv", list(rows[0]), rows)
    return rows


def monte_carlo_check(cfg: dict, out: Path) -> dict:
    rng = np.random.default_rng(int(cfg["seed"]))
    m = int(cfg["primary_M"])
    delta = float(cfg["primary_delta"])
    alpha = 0.5 * math.log(m)
    k = groups_for_rate(m, alpha)
    p = np.ones(m) / m
    exact_avg, exact_worst = grouping_risks(p, delta, k)
    theta = rng.integers(1, m + 1, size=int(cfg["monte_carlo_samples"]))
    group_indices = np.array_split(np.arange(1, m + 1), k)
    representative = np.zeros(m + 1)
    for group in group_indices:
        representative[group] = 2.0 * delta * np.mean(group)
    empirical = float(np.mean((representative[theta] - 2.0 * delta * theta) ** 2))
    result = {"seed": int(cfg["seed"]), "samples": int(cfg["monte_carlo_samples"]), "M": m,
              "delta": delta, "alpha": alpha, "K": k, "exact_average": exact_avg,
              "empirical_average": empirical, "absolute_error": abs(empirical - exact_avg),
              "exact_worst": exact_worst}
    (out / "monte_carlo_check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def write_summary(cfg: dict, out: Path, prior_rows: list[dict], mc: dict) -> None:
    seen = {}
    for row in prior_rows:
        seen.setdefault(row["prior"], (row["entropy"], row["effective_types"]))
    lines = ["# Numerical experiment summary", "", f"Primary M: {cfg['primary_M']}",
             f"Primary Delta: {cfg['primary_delta']}", "", "## Prior effective cardinalities", "",
             "| Prior | H(theta) | exp(H(theta)) |", "|---|---:|---:|"]
    lines += [f"| {name} | {h:.6f} | {eff:.3f} |" for name, (h, eff) in seen.items()]
    lines += ["", "## Monte Carlo implementation check", "",
              f"At alpha = {mc['alpha']:.6f} and K = {mc['K']}, the exact grouping average risk is "
              f"{mc['exact_average']:.8g}; the seeded Monte Carlo estimate is {mc['empirical_average']:.8g} "
              f"(absolute error {mc['absolute_error']:.3g}).", "",
              "Generated figures: `rate_sweep.pdf`, `prior_sweep.pdf`, "
              "`finite_sample_sweep.pdf`, `monte_carlo_rate_gap.pdf`, and "
              "`sensitivity_monte_carlo.pdf`."]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results"))
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    # Platform-independent boundary checks for the integer grouping rule.
    for m in map(int, cfg["M_values"] + [cfg["primary_M"]]):
        assert groups_for_rate(m, 0.0) == 1
        assert groups_for_rate(m, math.log(m)) == m
    rate_sweep(cfg, args.output)
    prior_rows = prior_sweep(cfg, args.output)
    scaling_sweep(cfg, args.output)
    finite_sample_sweep(cfg, args.output)
    monte_carlo_rate_gap(cfg, args.output)
    sensitivity_monte_carlo(cfg, args.output)
    mc = monte_carlo_check(cfg, args.output)
    write_summary(cfg, args.output, prior_rows, mc)
    print(f"Results written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
