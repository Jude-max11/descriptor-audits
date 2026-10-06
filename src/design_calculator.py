"""Theory-driven descriptor sizing calculator.

Usage: python design_calculator.py --M 128 --delta 1 --targets 1 0.25 0.01
"""
import argparse, csv, math, pathlib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=128)
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--targets", type=float, nargs="+", default=[1.0, .25, .01])
    ap.add_argument("--output", default="results/design_table.csv")
    a = ap.parse_args()
    out = pathlib.Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    rows=[]
    for eps in a.targets:
        if eps <= 0: raise ValueError("targets must be positive")
        # Exact inversion of Delta^2 (ceil(M/K)-1)^2 <= epsilon.
        qmax = math.floor(math.sqrt(eps)/a.delta)
        k = math.ceil(a.M/(qmax + 1))
        k = max(1, min(a.M, k))
        alpha = math.log(k)
        certified = a.delta*a.delta*(math.ceil(a.M/k)-1)**2
        rows.append({"M":a.M,"delta":a.delta,"epsilon_target":eps,
                     "required_groups":k,"required_budget_nats":alpha,
                     "certified_grouping_risk":certified})
    with out.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f"wrote {out}")
    for r in rows: print(r)

if __name__ == "__main__": main()
