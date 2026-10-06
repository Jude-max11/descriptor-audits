"""E20: VQ-resolution sensitivity for Gaussian at the matched grid point
sigma=0.5*scale (e15 protocol, bins 32-256)."""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from pipeline import (load_dataset, attrs_of, build_user_side, cluster_types,
                      mechanism_gaussian)
from e8_worstcase import (max_leakage, min_entropy_leakage, ml_conditional,
                          bootstrap_ci, il_inf_conditional,
                          advantage_percentiles, vq_labels)

RES = os.path.join(os.path.dirname(__file__), "..", "results")
ds = load_dataset("ml-1m")
emb, svd, idx_i, top = build_user_side(ds, top_items=500, svd_dim=32, seed=7)
theta, centers = cluster_types(emb, 128, seed=7)
yattrs = {a: ds.users[a].values for a in attrs_of(ds)}
scale = float(np.std(emb, axis=0).mean())
decode, _, Z, _ = mechanism_gaussian(emb, centers, 0.5 * scale, 7)
Z = np.asarray(Z)
rows = []
for bins in (32, 64, 89, 128, 256):
    zv = vq_labels(Z, n_bins=bins)
    for attr in ("gender", "age3"):
        y = yattrs[attr]
        r = {"mechanism": "gaussian 0.5s", "bins": bins, "attr": attr,
             "ML": max_leakage(zv, y),
             "IL_inf": min_entropy_leakage(zv, y)}
        r.update(ml_conditional(zv, y, theta))
        rows.append(r)
        print(r, flush=True)
pd.DataFrame(rows).to_csv(f"{RES}/e20_gaussian05_vq.csv", index=False)
print("e20 done")
