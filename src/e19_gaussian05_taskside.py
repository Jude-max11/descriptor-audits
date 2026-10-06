"""E19: task-side-information attack for Gaussian at the matched grid point
sigma=0.5*scale (e9 Part 2 protocol: 5-fold CV x 10 seeds, best-of)."""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from pipeline import (load_dataset, attrs_of, build_user_side, cluster_types,
                      mechanism_gaussian)
from e9_adversary import task_side

RES = os.path.join(os.path.dirname(__file__), "..", "results")
ds = load_dataset("ml-1m")
emb, svd, idx_i, top = build_user_side(ds, top_items=500, svd_dim=32, seed=7)
theta, centers = cluster_types(emb, 128, seed=7)
yattrs = {a: ds.users[a].values for a in attrs_of(ds)}
scale = float(np.std(emb, axis=0).mean())
decode, _, Z, _ = mechanism_gaussian(emb, centers, 0.5 * scale, 7)
rows = []
for attr in ("gender",):
    for r in task_side(np.asarray(Z), decode.astype(int), yattrs[attr]):
        r.update({"mechanism": "gaussian 0.5s", "attr": attr})
        rows.append(r)
        print(r, flush=True)
pd.DataFrame(rows).to_csv(f"{RES}/e19_gaussian05_taskside.csv", index=False)
