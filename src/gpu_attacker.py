"""GPU (torch) MLP attacker, faithful to the sklearn protocol but with
class-balanced cross-entropy (the depth ablation measures CAPACITY, so the
loss must not collapse to the majority class).

Mirrors sklearn MLPClassifier: hidden layers of 100, L2 alpha=1.0 on weights
(not biases), Adam lr=1e-3, max_iter=500 epochs, minibatch 200.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class TorchMLP:
    def __init__(self, depth=2, width=100, alpha=1e-4, lr=1e-3, epochs=1500,
                 batch=None, seed=0, device=None):
        self.depth, self.width = depth, width
        self.alpha, self.lr, self.epochs, self.batch = alpha, lr, epochs, batch
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        n, d = X.shape
        self.classes_ = np.unique(y)
        c = len(self.classes_)
        remap = {v: i for i, v in enumerate(self.classes_)}
        yt = torch.as_tensor([remap[v] for v in y], dtype=torch.long)
        Xt = torch.as_tensor(X, dtype=torch.float32)
        layers, in_dim = [], d
        for _ in range(self.depth):
            layers += [nn.Linear(in_dim, self.width), nn.ReLU()]
            in_dim = self.width
        layers += [nn.Linear(in_dim, c)]
        self.net = nn.Sequential(*layers).to(self.device)
        # class-balanced weights: w_c = n / (c * n_c)
        cnt = torch.bincount(yt, minlength=c).float()
        w = (len(yt) / (c * cnt.clamp(min=1))).to(self.device)
        lossf = nn.CrossEntropyLoss(weight=w)
        opt = torch.optim.Adam(
            [p for n_, p in self.net.named_parameters() if "weight" in n_],
            lr=self.lr, weight_decay=self.alpha)
        opt.add_param_group({
            "params": [p for n_, p in self.net.named_parameters()
                       if "bias" in n_], "weight_decay": 0.0})
        Xt, yt = Xt.to(self.device), yt.to(self.device)
        bs = n if self.batch is None else min(self.batch, n)
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                loss = lossf(self.net(Xt[idx]), yt[idx])
                loss.backward()
                opt.step()
        return self

    def predict(self, X):
        with torch.no_grad():
            Xt = torch.as_tensor(X, dtype=torch.float32).to(self.device)
            out = self.net(Xt).argmax(1).cpu().numpy()
        return self.classes_[out]

    def predict_proba(self, X):
        with torch.no_grad():
            Xt = torch.as_tensor(X, dtype=torch.float32).to(self.device)
            out = torch.softmax(self.net(Xt), 1).cpu().numpy()
        return out


def balanced_mlp_adv(Z, y, depths, seeds, folds=5):
    """Class-balanced GPU MLP: advantage/AUC per depth (mean +/- std over
    stratified 5-fold x seeds)."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    out = []
    for d in depths:
        advs, aucs = [], []
        for s in seeds:
            skf = StratifiedKFold(folds, shuffle=True, random_state=s)
            for tr, te in skf.split(Z, y):
                clf = TorchMLP(depth=d, seed=s).fit(Z[tr], y[tr])
                pred = clf.predict(Z[te])
                classes = np.unique(y[tr])
                per = [np.mean(pred[y[te] == c] == c)
                       if (y[te] == c).any() else 0.0 for c in classes]
                advs.append(np.mean(per) - 1.0 / len(classes))
                if len(classes) == 2:
                    try:
                        aucs.append(roc_auc_score(
                            y[te], clf.predict_proba(Z[te])[:, 1]))
                    except Exception:
                        pass
        out.append({"depth": d,
                    "adv_mean": float(np.mean(advs)),
                    "adv_std": float(np.std(advs)),
                    "auc_mean": float(np.mean(aucs)) if aucs else np.nan})
    return out
