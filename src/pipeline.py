"""Shared pipeline for the C&S supplementary experiments.

Threat model (mirrors the manuscript): a client holds local task data D
(user interaction history), releases a randomized descriptor Z, and a
honest-but-curious server personalizes from Z alone.  The adversary observes
Z and attempts (i) task identification (recover theta) and (ii) attribute
inference (recover protected attribute A).

Task families are built from REAL public datasets by clustering users into
M task types; the theory assumptions (equally spaced quadratic family) do not
hold there, so experiments validate the DESIGN RULE's conservativeness and
leakage TRENDS, not the literal bounds (stated openly in the paper).

Mechanisms:
  grouping : theta -> group label via bisecting k-means tree, K = floor(e^alpha)
  grr      : k-ary randomized response over M types (Erlingsson et al., CCS'14)
  brappor  : one-hot bit vector with permanent randomized response (ibid.)
  gaussian : x + N(0, sigma^2 I), server decodes nearest centroid
  pca      : release x projected to d' dims, server decodes nearest centroid

Information budget: for every mechanism we report the EMPIRICAL mutual
information I(theta; Z_disc) in nats, where Z_disc is the server's decoded
type for continuous/vector releases.  Curves are therefore comparable on a
common x-axis.
"""
from __future__ import annotations

import json
import math
import os
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

warnings.filterwarnings("ignore")

RNG_GLOBAL = np.random.default_rng(20260907)

# ----------------------------------------------------------------------
# Dataset loading
# ----------------------------------------------------------------------

AGE_BINS = [-np.inf, 34, 45, np.inf]          # Xue et al. 2017 binning
AGE_LABELS = ["<35", "35-45", ">45"]


@dataclass
class Dataset:
    name: str
    users: pd.DataFrame        # uid, gender, age3 (attributes)
    ratings: pd.DataFrame      # uid, iid, rating
    emb: np.ndarray            # user embedding used as local data summary
    uids: np.ndarray


def load_book_crossing(path_prefix: str, name: str = "book_crossing",
                       min_ratings: int = 5, max_users: int = 20000,
                       seed: int = 7) -> Dataset:
    """Book-Crossing (Ziegler 2004): semicolon/latin-1 schema.

    Keeps users with valid age (5-100) and >= min_ratings EXPLICIT ratings
    (implicit 0s excluded from the MSE pipeline), capped at max_users by a
    seeded random subsample for tractability.  No gender column: the
    attribute set is {age3} only.
    """
    import csv as _csv
    users = pd.read_csv(f"{path_prefix}/BX-Users.csv", sep=";",
                        encoding="latin-1", engine="python",
                        quoting=_csv.QUOTE_NONE, on_bad_lines="skip")
    users.columns = ["uid", "location", "age"]
    ratings = pd.read_csv(f"{path_prefix}/BX-Book-Ratings.csv", sep=";",
                          encoding="latin-1", engine="python",
                          quoting=_csv.QUOTE_NONE, on_bad_lines="skip")
    ratings.columns = ["uid", "iid", "rating"]
    for col in ("uid", "iid"):
        ratings[col] = (ratings[col].astype(str).str.strip('"')
                        .str.split(";").str[0])
    users["uid"] = users["uid"].astype(str).str.strip('"').str.split(";").str[0]
    ratings["rating"] = pd.to_numeric(ratings["rating"].astype(str)
                                      .str.strip('"'), errors="coerce")
    users["age"] = pd.to_numeric(users["age"].astype(str).str.strip('"')
                                 .str.split(";").str[0], errors="coerce")
    users["uid"] = pd.to_numeric(users["uid"], errors="coerce")
    ratings["uid"] = pd.to_numeric(ratings["uid"], errors="coerce")
    users = users.dropna(subset=["uid"])
    ratings = ratings.dropna(subset=["uid", "rating"])
    ratings = ratings[ratings["rating"] > 0]                # explicit only
    users["age"] = pd.to_numeric(users["age"], errors="coerce")
    users = users[(users["age"] >= 5) & (users["age"] <= 100)]
    counts = ratings.groupby("uid").size()
    keep = counts[counts >= min_ratings].index
    users = users[users["uid"].isin(keep)]
    if len(users) > max_users:
        users = users.sample(max_users, random_state=seed)
    keep = users["uid"]
    ratings = ratings[ratings["uid"].isin(keep)]
    users["gender"] = -1                                    # absent marker
    users["age3"] = pd.cut(users["age"], AGE_BINS,
                           labels=range(3)).astype(int)
    users = users.reset_index(drop=True)
    ratings = ratings.reset_index(drop=True)
    return Dataset(name=name, users=users, ratings=ratings, emb=None,
                   uids=users["uid"].values)


def load_dataset(name: str) -> Dataset:
    base = os.path.join(os.path.dirname(__file__), "..", "data")
    if name == "ml-1m":
        return load_movielens(f"{base}/ml-1m", name)
    if name == "ml-100k":
        return load_movielens(f"{base}/ml-100k", name)
    if name == "book_crossing":
        return load_book_crossing(f"{base}/book_crossing", name)
    raise ValueError(name)


def attrs_of(ds: Dataset):
    """Available protected attributes (gender skipped when absent)."""
    out = ["age3"]
    if getattr(ds, "name", "") != "book_crossing":
        out = ["gender", "age3"]
    return out


def load_movielens(path_prefix: str, name: str) -> Dataset:
    """Handles both ml-1m (ratings.dat/users.dat, '::') and ml-100k
    (u.data/u.user, tab/'|') layouts."""
    if os.path.exists(f"{path_prefix}/ratings.dat"):          # ml-1m
        ratings = pd.read_csv(
            f"{path_prefix}/ratings.dat", sep="::", engine="python",
            names=["uid", "iid", "rating", "ts"])
        users = pd.read_csv(
            f"{path_prefix}/users.dat", sep="::", engine="python",
            names=["uid", "gender", "age", "occ", "zip"])
    else:                                                     # ml-100k
        ratings = pd.read_csv(
            f"{path_prefix}/u.data", sep="\t",
            names=["uid", "iid", "rating", "ts"])
        # u.user layout: id | age | gender | occupation | zip  (differs from 1M)
        users = pd.read_csv(
            f"{path_prefix}/u.user", sep="|",
            names=["uid", "age", "gender", "occ", "zip"])
    users["gender"] = (users["gender"] == "M").astype(int)          # 0 F, 1 M
    users["age3"] = pd.cut(users["age"], AGE_BINS, labels=range(3)).astype(int)
    return Dataset(name=name, users=users, ratings=ratings, emb=None,
                   uids=users["uid"].values)


def build_user_side(dataset: Dataset, top_items: int = 500, svd_dim: int = 32,
                    seed: int = 0):
    """Compute the client-side summary x_u of each user's LOCAL ratings.

    The rating matrix is mean-centered per user so that x_u captures taste
    direction (the 'task'), not activity level.
    """
    r = dataset.ratings
    pop = r["iid"].value_counts()
    top = pop.index[:top_items]
    idx_i = {iid: j for j, iid in enumerate(top)}
    uids = dataset.uids
    idx_u = {uid: i for i, uid in enumerate(uids)}
    X = np.zeros((len(uids), len(top)))
    sub = r[r["iid"].isin(idx_i)]
    rows = sub["uid"].map(idx_u).values
    cols = sub["iid"].map(idx_i).values
    X[rows, cols] = sub["rating"].values
    # local mean-centering (client-side, uses only the client's own ratings)
    nz = (X > 0)
    cnt = nz.sum(1, keepdims=True)
    cnt[cnt == 0] = 1
    mu = (X.sum(1, keepdims=True)) / cnt
    Xc = np.where(nz, X - mu, 0.0)
    svd = TruncatedSVD(n_components=svd_dim, random_state=seed)
    emb = svd.fit_transform(Xc)
    dataset.emb = emb
    return emb, svd, idx_i, top


def user_holdout_split(ratings: pd.DataFrame, frac: float = 0.5, seed: int = 0):
    """Split each user's ratings into two halves (fresh-vs-training Z, E2)."""
    rng = np.random.default_rng(seed)
    r = ratings.sample(frac=1.0, random_state=seed).copy()
    r["u"] = r["uid"]
    r["rank"] = r.groupby("uid").cumcount()
    r["sz"] = r.groupby("uid")["uid"].transform("size")
    half_a = r[r["rank"] < (r["sz"] * frac).astype(int)]
    half_b = r[r["rank"] >= (r["sz"] * frac).astype(int)]
    return half_a.drop(columns=["u", "rank", "sz"]), half_b.drop(columns=["u", "rank", "sz"])


# ----------------------------------------------------------------------
# Task family construction
# ----------------------------------------------------------------------

def cluster_types(emb: np.ndarray, M: int, seed: int = 0):
    km = KMeans(n_clusters=M, n_init=10, random_state=seed)
    theta = km.fit_predict(emb)
    return theta, km.cluster_centers_


def balanced_group_partition(theta, centers, K, seed: int = 0):
    """Balanced grouping via bisecting 2-means on the centroid tree.

    Analogue of the manuscript's balanced grouping: recursively split the
    group with the largest spread*count until K groups exist.  Returns
    (per-user group labels, type->group array of length M,
     dict group -> group centroid in embedding space).
    """
    M = len(centers)
    if K >= M:
        t2g = np.arange(M)
        return t2g[theta], t2g, {m: centers[m] for m in range(M)}
    groups = [np.arange(M)]
    counts = np.bincount(theta, minlength=M)
    while len(groups) < K:
        best_g, best_score, best_parts = -1, -1.0, None
        for gi, g in enumerate(groups):
            if len(g) < 2:
                continue
            c = centers[g]
            spread = float(np.max(np.linalg.norm(c - c.mean(0), axis=1)))
            score = spread * math.sqrt(counts[g].sum())
            if score > best_score:
                km2 = KMeans(n_clusters=2, n_init=5, random_state=seed).fit(c)
                parts = [g[km2.labels_ == 0], g[km2.labels_ == 1]]
                if min(len(p) for p in parts) == 0:
                    continue
                best_g, best_score, best_parts = gi, score, parts
        if best_g < 0:               # cannot split further
            break
        groups[best_g] = best_parts[0]
        groups.append(best_parts[1])
    type2group = np.zeros(M, dtype=int)
    for k, g in enumerate(groups):
        type2group[g] = k
    gc = {k: centers[g].mean(0) for k, g in enumerate(groups)}
    return type2group[theta], type2group, gc


# ----------------------------------------------------------------------
# Descriptor mechanisms
# ----------------------------------------------------------------------

def mechanism_grouping(theta, centers, K, seed):
    """Theory mechanism: deterministic grouping descriptor (rate log K)."""
    user_g, _, group_centers = balanced_group_partition(theta, centers, K, seed)
    return user_g, group_centers, np.eye(max(K, 1))[user_g]   # label, decode, Z


def mechanism_grr(theta, M, eps, seed):
    """k-ary randomized response (GRR, Erlingsson et al. CCS'14)."""
    rng = np.random.default_rng(seed)
    p = math.exp(eps) / (math.exp(eps) + M - 1)
    q = (1 - p) / (M - 1)
    out = rng.choice(M, size=len(theta))
    same = rng.random(len(theta)) < p
    out[same] = theta[same]
    decode = out                                  # server serves decoded type
    Z = np.eye(M)[out]                            # released one-hot
    return decode, None, Z, (p, q)


def mechanism_brappor(theta, M, eps, seed):
    """Basic one-hot RAPPOR: permanent randomized response on M bits.

    Channel: P(Z_j=1 | b_j=1) = p with p = 1/(1+e^{eps/2}), and
    P(Z_j=1 | b_j=0) = q = 1-p (implemented as the isomorphic near-complement
    form: the true 1 flips to 0 with prob 1-p, true 0s flip to 1 with prob
    q).  Server decode: per-bit log-likelihood ratio under this channel,
    ll1 - ll0 = Z*log(p/q) + (1-Z)*log((1-p)/(1-q)); with q = 1-p the bits
    with Z_j = 0 (almost surely only the true bit at moderate eps) receive
    the positive score, so argmax recovers the type.

    Revision-2 bug fix: the previous ll1 used (1-p) on 1s, which under the
    symmetric channel q = 1-p makes ll1 and ll0 mathematically identical and
    leaves the decode resting on floating-point round-off (it happened to
    return the true bit for some eps, e.g. 12, and failed for others, e.g.
    16).  The formulas below follow the actual channel.
    """
    rng = np.random.default_rng(seed)
    p = 1.0 / (1.0 + math.exp(eps / 2.0))          # P(1|0) after transform
    q = 1.0 - p                                    # symmetric basic RAPPOR
    truth = np.eye(M)[theta]
    flip1 = rng.random(truth.shape) < (1 - p)      # true 1 -> 0
    flip0 = rng.random(truth.shape) < q            # true 0 -> 1
    Z = np.where(truth == 1, np.where(flip1, 0, 1),
                 np.where(flip0, 1, 0)).astype(float)
    ll1 = Z * math.log(p) + (1 - Z) * math.log(1 - p)      # P(Z | b=1)
    ll0 = Z * math.log(q) + (1 - Z) * math.log(1 - q)      # P(Z | b=0)
    decode = np.argmax(ll1 - ll0, axis=1)
    return decode, None, Z, (p, q)


def mechanism_gaussian(emb, centers, sigma, seed):
    rng = np.random.default_rng(seed)
    Z = emb + sigma * rng.standard_normal(emb.shape)
    d2 = ((Z[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
    decode = np.argmin(d2, axis=1)
    return decode, None, Z, sigma


def mechanism_pca(emb, centers, d_prime, seed):
    P = emb[:, :d_prime]
    C = centers[:, :d_prime]
    d2 = ((P[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    decode = np.argmin(d2, axis=1)
    return decode, None, P, d_prime


# ----------------------------------------------------------------------
# Empirical information measures
# ----------------------------------------------------------------------

def _plugin_mi(x, y):
    n = len(x)
    nx = np.bincount(x); ny = np.bincount(y)
    px = nx / n; py = ny / n
    joint = np.zeros((len(nx), len(ny)))
    np.add.at(joint, (x, y), 1.0)
    pxy = joint / n
    mask = pxy > 0
    return float((pxy[mask] * np.log(pxy[mask] /
               (px[:, None] * py[None, :])[mask])).sum())


def empirical_mi(x, y, n_perm: int = 20, seed: int = 0):
    """Discrete MI in nats with permutation-null bias subtraction.

    High-cardinality plug-in MI is upward-biased at small n; subtracting the
    mean MI under label permutation gives a nearly unbiased estimate
    (Miller-Madow-style correction is unreliable at |X|x|Y| ~ n).
    """
    x = np.asarray(x).astype(int); y = np.asarray(y).astype(int)
    raw = _plugin_mi(x, y)
    if n_perm <= 0 or raw == 0.0:
        return max(raw, 0.0)
    rng = np.random.default_rng(seed)
    null = float(np.mean([_plugin_mi(x, rng.permutation(y))
                          for _ in range(n_perm)]))
    return max(raw - null, 0.0)


def guessing_advantage(y_true, y_pred):
    """advantage = BAcc - random baseline (Jayaraman & Evans, CCS 2022)."""
    classes, counts = np.unique(y_true, return_counts=True)
    base = counts.max() / len(y_true)             # majority baseline
    # balanced accuracy
    per = [np.mean(y_pred[y_true == c] == c) for c in classes]
    bacc = float(np.mean(per))
    return bacc, bacc - base


# ----------------------------------------------------------------------
# Attacker protocol (RAID / WSDM'21 standard)
# ----------------------------------------------------------------------

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score


def make_attacker(kind: str):
    if kind == "mlp":
        return MLPClassifier(hidden_layer_sizes=(100,), alpha=1.0,
                             learning_rate_init=1e-3, max_iter=500,
                             random_state=None, early_stopping=False)
    if kind == "lr":
        # class-balanced: attribute priors are skewed (e.g., 71% male);
        # without balancing the attacker collapses to the majority class
        return LogisticRegression(max_iter=2000, C=1.0,
                                  class_weight="balanced")
    if kind == "rf":
        return RandomForestClassifier(n_estimators=200, n_jobs=8,
                                      class_weight="balanced")
    raise ValueError(kind)


def run_attack(Z, y, seeds=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), folds=5,
               attackers=("mlp",)):
    """5-fold stratified CV x seeds; returns mean/std of BAcc, advantage,
    micro-F1; AUC for binary tasks when probas available."""
    y = np.asarray(y).astype(int)
    rec = {"bacc": [], "adv": [], "f1": [], "auc": []}
    for s in seeds:
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=s)
        for tr, te in skf.split(Z, y):
            for kind in attackers:
                clf = make_attacker(kind)
                try:
                    clf.fit(Z[tr], y[tr])
                    pred = clf.predict(Z[te])
                except Exception:
                    continue
                classes = np.unique(y[tr])
                n_classes = len(classes)
                # advantage over RANDOM guessing (Jayaraman & Evans, CCS'22)
                per = [np.mean(pred[y[te] == c] == c) if (y[te] == c).any() else 0.0
                       for c in classes]
                bacc = float(np.mean(per))
                rec["bacc"].append(bacc)
                rec["adv"].append(bacc - 1.0 / n_classes)
                rec["f1"].append(f1_score(y[te], pred, average="micro"))
                if len(classes) == 2 and hasattr(clf, "predict_proba"):
                    try:
                        rec["auc"].append(roc_auc_score(
                            y[te], clf.predict_proba(Z[te])[:, list(classes).index(1)]))
                    except Exception:
                        pass
    out = {}
    for k, v in rec.items():
        if v:
            out[k] = (float(np.mean(v)), float(np.std(v)))
    return out


# ----------------------------------------------------------------------
# Utility model (personalization quality)
# ----------------------------------------------------------------------

def utility_excess(dataset, decode, svd, top_items_idx, half_train, half_test,
                   group_or_type_models):
    """Held-out excess risk: MSE(server model from descriptor) - MSE(client's
    own local model).  Server model predicts the user's held-out ratings."""
    raise NotImplementedError   # implemented in e2/e5 with rating matrices


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
