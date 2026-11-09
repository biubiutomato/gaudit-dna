"""Probe helpers copied unchanged from the original enhance.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import binary_cross_entropy_with_logits

SEEDS = [0, 1, 2]


def _stratified_split(y, test_size=0.35, seed=0):
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    te, tr = [], []
    for lab in np.unique(y):
        idx = np.where(y == lab)[0]
        rng.shuffle(idx)
        n_te = max(1, int(round(len(idx) * test_size)))
        te.extend(idx[:n_te])
        tr.extend(idx[n_te:])
    return np.array(tr), np.array(te)


def _auroc(y, s):
    y = np.asarray(y).astype(int)
    s = np.asarray(s)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Mann-Whitney
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).mean() + 0.5 * (diff == 0).mean()))


def probe_binary(X, y, seed):
    tr, te = _stratified_split(y, 0.35, seed)
    Xtr, Xte = X[tr], X[te]
    ytr, yte = y[tr], y[te]
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0) + 1e-6
    Xtr_s = (Xtr - mu) / sd
    Xte_s = (Xte - mu) / sd
    xt = torch.tensor(Xtr_s, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)
    w = torch.zeros(xt.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=50)

    def closure():
        opt.zero_grad()
        logit = xt @ w + b
        loss = binary_cross_entropy_with_logits(logit, yt)
        loss = loss + 1e-2 * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        scores = (torch.tensor(Xte_s) @ w + b).numpy()
    proba = 1 / (1 + np.exp(-scores))
    pred = (proba >= 0.5).astype(int)
    return {
        "acc": float((pred == yte).mean()),
        "auroc": _auroc(yte, scores),
    }


def summarize(rows):
    df = pd.DataFrame(rows)
    g = (
        df.groupby(["model", "perturbation", "setting"])
        .agg(
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", "std"),
            acc_mean=("acc", "mean"),
            acc_std=("acc", "std"),
            n=("auroc", "size"),
        )
        .reset_index()
    )
    return df, g


def gc_matched_subset(seqs, y, meta, n_per=180, seed=1):
    """Pair positives and negatives with similar GC (bin matching)."""
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    used_neg = set()
    picked_p, picked_n = [], []
    order = rng.permutation(pos_idx)
    for i in order:
        g = meta.loc[i, "gc"]
        cands = [
            j
            for j in neg_idx
            if j not in used_neg and abs(meta.loc[j, "gc"] - g) <= 0.03
        ]
        if not cands:
            continue
        j = int(rng.choice(cands))
        used_neg.add(j)
        picked_p.append(int(i))
        picked_n.append(j)
        if len(picked_p) >= n_per:
            break
    idx = picked_p + picked_n
    print(
        "gc-matched pairs",
        len(picked_p),
        "gc pos",
        meta.loc[picked_p, "gc"].mean(),
        "gc neg",
        meta.loc[picked_n, "gc"].mean(),
    )
    return [seqs[i] for i in idx], y[idx]


