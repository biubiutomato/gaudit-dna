#!/usr/bin/env python3
"""GC-matched corpus: same GC prior across classes, so probes cannot rely on %GC."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment import (
    CLASSES,
    DATA,
    RES,
    SEQ_LEN,
    HyenaEmbedder,
    PERTURBATIONS,
    embed_corpus,
    expand_iupac,
    gc_content,
    plant_motif,
    sample_from_gc,
    CODON_TABLE_RICH,
    STOP,
)

SEED = 7
N = 150
TARGET_GC = 0.50


def force_gc(seq: str, target: float, rng: random.Random) -> str:
    s = list(seq)
    n = len(s)
    want = int(round(target * n))
    gc_idx = [i for i, c in enumerate(s) if c in "GC"]
    at_idx = [i for i, c in enumerate(s) if c in "AT"]
    cur = len(gc_idx)
    if cur > want:
        for i in rng.sample(gc_idx, cur - want):
            s[i] = rng.choice("AT")
    elif cur < want:
        for i in rng.sample(at_idx, want - cur):
            s[i] = rng.choice("GC")
    return "".join(s)


def make_promoter_matched(length, rng):
    seq = sample_from_gc(length, TARGET_GC, rng)
    seq = plant_motif(seq, expand_iupac("TATAWAW", rng), length // 2 - 30)
    seq = plant_motif(seq, expand_iupac("GGGCGG", rng), length // 2 - 80)
    seq = plant_motif(seq, expand_iupac("CCAAT", rng), length // 2 - 110)
    seq = plant_motif(seq, expand_iupac("YYANWYY", rng), length // 2)
    for _ in range(6):
        pos = rng.randint(0, length - 4)
        seq = plant_motif(seq, rng.choice(["CGCG", "GCGC", "CCGG"]), pos)
    return force_gc(seq, TARGET_GC, rng)


def make_coding_matched(length, rng):
    n_codons = length // 3
    parts = ["ATG"]
    for _ in range(n_codons - 2):
        parts.append(rng.choice(CODON_TABLE_RICH))
    parts.append(rng.choice(STOP))
    seq = "".join(parts)
    # Fix: 256 // 3 codons give 255 nt; pad so every class has the same length
    # (same padding rule as make_coding in experiment.py).
    if len(seq) < length:
        seq += sample_from_gc(length - len(seq), TARGET_GC, rng)
    return force_gc(seq[:length], TARGET_GC, rng)


def make_background_matched(length, rng):
    seq = sample_from_gc(length, TARGET_GC, rng)
    return force_gc(seq, TARGET_GC, rng)


GEN = {
    "promoter": make_promoter_matched,
    "coding": make_coding_matched,
    "background": make_background_matched,
}


def probe(X, y, seed=0):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.35, random_state=seed, stratify=y)
    sc = StandardScaler()
    clf = LogisticRegression(max_iter=500)
    clf.fit(sc.fit_transform(Xtr), ytr)
    pred = clf.predict(sc.transform(Xte))
    acc = accuracy_score(yte, pred)
    yb = label_binarize(yte, classes=CLASSES)
    proba = clf.predict_proba(sc.transform(Xte))
    col = {c: i for i, c in enumerate(clf.classes_)}
    P = np.column_stack([proba[:, col[c]] for c in CLASSES])
    auroc = roc_auc_score(yb, P, average="macro", multi_class="ovr")
    return float(acc), float(auroc)


def main():
    rng = random.Random(SEED)
    rows = []
    for cls in CLASSES:
        for i in range(N):
            seq = GEN[cls](SEQ_LEN, rng)
            rows.append({"label": cls, "seq": seq, "gc": gc_content(seq)})
    df = pd.DataFrame(rows)
    df.to_csv(DATA / "corpus_gc_matched.csv", index=False)
    df["length"] = df.seq.str.len()
    print(df.groupby("label")[["gc", "length"]].agg(["mean", "std", "min", "max"]))
    assert df.length.nunique() == 1, "all classes must have the same length"
    seqs = df["seq"].tolist()
    y = df["label"].to_numpy()
    use_hyena = "--no-hyena" not in sys.argv
    hyena = HyenaEmbedder() if use_hyena else None
    models = ["gc_len", "2mer", "3mer", "4mer"] + (["hyena_tiny"] if use_hyena else [])
    perts = ["identity", "gc_shuffle", "dinuc_shuffle"]
    recs = []
    for p in perts:
        print("pert", p)
        if p == "identity":
            pseqs = seqs
        else:
            pseqs = [PERTURBATIONS[p](s, random.Random(SEED + i)) for i, s in enumerate(seqs)]
        for m in models:
            print(" ", m)
            X = embed_corpus(pseqs, m, hyena=hyena)
            acc, auroc = probe(X, y)
            recs.append({"model": m, "perturbation": p, "acc": acc, "auroc": auroc})
            print(f"    acc={acc:.3f} auroc={auroc:.3f}")
    out = pd.DataFrame(recs)
    name = "gc_matched_probe.csv" if use_hyena else "gc_matched_probe_nohyena.csv"
    out.to_csv(RES / name, index=False)
    print(out)


if __name__ == "__main__":
    main()
