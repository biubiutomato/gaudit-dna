#!/usr/bin/env python3
"""Real human non-TATA promoters (Genomic Benchmarks): GC+len, k-mer and HyenaDNA-tiny.

Same sampling, GC matching, perturbations and probe as the original run
(enhance_kmer.py / enhance.py), with HyenaDNA-tiny added.
Sequences are cached in data/real_promoters.csv after the first download.

Usage:  python real_promoters.py            # with HyenaDNA-tiny
        python real_promoters.py --no-hyena # compositional encoders only
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment import DATA, PERTURBATIONS, RES, HyenaEmbedder, cosine, embed_corpus, gc_content, pairwise_mean_cosine
from probe_utils import SEEDS, gc_matched_subset, probe_binary

CACHE = DATA / "real_promoters.csv"


def load_real(n_pos=280, n_neg=280, seed=0):
    if CACHE.exists():
        df = pd.read_csv(CACHE)
        print(f"loaded {len(df)} cached sequences from {CACHE}")
    else:
        from datasets import load_dataset

        ds = load_dataset("katarinagresova/Genomic_Benchmarks_human_nontata_promoters", split="train")
        pos, neg = [], []
        for ex in ds:  # identical filtering to the original enhance_kmer.py
            seq = ex["seq"].upper().replace("N", "A")[:251]
            if set(seq) - set("ACGT"):
                continue
            (pos if ex["label"] == 1 else neg).append(seq)
        rng = random.Random(seed)
        rng.shuffle(pos)
        rng.shuffle(neg)
        pos, neg = pos[:n_pos], neg[:n_neg]
        df = pd.DataFrame({"seq": pos + neg, "label": [1] * len(pos) + [0] * len(neg)})
        DATA.mkdir(parents=True, exist_ok=True)
        df.to_csv(CACHE, index=False)
    df["gc"] = [gc_content(s) for s in df.seq]
    print("n", len(df), "gc+", round(df[df.label == 1].gc.mean(), 3), "gc-", round(df[df.label == 0].gc.mean(), 3))
    return df.seq.tolist(), df.label.to_numpy(), df[["label", "gc"]].reset_index(drop=True)


def run_setting(name, seqs, y, models, perts, hyena):
    rows, drift = [], []
    orig = {}
    for m in models:
        print(f"[{name}] embed original {m}")
        orig[m] = embed_corpus(seqs, m, hyena=hyena)
        drift.append({"setting": name, "model": m, "perturbation": "random_pair",
                      "mean_sim": pairwise_mean_cosine(orig[m], 500, 1)})
    for p in perts:
        if p == "identity":
            Xp = orig
        else:
            pseqs = [PERTURBATIONS[p](s, random.Random(1000 + i)) for i, s in enumerate(seqs)]
            Xp = {}
            for m in models:
                print(f"[{name}] {p}: embed {m}")
                Xp[m] = embed_corpus(pseqs, m, hyena=hyena)
                sims = [cosine(orig[m][i], Xp[m][i]) for i in range(len(seqs))]
                drift.append({"setting": name, "model": m, "perturbation": p,
                              "mean_sim": float(np.mean(sims)), "p10": float(np.percentile(sims, 10))})
        for m in models:
            for seed in SEEDS:
                rows.append({"setting": name, "model": m, "perturbation": p, "seed": seed,
                             **probe_binary(Xp[m], y, seed)})
    return rows, drift


def main():
    use_hyena = "--no-hyena" not in sys.argv
    hyena = HyenaEmbedder() if use_hyena else None
    models = ["gc_len", "2mer", "4mer"] + (["hyena_tiny"] if use_hyena else [])
    perts = ["identity", "gc_shuffle", "dinuc_shuffle"]

    seqs, y, meta = load_real()
    rows, drift = run_setting("real_promoter", seqs, y, models, perts, hyena)
    mseqs, my = gc_matched_subset(seqs, y, meta, n_per=180)
    r2, d2 = run_setting("real_gc_matched", mseqs, my, models, perts, hyena)
    rows += r2
    drift += d2

    raw = pd.DataFrame(rows)
    summary = (raw.groupby(["model", "perturbation", "setting"])
               .agg(auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"),
                    acc_mean=("acc", "mean"), acc_std=("acc", "std"), n=("auroc", "size"))
               .reset_index())
    tag = "" if use_hyena else "_nohyena"
    raw.to_csv(RES / f"real_raw{tag}.csv", index=False)
    summary.to_csv(RES / f"real_summary{tag}.csv", index=False)
    pd.DataFrame(drift).to_csv(RES / f"real_cosine{tag}.csv", index=False)
    (RES / f"real_meta{tag}.json").write_text(json.dumps({
        "source": "Genomic Benchmarks human_nontata_promoters (train split)",
        "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum()),
        "n_gc_matched_pairs": int(len(my) // 2), "seeds": SEEDS, "models": models}, indent=2))
    pd.set_option("display.width", 200)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
