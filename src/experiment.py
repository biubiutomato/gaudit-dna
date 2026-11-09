#!/usr/bin/env python3
"""Genomic representation shortcut audit.

Controlled perturbations that preserve low-order composition while
destroying sequence organization, applied to compositional baselines
and a frozen genomic foundation model (HyenaDNA-tiny).
"""
from __future__ import annotations

import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, label_binarize
except ImportError:
    LogisticRegression = None

ROOT = Path(os.environ.get("GAUDIT_ROOT", Path(__file__).resolve().parents[1]))
FIG = ROOT / "figures"
RES = ROOT / "results"
DATA = ROOT / "data"
for p in (FIG, RES, DATA):
    p.mkdir(parents=True, exist_ok=True)

SEED = 42
N_PER_CLASS = 180
SEQ_LEN = 256
CLASSES = ("promoter", "coding", "background")
COMPLEMENT = str.maketrans("ACGT", "TGCA")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def gc_content(seq: str) -> float:
    if not seq:
        return 0.0
    return (seq.count("G") + seq.count("C")) / len(seq)


def sample_from_gc(length: int, gc: float, rng: random.Random) -> str:
    p_gc = min(max(gc, 0.05), 0.95)
    bases = []
    for _ in range(length):
        if rng.random() < p_gc:
            bases.append(rng.choice("GC"))
        else:
            bases.append(rng.choice("AT"))
    return "".join(bases)


def plant_motif(seq: str, motif: str, pos: int) -> str:
    pos = max(0, min(pos, len(seq) - len(motif)))
    return seq[:pos] + motif + seq[pos + len(motif) :]


PROMOTER_MOTIFS = [
    ("TATAWAW", 0.55),  # TATA box-like; W later expanded
    ("GGGCGG", 0.35),  # GC box / Sp1
    ("CCAAT", 0.30),
    ("YYANWYY", 0.25),  # initiator-like
    ("CGCG", 0.45),
]
IUPAC = {
    "A": "A",
    "C": "C",
    "G": "G",
    "T": "T",
    "W": "AT",
    "Y": "CT",
    "N": "ACGT",
    "S": "GC",
    "R": "AG",
    "K": "GT",
    "M": "AC",
}


def expand_iupac(motif: str, rng: random.Random) -> str:
    return "".join(rng.choice(IUPAC.get(ch, "ACGT")) for ch in motif)


CODON_TABLE_RICH = [
    "ATG",
    "GAA",
    "GAG",
    "AAA",
    "AAG",
    "CTG",
    "TTC",
    "TAC",
    "TGG",
    "CGC",
    "CGT",
    "GCC",
    "GCT",
    "GTG",
    "ATC",
    "AAC",
    "CAG",
    "GAC",
    "TCC",
    "AGC",
]
STOP = ["TAA", "TAG", "TGA"]


def make_promoter(length: int, rng: random.Random) -> str:
    gc = rng.uniform(0.52, 0.72)
    seq = sample_from_gc(length, gc, rng)
    # plant several regulatory motifs at stereotyped offsets
    seq = plant_motif(seq, expand_iupac("TATAWAW", rng), length // 2 - 30)
    if rng.random() < 0.7:
        seq = plant_motif(seq, expand_iupac("GGGCGG", rng), length // 2 - 80)
    if rng.random() < 0.6:
        seq = plant_motif(seq, expand_iupac("CCAAT", rng), length // 2 - 110)
    if rng.random() < 0.8:
        seq = plant_motif(seq, expand_iupac("YYANWYY", rng), length // 2)
    # extra CpG islands
    for _ in range(rng.randint(3, 8)):
        pos = rng.randint(0, length - 4)
        seq = plant_motif(seq, "CGCG" if rng.random() < 0.5 else "GCGC", pos)
    return seq[:length]


def make_coding(length: int, rng: random.Random) -> str:
    # ORF-like: ATG + codon run + avoid stops in frame
    n_codons = length // 3
    parts = ["ATG"]
    for _ in range(n_codons - 2):
        codon = rng.choice(CODON_TABLE_RICH)
        parts.append(codon)
    parts.append(rng.choice(STOP))
    seq = "".join(parts)
    if len(seq) < length:
        seq += sample_from_gc(length - len(seq), 0.48, rng)
    return seq[:length]


def make_background(length: int, rng: random.Random) -> str:
    # matched-ish genomic background, lower motif density, variable GC
    gc = rng.uniform(0.32, 0.52)
    seq = sample_from_gc(length, gc, rng)
    # occasional isolated motifs so the task is not trivial GC-only
    if rng.random() < 0.15:
        seq = plant_motif(seq, expand_iupac("TATAWAW", rng), rng.randint(10, length - 20))
    return seq[:length]


GENERATORS = {
    "promoter": make_promoter,
    "coding": make_coding,
    "background": make_background,
}


def build_corpus(n_per_class: int = N_PER_CLASS, length: int = SEQ_LEN, seed: int = SEED):
    rng = random.Random(seed)
    rows = []
    for cls in CLASSES:
        gen = GENERATORS[cls]
        for i in range(n_per_class):
            seq = gen(length, rng)
            rows.append(
                {
                    "id": f"{cls}_{i:04d}",
                    "label": cls,
                    "seq": seq,
                    "gc": gc_content(seq),
                    "length": len(seq),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(DATA / "corpus.csv", index=False)
    return df


# -------------------- perturbations --------------------

def gc_preserving_shuffle(seq: str, rng: random.Random) -> str:
    chars = list(seq)
    rng.shuffle(chars)
    return "".join(chars)


def dinucleotide_preserving_shuffle(seq: str, rng: random.Random, attempts: int = 40) -> str:
    """Altschul–Erikson style edge-shuffle on the dinucleotide graph."""
    if len(seq) < 3:
        return seq
    s = list(seq)
    n = len(s)
    # try random swaps that preserve dinucleotide counts approximately via
    # Eulerian-path reconstruction
    nodes = "ACGT"
    succs = {a: [] for a in nodes}
    for i in range(n - 1):
        succs[s[i]].append(s[i + 1])
    for a in nodes:
        rng.shuffle(succs[a])
    start = s[0]
    path = [start]
    # Hierholzer-like greedy walk; fall back to GC shuffle if it fails
    local = {a: list(succs[a]) for a in nodes}
    cur = start
    used = 0
    target = n - 1
    while used < target and local[cur]:
        nxt = local[cur].pop()
        path.append(nxt)
        cur = nxt
        used += 1
    if len(path) == n:
        return "".join(path)
    # fallback: many random transpositions that keep GC (not exact 2-mer)
    return gc_preserving_shuffle(seq, rng)


def exact_dinuc_shuffle(seq: str, rng: random.Random) -> str:
    """More reliable Euler reconstruction with retries."""
    if len(seq) < 4:
        return seq
    for _ in range(80):
        succs = defaultdict(list)
        for a, b in zip(seq, seq[1:]):
            succs[a].append(b)
        for k in succs:
            rng.shuffle(succs[k])
        path = [seq[0]]
        cur = seq[0]
        ok = True
        for _step in range(len(seq) - 1):
            if not succs[cur]:
                ok = False
                break
            nxt = succs[cur].pop()
            path.append(nxt)
            cur = nxt
        if ok and len(path) == len(seq):
            out = "".join(path)
            if out != seq or rng.random() < 0.05:
                return out
    return gc_preserving_shuffle(seq, rng)


def truncate(seq: str, keep: float = 0.5) -> str:
    k = max(16, int(len(seq) * keep))
    return seq[:k]


def reverse_seq(seq: str) -> str:
    return seq[::-1]


def reverse_complement(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]


def random_mismatch(seq: str, rate: float, rng: random.Random) -> str:
    alphabet = "ACGT"
    out = []
    for ch in seq:
        if rng.random() < rate:
            choices = [b for b in alphabet if b != ch]
            out.append(rng.choice(choices))
        else:
            out.append(ch)
    return "".join(out)


PERTURBATIONS = {
    "identity": lambda s, rng: s,
    "gc_shuffle": lambda s, rng: gc_preserving_shuffle(s, rng),
    "dinuc_shuffle": lambda s, rng: exact_dinuc_shuffle(s, rng),
    "trunc_50": lambda s, rng: truncate(s, 0.50),
    "trunc_25": lambda s, rng: truncate(s, 0.25),
    "reverse": lambda s, rng: reverse_seq(s),
    "revcomp": lambda s, rng: reverse_complement(s),
    "mismatch_10": lambda s, rng: random_mismatch(s, 0.10, rng),
}


# -------------------- embeddings --------------------

def all_kmers(k: int):
    alphabet = "ACGT"
    if k == 1:
        return list(alphabet)
    prev = all_kmers(k - 1)
    return [p + a for p in prev for a in alphabet]


K2 = all_kmers(2)
K3 = all_kmers(3)
K4 = all_kmers(4)


def kmer_freq(seq: str, k: int, vocab) -> np.ndarray:
    counts = Counter(seq[i : i + k] for i in range(len(seq) - k + 1))
    total = max(sum(counts.values()), 1)
    return np.array([counts.get(w, 0) / total for w in vocab], dtype=np.float32)


def gc_len_vec(seq: str, ref_len: int = SEQ_LEN) -> np.ndarray:
    return np.array([gc_content(seq), len(seq) / ref_len], dtype=np.float32)


class HyenaEmbedder:
    def __init__(self):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(
            "LongSafari/hyenadna-tiny-1k-seqlen-hf", trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            "LongSafari/hyenadna-tiny-1k-seqlen-hf", trust_remote_code=True
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def embed_one(self, seq: str) -> np.ndarray:
        torch = self.torch
        ids = self.tok(seq, return_tensors="pt")
        with torch.no_grad():
            out = self.model(**ids)
            h = out.last_hidden_state[0]  # [L, D]
            # mean pool excluding last token if it is EOS-like
            vec = h.mean(dim=0).cpu().numpy().astype(np.float32)
        return vec

    def embed_many(self, seqs) -> np.ndarray:
        out = []
        for i, s in enumerate(seqs):
            out.append(self.embed_one(s))
            if (i + 1) % 120 == 0:
                print(f"    hyena {i+1}/{len(seqs)}")
        return np.stack(out, axis=0)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def embed_corpus(seqs, kind: str, hyena=None) -> np.ndarray:
    if kind == "gc_len":
        return np.stack([gc_len_vec(s) for s in seqs])
    if kind == "1mer":
        return np.stack([kmer_freq(s, 1, list("ACGT")) for s in seqs])
    if kind == "2mer":
        return np.stack([kmer_freq(s, 2, K2) for s in seqs])
    if kind == "3mer":
        return np.stack([kmer_freq(s, 3, K3) for s in seqs])
    if kind == "4mer":
        return np.stack([kmer_freq(s, 4, K4) for s in seqs])
    if kind == "hyena_tiny":
        assert hyena is not None
        return hyena.embed_many(seqs)
    raise ValueError(kind)


def pairwise_mean_cosine(X: np.ndarray, n_pairs: int = 400, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    n = len(X)
    sims = []
    for _ in range(n_pairs):
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        sims.append(cosine(X[i], X[j]))
    return float(np.mean(sims)) if sims else 0.0


def class_probe(X: np.ndarray, y: np.ndarray, seed: int = 0):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.35, random_state=seed, stratify=y
    )
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    clf = LogisticRegression(max_iter=400)
    clf.fit(Xtr_s, ytr)
    pred = clf.predict(Xte_s)
    acc = accuracy_score(yte, pred)
    yte_b = label_binarize(yte, classes=CLASSES)
    proba = clf.predict_proba(Xte_s)
    # align columns to CLASSES
    col = {c: i for i, c in enumerate(clf.classes_)}
    P = np.column_stack([proba[:, col[c]] for c in CLASSES])
    try:
        auroc = roc_auc_score(yte_b, P, average="macro", multi_class="ovr")
    except Exception:
        auroc = float("nan")
    return {"acc": float(acc), "auroc": float(auroc)}


def run():
    set_seed(SEED)
    print("Building corpus...")
    df = build_corpus()
    print(df.groupby("label")["gc"].mean())
    seqs = df["seq"].tolist()
    labels = df["label"].to_numpy()
    y = labels

    print("Loading HyenaDNA-tiny...")
    hyena = HyenaEmbedder()

    models = ["gc_len", "1mer", "2mer", "3mer", "4mer", "hyena_tiny"]
    rng = random.Random(SEED)

    # precompute original embeddings
    orig_emb = {}
    for m in models:
        print(f"  embed original [{m}]")
        orig_emb[m] = embed_corpus(seqs, m, hyena=hyena)

    # random between-sequence baseline
    random_sim = {m: pairwise_mean_cosine(orig_emb[m], 500, 1) for m in models}

    # same-class vs different-class biological relatedness
    bio_sim = {}
    for m in models:
        same, diff = [], []
        X = orig_emb[m]
        idx = {c: np.where(y == c)[0] for c in CLASSES}
        rng_np = np.random.default_rng(7)
        for c in CLASSES:
            ids = idx[c]
            for _ in range(200):
                i, j = rng_np.choice(ids, 2, replace=False)
                same.append(cosine(X[i], X[j]))
            others = np.concatenate([idx[k] for k in CLASSES if k != c])
            for _ in range(200):
                i = rng_np.choice(ids)
                j = rng_np.choice(others)
                diff.append(cosine(X[i], X[j]))
        bio_sim[m] = {
            "same_class": float(np.mean(same)),
            "diff_class": float(np.mean(diff)),
        }

    # perturbation cosine tables
    pert_names = [k for k in PERTURBATIONS if k != "identity"]
    records = []
    probe_records = []

    # original probe
    for m in models:
        stats = class_probe(orig_emb[m], y, seed=0)
        probe_records.append({"model": m, "perturbation": "identity", **stats})

    for pname in pert_names:
        print(f"Perturbation: {pname}")
        fn = PERTURBATIONS[pname]
        pert_seqs = [fn(s, random.Random(SEED + i)) for i, s in enumerate(seqs)]
        for m in models:
            print(f"  embed {m}")
            Xp = embed_corpus(pert_seqs, m, hyena=hyena)
            Xo = orig_emb[m]
            sims = [cosine(Xo[i], Xp[i]) for i in range(len(seqs))]
            records.append(
                {
                    "model": m,
                    "perturbation": pname,
                    "mean_sim": float(np.mean(sims)),
                    "median_sim": float(np.median(sims)),
                    "std_sim": float(np.std(sims)),
                    "p10": float(np.percentile(sims, 10)),
                    "p90": float(np.percentile(sims, 90)),
                    "random_pair_sim": random_sim[m],
                    "same_class_sim": bio_sim[m]["same_class"],
                    "diff_class_sim": bio_sim[m]["diff_class"],
                }
            )
            stats = class_probe(Xp, y, seed=0)
            probe_records.append({"model": m, "perturbation": pname, **stats})
            # also store per-class mean sim
            for c in CLASSES:
                mask = y == c
                records[-1][f"sim_{c}"] = float(np.mean(np.array(sims)[mask]))

    drift = pd.DataFrame(records)
    probe = pd.DataFrame(probe_records)
    drift.to_csv(RES / "cosine_drift.csv", index=False)
    probe.to_csv(RES / "linear_probe.csv", index=False)
    with open(RES / "random_and_bio_sim.json", "w") as f:
        json.dump({"random": random_sim, "bio": bio_sim}, f, indent=2)

    # shortcut sensitivity = sim(x, T_shortcut(x)) / sim_same_class
    ss_rows = []
    for m in models:
        sc = bio_sim[m]["same_class"]
        for pname in ("gc_shuffle", "dinuc_shuffle"):
            row = drift[(drift.model == m) & (drift.perturbation == pname)].iloc[0]
            ss_rows.append(
                {
                    "model": m,
                    "perturbation": pname,
                    "shortcut_sensitivity": row.mean_sim / max(sc, 1e-6),
                    "mean_sim": row.mean_sim,
                    "same_class_sim": sc,
                }
            )
    ss = pd.DataFrame(ss_rows)
    ss.to_csv(RES / "shortcut_sensitivity.csv", index=False)

    make_figures(drift, probe, models, pert_names)
    summary = {
        "n_sequences": int(len(df)),
        "seq_len": SEQ_LEN,
        "classes": list(CLASSES),
        "models": models,
        "perturbations": pert_names,
        "hyena_dim": int(orig_emb["hyena_tiny"].shape[1]),
        "note": "HyenaDNA-tiny-1k frozen mean-pooled embeddings; compositional baselines.",
    }
    with open(RES / "run_meta.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Done. Wrote results to", RES)


def make_figures(drift: pd.DataFrame, probe: pd.DataFrame, models, pert_names):
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "figure.dpi": 160,
        }
    )
    model_labels = {
        "gc_len": "GC+len",
        "1mer": "1-mer",
        "2mer": "2-mer",
        "3mer": "3-mer",
        "4mer": "4-mer",
        "hyena_tiny": "HyenaDNA-tiny",
    }
    pert_labels = {
        "gc_shuffle": "GC shuffle",
        "dinuc_shuffle": "Dinuc. shuffle",
        "trunc_50": "Trunc. 50%",
        "trunc_25": "Trunc. 25%",
        "reverse": "Reverse",
        "revcomp": "Rev-comp",
        "mismatch_10": "10% mismatch",
    }

    # Figure 1: heatmap of mean cosine
    mat = np.zeros((len(models), len(pert_names)))
    for i, m in enumerate(models):
        for j, p in enumerate(pert_names):
            mat[i, j] = drift[(drift.model == m) & (drift.perturbation == p)].mean_sim.iloc[0]
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    im = ax.imshow(mat, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(pert_names)))
    ax.set_xticklabels([pert_labels[p] for p in pert_names], rotation=30, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([model_labels[m] for m in models])
    for i in range(len(models)):
        for j in range(len(pert_names)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if mat[i, j] < 0.45 else "black")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="mean cos(E(x), E(T(x)))")
    ax.set_title("Embedding stability under sequence-preserving perturbations")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_cosine_heatmap.png")
    fig.savefig(FIG / "fig1_cosine_heatmap.pdf")
    plt.close(fig)

    # Figure 2: grouped bars for key perts vs random pair
    key = ["gc_shuffle", "dinuc_shuffle", "trunc_50", "revcomp"]
    x = np.arange(len(models))
    width = 0.16
    fig, ax = plt.subplots(figsize=(8.2, 3.5))
    for k, p in enumerate(key):
        vals = [
            drift[(drift.model == m) & (drift.perturbation == p)].mean_sim.iloc[0]
            for m in models
        ]
        ax.bar(x + (k - 1.5) * width, vals, width, label=pert_labels[p])
    rand_vals = [
        drift[drift.model == m].random_pair_sim.iloc[0] for m in models
    ]
    ax.plot(x, rand_vals, "k--", marker="o", linewidth=1, label="random pair")
    ax.set_xticks(x)
    ax.set_xticklabels([model_labels[m] for m in models], rotation=15, ha="right")
    ax.set_ylabel("cosine similarity")
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=3, fontsize=8, frameon=False)
    ax.set_title("Perturbed-pair similarity versus unmatched random pairs")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_bars_vs_random.png")
    fig.savefig(FIG / "fig2_bars_vs_random.pdf")
    plt.close(fig)

    # Figure 3: linear probe AUROC original vs perturbed
    fig, ax = plt.subplots(figsize=(8.2, 3.5))
    probe_key = ["identity", "gc_shuffle", "dinuc_shuffle", "trunc_50", "revcomp"]
    width = 0.15
    x = np.arange(len(models))
    for k, p in enumerate(probe_key):
        vals = []
        for m in models:
            sub = probe[(probe.model == m) & (probe.perturbation == p)]
            vals.append(sub.auroc.iloc[0] if len(sub) else np.nan)
        label = "original" if p == "identity" else pert_labels.get(p, p)
        ax.bar(x + (k - 2) * width, vals, width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([model_labels[m] for m in models], rotation=15, ha="right")
    ax.set_ylabel("macro AUROC")
    ax.set_ylim(0.45, 1.05)
    ax.axhline(1 / 3, color="gray", ls=":", lw=1)
    ax.legend(ncol=3, fontsize=8, frameon=False)
    ax.set_title("Frozen linear probe: biological class information after perturbation")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_probe_auroc.png")
    fig.savefig(FIG / "fig3_probe_auroc.pdf")
    plt.close(fig)

    # Figure 4: shortcut sensitivity
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ss_gc = []
    ss_di = []
    for m in models:
        r1 = drift[(drift.model == m) & (drift.perturbation == "gc_shuffle")].iloc[0]
        r2 = drift[(drift.model == m) & (drift.perturbation == "dinuc_shuffle")].iloc[0]
        ss_gc.append(r1.mean_sim / max(r1.same_class_sim, 1e-6))
        ss_di.append(r2.mean_sim / max(r2.same_class_sim, 1e-6))
    x = np.arange(len(models))
    ax.bar(x - 0.18, ss_gc, 0.36, label="GC shuffle")
    ax.bar(x + 0.18, ss_di, 0.36, label="Dinucleotide shuffle")
    ax.axhline(1.0, color="black", ls="--", lw=1, label="same-class baseline")
    ax.set_xticks(x)
    ax.set_xticklabels([model_labels[m] for m in models], rotation=15, ha="right")
    ax.set_ylabel(r"S(x, T(x)) / S_same-class")
    ax.set_title("Shortcut sensitivity (higher = more composition-locked)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_shortcut_sensitivity.png")
    fig.savefig(FIG / "fig4_shortcut_sensitivity.pdf")
    plt.close(fig)


if __name__ == "__main__":
    run()
