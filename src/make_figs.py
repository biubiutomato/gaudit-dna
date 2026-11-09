"""Regenerate paper figures from the saved result CSVs (no titles; captions carry them)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#555555", "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42})
C = {"identity": "#2a78d6", "gc_shuffle": "#eb6834", "dinuc_shuffle": "#1baf7a"}
H = {"identity": "", "gc_shuffle": "////", "dinuc_shuffle": "\\\\\\\\"}
PL = {"identity": "Original", "gc_shuffle": "GC-preserving shuffle", "dinuc_shuffle": "Dinuc.-preserving shuffle"}
ML = {"gc_len": "GC+len", "1mer": "1-mer", "2mer": "2-mer", "3mer": "3-mer", "4mer": "4-mer", "hyena_tiny": "HyenaDNA-tiny"}
MLB = dict(ML, hyena_tiny="HyenaDNA\n-tiny")
PERTS = ["identity", "gc_shuffle", "dinuc_shuffle"]

def bars(ax, df, models, err=None):
    x = np.arange(len(models)); w = 0.26
    for k, p in enumerate(PERTS):
        v = [df[(df.model == m) & (df.perturbation == p)].iloc[0] for m in models]
        mean = [r["auroc_mean"] if "auroc_mean" in r else r["auroc"] for r in v]
        sd = [r["auroc_std"] for r in v] if "auroc_std" in v[0] else None
        ax.bar(x + (k - 1) * w, mean, w * 0.92, yerr=sd, capsize=2, error_kw={"lw": 0.7, "ecolor": "#333"},
               color=C[p], edgecolor="white", hatch=H[p], linewidth=0, label=PL[p])
    ax.set_xticks(x); ax.set_xticklabels([MLB[m] for m in models])
    ax.axhline(0.5, color="#888", ls=":", lw=0.8)
    ax.set_ylim(0.4, 1.02); ax.grid(axis="y", color="#e6e6e6", lw=0.5); ax.set_axisbelow(True)

# Fig 1: cosine heatmap
d = pd.read_csv(RES / "cosine_drift.csv")
pert = ["gc_shuffle", "dinuc_shuffle", "trunc_50", "trunc_25", "reverse", "revcomp", "mismatch_10"]
plab = ["GC shuf.", "Dinuc. shuf.", "Trunc. 50%", "Trunc. 25%", "Reverse", "Rev.-comp.", "10% mism."]
models = ["gc_len", "1mer", "2mer", "3mer", "4mer", "hyena_tiny"]
M = np.array([[d[(d.model == m) & (d.perturbation == p)].mean_sim.iloc[0] for p in pert] for m in models])
cmap = LinearSegmentedColormap.from_list("b", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
fig, ax = plt.subplots(figsize=(3.45, 2.1))
im = ax.imshow(M, cmap=cmap, vmin=0.5, vmax=1.0, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f"{M[i,j]:.2f}" if M[i,j] < 0.995 else f"{M[i,j]:.3f}", ha="center", va="center",
                fontsize=6.3, color="white" if M[i,j] > 0.8 else "#111")
ax.set_xticks(range(len(pert))); ax.set_xticklabels(plab, rotation=35, ha="right")
ax.set_yticks(range(len(models))); ax.set_yticklabels([ML[m] for m in models])
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=6.5)
cb.set_label(r"mean $\cos(E(x),E(T(x)))$", fontsize=7)
fig.tight_layout(pad=0.3); fig.savefig(OUT / "fig_cosine.pdf"); plt.close(fig)

# Fig 2: GC-matched synthetic probe
g = pd.read_csv(RES / "gc_matched_probe.csv")
fig, ax = plt.subplots(figsize=(3.45, 2.05))
bars(ax, g, [m for m in ["gc_len", "2mer", "3mer", "4mer", "hyena_tiny"] if m in set(g.model)])
ax.set_ylabel("Macro AUROC")
fig.legend(*ax.get_legend_handles_labels(), frameon=False, ncol=3, loc="upper center", handlelength=1.3, columnspacing=0.7, fontsize=6.5)
fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.9)); fig.savefig(OUT / "fig_gcmatched_synth.pdf"); plt.close(fig)

# Fig 3: real promoters, raw vs GC-matched
s = pd.read_csv(RES / "real_summary.csv") if (RES / "real_summary.csv").exists() else pd.read_csv(RES / "enhanced_summary.csv")
real_models = [m for m in ["gc_len", "2mer", "4mer", "hyena_tiny"] if m in set(s.model)]
fig, axs = plt.subplots(1, 2, figsize=(3.45 if len(real_models) < 4 else 7.0, 2.15), sharey=True)
for ax, st, t in zip(axs, ["real_promoter", "real_gc_matched"], ["(a) Full set, $n=560$", "(b) GC-matched, 153 pairs"]):
    bars(ax, s[s.setting == st], real_models)
    ax.set_xlabel(t, fontsize=7.5)
axs[0].set_ylabel("AUROC (mean $\\pm$ sd, 3 seeds)")
fig.legend(*axs[0].get_legend_handles_labels(), frameon=False, ncol=3, loc="upper center", handlelength=1.3, columnspacing=0.7, fontsize=6.5)
fig.tight_layout(pad=0.3, w_pad=0.6, rect=(0, 0, 1, 0.9)); fig.savefig(OUT / "fig_real.pdf"); plt.close(fig)
print("ok")
