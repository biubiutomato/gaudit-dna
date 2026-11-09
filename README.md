# G-Audit

Code, data, and results for:

> Ye Ye. **Composition or Grammar? A Shuffle-Based Audit of Frozen Genomic Sequence Representations.**
> IEEE BIBM 2026 Workshops (paper S37209).

G-Audit perturbs DNA sequences with composition-preserving shuffles (mononucleotide- and dinucleotide-preserving)
and with truncation, reversal, reverse complement, and random substitution, and compares two readouts on frozen
embeddings: the cosine displacement of each embedding, and the AUROC of a linear probe refitted on the perturbed
embeddings. Every experiment is paired with a GC-matched version so that composition cannot explain the signal.
Encoders: HyenaDNA-tiny (`LongSafari/hyenadna-tiny-1k-seqlen-hf`, mean-pooled, frozen), GC+length, and 1–4-mer spectra.

## Run (CPU, about 5–15 minutes)

```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
bash run_all.sh
```

The first run downloads HyenaDNA-tiny and the Genomic Benchmarks `human_nontata_promoters` dataset from Hugging Face.
If HyenaDNA fails to load, install `transformers==4.38.2`. Add `--no-hyena` to a script to run only the compositional baselines.
The natural-composition experiment (Tables II–III, Fig. 1) is produced by `python src/experiment.py`.

## Contents

| Path | Contents |
|---|---|
| `src/experiment.py` | Synthetic corpora, perturbations, encoders, cosine drift, natural-composition probes |
| `src/gc_matched_probe.py` | GC-matched synthetic corpus (all classes 256 nt, GC 0.50) and probes |
| `src/real_promoters.py` | Real human non-TATA promoters: sampling, GC matching, probes over three split seeds, cosine drift |
| `src/probe_utils.py` | Binary probe, stratified split, GC matching |
| `src/make_figs.py` | Paper figures from `results/` |
| `data/corpus.csv`, `data/corpus_gc_matched.csv` | Synthetic corpora |
| `results/` | All result tables used in the paper |
| `figures/` | Paper figures |

## Citation

```bibtex
@inproceedings{ye2026gaudit,
  author    = {Ye, Ye},
  title     = {Composition or Grammar? A Shuffle-Based Audit of Frozen Genomic Sequence Representations},
  booktitle = {Proc. IEEE International Conference on Bioinformatics and Biomedicine (BIBM) Workshops},
  year      = {2026}
}
```

## License

Code: MIT (see `LICENSE`).
