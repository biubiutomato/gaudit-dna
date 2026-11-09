#!/usr/bin/env bash
# One-command rerun. CPU only; needs internet access to huggingface.co on the first run.
set -euo pipefail
cd "$(dirname "$0")/src"
echo "== 1/3 GC-matched synthetic corpus (all classes 256 nt), incl. HyenaDNA-tiny"
python gc_matched_probe.py
echo "== 2/3 Real human non-TATA promoters, incl. HyenaDNA-tiny"
python real_promoters.py
echo "== 3/3 Figures"
python make_figs.py
echo "Done. Send back the results/ folder (and figures/ if you like)."
