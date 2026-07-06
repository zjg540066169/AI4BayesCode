#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# Single source of truth -> the two installable packages.
#
# EDIT ONLY the canonical copies at the repo ROOT:
#     skills/                  the AI codegen skill corpus
#     include/AI4BayesCode/    the core C++ library headers
#     examples/ , start.md
# then run:   bash tools/sync.sh
#
# This regenerates the committed copies that the R package (r-pkg/inst/…) and
# the Python package (python/AI4BayesCode/…) ship, so `install_github` /
# `pip install git+…` users always get the current skills + headers.
#
# The stable vendored C++ dependencies (eigen, autodiff, mcmclib,
# bart_pure_cpp, celerite, libgp_kernels) are committed as-is and only need a
# refresh when you actually bump a vendored dependency — they are NOT touched
# here, to keep the common (skills/header) edit fast and safe.
# ─────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"

sync_dir () { rm -rf "$2"; rsync -a --exclude '.DS_Store' "$1/" "$2/"; }

echo "• skills/        -> r-pkg/inst/skills , python/_skills"
sync_dir skills r-pkg/inst/skills
sync_dir skills python/AI4BayesCode/_skills

echo "• include/ (+ celerite, libgp_kernels relocated from the repo root) -> r-pkg/inst/include"
sync_dir include r-pkg/inst/include
rsync -a --exclude '.DS_Store' celerite/      r-pkg/inst/include/celerite/
rsync -a --exclude '.DS_Store' libgp_kernels/ r-pkg/inst/include/libgp_kernels/
echo "• include/AI4BayesCode -> python/_vendored_include/AI4BayesCode (core headers)"
sync_dir include/AI4BayesCode python/AI4BayesCode/_vendored_include/AI4BayesCode

echo "• bart_pure_cpp/ -> r-pkg/inst ; start.md -> both"
sync_dir bart_pure_cpp  r-pkg/inst/bart_pure_cpp
cp -f start.md r-pkg/inst/start.md
cp -f start.md python/AI4BayesCode/start.md

# examples/ are now UNIFIED tri-module .cpp: each file carries a fenced
# int main() + an #ifdef AI4BAYESCODE_RCPP_MODULE block + an
# #ifdef AI4BAYESCODE_PYBIND_MODULE block, so the SAME file runs standalone,
# in R (new() / ai4bayescode_example), AND in Python (pybind). Single source of
# truth = repo-root examples/; the .cpp are synced to BOTH packages here (edit
# once, in examples/, then run this script). R-only helper scripts under
# r-pkg/inst/examples (run_*.R / test_*.R / Makevars) are NOT part of the .cpp
# source-of-truth and are left untouched by this loop.
echo "• examples/*.cpp -> r-pkg/inst/examples , python/AI4BayesCode/_examples (tri-module)"
for f in examples/*.cpp; do
  cp -f "$f" "r-pkg/inst/examples/$(basename "$f")"
  cp -f "$f" "python/AI4BayesCode/_examples/$(basename "$f")"
done

find r-pkg/inst python/AI4BayesCode/_skills python/AI4BayesCode/_vendored_include \
     -name '.DS_Store' -delete 2>/dev/null || true
echo "✓ Synced. Commit the regenerated copies under r-pkg/inst and python/AI4BayesCode."
