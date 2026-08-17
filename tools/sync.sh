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

# Editing backups and the local _archive/ tree are never product; without
# these excludes a `pip install .` / R build from the working tree picks up
# ~312 KB of old header revisions.
sync_dir () { rm -rf "$2"; rsync -a --exclude '.DS_Store' --exclude '*.bak*' \
                            --exclude '_archive' "$1/" "$2/"; }

# The shipped skill corpus is what a USER receives. Two classes of file live
# under skills/ but are not product and must not ship:
#   sim*.md / sim*_workflow.md -- our own paper-experiment protocol
#   *.bak*                     -- local editing backups
# ai4bayescode_list_skills() / AI4BayesCode.list_skills() glob the shipped
# directory, so anything copied here shows up in a user's skill listing.
# Patterns, not a list of today's filenames: a literal set silently ships the
# next sim file somebody adds (probed: sim2.md and sim4_workflow.md both
# reached BOTH packages under the old four-name list).
SKILL_EXCLUDES=(--exclude 'sim*' --exclude '*.bak*' --exclude '* [0-9].*')

sync_skills () {
  rm -rf "$1"
  rsync -a --exclude '.DS_Store' "${SKILL_EXCLUDES[@]}" skills/ "$1"/
  # HARD GATE. The sim* files are our paper-experiment protocol, never product.
  # An exclude PATTERN can be outrun by a new naming convention, so assert the
  # outcome instead of trusting the rule: anything named sim* that reached the
  # package is a build failure, not a warning.
  local leaked
  leaked=$(find "$1" -iname 'sim*' -print 2>/dev/null || true)
  if [ -n "$leaked" ]; then
    echo "✗ experiment material reached $1 -- these must NEVER ship:"
    echo "$leaked" | sed 's/^/      /'
    exit 1
  fi
}

echo "• skills/        -> r-pkg/inst/skills , python/_skills (experiment + backup files excluded)"
sync_skills r-pkg/inst/skills
sync_skills python/AI4BayesCode/_skills

echo "• include/ (+ celerite, libgp_kernels relocated from the repo root) -> r-pkg/inst/include"
sync_dir include r-pkg/inst/include
rsync -a --exclude '.DS_Store' celerite/      r-pkg/inst/include/celerite/
rsync -a --exclude '.DS_Store' libgp_kernels/ r-pkg/inst/include/libgp_kernels/
echo "• include/AI4BayesCode -> python/_vendored_include/AI4BayesCode (core headers)"
sync_dir include/AI4BayesCode python/AI4BayesCode/_vendored_include/AI4BayesCode
# Vendored C++ deps that the core headers #include must ALSO reach the python
# package, or python/_vendored_include/mcmclib drifts stale vs the synced
# nuts_block.hpp (e.g. nuts_block.hpp references nuts_settings_t.precond_cache_valid
# but a stale mcmclib lacks the field -> compile error). r-pkg gets these via the
# whole-tree `sync_dir include r-pkg/inst/include` above; python needs them listed
# explicitly because it only vendors include/AI4BayesCode by default.
echo "• include/{mcmclib,eigen,autodiff,BaseMatrixOps} -> python/_vendored_include (vendored deps)"
sync_dir include/mcmclib      python/AI4BayesCode/_vendored_include/mcmclib
sync_dir include/eigen        python/AI4BayesCode/_vendored_include/eigen
sync_dir include/autodiff     python/AI4BayesCode/_vendored_include/autodiff
sync_dir include/BaseMatrixOps python/AI4BayesCode/_vendored_include/BaseMatrixOps

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

# Copying file-by-file cannot notice a DELETION, so an example removed from
# examples/ would live on in both packages as an orphan -- still listed by
# ai4bayescode_list_examples() / AI4BayesCode.examples_path(), still shipped.
# (The include/ and skills/ trees are safe: sync_dir rm -rf's the destination
# first. Only this loop copies in place, because r-pkg/inst/examples also holds
# hand-maintained run_*.R / test_*.R / Makevars that must NOT be wiped.)
for d in r-pkg/inst/examples python/AI4BayesCode/_examples; do
  for f in "$d"/*.cpp; do
    [ -e "$f" ] || continue
    if [ ! -e "examples/$(basename "$f")" ]; then
      echo "  - dropping orphaned $f (no longer in examples/)"
      rm -f "$f"
    fi
  done
done

find r-pkg/inst python/AI4BayesCode/_skills python/AI4BayesCode/_vendored_include \
     -name '.DS_Store' -delete 2>/dev/null || true
# Consistency report. NOTE this runs AFTER the orphan drop above, which
# already removes every dest-only .cpp, and the copy loop guarantees the
# forward direction -- so a mismatch here means a cp/rm actually failed
# (set -e would normally have aborted first). It is a cheap backstop, not the
# primary guard; the drop loop's own log line is what tells you something
# drifted.
sync_ok=1
for d in r-pkg/inst/examples python/AI4BayesCode/_examples; do
  if ! diff -q <(ls examples/*.cpp | xargs -n1 basename | sort) \
                <(ls "$d"/*.cpp 2>/dev/null | xargs -n1 basename | sort) >/dev/null; then
    echo "  ! $d does not match examples/:"
    diff <(ls examples/*.cpp | xargs -n1 basename | sort) \
         <(ls "$d"/*.cpp 2>/dev/null | xargs -n1 basename | sort) | sed 's/^/      /'
    sync_ok=0
  fi
done
if [ "$sync_ok" -ne 1 ]; then
  echo "✗ Sync finished with mismatches above -- fix before committing."
  exit 1
fi

echo "✓ Synced. Commit the regenerated copies under r-pkg/inst and python/AI4BayesCode."
