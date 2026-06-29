# blocks_local/ — user-designed (local-tier) sampler blocks

This directory holds **local-tier** `block_sampler` primitives created via the
`block_design` skill flow (`skills/block_design_skills/`). It is the move-on-"go"
landing target for a completed block bundle.

Local tier = self-trust, work-in-progress. These blocks are **not** vetted core and
are **not** shipped to the R package: `sync_from_core.sh` deliberately does NOT
package `blocks_local/` into `inst/` (same rule as `.bak` / `_archive` dev content).
"Shipping a block" is the FUTURE registry/submission path (`contrib.md`).

## Bundle layout

Each block is a self-contained bundle. `Block` is the one identity source;
`ClassName`, header filename, and bundle dir are all derived from it by convention.

```
blocks_local/<Block>/
  <Block>.hpp              # Tier-B block: implements the block_sampler C++ contract
  test_<Block>.cpp         # library test: T0-T4 regime ladder (FD / parity / recovery / split-Rhat / stress)
  manifest.dcf             # routing + metadata (13 required fields; see skill.md §3.1)
  skills/<Block>.md        # block-local skill (cites core, never restates)
  examples/<Model>.cpp     # REQUIRED frontend-independent C++ demo (int main; no Rcpp/pybind)
  vendor/                  # OPTIONAL pinned third-party source + its LICENSE (verbatim)
```

## Naming

CRAN-flat, globally unique, `snake_case` + `_block` suffix (e.g.
`poisson_icar_ess_block`). Core block names are reserved; uniqueness is checked at
creation against core (`include/AI4BayesCode/*_block*.hpp`) + existing local bundles
(`blocks_local/*/manifest.dcf`).

## How bundles get here

The `block_design` flow stages every artifact to
`.block_design_staging/<Block>/` first and **moves** it here only
on an explicit "go" (TOP RULE). Compile + library test run on the same "go".
