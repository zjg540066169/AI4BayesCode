# Third-party licenses

This package vendors the third-party libraries listed below. Each
retains its original upstream license. Most vendored files keep their
upstream copyright/license headers **unmodified**; the exceptions are
called out explicitly in the per-component "Modifications" notes (the
BART kernel carries an added dual-copyright + header-only-merge notice,
and the SoftBART vendor files ship without an upstream header — see
those sections).

> **Combined-work license.** When AI4BayesCode is distributed as a
> whole (project code plus the vendored dependencies it compiles
> against), the effective license of the combined work is
> **GPL-3.0-or-later**. This is the project's elected license. The
> reason it must be v3 (not v2) is the Apache-2.0 components (mcmclib /
> BaseMatrixOps): Apache-2.0 is one-way compatible with GPL **v3**, not
> with GPL-2.0-only. See "Root license summary" and "Compatibility
> notes" below.

## autodiff (generation-time only)

- Upstream: https://autodiff.github.io and
  https://github.com/autodiff/autodiff
- Version: v1.1.2 (vendored 2026-04-17 from `main` at depth-1 clone)
- License: MIT — see `include/autodiff/LICENSE`
- Copyright (c) 2018–2024 Allan Leal
- What we vendored: the inner `autodiff/autodiff/{common, forward,
  reverse}` header tree. We did NOT vendor
  `autodiff/autodiff/pybind11/`, the top-level `CMakeLists.txt`,
  `tests/`, `examples/`, `docs/`, or `python/`.
- Used only by the C++ code-generation / autodiff path; not required
  to run a delivered sampler.
- Modifications: none.

## Eigen (generation-time only)

- Upstream: https://eigen.tuxfamily.org and
  https://gitlab.com/libeigen/eigen
- Version: 3.4.0
- License: MPL-2.0 (with small portions under LGPL-2.1+, BSD, Apache,
  etc.; see `include/eigen/COPYING.MPL2` and `COPYING.README`)
- What we vendored: the `Eigen/` subfolder plus `COPYING.MPL2` and
  `COPYING.README`.
- Modifications: none.

## mcmclib (NUTS backend)

- Upstream: https://github.com/kthohr/mcmc
- License: Apache-2.0 — see `include/mcmclib/` upstream license file.
- What we vendored: the `include/mcmclib/mcmc/` header tree.
- Modifications: YES — see the header comments in
  `include/mcmclib/mcmc/nuts.hpp` and `include/mcmclib/mcmc/nuts.ipp`.
  The modifications enable persistent adaptation (needed for stateful
  Gibbs-wise NUTS) and fix the upstream tree-doubling endpoint bug.

## BaseMatrixOps (linear-algebra helpers for the NUTS backend)

- Upstream: https://github.com/kthohr/BaseMatrixOps (companion to
  mcmclib).
- Version: vendored alongside mcmclib.
- License: Apache-2.0.
- What we vendored: `include/mcmclib/BaseMatrixOps/`.
- Modifications: none.
- Note: being Apache-2.0, this component (with mcmclib) is what forces
  the combined-work license to the GPL **v3** branch — see
  "Compatibility notes".

## walnuts (design reference for `nuts_kernel_v1`)

- Upstream: https://github.com/flatironinstitute/walnuts
- License: MIT.
- Methodology / reference implementation: Bou-Rabee, Carpenter, Kleppe
  & Liu (2025), the WALNUTS within-orbit / mass-adaptation scheme.
- What we use: NO walnuts source is vendored. The header-only NUTS
  kernel under `include/AI4BayesCode/nuts_kernel_v1/` is an independent
  AI4BayesCode implementation (`Copyright (C) 2026 AI4BayesCode`,
  GPL-3.0-or-later) whose mass-matrix / dual-averaging conventions were
  cross-checked against walnuts (e.g. `adaptive_walnuts.hpp`); the
  inline comments cite walnuts where a default or convention is
  matched. MIT is GPL-compatible, so even a future vendoring of walnuts
  source would be combinable into the GPL-3 combined work.
- Modifications: not applicable (no vendored source).

## BART kernel — CRAN BART R package (vanilla Gaussian BART)

- Upstream: the **CRAN `BART` R package** (NOT `dbarts`).
  Methodology: Chipman, George & McCulloch (2010), *BART: Bayesian
  Additive Regression Trees*. Software: the `BART` package by Robert
  McCulloch, Rodney Sparapani and Charles Spanbauer.
- License: GPL-2.0-or-later. The vendored headers carry the upstream
  BART notice and point to https://www.R-project.org/Licenses/GPL-2.
- Copyright (dual): `Copyright (C) 2017-2018 Robert McCulloch,
  Rodney Sparapani and Charles Spanbauer` **and**
  `Copyright (C) 2024-2026 Jungang Zou` (the header-only adaptation).
- What we vendored: `bart_pure_cpp/src/BART/` plus the wrapper
  `bart_pure_cpp/src/bart_model.h` and the unity amalgamation
  `bart_pure_cpp/src/bart_kernel_unity.h`. The composing block header is
  `include/AI4BayesCode/bart_block.hpp`.
- Modifications: YES — the upstream `.cpp` definitions were merged into
  the corresponding headers so the kernel is header-only includable.
  Each modified file carries an explicit "Modifications by Jungang
  Zou, 2024 … comply with the terms of GPL-2" notice in addition to
  the original upstream copyright line (hence the dual-copyright
  header). The core tree-kernel logic is upstream and unchanged.
- **License-inheritance warning**: any file that `#include`s
  `bart_block.hpp` (directly or transitively) inherits
  GPL-2.0-or-later from this kernel.

## genBART kernel (Linero 2022 generalized BART via RJMCMC)

- Upstream methodology: Linero, A. R. (2022) "Generalized Bayesian
  Additive Regression Trees Models: Beyond Conditional Conjugacy",
  arXiv:2202.09924.
- Implementation: `Copyright (C) 2026 AI4BayesCode` — an independent
  C++17 port, GPL-2.0-or-later (headers state "GPL v2 or later").
- What we vendored: `bart_pure_cpp/src/GENBART/` (the generalized-BART RJMCMC
  moves, Laplace-proposal and likelihood code), plus the wrapper
  `bart_pure_cpp/src/genbart_model.h`. The composing block header is
  `include/AI4BayesCode/genbart_block.hpp`.
- Embedded upstream BART infrastructure: `bart_pure_cpp/src/GENBART/BART/`
  (e.g. `randomkit.h`, `rand_draws.h`, `rtnorm.h`) is derived from the
  same CRAN `BART` package code (McCulloch / Sparapani / Spanbauer),
  GPL-2.0-or-later, kept in a `genbart::`-style separation so it
  coexists with the vanilla BART kernel.
- Embedded MIT-licensed code: the `randomkit` Mersenne-Twister
  implementation originates from Jean-Sébastien Roy (2003, after
  Matsumoto–Nishimura 1997), MIT (GPL-compatible), preserved under
  its original license.
- **License-inheritance warning**: any file that `#include`s
  `genbart_block.hpp` (directly or transitively) inherits
  GPL-2.0-or-later.

## SoftBART kernel (Linero & Yang 2018 soft trees)

- Upstream methodology: Linero, A. R. & Yang, Y. (2018) "Bayesian
  Regression Tree Ensembles that Adapt to Smoothness and Sparsity",
  *JRSS-B* 80(5):1087–1110. Reference implementation:
  https://github.com/theodds/SoftBART (GPL-2.0-or-later).
- License: GPL-2.0-or-later (consistent with the upstream SoftBART R
  package and with the rest of the BART-family kernels here).
- Copyright: the upstream SoftBART reference implementation is
  `Copyright (c) Antonio R. Linero` (with the methodology due to
  Antonio R. Linero & Yun Yang, JRSS-B 2018).
- What we vendored: `bart_pure_cpp/src/SOFTBART_VENDOR/`
  (`soft_bart.h`, `soft_bart_impl.h`, `functions.h`,
  `slice_sampler.h`), plus the wrappers `bart_pure_cpp/src/softbart_model.h`
  and `bart_pure_cpp/src/softbart_kernel_unity.h`. The composing block header
  is `include/AI4BayesCode/softbart_block.hpp`.
- **Hygiene gap (action item):** the vendored `SOFTBART_VENDOR/*`
  files currently ship **without an upstream copyright/license
  header**. The applicable license is GPL-2.0-or-later as above; an
  explicit upstream attribution header should be added to each
  `SOFTBART_VENDOR/*` file before external distribution. This does not
  change the licensing of the combined work but should be corrected
  for attribution hygiene.
- **License-inheritance warning**: any file that `#include`s
  `softbart_block.hpp` (directly or transitively) inherits
  GPL-2.0-or-later.

## celerite (GP semiseparable kernels for time series)

- Upstream: https://github.com/dfm/celerite (the C++ `celerite`
  library underlying the celerite/celerite2 GP method).
- License: MIT — see `celerite/LICENSE`.
- Copyright (c) 2016–2020 Eric Agol & Dan Foreman-Mackey.
- What we vendored: `celerite/` (the `include/` header tree, `LICENSE`
  and `README.md`). Used by the GP time-series covariance path; it
  depends on the vendored Eigen headers.
- Modifications: none. MIT is GPL-compatible.

## libgp covariance kernels (GP covariance functions)

- Upstream: the `libgp` Gaussian-process library
  (https://github.com/mblum/libgp), original author Manuel Blum.
- License: BSD-3-Clause — see `libgp_kernels/COPYING`
  (`Copyright (c) 2025, libgp Contributors`); the individual kernel
  headers carry `Copyright (c) 2013, Manuel Blum
  <mblum@informatik.uni-freiburg.de>`.
- What we vendored: `libgp_kernels/` — the covariance-function
  headers (`cov_*.h`), `gp_utils.h`, the `src/` helpers, the unity
  header `libgp_kernels_unity.h`, plus `COPYING` and `README.md`.
  Used by the GP regression / classification covariance path.
- Modifications: none material to the covariance math (vendored as a
  trimmed kernel-only subset). BSD-3-Clause is GPL-compatible.

## librjmcmc (partial vendoring — transform classes only)

- Upstream: https://github.com/IGNF/librjmcmc
- Version: pre-2012 (last upstream activity 2012; stable, unmaintained).
- License: CeCILL-B v1 (French BSD-equivalent, GPL-compatible per the
  CeCILL-B compatibility clause).
- Copyright: Institut Géographique National (2008-2012), Mathieu
  Brédif, Olivier Tournaire, Didier Boldo. Upstream contact:
  librjmcmc@ign.fr.
- What we vendored: only the five auto-Jacobian transform classes
  (`identity_transform`, `linear_transform`,
  `diagonal_linear_transform`, `diagonal_affine_transform`,
  `affine_transform`) + their minimal `matrix::` utilities, ported
  into `include/AI4BayesCode/rjmcmc_transforms.hpp`, preserving the
  upstream copyright per CeCILL-B clauses 3, 4, 5.
- What we did NOT vendor: librjmcmc's kernel / view / variate /
  simulated-annealing infrastructure (geospatial-specific and
  architecturally incompatible with AI4BayesCode's stateful blocks).
- Modifications for AI4BayesCode (2026-04-19):
  - Namespace moved from `rjmcmc::` to
    `AI4BayesCode::rjmcmc_transforms::`.
  - C++17 modernisation (`if constexpr` for template recursion base
    cases), which also fixes a latent upstream bug in
    `determinant<1>` (the original `sum_k m[k] * cofactor<1>(k,0,m)`
    collapses to `m[0]^2` for N=1, propagating an error into
    `determinant<2>`); the port adds an explicit
    `if constexpr (N == 1) return m[0]` base case. Documented inline.
  - One `for (T *m = ...)` loop changed to `const T *m = ...` for
    const-correctness in a `const` method.
- Combined-work license: distributed as part of AI4BayesCode the
  combined work is GPL-3.0-or-later (see below); the upstream CeCILL-B
  attribution is preserved in the ported header.

## Root license summary

- **AI4BayesCode's own code** (all files under `include/AI4BayesCode/`
  and `examples/`, plus the top-level project files) —
  `Copyright (C) 2026 AI4BayesCode`. See `LICENSE` at the repo root.
- **The combined, distributed work — AI4BayesCode plus every vendored
  dependency it is compiled with — is licensed as a whole under the
  GNU General Public License, version 3.0 or later
  (GPL-3.0-or-later).** This is the project's elected license.
- Per-component upstream licenses are unchanged and remain in force on
  their own files:
  - BART (CRAN BART R package) kernel `bart_pure_cpp/src/BART/` — GPL-2.0-or-later
  - genBART kernel `bart_pure_cpp/src/GENBART/` — GPL-2.0-or-later
  - SoftBART kernel `bart_pure_cpp/src/SOFTBART_VENDOR/` — GPL-2.0-or-later
  - mcmclib `include/mcmclib/mcmc/` — Apache-2.0
  - BaseMatrixOps `include/mcmclib/BaseMatrixOps/` — Apache-2.0
  - autodiff `include/autodiff/` — MIT
  - Eigen `include/eigen/` — MPL-2.0
  - celerite `celerite/` — MIT
  - libgp kernels `libgp_kernels/` — BSD-3-Clause
  - librjmcmc transforms (in `rjmcmc_transforms.hpp`) — CeCILL-B v1
- Any file that `#include`s `bart_block.hpp`, `genbart_block.hpp` or
  `softbart_block.hpp` transitively pulls the corresponding
  GPL-2.0-or-later kernel.

### Compatibility notes

- GPL is one-directionally compatible with **MIT** (autodiff,
  celerite, and walnuts — the latter a design reference only, no
  vendored source), **BSD-3-Clause** (libgp), **MPL-2.0** (Eigen) and
  **CeCILL-B** (librjmcmc) — all may be combined into a GPL work.
- The BART-family kernels are **GPL-2.0-or-later**: the "or-later"
  clause permits redistribution under GPL-3.
- **Apache-2.0** (mcmclib, BaseMatrixOps) is one-way compatible with
  **GPL-3** but **NOT** with GPL-2.0-only. Because AI4BayesCode's
  NUTS path compiles in these Apache-2.0 components, the only
  self-consistent license for the *combined* work is the GPL-3 branch
  of the BART-family "or-later" grant.
- **Conclusion:** the combined distribution (AI4BayesCode + all
  vendored deps) is **GPL-3.0-or-later**. A strictly GPL-2.0-only
  build is only possible if the Apache-2.0 (mcmclib / BaseMatrixOps)
  pieces are removed — not a supported configuration.

All third-party source files retain their original copyright and
license headers except where a "Modifications" / "Hygiene gap" note
above states otherwise (BART dual-copyright header-only merge;
SoftBART missing upstream header — to be added before external
distribution).
