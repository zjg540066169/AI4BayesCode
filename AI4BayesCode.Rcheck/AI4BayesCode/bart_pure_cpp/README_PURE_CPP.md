# BART_unified — pure-C++ (R-free) core

This tree is a **dependency-decoupled port** of `../bart/`: the BART /
genBART / SoftBart kernels and model wrappers compile as **standalone C++**
with **no Rcpp and no R headers**, depending only on **Armadillo**. The goal
is a single C++ core that **both an R wrapper (Rcpp) and a Python wrapper
(pybind11/carma) can sit on top of** — the original `../bart/` is Rcpp-only,
so Python bindings could not be generated from it.

The original `../bart/` is left **untouched** (it remains the R/Rcpp version).

## How the decoupling works

The original BART code was already written **dual-mode** via `#ifdef NoRcpp`
(see `src/BART/common.h`, `tree.h`, `rtnorm/rtgamma/lambda`). The MCMC kernels
(`bartfuns/bd/bart/heterbart/treefuns`, the genBART RJMCMC, the vendored
SoftBart `Forest`) were already free of Rcpp in their hot paths. **No MCMC math
was rewritten.** Two things were added/changed:

1. **`src/r_compat.h`** — under `NoRcpp`, supplies pure-C++ replacements for the
   small set of R/Rcpp symbols the vendored numeric code references:
   - a **seedable `std::mt19937_64`** RNG (`bart_rng::set_seed(seed)`) exposed
     as `namespace R { norm_rand/unif_rand/rgamma/rchisq/rmultinom/... }`, the
     bare `unif_rand/norm_rand/exp_rand` SoftBart calls, and `Rf_*`;
   - standalone special functions (`pnorm/pgamma/qchisq/digamma/trigamma/
     lgammafn/...`), verified against R to ~5e-14;
   - `Rcout` → `std::cout`, `stop()` → `throw`, `RCPP_CHECK_INTERRUPT()` → no-op,
     `M_2PI`.

   `arn` (the R-named RNG class) thus resolves to the pure stream automatically —
   the standalone RNG that upstream had deleted is effectively restored, with
   **zero kernel edits**.

2. **Model wrappers transliterated Rcpp → Armadillo.** `bart_model.h`,
   `genbart_model.h`, `softbart_model.h` keep their class/method names and
   sampling semantics; only container types changed (`NumericMatrix/Vector/
   List` → `arma::mat/vec` + plain result structs). Batch results are
   `BartFit` / `GenbartFit` / `SoftbartFit`; histories are `*History` structs.

> **Reproducibility note:** the RNG stream is **not** bit-identical to R (R's
> Mersenne-Twister vs `std::mt19937_64`), so this is validated by *cross-chain
> agreement* with the original, not by bit-exactness. ESS, R-hat, and posterior
> means are statistically identical (see Validation).

## Families & entry points (pure C++)

| Family | Class (namespace) | Batch result | One-shot helper |
|---|---|---|---|
| Standard BART | `stdbart::bart_model` | `stdbart::BartFit` | `stdbart::bart_mcmc(...)` |
| genBART (RJMCMC, 10 likelihoods) | `genbart::genbart_model` (+ `genbart::lik::*`) | `genbart::GenbartFit` | `model.run_mcmc(...)` |
| SoftBart | `softbart::softbart_model` | `softbart::SoftbartFit`* | loop `update_step()` |

Uniform plug-in API on each class: `set_X / set_Y / set_data / set_offset /
set_s / get_s / update_step / predict / current_sigma / current_var_* /
get_history / set_tree / startdart`. All `arma::mat/vec`.

\* SoftBart batch is via the wrapper's `update_step()` + `predict()` (see the
runner); a `SoftbartFit` convenience can be added if needed.

## Build

```sh
clang++ -std=c++17 -O2 -DNoRcpp -I src -I/opt/homebrew/include \
    your_driver.cpp -L/opt/homebrew/lib -larmadillo
```

- Standard BART driver must also `#include "bart_kernel_unity.h"` (pulls the
  kernel `.cpp`s). genBART is header-only. SoftBart drivers `#include
  "softbart_kernel_unity.h"` (pulls `soft_bart_impl.h`).
- The model headers `#define NoRcpp` themselves, so `-DNoRcpp` is belt-and-
  suspenders.

## Where R / Python wrappers attach

The core takes/returns `arma::mat/vec` + POD structs. Thin adapters convert:
- **R**: Rcpp `NumericMatrix/Vector ↔ arma` (`Rcpp::as<arma::mat>`, the
  existing `build_*.cpp` style — these stay R-only and are regenerated).
- **Python**: pybind11 + carma (`numpy ↔ arma`).

## Validation — cross-chain rank-normalized R-hat vs the original

`validation/` runs the **original Rcpp** implementation and the **pure-C++**
implementation on identical X/Y, pools chains, and computes
**Vehtari (2021) rank-normalized split-R-hat** (`posterior::rhat`) for
WITHIN-new, WITHIN-old, and ACROSS. Criterion: the port ≡ the original —
**new ESS ≈ old ESS**, **ACROSS not inflated beyond WITHIN**, **cor(posterior
means) ≈ 1**.

| Family | cor(post means, new vs old) | ESS new ≈ old | verdict |
|---|---|---|---|
| Standard BART (4ch × 20k+20k) | **0.99996** | 1400 ≈ 1299 | same posterior; port ≥ original on R-hat |
| genBART (2ch dev) | **0.99785** | 15 ≈ 20 | same posterior; ACROSS ≤ WITHIN |
| SoftBart (2ch dev) | **0.99969** | 91 ≈ 82 | same posterior; ACROSS ≤ WITHIN |

Residual f(x_i) R-hat > 1.01 at finite chains is BART's intrinsic tree-posterior
multimodality — **present identically in the original** (for standard BART the
original is slightly *worse*). It is a sampler+data property, not the port.
Run scripts: `validation/crossref_rhat.R` (standard BART),
`genbart_crossref.R`, `softbart_crossref.R`; chain runners
`*_chain_runner.cpp`.

## Known follow-ups

- **Stale docs**: `README.md` / `CHANGELOG.md` describe a 4th family
  (log-linear **LOGBART** / `logbart_model.h` / `build_logbart.cpp` /
  `src/LOGBART/`) that **does not exist** in this tree. Those references are
  pre-existing rot inherited from `../bart/` and should be removed or the
  family restored.
- **R adapters** (`build_bart.cpp`, `build_genbart.cpp`, `build_softbart.cpp`)
  still target the *Rcpp* wrappers. To wrap the pure core from R, regenerate
  them against the arma API. Note `genbart_model::run_mcmc` dropped the
  `verbose`/`print_every` args in the port.
