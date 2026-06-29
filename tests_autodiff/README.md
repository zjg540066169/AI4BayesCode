# `tests_autodiff/` — extended autodiff / gradient-correctness suite

This directory holds AI4BayesCode's **extended integration-test suite**:
gradient-correctness checks, library-parity tests, and longer
multi-chain audit drivers that go beyond the fast unit tests in
`tests/`. It is a **development / maintenance** suite — **end users do
NOT need it** to compile or run a generated sampler. The shipped
library headers under `include/` and the standalone examples under
`examples/` are fully usable without anything here.

## What lives here

- **Gradient correctness (validator Check #12).** The core idea is to
  cross-check every hand-written log-density gradient against an
  independent autodiff (or finite-difference) computation. See
  `test_wrap_autodiff_vs_hand.cpp` and `run_wrap_check12.R`. During
  code generation the validator writes a throwaway
  `verify_<ClassName>.cpp` here, compiles it, confirms the max abs diff
  is below tolerance (≈1e-8 for AD, ≈1e-5 for the finite-difference
  fallback), and deletes the file on PASS — so you will normally see no
  `verify_*.cpp` checked in.
- **Library-parity tests** (`run_*_parity.R`, `block_tests/`) — confirm
  the specialized blocks (Gibbs / slice / ESS / ODE integrator, etc.)
  match a reference computation.
- **Block / kernel unit tests** (`test_*.cpp`) — e.g. NUTS adaptation,
  joint-NUTS dense metric, HMM, Pólya-Gamma, rjmcmc transforms.
- **Longer multi-chain audit drivers** (`audit_*.R`, `gbart_*.R`,
  `smoke_*.R`, `*.sh`) — R-hat / ESS / posterior-predictive checks at
  larger chain lengths than the smoke tests.

Agents extending the library may **add new gradient-check (and other
integration) tests here** when they touch a block's log-density or add
a new block.

## Running

Run from the **repository root** so the `-I include` paths resolve, for
example:

```bash
# a C++ gradient-correctness test
c++ -std=c++17 -I include -I include/eigen \
    tests_autodiff/test_wrap_autodiff_vs_hand.cpp -o /tmp/check12 && /tmp/check12

# an R-level parity / audit driver
Rscript tests_autodiff/run_wrap_check12.R
```

Some C++ tests additionally need the vendored Armadillo / mcmclib /
libgp / celerite include roots; mirror the `-I` flags that
`R/AI4BayesCode_helpers.R` assembles (`include`, `include/mcmclib`,
`include/mcmclib/BaseMatrixOps/include`, `include/eigen`,
`libgp_kernels`, `celerite/include`).
