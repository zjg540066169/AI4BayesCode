## Vendored libraries summary

| Library | Path | License | Purpose |
|---|---|---|---|
| Eigen 3.4 | `include/eigen/` | MPL-2.0 | Linear algebra |
| mcmclib | `include/mcmclib/` | Apache-2.0 | NUTS kernel |
| autodiff 1.1.2 | `include/autodiff/` | MIT | Check #12 |
| libgp kernels | `libgp_kernels/` | BSD-3 | GP kernel evaluators (SE, Matern, Periodic, RQ, ARD, noise + sum/prod composition) |
| celerite core | `celerite/include/celerite/` | MIT | 1-D time-series O(N) Cholesky |
| BART (legacy Gaussian) | `bart_pure_cpp/src/BART` | GPL-2.0+ | CRAN BART R package Gaussian tree kernel (backfitting) |
| genBART | `bart_pure_cpp/src/GENBART` | GPL-2.0+ | Linero 2022 generalized-BART RJMCMC with pluggable likelihoods (Normal / Logistic / Poisson / NB / Heteroscedastic / AFT / Beta / Gamma_shape / Beta-Binomial + user-supplied) |
| librjmcmc transforms | `include/AI4BayesCode/rjmcmc_transforms.hpp` | CeCILL-B | RJMCMC bijections |
