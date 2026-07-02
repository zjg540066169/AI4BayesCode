# BART_unified

A consolidated C++/Rcpp library bringing four BART variants under a single
`src/` tree with a uniform plug-in API.

## Models

| Family | C++ class | R-facing batch entry point |
|---|---|---|
| Standard BART (Chipman 2010)              | `bart::bart_model`           | `bart_MCMC()` |
| Log-linear BART  (Murray 2021 JASA)       | `loglinbart::logbart_model`  | `logbart_MCMC()` (Poisson),<br>`nb_train()`, `multinom_train()`, `zip_train()`,<br>`zinb_train()`, `logistic_train()` |
| RJMCMC generalized BART (Linero 2022)     | `genbart::genbart_model`     | `genbart_normal_MCMC()`, `genbart_logistic_MCMC()`,<br>`genbart_poisson_MCMC()`, `genbart_nb_MCMC()`,<br>`genbart_heteroscedastic_MCMC()`, `genbart_aft_*_MCMC()`,<br>`genbart_gamma_shape_MCMC()`, `genbart_beta_MCMC()`,<br>`genbart_beta_binomial_MCMC()`, `genbart_negative_binomial_MCMC()` |
| Soft BART (Linero & Yang 2018, JRSSB)     | `softbart::softbart_model`   | `softbart_MCMC()` |

Each model class exposes a uniform plug-in API for embedding in outer Gibbs
samplers, missing-data imputation routines, joint models, etc.:

```cpp
m.set_X(X);                              // values change, p must match
m.set_Y(Y, /*center_Y=*/false);          // store new response
m.set_data(X, Y, /*center_Y=*/false);    // both at once
m.set_offset(off);                       // additive offset

m.set_s(s);                              // override DART weights
m.get_s();                               // read current weights
m.startdart();                           // activate DART (no-op if dart=false)

m.update_step();                         // one Gibbs sweep, internal sigma
m.update_step(sigma_external);           // one sweep, external sigma (Gauss only)

m.predict(X_test);                       // current-state prediction
m.current_sigma();                       // residual sd (Gauss only)
m.current_sigma_mu();                    // leaf prior sd (genbart, softbart)
m.current_var_probs();                   // current s vector
m.current_var_counts();                  // current variable usage counts
m.get_invchi(n, rss);                    // sample sigma | n, rss (Gauss only)
m.N();  m.p();  m.ntrees();              // sizes
```

## Build

Each family has its own `build_<name>.cpp`.  Compile any subset by
calling `Rcpp::sourceCpp()`:

```r
library(Rcpp)
sourceCpp("build_bart.cpp")        # Standard BART -> bart_MCMC
sourceCpp("build_logbart.cpp")     # Log-linear BART -> logbart_MCMC
sourceCpp("build_genbart.cpp")     # Generalized BART -> genbart_*_MCMC
sourceCpp("build_softbart.cpp")    # Soft BART -> softbart_MCMC
```

The four shared libraries are independent; compile only what you need.

## Directory layout

```
BART_unified/
+- src/
|  +- bart_model.h            # standard BART class + bart_MCMC
|  +- logbart_model.h         # log-linear BART class + logbart_MCMC + nb_train, ...
|  +- genbart_model.h         # generalized BART class
|  +- softbart_model.h        # soft BART thin wrapper + softbart_MCMC
|  +- BART/                   # standard-BART internals (Chipman 2010 fork)
|  +- LOGBART/                # log-linear-BART internals (Murray 2021)
|  +- GENBART/                # RJMCMC-BART internals (Linero 2022)
|  |   +- BART/               # genBART's own copy of cutpoint code
|  |   +- likelihoods/
|  +- SOFTBART_VENDOR/        # Vendored SoftBart pkg sources (Linero, GPL-2)
+- build_bart.cpp
+- build_logbart.cpp
+- build_genbart.cpp
+- build_softbart.cpp
+- tests/
|  +- test_bart.R, test_logbart.R, test_genbart.R, test_softbart.R, audit.R
+- AUTHORS, CHANGELOG.md, LICENSE, README.md
```

## License

GPL-2 or later.  See LICENSE.  Vendored components carry their original
copyright and license notices in their respective source files.

## Provenance

The standard-BART internals (`src/BART/`) derive from the BART CRAN package
(McCulloch / Sparapani / Spanbauer).  The log-linear augmentation
(`src/LOGBART/`) implements Murray (2021, JASA) on top of the same tree
code.  The RJMCMC machinery (`src/GENBART/`) follows Linero (2022).  The
Soft BART vendored sources (`src/SOFTBART_VENDOR/`) are taken verbatim
from the SoftBart R package by Antonio R. Linero.

See AUTHORS and CHANGELOG.md for full attribution.
