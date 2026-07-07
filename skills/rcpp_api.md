---
name: AI4BayesCode-rcpp-api
description: |
  Common Rcpp and C++ API pitfalls to avoid when generating AI4BayesCode
  samplers. Reference this when writing code that interfaces with R.
---

# Rcpp API guardrails

## Data types

| R type | Rcpp type | arma type | Notes |
|--------|-----------|-----------|-------|
| numeric vector | `Rcpp::NumericVector` | `arma::vec` | `arma::vec(x.begin(), n, false)` for no-copy |
| numeric matrix | `Rcpp::NumericMatrix` | `arma::mat` | Column-major (like R), NOT row-major |
| integer vector | `Rcpp::IntegerVector` | `arma::ivec` | |
| list | `Rcpp::List` | -- | No compile-time key checking |

## Common pitfalls

### 1. NumericMatrix is column-major
R and Rcpp store matrices column-major. `X(i, j)` is `X[i + j*nrow]`.
BART's internal `fit()` function expects the transpose (`p x n`), which
is why `bart_model.h` transposes X on construction.

### 2. Rcpp::List has no compile-time key checking
Always validate keys at runtime:
```cpp
// WRONG: crashes if key missing
double x = Rcpp::as<double>(params["missing_key"]);

// RIGHT: check first
if (params.containsElementNamed("key")) {
    double x = Rcpp::as<double>(params["key"]);
}
```

### 3. Rcpp::as vs direct construction
```cpp
// Copy (safe, allocates new memory):
arma::vec v = Rcpp::as<arma::vec>(rcpp_vec);

// No-copy (fast but dangerous -- data dies with rcpp_vec):
arma::vec v(rcpp_vec.begin(), rcpp_vec.size(), /*copy=*/false);
```
Use no-copy only for temporary computation within a single function.

### 4. Rcpp::stop vs throw
In Rcpp context, use `Rcpp::stop()` not `throw std::runtime_error()`.
`Rcpp::stop` properly unwinds through R's error handling.

### 5. Where Rcpp lives in the header tree
All public `AI4BayesCode` headers under `include/AI4BayesCode/` depend on
Rcpp/RcppArmadillo **transitively**, because their public APIs
(`get_history()`, `get_dag()`) return `Rcpp::List` / `Rcpp::NumericMatrix`.
The include graph is:

- `block_sampler.hpp` -- single entry point that literally does
  `#include <RcppArmadillo.h>`.
- All other headers (`composite_block.hpp`, `nuts_block.hpp`,
  `joint_nuts_block.hpp`,
  `*_gibbs_block.hpp`, `bart_block.hpp`, `genbart_block.hpp`,
  `poisson_multinomial_aug_block.hpp`) include `block_sampler.hpp`
  and therefore get Rcpp transitively. They MUST NOT re-include
  `<Rcpp.h>` or `<RcppArmadillo.h>` directly.
- `bart_block.hpp` additionally pulls the BART unity header from
  `../../bart_pure_cpp/src/bart_kernel_unity.h` (unified BART_unified vendor as
  of 2026-05-03). `genbart_block.hpp` no longer needs a unity header --
  the new `genbart::genbart_model` (renamed from `gbart_model`) and its
  vendored kernel under `bart_pure_cpp/src/GENBART/` are header-only. Standard
  BART's `src/BART/` still has `.cpp` files, so its unity header folds
  them into the TU; genBART does not.
- `autodiff_wrap.hpp`, `constraints.hpp`, `shared_data.hpp` are
  pure-C++ (no Rcpp). Keep them that way.

When you add a new header, follow this rule: if it will expose an
`Rcpp::` type in a public signature, `#include "block_sampler.hpp"` for
the transitive Rcpp dep. Do not add a second direct include.

### 6. sourceCpp compiles one file
`Rcpp::sourceCpp` compiles a single `.cpp` translation unit. Multi-file
dependencies must use unity/amalgamation headers (like
`bart_pure_cpp/src/bart_kernel_unity.h` for standard BART). The unified genBART
vendor under `bart_pure_cpp/src/GENBART/` is header-only and needs no unity.

### 7. R's RNG vs C++ RNG
BART uses R's RNG via `arn` -- reproducible with `set.seed()` in R.
NUTS blocks use `std::mt19937_64` -- reproducible with the seed passed
to the constructor. These are independent RNG streams.

### 8. Rcpp Module method signatures
Rcpp Modules do NOT support C++ function overloading. Each method must
have a unique name. Use `Rcpp::List` for flexible input/output instead
of multiple overloads.

### 9. Empty Rcpp::List
`Rcpp::List::create()` returns an empty list with NULL names. Check
`new_data.size() == 0` before calling `.names()` to avoid errors:
```cpp
// WRONG: crashes on empty list
Rcpp::CharacterVector names = new_data.names();

// RIGHT: check size first
if (new_data.size() == 0) return Rcpp::List::create();
Rcpp::CharacterVector names = new_data.names();
```

### 10. arma::vec from Rcpp::NumericVector
When converting between Rcpp and arma, be explicit about copy semantics:
```cpp
// Rcpp -> arma (element-by-element, safe):
arma::vec v(n);
for (std::size_t i = 0; i < n; ++i) v[i] = rcpp_vec[i];

// arma -> Rcpp (element-by-element, safe):
Rcpp::NumericVector out(v.n_elem);
for (std::size_t i = 0; i < v.n_elem; ++i) out[i] = v[i];
```

### 11. Distribution parameterization -- COMMENT THE FORM YOU USE

**This is a critical source of silent bugs.** When using random
distributions in C++ code, ALWAYS write a comment next to the call
stating the exact parameterization. The comment must include the
expected mean so a reader can verify.

```cpp
// REQUIRED: comment the form and expected mean
std::gamma_distribution<double> gam(shape, scale);
// ^ C++ std::gamma_distribution uses shape-SCALE form
// E[X] = shape * scale

// WRONG -- no comment, reader cannot verify:
std::gamma_distribution<double> gam(a, b);
```

**C++ standard library parameterizations:**
- `std::gamma_distribution<double>(shape, scale)` -- shape-**SCALE**, E[X] = shape * scale
- `std::normal_distribution<double>(mean, stddev)` -- NOT variance
- `std::chi_squared_distribution<double>(df)` -- E[X] = df

**R function parameterizations (for cross-checking):**
- `rgamma(n, shape, rate)` -- shape-**RATE**, E[X] = shape / rate
- `rgamma(n, shape, scale=s)` -- shape-**SCALE**, E[X] = shape * s
- `rnorm(n, mean, sd)` -- NOT variance
- `MCMCpack::rinvgamma(n, shape, scale)` -- E[X] = scale / (shape - 1)

**C++ std uses SCALE. R rgamma default uses RATE. They are inverses.**
To draw Gamma(shape, rate=r) in C++:
```cpp
// Gamma with rate r: use scale = 1/r
std::gamma_distribution<double> gam(shape, 1.0 / rate);
// E[X] = shape / rate = shape * (1/rate)
```

**InvGamma(shape, scale) in C++:**
```cpp
// InvGamma(a, b): draw g ~ Gamma(a, rate=b), return 1/g
std::gamma_distribution<double> gam(shape, 1.0 / scale);  // scale param = 1/rate
double x = 1.0 / gam(rng);
// E[X] = scale / (shape - 1)
```
