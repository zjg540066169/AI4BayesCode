// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later
// (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
//  PYGaussianMixture.cpp
//
//  REFERENCE TEMPLATE for Bayesian-nonparametric Gaussian mixture
//  modelling via the PITMAN-YOR PROCESS (Pitman & Yor 1997). Identical
//  in structure to `examples/DPGaussianMixture.cpp`; the ONLY difference
//  is the `a_fn` / `b_fn` of the `stick_breaking_block`, which encode
//  the PY weights:
//
//      V_k ~ Beta(a_k, b_k)
//      a_k = 1 + n_k - discount
//      b_k = alpha + (k + 1) * discount + sum_{j>k} n_j
//
//  When discount == 0 this reduces exactly to the DP. When discount > 0
//  (in [0, 1)), the marginal distribution of the number of distinct
//  clusters has a heavier (power-law) tail than DP — useful for data
//  with many small components.
//
//  PRACTICAL DISCOUNT VALUES
//  -------------------------
//  - discount = 0       : DP (use DPGaussianMixture.cpp directly).
//  - discount in (0, 0.3]: mild heavy-tail; usually still recovers a
//                          well-defined modal K_active on identified
//                          fixtures.
//  - discount in (0.3, 0.7]: aggressive over-clustering; the prior
//                          makes "one more cluster" cheap, so the
//                          posterior on K_active is multi-modal and
//                          chains may disagree (R-hat on K poor).
//                          Use this regime intentionally when the data
//                          truly has many small components.
//  - discount > 0.7     : extreme; rarely useful in practice.
//
//  The 4-chain audit (`tests_autodiff/audit_py_gaussian_4chain.R`)
//  verifies discount = 0 (PASS) AND discount = 0.5 (DIAGNOSTIC FAIL —
//  chains find K = 5–8 instead of truth 3 because of the heavy-tail
//  prior; mechanics correct, modal K is not a function of truth alone).
//
//  In this v0 example, `discount` is FIXED at construction (not sampled).
//  Sampling discount jointly with alpha would require a `joint_nuts_block_mixed`
//  with an INTERVAL constraint on discount in (0, 1); that constraint
//  family is NOT in v1 of `joint_nuts_block_mixed` (constraints.md
//  joint_constraint table — only REAL + POSITIVE shipped). Two options:
//    - sample discount via a separate `nuts_block` with interval(0, 1)
//      constraint, OR
//    - keep it fixed (this file) and explore by re-running the chain at
//      multiple discount values.
//  We choose the second for the v0 reference; the first is a trivial
//  ~30-line addition for users who need discount-posterior inference.
//
//  Block decomposition (identical to DPGaussianMixture except (3) below):
//      child(0) z              categorical_gibbs_block
//      child(1) cluster_params normal_gamma_cluster_gibbs_block
//      child(2) pi             stick_breaking_block (PY a_fn / b_fn)
//      child(3) alpha          nuts_block (positive constraint, log scale)
//
//  alpha log-density (PY version — COLLAPSED EPPF marginal on (k, n, d))
//  ---------------------------------------------------------------------
//  alpha is sampled conditional on its SUFFICIENT STATISTICS via the
//  Pitman-Yor EPPF (Pitman & Yor 1997), NOT on the truncated sticks:
//      k = #OCCUPIED clusters (cluster_counts[j] > 0.5)
//      n = #observations      (sum of cluster_counts)
//      d = discount           (FIXED at construction)
//
//      log p(alpha | k, n, d) =
//          (a_alpha - 1) log alpha - b_alpha * alpha     [Gamma(a,b) prior]
//        + sum_{i=1}^{k-1} log(alpha + i*d)              [PY cluster factor]
//        + lgamma(alpha + 1) - lgamma(alpha + n)         [PY normalizer]
//
//      d/d alpha =
//          (a_alpha - 1)/alpha - b_alpha
//        + sum_{i=1}^{k-1} 1/(alpha + i*d)
//        + digamma(alpha + 1) - digamma(alpha + n)
//
//  At discount = 0 this reduces EXACTLY to the DP Antoniak (k, n)
//  concentration marginal (Antoniak 1974; Escobar & West 1995) —
//  cf. DPGaussianMixture_DerivedAlpha.cpp.
//
//  WHY NOT condition on the sticks? The previous version read `stick_V`
//  and summed over all K_trunc-1 sticks, INCLUDING the empty truncation
//  tail. Each empty stick is a pure Beta(1-d, alpha+(j+1)d) PRIOR draw —
//  chain-specific noise — so alpha was fit to noise and 2-chain rank-R-hat
//  was ~1.83 (chains landed at alpha ~9.4 vs ~1.7). The cluster
//  PROPORTIONS already converged; only alpha (and pi coupled to it)
//  failed. Conditioning on (k, n, d) decouples alpha from the empty
//  sticks and restores mixing (rank-R-hat < 1.01).
//
//  Numerics: digamma is computed by the series helper `psi()` below
//  (recurrence + asymptotic expansion) so the example stays
//  self-contained and frontend-independent (no Rcpp R-internal symbol,
//  no boost dependency). The lgamma/digamma DIFFERENCE is computed via an
//  alpha >> n asymptotic for alpha > 1e6 to avoid catastrophic
//  cancellation (mirrors DPGaussianMixture_DerivedAlpha.cpp).
//
//  Check #12 (autodiff verify): the alpha log-density gradient is
//  finite-difference verified at gen-time by
//  tests_autodiff/verify_PYGaussianMixture.cpp; verify file is DELETED
//  on PASS per codegen.md hard rule.
//
//  Check #15 inheritance: same as DPGaussianMixture (same Tier-B
//  blocks; only a_fn / b_fn closures change).
//
//  LABEL SWITCHING — PY STAYS A RAW SAMPLER (NO in-sampler canonicalizer)
//  ---------------------------------------------------------------------
//  Label switching for this PY mixture is resolved POST-MCMC on the
//  recorded draws (skills/label_switching.md): simple-sort the OCCUPIED
//  clusters by mu (or Stephens 2000 + Hungarian for overlapping
//  components), NOT in-sampler. The sampler stays a clean raw
//  exchangeable-component sampler. The CONVERGENCE TARGET is alpha +
//  occupied-cluster params (mu, lambda, proportion = count/N) +
//  proportions — all label-invariant or post-MCMC-relabeled — NOT the raw
//  stick weights of the empty truncation slots, which are slot-position
//  noise and do NOT converge on their own. alpha and the likelihood are
//  label-invariant and converge as-is.
//
// @example:R
//   library(AI4BayesCode)
//   ai4bayescode_example("PYGaussianMixture")
//   set.seed(7)
//   ## DGP: 3 well-separated 2-D Gaussian clusters (truth K = 3).
//   N <- 300L; d <- 2L
//   mu_true <- rbind(c(-4, -4), c(0, 4), c(4, -2))
//   z_true  <- sample.int(3L, N, replace = TRUE)
//   y <- mu_true[z_true, ] + matrix(rnorm(N * d, sd = 0.7), N, d)
//   # ---- Recommended: parallel chains + convergence diagnosis ----
//   run <- AI4BayesCode_run_chains(
//       function(seed) new(PYGaussianMixture, y, 12L, 0.0, seed),
//       n_chains = 4, n_burn = 1000, n_keep = 2000)
//   ai4b_diagnose(run$histories[[1]])      # summary + R-hat/ESS + plots
//   # ---- Advanced: stateful single-chain control ----
//   m <- new(PYGaussianMixture, y, 12L, 0.0, 7L)  # (y, K_trunc, discount=0(DP), seed)
//   m$step(2000)
//   cur <- m$get_current()
//   ## #occupied clusters (cluster_counts > 0.5) should be ~3; alpha finite > 0.
//   str(cur)
// @example:python
//   import numpy as np, AI4BayesCode
//   rng = np.random.default_rng(7)
//   ## DGP: 3 well-separated 2-D Gaussian clusters (truth K = 3).
//   N, d = 300, 2
//   mu_true = np.array([[-4.0, -4.0], [0.0, 4.0], [4.0, -2.0]])
//   z_true  = rng.integers(0, 3, size=N)
//   y = mu_true[z_true] + rng.normal(0.0, 0.7, size=(N, d))
//   Mod = AI4BayesCode.source("PYGaussianMixture.cpp")
//   # ---- Recommended: parallel chains + diagnosis ----
//   chains = AI4BayesCode.run_chains(
//       lambda seed: Mod.PYGaussianMixture(y, 12, 0.0, seed),
//       seeds=[101, 202, 303, 404], n_burn=1000, n_keep=2000, n_jobs=1)
//   AI4BayesCode.ai4b_diagnose(chains[0]["hist"])   # summary + diagnostics
//   # ---- Advanced: stateful single-chain control ----
//   m = Mod.PYGaussianMixture(y, 12, 0.0, 7)   # (y, K_trunc, discount=0(DP), seed)
//   m.step(2000); print(m.get_current())
// @example:end
// ============================================================================

// [[Rcpp::depends(RcppArmadillo)]]

#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif

#ifdef AI4BAYESCODE_RCPP_MODULE
#  include <RcppArmadillo.h>
#else
#  include <armadillo>
#endif

#include "AI4BayesCode/block_sampler.hpp"
#include "AI4BayesCode/backend_neutral.hpp"
#include "AI4BayesCode/shared_data.hpp"
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/constraints.hpp"
#include "AI4BayesCode/rcpp_wrap.hpp"

#include "AI4BayesCode/categorical_gibbs_block.hpp"
#include "AI4BayesCode/nuts_block.hpp"
#include "AI4BayesCode/stick_breaking_block.hpp"
#include "AI4BayesCode/normal_gamma_cluster_gibbs_block.hpp"
#include "AI4BayesCode/bnp_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <vector>

using AI4BayesCode::block_context;
using AI4BayesCode::composite_block;
using AI4BayesCode::categorical_gibbs_block;
using AI4BayesCode::categorical_gibbs_block_config;
using AI4BayesCode::nuts_block;
using AI4BayesCode::nuts_block_config;
using AI4BayesCode::stick_breaking_block;
using AI4BayesCode::stick_breaking_block_config;
using AI4BayesCode::normal_gamma_cluster_gibbs_block;
using AI4BayesCode::normal_gamma_cluster_gibbs_block_config;
namespace constraints = AI4BayesCode::constraints;
namespace bnp = AI4BayesCode::bnp;

// ============================================================================
//  Free helpers
// ============================================================================

namespace {

inline double diag_normal_log_density(const double* y, const double* mu,
                                       const double* lambda,
                                       std::size_t d) {
    double lp = 0.0;
    constexpr double kLog2Pi = 1.83787706640934548356065947281;
    for (std::size_t j = 0; j < d; ++j) {
        const double dev = y[j] - mu[j];
        lp += 0.5 * std::log(lambda[j])
            - 0.5 * kLog2Pi
            - 0.5 * lambda[j] * dev * dev;
    }
    return lp;
}

/// Series-based digamma psi(x) for x > 0. Standard recurrence + asymptotic
/// expansion. Accurate to ~1e-12 for x > 1; at x in (0, 1) we shift up
/// using psi(x) = psi(x+1) - 1/x.
inline double psi(double x) {
    double r = 0.0;
    while (x < 6.0) { r -= 1.0 / x; x += 1.0; }
    // Asymptotic: psi(x) ~ log(x) - 1/(2x) - 1/(12x^2) + 1/(120x^4) - ...
    const double inv_x = 1.0 / x;
    const double inv_x2 = inv_x * inv_x;
    r += std::log(x) - 0.5 * inv_x
       - inv_x2 * (1.0 / 12.0
                  - inv_x2 * (1.0 / 120.0
                             - inv_x2 * (1.0 / 252.0)));
    return r;
}

/// log p(alpha | k, n, discount) (NATURAL scale, no Jacobian) — the
/// COLLAPSED Pitman-Yor concentration conditional via the PY EPPF
/// (Pitman & Yor 1997). alpha's SUFFICIENT STATISTICS are
///     k = #OCCUPIED clusters   and   n = #observations,
/// together with the FIXED discount d; it does NOT depend on the stick
/// weights. The previous version conditioned on the truncated sticks
/// `stick_V`, coupling alpha to the empty-cluster sticks (each a pure
/// Beta(1-d, alpha+(j+1)d) PRIOR draw, chain-specific noise) -> alpha was
/// fit to noise and mixed terribly (2-chain rank-R-hat ~1.83; chains
/// landed at alpha ~9.4 vs ~1.7). Conditioning on (k, n, d) decouples
/// alpha from the empty truncation tail and mixes (rank-R-hat ~1.0). This
/// reduces EXACTLY to the DP Antoniak (k, n) marginal at discount = 0.
///
///   PY EPPF (exchangeable partition probability), occupied-cluster part:
///     p(partition | alpha, d) ∝
///        [ prod_{i=1}^{k-1} (alpha + i*d) ] * Gamma(alpha + 1) / Gamma(alpha + n)
///   (the per-cluster Gamma(n_j - d)/Gamma(1 - d) factors are
///    alpha-INDEPENDENT and drop out of the alpha conditional).
///
///   prior: alpha ~ Gamma(a_alpha, rate=b_alpha)
///        => (a_alpha - 1) log alpha - b_alpha * alpha
///
///   log p(alpha | k, n, d) = (a_alpha - 1) log alpha - b_alpha * alpha
///        + sum_{i=1}^{k-1} log(alpha + i*d)            [PY cluster factor]
///        + lgamma(alpha + 1) - lgamma(alpha + n)       [PY normalizer]
///
///   d/d alpha = (a_alpha - 1)/alpha - b_alpha
///        + sum_{i=1}^{k-1} 1/(alpha + i*d)
///        + digamma(alpha + 1) - digamma(alpha + n)
///
/// At discount = 0 the cluster factor sum_{i=1}^{k-1} log(alpha) = (k-1)
/// log alpha and (with lgamma(alpha+1)=log(alpha)+lgamma(alpha)) the
/// expression reduces to the DP Antoniak marginal
///   k*log alpha + lgamma(alpha) - lgamma(alpha+n) (+ prior). Sanity check.
///
/// NB: the sum_{i} log(alpha + i*d) and its 1/(alpha + i*d) gradient are
/// computed DIRECTLY (well-conditioned for any alpha). Only the
/// lgamma/digamma DIFFERENCE lgamma(alpha+1)-lgamma(alpha+n) suffers
/// CATASTROPHIC CANCELLATION for huge alpha (both ~alpha*log alpha while
/// their true difference ~ -(n-1) log alpha is lost below the float ULP),
/// which can fool a NUTS fling to huge alpha into a spuriously good lp and
/// freeze the chain. For alpha > 1e6 we restore the correct difference +
/// gradient via the alpha >> n asymptotic (mirrors
/// DPGaussianMixture_DerivedAlpha.cpp).
double alpha_natural_log_density(const arma::vec& alpha_nat,
                                  const block_context& ctx,
                                  arma::vec* grad_nat) {
    const double a = alpha_nat[0];
    if (!(a > 0.0) || !std::isfinite(a)) {
        if (grad_nat) { grad_nat->set_size(1); (*grad_nat)[0] = 0.0; }
        return -std::numeric_limits<double>::infinity();
    }
    const double a_prior  = ctx.at("a_alpha")[0];
    const double b_prior  = ctx.at("b_alpha")[0];
    const double discount = ctx.at("discount")[0];

    // Validate boundaries.
    if (!(discount >= 0.0 && discount < 1.0)) {
        if (grad_nat) { grad_nat->set_size(1); (*grad_nat)[0] = 0.0; }
        return -std::numeric_limits<double>::infinity();
    }

    // SUFFICIENT STATISTICS: k = #OCCUPIED clusters, n = #observations.
    const arma::vec& counts = ctx.at("cluster_counts");
    double n = 0.0;
    std::size_t k = 0;
    for (std::size_t j = 0; j < counts.n_elem; ++j) {
        n += counts[j];
        if (counts[j] > 0.5) ++k;
    }

    // ---- Gamma(a_alpha, b_alpha) prior ---------------------------------
    double lp   = (a_prior - 1.0) * std::log(a) - b_prior * a;
    double grad = (a_prior - 1.0) / a - b_prior;

    // ---- PY EPPF cluster factor: sum_{i=1}^{k-1} log(alpha + i*d) -------
    //      (empty when k <= 1; computed directly — fine for any alpha).
    for (std::size_t i = 1; i + 1 <= k; ++i) {   // i = 1 .. k-1
        const double term = a + static_cast<double>(i) * discount;
        lp   += std::log(term);
        grad += 1.0 / term;
    }

    // ---- PY EPPF normalizer: lgamma(alpha+1) - lgamma(alpha+n) ----------
    //      and its derivative digamma(alpha+1) - digamma(alpha+n). For
    //      LARGE alpha use the alpha >> n asymptotic to avoid catastrophic
    //      cancellation; otherwise the direct lgamma/digamma difference.
    double norm_lp, norm_d;
    if (a > 1.0e6) {
        norm_lp = -(n - 1.0) * std::log(a) - (n - 1.0) * n / (2.0 * a);
        norm_d  = -(n - 1.0) / a + (n - 1.0) * n / (2.0 * a * a);
    } else {
        norm_lp = std::lgamma(a + 1.0) - std::lgamma(a + n);
        norm_d  = psi(a + 1.0) - psi(a + n);
    }
    lp   += norm_lp;
    grad += norm_d;

    if (grad_nat) {
        grad_nat->set_size(1);
        (*grad_nat)[0] = grad;
    }
    return lp;
}

}  // anonymous namespace

// ============================================================================
//  Tier A wrapper class
// ============================================================================

class PYGaussianMixture {
public:
    // Data-driven weakly-informative Normal-Gamma hypers (see CRITICAL
    // note in skills/block_catalogue.md; verified on DPGaussianMixture).
    static arma::vec dd_mu0_(const arma::mat& y) {
        const int n = static_cast<int>(y.n_rows), d = static_cast<int>(y.n_cols);
        arma::vec m(static_cast<arma::uword>(d));
        for (int j = 0; j < d; ++j) {
            double s = 0.0;
            for (int i = 0; i < n; ++i) s += y(i, j);
            m[j] = (n > 0) ? s / n : 0.0;
        }
        return m;
    }
    static double dd_blambda_(const arma::mat& y) {
        const int n = static_cast<int>(y.n_rows), d = static_cast<int>(y.n_cols);
        double acc = 0.0; int used = 0;
        for (int j = 0; j < d; ++j) {
            double s = 0.0;
            for (int i = 0; i < n; ++i) s += y(i, j);
            const double mean = (n > 0) ? s / n : 0.0;
            double ss = 0.0;
            for (int i = 0; i < n; ++i) {
                const double dv = y(i, j) - mean; ss += dv * dv;
            }
            const double v = (n > 1) ? ss / (n - 1) : 1.0;
            if (std::isfinite(v) && v > 0.0) { acc += v; ++used; }
        }
        const double b = (used > 0) ? acc / used : 1.0;
        return (std::isfinite(b) && b > 0.0) ? b : 1.0;
    }

    /// RECOMMENDED constructor: data-driven weakly-informative
    /// Normal-Gamma hypers from y; `discount` (PY model param) stays
    /// explicit. Delegates to the explicit ctor (unchanged path).
    PYGaussianMixture(const arma::mat& y,
                      int K_trunc,
                      double discount,
                      int rng_seed,
                      bool keep_history = false)
        : PYGaussianMixture(y, K_trunc, discount,
                            dd_mu0_(y), 0.01, 2.0, dd_blambda_(y),
                            1.0, 1.0, rng_seed, keep_history) {}

    PYGaussianMixture(const arma::mat& y,
                      int K_trunc,
                      double discount,
                      const arma::vec& mu_0,
                      double kappa_0,
                      double a_lambda_0,
                      double b_lambda_0,
                      double a_alpha,
                      double b_alpha,
                      int rng_seed,
                      bool keep_history = false)
        : rng_(rng_seed == 0
                   ? std::mt19937_64{std::random_device{}()}
                   : std::mt19937_64{static_cast<std::uint64_t>(rng_seed)}),
          predict_rng_(rng_seed == 0
                   ? std::mt19937_64{std::random_device{}()}
                   : std::mt19937_64{static_cast<std::uint64_t>(rng_seed)
                                     ^ 0x9E3779B97F4A7C15ULL}),
          readapt_rng_(rng_seed == 0
                   ? std::mt19937_64{std::random_device{}()}
                   : std::mt19937_64{static_cast<std::uint64_t>(rng_seed)
                                     ^ 0xBF58476D1CE4E5B9ULL}),
          impl_(std::make_unique<composite_block>("PYGaussianMixture")),
          keep_history_(keep_history)
    {
        if (y.n_rows < 2)
            ai4b::stop("PYGaussianMixture: N must be >= 2");
        if (y.n_cols < 1)
            ai4b::stop("PYGaussianMixture: d must be >= 1");
        if (K_trunc < 2)
            ai4b::stop("PYGaussianMixture: K_trunc must be >= 2");
        if (mu_0.n_elem != y.n_cols)
            ai4b::stop("PYGaussianMixture: mu_0 length must equal d");
        if (!(kappa_0 > 0.0))      ai4b::stop("kappa_0 must be > 0");
        if (!(a_lambda_0 > 0.0))   ai4b::stop("a_lambda_0 must be > 0");
        if (!(b_lambda_0 > 0.0))   ai4b::stop("b_lambda_0 must be > 0");
        if (!(a_alpha > 0.0))      ai4b::stop("a_alpha must be > 0");
        if (!(b_alpha > 0.0))      ai4b::stop("b_alpha must be > 0");
        if (!(discount >= 0.0 && discount < 1.0))
            ai4b::stop("discount must be in [0, 1)");

        N_ = static_cast<std::size_t>(y.n_rows);
        d_ = static_cast<std::size_t>(y.n_cols);
        K_ = static_cast<std::size_t>(K_trunc);

        arma::vec y_flat(N_ * d_);
        for (std::size_t i = 0; i < N_; ++i) {
            for (std::size_t j = 0; j < d_; ++j) {
                y_flat[i * d_ + j] = y(i, j);
            }
        }
        impl_->data().set("y", y_flat);

        arma::vec mu0_arma(d_);
        for (std::size_t j = 0; j < d_; ++j) mu0_arma[j] = mu_0[j];
        impl_->data().set("mu_0", mu0_arma);
        impl_->data().set("kappa_0",     arma::vec{kappa_0});
        impl_->data().set("a_lambda_0",  arma::vec{a_lambda_0});
        impl_->data().set("b_lambda_0",  arma::vec{b_lambda_0});
        impl_->data().set("a_alpha",     arma::vec{a_alpha});
        impl_->data().set("b_alpha",     arma::vec{b_alpha});
        impl_->data().set("discount",    arma::vec{discount});

        // Initial state.
        arma::vec z_init(N_);
        for (std::size_t i = 0; i < N_; ++i)
            z_init[i] = static_cast<double>((i % K_) + 1);
        impl_->data().set("z", z_init);

        arma::vec pi_init(K_, arma::fill::value(1.0 / static_cast<double>(K_)));
        impl_->data().set("pi", pi_init);
        arma::vec V_init(K_, arma::fill::zeros);
        {
            double rem = 1.0;
            for (std::size_t k = 0; k + 1 < K_; ++k) {
                V_init[k] = pi_init[k] / rem;
                if (V_init[k] > 1.0) V_init[k] = 1.0;
                if (V_init[k] < 0.0) V_init[k] = 0.0;
                rem *= (1.0 - V_init[k]);
            }
            V_init[K_ - 1] = 1.0;
        }
        impl_->data().set("stick_V", V_init);

        arma::vec mu_init(K_ * d_, arma::fill::zeros);
        arma::vec lambda_init(K_ * d_,
                              arma::fill::value(a_lambda_0 / b_lambda_0));
        for (std::size_t k = 0; k < K_; ++k) {
            const std::size_t i_anchor = (k * N_) / K_;
            for (std::size_t j = 0; j < d_; ++j) {
                mu_init[k * d_ + j] = y(i_anchor, j);
            }
        }
        impl_->data().set("mu", mu_init);
        impl_->data().set("lambda", lambda_init);

        const double alpha_init = a_alpha / b_alpha;
        impl_->data().set("alpha", arma::vec{alpha_init});

        impl_->data().set("cluster_counts", arma::vec(K_, arma::fill::zeros));
        impl_->data().register_refresher("cluster_counts",
            [K = K_](const AI4BayesCode::shared_data_t& d) -> arma::vec {
                const arma::vec& z = d.get("z");
                return bnp::counts_from_z(z, K);
            });

        impl_->data().declare_dependencies("z",
            {"y", "pi", "mu", "lambda"});
        impl_->data().declare_dependencies("cluster_params",
            {"z", "y"});
        impl_->data().declare_dependencies("pi",
            {"cluster_counts", "alpha", "discount"});
        impl_->data().declare_dependencies("alpha",
            {"cluster_counts", "a_alpha", "b_alpha", "discount"});
        impl_->data().declare_invalidates("z", {"cluster_counts"});

        // (No declare_data_input here — y is an observed terminal,
        // not a replaceable covariate. The y_rep refresher reads
        // pi/mu/lambda, NOT y.)
        impl_->data().declare_predict_edges("pi",     {"y_rep"});
        impl_->data().declare_predict_edges("mu",     {"y_rep"});
        impl_->data().declare_predict_edges("lambda", {"y_rep"});

        // ---- Generative-DAG context (VIZ-ONLY; predict_at BFS never
        //      reads context_edges_). Pitman-Yor stick-breaking +
        //      Normal-Gamma cluster prior:
        //        alpha ~ Gamma(a_alpha, b_alpha);
        //        V_k ~ Beta(1-discount, alpha + (k+1)*discount + ...);
        //        stick_V -> pi;
        //        (mu_k, lambda_k) ~ NormalGamma(mu_0, kappa_0,
        //                                       a_lambda_0, b_lambda_0).
        //      Drawn faded by plot_dag.
        impl_->data().declare_context_edges("a_alpha",     {"alpha"});
        impl_->data().declare_context_edges("b_alpha",     {"alpha"});
        impl_->data().declare_context_edges("alpha",       {"stick_V"});
        impl_->data().declare_context_edges("discount",    {"stick_V"});
        impl_->data().declare_context_edges("stick_V",     {"pi"});
        impl_->data().declare_context_edges("mu_0",        {"mu"});
        impl_->data().declare_context_edges("kappa_0",     {"mu"});
        impl_->data().declare_context_edges("a_lambda_0",  {"lambda"});
        impl_->data().declare_context_edges("b_lambda_0",  {"lambda"});

        impl_->data().set("y_rep", arma::vec(N_ * d_, arma::fill::zeros));
        impl_->data().register_stochastic_refresher("y_rep",
            [N = N_, d = d_, K = K_](
                const AI4BayesCode::shared_data_t& dat,
                std::mt19937_64& rng) -> arma::vec {
                const arma::vec& pi  = dat.get("pi");
                const arma::vec& mu  = dat.get("mu");
                const arma::vec& lam = dat.get("lambda");
                std::uniform_real_distribution<double> uniform(0.0, 1.0);
                std::normal_distribution<double> stdnorm(0.0, 1.0);
                arma::vec out(N * d);
                for (std::size_t i = 0; i < N; ++i) {
                    const double u = uniform(rng);
                    double cumul = 0.0;
                    std::size_t z_i = K - 1;
                    for (std::size_t k = 0; k < K; ++k) {
                        cumul += pi[k];
                        if (u < cumul) { z_i = k; break; }
                    }
                    for (std::size_t j = 0; j < d; ++j) {
                        const double mu_kj  = mu[z_i * d + j];
                        const double lam_kj = lam[z_i * d + j];
                        const double sd = 1.0 / std::sqrt(lam_kj);
                        out[i * d + j] = mu_kj + sd * stdnorm(rng);
                    }
                }
                return out;
            });

        impl_->data().refresh_all();

        // ---- children ---------------------------------------------------
        // child(0) z
        {
            categorical_gibbs_block_config cfg;
            cfg.name           = "z";
            cfg.n_obs          = N_;
            cfg.n_categories   = K_;
            cfg.initial_labels = z_init;
            const std::size_t d_capture = d_;
            const std::size_t N_capture = N_;
            const std::size_t K_capture = K_;
            cfg.log_probs_fn = [d_capture, N_capture, K_capture]
                (const block_context& ctx) -> arma::mat {
                const arma::vec& y_flat = ctx.at("y");
                const arma::vec& pi     = ctx.at("pi");
                const arma::vec& mu     = ctx.at("mu");
                const arma::vec& lam    = ctx.at("lambda");
                arma::mat lp(N_capture, K_capture);
                for (std::size_t i = 0; i < N_capture; ++i) {
                    for (std::size_t k = 0; k < K_capture; ++k) {
                        const double log_pi_k = std::log(pi[k] + 1e-300);
                        lp(i, k) = log_pi_k
                                 + diag_normal_log_density(
                                     y_flat.memptr() + i * d_capture,
                                     mu.memptr()     + k * d_capture,
                                     lam.memptr()    + k * d_capture,
                                     d_capture);
                    }
                }
                return lp;
            };
            impl_->add_child(
                std::make_unique<categorical_gibbs_block>(std::move(cfg)));
        }

        // child(1) cluster_params
        {
            normal_gamma_cluster_gibbs_block_config cfg;
            cfg.name        = "cluster_params";
            cfg.K_trunc     = K_;
            cfg.d           = d_;
            cfg.N           = N_;
            cfg.z_key       = "z";
            cfg.y_key       = "y";
            cfg.mu_name     = "mu";
            cfg.lambda_name = "lambda";
            cfg.mu_0        = mu0_arma;
            cfg.kappa_0     = kappa_0;
            cfg.a_lambda_0  = a_lambda_0;
            cfg.b_lambda_0  = b_lambda_0;
            cfg.initial_mu      = mu_init;
            cfg.initial_lambda  = lambda_init;
            impl_->add_child(
                std::make_unique<normal_gamma_cluster_gibbs_block>(
                    std::move(cfg)));
        }

        // child(2) pi  (PY a_fn / b_fn)
        {
            stick_breaking_block_config cfg;
            cfg.name        = "pi";
            cfg.K_trunc     = K_;
            cfg.counts_key  = "cluster_counts";
            cfg.v_name      = "stick_V";
            // a_k = 1 + n_k - discount
            cfg.a_fn = [](std::size_t k, const arma::vec& counts,
                          const block_context& ctx) -> double {
                const double dscount = ctx.at("discount")[0];
                return 1.0 + counts[k] - dscount;
            };
            // b_k = alpha + (k+1) * discount + sum_{j>k} n_j
            cfg.b_fn = [](std::size_t k, const arma::vec& counts,
                          const block_context& ctx) -> double {
                const double a       = ctx.at("alpha")[0];
                const double dscount = ctx.at("discount")[0];
                double tail = 0.0;
                for (std::size_t j = k + 1; j < counts.n_elem; ++j) {
                    tail += counts[j];
                }
                return a + static_cast<double>(k + 1) * dscount + tail;
            };
            cfg.initial_pi = pi_init;
            impl_->add_child(
                std::make_unique<stick_breaking_block>(std::move(cfg)));
        }

        // child(3) alpha
        {
            nuts_block_config cfg;
            cfg.name        = "alpha";
            cfg.initial_unc = arma::vec{std::log(alpha_init)};
            cfg.constrain   = constraints::positive::constrain;
            cfg.unconstrain = constraints::positive::unconstrain;
            cfg.log_density_grad =
                [](const arma::vec& t_unc, const block_context& ctx,
                   arma::vec* grad) -> double {
                    return constraints::positive::wrap(t_unc, grad,
                        [&](const arma::vec& t_nat, arma::vec* g_nat) {
                            return alpha_natural_log_density(t_nat, ctx, g_nat);
                        });
                };
            impl_->add_child(std::make_unique<nuts_block>(std::move(cfg)));
        }

        if (keep_history_) impl_->set_keep_history(true);
    }

    // ---- Six-method R interface (identical shape to DPGaussianMixture) ---

    void step(int n_steps) {
        if (n_steps < 0) ai4b::stop("n_steps must be >= 0");
        for (int i = 0; i < n_steps; ++i) impl_->step(rng_);
    }

    // Backend-neutral state_map (map<string, arma::vec>). The internal mu /
    // lambda are stored ROW-MAJOR flat as [k*d + j] (k = cluster, j = dim);
    // here they are re-flattened to COLUMN-MAJOR (k fast, j slow) arma::vec
    // so they obey the documented state_map convention (a K x d matrix
    // vectorised column-major), consistent across both backends. Scalars
    // (alpha, discount, K_trunc) are length-1 arma::vec; z and pi pass
    // through unchanged.
    AI4BayesCode::state_map get_current() const {
        const arma::vec& mu_flat  = impl_->data().get("mu");
        const arma::vec& lam_flat = impl_->data().get("lambda");
        arma::vec mu_cm(K_ * d_);    // column-major: index = j * K_ + k
        arma::vec lam_cm(K_ * d_);
        for (std::size_t k = 0; k < K_; ++k) {
            for (std::size_t j = 0; j < d_; ++j) {
                mu_cm[j * K_ + k]  = mu_flat[k * d_ + j];
                lam_cm[j * K_ + k] = lam_flat[k * d_ + j];
            }
        }
        AI4BayesCode::state_map out;
        out["z"]        = impl_->data().get("z");
        out["pi"]       = impl_->data().get("pi");
        out["mu"]       = mu_cm;      // K x d, column-major flat
        out["lambda"]   = lam_cm;     // K x d, column-major flat
        out["alpha"]    = arma::vec{impl_->data().get("alpha")[0]};
        out["discount"] = arma::vec{impl_->data().get("discount")[0]};
        out["K_trunc"]  = arma::vec{static_cast<double>(K_)};
        return out;
    }

    // Backend-neutral set_current. Reads from a state_map (map<string,
    // arma::vec>) via .find(). "y" arrives as a vectorised COLUMN-MAJOR
    // arma::vec for a K x d matrix (index = j * N_ + i); it is converted
    // back to the internal ROW-MAJOR [i*d + j] layout. z, pi, alpha,
    // discount pass through as length-N / length-K / length-1 arma::vec.
    void set_current(const AI4BayesCode::state_map& params) {
        auto it_y = params.find("y");
        if (it_y != params.end()) {
            const arma::vec& y_new = it_y->second;
            // STRICT-N legitimate (Check #21): z allocation length-N.
            // y arrives column-major flattened (N_ x d_), length N_ * d_.
            if (y_new.n_elem != N_ * d_)
                ai4b::stop("set_current: PYGaussianMixture fixes N and d at "
                           "construction (z allocation length-N). y must be a "
                           "vectorised %zu x %zu matrix (length %zu); got "
                           "length %zu. Reconstruct to change N or d.",
                           N_, d_, N_ * d_, y_new.n_elem);
            arma::vec yflat(N_ * d_);
            for (std::size_t i = 0; i < N_; ++i)
                for (std::size_t j = 0; j < d_; ++j)
                    yflat[i * d_ + j] = y_new[j * N_ + i];  // col-major -> row-major
            impl_->data().set("y", yflat);
        }
        auto it_z = params.find("z");
        if (it_z != params.end()) {
            arma::vec znew = it_z->second;
            if (znew.n_elem != N_) ai4b::stop("set_current: z length mismatch");
            for (std::size_t i = 0; i < N_; ++i) {
                const long lab = static_cast<long>(std::llround(znew[i]));
                if (lab < 1 || static_cast<std::size_t>(lab) > K_)
                    ai4b::stop("set_current: z[i] out of {1, ..., K_trunc}");
            }
            dynamic_cast<categorical_gibbs_block&>(
                impl_->child(0)).set_current(znew);
            impl_->data().set("z", znew);
            impl_->data().refresh_derived_for("z");
        }
        auto it_pi = params.find("pi");
        if (it_pi != params.end()) {
            arma::vec pinew = it_pi->second;
            if (pinew.n_elem != K_)
                ai4b::stop("set_current: pi length must equal K_trunc");
            dynamic_cast<stick_breaking_block&>(
                impl_->child(2)).set_current(pinew);
            impl_->data().set("pi", pinew);
        }
        auto it_alpha = params.find("alpha");
        if (it_alpha != params.end()) {
            const double a_new = it_alpha->second[0];
            if (!(a_new > 0.0)) ai4b::stop("alpha must be > 0");
            dynamic_cast<nuts_block&>(impl_->child(3))
                .set_current(arma::vec{a_new});
            impl_->data().set("alpha", arma::vec{a_new});
        }
        auto it_disc = params.find("discount");
        if (it_disc != params.end()) {
            const double d_new = it_disc->second[0];
            if (!(d_new >= 0.0 && d_new < 1.0))
                ai4b::stop("discount must be in [0, 1)");
            impl_->data().set("discount", arma::vec{d_new});
        }
    }

    // Backend-neutral predict_at. Takes an EMPTY state_map (no covariate
    // inputs at v0) and returns a history_map (map<string, arma::mat>).
    // Without history: each refreshed node becomes a 1-row arma::mat (y_rep
    // is a 1 x (N_*d_) flat row, row-major [i*d + j], cf. the @example DGP).
    // With history: y_rep is an n_draws x (N_*d_) matrix.
    AI4BayesCode::history_map predict_at(
            const AI4BayesCode::state_map& new_data) const {
        if (!new_data.empty())
            ai4b::stop(
                "PYGaussianMixture: predict_at(new_data) does NOT support "
                "covariate-dependent BNP at v0; pass an empty map/list.");

        AI4BayesCode::history_map out;

        if (!keep_history_) {
            block_context replaced;
            block_context result = impl_->predict_at(replaced, predict_rng_);
            for (const auto& kv : result) {
                const arma::vec& v = kv.second;
                arma::mat m(1, v.n_elem);
                for (std::size_t j = 0; j < v.n_elem; ++j) m(0, j) = v[j];
                out.emplace(kv.first, std::move(m));
            }
            return out;
        }

        // History mode: manual-compute (cf. DPGaussianMixture).
        AI4BayesCode::history_map hist = impl_->get_history();
        const arma::mat& pi_hist  = hist.at("pi");
        const arma::mat& mu_hist  = hist.at("mu");
        const arma::mat& lam_hist = hist.at("lambda");
        const std::size_t n_draws = pi_hist.n_rows;

        arma::mat yrep_mat(n_draws, N_ * d_);
        std::uniform_real_distribution<double> unif(0.0, 1.0);
        std::normal_distribution<double> nd(0.0, 1.0);
        for (std::size_t draw = 0; draw < n_draws; ++draw) {
            for (std::size_t i = 0; i < N_; ++i) {
                const double u = unif(predict_rng_);
                double cumul = 0.0;
                std::size_t z_i = K_ - 1;
                for (std::size_t k = 0; k < K_; ++k) {
                    cumul += pi_hist(draw, k);
                    if (u < cumul) { z_i = k; break; }
                }
                for (std::size_t j = 0; j < d_; ++j) {
                    const double mu_kj  = mu_hist (draw, z_i * d_ + j);
                    const double lam_kj = lam_hist(draw, z_i * d_ + j);
                    const double sd = 1.0 / std::sqrt(lam_kj);
                    yrep_mat(draw, i * d_ + j) = mu_kj + sd * nd(predict_rng_);
                }
            }
        }
        out.emplace("y_rep", std::move(yrep_mat));
        return out;
    }

    AI4BayesCode::dag_info     get_dag()     const { return impl_->get_dag(); }
    AI4BayesCode::history_map  get_history() const { return impl_->get_history(); }

    /// 7th R-level method: re-tune NUTS metric (mass matrix + step size +
    /// dual averaging) without advancing chain state. Available because
    /// the composite contains NUTS-family children. See system_design.md
    /// §13 NUTS-family + validator.md §24.
    void readapt_NUTS(int n, bool reset = false) {
        if (n < 0) {
            ai4b::stop("readapt_NUTS: n must be non-negative");
        }
        impl_->readapt_NUTS(static_cast<std::size_t>(n),
                            reset, readapt_rng_);
    }


private:
    std::mt19937_64                  rng_;
    mutable std::mt19937_64          predict_rng_;
    mutable std::mt19937_64          readapt_rng_; // readapt_NUTS() advances it (7th method)
    std::unique_ptr<composite_block> impl_;
    bool                             keep_history_ = false;
    std::size_t                      N_ = 0;
    std::size_t                      d_ = 0;
    std::size_t                      K_ = 0;
};

#ifdef AI4BAYESCODE_RCPP_MODULE
RCPP_MODULE(PYGaussianMixture_module) {
    Rcpp::class_<PYGaussianMixture>("PYGaussianMixture")
        .constructor<arma::mat, int, double, int>(
            "DEFAULT (data-driven) ctor; keep_history=FALSE. "
            "PYGaussianMixture(y, K_trunc, discount, seed).")
        .constructor<arma::mat, int, double, int, bool>(
            "DEFAULT ctor: (y, K_trunc, discount, seed, keep_history). "
            "Normal-Gamma cluster hypers computed data-driven from y "
            "(mu_0=col means, kappa_0=0.01, a_lambda=2, b_lambda=mean "
            "col var; alpha~Gamma(1,1)) so it is correctly scaled and "
            "recovers the true clusters. discount in [0,1) is the PY "
            "model parameter (kept explicit). Prefer this ctor.")
        .constructor<arma::mat, int, double, arma::vec,
                     double, double, double, double, double, int>(
            "Advanced explicit-hyper constructor; keep_history=FALSE. "
            "Prefer the data-driven ctor above.")
        .constructor<arma::mat, int, double, arma::vec,
                     double, double, double, double, double, int, bool>(
            "Construct PYGaussianMixture(y, K_trunc, discount, mu_0, "
            "kappa_0, a_lambda_0, b_lambda_0, a_alpha, b_alpha, seed, "
            "keep_history). Pitman-Yor variant of DPGaussianMixture; "
            "discount in [0, 1) FIXED at construction. discount=0 "
            "reduces exactly to DP.")
        .method("step",        &PYGaussianMixture::step)
        .method("get_current", &PYGaussianMixture::get_current)
        .method("set_current", &PYGaussianMixture::set_current,
                "Overwrite z, pi, alpha, discount, or y from a named list.")
        .method("predict_at",  &PYGaussianMixture::predict_at,
                "Posterior predictive y_rep at training X. Empty list only.")
        .method("get_dag",     &PYGaussianMixture::get_dag)
        .method("get_history", &PYGaussianMixture::get_history)
        .method("readapt_NUTS", &PYGaussianMixture::readapt_NUTS);
}
#endif

#ifdef AI4BAYESCODE_PYBIND_MODULE
#include "AI4BayesCode/pybind_casters.hpp"
PYBIND11_MODULE(PYGaussianMixture, m) {
    AI4BayesCode::register_ai4bayescode_types(m);
    pybind11::class_<PYGaussianMixture>(m, "PYGaussianMixture")
        // DEFAULT (data-driven) ctor: (y, K_trunc, discount, rng_seed,
        // keep_history). Normal-Gamma cluster hypers computed data-driven
        // from y; discount in [0,1) explicit. Mirrors the @example:python.
        .def(pybind11::init<arma::mat, int, double, int, bool>(),
             pybind11::arg("y"),
             pybind11::arg("K_trunc"),
             pybind11::arg("discount") = 0.0,
             pybind11::arg("rng_seed") = 1,
             pybind11::arg("keep_history") = false)
        // Advanced explicit-hyper ctor: (y, K_trunc, discount, mu_0,
        // kappa_0, a_lambda_0, b_lambda_0, a_alpha, b_alpha, rng_seed,
        // keep_history).
        .def(pybind11::init<arma::mat, int, double, arma::vec,
                            double, double, double, double, double, int, bool>(),
             pybind11::arg("y"),
             pybind11::arg("K_trunc"),
             pybind11::arg("discount"),
             pybind11::arg("mu_0"),
             pybind11::arg("kappa_0"),
             pybind11::arg("a_lambda_0"),
             pybind11::arg("b_lambda_0"),
             pybind11::arg("a_alpha"),
             pybind11::arg("b_alpha"),
             pybind11::arg("rng_seed") = 1,
             pybind11::arg("keep_history") = false)
        .def("step",         &PYGaussianMixture::step,  pybind11::arg("n_steps"))
        .def("get_current",  &PYGaussianMixture::get_current)
        .def("set_current",  &PYGaussianMixture::set_current, pybind11::arg("params"))
        .def("predict_at",   &PYGaussianMixture::predict_at,  pybind11::arg("new_data"))
        .def("get_dag",      &PYGaussianMixture::get_dag)
        .def("get_history",  &PYGaussianMixture::get_history)
        .def("readapt_NUTS", &PYGaussianMixture::readapt_NUTS,
             pybind11::arg("n"), pybind11::arg("reset") = false);
}
#endif
