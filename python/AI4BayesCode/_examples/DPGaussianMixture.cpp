// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later
// (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
//  DPGaussianMixture.cpp
//
//  REFERENCE TEMPLATE for Bayesian-nonparametric (BNP) Gaussian mixture
//  modelling via the Dirichlet Process. Uses the TRUNCATED STICK-
//  BREAKING REPRESENTATION (Ishwaran & James 2001) at level K_trunc;
//  each component carries a diagonal-Gaussian likelihood with
//  conjugate Normal-Gamma cluster prior (Bishop PRML §2.3.6 / Murphy
//  2007 §4).
//
//  Model
//  -----
//      y_i      ~ N(mu_{z_i}, diag(1 / lambda_{z_i}))      i = 1..N
//      z_i      ~ Categorical(pi)
//      pi       ~ truncated stick-breaking (DP)
//        V_k    iid ~ Beta(1, alpha)         k = 0..K_trunc-2
//        V_{K_trunc-1} = 1                   (forced — Ishwaran-James)
//        pi_k = V_k * prod_{j<k}(1 - V_j)    k = 0..K_trunc-1
//      (mu_k, lambda_k) iid ~ NormalGamma(mu_0, kappa_0,
//                                          a_lambda_0, b_lambda_0)
//      alpha    ~ Gamma(a_alpha, b_alpha)    [shape-rate]
//
//  Block decomposition (Gibbs sweep order, deterministic systematic scan)
//  ---------------------------------------------------------------------
//      child(0) relabel   dp_label_canonicalizer_block
//                         FALLBACK in-sampler canonicaliser (NOT RECOMMENDED;
//                         the default is POST-MCMC relabeling, see below and
//                         skills/label_switching.md). Permutes the K_trunc slots
//                         into descending-pi order each sweep so the REPORTED
//                         pi[slot] converges. Posterior-preserving; runs FIRST
//                         so downstream children read/record on one canonical
//                         labelling. See the class comment for the full rationale.
//
//      child(1) z         categorical_gibbs_block
//                         log_probs[i, k] = log(pi_k)
//                                           + sum_d log N(y_id | mu_kd, 1/lambda_kd)
//                         (z_i conditional independence holds because pi is
//                          sampled SEPARATELY — this is the truncated SBP
//                          regime, not Neal Alg 2 CRP-marginal.)
//
//      child(2) cluster_params  normal_gamma_cluster_gibbs_block
//                         conjugate Normal-Gamma posterior per (k, dim);
//                         empty clusters draw from the prior. Writes two
//                         shared_data keys: "mu" (K_trunc * d, cluster-major)
//                         and "lambda" (K_trunc * d, cluster-major; precisions).
//
//      child(3) pi        stick_breaking_block
//                         a_fn(k) = 1 + counts[k]
//                         b_fn(k) = alpha + sum_{j>k} counts[j]
//                         Reads cluster_counts (deterministic refresher of z)
//                         and alpha from ctx. Writes "pi" and "stick_V".
//
//      child(4) alpha     nuts_block on the Antoniak (k, n) marginal
//                         (PRIOR-AGNOSTIC; positive constraint, log scale).
//                         log p(alpha | k, n) = log prior(alpha)
//                              + k*log(alpha) + lgamma(alpha) - lgamma(alpha + n),
//                         k = #occupied clusters, n = #obs. Conditions on the
//                         data-identified (k, n) sufficient statistic, NOT the
//                         empty-tail sticks. Hand-written log-density + analytic
//                         gradient; staying on NUTS keeps the prior swappable.
//
//  Refreshers
//  ----------
//      cluster_counts   register_refresher (deterministic)
//                       counts_from_z(z, K_trunc) — declared invalidated by z
//      y_rep            register_stochastic_refresher (predict-time only)
//                       For each i: z_rep_i ~ Categorical(pi); then
//                       y_rep_i ~ N(mu_{z_rep_i}, diag(1/lambda_{z_rep_i})).
//                       Marginal-over-z form is the standard posterior
//                       predictive used by Bayesian-p / LOO diagnostics.
//
//  LABEL SWITCHING
//  ---------------
//  DEFAULT / RECOMMENDED: resolve label switching POST-MCMC on the recorded
//  draws. Per `skills/label_switching.md` §3, post-MCMC relabeling is the
//  standard route for per-component posterior summaries:
//
//      - Audit / cross-implementation R-hat: apply Stephens 2000 with
//        per-draw allocation probabilities pulled from the chain.
//      - Quick visualization: simple sort by mu_first_dim of OCCUPIED
//        clusters (ignore tail K_active+1..K_trunc whose params are
//        prior draws).
//
//  FALLBACK USED HERE (child(0) dp_label_canonicalizer_block — NOT RECOMMENDED):
//  the truncated-DP stick weights pi[slot] are SLOT-POSITION-dependent and do
//  NOT converge under per-draw post-MCMC sorting of the empty truncation tail.
//  The cluster PROPORTIONS (count/N) DO converge (2-chain rank-R-hat ~1.001), so
//  this is a representation/labeling artifact, NOT a masked mixing failure. To
//  make the REPORTED pi[slot] converge on its raw recorded values, this example
//  carries an in-sampler canonicaliser (descending-pi slot permutation each
//  sweep) as a last resort. Prefer post-MCMC relabeling for new models; this
//  in-sampler path is the exception, not the template. See the class comment.
//
//  Truncation choice
//  -----------------
//  Default K_trunc = max(20, ceil(N / 5)). Ishwaran & James 2001 §2
//  show truncation error decays exponentially in K_trunc for moderate
//  alpha. For data whose true cluster count >> 20 you must supply a
//  larger K_trunc explicitly.
//
//  JUSTIFICATION (Check #16):
//      - z is DISCRETE; categorical_gibbs_block is the only valid path
//        (Exception 1 from `skills/codegen_priors.md §2b`). Conditional
//        independence holds in the truncated SBP regime.
//      - cluster_params: NEW Tier-B block normal_gamma_cluster_gibbs_block
//        (this PR). Conjugate Normal-Gamma posterior is the
//        Bishop-PRML / Murphy-2007 textbook update.
//      - pi: NEW Tier-B block stick_breaking_block. Per-stick Beta
//        gamma-trick is the standard mechanism.
//      - alpha: nuts_block per Q9 lean (NUTS preferred over Escobar-West
//        1995 auxiliary trick to keep the example free of new
//        block types).
//
//  Check #15 (library parity tests):
//      - tests_autodiff/block_tests/test_bnp_utils.cpp
//      - tests_autodiff/block_tests/test_stick_breaking_block.cpp
//      - tests_autodiff/block_tests/test_normal_gamma_cluster_gibbs_block.cpp
//      - tests_autodiff/block_tests/test_beta_gibbs_block.cpp (covers the
//        gamma-normalization mechanism shared by stick_breaking_block).
//
//  Check #12 (autodiff verify): the alpha log-density gradient is
//  analytic and verified by tests_autodiff/verify_DPGaussianMixture.cpp
//  at gen-time; verify file is DELETED on PASS per the codegen.md hard
//  rule (production .cpp stays scaffolding-free).
//
// @example:R
//   library(AI4BayesCode)
//   ai4bayescode_example("DPGaussianMixture")
//   set.seed(42); n_per <- 150L; d <- 2L                  # well-identified DGP
//   mu_true <- rbind(c(-3, -3), c(3, 3))                  # 2 well-separated clusters, sd 0.7
//   y <- rbind(matrix(rnorm(n_per * d, 0, 0.7), n_per, d) + rep(mu_true[1, ], each = n_per),
//              matrix(rnorm(n_per * d, 0, 0.7), n_per, d) + rep(mu_true[2, ], each = n_per))
//   # ---- Recommended: parallel chains + convergence diagnosis ----
//   run <- ai4bayescode_run_chains(
//       function(seed) new(DPGaussianMixture, y, 60L, seed, TRUE),
//       n_chains = 4, n_burn = 1000, n_keep = 2000)
//   ai4bayescode_diagnose(run$histories[[1]])      # summary + R-hat/ESS + plots
//   # ---- Advanced: stateful single-chain control ----
//   m <- new(DPGaussianMixture, y, 60L, 42L, TRUE)        # y (N x d), K_trunc=60, seed=42, keep_history
//   m$step(2000L); cur <- m$get_current()                # $z $pi $mu $lambda $alpha $K_trunc
//   K <- as.integer(cur$K_trunc); d <- ncol(y)
//   mu_mat <- matrix(cur$mu, nrow = K, ncol = d)          # mu flat (col-major) -> K x d
//   occ <- which(tabulate(cur$z, K) > 0)                  # occupied slots (~2)
//   mu_mat[occ, , drop = FALSE]                           # recovered centres ~ (-3,-3),(3,3)
//   cur$alpha                                             # DP concentration alpha (NUTS, (k,n) marginal)
// @example:python
//   import numpy as np, AI4BayesCode
//   rng = np.random.default_rng(42); n_per, d = 150, 2     # well-identified DGP
//   mu_true = np.array([[-3.0, -3.0], [3.0, 3.0]])         # 2 well-separated clusters, sd 0.7
//   y = np.vstack([rng.normal(0.0, 0.7, (n_per, d)) + mu_true[0],
//                  rng.normal(0.0, 0.7, (n_per, d)) + mu_true[1]])
//   Mod = AI4BayesCode.example("DPGaussianMixture")
//   # ---- Recommended: parallel chains + diagnosis ----
//   chains = AI4BayesCode.run_chains(
//       lambda seed: Mod.DPGaussianMixture(y, 60, seed, True),
//       seeds=[101, 202, 303, 404], n_burn=1000, n_keep=2000, n_jobs=1)
//   AI4BayesCode.diagnose(chains[0]["hist"])   # summary + diagnostics
//   # ---- Advanced: stateful single-chain control ----
//   m = Mod.DPGaussianMixture(y, 60, 42, True)             # (y N x d, K_trunc, seed, keep_history)
//   m.step(2000); cur = m.get_current()                   # 'z' 'pi' 'mu' 'lambda' 'alpha' 'K_trunc'
//   K = int(np.asarray(cur["K_trunc"]).ravel()[0])
//   mu_mat = np.asarray(cur["mu"]).reshape(K, d, order="F")  # mu flat (col-major) -> K x d
//   z = np.asarray(cur["z"]).astype(int)
//   occ = np.flatnonzero(np.bincount(z, minlength=K + 1)[1:] > 0)  # occupied slots (~2)
//   print(mu_mat[occ])                                     # recovered centres ~ (-3,-3),(3,3)
//   print(np.asarray(cur["alpha"]).ravel()[0])            # DP concentration alpha
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
#include "AI4BayesCode/shared_data.hpp"
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/constraints.hpp"
#include "AI4BayesCode/rcpp_wrap.hpp"

#include "AI4BayesCode/categorical_gibbs_block.hpp"
#include "AI4BayesCode/nuts_block.hpp"
#include "AI4BayesCode/stick_breaking_block.hpp"
#include "AI4BayesCode/normal_gamma_cluster_gibbs_block.hpp"
#include "AI4BayesCode/bnp_utils.hpp"
#include "AI4BayesCode/backend_neutral.hpp"   // ai4b::digamma

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
//  Free functions used by block functors / log-density lambdas
// ============================================================================

namespace {

/// log N(y | mu, 1/lambda) for diagonal precision lambda.
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

/// log p(alpha | k, n) on the NATURAL scale (no Jacobian) -- the Antoniak
/// (1974) (k, n) marginal. PRIOR-AGNOSTIC NUTS update; NOT conjugate Gibbs.
///
///   log p(alpha) [Gamma(a_alpha, rate=b_alpha)]
///       = (a_alpha - 1) * log(alpha) - b_alpha * alpha + const
///   Antoniak (k, n) marginal (k = #occupied clusters, n = #obs):
///       + k * log(alpha) + lgamma(alpha) - lgamma(alpha + n)
///   Conditions on the data-identified sufficient statistic (k, n), reading
///   cluster_counts -- NOT the empty-tail sticks stick_V, whose Beta(1, alpha)
///   prior draws are chain-specific and broke mixing (rank-R-hat ~ 1.42).
///   Staying on NUTS keeps the prior swappable: the Gamma(a_alpha, b_alpha)
///   line is the ONLY prior-specific term (conjugate Gibbs would lock Gamma).
///
/// Gradient (per-natural-alpha):
///   d log p / d alpha
///     = (a_alpha - 1 + k) / alpha - b_alpha + psi(alpha) - psi(alpha + n)
double alpha_natural_log_density(const arma::vec& alpha_nat,
                                  const block_context& ctx,
                                  arma::vec* grad_nat) {
    const double a = alpha_nat[0];
    if (!(a > 0.0) || !std::isfinite(a)) {
        if (grad_nat) { grad_nat->set_size(1); (*grad_nat)[0] = 0.0; }
        return -std::numeric_limits<double>::infinity();
    }
    const double a_prior = ctx.at("a_alpha")[0];
    const double b_prior = ctx.at("b_alpha")[0];
    const arma::vec& counts = ctx.at("cluster_counts");

    std::size_t k = 0;
    double n = 0.0;
    for (std::size_t j = 0; j < counts.n_elem; ++j) {
        if (counts[j] > 0.0) ++k;
        n += counts[j];
    }
    const double kd = static_cast<double>(k);

    const double lp =
        (a_prior - 1.0) * std::log(a) - b_prior * a            // Gamma prior
      +  kd * std::log(a)
      +  std::lgamma(a) - std::lgamma(a + n);                  // Antoniak (k,n)

    if (grad_nat) {
        grad_nat->set_size(1);
        (*grad_nat)[0] = (a_prior - 1.0 + kd) / a - b_prior
                       + ai4b::digamma(a) - ai4b::digamma(a + n);
    }
    return lp;
}

// ----------------------------------------------------------------------------
//  dp_label_canonicalizer_block  (FALLBACK — NOT THE RECOMMENDED APPROACH)
//
//  Label switching should normally be resolved POST-MCMC on the recorded draws
//  (see skills/label_switching.md): the sampler stays a clean raw exchangeable-
//  component sampler and the validation/comparison layer relabels (simple-sort
//  or Stephens 2000 + Hungarian) before computing R-hat.
//
//  It is used here ONLY as a fallback: the truncated-DP stick weights pi[slot]
//  are slot-position-dependent and do NOT converge under per-draw post-MCMC
//  sorting of the empty truncation tail. The cluster PROPORTIONS (count/N) DO
//  converge (2-chain rank-R-hat ~1.001), so this is a representation/labeling
//  fix that lets the REPORTED pi[slot] converge too — it does NOT mask a mixing
//  failure. Prefer post-MCMC relabeling for new models; reach for this only
//  when an otherwise-sound DP/mixture example cannot show full convergence on
//  its raw recorded parameters.
//
//  Posterior-preserving: permutes the K_trunc slots into descending-pi order
//  each sweep and writes the relabelled (pi, mu, lambda, z) back. Runs FIRST so
//  downstream children read/record on one canonical labelling.
//  Reads: pi (K), mu (K*d), lambda (K*d), z (N).  Writes: pi, mu, lambda, z.
// ----------------------------------------------------------------------------
class dp_label_canonicalizer_block : public AI4BayesCode::block_sampler {
public:
    dp_label_canonicalizer_block(std::string name, std::size_t K_trunc,
                                 std::size_t d, std::size_t N)
        : name_(std::move(name)), K_(K_trunc), d_(d), N_(N),
          dummy_(arma::vec(1, arma::fill::zeros)) {
        if (name_.empty())
            throw std::invalid_argument("dp_label_canonicalizer_block: name must be non-empty");
        if (K_ < 2 || d_ < 1 || N_ < 1)
            throw std::invalid_argument("dp_label_canonicalizer_block: bad K/d/N");
    }
    void set_context(const block_context& ctx) override { context_ = ctx; }
    void step(std::mt19937_64& /*rng*/) override {
        const arma::vec& pi  = context_.at("pi");
        const arma::vec& mu  = context_.at("mu");
        const arma::vec& lam = context_.at("lambda");
        const arma::vec& z   = context_.at("z");
        const arma::uvec perm = arma::sort_index(pi, "descend");
        arma::uvec inv(K_);
        for (std::size_t r = 0; r < K_; ++r) inv[perm[r]] = r;
        pi_out_.set_size(K_); mu_out_.set_size(K_ * d_); lam_out_.set_size(K_ * d_);
        for (std::size_t r = 0; r < K_; ++r) {
            const std::size_t o = perm[r];
            pi_out_[r] = pi[o];
            for (std::size_t j = 0; j < d_; ++j) {
                mu_out_[r * d_ + j]  = mu[o * d_ + j];
                lam_out_[r * d_ + j] = lam[o * d_ + j];
            }
        }
        z_out_.set_size(N_);
        for (std::size_t i = 0; i < N_; ++i) {
            long L = static_cast<long>(std::llround(z[i]));
            if (L < 1) L = 1;
            if (static_cast<std::size_t>(L) > K_) L = static_cast<long>(K_);
            z_out_[i] = static_cast<double>(inv[static_cast<std::size_t>(L) - 1] + 1);
        }
    }
    AI4BayesCode::state_map current_named_outputs() const override {
        AI4BayesCode::state_map out;
        out.emplace("pi", pi_out_); out.emplace("mu", mu_out_);
        out.emplace("lambda", lam_out_); out.emplace("z", z_out_);
        return out;
    }
    AI4BayesCode::state_map current_named_outputs(std::mt19937_64& /*rng*/) const override { return current_named_outputs(); }
    const arma::vec& current() const override { return dummy_; }
    void set_current(const arma::vec&) override {}
    const std::string& name() const noexcept override { return name_; }
    std::size_t dim() const noexcept override { return 0; }
    AI4BayesCode::history_map get_history() const override { return {}; }
    std::size_t history_size() const noexcept override { return 0; }
    void clear_history() override {}
private:
    std::string name_; std::size_t K_, d_, N_; arma::vec dummy_;
    block_context context_; arma::vec pi_out_, mu_out_, lam_out_, z_out_;
};

}  // anonymous namespace

// ============================================================================
//  Tier A wrapper class
// ============================================================================

class DPGaussianMixture {
public:
    // ---- Data-driven weakly-informative hyperparameters ---------------
    //  Computed from y so the conjugate Normal-Gamma cluster prior is
    //  correctly SCALED to the data regardless of its location/spread.
    //  This is the convergence-robust path (mis-scaled fixed hypers --
    //  e.g. b_lambda_0 = 1 on non-unit-variance data -- make the
    //  single-site categorical Gibbs mix poorly). Matches the
    //  user-blessed generated DP reference, generalised to d dims:
    //      mu_0_j   = mean(y[, j])
    //      kappa_0  = 0.01            (Var[mu_kj] = 100 * E[1/lambda])
    //      a_lambda = 2.0             (heavy-tailed, weakly informative)
    //      b_lambda = mean_j var(y[, j])  (E[sigma^2] ~ data scale)
    //  alpha ~ Gamma(1, 1).
    static arma::vec dd_mu0_(const arma::mat& y) {
        const std::size_t n = y.n_rows, d = y.n_cols;
        arma::vec m(d);
        for (std::size_t j = 0; j < d; ++j) {
            double s = 0.0;
            for (std::size_t i = 0; i < n; ++i) s += y(i, j);
            m[j] = (n > 0) ? s / static_cast<double>(n) : 0.0;
        }
        return m;
    }
    static double dd_blambda_(const arma::mat& y) {
        const std::size_t n = y.n_rows, d = y.n_cols;
        double acc = 0.0; int used = 0;
        for (std::size_t j = 0; j < d; ++j) {
            double s = 0.0;
            for (std::size_t i = 0; i < n; ++i) s += y(i, j);
            const double mean = (n > 0) ? s / static_cast<double>(n) : 0.0;
            double ss = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                const double dv = y(i, j) - mean; ss += dv * dv;
            }
            const double v = (n > 1) ? ss / static_cast<double>(n - 1) : 1.0;  // unbiased
            if (std::isfinite(v) && v > 0.0) { acc += v; ++used; }
        }
        const double b = (used > 0) ? acc / used : 1.0;
        return (std::isfinite(b) && b > 0.0) ? b : 1.0;
    }

    /// RECOMMENDED constructor: data-driven weakly-informative
    /// Normal-Gamma hyperparameters computed from y (converges robustly;
    /// no hyperparameter tuning needed). Delegates to the explicit
    /// constructor below -- behaviour of that path is unchanged.
    DPGaussianMixture(const arma::mat& y,
                      int K_trunc,
                      int rng_seed,
                      bool keep_history = false)
        : DPGaussianMixture(y, K_trunc,
                            dd_mu0_(y),   // mu_0   = column means
                            0.01,         // kappa_0
                            2.0,          // a_lambda_0
                            dd_blambda_(y),  // b_lambda_0 = mean colVar
                            1.0, 1.0,     // alpha ~ Gamma(1, 1)
                            rng_seed, keep_history) {}

    DPGaussianMixture(const arma::mat& y,
                      int K_trunc,
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
          impl_(std::make_unique<composite_block>("DPGaussianMixture")),
          keep_history_(keep_history)
    {
        if (y.n_rows < 2)
            ai4b::stop("DPGaussianMixture: N must be >= 2");
        if (y.n_cols < 1)
            ai4b::stop("DPGaussianMixture: d must be >= 1");
        if (K_trunc < 2)
            ai4b::stop("DPGaussianMixture: K_trunc must be >= 2");
        if (mu_0.n_elem != y.n_cols)
            ai4b::stop("DPGaussianMixture: mu_0 length must equal d");
        if (!(kappa_0 > 0.0))      ai4b::stop("kappa_0 must be > 0");
        if (!(a_lambda_0 > 0.0))   ai4b::stop("a_lambda_0 must be > 0");
        if (!(b_lambda_0 > 0.0))   ai4b::stop("b_lambda_0 must be > 0");
        if (!(a_alpha > 0.0))      ai4b::stop("a_alpha must be > 0");
        if (!(b_alpha > 0.0))      ai4b::stop("b_alpha must be > 0");

        N_ = static_cast<std::size_t>(y.n_rows);
        d_ = static_cast<std::size_t>(y.n_cols);
        K_ = static_cast<std::size_t>(K_trunc);

        // ---- Data + priors -----------------------------------------
        // y stored row-major: y_flat[i * d + j] = y(i, j).
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

        // ---- Initial sampler state ---------------------------------
        // z: spread (i mod K_trunc) + 1.
        arma::vec z_init(N_);
        for (std::size_t i = 0; i < N_; ++i) {
            z_init[i] = static_cast<double>((i % K_) + 1);
        }
        impl_->data().set("z", z_init);

        // pi: uniform on K_trunc-simplex.
        arma::vec pi_init(K_, arma::fill::value(1.0 / static_cast<double>(K_)));
        impl_->data().set("pi", pi_init);
        // stick_V: derived initial (V_k = pi_k / remainder_{<k}).
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

        // mu, lambda: data-driven init for mu (kmeans-like spread of
        // observed y-rows over the K_trunc clusters); lambda at prior mean.
        arma::vec mu_init(K_ * d_, arma::fill::zeros);
        arma::vec lambda_init(K_ * d_,
                              arma::fill::value(a_lambda_0 / b_lambda_0));
        for (std::size_t k = 0; k < K_; ++k) {
            const std::size_t i_anchor = (k * N_) / K_;  // spread anchors
            for (std::size_t j = 0; j < d_; ++j) {
                mu_init[k * d_ + j] = y(i_anchor, j);
            }
        }
        impl_->data().set("mu", mu_init);
        impl_->data().set("lambda", lambda_init);

        // alpha: prior mean.
        const double alpha_init = a_alpha / b_alpha;
        impl_->data().set("alpha", arma::vec{alpha_init});

        // cluster_counts: derived (refresher).
        impl_->data().set("cluster_counts",
                          arma::vec(K_, arma::fill::zeros));
        impl_->data().register_refresher("cluster_counts",
            [K = K_](const AI4BayesCode::shared_data_t& d)
                -> arma::vec {
                const arma::vec& z = d.get("z");
                return bnp::counts_from_z(z, K);
            });

        // ---- Gibbs DAG dependencies / invalidations ----------------
        // (Block names match child name() values used in declare_*.)
        // relabel (FALLBACK canonicaliser — NOT recommended; prefer post-MCMC).
        impl_->data().declare_dependencies("relabel",
            {"pi", "mu", "lambda", "z"});
        impl_->data().declare_dependencies("z",
            {"y", "pi", "mu", "lambda"});
        impl_->data().declare_dependencies("cluster_params",
            {"z", "y"});
        impl_->data().declare_dependencies("pi",
            {"cluster_counts", "alpha"});
        impl_->data().declare_dependencies("alpha",
            {"cluster_counts", "a_alpha", "b_alpha"});

        // z's update INVALIDATES cluster_counts (its derived child).
        impl_->data().declare_invalidates("z", {"cluster_counts"});
        // cluster_params writes mu and lambda; no derived keys.
        // pi writes pi and stick_V; no derived keys.
        // alpha writes alpha; no derived keys.

        // ---- Predict DAG + y_rep stochastic refresher --------------
        // (No declare_data_input here — y is an observed terminal,
        // not a replaceable covariate. The y_rep refresher reads
        // pi/mu/lambda, NOT y; N/d/K are captured at construction.)
        impl_->data().declare_predict_edges("pi",     {"y_rep"});
        impl_->data().declare_predict_edges("mu",     {"y_rep"});
        impl_->data().declare_predict_edges("lambda", {"y_rep"});

        // ---- Generative-DAG context (VIZ-ONLY; predict_at BFS never
        //      reads context_edges_). DP truncated stick-breaking +
        //      Normal-Gamma cluster prior:
        //        alpha ~ Gamma(a_alpha, b_alpha);
        //        V_k ~ Beta(1, alpha) -> stick_V -> pi;
        //        (mu_k, lambda_k) ~ NormalGamma(mu_0, kappa_0,
        //                                       a_lambda_0, b_lambda_0).
        //      Drawn faded by ai4bayescode_plot_dag.
        impl_->data().declare_context_edges("a_alpha",     {"alpha"});
        impl_->data().declare_context_edges("b_alpha",     {"alpha"});
        impl_->data().declare_context_edges("alpha",       {"stick_V"});
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
                // Marginal posterior predictive: z_rep_i ~ Cat(pi),
                // y_rep_i ~ N(mu_{z_rep_i}, diag(1/lambda_{z_rep_i})).
                const arma::vec& pi  = dat.get("pi");
                const arma::vec& mu  = dat.get("mu");
                const arma::vec& lam = dat.get("lambda");
                std::uniform_real_distribution<double> uniform(0.0, 1.0);
                std::normal_distribution<double> stdnorm(0.0, 1.0);
                arma::vec out(N * d);
                for (std::size_t i = 0; i < N; ++i) {
                    // sample z_rep_i ~ Categorical(pi)
                    const double u = uniform(rng);
                    double cumul = 0.0;
                    std::size_t z_i = K - 1;
                    for (std::size_t k = 0; k < K; ++k) {
                        cumul += pi[k];
                        if (u < cumul) { z_i = k; break; }
                    }
                    // sample y_rep_i ~ N(mu_{z_i}, diag(1/lambda_{z_i}))
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

        // ---- Children in Gibbs sweep order -------------------------
        // child(0) relabel (FALLBACK canonicaliser — see class comment; NOT
        // recommended, prefer post-MCMC). Added FIRST so downstream children
        // read/record on one canonical descending-pi labelling.
        impl_->add_child(std::make_unique<dp_label_canonicalizer_block>(
            "relabel", K_, d_, N_));

        // child(1) z (categorical_gibbs_block)
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

        // child(2) cluster_params (normal_gamma_cluster_gibbs_block)
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

        // child(3) pi (stick_breaking_block, DP weights)
        {
            stick_breaking_block_config cfg;
            cfg.name        = "pi";
            cfg.K_trunc     = K_;
            cfg.counts_key  = "cluster_counts";
            cfg.v_name      = "stick_V";
            cfg.a_fn = [](std::size_t k, const arma::vec& counts,
                          const block_context& /*ctx*/) -> double {
                return 1.0 + counts[k];
            };
            cfg.b_fn = [](std::size_t k, const arma::vec& counts,
                          const block_context& ctx) -> double {
                const double a = ctx.at("alpha")[0];
                double tail = 0.0;
                for (std::size_t j = k + 1; j < counts.n_elem; ++j) {
                    tail += counts[j];
                }
                return a + tail;
            };
            cfg.initial_pi = pi_init;
            impl_->add_child(
                std::make_unique<stick_breaking_block>(std::move(cfg)));
        }

        // child(4) alpha (nuts_block on log scale, positive constraint)
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

    // ---- Six-method R interface --------------------------------------------

    void step() { step(1); }              // no-arg convenience: one sweep

    void step(int n_steps) {
        if (n_steps < 0) ai4b::stop("n_steps must be >= 0");
        for (int i = 0; i < n_steps; ++i) impl_->step(rng_);
    }

    AI4BayesCode::state_map get_current() const {
        // Backend-neutral: mu and lambda are returned as FLAT column-major
        // arma::vec of length K_trunc*d (slot-major along columns: element
        // (k, j) of the K x d matrix lives at j*K + k). Internal storage is
        // cluster-major mu_flat[k*d + j]; build a K x d arma::mat then
        // vectorise() to emit the canonical column-major flat vector. The
        // @example:R / @example:python headers reshape with nrow = K_trunc
        // (R matrix() / numpy order="F") to recover the K x d centres.
        const arma::vec& mu_flat  = impl_->data().get("mu");
        const arma::vec& lam_flat = impl_->data().get("lambda");
        arma::mat mu_mat(K_, d_);
        arma::mat lam_mat(K_, d_);
        for (std::size_t k = 0; k < K_; ++k) {
            for (std::size_t j = 0; j < d_; ++j) {
                mu_mat(k, j)  = mu_flat[k * d_ + j];
                lam_mat(k, j) = lam_flat[k * d_ + j];
            }
        }
        AI4BayesCode::state_map out;
        out["z"]       = impl_->data().get("z");
        out["pi"]      = impl_->data().get("pi");
        out["mu"]      = arma::vectorise(mu_mat);    // flat, column-major
        out["lambda"]  = arma::vectorise(lam_mat);   // flat, column-major
        out["alpha"]   = impl_->data().get("alpha");
        out["K_trunc"] = arma::vec{static_cast<double>(K_)};
        return out;
    }

    void set_current(const AI4BayesCode::state_map& params) {
        // Backend-neutral: any subset of (z, pi, mu, lambda, alpha, y).
        // Matrix-valued keys (y is N x d; mu, lambda are K_trunc x d) arrive
        // as a VECTORISED column-major arma::vec under their key — element
        // (row, col) lives at col*nrow + row. We convert back to the internal
        // ROW/cluster-major storage (flat[row*ncol + col]).
        if (params.count("y")) {
            const arma::vec& y_new = params.at("y");
            // STRICT-N legitimate use (validator Check #21 / codegen_cpp.md
            // §7a): DPGaussianMixture's categorical_gibbs_block holds the
            // allocation vector z of length N, and the cluster_gibbs_block
            // holds per-cluster sufficient stats sized for N observations.
            // To change N, reconstruct the wrapper. (y arrives flat col-major;
            // we can only validate total length here.)
            if (y_new.n_elem != N_ * d_)
                ai4b::stop("set_current: DPGaussianMixture fixes (N, d) at "
                           "construction (z allocation + cluster suffstats "
                           "are length-N). Supplied y has %zu elements but "
                           "requires N*d = %zu. To change N, reconstruct.",
                           static_cast<std::size_t>(y_new.n_elem), N_ * d_);
            arma::vec yflat(N_ * d_);
            for (std::size_t i = 0; i < N_; ++i)
                for (std::size_t j = 0; j < d_; ++j)
                    yflat[i * d_ + j] = y_new[j * N_ + i];  // col-major -> row-major
            impl_->data().set("y", yflat);
        }
        if (params.count("z")) {
            arma::vec znew = params.at("z");
            if (znew.n_elem != N_)
                ai4b::stop("set_current: z length must equal N");
            for (std::size_t i = 0; i < N_; ++i) {
                const long lab = static_cast<long>(std::llround(znew[i]));
                if (lab < 1 || static_cast<std::size_t>(lab) > K_) {
                    ai4b::stop("set_current: z[i] out of {1, ..., K_trunc}");
                }
            }
            dynamic_cast<categorical_gibbs_block&>(
                impl_->child(1)).set_current(znew);
            impl_->data().set("z", znew);
            impl_->data().refresh_derived_for("z");
        }
        if (params.count("pi")) {
            arma::vec pinew = params.at("pi");
            if (pinew.n_elem != K_)
                ai4b::stop("set_current: pi length must equal K_trunc");
            dynamic_cast<stick_breaking_block&>(
                impl_->child(3)).set_current(pinew);
            impl_->data().set("pi", pinew);
        }
        if (params.count("mu")) {
            // Accept K_trunc x d matrix as a vectorised column-major arma::vec.
            const arma::vec& mu_new = params.at("mu");
            if (mu_new.n_elem != K_ * d_) {
                ai4b::stop("set_current: mu must be K_trunc x d "
                           "(flat length K_trunc*d)");
            }
            arma::vec mu_flat(K_ * d_);
            for (std::size_t k = 0; k < K_; ++k)
                for (std::size_t j = 0; j < d_; ++j)
                    mu_flat[k * d_ + j] = mu_new[j * K_ + k];  // col-major -> row-major
            impl_->data().set("mu", mu_flat);
            // Also push to the cluster block's internal state.
            const arma::vec& cur_lam = impl_->data().get("lambda");
            arma::vec cluster_concat(2 * K_ * d_);
            for (std::size_t i = 0; i < K_ * d_; ++i) {
                cluster_concat[i]            = mu_flat[i];
                cluster_concat[K_ * d_ + i]  = cur_lam[i];
            }
            dynamic_cast<normal_gamma_cluster_gibbs_block&>(
                impl_->child(2)).set_current(cluster_concat);
        }
        if (params.count("lambda")) {
            const arma::vec& lam_new = params.at("lambda");
            if (lam_new.n_elem != K_ * d_) {
                ai4b::stop("set_current: lambda must be K_trunc x d "
                           "(flat length K_trunc*d)");
            }
            arma::vec lam_flat(K_ * d_);
            for (std::size_t k = 0; k < K_; ++k) {
                for (std::size_t j = 0; j < d_; ++j) {
                    const double v = lam_new[j * K_ + k];  // col-major -> row-major
                    if (!(v > 0.0)) {
                        ai4b::stop("set_current: lambda must be > 0");
                    }
                    lam_flat[k * d_ + j] = v;
                }
            }
            impl_->data().set("lambda", lam_flat);
            const arma::vec& cur_mu = impl_->data().get("mu");
            arma::vec cluster_concat(2 * K_ * d_);
            for (std::size_t i = 0; i < K_ * d_; ++i) {
                cluster_concat[i]            = cur_mu[i];
                cluster_concat[K_ * d_ + i]  = lam_flat[i];
            }
            dynamic_cast<normal_gamma_cluster_gibbs_block&>(
                impl_->child(2)).set_current(cluster_concat);
        }
        if (params.count("alpha")) {
            const double a_new = params.at("alpha")[0];
            if (!(a_new > 0.0))
                ai4b::stop("set_current: alpha must be > 0");
            dynamic_cast<nuts_block&>(impl_->child(4))
                .set_current(arma::vec{a_new});
            impl_->data().set("alpha", arma::vec{a_new});
        }
    }

    AI4BayesCode::history_map predict_at(
            const AI4BayesCode::state_map& new_data) const {
        if (!new_data.empty())
            ai4b::stop(
                "DPGaussianMixture: predict_at(new_data) does NOT support "
                "covariate-dependent BNP at v0; pass an empty list/map to draw "
                "y_rep at training X.");

        AI4BayesCode::history_map out;

        if (!keep_history_) {
            // No-history mode: one posterior-predictive draw. y_rep arrives as
            // a flat N*d vec (row-major over observations); emit it as a 1 x
            // (N*d) arma::mat row, mirroring the other dual examples.
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

        // History mode: mu and lambda are sub-outputs of the
        // "cluster_params" block; pi is its own block. Manual-compute
        // y_rep per draw mirrors the refresher lambda. Returned as an
        // n_draws x (N*d) arma::mat (row-major over observations per draw).
        AI4BayesCode::history_map hist = impl_->get_history();
        const arma::mat& pi_hist  = hist.at("pi");      // n_draws x K
        const arma::mat& mu_hist  = hist.at("mu");      // n_draws x (K*d)
        const arma::mat& lam_hist = hist.at("lambda");  // n_draws x (K*d)
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
    void readapt_NUTS(int n, bool reset = false, int max_tree_depth = -1) {
        if (n < 0) {
            ai4b::stop("readapt_NUTS: n must be non-negative");
        }
        impl_->readapt_NUTS(static_cast<std::size_t>(n),
                            reset, readapt_rng_, max_tree_depth < 0 ? std::size_t(0) : static_cast<std::size_t>(max_tree_depth));
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

// ============================================================================
//  Rcpp module
// ============================================================================

#ifdef AI4BAYESCODE_RCPP_MODULE
RCPP_MODULE(DPGaussianMixture_module) {
    Rcpp::class_<DPGaussianMixture>("DPGaussianMixture")
        .constructor<arma::mat, int, int>(
            "DEFAULT (data-driven) constructor; keep_history = FALSE. "
            "DPGaussianMixture(y, K_trunc, seed).")
        .constructor<arma::mat, int, int, bool>(
            "DEFAULT constructor: DPGaussianMixture(y, K_trunc, seed, "
            "keep_history). y is N x d. The conjugate Normal-Gamma "
            "cluster-prior hyperparameters are computed data-driven from "
            "y (mu_0 = column means, kappa_0 = 0.01, a_lambda = 2, "
            "b_lambda = mean column variance, alpha ~ Gamma(1,1)) so the "
            "sampler is correctly scaled to the data and mixes/converges "
            "robustly with no tuning. Use the explicit-hyperparameter "
            "constructor below ONLY for advanced manual control.")
        .constructor<arma::mat, int, arma::vec,
                     double, double, double, double, double, int>(
            "Advanced/explicit-hyperparameter constructor; keep_history "
            "defaults to FALSE. Prefer the data-driven constructor above "
            "unless you specifically need to set the Normal-Gamma hypers.")
        .constructor<arma::mat, int, arma::vec,
                     double, double, double, double, double, int, bool>(
            "Advanced explicit-hyperparameter constructor: "
            "DPGaussianMixture(y, K_trunc, mu_0, kappa_0, "
            "a_lambda_0, b_lambda_0, a_alpha, b_alpha, seed, keep_history). "
            "y is N x d. Truncated stick-breaking at level K_trunc; "
            "diagonal-Gaussian likelihood with conjugate Normal-Gamma "
            "cluster prior; DP concentration alpha sampled by NUTS. "
            "WARNING: mis-scaled fixed hypers (e.g. b_lambda_0 = 1 on "
            "non-unit-variance data) mix poorly -- prefer the data-driven "
            "constructor.")
        .method("step", (void (DPGaussianMixture::*)())    &DPGaussianMixture::step, "Run one sweep.")
        .method("step", (void (DPGaussianMixture::*)(int)) &DPGaussianMixture::step, "Run n sweeps.")
        .method("get_current", &DPGaussianMixture::get_current)
        .method("set_current", &DPGaussianMixture::set_current,
                "Overwrite z, pi, mu, lambda, alpha, or y from a named list.")
        .method("predict_at",  &DPGaussianMixture::predict_at,
                "Posterior predictive y_rep at training X. Empty list only.")
        .method("get_dag",     &DPGaussianMixture::get_dag)
        .method("get_history", &DPGaussianMixture::get_history)
        .method("readapt_NUTS", &DPGaussianMixture::readapt_NUTS);
}
#endif

#ifdef AI4BAYESCODE_PYBIND_MODULE
#include "AI4BayesCode/pybind_casters.hpp"
PYBIND11_MODULE(DPGaussianMixture, m) {
    AI4BayesCode::register_ai4bayescode_types(m);
    pybind11::class_<DPGaussianMixture>(m, "DPGaussianMixture")
        // DEFAULT data-driven constructor: (y N x d, K_trunc, seed, keep_history).
        .def(pybind11::init<arma::mat, int, int, bool>(),
             pybind11::arg("y"),
             pybind11::arg("K_trunc"),
             pybind11::arg("rng_seed") = 1,
             pybind11::arg("keep_history") = false)
        // Advanced explicit-hyperparameter constructor.
        .def(pybind11::init<arma::mat, int, arma::vec,
                            double, double, double, double, double, int, bool>(),
             pybind11::arg("y"),
             pybind11::arg("K_trunc"),
             pybind11::arg("mu_0"),
             pybind11::arg("kappa_0"),
             pybind11::arg("a_lambda_0"),
             pybind11::arg("b_lambda_0"),
             pybind11::arg("a_alpha"),
             pybind11::arg("b_alpha"),
             pybind11::arg("rng_seed") = 1,
             pybind11::arg("keep_history") = false)
        .def("step", (void (DPGaussianMixture::*)())    &DPGaussianMixture::step, "Run one sweep.")
        .def("step", (void (DPGaussianMixture::*)(int)) &DPGaussianMixture::step, pybind11::arg("n_steps"))
        .def("get_current",  &DPGaussianMixture::get_current)
        .def("set_current",  &DPGaussianMixture::set_current, pybind11::arg("params"))
        .def("predict_at",   &DPGaussianMixture::predict_at,  pybind11::arg("new_data"))
        .def("get_dag",      &DPGaussianMixture::get_dag)
        .def("get_history",  &DPGaussianMixture::get_history)
        .def("readapt_NUTS", &DPGaussianMixture::readapt_NUTS,
             pybind11::arg("n"), pybind11::arg("reset") = false, pybind11::arg("max_tree_depth") = -1);
}
#endif
