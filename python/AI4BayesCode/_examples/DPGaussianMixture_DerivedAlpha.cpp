// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later
// (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
//  DPGaussianMixture_DerivedAlpha.cpp
//
//  REFERENCE TEMPLATE for the "alpha-as-function-of-other-params" pattern
//  surfaced in `DESIGN_NOTES_BNP_GP_2026-04-20.md` Q6:
//
//     "User's other project: alpha = complex_function(phi), phi is prior
//      hyperparams of the base distribution, possibly high-dim."
//
//  This file demonstrates the COMPOSITION mechanics. Identical to
//  `DPGaussianMixture.cpp` except:
//
//   - We introduce a real-valued scalar `phi` sampled by a `nuts_block`
//     with prior phi ~ Normal(0, 1).
//   - alpha is REGISTERED as a deterministic refresher of phi:
//                   alpha = exp(phi)
//     via `register_refresher("alpha", ...)` and
//     `declare_invalidates("phi", {"alpha"})`. Whenever phi is updated
//     by its NUTS step, alpha is automatically refreshed.
//   - Downstream blocks (`stick_breaking_block`'s b_fn, etc.) read
//     `alpha` from ctx UNCHANGED — they don't know or care that alpha
//     came from a refresher rather than a sampler. This is the core
//     of the modular-block architecture.
//
//  In real applications, alpha could be any function of any subset of
//  the prior hyperparameters (kappa_0, mu_0_norm, etc.). The pattern
//  is identical: write the function inside the refresher closure, and
//  declare the appropriate invalidates edges.
//
//  Block decomposition (Gibbs sweep order)
//  ---------------------------------------
//      child(0) relabel        dp_label_canonicalizer_block
//                              FALLBACK in-sampler canonicaliser (NOT
//                              RECOMMENDED; the default is POST-MCMC relabeling,
//                              see LABEL SWITCHING below and
//                              skills/label_switching.md). Permutes the K_trunc
//                              slots into descending-pi order each sweep so the
//                              REPORTED pi[slot] converges. Posterior-preserving;
//                              runs FIRST so downstream children read/record on
//                              one canonical labelling. See the class comment.
//      child(1) z              categorical_gibbs_block
//      child(2) cluster_params normal_gamma_cluster_gibbs_block
//      child(3) pi             stick_breaking_block (DP a_fn / b_fn)
//      child(4) phi            nuts_block (REAL constraint)
//
//  LABEL SWITCHING
//  ---------------
//  DEFAULT / RECOMMENDED: resolve label switching POST-MCMC on the recorded
//  draws (skills/label_switching.md) — simple descending-pi / mu sort for
//  well-separated components, or Stephens 2000 + Hungarian for overlapping
//  ones. The truncated-DP mixture likelihood is invariant under permutation of
//  the K_trunc component labels, so a clean raw exchangeable-component sampler
//  is the preferred design and per-slot rank-R-hat is computed AFTER relabeling.
//  alpha (= exp(phi)) and the likelihood are label-invariant and converge as-is.
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
//  Note alpha has NO sampler block — it is a derived key. The alpha
//  log-density is rewritten as a phi log-density via change of variables;
//  the Jacobian d alpha / d phi = exp(phi) IS automatically handled
//  because phi is on the user-natural scale (Normal prior at phi-level)
//  and the LIKELIHOOD term is written directly in terms of phi (no
//  additional Jacobian).
//
//  Specifically:
//      log p(phi | V) = log Normal(phi; 0, 1)
//                     + sum_{k=0..K_trunc-2}
//                         [log(exp(phi)) + (exp(phi) - 1) log(1 - V_k)]
//                     = -0.5 phi^2 - 0.5 log(2 pi)
//                     + (K_trunc - 1) phi
//                     + (exp(phi) - 1) sum_k log(1 - V_k)
//
//      d log p(phi) / d phi
//          = -phi
//          + (K_trunc - 1)
//          + exp(phi) * sum_k log(1 - V_k)
//
//  Verified at gen-time via FD verify file (deleted on PASS).
//
// @example:R
//   library(AI4BayesCode)
//   ai4bayescode_example("DPGaussianMixture_DerivedAlpha")
//   set.seed(2026)
//   ## DGP: 3 well-separated 2-D Gaussian clusters; fit DP mixture with
//   ## K_trunc = 8 truncation slots, alpha = exp(phi) derived from phi.
//   N_k <- 100L; d <- 2L
//   centers <- rbind(c(-4, -4), c(0, 4), c(5, -1))
//   y <- do.call(rbind, lapply(1:3, function(k)
//            matrix(rnorm(N_k * d, mean = centers[k, ], sd = 0.7),
//                   ncol = d, byrow = TRUE)))
//   # ---- Recommended: parallel chains + convergence diagnosis ----
//   run <- ai4bayescode_run_chains(
//       function(seed) new(DPGaussianMixture_DerivedAlpha, y, 8L, seed),
//       n_chains = 4, n_burn = 1000, n_keep = 2000)
//   ai4bayescode_diagnose(run$histories[[1]])      # summary + R-hat/ESS + plots
//   # ---- Advanced: stateful single-chain control ----
//   m <- new(DPGaussianMixture_DerivedAlpha, y, 8L, 7L)  # (y, K_trunc, seed)
//   m$step(1500)
//   cur <- m$get_current()
//   ## # occupied clusters ~ 3; alpha = exp(phi) finite & positive
//   cat("alpha =", cur$alpha, " n_occupied =",
//       length(unique(round(cur$z))), "\n")
// @example:python
//   import numpy as np, AI4BayesCode
//   rng = np.random.default_rng(2026)
//   ## DGP: 3 well-separated 2-D Gaussian clusters.
//   N_k, d = 100, 2
//   centers = np.array([[-4.0, -4.0], [0.0, 4.0], [5.0, -1.0]])
//   y = np.vstack([rng.normal(centers[k], 0.7, size=(N_k, d)) for k in range(3)])
//   Mod = AI4BayesCode.example("DPGaussianMixture_DerivedAlpha")
//   # ---- Recommended: parallel chains + diagnosis ----
//   chains = AI4BayesCode.run_chains(
//       lambda seed: Mod.DPGaussianMixture_DerivedAlpha(y, 8, seed),
//       seeds=[101, 202, 303, 404], n_burn=1000, n_keep=2000, n_jobs=1)
//   AI4BayesCode.diagnose(chains[0]["hist"])   # summary + diagnostics
//   # ---- Advanced: stateful single-chain control ----
//   m = Mod.DPGaussianMixture_DerivedAlpha(y, 8, 7)  # (y, K_trunc, seed)
//   m.step(1500)
//   cur = m.get_current()
//   alpha = float(np.asarray(cur["alpha"]).ravel()[0])
//   n_occ = len(np.unique(np.round(np.asarray(cur["z"]))))
//   print("alpha =", alpha, " n_occupied =", n_occ)
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

/// digamma (psi) for x > 0: recurrence to x>=6 then the asymptotic expansion;
/// matches R's digamma() to ~1e-12. std has no digamma; example is frontend-independent.
double digamma_psi(double x) {
    double result = 0.0;
    while (x < 6.0) { result -= 1.0 / x; x += 1.0; }
    const double inv  = 1.0 / x;
    const double inv2 = inv * inv;
    result += std::log(x) - 0.5 * inv
            - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 * (1.0 / 252.0)));
    return result;
}

/// log p(phi | k, n) where alpha = exp(phi) — the COLLAPSED / exact-DP concentration
/// conditional (Antoniak 1974; Escobar & West 1995) in the log-parameterisation.
/// alpha's SUFFICIENT STATISTICS are k = #OCCUPIED clusters and n = #observations;
/// it does NOT depend on the stick weights. The previous version conditioned on the
/// truncated sticks, coupling alpha to the empty-cluster sticks (each Beta(1,alpha),
/// pure prior draws) -> alpha mixed terribly (split-R-hat ~2-3; an exact draw is just
/// as bad). Conditioning on (k, n) decouples them and mixes (R-hat ~1.0). NOTE this
/// demonstrates the Antoniak update with a NON-Gamma prior: here alpha = exp(phi),
/// phi ~ N(0,1), i.e. alpha is LOG-NORMAL — the (k,n) marginal is prior-agnostic.
///
///   p(k | alpha, n) ∝ alpha^k * Gamma(alpha) / Gamma(alpha + n)   [Antoniak]
///   prior: phi ~ N(0,1)   <-- swap THIS term for any prior on phi/alpha
///   alpha = exp(phi);  alpha^k = exp(k phi)
///   => log p(phi|k,n) = -0.5 phi^2 - 0.5 log(2 pi)
///                       + k*phi + lgamma(alpha) - lgamma(alpha + n)
///   d/d phi = -phi + k + alpha*( digamma(alpha) - digamma(alpha + n) )
double phi_natural_log_density(const arma::vec& phi_nat,
                                const block_context& ctx,
                                arma::vec* grad_nat) {
    constexpr double kLog2Pi = 1.83787706640934548356065947281;
    const double phi = phi_nat[0];
    if (!std::isfinite(phi)) {
        if (grad_nat) { grad_nat->set_size(1); (*grad_nat)[0] = 0.0; }
        return -std::numeric_limits<double>::infinity();
    }
    const arma::vec& counts = ctx.at("cluster_counts");
    // k = number of OCCUPIED clusters; n = total observations = sum(counts).
    double n = 0.0;
    std::size_t k = 0;
    for (std::size_t j = 0; j < counts.n_elem; ++j) {
        n += counts[j];
        if (counts[j] > 0.5) ++k;
    }
    const double kk    = static_cast<double>(k);
    const double alpha = std::exp(phi);

    // (k,n) term: lgamma(alpha)-lgamma(alpha+n) and its derivative
    // digamma(alpha)-digamma(alpha+n). For LARGE alpha the direct lgamma/digamma
    // differences suffer CATASTROPHIC CANCELLATION: both lgammas are ~alpha*log(alpha)
    // (~4e25 at alpha=8e23) and their true difference (~ -n*log(alpha)) is lost
    // below the float ULP, computing as ~0. That makes a NUTS fling to huge alpha
    // (here alpha=exp(phi)) get a SPURIOUSLY GOOD lp -> the chain freezes there and
    // the escape guards are fooled. The alpha>>n asymptotic restores the correct
    // value + gradient (matches the exact -sum log(alpha+i) to many digits).
    double kn_lp, kn_d;   // lgamma(a)-lgamma(a+n) ; digamma(a)-digamma(a+n)
    if (alpha > 1.0e6) {
        kn_lp = -(n * std::log(alpha) + n * (n - 1.0) / (2.0 * alpha));
        kn_d  = -n / alpha + n * (n - 1.0) / (2.0 * alpha * alpha);
    } else {
        kn_lp = std::lgamma(alpha) - std::lgamma(alpha + n);
        kn_d  = digamma_psi(alpha) - digamma_psi(alpha + n);
    }

    const double lp =
        -0.5 * phi * phi - 0.5 * kLog2Pi + kk * phi + kn_lp;

    if (grad_nat) {
        grad_nat->set_size(1);
        (*grad_nat)[0] = -phi + kk + alpha * kn_d;
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

class DPGaussianMixture_DerivedAlpha {
public:
    // Data-driven weakly-informative Normal-Gamma hypers (see CRITICAL
    // note in skills/block_catalogue/index.md; verified on DPGaussianMixture).
    // Fixed mis-scaled hypers over-segment into a WRONG posterior R-hat
    // cannot flag; computing from y fixes it.
    static arma::vec dd_mu0_(const arma::mat& y) {
        const int n = static_cast<int>(y.n_rows), d = static_cast<int>(y.n_cols);
        arma::vec m(d);
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
    /// Normal-Gamma hypers from y. Delegates to the explicit ctor
    /// (that path is unchanged).
    DPGaussianMixture_DerivedAlpha(const arma::mat& y,
                                   int K_trunc,
                                   int rng_seed,
                                   bool keep_history = false)
        : DPGaussianMixture_DerivedAlpha(y, K_trunc,
                                         dd_mu0_(y), 0.01, 2.0,
                                         dd_blambda_(y),
                                         rng_seed, keep_history) {}

    DPGaussianMixture_DerivedAlpha(const arma::mat& y,
                                    int K_trunc,
                                    const arma::vec& mu_0,
                                    double kappa_0,
                                    double a_lambda_0,
                                    double b_lambda_0,
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
          impl_(std::make_unique<composite_block>(
                "DPGaussianMixture_DerivedAlpha")),
          keep_history_(keep_history)
    {
        if (y.n_rows < 2)        ai4b::stop("N must be >= 2");
        if (y.n_cols < 1)        ai4b::stop("d must be >= 1");
        if (K_trunc < 2)         ai4b::stop("K_trunc must be >= 2");
        if (mu_0.n_elem != y.n_cols)
            ai4b::stop("mu_0 length must equal d");
        if (!(kappa_0 > 0.0))      ai4b::stop("kappa_0 must be > 0");
        if (!(a_lambda_0 > 0.0))   ai4b::stop("a_lambda_0 must be > 0");
        if (!(b_lambda_0 > 0.0))   ai4b::stop("b_lambda_0 must be > 0");

        N_ = static_cast<std::size_t>(y.n_rows);
        d_ = static_cast<std::size_t>(y.n_cols);
        K_ = static_cast<std::size_t>(K_trunc);

        arma::vec y_flat(N_ * d_);
        for (std::size_t i = 0; i < N_; ++i)
            for (std::size_t j = 0; j < d_; ++j)
                y_flat[i * d_ + j] = y(i, j);
        impl_->data().set("y", y_flat);

        arma::vec mu0_arma(d_);
        for (std::size_t j = 0; j < d_; ++j) mu0_arma[j] = mu_0[j];
        impl_->data().set("mu_0", mu0_arma);
        impl_->data().set("kappa_0",     arma::vec{kappa_0});
        impl_->data().set("a_lambda_0",  arma::vec{a_lambda_0});
        impl_->data().set("b_lambda_0",  arma::vec{b_lambda_0});

        // phi sampled directly; alpha derived. Initial phi = 0 => alpha=1.
        impl_->data().set("phi",   arma::vec{0.0});
        impl_->data().set("alpha", arma::vec{1.0});

        // Initial state for the rest.
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

        impl_->data().set("cluster_counts", arma::vec(K_, arma::fill::zeros));
        impl_->data().register_refresher("cluster_counts",
            [K = K_](const AI4BayesCode::shared_data_t& d) -> arma::vec {
                const arma::vec& z = d.get("z");
                return bnp::counts_from_z(z, K);
            });

        // *** KEY DEMONSTRATION ***
        // Register alpha as a deterministic refresher of phi:
        //     alpha = exp(phi)
        impl_->data().register_refresher("alpha",
            [](const AI4BayesCode::shared_data_t& d) -> arma::vec {
                const double phi = d.get("phi")[0];
                return arma::vec{std::exp(phi)};
            });

        // ---- DAG ---------------------------------------------------
        // relabel (FALLBACK canonicaliser — NOT recommended; prefer post-MCMC).
        impl_->data().declare_dependencies("relabel",
            {"pi", "mu", "lambda", "z"});
        impl_->data().declare_dependencies("z",
            {"y", "pi", "mu", "lambda"});
        impl_->data().declare_dependencies("cluster_params",
            {"z", "y"});
        impl_->data().declare_dependencies("pi",
            {"cluster_counts", "alpha"});
        // cluster_counts — the exact-DP (Antoniak) update, NOT on the sticks.
        impl_->data().declare_dependencies("phi",
            {"cluster_counts"});

        impl_->data().declare_invalidates("z",   {"cluster_counts"});
        impl_->data().declare_invalidates("phi", {"alpha"});

        // (No declare_data_input here — y is an observed terminal,
        // not a replaceable covariate. The y_rep refresher reads
        // pi/mu/lambda, NOT y.)
        impl_->data().declare_predict_edges("pi",     {"y_rep"});
        impl_->data().declare_predict_edges("mu",     {"y_rep"});
        impl_->data().declare_predict_edges("lambda", {"y_rep"});

        // ---- Generative-DAG context (VIZ-ONLY; predict_at BFS never
        //      reads context_edges_). alpha = exp(phi), phi ~ N(0,1)
        //      (derived-alpha pattern, alpha is a refresher of phi);
        //      alpha drives the stick-breaking; (mu_k, lambda_k) ~
        //      NormalGamma(mu_0, kappa_0, a_lambda_0, b_lambda_0).
        //      Drawn faded by ai4bayescode_plot_dag.
        impl_->data().declare_context_edges("phi",         {"alpha"});
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
        // child(0) relabel (FALLBACK canonicaliser — see class comment; NOT
        // recommended, prefer post-MCMC). Added FIRST so downstream children
        // read/record on one canonical descending-pi labelling.
        impl_->add_child(std::make_unique<dp_label_canonicalizer_block>(
            "relabel", K_, d_, N_));

        // child(1) z
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

        // child(2) cluster_params
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

        // child(3) pi (DP weights, reads alpha from refresher)
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
                const double a = ctx.at("alpha")[0];  // from refresher!
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

        // child(4) phi (NUTS on REAL)
        {
            nuts_block_config cfg;
            cfg.name        = "phi";
            cfg.initial_unc = arma::vec{0.0};
            // No constrain/unconstrain => identity (REAL).
            cfg.log_density_grad =
                [](const arma::vec& t_unc, const block_context& ctx,
                   arma::vec* grad) -> double {
                    return phi_natural_log_density(t_unc, ctx, grad);
                };
            impl_->add_child(std::make_unique<nuts_block>(std::move(cfg)));
        }

        if (keep_history_) impl_->set_keep_history(true);
    }

    void step() { step(1); }              // no-arg convenience: one sweep

    void step(int n_steps) {
        if (n_steps < 0) ai4b::stop("n_steps must be >= 0");
        for (int i = 0; i < n_steps; ++i) impl_->step(rng_);
    }

    // Backend-neutral get_current: returns a state_map (map<string, arma::vec>).
    // mu and lambda are returned as FLAT row-major (k*d_+j) arma::vec of length
    // K_trunc*d (reshape to K_trunc x d in R/Python if a matrix is wanted);
    // phi/alpha/K_trunc are 1-element arma::vec (read scalar as x[0] / x.ravel()[0]).
    AI4BayesCode::state_map get_current() const {
        AI4BayesCode::state_map out;
        out["z"]       = impl_->data().get("z");
        out["pi"]      = impl_->data().get("pi");
        out["mu"]      = impl_->data().get("mu");      // flat K_trunc*d (k*d_+j)
        out["lambda"]  = impl_->data().get("lambda");  // flat K_trunc*d (k*d_+j)
        out["phi"]     = impl_->data().get("phi");     // length-1
        out["alpha"]   = impl_->data().get("alpha");   // length-1
        out["K_trunc"] = arma::vec{static_cast<double>(K_)};
        return out;
    }

    // Backend-neutral set_current: read from a state_map (map<string,arma::vec>).
    // "y" arrives as a vectorised COLUMN-MAJOR arma::vec of a N_ x d_ matrix
    // (the standard state_map matrix convention), i.e. y_col[j*N_ + i] = y(i,j);
    // it is restored into the model's row-major (i*d_+j) "y" buffer.
    void set_current(const AI4BayesCode::state_map& params) {
        auto it_y = params.find("y");
        if (it_y != params.end()) {
            const arma::vec& y_col = it_y->second;   // column-major N_ x d_
            // STRICT-N legitimate (Check #21): z allocation length-N.
            if (y_col.n_elem != N_ * d_)
                ai4b::stop("set_current: DPGaussianMixture_DerivedAlpha "
                           "fixes N and d at construction (z is length-N). "
                           "Supplied y has %d elems; required N*d = %d. "
                           "Reconstruct to change N/d.",
                           static_cast<int>(y_col.n_elem),
                           static_cast<int>(N_ * d_));
            arma::vec yflat(N_ * d_);
            for (std::size_t i = 0; i < N_; ++i)
                for (std::size_t j = 0; j < d_; ++j)
                    yflat[i * d_ + j] = y_col[j * N_ + i];
            impl_->data().set("y", yflat);
        }
        auto it_z = params.find("z");
        if (it_z != params.end()) {
            arma::vec znew = it_z->second;
            if (znew.n_elem != N_) ai4b::stop("z length mismatch");
            for (std::size_t i = 0; i < N_; ++i) {
                const long lab = static_cast<long>(std::llround(znew[i]));
                if (lab < 1 || static_cast<std::size_t>(lab) > K_)
                    ai4b::stop("z[i] out of {1, ..., K_trunc}");
            }
            dynamic_cast<categorical_gibbs_block&>(
                impl_->child(1)).set_current(znew);
            impl_->data().set("z", znew);
            impl_->data().refresh_derived_for("z");
        }
        auto it_pi = params.find("pi");
        if (it_pi != params.end()) {
            arma::vec pinew = it_pi->second;
            if (pinew.n_elem != K_)
                ai4b::stop("pi length must equal K_trunc");
            dynamic_cast<stick_breaking_block&>(
                impl_->child(3)).set_current(pinew);
            impl_->data().set("pi", pinew);
        }
        auto it_phi = params.find("phi");
        if (it_phi != params.end()) {
            const double p_new = it_phi->second[0];
            if (!std::isfinite(p_new)) ai4b::stop("phi must be finite");
            dynamic_cast<nuts_block&>(impl_->child(4))
                .set_current(arma::vec{p_new});
            impl_->data().set("phi", arma::vec{p_new});
            impl_->data().refresh_derived_for("phi");  // updates alpha
        }
    }

    // Backend-neutral predict_at: takes an empty state_map (no covariates at v0)
    // and returns a history_map (map<string, arma::mat>). In no-history mode each
    // refreshed node is a 1-row arma::mat (y_rep is 1 x N_*d_, FLAT row-major
    // i*d_+j — reshape to N_ x d_ downstream). In history mode y_rep is
    // n_draws x N_*d_.
    AI4BayesCode::history_map predict_at(const AI4BayesCode::state_map& new_data) const {
        if (!new_data.empty())
            ai4b::stop(
                "DPGaussianMixture_DerivedAlpha: predict_at(new_data) "
                "does NOT support covariate-dependent BNP at v0; "
                "pass an empty map/list.");

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

        // History mode: see DPGaussianMixture.cpp predict_at for the
        // canonical mixture history-mode pattern. Mu/lambda are sub-outputs
        // of "cluster_params" so we manual-compute per draw.
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

#ifdef AI4BAYESCODE_RCPP_MODULE
RCPP_MODULE(DPGaussianMixture_DerivedAlpha_module) {
    Rcpp::class_<DPGaussianMixture_DerivedAlpha>(
        "DPGaussianMixture_DerivedAlpha")
        .constructor<arma::mat, int, int>(
            "DEFAULT (data-driven) ctor; keep_history=FALSE. "
            "DPGaussianMixture_DerivedAlpha(y, K_trunc, seed).")
        .constructor<arma::mat, int, int, bool>(
            "DEFAULT ctor: (y, K_trunc, seed, keep_history). Normal-Gamma "
            "cluster hypers computed data-driven from y (mu_0=col means, "
            "kappa_0=0.01, a_lambda=2, b_lambda=mean col var) so the "
            "sampler is correctly scaled and recovers the true cluster "
            "structure. Prefer this; the explicit-hyper ctor is advanced.")
        .constructor<arma::mat, int, arma::vec,
                     double, double, double, int>(
            "Advanced explicit-hyper constructor; keep_history=FALSE. "
            "Prefer the data-driven ctor above.")
        .constructor<arma::mat, int, arma::vec,
                     double, double, double, int, bool>(
            "Construct DPGaussianMixture_DerivedAlpha(y, K_trunc, mu_0, "
            "kappa_0, a_lambda_0, b_lambda_0, seed, keep_history). "
            "DP mixture where alpha = exp(phi) is DERIVED from phi via "
            "register_refresher; phi has Normal(0, 1) prior and is "
            "sampled by NUTS. Demonstrates the alpha-as-derived "
            "composition pattern from DESIGN_NOTES_BNP_GP_2026-04-20.md "
            "Q6.")
        .method("step", (void (DPGaussianMixture_DerivedAlpha::*)())    &DPGaussianMixture_DerivedAlpha::step, "Run one sweep.")
        .method("step", (void (DPGaussianMixture_DerivedAlpha::*)(int)) &DPGaussianMixture_DerivedAlpha::step, "Run n sweeps.")
        .method("get_current", &DPGaussianMixture_DerivedAlpha::get_current)
        .method("set_current", &DPGaussianMixture_DerivedAlpha::set_current,
                "Overwrite z, pi, phi, or y from a named list.")
        .method("predict_at",  &DPGaussianMixture_DerivedAlpha::predict_at,
                "Posterior predictive y_rep at training X.")
        .method("get_dag",     &DPGaussianMixture_DerivedAlpha::get_dag)
        .method("get_history", &DPGaussianMixture_DerivedAlpha::get_history)
        .method("readapt_NUTS", &DPGaussianMixture_DerivedAlpha::readapt_NUTS);
}
#endif

#ifdef AI4BAYESCODE_PYBIND_MODULE
#include "AI4BayesCode/pybind_casters.hpp"
PYBIND11_MODULE(DPGaussianMixture_DerivedAlpha, m) {
    AI4BayesCode::register_ai4bayescode_types(m);
    pybind11::class_<DPGaussianMixture_DerivedAlpha>(
        m, "DPGaussianMixture_DerivedAlpha")
        // DEFAULT data-driven ctor: (y, K_trunc, rng_seed, keep_history).
        // Normal-Gamma cluster hypers computed data-driven from y. alpha =
        // exp(phi) DERIVED from phi (Normal(0,1) prior, NUTS-sampled) via
        // register_refresher (DESIGN_NOTES_BNP_GP_2026-04-20.md Q6).
        .def(pybind11::init<arma::mat, int, int, bool>(),
             pybind11::arg("y"),
             pybind11::arg("K_trunc"),
             pybind11::arg("rng_seed") = 1,
             pybind11::arg("keep_history") = false)
        // Advanced explicit-hyper ctor.
        .def(pybind11::init<arma::mat, int, arma::vec,
                            double, double, double, int, bool>(),
             pybind11::arg("y"),
             pybind11::arg("K_trunc"),
             pybind11::arg("mu_0"),
             pybind11::arg("kappa_0"),
             pybind11::arg("a_lambda_0"),
             pybind11::arg("b_lambda_0"),
             pybind11::arg("rng_seed") = 1,
             pybind11::arg("keep_history") = false)
        .def("step", (void (DPGaussianMixture_DerivedAlpha::*)())    &DPGaussianMixture_DerivedAlpha::step, "Run one sweep.")
        .def("step", (void (DPGaussianMixture_DerivedAlpha::*)(int)) &DPGaussianMixture_DerivedAlpha::step, pybind11::arg("n_steps"))
        .def("get_current",  &DPGaussianMixture_DerivedAlpha::get_current)
        .def("set_current",  &DPGaussianMixture_DerivedAlpha::set_current,
             pybind11::arg("params"))
        .def("predict_at",   &DPGaussianMixture_DerivedAlpha::predict_at,
             pybind11::arg("new_data"))
        .def("get_dag",      &DPGaussianMixture_DerivedAlpha::get_dag)
        .def("get_history",  &DPGaussianMixture_DerivedAlpha::get_history)
        .def("readapt_NUTS", &DPGaussianMixture_DerivedAlpha::readapt_NUTS,
             pybind11::arg("n"), pybind11::arg("reset") = false, pybind11::arg("max_tree_depth") = -1);
}
#endif
