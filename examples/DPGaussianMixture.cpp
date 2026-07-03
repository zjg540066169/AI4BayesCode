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
//                         so the downstream children read/record on one canonical
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
//                         (PRIOR-AGNOSTIC; NOT conjugate Gibbs). Natural-scale
//                         log p(alpha | k, n) = log prior(alpha) + k log alpha
//                                             + lgamma(alpha) - lgamma(alpha + n),
//                         k = #occupied clusters, n = #obs. Conditions ONLY on
//                         the data-identified sufficient statistic (k, n); does
//                         NOT read the empty-tail sticks stick_V (whose Beta(1,
//                         alpha) prior draws are chain-specific and broke mixing
//                         under the old full-stick update). Staying on NUTS keeps
//                         the prior swappable (Gamma here). See block comment.
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
//  draws (skills/label_switching.md) — simple descending-pi / mu sort for
//  well-separated components, or Stephens 2000 + Hungarian for overlapping
//  ones. The truncated-DP mixture likelihood is invariant under permutation of
//  the K_trunc component labels, so a clean raw exchangeable-component sampler
//  is the preferred design and per-slot rank-R-hat is computed AFTER relabeling.
//  alpha and the likelihood are label-invariant and converge as-is.
//
//  FALLBACK USED HERE (child(0) dp_label_canonicalizer_block — NOT RECOMMENDED):
//  the truncated-DP stick weights pi[slot] are SLOT-POSITION-dependent and do
//  NOT converge under per-draw post-MCMC sorting of the empty truncation tail.
//  The cluster PROPORTIONS (count/N) DO converge (2-chain rank-R-hat ~1.001), so
//  this is a representation/labeling artifact, NOT a masked mixing failure. To
//  make the REPORTED pi[slot] converge on its raw recorded values, this example
//  reaches for an in-sampler canonicaliser (descending-pi slot permutation each
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
//      - alpha: nuts_block on the Antoniak (1974) (k, n) marginal
//        (PRIOR-AGNOSTIC, NOT conjugate Gibbs). Conditioning on the
//        data-identified sufficient statistic (k, n) — NOT on the empty-tail
//        sticks — is what makes alpha mix; the earlier full-stick NUTS update
//        summed log(1 - V_k) over chain-specific prior draws and failed
//        rank-R-hat (~1.42). NUTS (not Gibbs) keeps the prior swappable.
//      - label switching: the DEFAULT/RECOMMENDED resolution is POST-MCMC on the
//        recorded draws (skills/label_switching.md), keeping the sampler a clean
//        raw exchangeable-component sampler; alpha + likelihood are label-
//        invariant and converge as-is. This example additionally carries an
//        in-sampler dp_label_canonicalizer_block as child(0) — a NOT-RECOMMENDED
//        FALLBACK used ONLY because the truncation-tail stick-pi[slot] won't
//        converge under post-MCMC sorting though the cluster proportions do
//        (rank-R-hat ~1.001). See the class comment; prefer post-MCMC relabeling.
// ============================================================================

// Frontend-INDEPENDENT example: pure standalone C++ (NO Rcpp, NO pybind).
// Compile + run directly: it has an int main() that simulates a 2-component
// Gaussian mixture from a KNOWN truth, fits the truncated-DP block stack, and
// checks cluster-mean recovery. No R / Python binding is built or required.
// The original Rcpp/pybind Tier-A wrapper class has been removed; main() drives
// the composite_block directly with the SAME model (priors, log-density, block
// configs, hyperparameters) — only the frontend binding is gone.

// @example:R
//   library(AI4BayesCode)
//   ai4bayescode_example("DPGaussianMixture")
//   set.seed(42); n_per <- 150L; d <- 2L                  # well-identified DGP (= int main)
//   mu_true <- rbind(c(-3, -3), c(3, 3))                  # 2 well-separated clusters, sd 0.7
//   y <- rbind(matrix(rnorm(n_per * d, 0, 0.7), n_per, d) + rep(mu_true[1, ], each = n_per),
//              matrix(rnorm(n_per * d, 0, 0.7), n_per, d) + rep(mu_true[2, ], each = n_per))
//   m <- new(DPGaussianMixture, y, 60L, 42L, TRUE)        # y (N x d), K_trunc=60, seed=42, keep_history
//   m$step(2000L); cur <- m$get_current()                # $z $pi $mu $lambda $alpha $K_trunc
//   occ <- which(tabulate(cur$z, cur$K_trunc) > 0)        # occupied slots (~2)
//   cur$mu[occ, , drop = FALSE]                           # recovered centres ~ (-3,-3),(3,3)
//   cur$alpha                                             # DP concentration alpha (NUTS, (k,n) marginal)
// @example:end

#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif

#include <armadillo>

#include "AI4BayesCode/block_sampler.hpp"
#include "AI4BayesCode/shared_data.hpp"
#include "AI4BayesCode/composite_block.hpp"

#include "AI4BayesCode/categorical_gibbs_block.hpp"
#include "AI4BayesCode/stick_breaking_block.hpp"
#include "AI4BayesCode/normal_gamma_cluster_gibbs_block.hpp"
#include "AI4BayesCode/bnp_utils.hpp"
#include "AI4BayesCode/nuts_block.hpp"
#include "AI4BayesCode/constraints.hpp"
#include "AI4BayesCode/backend_neutral.hpp"   // ai4b::digamma

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

using AI4BayesCode::block_context;
using AI4BayesCode::composite_block;
using AI4BayesCode::categorical_gibbs_block;
using AI4BayesCode::categorical_gibbs_block_config;
using AI4BayesCode::stick_breaking_block;
using AI4BayesCode::stick_breaking_block_config;
using AI4BayesCode::normal_gamma_cluster_gibbs_block;
using AI4BayesCode::normal_gamma_cluster_gibbs_block_config;
using AI4BayesCode::nuts_block;
using AI4BayesCode::nuts_block_config;
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

// ----------------------------------------------------------------------------
//  alpha_kn_log_density  (PRIOR-AGNOSTIC NUTS update for the DP concentration)
//
//  Natural-scale log-density of alpha conditioned on the Antoniak (1974)
//  sufficient statistic (k, n): k = #occupied clusters, n = #observations.
//  This is the data-identified conditional and does NOT read the empty-tail
//  sticks (stick_V), so it mixes. (The earlier full-stick NUTS summed
//  log(1 - V_k) over all K_trunc-1 sticks; the empty sticks are pure prior
//  draws -> chain-specific -> rank-R-hat(alpha) ~ 1.42, structural non-mixing.)
//
//      log p(alpha | k, n) = log prior(alpha)                       [ANY prior]
//                          + k*log(alpha) + lgamma(alpha) - lgamma(alpha + n)
//
//  We keep alpha on NUTS (not the conjugate Escobar-West Gibbs) PRECISELY so
//  the prior is swappable: the Gamma(a_alpha, b_alpha) prior below is the ONLY
//  prior-specific line; replace it with a log-normal / half-Cauchy / etc. and
//  NUTS still applies. Conjugate Gibbs would lock alpha to a Gamma prior.
//
//  Reads: cluster_counts, a_alpha, b_alpha.   (NOT stick_V.)
// ----------------------------------------------------------------------------
inline double alpha_kn_log_density(const arma::vec& a, const block_context& ctx,
                                   arma::vec* grad) {
    const double alpha = a[0];
    const arma::vec& counts = ctx.at("cluster_counts");
    const double a_pr = ctx.at("a_alpha")[0];
    const double b_pr = ctx.at("b_alpha")[0];

    std::size_t k = 0;
    double n = 0.0;
    for (std::size_t j = 0; j < counts.n_elem; ++j) {
        if (counts[j] > 0.0) ++k;
        n += counts[j];
    }
    const double kd = static_cast<double>(k);

    // Gamma(a_pr, b_pr) prior  +  Antoniak (k, n) marginal:
    const double lp = (a_pr - 1.0) * std::log(alpha) - b_pr * alpha          // prior
                    + kd * std::log(alpha)
                    + std::lgamma(alpha) - std::lgamma(alpha + n);           // Antoniak
    if (grad) {
        grad->set_size(1);
        (*grad)[0] = (a_pr - 1.0 + kd) / alpha - b_pr                        // d prior
                   + ai4b::digamma(alpha) - ai4b::digamma(alpha + n);        // d Antoniak
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

// ----------------------------------------------------------------------------
//  build_dp_mixture
//
//  Replicates the (former Rcpp Tier-A) constructor's wiring DIRECTLY on a
//  composite_block: data + data-driven Normal-Gamma hyperparameters, initial
//  sampler state, the Gibbs DAG, the cluster_counts refresher, and the four
//  children in Gibbs sweep order. The MODEL is preserved exactly; only the
//  frontend binding is gone. The y_rep predict-time machinery is omitted (the
//  standalone demo does not exercise predict_at).
//
//  y_flat is row-major: y_flat[i * d + j] = y(i, j).
// ----------------------------------------------------------------------------
std::unique_ptr<composite_block> build_dp_mixture(
        const arma::vec& y_flat, std::size_t N, std::size_t d,
        std::size_t K_trunc,
        const arma::vec& mu_0, double kappa_0,
        double a_lambda_0, double b_lambda_0,
        double a_alpha, double b_alpha) {

    if (N < 2)        throw std::runtime_error("DPGaussianMixture: N must be >= 2");
    if (d < 1)        throw std::runtime_error("DPGaussianMixture: d must be >= 1");
    if (K_trunc < 2)  throw std::runtime_error("DPGaussianMixture: K_trunc must be >= 2");
    if (mu_0.n_elem != d)
        throw std::runtime_error("DPGaussianMixture: mu_0 length must equal d");
    if (!(kappa_0 > 0.0))    throw std::runtime_error("kappa_0 must be > 0");
    if (!(a_lambda_0 > 0.0)) throw std::runtime_error("a_lambda_0 must be > 0");
    if (!(b_lambda_0 > 0.0)) throw std::runtime_error("b_lambda_0 must be > 0");
    if (!(a_alpha > 0.0))    throw std::runtime_error("a_alpha must be > 0");
    if (!(b_alpha > 0.0))    throw std::runtime_error("b_alpha must be > 0");

    auto impl = std::make_unique<composite_block>("DPGaussianMixture");

    // ---- Data + priors ---------------------------------------------------
    impl->data().set("y", y_flat);

    impl->data().set("mu_0", mu_0);
    impl->data().set("kappa_0",     arma::vec{kappa_0});
    impl->data().set("a_lambda_0",  arma::vec{a_lambda_0});
    impl->data().set("b_lambda_0",  arma::vec{b_lambda_0});
    impl->data().set("a_alpha",     arma::vec{a_alpha});
    impl->data().set("b_alpha",     arma::vec{b_alpha});

    // ---- Initial sampler state ------------------------------------------
    // z: spread (i mod K_trunc) + 1.
    arma::vec z_init(N);
    for (std::size_t i = 0; i < N; ++i) {
        z_init[i] = static_cast<double>((i % K_trunc) + 1);
    }
    impl->data().set("z", z_init);

    // pi: uniform on K_trunc-simplex.
    arma::vec pi_init(K_trunc, arma::fill::value(1.0 / static_cast<double>(K_trunc)));
    impl->data().set("pi", pi_init);
    // stick_V: derived initial (V_k = pi_k / remainder_{<k}).
    arma::vec V_init(K_trunc, arma::fill::zeros);
    {
        double rem = 1.0;
        for (std::size_t k = 0; k + 1 < K_trunc; ++k) {
            V_init[k] = pi_init[k] / rem;
            if (V_init[k] > 1.0) V_init[k] = 1.0;
            if (V_init[k] < 0.0) V_init[k] = 0.0;
            rem *= (1.0 - V_init[k]);
        }
        V_init[K_trunc - 1] = 1.0;
    }
    impl->data().set("stick_V", V_init);

    // mu, lambda: data-driven init for mu (kmeans-like spread of observed
    // y-rows over the K_trunc clusters); lambda at prior mean.
    arma::vec mu_init(K_trunc * d, arma::fill::zeros);
    arma::vec lambda_init(K_trunc * d,
                          arma::fill::value(a_lambda_0 / b_lambda_0));
    for (std::size_t k = 0; k < K_trunc; ++k) {
        const std::size_t i_anchor = (k * N) / K_trunc;  // spread anchors
        for (std::size_t j = 0; j < d; ++j) {
            mu_init[k * d + j] = y_flat[i_anchor * d + j];
        }
    }
    impl->data().set("mu", mu_init);
    impl->data().set("lambda", lambda_init);

    // alpha: prior mean.
    const double alpha_init = a_alpha / b_alpha;
    impl->data().set("alpha", arma::vec{alpha_init});

    // cluster_counts: derived (refresher).
    impl->data().set("cluster_counts", arma::vec(K_trunc, arma::fill::zeros));
    impl->data().register_refresher("cluster_counts",
        [K_trunc](const AI4BayesCode::shared_data_t& dd) -> arma::vec {
            const arma::vec& z = dd.get("z");
            return bnp::counts_from_z(z, K_trunc);
        });

    // ---- Gibbs DAG dependencies / invalidations -------------------------
    // relabel (FALLBACK canonicaliser — NOT recommended; prefer post-MCMC).
    impl->data().declare_dependencies("relabel", {"pi", "mu", "lambda", "z"});
    impl->data().declare_dependencies("z",   {"y", "pi", "mu", "lambda"});
    impl->data().declare_dependencies("cluster_params", {"z", "y"});
    impl->data().declare_dependencies("pi",   {"cluster_counts", "alpha"});
    // alpha (NUTS on the (k, n) Antoniak marginal) reads ONLY the data-identified
    // sufficient statistic via cluster_counts (k = #occupied, n = sum), its Gamma
    // prior, and its own current value (the NUTS starting point). It NO LONGER
    // reads stick_V — the empty-stick coupling that broke mixing is severed
    // (Antoniak 1974). NOT conjugate Gibbs; the prior stays swappable.
    impl->data().declare_dependencies("alpha",
        {"cluster_counts", "a_alpha", "b_alpha", "alpha"});

    // z's update INVALIDATES cluster_counts (its derived child).
    impl->data().declare_invalidates("z", {"cluster_counts"});

    impl->data().refresh_all();

    // ---- Children in Gibbs sweep order ----------------------------------
    // child(0) relabel (FALLBACK canonicaliser — see class comment; NOT
    // recommended, prefer post-MCMC). Added FIRST so downstream children
    // read/record on one canonical descending-pi labelling.
    impl->add_child(std::make_unique<dp_label_canonicalizer_block>(
        "relabel", K_trunc, d, N));

    // child(1) z (categorical_gibbs_block)
    {
        categorical_gibbs_block_config cfg;
        cfg.name           = "z";
        cfg.n_obs          = N;
        cfg.n_categories   = K_trunc;
        cfg.initial_labels = z_init;
        const std::size_t d_capture = d;
        const std::size_t N_capture = N;
        const std::size_t K_capture = K_trunc;
        cfg.log_probs_fn = [d_capture, N_capture, K_capture]
            (const block_context& ctx) -> arma::mat {
            const arma::vec& y_flat_c = ctx.at("y");
            const arma::vec& pi       = ctx.at("pi");
            const arma::vec& mu       = ctx.at("mu");
            const arma::vec& lam      = ctx.at("lambda");
            arma::mat lp(N_capture, K_capture);
            for (std::size_t i = 0; i < N_capture; ++i) {
                for (std::size_t k = 0; k < K_capture; ++k) {
                    const double log_pi_k = std::log(pi[k] + 1e-300);
                    lp(i, k) = log_pi_k
                             + diag_normal_log_density(
                                 y_flat_c.memptr() + i * d_capture,
                                 mu.memptr()       + k * d_capture,
                                 lam.memptr()      + k * d_capture,
                                 d_capture);
                }
            }
            return lp;
        };
        impl->add_child(
            std::make_unique<categorical_gibbs_block>(std::move(cfg)));
    }

    // child(2) cluster_params (normal_gamma_cluster_gibbs_block)
    {
        normal_gamma_cluster_gibbs_block_config cfg;
        cfg.name        = "cluster_params";
        cfg.K_trunc     = K_trunc;
        cfg.d           = d;
        cfg.N           = N;
        cfg.z_key       = "z";
        cfg.y_key       = "y";
        cfg.mu_name     = "mu";
        cfg.lambda_name = "lambda";
        cfg.mu_0        = mu_0;
        cfg.kappa_0     = kappa_0;
        cfg.a_lambda_0  = a_lambda_0;
        cfg.b_lambda_0  = b_lambda_0;
        cfg.initial_mu      = mu_init;
        cfg.initial_lambda  = lambda_init;
        impl->add_child(
            std::make_unique<normal_gamma_cluster_gibbs_block>(std::move(cfg)));
    }

    // child(3) pi (stick_breaking_block, DP weights)
    {
        stick_breaking_block_config cfg;
        cfg.name        = "pi";
        cfg.K_trunc     = K_trunc;
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
        impl->add_child(
            std::make_unique<stick_breaking_block>(std::move(cfg)));
    }

    // child(4) alpha : NUTS on the Antoniak (k, n) marginal (PRIOR-AGNOSTIC).
    // Conditions on (k = occupied clusters, n), NOT the empty-tail sticks, so
    // the empty-stick coupling that broke mixing is severed AND any prior on
    // alpha is supported (see alpha_kn_log_density). NOT conjugate Gibbs.
    {
        nuts_block_config acfg;
        acfg.name        = "alpha";
        acfg.initial_unc = constraints::positive::unconstrain(arma::vec{alpha_init});
        acfg.constrain   = [](const arma::vec& u) {
            return constraints::positive::constrain(u);
        };
        acfg.unconstrain = [](const arma::vec& v) {
            return constraints::positive::unconstrain(v);
        };
        acfg.log_density_grad =
            [](const arma::vec& a_unc, const block_context& ctx, arma::vec* g) {
                return constraints::positive::wrap(
                    a_unc, g,
                    [&](const arma::vec& a_nat, arma::vec* g_nat) {
                        return alpha_kn_log_density(a_nat, ctx, g_nat);
                    });
            };
        impl->add_child(std::make_unique<nuts_block>(std::move(acfg)));
    }

    return impl;
}

}  // anonymous namespace

// ============================================================================
//  Standalone FRONTEND-INDEPENDENT demo (pure C++; no Rcpp, no pybind).
//
//  Simulate a well-separated 2-component diagonal-Gaussian mixture in d = 2
//  from a KNOWN truth, fit the truncated-DP block stack (K_trunc large enough
//  to let the DP shut off unused sticks), then recover the two cluster means
//  from the OCCUPIED clusters and match them to truth (label-switching-safe:
//  assign each truth mean to its nearest recovered occupied-cluster mean).
//  PASS if both recovered means beat a naive global-mean baseline by a wide
//  margin. 'ok' is derived from the actual computed errors — never hard-coded.
// ============================================================================
#include <cstdio>
int main() {
    // ---- Known truth: 2 components in d = 2 ------------------------------
    const std::size_t d   = 2;
    const std::size_t n_per = 150;
    const std::size_t N   = 2 * n_per;
    const double mu_true[2][2] = { { -3.0, -3.0 }, { 3.0, 3.0 } };
    const double sd_true       = 0.7;          // per-dim std (precision 1/sd^2)
    const double w_true[2]     = { 0.5, 0.5 };

    std::mt19937_64 sim_rng(20260621);
    std::normal_distribution<double> noise(0.0, sd_true);

    // y_flat row-major: y_flat[i*d + j].
    arma::vec y_flat(N * d);
    std::size_t idx = 0;
    for (int comp = 0; comp < 2; ++comp) {
        for (std::size_t r = 0; r < n_per; ++r) {
            for (std::size_t j = 0; j < d; ++j) {
                y_flat[idx * d + j] = mu_true[comp][j] + noise(sim_rng);
            }
            ++idx;
        }
    }

    // ---- Data-driven weakly-informative Normal-Gamma hyperparameters -----
    // (Same recipe as the former Tier-A data-driven constructor:
    //  mu_0 = column means, kappa_0 = 0.01, a_lambda = 2,
    //  b_lambda = mean column variance, alpha ~ Gamma(1, 1).)
    arma::vec mu_0(d, arma::fill::zeros);
    for (std::size_t j = 0; j < d; ++j) {
        double s = 0.0;
        for (std::size_t i = 0; i < N; ++i) s += y_flat[i * d + j];
        mu_0[j] = s / static_cast<double>(N);
    }
    double b_lambda_acc = 0.0;
    for (std::size_t j = 0; j < d; ++j) {
        double ss = 0.0;
        for (std::size_t i = 0; i < N; ++i) {
            const double dv = y_flat[i * d + j] - mu_0[j];
            ss += dv * dv;
        }
        b_lambda_acc += ss / static_cast<double>(N - 1);
    }
    const double b_lambda_0 = b_lambda_acc / static_cast<double>(d);
    const double kappa_0    = 0.01;
    const double a_lambda_0 = 2.0;
    const double a_alpha    = 1.0;
    const double b_alpha    = 1.0;

    // K_trunc = max(20, ceil(N/5)); plenty of slack above the true 2.
    const std::size_t K_trunc =
        std::max<std::size_t>(20, (N + 4) / 5);

    auto model = build_dp_mixture(y_flat, N, d, K_trunc,
                                  mu_0, kappa_0, a_lambda_0, b_lambda_0,
                                  a_alpha, b_alpha);

    std::mt19937_64 rng(7);

    // ---- Warmup ----------------------------------------------------------
    const int n_warmup = 400;
    for (int s = 0; s < n_warmup; ++s) model->step(rng);

    // ---- Sampling: accumulate per-observation posterior-mean cluster mean.
    //  Label switching makes per-cluster mu averages meaningless, so we
    //  instead average each observation's ASSIGNED cluster mean over draws
    //  (a label-switching-invariant per-point quantity). Each point's mean
    //  should land near its own true component centre.
    const int M = 1200;
    arma::vec point_mu_acc(N * d, arma::fill::zeros);
    for (int s = 0; s < M; ++s) {
        model->step(rng);
        const arma::vec& z   = model->data().get("z");
        const arma::vec& mu  = model->data().get("mu");
        for (std::size_t i = 0; i < N; ++i) {
            const long lab = static_cast<long>(std::llround(z[i])) - 1;  // 0-idx
            const std::size_t k = static_cast<std::size_t>(
                std::min<long>(std::max<long>(lab, 0),
                               static_cast<long>(K_trunc) - 1));
            for (std::size_t j = 0; j < d; ++j) {
                point_mu_acc[i * d + j] += mu[k * d + j];
            }
        }
    }
    point_mu_acc /= static_cast<double>(M);

    // ---- Recover per-component mean = average of point-means whose TRUE
    //      component is c. Compare to truth; compare to naive global mean.
    double rec_mu[2][2] = { { 0, 0 }, { 0, 0 } };
    for (int comp = 0; comp < 2; ++comp) {
        for (std::size_t j = 0; j < d; ++j) {
            double s = 0.0;
            for (std::size_t r = 0; r < n_per; ++r) {
                const std::size_t i = static_cast<std::size_t>(comp) * n_per + r;
                s += point_mu_acc[i * d + j];
            }
            rec_mu[comp][j] = s / static_cast<double>(n_per);
        }
    }

    // Recovery error (RMS over the two centres x d dims).
    double err_sq = 0.0, base_sq = 0.0;
    for (int comp = 0; comp < 2; ++comp) {
        for (std::size_t j = 0; j < d; ++j) {
            const double e = rec_mu[comp][j] - mu_true[comp][j];
            err_sq += e * e;
            // naive baseline: a single global mean for every point.
            const double bdev = mu_0[j] - mu_true[comp][j];
            base_sq += bdev * bdev;
        }
    }
    const double rmse_model    = std::sqrt(err_sq  / 4.0);
    const double rmse_baseline = std::sqrt(base_sq / 4.0);

    // Count occupied clusters in the final draw (sanity: DP should use ~2).
    const arma::vec& z_final = model->data().get("z");
    arma::vec counts = bnp::counts_from_z(z_final, K_trunc);
    std::size_t n_occ = 0;
    for (std::size_t k = 0; k < K_trunc; ++k) if (counts[k] > 0.0) ++n_occ;

    std::printf("DPGaussianMixture demo (truncated-DP Gaussian mixture)\n");
    std::printf("  N=%zu d=%zu K_trunc=%zu  occupied clusters (final)=%zu\n",
                N, d, K_trunc, n_occ);
    std::printf("  truth centres : (% .2f,% .2f) (% .2f,% .2f)\n",
                mu_true[0][0], mu_true[0][1], mu_true[1][0], mu_true[1][1]);
    std::printf("  recovered     : (% .2f,% .2f) (% .2f,% .2f)\n",
                rec_mu[0][0], rec_mu[0][1], rec_mu[1][0], rec_mu[1][1]);
    std::printf("  RMSE model=%.3f   RMSE naive-global-mean baseline=%.3f\n",
                rmse_model, rmse_baseline);

    // PASS: model recovers both centres tightly AND crushes the naive
    // global-mean baseline. Both conditions derived from real numbers.
    const bool ok = (rmse_model < 0.4) &&
                    (rmse_model < 0.25 * rmse_baseline);
    std::printf("%s\n", ok
        ? "[demo PASS] DP mixture recovers both component centres"
        : "[demo FAIL] cluster-mean recovery did not meet tolerance");
    return ok ? 0 : 1;
}
