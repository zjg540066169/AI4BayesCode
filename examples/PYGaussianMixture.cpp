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
//  Sampling discount jointly with alpha would put it in a `joint_nuts_block`
//  with an INTERVAL constraint on discount in (0, 1) (the joint_constraint
//  table includes INTERVAL). For this v0 reference we keep it simpler. Two
//  options:
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
// ============================================================================

// Frontend-INDEPENDENT example: pure standalone C++ (NO Rcpp, NO pybind).
// Compile + run directly: it has an int main() that simulates 3-cluster
// data, drives the Pitman-Yor mixture composite directly, and checks
// cluster recovery (label-permutation tolerant). No R / Python binding is
// built or required.

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
#include "AI4BayesCode/constraints.hpp"

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
#include <stdexcept>
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
//  build_py_mixture: wire the Pitman-Yor Gaussian-mixture composite directly.
//
//  This replaces the (formerly Rcpp-typed) Tier-A wrapper class. It builds
//  the exact same model — same priors, same block configs, same PY a_fn/b_fn,
//  same alpha log-density — but drives the composite_block directly so the
//  example is frontend-independent. Data-driven Normal-Gamma hypers (col
//  means for mu_0, kappa_0=0.01, a_lambda=2, b_lambda=mean col var; the same
//  weakly-informative defaults the data-driven ctor used) are computed here.
// ============================================================================

namespace {

std::unique_ptr<composite_block>
build_py_mixture(const arma::mat& y,      // N x d
                 std::size_t       K_trunc,
                 double            discount,
                 double            a_alpha,
                 double            b_alpha) {
    const std::size_t N = y.n_rows;
    const std::size_t d = y.n_cols;
    if (N < 2)       throw std::runtime_error("PYGaussianMixture: N must be >= 2");
    if (d < 1)       throw std::runtime_error("PYGaussianMixture: d must be >= 1");
    if (K_trunc < 2) throw std::runtime_error("PYGaussianMixture: K_trunc must be >= 2");
    if (!(discount >= 0.0 && discount < 1.0))
        throw std::runtime_error("discount must be in [0, 1)");
    if (!(a_alpha > 0.0)) throw std::runtime_error("a_alpha must be > 0");
    if (!(b_alpha > 0.0)) throw std::runtime_error("b_alpha must be > 0");

    // ---- Data-driven weakly-informative Normal-Gamma hypers (matches the
    //      data-driven ctor the Rcpp wrapper used). ------------------------
    arma::vec mu0_arma(d);
    for (std::size_t j = 0; j < d; ++j) mu0_arma[j] = arma::mean(y.col(j));
    const double kappa_0    = 0.01;
    const double a_lambda_0 = 2.0;
    double b_acc = 0.0; std::size_t b_used = 0;
    for (std::size_t j = 0; j < d; ++j) {
        const double v = arma::var(y.col(j));      // unbiased (n-1)
        if (std::isfinite(v) && v > 0.0) { b_acc += v; ++b_used; }
    }
    const double b_lambda_0 = (b_used > 0) ? b_acc / static_cast<double>(b_used) : 1.0;

    auto impl = std::make_unique<composite_block>("PYGaussianMixture");

    // ---- flatten y row-major into the shared "y" key --------------------
    arma::vec y_flat(N * d);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < d; ++j)
            y_flat[i * d + j] = y(i, j);
    impl->data().set("y", y_flat);

    impl->data().set("mu_0", mu0_arma);
    impl->data().set("kappa_0",     arma::vec{kappa_0});
    impl->data().set("a_lambda_0",  arma::vec{a_lambda_0});
    impl->data().set("b_lambda_0",  arma::vec{b_lambda_0});
    impl->data().set("a_alpha",     arma::vec{a_alpha});
    impl->data().set("b_alpha",     arma::vec{b_alpha});
    impl->data().set("discount",    arma::vec{discount});

    // ---- Initial state --------------------------------------------------
    arma::vec z_init(N);
    for (std::size_t i = 0; i < N; ++i)
        z_init[i] = static_cast<double>((i % K_trunc) + 1);
    impl->data().set("z", z_init);

    arma::vec pi_init(K_trunc, arma::fill::value(1.0 / static_cast<double>(K_trunc)));
    impl->data().set("pi", pi_init);
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

    arma::vec mu_init(K_trunc * d, arma::fill::zeros);
    arma::vec lambda_init(K_trunc * d,
                          arma::fill::value(a_lambda_0 / b_lambda_0));
    for (std::size_t k = 0; k < K_trunc; ++k) {
        const std::size_t i_anchor = (k * N) / K_trunc;
        for (std::size_t j = 0; j < d; ++j)
            mu_init[k * d + j] = y(i_anchor, j);
    }
    impl->data().set("mu", mu_init);
    impl->data().set("lambda", lambda_init);

    const double alpha_init = a_alpha / b_alpha;
    impl->data().set("alpha", arma::vec{alpha_init});

    impl->data().set("cluster_counts", arma::vec(K_trunc, arma::fill::zeros));
    impl->data().register_refresher("cluster_counts",
        [K = K_trunc](const AI4BayesCode::shared_data_t& d) -> arma::vec {
            const arma::vec& z = d.get("z");
            return bnp::counts_from_z(z, K);
        });

    impl->data().declare_dependencies("z",
        {"y", "pi", "mu", "lambda"});
    impl->data().declare_dependencies("cluster_params",
        {"z", "y"});
    impl->data().declare_dependencies("pi",
        {"cluster_counts", "alpha", "discount"});
    impl->data().declare_dependencies("alpha",
        {"cluster_counts", "a_alpha", "b_alpha", "discount"});
    impl->data().declare_invalidates("z", {"cluster_counts"});

    // (No declare_data_input here — y is an observed terminal, not a
    // replaceable covariate. The y_rep refresher reads pi/mu/lambda, NOT y.)
    impl->data().declare_predict_edges("pi",     {"y_rep"});
    impl->data().declare_predict_edges("mu",     {"y_rep"});
    impl->data().declare_predict_edges("lambda", {"y_rep"});

    // ---- Generative-DAG context (VIZ-ONLY). Pitman-Yor stick-breaking +
    //      Normal-Gamma cluster prior. -------------------------------------
    impl->data().declare_context_edges("a_alpha",     {"alpha"});
    impl->data().declare_context_edges("b_alpha",     {"alpha"});
    impl->data().declare_context_edges("alpha",       {"stick_V"});
    impl->data().declare_context_edges("discount",    {"stick_V"});
    impl->data().declare_context_edges("stick_V",     {"pi"});
    impl->data().declare_context_edges("mu_0",        {"mu"});
    impl->data().declare_context_edges("kappa_0",     {"mu"});
    impl->data().declare_context_edges("a_lambda_0",  {"lambda"});
    impl->data().declare_context_edges("b_lambda_0",  {"lambda"});

    impl->data().set("y_rep", arma::vec(N * d, arma::fill::zeros));
    impl->data().register_stochastic_refresher("y_rep",
        [N, d, K = K_trunc](
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

    impl->data().refresh_all();

    // ---- children -------------------------------------------------------
    // child(0) z
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
        impl->add_child(
            std::make_unique<categorical_gibbs_block>(std::move(cfg)));
    }

    // child(1) cluster_params
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
        cfg.mu_0        = mu0_arma;
        cfg.kappa_0     = kappa_0;
        cfg.a_lambda_0  = a_lambda_0;
        cfg.b_lambda_0  = b_lambda_0;
        cfg.initial_mu      = mu_init;
        cfg.initial_lambda  = lambda_init;
        impl->add_child(
            std::make_unique<normal_gamma_cluster_gibbs_block>(
                std::move(cfg)));
    }

    // child(2) pi  (PY a_fn / b_fn)
    {
        stick_breaking_block_config cfg;
        cfg.name        = "pi";
        cfg.K_trunc     = K_trunc;
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
            for (std::size_t j = k + 1; j < counts.n_elem; ++j)
                tail += counts[j];
            return a + static_cast<double>(k + 1) * dscount + tail;
        };
        cfg.initial_pi = pi_init;
        impl->add_child(
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
        impl->add_child(std::make_unique<nuts_block>(std::move(cfg)));
    }

    return impl;
}

}  // anonymous namespace

// ============================================================================
//  Standalone FRONTEND-INDEPENDENT demo (pure C++; no Rcpp, no pybind).
//
//  Simulates a well-separated 3-component Gaussian mixture in d=2, fits the
//  Pitman-Yor mixture composite at discount=0 (the identified DP regime), and
//  checks cluster recovery. Because mixture labels are permutation-symmetric,
//  the check is LABEL-PERMUTATION TOLERANT: for each true cluster centre we
//  require that SOME fitted component carrying meaningful posterior weight
//  sits close to it (posterior-mean of mu/pi over kept draws). This is the
//  honest analogue of "recover the truth" for a label-switching model — we do
//  NOT compare component k to component k.
// ============================================================================
#include <cstdio>
int main() {
    // ---- simulate from a KNOWN 3-cluster truth --------------------------
    const std::size_t d     = 2;
    const std::size_t per   = 120;          // points per cluster
    const std::size_t N     = 3 * per;
    const double      sd    = 0.6;          // tight, well-separated

    // True centres (well separated relative to sd):
    const double centres[3][2] = {
        { -6.0, -6.0 },
        {  6.0,  0.0 },
        {  0.0,  6.0 }
    };

    std::mt19937_64 sim_rng(20240621ULL);
    std::normal_distribution<double> noise(0.0, sd);
    arma::mat y(N, d);
    {
        std::size_t row = 0;
        for (std::size_t c = 0; c < 3; ++c)
            for (std::size_t i = 0; i < per; ++i, ++row)
                for (std::size_t j = 0; j < d; ++j)
                    y(row, j) = centres[c][j] + noise(sim_rng);
    }

    // ---- build + fit the PY mixture (discount = 0 -> identified DP) ------
    const std::size_t K_trunc  = 10;        // over-truncated; PY/DP prunes
    const double      discount = 0.0;
    auto model = build_py_mixture(y, K_trunc, discount,
                                  /*a_alpha=*/1.0, /*b_alpha=*/1.0);

    std::mt19937_64 mcmc_rng(7);

    // warmup
    const int n_warm = 1500;
    for (int s = 0; s < n_warm; ++s) model->step(mcmc_rng);

    // sampling: accumulate posterior-mean of pi and mu over kept draws.
    const int M = 2000;
    arma::vec pi_bar(K_trunc, arma::fill::zeros);
    arma::mat mu_bar(K_trunc, d, arma::fill::zeros);
    double alpha_bar = 0.0;
    for (int s = 0; s < M; ++s) {
        model->step(mcmc_rng);
        const arma::vec& pi  = model->data().get("pi");
        const arma::vec& mu  = model->data().get("mu");
        for (std::size_t k = 0; k < K_trunc; ++k) {
            pi_bar[k] += pi[k];
            for (std::size_t j = 0; j < d; ++j)
                mu_bar(k, j) += mu[k * d + j];
        }
        alpha_bar += model->data().get("alpha")[0];
    }
    pi_bar    /= static_cast<double>(M);
    mu_bar    /= static_cast<double>(M);
    alpha_bar /= static_cast<double>(M);

    // ---- recovery check (label-permutation tolerant) -------------------
    // A component is "active" if its mean posterior weight exceeds a floor.
    const double weight_floor = 0.05;
    std::size_t n_active = 0;
    for (std::size_t k = 0; k < K_trunc; ++k)
        if (pi_bar[k] > weight_floor) ++n_active;

    // For each TRUE centre, find the nearest active fitted component and
    // record (a) the distance and (b) that it carries meaningful weight.
    bool all_centres_matched = true;
    double worst_dist = 0.0;
    std::printf("PYGaussianMixture demo (PY discount=%.1f, K_trunc=%zu, "
                "N=%zu, d=%zu)\n", discount, K_trunc, N, d);
    std::printf("  posterior-mean alpha = %.3f, n_active(pi>%.2f) = %zu "
                "(truth K=3)\n", alpha_bar, weight_floor, n_active);
    for (std::size_t c = 0; c < 3; ++c) {
        double best = std::numeric_limits<double>::infinity();
        std::size_t best_k = 0;
        for (std::size_t k = 0; k < K_trunc; ++k) {
            if (pi_bar[k] <= weight_floor) continue;
            double dist2 = 0.0;
            for (std::size_t j = 0; j < d; ++j) {
                const double dv = mu_bar(k, j) - centres[c][j];
                dist2 += dv * dv;
            }
            const double dist = std::sqrt(dist2);
            if (dist < best) { best = dist; best_k = k; }
        }
        if (best > worst_dist) worst_dist = best;
        const bool matched = std::isfinite(best) && best < 0.75;
        if (!matched) all_centres_matched = false;
        std::printf("  true centre (%.1f, %.1f): nearest active comp %zu at "
                    "(%.2f, %.2f), dist=%.3f, weight=%.3f  [%s]\n",
                    centres[c][0], centres[c][1], best_k,
                    std::isfinite(best) ? mu_bar(best_k, 0) : 0.0,
                    std::isfinite(best) ? mu_bar(best_k, 1) : 0.0,
                    best,
                    std::isfinite(best) ? pi_bar[best_k] : 0.0,
                    matched ? "OK" : "MISS");
    }

    // Pass criterion: all 3 true centres matched by a distinct, meaningful
    // component, modal active count is 3 (not over/under clustered), and the
    // worst centre-recovery distance is small relative to sd.
    const bool ok = all_centres_matched && (n_active == 3) && (worst_dist < 0.75);
    std::printf("%s\n",
                ok ? "[demo PASS] PY mixture recovers the 3 true clusters "
                     "(label-permutation tolerant)"
                   : "[demo FAIL] cluster recovery did not meet criterion");
    return ok ? 0 : 1;
}
