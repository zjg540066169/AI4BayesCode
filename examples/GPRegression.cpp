// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later
// (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
// GPRegression.cpp
//
// REFERENCE TEMPLATE for generic Gaussian Process regression with
// Elliptical Slice Sampling on latent f + NUTS on kernel hyperparameters
// + NUTS on noise sigma. Uses the vendored libgp kernel subsystem for
// kernel evaluation (stateless-per-pair, stateful-per-hyperparam).
//
// MODEL
// -----
//   y_i | f(x_i), sigma  ~ Normal(f(x_i), sigma^2),   i = 1..N
//   f                     ~ GP(0, K_amplitude_lengthscale(X, X))
//   amplitude             ~ half-Normal(0, sd(y))     (weakly informative)
//   lengthscale           ~ InverseGamma(5, s_med)    (s_med = median pairwise dist)
//   sigma (noise)         ~ Jeffreys  p(sigma) oc 1/sigma
//
// The GP prior K is built via libgp's CovSEiso (can be swapped for any
// other libgp kernel: SE-ARD, Matern32/52, Periodic, etc.). libgp
// kernel object is held as a member of this wrapper and configured
// (via set_loghyper) inside the K_matrix shared_data refresher.
//
// BLOCKS (Gibbs sweep order: hyperparams FIRST, f LAST)
// ------
//   log_amplitude    : nuts_block with positive constraint + half-Normal prior
//   log_lengthscale  : nuts_block with positive constraint + InverseGamma prior
//   log_sigma        : nuts_block with positive constraint + Jeffreys prior
//   f                : elliptical_slice_sampling_block (reads L_prior from ctx
//                      -- the GP PRIOR covariance Cholesky chol(K), NOT the
//                      noisy chol(K+sigma^2 I))
//
// SHARED_DATA DAG
// ---------------
//   y (fixed data), X (fixed data, flat column-major)
//   amplitude, lengthscale, sigma: sampled parameters
//   K_matrix  (derived, N^2 flat): refresher reads X + amplitude + lengthscale
//                                   + libgp cf; invalidated by amp / ell updates
//   L_prior   (derived, N^2 flat): refresher reads K_matrix; chol(K + jitter)
//                                   = GP prior covariance, consumed by ESS;
//                                   invalidated by amp / ell (NOT sigma)
//   L_chol    (derived, N^2 flat): refresher reads K_matrix + sigma;
//                                   chol(K + sigma^2 I) used by predict_at;
//                                   invalidated by amp / ell / sigma updates
//   f         : sampled by ESS block
//   prob ... (not applicable for Gaussian-likelihood regression)
//   y_rep     : stochastic refresher at predict time
//
// PREDICT DAG
// -----------
//   X     -> K_matrix
//   K_matrix + sigma -> L_chol
//   L_chol + f -> (no simple refresher; predict_at uses the BartNoise
//                   pattern: wrapper calls kernel routines directly to
//                   build cross-kernel K_star_X and compute posterior
//                   mean/cov at X_new; injects f_star + sigma into a
//                   scratch context, then y_rep stochastic refresher
//                   fires)
//
// LICENSE WARNING
// ---------------
// libgp_kernels is BSD-3; GPL-compatible. AI4BayesCode as a whole is
// GPL-2.0-or-later; the combined work stays GPL-2.0-or-later. The
// BSD-3 attribution is preserved in AI4BayesCode/libgp_kernels/COPYING and
// listed in THIRD_PARTY_LICENSES.md.
// ============================================================================

// Frontend-INDEPENDENT example: pure standalone C++ (NO Rcpp, NO pybind).
// Compile + run directly: it has an int main() that simulates GP-regression
// data, fits the latent f + kernel hyperparameters + noise, and checks
// recovery. No R / Python binding is built or required.

#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif

#include <armadillo>

#include "AI4BayesCode/block_sampler.hpp"
#include "AI4BayesCode/shared_data.hpp"
#include "AI4BayesCode/nuts_block.hpp"
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/constraints.hpp"
#include "AI4BayesCode/elliptical_slice_sampling_block.hpp"

// Vendored libgp kernel subsystem (BSD-3). Unity header includes both
// headers and .cc sources so the single translation unit picks everything up.
#include "libgp_kernels_unity.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

using AI4BayesCode::block_context;
using AI4BayesCode::composite_block;
using AI4BayesCode::nuts_block;
using AI4BayesCode::nuts_block_config;
using AI4BayesCode::elliptical_slice_sampling_block;
using AI4BayesCode::elliptical_slice_sampling_block_config;
namespace constraints = AI4BayesCode::constraints;

// ============================================================================
// Natural-scale log-densities for the three hyperparameter blocks.
//
// Per codegen.md Jacobian discipline: user lambdas on natural scale
// contain ONLY the posterior's natural-scale log-density. The wrap
// function in constraints::positive::wrap automatically adds the
// log|dsigma/dlog sigma| = log sigma Jacobian term. So NO log sigma
// in these lambdas.
// ============================================================================

namespace {

// ------- amplitude: half-Normal(0, s) prior --------
// Natural scale sigma_f > 0.
//   log p(sigma_f) = -sigma_f^2 / (2 s^2)   (const dropped)
//   d/dsigma_f     = -sigma_f / s^2
double amp_natural_log_density(const arma::vec& amp_nat,
                               const block_context& ctx,
                               arma::vec* grad_nat) {
    const double a = amp_nat[0];
    const double s = ctx.at("amp_prior_sd")[0];
    const double s2 = s * s;
    if (grad_nat) {
        grad_nat->set_size(1);
        (*grad_nat)[0] = -a / s2;
    }
    return -0.5 * a * a / s2;
}

// ------- lengthscale: InverseGamma(shape, scale) prior --------
// Natural scale ell > 0.
//   log p(ell) = -(shape + 1) log(ell) - scale / ell
//   d/dell     = -(shape + 1) / ell + scale / ell^2
double ell_natural_log_density(const arma::vec& ell_nat,
                               const block_context& ctx,
                               arma::vec* grad_nat) {
    const double ell = ell_nat[0];
    const double shape = ctx.at("ell_prior_shape")[0];
    const double scale = ctx.at("ell_prior_scale")[0];
    if (grad_nat) {
        grad_nat->set_size(1);
        (*grad_nat)[0] = -(shape + 1.0) / ell + scale / (ell * ell);
    }
    return -(shape + 1.0) * std::log(ell) - scale / ell;
}

// ------- sigma (noise): Gaussian likelihood + Jeffreys prior --------
// Natural scale sigma > 0.
// Joint:
//   log p(y|f,sigma) + log p(sigma)
//     = -N log(sigma) - 0.5 * SSE / sigma^2  +  (-log sigma)
//     = -(N+1) log(sigma) - 0.5 * SSE / sigma^2
//   d/dsigma = -(N+1)/sigma + SSE / sigma^3
double sigma_natural_log_density(const arma::vec& sigma_nat,
                                 const block_context& ctx,
                                 arma::vec* grad_nat) {
    const double sig = sigma_nat[0];
    const arma::vec& y = ctx.at("y");
    const arma::vec& f = ctx.at("f");
    const double N = static_cast<double>(y.n_elem);
    double sse = 0.0;
    for (std::size_t i = 0; i < y.n_elem; ++i) {
        double d = y[i] - f[i];
        sse += d * d;
    }
    const double s2 = sig * sig;
    if (grad_nat) {
        grad_nat->set_size(1);
        (*grad_nat)[0] = -(N + 1.0) / sig + sse / (sig * s2);
    }
    return -(N + 1.0) * std::log(sig) - 0.5 * sse / s2;
}

// Pairwise distance helper for InvGamma scale heuristic.
double median_pairwise_distance(const arma::mat& X) {
    const std::size_t N = X.n_rows;
    if (N < 2) return 1.0;
    std::vector<double> dists;
    dists.reserve(N * (N - 1) / 2);
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = i + 1; j < N; ++j) {
            const double d = arma::norm(X.row(i) - X.row(j), 2);
            dists.push_back(d);
        }
    }
    std::sort(dists.begin(), dists.end());
    const std::size_t m = dists.size() / 2;
    return (dists.size() % 2 == 1) ? dists[m]
                                   : 0.5 * (dists[m - 1] + dists[m]);
}

}  // anonymous namespace

// ============================================================================
// User-facing model class (frontend-independent, neutral-typed).
// ============================================================================

class GPRegression {
public:
    GPRegression(const arma::mat& X,
                 const arma::vec& y,
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
          impl_(std::make_unique<composite_block>("GPRegression")),
          keep_history_(keep_history)
    {
        if (X.n_rows != y.n_elem) {
            throw std::runtime_error(
                "GPRegression: X and y must have matching row counts");
        }
        if (X.n_rows < 2) {
            throw std::runtime_error("GPRegression: N must be >= 2");
        }
        const std::size_t N = X.n_rows;
        const std::size_t p = X.n_cols;
        N_ = N;
        p_ = p;

        // ---- Install fixed data -----------------------------------------
        arma::vec y_arma = y;
        impl_->data().set("y", y_arma);

        // Store X as arma::mat (for heuristics + predict) AND cache per-row
        // as Eigen::VectorXd for libgp kernel evaluation.
        X_arma_ = arma::mat(N, p);
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = 0; j < p; ++j) X_arma_(i, j) = X(i, j);
        }
        X_rows_.reserve(N);
        for (std::size_t i = 0; i < N; ++i) {
            Eigen::VectorXd row(p);
            for (std::size_t j = 0; j < p; ++j) row[j] = X(i, j);
            X_rows_.push_back(row);
        }
        // X flat for shared_data (column-major)
        arma::vec x_flat = arma::vectorise(X_arma_);
        impl_->data().set("X", x_flat);

        // ---- Prior hyperparameters -------------------------------------
        // amplitude half-Normal(0, sd(y))
        const double sd_y = arma::stddev(y_arma);
        impl_->data().set("amp_prior_sd",
            arma::vec{std::max(sd_y, 0.1)});

        // lengthscale InverseGamma(5, median_pair_dist)
        const double ell_scale = median_pairwise_distance(X_arma_);
        impl_->data().set("ell_prior_shape", arma::vec{5.0});
        impl_->data().set("ell_prior_scale", arma::vec{ell_scale});

        // Initial hyperparam values (sensible data-driven defaults)
        const double amp_init   = std::max(sd_y, 0.1);
        const double ell_init   = ell_scale / 5.0;  // InvGamma posterior-ish mean
        const double sigma_init = 0.3 * sd_y;

        impl_->data().set("amplitude",   arma::vec{amp_init});
        impl_->data().set("lengthscale", arma::vec{ell_init});
        impl_->data().set("sigma",       arma::vec{sigma_init});

        // ---- libgp kernel object ----------------------------------------
        // CovSEiso: params are [log(ell), log(sf)] in that order.
        cf_ = std::make_unique<libgp::CovSEiso>();
        cf_->init(static_cast<int>(p));

        // Initial f seed = zeros
        impl_->data().set("f", arma::vec(N, arma::fill::zeros));

        // Seed K_matrix + L_chol slots (refreshers will populate)
        impl_->data().set("K_matrix",
            arma::vec(N * N, arma::fill::zeros));
        impl_->data().set("L_chol",
            arma::vec(N * N, arma::fill::zeros));
        // L_prior = chol(K + jitter): the GP PRIOR-covariance Cholesky
        // consumed by the elliptical slice sampling block. ESS draws its
        // prior proposal nu = L_prior z ~ N(0, K). The noise sigma^2 must
        // NOT be folded in here (it enters only via the ESS likelihood);
        // doing so over-shrinks f and biases the conditional sigma low.
        // (L_chol = chol(K + sigma^2 I) is kept separately for predict_at.)
        impl_->data().set("L_prior",
            arma::vec(N * N, arma::fill::zeros));

        // ---- Declare Gibbs dependencies -----------------------------
        impl_->data().declare_dependencies("amplitude",
            {"amp_prior_sd"});
        impl_->data().declare_dependencies("lengthscale",
            {"ell_prior_shape", "ell_prior_scale"});
        impl_->data().declare_dependencies("sigma",
            {"y", "f"});
        impl_->data().declare_dependencies("f",
            {"L_prior", "y", "sigma"});

        // ---- Invalidation chain -----------------------------------------
        // amplitude / lengthscale update -> rebuild K_matrix then L_chol
        impl_->data().declare_invalidates("amplitude",
            {"K_matrix", "L_prior", "L_chol"});
        impl_->data().declare_invalidates("lengthscale",
            {"K_matrix", "L_prior", "L_chol"});
        // sigma update -> only L_chol (noise on diagonal)
        impl_->data().declare_invalidates("sigma", {"L_chol"});

        // ---- K_matrix refresher (uses libgp) -----------------------
        const libgp::CovSEiso* cf_ptr = cf_.get();
        const std::vector<Eigen::VectorXd>* X_rows_ptr = &X_rows_;
        impl_->data().register_refresher("K_matrix",
            [cf_ptr, X_rows_ptr, N](
                const AI4BayesCode::shared_data_t& d) -> arma::vec {
                const double amp = d.get("amplitude")[0];
                const double ell = d.get("lengthscale")[0];
                // libgp CovSEiso expects [log(ell), log(sf)].
                Eigen::VectorXd hyper(2);
                hyper[0] = std::log(std::max(ell, 1e-10));
                hyper[1] = std::log(std::max(amp, 1e-10));
                auto* cf_mut = const_cast<libgp::CovSEiso*>(cf_ptr);
                cf_mut->set_loghyper(hyper);

                arma::mat K(N, N);
                for (std::size_t i = 0; i < N; ++i) {
                    K(i, i) = cf_mut->get((*X_rows_ptr)[i],
                                           (*X_rows_ptr)[i]);
                    for (std::size_t j = i + 1; j < N; ++j) {
                        double k = cf_mut->get((*X_rows_ptr)[i],
                                                (*X_rows_ptr)[j]);
                        K(i, j) = k;
                        K(j, i) = k;
                    }
                }
                return arma::vectorise(K);
            });

        // ---- L_chol refresher: chol(K + sigma^2 I + jitter I) --------
        const double jitter = 1e-6;
        impl_->data().register_refresher("L_chol",
            [N, jitter](const AI4BayesCode::shared_data_t& d) -> arma::vec {
                const arma::vec& K_flat = d.get("K_matrix");
                const double sig = d.get("sigma")[0];
                arma::mat K(const_cast<double*>(K_flat.memptr()), N, N,
                             /*copy_aux_mem=*/false, /*strict=*/true);
                arma::mat M = K;
                M.diag() += sig * sig + jitter;
                arma::mat L;
                if (!arma::chol(L, M, "lower")) {
                    // Should be extremely rare with positive jitter;
                    // retry with larger jitter.
                    M.diag() += 1e-3;
                    arma::chol(L, M, "lower");
                }
                return arma::vectorise(L);
            });

        // ---- L_prior refresher: chol(K + jitter I) -- the GP PRIOR
        //      covariance Cholesky (NO sigma^2). Consumed by the ESS block
        //      as its prior-proposal covariance nu = L_prior z ~ N(0, K).
        impl_->data().register_refresher("L_prior",
            [N, jitter](const AI4BayesCode::shared_data_t& d) -> arma::vec {
                const arma::vec& K_flat = d.get("K_matrix");
                arma::mat K(const_cast<double*>(K_flat.memptr()), N, N,
                             /*copy_aux_mem=*/false, /*strict=*/true);
                arma::mat M = K;
                M.diag() += jitter;
                arma::mat L;
                if (!arma::chol(L, M, "lower")) {
                    // Smooth SE kernel on a dense grid can be near-singular;
                    // bump jitter (matches the L_chol fallback policy).
                    M.diag() += 1e-3;
                    arma::chol(L, M, "lower");
                }
                return arma::vectorise(L);
            });

        // Prime K and L so the first step sees consistent state
        impl_->data().refresh_all();

        // ---- Predict DAG (for predict_at) -----------------------------
        impl_->data().declare_data_input("X");
        impl_->data().declare_predict_edges("X", {"K_matrix"});
        impl_->data().declare_predict_edges("K_matrix", {"L_chol"});
        impl_->data().declare_predict_edges("f", {"y_rep"});
        impl_->data().declare_predict_edges("sigma", {"y_rep"});

        // ---- Generative-DAG context (VIZ-ONLY; predict_at BFS never
        //      reads context_edges_). The GP kernel K is built from the
        //      sampled hyperparameters (amplitude, lengthscale) — they
        //      are generative parents of K_matrix, analogous to the BART
        //      forest -> f_bart edge. Their priors: amplitude ~
        //      half-Normal(0, amp_prior_sd); lengthscale ~
        //      InverseGamma(ell_prior_shape, ell_prior_scale). f ~
        //      GP(0,K) so L_chol is f's generative parent. sigma ~
        //      Jeffreys (no slot). Drawn faded by ai4bayescode_plot_dag.
        impl_->data().declare_context_edges("amp_prior_sd",   {"amplitude"});
        impl_->data().declare_context_edges("ell_prior_shape",{"lengthscale"});
        impl_->data().declare_context_edges("ell_prior_scale",{"lengthscale"});
        impl_->data().declare_context_edges("amplitude",      {"K_matrix"});
        impl_->data().declare_context_edges("lengthscale",    {"K_matrix"});
        impl_->data().declare_context_edges("L_chol",         {"f"});

        impl_->data().set("y_rep", arma::vec(N, arma::fill::zeros));
        impl_->data().register_stochastic_refresher("y_rep",
            [](const AI4BayesCode::shared_data_t& d,
               std::mt19937_64& rng) {
                const arma::vec& f = d.get("f");
                const double sig = d.get("sigma")[0];
                std::normal_distribution<double> nd(0.0, 1.0);
                arma::vec out(f.n_elem);
                for (std::size_t i = 0; i < f.n_elem; ++i) {
                    out[i] = f[i] + sig * nd(rng);
                }
                return out;
            });

        // ---- Add blocks in Gibbs order ---------------------------------
        // Order: amplitude -> lengthscale -> sigma -> f
        // Reason: hyperparams update first, invalidates K/L, then f
        // samples against fresh K/L.

        {
            nuts_block_config cfg;
            cfg.name        = "amplitude";
            cfg.initial_unc = arma::vec{std::log(amp_init)};
            cfg.constrain   = constraints::positive::constrain;
            cfg.unconstrain = constraints::positive::unconstrain;
            cfg.log_density_grad =
                [](const arma::vec& t_unc, const block_context& ctx,
                   arma::vec* grad) {
                    return constraints::positive::wrap(t_unc, grad,
                        [&](const arma::vec& t_nat, arma::vec* g_nat) {
                            return amp_natural_log_density(t_nat, ctx, g_nat);
                        });
                };
            impl_->add_child(
                std::make_unique<nuts_block>(std::move(cfg)));
        }
        {
            nuts_block_config cfg;
            cfg.name        = "lengthscale";
            cfg.initial_unc = arma::vec{std::log(ell_init)};
            cfg.constrain   = constraints::positive::constrain;
            cfg.unconstrain = constraints::positive::unconstrain;
            cfg.log_density_grad =
                [](const arma::vec& t_unc, const block_context& ctx,
                   arma::vec* grad) {
                    return constraints::positive::wrap(t_unc, grad,
                        [&](const arma::vec& t_nat, arma::vec* g_nat) {
                            return ell_natural_log_density(t_nat, ctx, g_nat);
                        });
                };
            impl_->add_child(
                std::make_unique<nuts_block>(std::move(cfg)));
        }
        {
            nuts_block_config cfg;
            cfg.name        = "sigma";
            cfg.initial_unc = arma::vec{std::log(sigma_init)};
            cfg.constrain   = constraints::positive::constrain;
            cfg.unconstrain = constraints::positive::unconstrain;
            cfg.log_density_grad =
                [](const arma::vec& t_unc, const block_context& ctx,
                   arma::vec* grad) {
                    return constraints::positive::wrap(t_unc, grad,
                        [&](const arma::vec& t_nat, arma::vec* g_nat) {
                            return sigma_natural_log_density(t_nat, ctx, g_nat);
                        });
                };
            impl_->add_child(
                std::make_unique<nuts_block>(std::move(cfg)));
        }
        {
            elliptical_slice_sampling_block_config cfg;
            cfg.name        = "f";
            cfg.N           = N;
            // ESS prior covariance MUST be the GP prior K (not K+sigma^2 I).
            cfg.L_chol_key  = "L_prior";
            cfg.initial_f   = arma::vec(N, arma::fill::zeros);
            cfg.log_lik =
                [](const arma::vec& f, const block_context& ctx) -> double {
                    const arma::vec& y = ctx.at("y");
                    const double sig = ctx.at("sigma")[0];
                    const double s2 = sig * sig;
                    double sse = 0.0;
                    for (std::size_t i = 0; i < y.n_elem; ++i) {
                        double d = y[i] - f[i];
                        sse += d * d;
                    }
                    return -0.5 * sse / s2;
                };
            impl_->add_child(
                std::make_unique<elliptical_slice_sampling_block>(
                    std::move(cfg)));
        }

        if (keep_history_) {
            impl_->set_keep_history(true);
        }
    }

    void step() { step(1); }              // no-arg convenience: one sweep
    void step(int n_steps) {
        for (int i = 0; i < n_steps; ++i) impl_->step(rng_);
    }

    AI4BayesCode::state_map get_current() const {
        AI4BayesCode::state_map out;
        out["f"]           = impl_->data().get("f");
        out["amplitude"]   = impl_->data().get("amplitude");
        out["lengthscale"] = impl_->data().get("lengthscale");
        out["sigma"]       = impl_->data().get("sigma");
        return out;
    }

    AI4BayesCode::dag_info get_dag() const { return impl_->get_dag(); }

    AI4BayesCode::history_map get_history() const { return impl_->get_history(); }

    /// 7th R-level method: re-tune NUTS metric (mass matrix + step size +
    /// dual averaging) without advancing chain state. Available because
    /// the composite contains NUTS-family children. See system_design.md
    /// §13 NUTS-family + validator.md §24.
    void readapt_NUTS(int n, bool reset = false) {
        if (n < 0) {
            throw std::runtime_error("readapt_NUTS: n must be non-negative");
        }
        impl_->readapt_NUTS(static_cast<std::size_t>(n),
                            reset, readapt_rng_);
    }


private:
    std::mt19937_64                  rng_;
    mutable std::mt19937_64          predict_rng_;
    mutable std::mt19937_64          readapt_rng_; // readapt_NUTS() advances it (7th method)
    std::unique_ptr<composite_block> impl_;
    std::size_t                      N_ = 0;
    std::size_t                      p_ = 0;
    bool                             keep_history_ = false;

    // libgp kernel object (long-lived; configured by K-matrix refresher)
    std::unique_ptr<libgp::CovSEiso> cf_;
    arma::mat                        X_arma_;
    std::vector<Eigen::VectorXd>     X_rows_;
};

// ============================================================================
// Standalone FRONTEND-INDEPENDENT demo (pure C++; no Rcpp, no pybind).
//
// Simulates 1-D GP-regression data from a KNOWN smooth latent function
//     f_true(x) = sin(3x) + 0.5 x
// observed with Gaussian noise   y_i = f_true(x_i) + eps_i,  eps ~ N(0, s^2).
// Fits the full model (ESS on latent f + NUTS on amplitude/lengthscale/
// noise sigma), then checks that:
//   (1) the posterior-mean latent f recovers f_true much better than the
//       naive "y is the signal" baseline (RMSE(f_hat, f_true) <
//       RMSE(y, f_true), i.e. the GP smooths out noise), and
//   (2) the noise-sigma posterior mean is in the right ballpark of truth.
// ============================================================================
#include <cstdio>

int main() {
    const std::size_t N          = 40;
    const double      sigma_true = 0.30;

    // ---- Simulate from a known smooth truth -----------------------------
    std::mt19937_64 sim_rng(2024);
    std::normal_distribution<double> noise(0.0, sigma_true);

    arma::mat X(N, 1);
    arma::vec f_true(N);
    arma::vec y(N);
    for (std::size_t i = 0; i < N; ++i) {
        const double x = -3.0 + 6.0 * static_cast<double>(i)
                                / static_cast<double>(N - 1);  // grid on [-3,3]
        X(i, 0)   = x;
        f_true[i] = std::sin(3.0 * x) + 0.5 * x;
        y[i]      = f_true[i] + noise(sim_rng);
    }

    // ---- Fit -------------------------------------------------------------
    GPRegression model(X, y, /*rng_seed=*/11, /*keep_history=*/false);
    model.step(800);   // warmup

    const int    M = 1500;
    arma::vec    f_sum(N, arma::fill::zeros);
    double       sigma_bar = 0.0;
    double       amp_bar    = 0.0;
    double       ell_bar    = 0.0;
    for (int s = 0; s < M; ++s) {
        model.step(1);
        const auto cur = model.get_current();
        f_sum     += cur.at("f");
        sigma_bar += cur.at("sigma")[0];
        amp_bar   += cur.at("amplitude")[0];
        ell_bar   += cur.at("lengthscale")[0];
    }
    const arma::vec f_hat = f_sum / static_cast<double>(M);
    sigma_bar /= static_cast<double>(M);
    amp_bar   /= static_cast<double>(M);
    ell_bar   /= static_cast<double>(M);

    // ---- Recovery metrics ------------------------------------------------
    double sse_fit = 0.0, sse_naive = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        const double df = f_hat[i] - f_true[i];
        const double dn = y[i]     - f_true[i];   // naive: observed == signal
        sse_fit   += df * df;
        sse_naive += dn * dn;
    }
    const double rmse_fit   = std::sqrt(sse_fit   / static_cast<double>(N));
    const double rmse_naive = std::sqrt(sse_naive / static_cast<double>(N));

    std::printf("GPRegression demo (N=%zu):\n", N);
    std::printf("  posterior-mean f RMSE vs truth = %.4f\n", rmse_fit);
    std::printf("  naive (y as signal)  RMSE       = %.4f\n", rmse_naive);
    std::printf("  sigma_hat = %.4f  (truth %.2f)\n", sigma_bar, sigma_true);
    std::printf("  amplitude_hat = %.4f   lengthscale_hat = %.4f\n",
                amp_bar, ell_bar);

    // The GP must denoise: smoothed latent f beats raw observations, and the
    // noise scale lands in a sane band around the truth.
    const bool denoises   = rmse_fit < rmse_naive;
    const bool sigma_sane = sigma_bar > 0.5 * sigma_true &&
                            sigma_bar < 2.0 * sigma_true;
    const bool ok = denoises && sigma_sane;

    std::printf("%s\n",
        ok ? "[demo PASS] GP recovers latent f (beats naive) + sane noise sigma"
           : "[demo FAIL] GP did not recover the latent function / noise scale");
    return ok ? 0 : 1;
}
