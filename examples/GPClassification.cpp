// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later
// (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
// GPClassification_joint.cpp
//
// JOINT-NUTS rewrite of GPClassification.cpp. Same model, same priors,
// same posterior — but amplitude and lengthscale are sampled by ONE
// joint_nuts_block instead of two separate single nuts_blocks alternated
// Gibbs-style. The elliptical_slice_sampling_block for f stays separate.
//
// MODEL (identical to GPClassification.cpp)
// ------------------------------------------
//   y_i | f(x_i)  ~ Bernoulli(sigmoid(f(x_i))),   y_i in {0, 1}
//   f             ~ GP(0, K_amplitude_lengthscale(X, X))
//   amplitude     ~ half-Normal(0, 1)
//   lengthscale   ~ InverseGamma(5, median pairwise distance)
//
// JOINT natural-scale log-density over (amplitude, lengthscale):
//
//   log p(amplitude, lengthscale) =
//       -0.5 * amplitude^2 / amp_prior_sd^2                (half-Normal prior)
//     - (shape+1)*log(lengthscale) - scale/lengthscale      (InvGamma prior)
//
//   Each prior term appears exactly ONCE. The ESS block for f absorbs the
//   Bernoulli log-likelihood + GP prior on f; the NUTS hyperparameter
//   blocks sample only from the marginal hyperprior. No shared terms exist
//   between the two priors, so no double-counting is possible.
//
//   grad_{amplitude}   = -amplitude / amp_prior_sd^2
//   grad_{lengthscale} = -(shape+1)/lengthscale + scale/lengthscale^2
//
// BLOCKS (Gibbs sweep order: hyperparams FIRST, f LAST)
// -------------------------------------------------------
//   (amplitude, lengthscale) : ONE joint_nuts_block, sub_params
//                              [{amplitude,1,POSITIVE},{lengthscale,1,POSITIVE}]
//   f                        : elliptical_slice_sampling_block (unchanged)
//
// This file coexists with GPClassification.cpp for cross-validation.
// Class name: GPClassification. Module: GPClassification_module.
//
// LICENSE: libgp_kernels is BSD-3 (GPL-compatible). AI4BayesCode itself is
// GPL-2.0-or-later; combined work remains GPL-2.0-or-later.
// ============================================================================

// Frontend-INDEPENDENT example: pure standalone C++ (NO Rcpp, NO pybind).
// Compile + run directly: it has an int main() that simulates a 1D GP
// classification dataset from a known latent function, fits the joint-NUTS +
// ESS blocks, and checks that the recovered class probabilities track the
// ground truth (and beat a naive constant-rate baseline). No R / Python
// binding is built or required.

#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif

#include <armadillo>

#include "AI4BayesCode/block_sampler.hpp"
#include "AI4BayesCode/shared_data.hpp"
#include "AI4BayesCode/joint_nuts_block.hpp"
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/constraints.hpp"
#include "AI4BayesCode/elliptical_slice_sampling_block.hpp"

// Vendored libgp kernel subsystem (BSD-3).
#include "libgp_kernels_unity.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

using AI4BayesCode::block_context;
using AI4BayesCode::composite_block;
using AI4BayesCode::joint_nuts_block;
using AI4BayesCode::joint_nuts_block_config;
using AI4BayesCode::joint_nuts_sub_param;
using AI4BayesCode::joint_constraint;
using AI4BayesCode::elliptical_slice_sampling_block;
using AI4BayesCode::elliptical_slice_sampling_block_config;
namespace constraints = AI4BayesCode::constraints;

namespace {

// Numerically stable softplus: log(1 + exp(x))
inline double stable_softplus(double x) {
    if (x > 0.0)  return x + std::log1p(std::exp(-x));
    else          return std::log1p(std::exp(x));
}

// Numerically stable sigmoid: 1 / (1 + exp(-x))
inline double stable_sigmoid(double x) {
    if (x >= 0.0) {
        const double e = std::exp(-x);
        return 1.0 / (1.0 + e);
    } else {
        const double e = std::exp(x);
        return e / (1.0 + e);
    }
}

// Pairwise-distance helper for InvGamma scale heuristic.
double median_pairwise_distance(const arma::mat& X) {
    const std::size_t N = X.n_rows;
    if (N < 2) return 1.0;
    std::vector<double> dists;
    dists.reserve(N * (N - 1) / 2);
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = i + 1; j < N; ++j) {
            dists.push_back(arma::norm(X.row(i) - X.row(j), 2));
        }
    }
    std::sort(dists.begin(), dists.end());
    const std::size_t m = dists.size() / 2;
    return (dists.size() % 2 == 1) ? dists[m]
                                   : 0.5 * (dists[m - 1] + dists[m]);
}

// ----------------------------------------------------------------------------
//  Joint natural-scale log-density over (amplitude, lengthscale).
//
//  theta_cat = [ amplitude, lengthscale ]  (NATURAL scale; both > 0,
//  guaranteed by the POSITIVE sub-param transforms inside joint_nuts_block).
//
//  log p(amplitude, lengthscale) =
//      -0.5 * amplitude^2 / amp_prior_sd^2        (half-Normal)
//    + (-(shape+1)*log(lengthscale) - scale/ell)   (InvGamma)
//
//  d/d amplitude   = -amplitude / amp_prior_sd^2
//  d/d lengthscale = -(shape+1)/lengthscale + scale/lengthscale^2
//
//  The ESS block absorbs the Bernoulli likelihood and GP prior on f; these
//  NUTS hyperparameter steps sample only from the marginal hyperprior.
//  joint_nuts_block adds POSITIVE-slice Jacobians (+log amplitude, +log ell)
//  internally; this function must NOT include them.
// ----------------------------------------------------------------------------
double amp_ell_joint_log_density(const arma::vec& theta_cat,
                                 const block_context& ctx,
                                 arma::vec* grad_nat) {
    const double amp   = theta_cat[0];
    const double ell   = theta_cat[1];

    if (amp <= 0.0 || ell <= 0.0) {
        if (grad_nat) grad_nat->set_size(2);
        return -std::numeric_limits<double>::infinity();
    }

    const double s        = ctx.at("amp_prior_sd")[0];
    const double s2       = s * s;
    const double shape    = ctx.at("ell_prior_shape")[0];
    const double scale    = ctx.at("ell_prior_scale")[0];

    // half-Normal prior on amplitude: -0.5 * amp^2 / s^2
    // InvGamma prior on lengthscale: -(shape+1)*log(ell) - scale/ell
    const double lp =
          -0.5 * amp * amp / s2
        + (-(shape + 1.0) * std::log(ell) - scale / ell);

    if (grad_nat) {
        grad_nat->set_size(2);
        (*grad_nat)[0] = -amp / s2;                           // d/d amplitude
        (*grad_nat)[1] = -(shape + 1.0) / ell + scale / (ell * ell); // d/d lengthscale
    }
    return lp;
}

}  // anonymous namespace

// ============================================================================
// User-facing class
// ============================================================================

class GPClassification {
public:
    GPClassification(const arma::mat& X,
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
          impl_(std::make_unique<composite_block>("GPClassification")),
          keep_history_(keep_history)
    {
        if (X.n_rows != y.n_elem)
            throw std::runtime_error("GPClassification: X and y row/length mismatch");
        if (X.n_rows < 2)
            throw std::runtime_error("GPClassification: N must be >= 2");

        const std::size_t N = static_cast<std::size_t>(X.n_rows);
        const std::size_t p = static_cast<std::size_t>(X.n_cols);
        N_ = N;
        p_ = p;

        // Validate y in {0, 1}
        arma::vec y_arma(N);
        for (std::size_t i = 0; i < N; ++i) {
            const double yi = y[i];
            if (yi != 0.0 && yi != 1.0)
                throw std::runtime_error("GPClassification: y must contain only 0/1 values");
            y_arma[i] = yi;
        }
        impl_->data().set("y", y_arma);

        // Cache X as arma::mat + per-row Eigen::VectorXd for libgp.
        X_arma_ = X;
        X_rows_.reserve(N);
        for (std::size_t i = 0; i < N; ++i) {
            Eigen::VectorXd row(p);
            for (std::size_t j = 0; j < p; ++j) row[j] = X(i, j);
            X_rows_.push_back(row);
        }
        impl_->data().set("X", arma::vectorise(X_arma_));

        // Prior hyperparameters:
        //   amplitude: half-Normal(0, 1)   -- weakly-informative default
        //   lengthscale: InverseGamma(5, median_pair_dist)
        impl_->data().set("amp_prior_sd", arma::vec{1.0});
        const double ell_scale = median_pairwise_distance(X_arma_);
        impl_->data().set("ell_prior_shape", arma::vec{5.0});
        impl_->data().set("ell_prior_scale", arma::vec{ell_scale});

        // Initial hyperparameter values
        const double amp_init = 1.0;
        const double ell_init = ell_scale / 5.0;  // InvGamma mean ~ scale/(shape-1)
        impl_->data().set("amplitude",   arma::vec{amp_init});
        impl_->data().set("lengthscale", arma::vec{ell_init});

        // libgp kernel
        cf_ = std::make_unique<libgp::CovSEiso>();
        cf_->init(static_cast<int>(p));

        // Initial latent f = zeros (sigmoid(0) = 0.5; neutral prior start)
        impl_->data().set("f", arma::vec(N, arma::fill::zeros));

        // Seed K_matrix + L_chol buffers
        impl_->data().set("K_matrix", arma::vec(N * N, arma::fill::zeros));
        impl_->data().set("L_chol",   arma::vec(N * N, arma::fill::zeros));

        // ---- Gibbs dependencies -----------------------------------------
        // The joint block's dependencies are keyed under the JOINT BLOCK NAME
        // ("amp_ell_joint") and list only the true external data() inputs
        // (the union of what amplitude and lengthscale each read from data(),
        // minus the now-internal cross-reads of each other).
        impl_->data().declare_dependencies("amp_ell_joint",
            {"amp_prior_sd", "ell_prior_shape", "ell_prior_scale"});
        impl_->data().declare_dependencies("f", {"L_chol", "y"});

        // ---- Invalidation chain -----------------------------------------
        // Keyed under the JOINT BLOCK NAME (composite_block calls
        // refresh_derived_for(block_name) after the joint block steps); keying
        // under sub-param names would never fire -> stale K_matrix/L_chol.
        impl_->data().declare_invalidates("amp_ell_joint",
            {"K_matrix", "L_chol"});

        // ---- K_matrix refresher (libgp CovSEiso) ------------------------
        const libgp::CovSEiso* cf_ptr = cf_.get();
        const std::vector<Eigen::VectorXd>* X_rows_ptr = &X_rows_;
        impl_->data().register_refresher("K_matrix",
            [cf_ptr, X_rows_ptr, N](
                const AI4BayesCode::shared_data_t& d) -> arma::vec {
                const double amp = d.get("amplitude")[0];
                const double ell = d.get("lengthscale")[0];
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

        // ---- L_chol refresher: chol(K + jitter I). NO sigma^2 here. ----
        const double jitter = 1e-5;
        impl_->data().register_refresher("L_chol",
            [N, jitter](const AI4BayesCode::shared_data_t& d) -> arma::vec {
                const arma::vec& K_flat = d.get("K_matrix");
                arma::mat K(const_cast<double*>(K_flat.memptr()), N, N,
                             /*copy_aux_mem=*/false, /*strict=*/true);
                arma::mat M = K;
                M.diag() += jitter;
                arma::mat L;
                if (!arma::chol(L, M, "lower")) {
                    M.diag() += 1e-3;
                    arma::chol(L, M, "lower");
                }
                return arma::vectorise(L);
            });

        impl_->data().refresh_all();

        // ---- Predict DAG + y_rep stochastic refresher -------------------
        impl_->data().declare_data_input("X");
        impl_->data().declare_predict_edges("X",        {"K_matrix"});
        impl_->data().declare_predict_edges("K_matrix", {"L_chol"});
        impl_->data().declare_predict_edges("f",        {"y_rep"});

        // ---- Generative-DAG context (VIZ-ONLY; predict_at BFS never
        //      reads context_edges_). declare_predict_edges + context_edges
        //      stay keyed by SUB-PARAM name (unchanged from original).
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
                std::uniform_real_distribution<double> ud(0.0, 1.0);
                arma::vec out(f.n_elem);
                for (std::size_t i = 0; i < f.n_elem; ++i) {
                    const double pi = stable_sigmoid(f[i]);
                    out[i] = (ud(rng) < pi) ? 1.0 : 0.0;
                }
                return out;
            });

        // ---- Child blocks in Gibbs order --------------------------------
        //   child(0) amp_ell_joint (joint_nuts_block for amplitude+lengthscale)
        //   child(1) f             (elliptical_slice_sampling_block)
        {
            joint_nuts_block_config cfg;
            cfg.name = "amp_ell_joint";
            cfg.sub_params.push_back(
                joint_nuts_sub_param{ "amplitude",   1,
                                      joint_constraint::POSITIVE });
            cfg.sub_params.push_back(
                joint_nuts_sub_param{ "lengthscale", 1,
                                      joint_constraint::POSITIVE });
            // initial_cat is NATURAL-scale: [amp_init, ell_init].
            cfg.initial_cat = arma::vec{ amp_init, ell_init };
            cfg.log_density_grad = &amp_ell_joint_log_density;
            // amplitude and lengthscale operate on very different scales
            // (amp ~ O(1), ell ~ O(median pairwise dist)); diagonal metric
            // is required for faithful sampling of heterogeneous scales.
            cfg.use_diagonal_metric = true;
            // Give the joint block warmup runway.
            cfg.n_warmup_first_call = 800;
            impl_->add_child(std::make_unique<joint_nuts_block>(std::move(cfg)));
        }
        {
            elliptical_slice_sampling_block_config cfg;
            cfg.name       = "f";
            cfg.N          = N;
            cfg.L_chol_key = "L_chol";
            cfg.initial_f  = arma::vec(N, arma::fill::zeros);
            // Bernoulli-logit log-likelihood: sum y_i f_i - softplus(f_i).
            cfg.log_lik =
                [](const arma::vec& f, const block_context& ctx) -> double {
                    const arma::vec& y = ctx.at("y");
                    double lp = 0.0;
                    for (std::size_t i = 0; i < y.n_elem; ++i) {
                        lp += y[i] * f[i] - stable_softplus(f[i]);
                    }
                    return lp;
                };
            impl_->add_child(
                std::make_unique<elliptical_slice_sampling_block>(
                    std::move(cfg)));
        }

        if (keep_history_) impl_->set_keep_history(true);
    }

    void step(int n_steps) {
        for (int i = 0; i < n_steps; ++i) impl_->step(rng_);
    }

    AI4BayesCode::state_map get_current() const {
        // sub-params are written back to data() under sub-param names by
        // joint_nuts_block; read directly from data().
        AI4BayesCode::state_map out;
        out["f"]           = impl_->data().get("f");
        out["amplitude"]   = impl_->data().get("amplitude");
        out["lengthscale"] = impl_->data().get("lengthscale");
        return out;
    }

    void set_current(const AI4BayesCode::state_map& params) {
        // For the joint block: read its current concatenated vector, overwrite
        // the relevant slice(s), set_current, mirror to data().
        auto& jblk = dynamic_cast<joint_nuts_block&>(impl_->child(0));
        arma::vec cat_new = jblk.current();   // [amplitude, lengthscale]
        bool touched_joint = false;

        auto it_amp = params.find("amplitude");
        if (it_amp != params.end()) {
            const double a = it_amp->second[0];
            if (!(a > 0.0)) throw std::runtime_error("amplitude must be positive");
            cat_new[0] = a;
            touched_joint = true;
        }
        auto it_ell = params.find("lengthscale");
        if (it_ell != params.end()) {
            const double l = it_ell->second[0];
            if (!(l > 0.0)) throw std::runtime_error("lengthscale must be positive");
            cat_new[1] = l;
            touched_joint = true;
        }
        if (touched_joint) {
            jblk.set_current(cat_new);
            impl_->data().set("amplitude",   arma::vec{cat_new[0]});
            impl_->data().set("lengthscale", arma::vec{cat_new[1]});
            impl_->data().refresh_derived_for("amplitude");
            impl_->data().refresh_derived_for("lengthscale");
        }

        auto it_f = params.find("f");
        if (it_f != params.end()) {
            const arma::vec& f_new = it_f->second;
            if (static_cast<std::size_t>(f_new.n_elem) != N_)
                throw std::runtime_error("GPClassification::set_current: f length mismatch");
            impl_->data().set("f", f_new);
            dynamic_cast<elliptical_slice_sampling_block&>(
                impl_->child(1)).set_current(f_new);
        }
    }

    // NOTE: the Rcpp/pybind frontend exposed a predict_at() method (GP
    // posterior predictive p(f* | X*) -> sigmoid -> Bernoulli y_rep). It is
    // omitted from this frontend-independent build: the standalone demo
    // checks recovery directly from get_current() / the latent f, and
    // predict_at returned frontend (Rcpp::List) types. The MODEL (priors,
    // log-density, block config, hyperparameters) is preserved exactly.

    AI4BayesCode::dag_info get_dag() const { return impl_->get_dag(); }
    AI4BayesCode::history_map get_history() const { return impl_->get_history(); }

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
    mutable std::mt19937_64          readapt_rng_;
    std::unique_ptr<composite_block> impl_;
    std::size_t                      N_ = 0;
    std::size_t                      p_ = 0;
    bool                             keep_history_ = false;

    // libgp kernel object
    std::unique_ptr<libgp::CovSEiso> cf_;
    arma::mat                        X_arma_;
    std::vector<Eigen::VectorXd>     X_rows_;
};

//==============================================================================
//  Standalone FRONTEND-INDEPENDENT demo (pure C++; no Rcpp, no pybind).
//
//  Simulates a 1D GP-classification dataset from a KNOWN latent function:
//      x_i      ~ Uniform(-3, 3)
//      f_true_i = 1.6 * sin(1.3 * x_i)          (a smooth latent signal)
//      p_true_i = sigmoid(f_true_i)
//      y_i      ~ Bernoulli(p_true_i)
//
//  Fits the model (joint-NUTS on amplitude+lengthscale, ESS on latent f,
//  Bernoulli-logit likelihood), averages the posterior latent f -> posterior
//  class probability at each training point, and checks that:
//    (a) the posterior probabilities track the TRUE probabilities better than
//        a naive constant-rate baseline (mean(y)) — i.e. lower mean abs error;
//    (b) the recovered probabilities are accurate in absolute terms.
//==============================================================================
#include <cstdio>
int main() {
    const std::size_t N = 60;

    std::mt19937_64 sim_rng(20260621ULL);
    std::uniform_real_distribution<double> xdist(-3.0, 3.0);
    std::uniform_real_distribution<double> udist(0.0, 1.0);

    // Stable sigmoid for the simulation side.
    auto sig = [](double z) {
        if (z >= 0.0) { const double e = std::exp(-z); return 1.0 / (1.0 + e); }
        const double e = std::exp(z); return e / (1.0 + e);
    };

    arma::mat X(N, 1);
    arma::vec y(N);
    arma::vec p_true(N);
    double y_mean = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        const double x = xdist(sim_rng);
        X(i, 0) = x;
        const double f_true = 1.6 * std::sin(1.3 * x);
        const double pi = sig(f_true);
        p_true[i] = pi;
        y[i] = (udist(sim_rng) < pi) ? 1.0 : 0.0;
        y_mean += y[i];
    }
    y_mean /= static_cast<double>(N);

    GPClassification model(X, y, /*rng_seed=*/11, /*keep_history=*/false);
    model.step(300);   // warmup (joint block also self-warms n_warmup_first_call)

    // Posterior mean of sigmoid(f) at each training point.
    arma::vec prob_bar(N, arma::fill::zeros);
    const int M = 1500;
    for (int s = 0; s < M; ++s) {
        model.step(1);
        const auto cur = model.get_current();
        const arma::vec& f = cur.at("f");
        for (std::size_t i = 0; i < N; ++i) {
            const double fi = f[i];
            const double pi = (fi >= 0.0)
                ? 1.0 / (1.0 + std::exp(-fi))
                : std::exp(fi) / (1.0 + std::exp(fi));
            prob_bar[i] += pi;
        }
    }
    prob_bar /= static_cast<double>(M);

    // Mean absolute error of posterior prob vs TRUE prob, and of the naive
    // constant-rate baseline mean(y) vs TRUE prob.
    double mae_model = 0.0, mae_naive = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        mae_model += std::abs(prob_bar[i] - p_true[i]);
        mae_naive += std::abs(y_mean     - p_true[i]);
    }
    mae_model /= static_cast<double>(N);
    mae_naive /= static_cast<double>(N);

    const auto cur = model.get_current();
    const double amp_hat = cur.at("amplitude")[0];
    const double ell_hat = cur.at("lengthscale")[0];

    std::printf("GPClassification demo (N=%zu, 1D GP-logit):\n", N);
    std::printf("  amplitude_hat = %.3f   lengthscale_hat = %.3f\n",
                amp_hat, ell_hat);
    std::printf("  MAE(post prob vs true) = %.4f\n", mae_model);
    std::printf("  MAE(naive mean(y)=%.3f vs true) = %.4f\n", y_mean, mae_naive);

    // Pass: model recovers the latent class-probability surface better than the
    // naive constant baseline AND is accurate in absolute terms.
    const bool ok = (mae_model < mae_naive) && (mae_model < 0.20);
    std::printf("%s\n", ok
        ? "[demo PASS] GP-logit recovers latent class probabilities, "
          "beats naive baseline"
        : "[demo FAIL] posterior probabilities did not recover the latent "
          "surface");
    return ok ? 0 : 1;
}
