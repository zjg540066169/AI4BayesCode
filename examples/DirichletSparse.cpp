// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later
// (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
//  DirichletSparse_joint.cpp
//
//  JOINT-NUTS rewrite of DirichletSparse.cpp. Same model/priors/posterior, but
//  (s, theta) are sampled by ONE joint_nuts_block (sub-params SIMPLEX + POSITIVE)
//  instead of two separate single nuts_blocks alternated Gibbs-style. s and theta
//  are tightly coupled through alpha = theta/P, so a single joint NUTS trajectory
//  mixes the dependence directly. This also exercises the dim-CHANGING SIMPLEX
//  constraint as a joint sub-param (unconstrained width P-1, natural width P).
//
//  Model (identical to DirichletSparse.cpp)
//  ----------------------------------------
//      y      ~ Multinomial(N, s)                       (P categories)
//      s      ~ Dirichlet(theta/P, ..., theta/P)
//      theta / (theta + P) ~ Beta(0.5, 1)
//
//  JOINT natural-scale log-density over x = [s (P), theta (1)] (each model term
//  exactly ONCE; the shared Dirichlet-prior kernel (theta/P-1)*sum_i log s_i is
//  counted ONCE, NOT once per conditional):
//
//    log p(s, theta | y) =
//        sum_i y_i * log s_i                         # Multinomial likelihood
//      + lgamma(theta) - P*lgamma(theta/P)           # Dirichlet normalizer
//      + (theta/P - 1) * sum_i log s_i               # Dirichlet prior kernel (ONCE)
//      - 0.5*log(theta) - 1.5*log(theta + P)         # Beta(0.5,1) hyperprior
//
//    grad_{s_i}  = (y_i + theta/P - 1) / s_i
//    grad_theta  = psi(theta) - psi(theta/P) + (1/P)*sum_i log s_i
//                  - 0.5/theta - 1.5/(theta + P)
//
//  joint_nuts_block adds the per-slice Jacobians (simplex stick-breaking on s,
//  log on theta) internally; this function stays on the NATURAL scale.
//  Cross-validated against DirichletSparse.cpp (posteriors match, R-hat).
// ============================================================================

// @example:R
//   library(AI4BayesCode)
//   ai4bayescode_example("DirichletSparse")
//   set.seed(2026)
//   s_true <- c(0.50, 0.25, 0.15, 0.05, 0.03, 0.02)  # sparse P=6 simplex
//   N      <- 1000L                                   # multinomial trials (N >> P)
//   y      <- as.numeric(rmultinom(1, N, s_true))     # observed category counts
//   # ---- Recommended: parallel chains + convergence diagnosis ----
//   run <- ai4bayescode_run_chains(
//       function(seed) new(DirichletSparse, y, seed, TRUE),
//       n_chains = 4, n_burn = 1000, n_keep = 2000)
//   ai4bayescode_diagnose(run$histories[[1]])      # summary + R-hat/ESS + plots
//   # ---- Advanced: stateful single-chain control ----
//   m <- new(DirichletSparse, y, 7L, TRUE)            # y_counts, rng_seed, keep_history
//   m$step(2500); str(m$get_current())                # s (P-simplex) + theta (>0)
// @example:python
//   import numpy as np, AI4BayesCode
//   rng = np.random.default_rng(2026)
//   s_true = np.array([0.50, 0.25, 0.15, 0.05, 0.03, 0.02])  # sparse P=6 simplex
//   N = 1000                                                 # multinomial trials (N >> P)
//   y = rng.multinomial(N, s_true).astype(float)             # observed category counts
//   Mod = AI4BayesCode.example("DirichletSparse")
//   # ---- Recommended: parallel chains + diagnosis ----
//   chains = AI4BayesCode.run_chains(
//       lambda seed: Mod.DirichletSparse(y, seed, True),
//       seeds=[101, 202, 303, 404], n_burn=1000, n_keep=2000, n_jobs=1)
//   AI4BayesCode.diagnose(chains[0]["hist"])   # summary + diagnostics
//   # ---- Advanced: stateful single-chain control ----
//   m = Mod.DirichletSparse(y, 7, True)  # (y_counts, rng_seed, keep_history)
//   m.step(2500); print(m.get_current())                     # s (P-simplex) + theta (>0)
// @example:end

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
#include "AI4BayesCode/joint_nuts_block.hpp"
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/constraints.hpp"
#include "AI4BayesCode/rcpp_wrap.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>

using AI4BayesCode::block_context;
using AI4BayesCode::composite_block;
using AI4BayesCode::joint_nuts_block;
using AI4BayesCode::joint_nuts_block_config;
using AI4BayesCode::joint_nuts_sub_param;
using AI4BayesCode::joint_constraint;

namespace constraints = AI4BayesCode::constraints;

namespace {

// ----------------------------------------------------------------------------
//  Joint natural-scale log-density of (s, theta) given y.
//  x = [ s (P entries, simplex), theta (1, positive) ], length P+1.
// ----------------------------------------------------------------------------
double s_theta_joint_log_density(const arma::vec& x,
                                 const block_context& ctx,
                                 arma::vec* grad_nat) {
    const arma::vec& y  = ctx.at("y");          // counts, length P
    const std::size_t P = y.n_elem;
    const double theta  = x[P];
    if (theta <= 0.0) return -std::numeric_limits<double>::infinity();
    const double Pd     = static_cast<double>(P);
    const double alpha  = theta / Pd;

    double sum_log_s = 0.0;
    double like_s    = 0.0;   // sum_i y_i log s_i
    for (std::size_t i = 0; i < P; ++i) {
        const double si = x[i];
        if (si <= 0.0) return -std::numeric_limits<double>::infinity();
        const double ls = std::log(si);
        sum_log_s += ls;
        like_s    += y[i] * ls;
    }

    const double lp =
          like_s                                            // Multinomial likelihood
        + ai4b::lgammafn(theta) - Pd * ai4b::lgammafn(alpha)// Dirichlet normalizer
        + (alpha - 1.0) * sum_log_s                         // Dirichlet prior kernel (ONCE)
        - 0.5 * std::log(theta) - 1.5 * std::log(theta + Pd); // Beta(0.5,1) hyperprior

    if (grad_nat) {
        grad_nat->set_size(P + 1);
        for (std::size_t i = 0; i < P; ++i)
            (*grad_nat)[i] = (y[i] + alpha - 1.0) / x[i];   // d/d s_i
        (*grad_nat)[P] =                                     // d/d theta
              ai4b::digamma(theta) - ai4b::digamma(alpha)
            + sum_log_s / Pd
            - 0.5 / theta - 1.5 / (theta + Pd);
    }
    return lp;
}

} // anonymous namespace

class DirichletSparse {
public:
    DirichletSparse(const arma::vec& y_counts,
                         int    rng_seed,
                         bool   keep_history = false)
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
          impl_(std::make_unique<composite_block>("DirichletSparse")),
          keep_history_(keep_history)
    {
        const int P = static_cast<int>(y_counts.n_elem);
        if (P < 2) ai4b::stop("y_counts must have length P >= 2");
        if (arma::any(y_counts < 0.0)) ai4b::stop("y_counts entries must be non-negative");

        const double theta_init = 1.0;
        impl_->data().set("y", y_counts);

        arma::vec s_init = y_counts + 1.0;
        s_init /= arma::sum(s_init);
        impl_->data().set("s",     s_init);
        impl_->data().set("theta", arma::vec{theta_init});

        // Joint block's dependencies are keyed under the JOINT BLOCK NAME and
        // are only the true external data() inputs (s<->theta are now INTERNAL
        // to the joint block, not data() cross-reads).
        impl_->data().declare_dependencies("s_theta", {"y"});

        // predict edges + viz context keyed by SUB-PARAM name (unchanged).
        impl_->data().declare_predict_edges("s", {"y_rep"});
        impl_->data().declare_context_edges("theta", {"s"});

        impl_->data().set("y_rep", arma::vec(y_counts.n_elem, arma::fill::zeros));
        N_total_ = static_cast<int>(arma::sum(y_counts));
        const int N_total = N_total_;
        impl_->data().register_stochastic_refresher(
            "y_rep",
            [N_total](const AI4BayesCode::shared_data_t& d,
                      std::mt19937_64& rng) {
                const arma::vec& s = d.get("s");
                const std::size_t K = s.n_elem;
                std::discrete_distribution<int> cat(s.begin(), s.end());
                arma::vec counts(K, arma::fill::zeros);
                for (int n = 0; n < N_total; ++n)
                    counts[static_cast<std::size_t>(cat(rng))] += 1.0;
                return counts;
            });

        // ---- ONE joint_nuts_block over (s SIMPLEX, theta POSITIVE) -------
        {
            joint_nuts_block_config cfg;
            cfg.name = "s_theta";
            cfg.sub_params.push_back(
                joint_nuts_sub_param{ "s", static_cast<std::size_t>(P),
                                      joint_constraint::SIMPLEX });
            cfg.sub_params.push_back(
                joint_nuts_sub_param{ "theta", 1, joint_constraint::POSITIVE });
            // initial_cat is NATURAL-scale: [s_init (P), theta_init].
            arma::vec init_cat(P + 1);
            for (int i = 0; i < P; ++i) init_cat[i] = s_init[i];
            init_cat[P] = theta_init;
            cfg.initial_cat = init_cat;
            cfg.log_density_grad = &s_theta_joint_log_density;
            // s and theta are tightly coupled (alpha = theta/P) + the simplex
            // geometry is the harder slice; give the joint warmup runway.
            cfg.n_warmup_first_call = 600;
            cfg.use_diagonal_metric = true;  // s super-concentrated, theta heavy-tail: very different scales
            impl_->add_child(std::make_unique<joint_nuts_block>(std::move(cfg)));
        }

        if (keep_history_) impl_->set_keep_history(true);
    }

    void step() { step(1); }              // no-arg convenience: one sweep
    void step(int n_steps) {
        for (int i = 0; i < n_steps; ++i) impl_->step(rng_);
    }

    AI4BayesCode::state_map get_current() const {
        AI4BayesCode::state_map out;
        out["s"]     = impl_->data().get("s");      // length-P simplex vector
        out["theta"] = impl_->data().get("theta");  // length-1 arma::vec
        return out;
    }

    void set_current(const AI4BayesCode::state_map& params) {
        auto& jblk = dynamic_cast<joint_nuts_block&>(impl_->child(0));
        const std::size_t P = jblk.sub_param_dim("s");
        arma::vec cat_new = jblk.current();    // [s (P), theta]
        bool touched = false;

        auto it_s = params.find("s");
        if (it_s != params.end()) {
            arma::vec s_new = it_s->second;
            if (s_new.n_elem != P) ai4b::stop("s has wrong length");
            if (arma::any(s_new <= 0.0)) ai4b::stop("s entries must be strictly positive");
            const double ssum = arma::sum(s_new);
            if (std::abs(ssum - 1.0) > 1e-8) s_new /= ssum;
            for (std::size_t i = 0; i < P; ++i) cat_new[i] = s_new[i];
            touched = true;
        }
        auto it_theta = params.find("theta");
        if (it_theta != params.end()) {
            const double th_new = it_theta->second[0];
            if (th_new <= 0.0) ai4b::stop("theta must be strictly positive");
            cat_new[P] = th_new;
            touched = true;
        }
        if (touched) {
            jblk.set_current(cat_new);
            impl_->data().set("s",     cat_new.subvec(0, P - 1));
            impl_->data().set("theta", arma::vec{cat_new[P]});
        }
    }

    AI4BayesCode::history_map predict_at(const AI4BayesCode::state_map& new_data) const {
        if (!new_data.empty())
            ai4b::stop("DirichletSparse has no covariate inputs. "
                       "predict_at() takes an empty map/list.");
        AI4BayesCode::history_map out;

        if (!keep_history_) {
            block_context replaced;
            block_context result = impl_->predict_at(replaced, predict_rng_);
            for (const auto& kv : result) {
                arma::mat m(1, kv.second.n_elem);
                for (std::size_t j = 0; j < kv.second.n_elem; ++j)
                    m(0, j) = kv.second[j];
                out.emplace(kv.first, std::move(m));
            }
            return out;
        }

        // History mode: s is a joint sub-output (keyed by sub-param name "s"),
        // so the composite predict_at rejects it as replaced context; compute
        // y_rep manually per draw (cf. DirichletHierarchical_joint / IRT1PL_joint).
        // y_rep ~ Multinomial(N_total, s_draw), reported as category counts.
        AI4BayesCode::history_map hist = impl_->get_history();
        const arma::mat& s_hist = hist.at("s");      // n_draws x P
        const std::size_t n_draws = s_hist.n_rows;
        const std::size_t K       = s_hist.n_cols;

        arma::mat yrep(n_draws, K);
        for (std::size_t d = 0; d < n_draws; ++d) {
            std::vector<double> probs(K);
            for (std::size_t j = 0; j < K; ++j) probs[j] = s_hist(d, j);
            std::discrete_distribution<int> cat(probs.begin(), probs.end());
            arma::vec counts(K, arma::fill::zeros);
            for (int n = 0; n < N_total_; ++n)
                counts[static_cast<std::size_t>(cat(predict_rng_))] += 1.0;
            for (std::size_t j = 0; j < K; ++j) yrep(d, j) = counts[j];
        }
        out.emplace("y_rep", std::move(yrep));
        return out;
    }

    AI4BayesCode::dag_info      get_dag()     const { return impl_->get_dag(); }
    AI4BayesCode::history_map   get_history() const { return impl_->get_history(); }

    void readapt_NUTS(int n, bool reset = false, int max_tree_depth = -1) {
        if (n < 0) ai4b::stop("readapt_NUTS: n must be non-negative");
        impl_->readapt_NUTS(static_cast<std::size_t>(n), reset, readapt_rng_, max_tree_depth < 0 ? std::size_t(0) : static_cast<std::size_t>(max_tree_depth));
    }

private:
    std::mt19937_64                  rng_;
    mutable std::mt19937_64          predict_rng_;
    mutable std::mt19937_64          readapt_rng_;
    std::unique_ptr<composite_block> impl_;
    bool                             keep_history_ = false;
    int                              N_total_      = 0;
};

#ifdef AI4BAYESCODE_RCPP_MODULE
RCPP_MODULE(DirichletSparse_module) {
    Rcpp::class_<DirichletSparse>("DirichletSparse")
        .constructor<arma::vec, int>("Legacy constructor; keep_history defaults to FALSE.")
        .constructor<arma::vec, int, bool>(
            "Joint-NUTS DirichletSparse: (s, theta) in one joint_nuts_block.")
        .method("step", (void (DirichletSparse::*)())    &DirichletSparse::step, "Run one sweep.")
        .method("step", (void (DirichletSparse::*)(int)) &DirichletSparse::step,
                "Run n sweeps (each: one JOINT NUTS update of (s, theta)).")
        .method("get_current", &DirichletSparse::get_current)
        .method("set_current", &DirichletSparse::set_current)
        .method("predict_at",  &DirichletSparse::predict_at)
        .method("get_dag",     &DirichletSparse::get_dag)
        .method("get_history", &DirichletSparse::get_history)
        .method("readapt_NUTS",&DirichletSparse::readapt_NUTS);
}
#endif

#ifdef AI4BAYESCODE_PYBIND_MODULE
#include "AI4BayesCode/pybind_casters.hpp"
PYBIND11_MODULE(DirichletSparse, m) {
    AI4BayesCode::register_ai4bayescode_types(m);
    pybind11::class_<DirichletSparse>(m, "DirichletSparse")
        .def(pybind11::init<arma::vec, int, bool>(),
             pybind11::arg("y_counts"),
             pybind11::arg("rng_seed") = 1,
             pybind11::arg("keep_history") = false)
        .def("step", (void (DirichletSparse::*)())    &DirichletSparse::step, "Run one sweep.")
        .def("step", (void (DirichletSparse::*)(int)) &DirichletSparse::step,    pybind11::arg("n_steps"))
        .def("get_current",  &DirichletSparse::get_current)
        .def("set_current",  &DirichletSparse::set_current, pybind11::arg("params"))
        .def("predict_at",   &DirichletSparse::predict_at,  pybind11::arg("new_data"))
        .def("get_dag",      &DirichletSparse::get_dag)
        .def("get_history",  &DirichletSparse::get_history)
        .def("readapt_NUTS", &DirichletSparse::readapt_NUTS,
             pybind11::arg("n"), pybind11::arg("reset") = false, pybind11::arg("max_tree_depth") = -1);
}
#endif

//==============================================================================
//  Standalone FRONTEND-INDEPENDENT demo (active only when NO binding macro is
//  defined). Simulates multinomial counts from a known sparse probability
//  vector s_true, fits via the joint-NUTS block (s SIMPLEX, theta POSITIVE),
//  and checks that the posterior-mean of s recovers s_true. State is read via
//  the full contract get_current() (keys "s", "theta"). PASS is derived from a
//  real computed error vs the truth, never hard-coded.
//==============================================================================
#if !defined(AI4BAYESCODE_RCPP_MODULE) && !defined(AI4BAYESCODE_PYBIND_MODULE)
#include <cstdio>
int main() {
    // Sparse truth: most mass on a few categories.
    const arma::vec s_true = arma::vec{0.50, 0.25, 0.15, 0.05, 0.03, 0.02};
    const std::size_t P = s_true.n_elem;
    const int N = 500;   // multinomial trials

    // Simulate y ~ Multinomial(N, s_true).
    std::mt19937_64 sim_rng(2026);
    std::discrete_distribution<int> cat(s_true.begin(), s_true.end());
    arma::vec y(P, arma::fill::zeros);
    for (int n = 0; n < N; ++n) y[static_cast<std::size_t>(cat(sim_rng))] += 1.0;

    // Fit.
    DirichletSparse model(y, /*rng_seed=*/7, /*keep_history=*/false);
    model.step(800);   // warmup sweeps (block also self-warms n_warmup_first_call)

    arma::vec s_bar(P, arma::fill::zeros);
    double theta_bar = 0.0;
    const int M = 3000;
    for (int it = 0; it < M; ++it) {
        model.step(1);
        const auto gc = model.get_current();          // copy (avoids dangling ref)
        const arma::vec& s_cur     = gc.at("s");
        const arma::vec& theta_cur = gc.at("theta");
        s_bar     += s_cur;
        theta_bar += theta_cur[0];
    }
    s_bar     /= static_cast<double>(M);
    theta_bar /= static_cast<double>(M);

    // Posterior mean of s should recover s_true. Naive frequency baseline = y/N.
    const arma::vec s_freq = y / static_cast<double>(N);
    const double err_post = arma::norm(s_bar  - s_true, 2);
    const double err_freq = arma::norm(s_freq - s_true, 2);

    std::printf("DirichletSparse demo (P=%zu, N=%d):\n", P, N);
    std::printf("  s_true = ");
    for (std::size_t i = 0; i < P; ++i) std::printf("%.3f ", s_true[i]);
    std::printf("\n  s_hat  = ");
    for (std::size_t i = 0; i < P; ++i) std::printf("%.3f ", s_bar[i]);
    std::printf("\n  s_freq = ");
    for (std::size_t i = 0; i < P; ++i) std::printf("%.3f ", s_freq[i]);
    std::printf("\n  L2(post - truth) = %.4f   L2(freq - truth) = %.4f\n",
                err_post, err_freq);
    std::printf("  theta_hat = %.3f\n", theta_bar);

    // PASS criterion: posterior mean is close to truth in an absolute sense AND
    // is at least as good as the naive frequency estimate (within slack).
    const bool ok = (err_post < 0.05) && (err_post <= err_freq + 0.01);
    std::printf("%s\n", ok ? "[demo PASS] joint-NUTS recovers sparse s"
                           : "[demo FAIL]");
    return ok ? 0 : 1;
}
#endif
