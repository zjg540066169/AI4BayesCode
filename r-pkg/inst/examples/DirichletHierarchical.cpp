// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later
// (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
//  DirichletHierarchical_joint.cpp
//
//  JOINT-NUTS rewrite of DirichletHierarchical.cpp. Same model, same priors,
//  same posterior — but (s, kappa, theta) are sampled by ONE joint_nuts_block
//  instead of three separate single nuts_blocks updated Gibbs-style. This
//  collapses the three tightly-coupled parameters (s and kappa share a
//  kappa*s_i argument to lgamma, and s and theta share the Dirichlet prior
//  alpha = theta/P) into a single HMC trajectory.
//
//  Model (identical to DirichletHierarchical.cpp)
//  -----------------------------------------------
//      s_k | s, kappa         ~ Dirichlet(kappa * s),   k = 1..K
//      s                      ~ Dirichlet(theta/P, ..., theta/P)
//      kappa                  ~ Gamma(shape = 1, rate = 1)    ( == Exp(1) )
//      theta / (theta + P)    ~ Beta(a, b)
//
//  Sub-params: [{s, P, SIMPLEX}, {kappa, 1, POSITIVE}, {theta, 1, POSITIVE}].
//  Concatenated natural vector: x = [s(0)...s(P-1), kappa, theta], length P+2.
//
//  JOINT natural-scale log-density (each model term EXACTLY ONCE):
//
//    log p(s, kappa, theta | L, K) =
//        sum_i [ -K * lgamma(kappa*s_i) + kappa*s_i*L_i + (theta/P-1)*log s_i ]
//      + K * lgamma(kappa) - kappa
//      + lgamma(theta) - P*lgamma(theta/P) + (a-1)*log(theta) - (a+b)*log(theta+P)
//
//  The three original per-block conditionals SHARE cross-terms:
//    - The s conditional contains kappa cross-terms and the Dir(s) kernel.
//    - The kappa conditional ALSO contains -K*sum_i lgamma(kappa*s_i) and
//      kappa*sum_i s_i*L_i from the observation likelihood. These are ALREADY
//      in the s block's sum, so kappa contributes ONLY its unique terms:
//      K*lgamma(kappa) (Dir-normalizer contribution) and -kappa (Exp(1) prior).
//    - The theta conditional also contains (theta/P-1)*sum_i log s_i (already
//      in the s block sum), so theta contributes ONLY its normalizer and prior.
//
//  Gradients (natural scale, no Jacobians — joint_nuts_block adds them):
//    d/d s_i   = -K*kappa*psi(kappa*s_i) + kappa*L_i + (theta/P-1)/s_i
//    d/d kappa = K*psi(kappa) - K*sum_i s_i*psi(kappa*s_i) + sum_i s_i*L_i - 1
//    d/d theta = psi(theta) - psi(theta/P) + (1/P)*sum_i log s_i
//                + (a-1)/theta - (a+b)/(theta+P)
//
//  joint_nuts_block adds per-slice Jacobians (stick-breaking for s, log for
//  kappa and theta) internally; this function stays on the NATURAL scale.
// ============================================================================
//
// @example:R
//   library(AI4BayesCode)
//   ai4bayescode_example("DirichletHierarchical")
//   set.seed(2026)
//   ## Simulate K=400 simplexes ~ Dirichlet(kappa_true * s_true): well
//   ## replicated (400 obs to identify a length-4 base simplex + kappa + theta).
//   P <- 4L; K <- 400L; kappa_true <- 30
//   s_true <- c(0.50, 0.25, 0.15, 0.10); s_true <- s_true / sum(s_true)
//   rdir <- function(a) { g <- rgamma(length(a), shape = a, rate = 1); g / sum(g) }
//   S_obs <- t(replicate(K, pmax(rdir(kappa_true * s_true), 1e-8)))
//   S_obs <- S_obs / rowSums(S_obs)
//   # ---- Recommended: parallel chains + convergence diagnosis ----
//   run <- AI4BayesCode_run_chains(
//       function(seed) new(DirichletHierarchical, S_obs, 0.5, 1.0, seed, TRUE),
//       n_chains = 4, n_burn = 1000, n_keep = 2000)
//   ai4b_diagnose(run$histories[[1]])      # summary + R-hat/ESS + plots
//   # ---- Advanced: stateful single-chain control ----
//   m <- new(DirichletHierarchical,
//            S_obs,   # K x P matrix of observed simplexes
//            0.5,     # beta_a : Beta hyperprior shape on theta/(theta+P)
//            1.0,     # beta_b : Beta hyperprior shape on theta/(theta+P)
//            7L,      # rng_seed (0 = random)
//            TRUE)    # keep_history
//   m$step(2500); str(m$get_current())
// @example:python
//   import numpy as np, AI4BayesCode
//   rng = np.random.default_rng(2026)
//   ## Simulate K=400 simplexes ~ Dirichlet(kappa_true * s_true).
//   P, K, kappa_true = 4, 400, 30.0
//   s_true = np.array([0.50, 0.25, 0.15, 0.10]); s_true = s_true / s_true.sum()
//   def rdir(a):
//       g = rng.gamma(shape=a, scale=1.0); return g / g.sum()
//   S_obs = np.vstack([np.maximum(rdir(kappa_true * s_true), 1e-8) for _ in range(K)])
//   S_obs = S_obs / S_obs.sum(axis=1, keepdims=True)
//   Mod = AI4BayesCode.source("DirichletHierarchical.cpp")
//   # ---- Recommended: parallel chains + diagnosis ----
//   chains = AI4BayesCode.run_chains(
//       lambda seed: Mod.DirichletHierarchical(S_obs, 0.5, 1.0, seed, True),
//       seeds=[101, 202, 303, 404], n_burn=1000, n_keep=2000, n_jobs=1)
//   AI4BayesCode.ai4b_diagnose(chains[0]["hist"])   # summary + diagnostics
//   # ---- Advanced: stateful single-chain control ----
//   m = Mod.DirichletHierarchical(S_obs, 0.5, 1.0, 7, True)  # (S_obs, beta_a, beta_b, seed, keep_history)
//   m.step(2500); print(m.get_current())
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

using AI4BayesCode::block_context;
using AI4BayesCode::composite_block;
using AI4BayesCode::joint_nuts_block;
using AI4BayesCode::joint_nuts_block_config;
using AI4BayesCode::joint_nuts_sub_param;
using AI4BayesCode::joint_constraint;

namespace constraints = AI4BayesCode::constraints;

namespace {

// ----------------------------------------------------------------------------
//  Joint natural-scale log-density of (s, kappa, theta) given (L, K, a, b).
//
//  x layout: [s(0), ..., s(P-1), kappa, theta],  length P+2.
//
//  log p(s, kappa, theta | L, K, beta_a, beta_b) =
//      sum_i [ -K*lgamma(kappa*s_i) + kappa*s_i*L_i + (theta/P-1)*log s_i ]
//    + K*lgamma(kappa) - kappa
//    + lgamma(theta) - P*lgamma(theta/P) + (a-1)*log(theta) - (a+b)*log(theta+P)
//
//  Cross-term accounting (to avoid double-counting):
//    The s conditional includes: -K*lgamma(kappa*s_i)+kappa*s_i*L_i+(theta/P-1)*log s_i.
//    kappa's conditional also has those kappa-s cross-terms, so kappa contributes
//    ONLY its unique prior terms: K*lgamma(kappa) - kappa.
//    theta's conditional also has (theta/P-1)*sum log s_i, so theta contributes
//    ONLY its normalizer and hyperprior.
//
//  joint_nuts_block adds Jacobians for each slice (SIMPLEX stick-breaking,
//  POSITIVE log) internally; NO Jacobians here.
// ----------------------------------------------------------------------------
double s_kappa_theta_joint_log_density(const arma::vec& x,
                                       const block_context& ctx,
                                       arma::vec* grad_nat) {
    const arma::vec& L  = ctx.at("L");         // length P
    const double K      = ctx.at("K")[0];
    const double a      = ctx.at("beta_a")[0];
    const double b      = ctx.at("beta_b")[0];
    const std::size_t P = L.n_elem;

    // Slice the concatenated vector.
    const double kappa  = x[P];
    const double theta  = x[P + 1];

    if (kappa <= 0.0 || theta <= 0.0) {
        if (grad_nat) grad_nat->set_size(P + 2);
        return -std::numeric_limits<double>::infinity();
    }

    const double Pd     = static_cast<double>(P);
    const double alpha  = theta / Pd;          // Dirichlet shape parameter for s

    // --- Accumulate the s-block sum and kappa/theta shared summands --------
    double sum_log_s   = 0.0;   // sum_i log s_i  (needed for theta grad)
    double sum_s_L     = 0.0;   // sum_i s_i * L_i
    double sum_lgamma_ks = 0.0; // sum_i lgamma(kappa * s_i)
    double sum_s_psi_ks  = 0.0; // sum_i s_i * psi(kappa * s_i)

    double lp_s = 0.0;
    for (std::size_t i = 0; i < P; ++i) {
        const double si = x[i];
        if (si <= 0.0) {
            if (grad_nat) grad_nat->set_size(P + 2);
            return -std::numeric_limits<double>::infinity();
        }
        const double ks  = kappa * si;
        const double lsi = std::log(si);
        const double lgks = ai4b::lgammafn(ks);
        sum_log_s     += lsi;
        sum_s_L       += si * L[i];
        sum_lgamma_ks += lgks;
        sum_s_psi_ks  += si * ai4b::digamma(ks);
        lp_s += -K * lgks + kappa * si * L[i] + (alpha - 1.0) * lsi;
    }

    // kappa unique terms: K*lgamma(kappa) - kappa
    const double lp_kappa = K * ai4b::lgammafn(kappa) - kappa;

    // theta unique terms: lgamma(theta) - P*lgamma(theta/P) + (a-1)*log(theta)
    //                     - (a+b)*log(theta+P)
    const double lp_theta =
          ai4b::lgammafn(theta)
        - Pd * ai4b::lgammafn(alpha)
        + (a - 1.0) * std::log(theta)
        - (a + b) * std::log(theta + Pd);

    const double lp = lp_s + lp_kappa + lp_theta;

    if (!std::isfinite(lp)) {
        if (grad_nat) grad_nat->set_size(P + 2);
        return -std::numeric_limits<double>::infinity();
    }

    if (grad_nat) {
        grad_nat->set_size(P + 2);
        // d/d s_i = -K*kappa*psi(kappa*s_i) + kappa*L_i + (alpha-1)/s_i
        for (std::size_t i = 0; i < P; ++i) {
            const double si = x[i];
            const double ks = kappa * si;
            (*grad_nat)[i] =
                  -K * kappa * ai4b::digamma(ks)
                + kappa * L[i]
                + (alpha - 1.0) / si;
        }
        // d/d kappa = K*psi(kappa) - K*sum_i s_i*psi(kappa*s_i) + sum_i s_i*L_i - 1
        (*grad_nat)[P] =
              K * ai4b::digamma(kappa)
            - K * sum_s_psi_ks
            + sum_s_L
            - 1.0;
        // d/d theta = psi(theta) - psi(theta/P) + (1/P)*sum_i log s_i
        //             + (a-1)/theta - (a+b)/(theta+P)
        (*grad_nat)[P + 1] =
              ai4b::digamma(theta)
            - ai4b::digamma(alpha)
            + sum_log_s / Pd
            + (a - 1.0) / theta
            - (a + b) / (theta + Pd);
    }
    return lp;
}

} // anonymous namespace

// ============================================================================
//  User-facing class exposed to R.
// ============================================================================

class DirichletHierarchical {
public:
    DirichletHierarchical(const arma::mat& S_obs,  // K x P matrix of data
                               double beta_a,
                               double beta_b,
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
          impl_(std::make_unique<composite_block>("DirichletHierarchical")),
          keep_history_(keep_history)
    {
        const int K = static_cast<int>(S_obs.n_rows);
        const int P = static_cast<int>(S_obs.n_cols);
        if (K < 1) {
            ai4b::stop("S_obs must have at least 1 row");
        }
        if (P < 2) {
            ai4b::stop("S_obs must have at least 2 columns");
        }
        if (beta_a <= 0.0 || beta_b <= 0.0) {
            ai4b::stop("beta_a, beta_b must be > 0");
        }

        // ---- Pick initial values internally --------------------------------
        const double kappa_init = 1.0;
        const double theta_init = beta_a / (beta_a + beta_b);

        // ---- Precompute sufficient statistic L_i = sum_k log s_{k,i} ------
        arma::vec L(static_cast<arma::uword>(P), arma::fill::zeros);
        for (int k = 0; k < K; ++k) {
            for (int i = 0; i < P; ++i) {
                const double sk = S_obs(k, i);
                if (sk <= 0.0) {
                    ai4b::stop("S_obs entries must be strictly positive");
                }
                L[i] += std::log(sk);
            }
        }

        // ---- Install fixed data -------------------------------------------
        impl_->data().set("L",      L);
        impl_->data().set("K",      arma::vec{static_cast<double>(K)});
        impl_->data().set("beta_a", arma::vec{beta_a});
        impl_->data().set("beta_b", arma::vec{beta_b});

        // ---- Initial s: column means of S_obs, jittered toward Dir(1) -----
        const double s_init_alpha = 1.0;
        arma::vec s_init = arma::vectorise(arma::mean(S_obs, 0));
        s_init += s_init_alpha;
        s_init /= arma::sum(s_init);

        impl_->data().set("s",     s_init);
        impl_->data().set("kappa", arma::vec{kappa_init});
        impl_->data().set("theta", arma::vec{theta_init});

        // ---- Dependency DAG -----------------------------------------------
        // Joint block dependencies keyed under the JOINT BLOCK NAME.
        // These are the union of external data() inputs that the sub-params
        // read from data() (minus the now-internal cross-reads of each other).
        // s, kappa, theta all read L and K from data(); theta also reads
        // beta_a and beta_b. The mutual reads of s/kappa/theta are now
        // internal to the joint block.
        impl_->data().declare_dependencies("s_kappa_theta_joint",
            {"L", "K", "beta_a", "beta_b"});

        // Predict edges keyed by SUB-PARAM name (unchanged).
        impl_->data().declare_predict_edges("s",     {"y_rep"});
        impl_->data().declare_predict_edges("kappa", {"y_rep"});

        // Generative-DAG context (VIZ-ONLY; predict_at BFS never reads it).
        impl_->data().declare_context_edges("theta",  {"s"});
        impl_->data().declare_context_edges("beta_a", {"theta"});
        impl_->data().declare_context_edges("beta_b", {"theta"});

        // y_rep: one new simplex observation ~ Dirichlet(kappa * s), length K.
        // (Same stochastic refresher as DirichletHierarchical.cpp.)
        impl_->data().set("y_rep", arma::vec(static_cast<arma::uword>(K),
                                              arma::fill::zeros));
        // Check #17 whitelist: std::gamma_distribution used inside
        // register_stochastic_refresher (one of the three whitelisted contexts).
        impl_->data().register_stochastic_refresher(
            "y_rep",
            [](const AI4BayesCode::shared_data_t& d,
               std::mt19937_64& rng) {
                const arma::vec& s  = d.get("s");
                const double kappa  = d.get("kappa")[0];
                const std::size_t K = s.n_elem;
                arma::vec y_rep(K);
                double sum_g = 0.0;
                for (std::size_t k = 0; k < K; ++k) {
                    const double alpha_k = std::max(kappa * s[k], 1e-10);
                    std::gamma_distribution<double> gam(alpha_k, 1.0);
                    double g = gam(rng);
                    if (g < 1e-300) g = 1e-300;
                    y_rep[k] = g;
                    sum_g   += g;
                }
                if (sum_g > 0.0) y_rep /= sum_g;
                return y_rep;
            });

        // ---- ONE joint_nuts_block over (s SIMPLEX, kappa POSITIVE,
        //       theta POSITIVE), total natural dim = P+2. -----------------
        {
            joint_nuts_block_config cfg;
            cfg.name = "s_kappa_theta_joint";
            cfg.sub_params.push_back(
                joint_nuts_sub_param{ "s",    static_cast<std::size_t>(P),
                                      joint_constraint::SIMPLEX });
            cfg.sub_params.push_back(
                joint_nuts_sub_param{ "kappa", 1, joint_constraint::POSITIVE });
            cfg.sub_params.push_back(
                joint_nuts_sub_param{ "theta", 1, joint_constraint::POSITIVE });
            // initial_cat is NATURAL-scale: [s_init (P), kappa_init, theta_init].
            arma::vec init_cat(static_cast<arma::uword>(P) + 2);
            for (int i = 0; i < P; ++i) init_cat[static_cast<arma::uword>(i)] = s_init[i];
            init_cat[static_cast<arma::uword>(P)]     = kappa_init;
            init_cat[static_cast<arma::uword>(P) + 1] = theta_init;
            cfg.initial_cat = init_cat;
            cfg.log_density_grad = &s_kappa_theta_joint_log_density;
            // Simplex + two positive params at very different scales; diagonal
            // metric adapts independently per dimension.
            cfg.use_diagonal_metric = true;
            // Give the joint block more warmup runway — adapting over (P+2)
            // dimensions with heterogeneous scales (simplex vs. positive)
            // needs more trajectory exploration than a single-param block.
            cfg.n_warmup_first_call = 800;
            impl_->add_child(std::make_unique<joint_nuts_block>(std::move(cfg)));
        }

        if (keep_history_) {
            impl_->set_keep_history(true);
        }
    }

    void step(int n_steps) {
        for (int i = 0; i < n_steps; ++i) impl_->step(rng_);
    }

    AI4BayesCode::state_map get_current() const {
        AI4BayesCode::state_map out;
        out["s"]     = impl_->data().get("s");        // length-P simplex vector
        out["kappa"] = impl_->data().get("kappa");    // length-1 arma::vec
        out["theta"] = impl_->data().get("theta");    // length-1 arma::vec
        return out;
    }

    void set_current(const AI4BayesCode::state_map& params) {
        auto& jblk = dynamic_cast<joint_nuts_block&>(impl_->child(0));
        const std::size_t P = jblk.sub_param_dim("s");
        arma::vec cat_new = jblk.current();   // [s(0)...s(P-1), kappa, theta]
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
        auto it_kappa = params.find("kappa");
        if (it_kappa != params.end()) {
            const double k_new = it_kappa->second[0];
            if (k_new <= 0.0) ai4b::stop("kappa must be > 0");
            cat_new[P] = k_new;
            touched = true;
        }
        auto it_theta = params.find("theta");
        if (it_theta != params.end()) {
            const double th_new = it_theta->second[0];
            if (th_new <= 0.0) ai4b::stop("theta must be > 0");
            cat_new[P + 1] = th_new;
            touched = true;
        }
        if (touched) {
            jblk.set_current(cat_new);
            impl_->data().set("s",     cat_new.subvec(0, P - 1));
            impl_->data().set("kappa", arma::vec{cat_new[P]});
            impl_->data().set("theta", arma::vec{cat_new[P + 1]});
        }
    }

    // Posterior-predictive: y_rep ~ Dirichlet(kappa * s), one new simplex obs.
    AI4BayesCode::history_map predict_at(const AI4BayesCode::state_map& new_data) const {
        if (!new_data.empty()) {
            ai4b::stop("DirichletHierarchical has no covariate inputs. "
                       "predict_at() takes an empty map/list.");
        }
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

        // History mode: s and kappa are sub-outputs of the joint block (keyed
        // by sub-param name in get_history()). Compute y_rep manually per
        // draw (cf. IRT1PL_joint.cpp — composite predict_at rejects sub-param
        // keys as replaced context for joint sub-outputs).
        AI4BayesCode::history_map hist = impl_->get_history();
        const arma::mat& s_hist     = hist.at("s");      // n_draws x P
        const arma::mat& kappa_hist = hist.at("kappa");  // n_draws x 1
        const std::size_t n_draws = s_hist.n_rows;
        const std::size_t P       = s_hist.n_cols;

        // y_rep has length K = P (same simplex dimensionality as the observations)
        const std::size_t K_rep = P;
        arma::mat yrep(n_draws, K_rep);

        for (std::size_t d = 0; d < n_draws; ++d) {
            const double kappa = kappa_hist(d, 0);
            double sum_g = 0.0;
            for (std::size_t k = 0; k < K_rep; ++k) {
                const double alpha_k = std::max(kappa * s_hist(d, k), 1e-10);
                std::gamma_distribution<double> gam(alpha_k, 1.0);
                double g = gam(predict_rng_);
                if (g < 1e-300) g = 1e-300;
                yrep(d, k) = g;
                sum_g += g;
            }
            if (sum_g > 0.0) {
                for (std::size_t k = 0; k < K_rep; ++k)
                    yrep(d, k) /= sum_g;
            }
        }

        out.emplace("y_rep", std::move(yrep));
        return out;
    }

    AI4BayesCode::dag_info get_dag() const { return impl_->get_dag(); }

    AI4BayesCode::history_map get_history() const { return impl_->get_history(); }

    /// Re-tune NUTS metric (mass matrix + step size + dual averaging) without
    /// advancing chain state.
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
    mutable std::mt19937_64          readapt_rng_;
    std::unique_ptr<composite_block> impl_;
    bool                             keep_history_ = false;
};

// ============================================================================
//  Rcpp module
// ============================================================================

#ifdef AI4BAYESCODE_RCPP_MODULE
RCPP_MODULE(DirichletHierarchical_module) {
    Rcpp::class_<DirichletHierarchical>("DirichletHierarchical")
        .constructor<arma::mat, double, double, int>(
            "Legacy constructor; keep_history defaults to FALSE.")
        .constructor<arma::mat, double, double, int, bool>(
            "Joint-NUTS DirichletHierarchical: (s, kappa, theta) in one "
            "joint_nuts_block (SIMPLEX + POSITIVE + POSITIVE). Construct with "
            "S_obs (K x P), Beta hyperprior shapes (beta_a, beta_b), RNG seed "
            "(0 = random), and keep_history (default FALSE).")
        .method("step",        &DirichletHierarchical::step,
                "Run n sweeps (each: one JOINT NUTS update of (s, kappa, theta)).")
        .method("get_current", &DirichletHierarchical::get_current,
                "Return current draw as a named list.")
        .method("set_current", &DirichletHierarchical::set_current,
                "Overwrite s, kappa, or theta from a named list.")
        .method("predict_at",  &DirichletHierarchical::predict_at,
                "No covariate inputs; takes empty list, returns y_rep.")
        .method("get_dag",     &DirichletHierarchical::get_dag,
                "Return the predict DAG.")
        .method("get_history", &DirichletHierarchical::get_history,
                "History of [s, kappa, theta] draws.")
        .method("readapt_NUTS", &DirichletHierarchical::readapt_NUTS);
}
#endif

#ifdef AI4BAYESCODE_PYBIND_MODULE
#include "AI4BayesCode/pybind_casters.hpp"
PYBIND11_MODULE(DirichletHierarchical, m) {
    AI4BayesCode::register_ai4bayescode_types(m);
    pybind11::class_<DirichletHierarchical>(m, "DirichletHierarchical")
        .def(pybind11::init<arma::mat, double, double, int, bool>(),
             pybind11::arg("S_obs"),
             pybind11::arg("beta_a") = 0.5,
             pybind11::arg("beta_b") = 1.0,
             pybind11::arg("rng_seed") = 1,
             pybind11::arg("keep_history") = false)
        .def("step",         &DirichletHierarchical::step,    pybind11::arg("n_steps"))
        .def("get_current",  &DirichletHierarchical::get_current)
        .def("set_current",  &DirichletHierarchical::set_current, pybind11::arg("params"))
        .def("predict_at",   &DirichletHierarchical::predict_at,  pybind11::arg("new_data"))
        .def("get_dag",      &DirichletHierarchical::get_dag)
        .def("get_history",  &DirichletHierarchical::get_history)
        .def("readapt_NUTS", &DirichletHierarchical::readapt_NUTS,
             pybind11::arg("n"), pybind11::arg("reset") = false);
}
#endif
