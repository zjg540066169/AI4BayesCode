// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v3.0 or later
// (GPL-3.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
//  HDPGaussianMixture.cpp
//
//  Hierarchical Dirichlet Process Gaussian mixture in TRUNCATED form
//  (Wang-Paisley-Blei 2011 simplified architecture, after Teh et al.
//  2006). Demonstrates COMPOSITION of multiple shipped blocks
//  (`stick_breaking_block` + G × `dirichlet_gibbs_block` +
//  `niw_cluster_gibbs_block` + `categorical_gibbs_block`) into a
//  hierarchical clustering model that shares atoms across groups.
//
//  IMPORTANT: V0 SIMPLIFICATION
//  ============================
//  The β top-level update implemented here uses a HEURISTIC: it treats
//  β | counts as a stick-breaking conditional with a_fn = 1 + Σ_j n_jt
//  and b_fn = γ + Σ_j (Σ_{s>t} n_js). This is NOT the rigorous HDP
//  posterior. The full HDP requires per-(j, t) auxiliary "table counts"
//  m_jt sampled from the Antoniak distribution (Teh et al. 2006 §5.1),
//  with β | m_·1, ..., m_·T ~ Dirichlet(γ/T + m_·1, ..., γ/T + m_·T).
//
//  For a v0 demonstration of the COMPOSITION PATTERN, the heuristic
//  works correctly on simple group-clustering fixtures and matches the
//  posterior intuitively (more populated atoms get larger β). For
//  rigorous HDP inference, future work needs an `antoniak_aug_block`
//  that samples table counts; with that, the existing β block can be
//  re-pointed to read m_·t instead of n_·t.
//
//  This caveat is documented prominently here (in the header comment),
//  in `skills/block_catalogue/index.md` (HDP example summary), and in
//  `BNP_AUDIT_STATUS.md`. Users who need rigorous HDP should look at
//  BayesMix (Beraha et al. 2022, arXiv 2205.08144) for a reference
//  implementation.
//
//  MODEL
//  -----
//      β ~ truncated stick-breaking with concentration γ
//      π_j | β, α ~ Dirichlet(α · β_1, ..., α · β_T)   for j = 1, ..., G
//      z_{ji} | π_j ~ Categorical(π_j)
//      y_{ji} | z_{ji} = t ~ N(μ_t, Σ_t)               (atoms shared)
//      (μ_t, Σ_t) ~ NIW(μ_0, κ_0, Ψ_0, ν_0)
//      α, γ : FIXED at construction (v0; future v0.5 sample via
//             gamma_gibbs_block).
//
//  BLOCK DECOMPOSITION (Gibbs sweep order)
//  ---------------------------------------
//      child(0)             z              categorical_gibbs_block
//      child(1)             cluster_params niw_cluster_gibbs_block
//      child(2)             β              stick_breaking_block (heuristic)
//      child(3..2+G)        π_j (j = 1..G) dirichlet_gibbs_block (one per group)
//
//  LABEL SWITCHING
//  ---------------
//  DEFAULT / RECOMMENDED: resolve label switching POST-MCMC on the recorded
//  draws (skills/label_switching.md). The truncated-HDP likelihood is invariant
//  under permutation of the T GLOBAL atoms (shared mu_t/Sigma_t, the global
//  weight beta_t, and every group's pi_{g,t}); the label-INVARIANT summaries
//  (K_active, sorted occupied-atom proportions / mu / beta order statistics)
//  converge as-is (2-chain rank-R-hat ~1.0). The two concentrations gamma_0 and
//  alpha are FIXED constants in this v0 model (not sampled), hence trivially
//  label-invariant — there is NO concentration-mixing problem to fix (unlike the
//  DP example where alpha is sampled by NUTS on the Antoniak (k,n) marginal).
//
//  THIS SAMPLER STAYS RAW — no in-sampler canonicalizer. The raw per-slot
//  beta[t] / mu[t] / pi_{g,t} are exchangeable across the T atoms and not
//  identified, so their raw R-hat looks high (measured ~1.83 on raw beta[t]) —
//  that is benign label switching, not a mixing failure. The label-invariant
//  summaries (K_active, sorted occupied-atom proportions / mu / beta order
//  statistics) converge as-is (~1.0). Resolve labelling POST-MCMC via
//  ai4bayescode_diagnose(..., order_components = TRUE), which reports
//  order-statistic R-hat on the sorted components. An in-sampler ordering /
//  canonicalizer is a discouraged fallback (skills/label_switching.md) and is
//  deliberately NOT used here.
//
//  REFRESHERS
//  ----------
//      counts_jt   register_refresher (deterministic). Group×Atom count
//                  matrix derived from (z, group_idx). Stored flat as
//                  G*T row-major in shared_data["counts_jt"].
//      counts_t    register_refresher (deterministic). Top-level
//                  marginal: counts_t = Σ_j counts_jt. Length T.
//      y_rep       register_stochastic_refresher (predict-time).
//
//  STORAGE CONVENTIONS
//  -------------------
//  - y         length N * d, row-major
//  - group_idx length N, integer 0..G-1 (stored as double)
//  - z         length N, integer 1..T (1-indexed)
//  - β         length T (T = K_trunc)
//  - stick_V_β length T (output of β stick_breaking_block via v_name)
//  - π_g       length T, exposed under shared_data key "pi_<g>" for
//              g = 0, 1, ..., G-1 (no aggregated group-major flat key)
//  - μ         length T * d, cluster-major (atoms shared)
//  - σ         length T * d * d, cluster-major row-major (atoms shared)
//  - counts_jt length G * T, group-major row-major
//  - counts_t  length T
//
//  JUSTIFICATION (Check #16):
//  - z         categorical_gibbs_block. Class-1 conditional independence
//              given (π_j, μ, Σ).
//  - π_g       dirichlet_gibbs_block. Conjugate on the simplex.
//  - β         stick_breaking_block. Heuristic update; documented
//              approximation.
//  - cluster_params  niw_cluster_gibbs_block. Conjugate.
//
//  No hand-written log-density (no NUTS); all 3 + G sampling children are
//  conjugate Gibbs / MH-deterministic.
//  Check #12 vacuous.
//
// @example:R
//   library(AI4BayesCode)
//   ai4bayescode_example("HDPGaussianMixture")
//   set.seed(2026)
//   ## Two groups (G=2) sharing T=6 truncated atoms in 2-D. Three well-
//   ## separated true clusters; groups differ in mixing weights.
//   d <- 2L; G <- 2L; Ng <- 120L
//   centers <- rbind(c(-4, -4), c(4, 4), c(0, 6))   # 3 atoms
//   draw_group <- function(g) {
//     w <- if (g == 0L) c(0.6, 0.3, 0.1) else c(0.1, 0.4, 0.5)
//     z <- sample(1:3, Ng, replace = TRUE, prob = w)
//     centers[z, ] + matrix(rnorm(Ng * d, sd = 0.6), Ng, d)
//   }
//   y <- rbind(draw_group(0L), draw_group(1L))
//   group_idx <- c(rep(0L, Ng), rep(1L, Ng))
//   # ---- Recommended: parallel chains + convergence diagnosis ----
//   run <- ai4bayescode_run_chains(
//       function(seed) new(HDPGaussianMixture, y, group_idx, 6L, c(0, 0), 0.1, diag(d), 4.0, 1.0, 1.0, seed, TRUE),
//       n_chains = 4, n_burn = 1000, n_keep = 2000)
//   ai4bayescode_diagnose(run$histories[[1]])      # summary + R-hat/ESS + plots
//   # ---- Advanced: stateful single-chain control ----
//   m <- new(HDPGaussianMixture,
//            y, group_idx, 6L,                 # y (N x d), group_idx, K_trunc
//            c(0, 0), 0.1,                      # mu_0, kappa_0
//            diag(d), 4.0,                      # Psi_0, nu_0
//            1.0, 1.0,                          # alpha, gamma_0
//            7L, FALSE)                         # rng_seed, keep_history
//   m$step(800); cur <- m$get_current()
//   ## mu is a flat length-(T*d) cluster-major vector; reshape byrow=TRUE.
//   mu  <- matrix(cur$mu, nrow = 6L, ncol = d, byrow = TRUE)
//   ord <- order(cur$beta, decreasing = TRUE)
//   ## occupied atoms recover the 3 true centers (up to label permutation)
//   print(round(mu[ord, ][1:3, ], 1))
// @example:python
//   import numpy as np, AI4BayesCode
//   rng = np.random.default_rng(2026)
//   d, G, Ng = 2, 2, 120
//   centers = np.array([[-4.0, -4.0], [4.0, 4.0], [0.0, 6.0]])  # 3 atoms
//   def draw_group(g):
//       w = np.array([0.6, 0.3, 0.1]) if g == 0 else np.array([0.1, 0.4, 0.5])
//       z = rng.choice(3, size=Ng, p=w)
//       return centers[z] + rng.normal(0.0, 0.6, size=(Ng, d))
//   y = np.vstack([draw_group(0), draw_group(1)])
//   group_idx = np.concatenate([np.zeros(Ng), np.ones(Ng)])   # 0/1 group labels
//   Mod = AI4BayesCode.example("HDPGaussianMixture")
//   # ---- Recommended: parallel chains + diagnosis ----
//   chains = AI4BayesCode.run_chains(
//       lambda seed: Mod.HDPGaussianMixture(y, group_idx, 6, np.zeros(d), 0.1, np.eye(d), 4.0, 1.0, 1.0, seed, True),
//       seeds=[101, 202, 303, 404], n_burn=1000, n_keep=2000, n_jobs=1)
//   AI4BayesCode.diagnose(chains[0]["hist"])   # summary + diagnostics
//   # ---- Advanced: stateful single-chain control ----
//   m = Mod.HDPGaussianMixture(y, group_idx, 6, np.zeros(d), 0.1,
//                              np.eye(d), 4.0, 1.0, 1.0, 7, False)
//   m.step(800); cur = m.get_current()
//   mu = cur["mu"].reshape(6, d); beta = cur["beta"]
//   print(np.round(mu[np.argsort(beta)[::-1]][:3], 1))  # ~ the 3 true centers
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
#include "AI4BayesCode/rcpp_wrap.hpp"

#include "AI4BayesCode/categorical_gibbs_block.hpp"
#include "AI4BayesCode/dirichlet_gibbs_block.hpp"
#include "AI4BayesCode/niw_cluster_gibbs_block.hpp"
#include "AI4BayesCode/stick_breaking_block.hpp"
#include "AI4BayesCode/bnp_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <vector>

using AI4BayesCode::block_context;
using AI4BayesCode::composite_block;
using AI4BayesCode::categorical_gibbs_block;
using AI4BayesCode::categorical_gibbs_block_config;
using AI4BayesCode::dirichlet_gibbs_block;
using AI4BayesCode::dirichlet_gibbs_block_config;
using AI4BayesCode::niw_cluster_gibbs_block;
using AI4BayesCode::niw_cluster_gibbs_block_config;
using AI4BayesCode::stick_breaking_block;
using AI4BayesCode::stick_breaking_block_config;
namespace bnp = AI4BayesCode::bnp;

namespace {

inline std::string pi_key_for_group(std::size_t g) {
    std::ostringstream oss;
    oss << "pi_" << g;
    return oss.str();
}

inline double full_normal_log_density(const double* y, const double* mu,
                                       const arma::mat& L,
                                       std::size_t d) {
    constexpr double kLog2Pi = 1.83787706640934548356065947281;
    arma::vec dev(d);
    for (std::size_t j = 0; j < d; ++j) dev[j] = y[j] - mu[j];
    double log_det = 0.0;
    for (std::size_t a = 0; a < d; ++a) log_det += 2.0 * std::log(L(a, a));
    arma::vec u = arma::solve(arma::trimatl(L), dev);
    return -0.5 * static_cast<double>(d) * kLog2Pi
         - 0.5 * log_det - 0.5 * arma::dot(u, u);
}


}  // anonymous namespace

class HDPGaussianMixture {
public:
    HDPGaussianMixture(const arma::mat& y,
                       const arma::vec& group_idx,  // length N, values 0..G-1
                       int K_trunc,
                       const arma::vec& mu_0,
                       double kappa_0,
                       const arma::mat& Psi_0,
                       double nu_0,
                       double alpha,    // group-level concentration
                       double gamma_0,  // top-level concentration
                       int rng_seed,
                       bool keep_history = false)
        : rng_(rng_seed == 0
                   ? std::mt19937_64{std::random_device{}()}
                   : std::mt19937_64{static_cast<std::uint64_t>(rng_seed)}),
          predict_rng_(rng_seed == 0
                   ? std::mt19937_64{std::random_device{}()}
                   : std::mt19937_64{static_cast<std::uint64_t>(rng_seed)
                                     ^ 0x9E3779B97F4A7C15ULL}),
          impl_(std::make_unique<composite_block>("HDPGaussianMixture")),
          keep_history_(keep_history)
    {
        if (y.n_rows < 2)
            ai4b::stop("HDPGaussianMixture: N must be >= 2");
        if (y.n_cols < 1)
            ai4b::stop("HDPGaussianMixture: d must be >= 1");
        if (group_idx.n_elem != y.n_rows)
            ai4b::stop("HDPGaussianMixture: group_idx length must equal N");
        if (K_trunc < 2)
            ai4b::stop("HDPGaussianMixture: K_trunc must be >= 2");
        if (mu_0.n_elem != y.n_cols)
            ai4b::stop("HDPGaussianMixture: mu_0 length must equal d");
        if (Psi_0.n_rows != y.n_cols || Psi_0.n_cols != y.n_cols)
            ai4b::stop("HDPGaussianMixture: Psi_0 must be d x d");
        if (!(kappa_0 > 0.0)) ai4b::stop("kappa_0 must be > 0");
        if (!(nu_0 > static_cast<double>(y.n_cols) - 1.0))
            ai4b::stop("nu_0 must be > d - 1");
        if (!(alpha > 0.0)) ai4b::stop("alpha must be > 0");
        if (!(gamma_0 > 0.0)) ai4b::stop("gamma_0 must be > 0");

        N_ = static_cast<std::size_t>(y.n_rows);
        d_ = static_cast<std::size_t>(y.n_cols);
        T_ = static_cast<std::size_t>(K_trunc);

        // Determine G from group_idx (stored as double; values are integer
        // group labels 0..G-1).
        int g_max = 0;
        for (std::size_t i = 0; i < N_; ++i) {
            const long g = static_cast<long>(std::llround(group_idx[i]));
            if (g < 0)
                ai4b::stop("HDPGaussianMixture: group_idx must be non-negative");
            if (g > g_max) g_max = static_cast<int>(g);
        }
        G_ = static_cast<std::size_t>(g_max + 1);
        if (G_ < 1) ai4b::stop("HDPGaussianMixture: must have >= 1 group");

        // ---- Data + priors ------------------------------------------
        arma::vec y_flat(N_ * d_);
        for (std::size_t i = 0; i < N_; ++i)
            for (std::size_t j = 0; j < d_; ++j)
                y_flat[i * d_ + j] = y(i, j);
        impl_->data().set("y", y_flat);

        arma::vec g_arma(N_);
        for (std::size_t i = 0; i < N_; ++i)
            g_arma[i] = static_cast<double>(
                std::llround(group_idx[i]));
        impl_->data().set("group_idx", g_arma);

        arma::vec mu0_arma(d_);
        for (std::size_t j = 0; j < d_; ++j) mu0_arma[j] = mu_0[j];
        impl_->data().set("mu_0", mu0_arma);
        impl_->data().set("kappa_0",     arma::vec{kappa_0});
        impl_->data().set("nu_0",        arma::vec{nu_0});
        impl_->data().set("alpha",       arma::vec{alpha});
        impl_->data().set("gamma_0",     arma::vec{gamma_0});

        arma::vec psi0_flat(d_ * d_);
        for (std::size_t i = 0; i < d_; ++i)
            for (std::size_t j = 0; j < d_; ++j)
                psi0_flat[i * d_ + j] = Psi_0(i, j);
        impl_->data().set("Psi_0", psi0_flat);

        // ---- Initial state -----------------------------------------
        // z: spread (i mod T) + 1.
        arma::vec z_init(N_);
        for (std::size_t i = 0; i < N_; ++i)
            z_init[i] = static_cast<double>((i % T_) + 1);
        impl_->data().set("z", z_init);

        // β: uniform 1/T.
        arma::vec beta_init(T_, arma::fill::value(1.0 / static_cast<double>(T_)));
        impl_->data().set("beta", beta_init);
        // Stick fractions (output of stick_breaking_block via v_name).
        arma::vec V_beta_init(T_, arma::fill::zeros);
        {
            double rem = 1.0;
            for (std::size_t k = 0; k + 1 < T_; ++k) {
                V_beta_init[k] = beta_init[k] / rem;
                if (V_beta_init[k] > 1.0) V_beta_init[k] = 1.0;
                rem *= (1.0 - V_beta_init[k]);
            }
            V_beta_init[T_ - 1] = 1.0;
        }
        impl_->data().set("stick_V_beta", V_beta_init);

        // π_g for g = 0..G-1: each uniform 1/T.
        for (std::size_t g = 0; g < G_; ++g) {
            const std::string key = pi_key_for_group(g);
            impl_->data().set(key,
                arma::vec(T_, arma::fill::value(1.0 / static_cast<double>(T_))));
        }

        // μ, Σ atoms: data-driven init for μ; Σ = identity.
        arma::vec mu_init(T_ * d_, arma::fill::zeros);
        for (std::size_t k = 0; k < T_; ++k) {
            const std::size_t i_anchor = (k * N_) / T_;
            for (std::size_t j = 0; j < d_; ++j)
                mu_init[k * d_ + j] = y(i_anchor, j);
        }
        arma::vec sigma_init(T_ * d_ * d_, arma::fill::zeros);
        for (std::size_t k = 0; k < T_; ++k)
            for (std::size_t i = 0; i < d_; ++i)
                sigma_init[k * d_ * d_ + i * d_ + i] = 1.0;
        impl_->data().set("mu", mu_init);
        impl_->data().set("sigma", sigma_init);

        // counts_jt + counts_t: derived (refreshers).
        impl_->data().set("counts_jt", arma::vec(G_ * T_, arma::fill::zeros));
        impl_->data().set("counts_t",  arma::vec(T_,      arma::fill::zeros));
        impl_->data().register_refresher("counts_jt",
            [G = G_, T = T_, N = N_](const AI4BayesCode::shared_data_t& d)
                -> arma::vec {
                const arma::vec& z = d.get("z");
                const arma::vec& g_idx = d.get("group_idx");
                arma::vec out(G * T, arma::fill::zeros);
                for (std::size_t i = 0; i < N; ++i) {
                    const long lab = static_cast<long>(std::llround(z[i]));
                    if (lab < 1 || static_cast<std::size_t>(lab) > T) continue;
                    const std::size_t t = static_cast<std::size_t>(lab) - 1;
                    const std::size_t g = static_cast<std::size_t>(
                        std::llround(g_idx[i]));
                    if (g >= G) continue;
                    out[g * T + t] += 1.0;
                }
                return out;
            });
        impl_->data().register_refresher("counts_t",
            [G = G_, T = T_](const AI4BayesCode::shared_data_t& d)
                -> arma::vec {
                const arma::vec& cjt = d.get("counts_jt");
                arma::vec out(T, arma::fill::zeros);
                for (std::size_t g = 0; g < G; ++g)
                    for (std::size_t t = 0; t < T; ++t)
                        out[t] += cjt[g * T + t];
                return out;
            });

        // ---- Gibbs DAG dependencies / invalidations ----------------
        std::vector<std::string> z_reads = {"y", "group_idx", "mu", "sigma"};
        for (std::size_t g = 0; g < G_; ++g)
            z_reads.push_back(pi_key_for_group(g));
        impl_->data().declare_dependencies("z", z_reads);

        impl_->data().declare_dependencies("cluster_params",
            {"z", "y", "Psi_0", "kappa_0", "nu_0", "mu_0"});
        impl_->data().declare_dependencies("beta",
            {"counts_t", "gamma_0"});
        for (std::size_t g = 0; g < G_; ++g) {
            impl_->data().declare_dependencies(
                "pi_" + std::to_string(g),
                {"counts_jt", "beta", "alpha"});
        }

        impl_->data().declare_invalidates("z", {"counts_jt", "counts_t"});

        // ---- Predict DAG --------------------------------------------
        // (No declare_data_input here — y is an observed terminal,
        // not a replaceable covariate. The y_rep refresher reads
        // pi/mu/sigma/z, NOT y.)
        std::vector<std::string> edges_to_yrep =
            {"mu", "sigma", "z"};
        for (std::size_t g = 0; g < G_; ++g)
            edges_to_yrep.push_back(pi_key_for_group(g));
        for (const auto& src : edges_to_yrep)
            impl_->data().declare_predict_edges(src, {"y_rep"});

        // ---- Generative-DAG context (VIZ-ONLY; predict_at BFS never
        //      reads context_edges_). Truncated HDP:
        //        beta | m ~ Dir(gamma_0/T + m)  (top-level sticks);
        //        pi_j | beta, alpha ~ Dir(alpha * beta)  per group j;
        //        (mu_k, Sigma_k) ~ NIW(mu_0, kappa_0, Psi_0, nu_0).
        //      Drawn faded by ai4bayescode_plot_dag.
        std::vector<std::string> pi_keys;
        for (std::size_t g = 0; g < G_; ++g)
            pi_keys.push_back(pi_key_for_group(g));
        impl_->data().declare_context_edges("gamma_0",     {"stick_V_beta"});
        impl_->data().declare_context_edges("stick_V_beta",{"beta"});
        impl_->data().declare_context_edges("beta",        pi_keys);
        impl_->data().declare_context_edges("alpha",       pi_keys);
        impl_->data().declare_context_edges("mu_0",        {"mu"});
        impl_->data().declare_context_edges("kappa_0",     {"mu"});
        impl_->data().declare_context_edges("Psi_0",       {"sigma"});
        impl_->data().declare_context_edges("nu_0",        {"sigma"});

        impl_->data().set("y_rep", arma::vec(N_ * d_, arma::fill::zeros));
        impl_->data().register_stochastic_refresher("y_rep",
            [G = G_, N = N_, d = d_, T = T_](
                const AI4BayesCode::shared_data_t& dat,
                std::mt19937_64& rng) -> arma::vec {
                const arma::vec& g_idx = dat.get("group_idx");
                const arma::vec& mu  = dat.get("mu");
                const arma::vec& sig = dat.get("sigma");
                std::vector<arma::vec> pi_vec;
                pi_vec.reserve(G);
                for (std::size_t g = 0; g < G; ++g)
                    pi_vec.push_back(dat.get(pi_key_for_group(g)));
                std::uniform_real_distribution<double> uniform(0.0, 1.0);
                std::normal_distribution<double> stdnorm(0.0, 1.0);
                arma::vec out(N * d);
                for (std::size_t i = 0; i < N; ++i) {
                    const std::size_t g = static_cast<std::size_t>(
                        std::llround(g_idx[i]));
                    const arma::vec& pi_g = pi_vec[g];
                    // Sample atom z_rep_i ~ Cat(pi_g)
                    const double u = uniform(rng);
                    double cumul = 0.0;
                    std::size_t z_i = T - 1;
                    for (std::size_t t = 0; t < T; ++t) {
                        cumul += pi_g[t];
                        if (u < cumul) { z_i = t; break; }
                    }
                    // Sample y_rep_i ~ N(mu_z, sigma_z)
                    arma::mat S(d, d);
                    for (std::size_t a = 0; a < d; ++a)
                        for (std::size_t b = 0; b < d; ++b)
                            S(a, b) = sig[z_i * d * d + a * d + b];
                    arma::mat L;
                    if (!arma::chol(L, S, "lower")) {
                        S.diag() += 1e-8;
                        arma::chol(L, S, "lower");
                    }
                    arma::vec eps(d);
                    for (std::size_t j = 0; j < d; ++j) eps[j] = stdnorm(rng);
                    arma::vec L_eps = L * eps;  // evaluate before subscript
                    for (std::size_t j = 0; j < d; ++j)
                        out[i * d + j] = mu[z_i * d + j] + L_eps[j];
                }
                return out;
            });

        impl_->data().refresh_all();

        // ---- Children ---------------------------------------------
        // child(0) z (categorical_gibbs_block)
        {
            categorical_gibbs_block_config cfg;
            cfg.name = "z";
            cfg.n_obs = N_;
            cfg.n_categories = T_;
            cfg.initial_labels = z_init;
            const std::size_t d_capture = d_;
            const std::size_t N_capture = N_;
            const std::size_t T_capture = T_;
            const std::size_t G_capture = G_;
            cfg.log_probs_fn = [d_capture, N_capture, T_capture, G_capture]
                (const block_context& ctx) -> arma::mat {
                const arma::vec& y_flat = ctx.at("y");
                const arma::vec& g_idx  = ctx.at("group_idx");
                const arma::vec& mu     = ctx.at("mu");
                const arma::vec& sig    = ctx.at("sigma");
                std::vector<const arma::vec*> pi_ptrs(G_capture, nullptr);
                for (std::size_t g = 0; g < G_capture; ++g) {
                    pi_ptrs[g] = &ctx.at(pi_key_for_group(g));
                }
                // Pre-compute chol(Sigma_t) for each atom.
                std::vector<arma::mat> L_atoms(T_capture);
                for (std::size_t k = 0; k < T_capture; ++k) {
                    arma::mat S(d_capture, d_capture);
                    for (std::size_t a = 0; a < d_capture; ++a)
                        for (std::size_t b = 0; b < d_capture; ++b)
                            S(a, b) = sig[k * d_capture * d_capture
                                        + a * d_capture + b];
                    if (!arma::chol(L_atoms[k], S, "lower")) {
                        S.diag() += 1e-8;
                        arma::chol(L_atoms[k], S, "lower");
                    }
                }
                arma::mat lp(N_capture, T_capture);
                for (std::size_t i = 0; i < N_capture; ++i) {
                    const std::size_t g = static_cast<std::size_t>(
                        std::llround(g_idx[i]));
                    const arma::vec& pi_g = *pi_ptrs[g];
                    for (std::size_t t = 0; t < T_capture; ++t) {
                        const double log_pi_t =
                            std::log(pi_g[t] + 1e-300);
                        lp(i, t) = log_pi_t
                                 + full_normal_log_density(
                                     y_flat.memptr() + i * d_capture,
                                     mu.memptr()     + t * d_capture,
                                     L_atoms[t], d_capture);
                    }
                }
                return lp;
            };
            impl_->add_child(
                std::make_unique<categorical_gibbs_block>(std::move(cfg)));
        }

        // child(1) cluster_params (niw_cluster_gibbs_block)
        {
            niw_cluster_gibbs_block_config cfg;
            cfg.name = "cluster_params";
            cfg.K_trunc = T_;
            cfg.d = d_;
            cfg.N = N_;
            cfg.z_key = "z"; cfg.y_key = "y";
            cfg.mu_name = "mu"; cfg.sigma_name = "sigma";
            cfg.mu_0 = mu0_arma;
            cfg.kappa_0 = kappa_0;
            cfg.Psi_0_flat = psi0_flat;
            cfg.nu_0 = nu_0;
            cfg.initial_mu = mu_init;
            cfg.initial_sigma = sigma_init;
            impl_->add_child(
                std::make_unique<niw_cluster_gibbs_block>(std::move(cfg)));
        }

        // child(2) β (stick_breaking_block, heuristic update on counts_t)
        {
            stick_breaking_block_config cfg;
            cfg.name = "beta";
            cfg.K_trunc = T_;
            cfg.counts_key = "counts_t";
            cfg.v_name = "stick_V_beta";
            cfg.a_fn = [](std::size_t k, const arma::vec& counts,
                          const block_context& /*ctx*/) -> double {
                return 1.0 + counts[k];
            };
            cfg.b_fn = [](std::size_t k, const arma::vec& counts,
                          const block_context& ctx) -> double {
                const double g = ctx.at("gamma_0")[0];
                double tail = 0.0;
                for (std::size_t j = k + 1; j < counts.n_elem; ++j)
                    tail += counts[j];
                return g + tail;
            };
            cfg.initial_pi = beta_init;
            impl_->add_child(
                std::make_unique<stick_breaking_block>(std::move(cfg)));
        }

        // child(3..3+G-1) π_g (one dirichlet_gibbs_block per group)
        for (std::size_t g = 0; g < G_; ++g) {
            dirichlet_gibbs_block_config cfg;
            cfg.name = pi_key_for_group(g);
            cfg.n_categories = T_;
            cfg.initial_values =
                arma::vec(T_, arma::fill::value(1.0 / static_cast<double>(T_)));
            const std::size_t g_capture = g;
            const std::size_t T_capture = T_;
            cfg.alpha_post_fn = [g_capture, T_capture]
                (const block_context& ctx) -> arma::vec {
                const arma::vec& cjt   = ctx.at("counts_jt");
                const arma::vec& beta  = ctx.at("beta");
                const double a         = ctx.at("alpha")[0];
                arma::vec out(T_capture);
                for (std::size_t t = 0; t < T_capture; ++t) {
                    out[t] = a * beta[t] + cjt[g_capture * T_capture + t];
                }
                return out;
            };
            impl_->add_child(
                std::make_unique<dirichlet_gibbs_block>(std::move(cfg)));
        }

        if (keep_history_) impl_->set_keep_history(true);
    }

    // ---- Six-method R interface ----

    void step() { step(1); }              // no-arg convenience: one sweep
    void step(int n_steps) {
        if (n_steps < 0) ai4b::stop("n_steps must be >= 0");
        for (int i = 0; i < n_steps; ++i) impl_->step(rng_);
    }

    // Backend-neutral current draw. Every entry is a FLAT arma::vec (the
    // state_map value type). Matrix-shaped parameters are flattened ROW-MAJOR
    // so the R/Python @example code can `matrix(.., byrow=TRUE)` / numpy
    // `.reshape(rows, cols)` (C-order) back to (atom/group)-major rows:
    //   - mu         : length T*d, cluster-major  (mu[k*d + j])   -> reshape(T, d)
    //   - sigma_flat : length T*d*d, cluster-major, row-major within each dxd
    //   - pi         : length G*T, group-major     (pi[g*T + t])  -> reshape(G, T)
    //   - z, beta    : as-is (length N, length T)
    //   - alpha, gamma_0, K_trunc, G : 1-element vectors (scalars).
    AI4BayesCode::state_map get_current() const {
        AI4BayesCode::state_map out;

        out.emplace("z",    impl_->data().get("z"));
        out.emplace("beta", impl_->data().get("beta"));

        // mu: cluster-major flat (already stored that way), length T*d.
        out.emplace("mu",    impl_->data().get("mu"));
        // sigma: cluster-major, row-major within each dxd block (as stored).
        out.emplace("sigma_flat", impl_->data().get("sigma"));

        // pi: G*T flat, group-major rows (pi[g*T + t]).
        arma::vec pi_flat(G_ * T_);
        for (std::size_t g = 0; g < G_; ++g) {
            const arma::vec& pi_g = impl_->data().get(pi_key_for_group(g));
            for (std::size_t t = 0; t < T_; ++t)
                pi_flat[g * T_ + t] = pi_g[t];
        }
        out.emplace("pi", std::move(pi_flat));

        out.emplace("alpha",   impl_->data().get("alpha"));
        out.emplace("gamma_0", impl_->data().get("gamma_0"));
        out.emplace("K_trunc",
                    arma::vec{static_cast<double>(T_)});
        out.emplace("G",
                    arma::vec{static_cast<double>(G_)});
        return out;
    }

    // Backend-neutral set_current. Overwrite z, alpha, gamma_0, or y from a
    // state_map. A "y" entry is a vectorised (column-major from numpy/R's
    // matrix caster) length-(N*d) flat vector; here it represents the N x d
    // data matrix and is read row-major (y[i*d + j]) — callers pass the same
    // cluster-major flattening get_current uses, OR a flat vector built from
    // the original N x d matrix row-major. (set_current("y") is an internal
    // refit hook; N and d are fixed at construction.)
    void set_current(const AI4BayesCode::state_map& params) {
        auto it_y = params.find("y");
        if (it_y != params.end()) {
            const arma::vec& y_new = it_y->second;
            // STRICT-N legitimate (Check #21): z allocation length-N
            // + per-group sufficient stats; HDP also fixes group count.
            if (y_new.n_elem != N_ * d_)
                ai4b::stop("set_current: HDPGaussianMixture fixes N and d at "
                           "construction; y must be a flat length-(N*d) "
                           "vector (N=%zu, d=%zu). Reconstruct to change "
                           "shape.", N_, d_);
            arma::vec yflat(N_ * d_);
            for (std::size_t i = 0; i < N_ * d_; ++i) yflat[i] = y_new[i];
            impl_->data().set("y", yflat);
        }
        auto it_z = params.find("z");
        if (it_z != params.end()) {
            arma::vec znew = it_z->second;
            if (znew.n_elem != N_)
                ai4b::stop("set_current: z length must equal N");
            for (std::size_t i = 0; i < N_; ++i) {
                const long lab = static_cast<long>(std::llround(znew[i]));
                if (lab < 1 || static_cast<std::size_t>(lab) > T_)
                    ai4b::stop("set_current: z[i] out of {1, ..., T}");
            }
            // child(0) is the categorical z block (first child).
            dynamic_cast<categorical_gibbs_block&>(
                impl_->child(0)).set_current(znew);
            impl_->data().set("z", znew);
            impl_->data().refresh_derived_for("z");
        }
        auto it_alpha = params.find("alpha");
        if (it_alpha != params.end()) {
            const double a = it_alpha->second[0];
            if (!(a > 0.0)) ai4b::stop("alpha must be > 0");
            impl_->data().set("alpha", arma::vec{a});
        }
        auto it_gamma = params.find("gamma_0");
        if (it_gamma != params.end()) {
            const double g = it_gamma->second[0];
            if (!(g > 0.0)) ai4b::stop("gamma_0 must be > 0");
            impl_->data().set("gamma_0", arma::vec{g});
        }
    }

    // Posterior-predictive y_rep at the training groups. No covariate inputs
    // at v0; new_data must be empty. Returns a history_map: each entry is a
    // 2-D arma::mat (rows = draws in history mode, or a single row otherwise).
    // y_rep is returned FLAT (length N*d, cluster/obs-major row-major) per
    // row, matching the y_rep refresher's flat layout.
    AI4BayesCode::history_map predict_at(
            const AI4BayesCode::state_map& new_data) const {
        if (!new_data.empty())
            ai4b::stop(
                "HDPGaussianMixture: predict_at(new_data) does NOT support "
                "covariate-dependent inference at v0; pass an empty map/list.");

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

        // History mode: per-draw resample. mu and sigma are sub-outputs of
        // the cluster_params block; pi_g_key for each group g is its own
        // block. Replicates the stochastic refresher's logic per draw.
        AI4BayesCode::history_map hist = impl_->get_history();
        const arma::mat& mu_hist  = hist.at("mu");      // n_draws x (T*d)
        const arma::mat& sig_hist = hist.at("sigma");   // n_draws x (T*d*d)
        std::vector<arma::mat> pi_g_hist; pi_g_hist.reserve(G_);
        for (std::size_t g = 0; g < G_; ++g)
            pi_g_hist.push_back(hist.at(pi_key_for_group(g)));
        const std::size_t n_draws = mu_hist.n_rows;

        const arma::vec& g_idx = impl_->data().get("group_idx");

        arma::mat yrep_mat(n_draws, N_ * d_);
        std::uniform_real_distribution<double> unif(0.0, 1.0);
        std::normal_distribution<double> nd(0.0, 1.0);
        for (std::size_t draw = 0; draw < n_draws; ++draw) {
            for (std::size_t i = 0; i < N_; ++i) {
                const std::size_t g = static_cast<std::size_t>(
                    std::llround(g_idx[i]));
                const arma::mat& pi_mat = pi_g_hist[g];
                // Sample z_i ~ Cat(pi_g_d)
                const double u = unif(predict_rng_);
                double cumul = 0.0;
                std::size_t z_i = T_ - 1;
                for (std::size_t t = 0; t < T_; ++t) {
                    cumul += pi_mat(draw, t);
                    if (u < cumul) { z_i = t; break; }
                }
                // Sample y ~ N(mu_z, sigma_z)
                arma::mat S(d_, d_);
                for (std::size_t a = 0; a < d_; ++a)
                    for (std::size_t b = 0; b < d_; ++b)
                        S(a, b) = sig_hist(draw, z_i * d_ * d_ + a * d_ + b);
                arma::mat L;
                if (!arma::chol(L, S, "lower")) {
                    S.diag() += 1e-8;
                    arma::chol(L, S, "lower");
                }
                arma::vec eps(d_);
                for (std::size_t j = 0; j < d_; ++j) eps[j] = nd(predict_rng_);
                arma::vec L_eps = L * eps;
                for (std::size_t j = 0; j < d_; ++j)
                    yrep_mat(draw, i * d_ + j) =
                        mu_hist(draw, z_i * d_ + j) + L_eps[j];
            }
        }
        out.emplace("y_rep", std::move(yrep_mat));
        return out;
    }

    AI4BayesCode::dag_info     get_dag()     const { return impl_->get_dag(); }
    AI4BayesCode::history_map  get_history() const { return impl_->get_history(); }

private:
    std::mt19937_64                  rng_;
    mutable std::mt19937_64          predict_rng_;
    std::unique_ptr<composite_block> impl_;
    bool                             keep_history_ = false;
    std::size_t                      N_ = 0;
    std::size_t                      d_ = 0;
    std::size_t                      T_ = 0;  // K_trunc
    std::size_t                      G_ = 0;
};

#ifdef AI4BAYESCODE_RCPP_MODULE
RCPP_MODULE(HDPGaussianMixture_module) {
    Rcpp::class_<HDPGaussianMixture>("HDPGaussianMixture")
        .constructor<arma::mat, arma::vec, int,
                     arma::vec, double, arma::mat,
                     double, double, double, int>(
            "Legacy constructor; keep_history defaults to FALSE.")
        .constructor<arma::mat, arma::vec, int,
                     arma::vec, double, arma::mat,
                     double, double, double, int, bool>(
            "Construct HDPGaussianMixture(y, group_idx, K_trunc, mu_0, "
            "kappa_0, Psi_0, nu_0, alpha, gamma_0, seed, keep_history). "
            "Truncated HDP with V0 SIMPLIFIED β update (heuristic on "
            "combined counts_t; NOT the rigorous Antoniak-table CRF). "
            "See header for caveat. group_idx: numeric vector length N "
            "with integer values in {0, 1, ..., G-1}.")
        .method("step", (void (HDPGaussianMixture::*)())    &HDPGaussianMixture::step, "Run one sweep.")
        .method("step", (void (HDPGaussianMixture::*)(int)) &HDPGaussianMixture::step, "Run n sweeps.")
        .method("get_current", &HDPGaussianMixture::get_current)
        .method("set_current", &HDPGaussianMixture::set_current,
                "Overwrite z, alpha, gamma_0, or y from a named list.")
        .method("predict_at",  &HDPGaussianMixture::predict_at,
                "Posterior predictive y_rep at training X. Empty list only.")
        .method("get_dag",     &HDPGaussianMixture::get_dag)
        .method("get_history", &HDPGaussianMixture::get_history);
}
#endif

#ifdef AI4BAYESCODE_PYBIND_MODULE
#include "AI4BayesCode/pybind_casters.hpp"
PYBIND11_MODULE(HDPGaussianMixture, m) {
    AI4BayesCode::register_ai4bayescode_types(m);
    pybind11::class_<HDPGaussianMixture>(m, "HDPGaussianMixture")
        .def(pybind11::init<arma::mat, arma::vec, int,
                            arma::vec, double, arma::mat,
                            double, double, double, int, bool>(),
             pybind11::arg("y"),
             pybind11::arg("group_idx"),
             pybind11::arg("K_trunc"),
             pybind11::arg("mu_0"),
             pybind11::arg("kappa_0"),
             pybind11::arg("Psi_0"),
             pybind11::arg("nu_0"),
             pybind11::arg("alpha"),
             pybind11::arg("gamma_0"),
             pybind11::arg("rng_seed") = 1,
             pybind11::arg("keep_history") = false)
        .def("step", (void (HDPGaussianMixture::*)())    &HDPGaussianMixture::step, "Run one sweep.")
        .def("step", (void (HDPGaussianMixture::*)(int)) &HDPGaussianMixture::step, pybind11::arg("n_steps"))
        .def("get_current", &HDPGaussianMixture::get_current)
        .def("set_current", &HDPGaussianMixture::set_current,
             pybind11::arg("params"))
        .def("predict_at",  &HDPGaussianMixture::predict_at,
             pybind11::arg("new_data"))
        .def("get_dag",     &HDPGaussianMixture::get_dag)
        .def("get_history", &HDPGaussianMixture::get_history);
}
#endif

//==============================================================================
//  Standalone FRONTEND-INDEPENDENT demo (pure C++; no Rcpp, no pybind).
//
//  Simulates a 2-group, 1-D Gaussian mixture with THREE shared atoms at known
//  locations. Each group mixes the shared atoms in different proportions (atom
//  sharing across groups is the whole point of an HDP). We fit the truncated
//  HDP sampler, then recover the posterior-mean cluster locations of the three
//  POPULATED atoms and check that the set of recovered locations matches the
//  three true atom means. Recovery is label-invariant: we match each true mean
//  to its nearest recovered populated-atom mean.
//
//  State is read via the full six-method contract (get_current()); the keys
//  used here ("mu", "z") come from HDPGaussianMixture::get_current().
//==============================================================================
#if !defined(AI4BAYESCODE_RCPP_MODULE) && !defined(AI4BAYESCODE_PYBIND_MODULE)
#include <cstdio>

int main() {
    // ---- Ground truth: 3 shared atoms, 2 groups -----------------------------
    const std::size_t d        = 1;
    const std::size_t G        = 2;
    const int         K_trunc  = 8;          // truncation > 3 true atoms
    const double      atom_mu[3]  = { -6.0, 0.0, 6.0 };
    const double      atom_sd     = 0.6;     // tight, well-separated atoms

    // Per-group mixing weights over the 3 shared atoms (rows sum to 1).
    const double w[G][3] = {
        { 0.60, 0.30, 0.10 },   // group 0 favours atom 0
        { 0.10, 0.30, 0.60 }    // group 1 favours atom 2
    };
    const std::size_t n_per_group = 150;
    const std::size_t N           = G * n_per_group;

    std::mt19937_64 sim_rng(123);
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    std::normal_distribution<double>       gauss(0.0, 1.0);

    arma::mat  y(N, d);
    arma::vec  group_idx(N);   // package ctor takes arma::vec (double labels)
    std::size_t row = 0;
    for (std::size_t g = 0; g < G; ++g) {
        for (std::size_t i = 0; i < n_per_group; ++i) {
            // pick a true atom from this group's weights
            const double u = unif(sim_rng);
            double cum = 0.0; std::size_t a = 2;
            for (std::size_t k = 0; k < 3; ++k) {
                cum += w[g][k];
                if (u < cum) { a = k; break; }
            }
            y(row, 0)       = atom_mu[a] + atom_sd * gauss(sim_rng);
            group_idx[row]  = static_cast<double>(g);
            ++row;
        }
    }

    // ---- NIW prior: weak, centred on the data mean -------------------------
    arma::vec mu_0  = { arma::mean(y.col(0)) };
    const double kappa_0 = 0.05;        // weak prior on the mean
    arma::mat Psi_0 = arma::mat(1, 1, arma::fill::value(1.0));
    const double nu_0    = 3.0;         // > d - 1
    const double alpha   = 1.0;         // group-level concentration
    const double gamma_0 = 1.0;         // top-level concentration

    HDPGaussianMixture model(y, group_idx, K_trunc, mu_0, kappa_0, Psi_0,
                             nu_0, alpha, gamma_0, /*rng_seed=*/7,
                             /*keep_history=*/false);

    // ---- Warmup + sampling --------------------------------------------------
    model.step(500);   // warmup

    const int  M = 1500;
    const std::size_t T = static_cast<std::size_t>(K_trunc);
    arma::vec mu_sum(T, arma::fill::zeros);   // sum of mu_t over draws
    arma::vec occ_sum(T, arma::fill::zeros);  // sum of occupancy (counts) over draws
    for (int s = 0; s < M; ++s) {
        model.step(1);
        const auto gc = model.get_current();          // copy (avoids dangling ref)
        const arma::vec& mu = gc.at("mu");            // length T*d (d=1)
        const arma::vec& z  = gc.at("z");             // length N, labels 1..T
        for (std::size_t t = 0; t < T; ++t) mu_sum[t] += mu[t];
        for (std::size_t i = 0; i < N; ++i) {
            const long lab = static_cast<long>(std::llround(z[i]));
            if (lab >= 1 && static_cast<std::size_t>(lab) <= T)
                occ_sum[static_cast<std::size_t>(lab) - 1] += 1.0;
        }
    }
    arma::vec mu_bar  = mu_sum  / static_cast<double>(M);
    arma::vec occ_bar = occ_sum / static_cast<double>(M);   // avg #points per atom

    // ---- Identify POPULATED atoms (avg occupancy > threshold) --------------
    const double occ_thresh = 0.05 * static_cast<double>(N);  // >= 5% of data
    std::vector<double> populated_mu;
    for (std::size_t t = 0; t < T; ++t)
        if (occ_bar[t] > occ_thresh) populated_mu.push_back(mu_bar[t]);

    std::printf("HDPGaussianMixture demo: N=%zu, G=%zu, K_trunc=%d, "
                "true atoms = {%.1f, %.1f, %.1f}\n",
                N, G, K_trunc, atom_mu[0], atom_mu[1], atom_mu[2]);
    std::printf("  populated atoms found: %zu (occupancy > %.0f pts)\n",
                populated_mu.size(), occ_thresh);
    for (std::size_t t = 0; t < T; ++t)
        if (occ_bar[t] > occ_thresh)
            std::printf("    atom %zu: mu_hat=%+.3f  (avg occ %.1f pts)\n",
                        t, mu_bar[t], occ_bar[t]);

    // ---- Label-invariant recovery check ------------------------------------
    // Each true atom mean must have a populated recovered atom within tol.
    const double tol = 0.6;
    bool all_matched = true;
    for (std::size_t a = 0; a < 3; ++a) {
        double best = std::numeric_limits<double>::infinity();
        for (double mh : populated_mu)
            best = std::min(best, std::abs(mh - atom_mu[a]));
        std::printf("  true atom %.1f -> nearest recovered err = %.3f%s\n",
                    atom_mu[a], best, best <= tol ? "" : "  <-- MISS");
        if (best > tol) all_matched = false;
    }

    // We expect exactly the 3 populated atoms (allow 1 spurious tiny extra,
    // but the 3 true ones must each be matched and recovered count >= 3).
    const bool count_ok = populated_mu.size() >= 3;
    const bool ok = all_matched && count_ok;

    std::printf("%s\n",
                ok ? "[demo PASS] HDP recovers the 3 shared atoms across groups"
                   : "[demo FAIL] HDP did not recover the shared atoms");
    return ok ? 0 : 1;
}
#endif
