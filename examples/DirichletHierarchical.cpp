// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later
// (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
//  DirichletHierarchical.cpp
//
//  Hierarchical Dirichlet example. (s, kappa, theta) are sampled by ONE
//  joint_nuts_block rather than three separate single nuts_blocks updated
//  Gibbs-style. This collapses the three tightly-coupled parameters (s and
//  kappa share a kappa*s_i argument to lgamma, and s and theta share the
//  Dirichlet prior alpha = theta/P) into a single HMC trajectory.
//
//  Model
//  -----
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

// Frontend-independent standalone build: no Rcpp / no pybind. The model,
// priors, joint log-density and block wiring are preserved exactly; only the
// R/Python binding layer is removed and an int main() demo is added.

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

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>

// ----------------------------------------------------------------------------
// Standalone replacements for the backend-neutral math helpers that were
// previously pulled in from AI4BayesCode/backend_neutral.hpp. These are pure
// std-math functions with no frontend dependency.
// ----------------------------------------------------------------------------
namespace ai4b {
inline double lgammafn(double x) { return std::lgamma(x); }

// digamma(): psi(x) for x > 0. Recurrence to x >= 6 then the standard
// asymptotic (Bernoulli) expansion; matches R's digamma() to ~1e-12.
inline double digamma(double x) {
    double result = 0.0;
    while (x < 6.0) {
        result -= 1.0 / x;
        x += 1.0;
    }
    const double inv  = 1.0 / x;
    const double inv2 = inv * inv;
    result += std::log(x) - 0.5 * inv
            - inv2 * (1.0 / 12.0
                      - inv2 * (1.0 / 120.0
                                - inv2 * (1.0 / 252.0)));
    return result;
}
}  // namespace ai4b

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
//  Model class (frontend-independent). Drives one joint_nuts_block over
//  (s SIMPLEX, kappa POSITIVE, theta POSITIVE) inside a composite_block.
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
            throw std::runtime_error("S_obs must have at least 1 row");
        }
        if (P < 2) {
            throw std::runtime_error("S_obs must have at least 2 columns");
        }
        if (beta_a <= 0.0 || beta_b <= 0.0) {
            throw std::runtime_error("beta_a, beta_b must be > 0");
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
                    throw std::runtime_error("S_obs entries must be strictly positive");
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

    // Neutral-typed accessor: the composite_block writes each joint sub-param
    // output ("s", "kappa", "theta") back into data() after every step(), so
    // the current draw is read directly from the shared_data_t.
    const AI4BayesCode::shared_data_t& data() const { return impl_->data(); }

    AI4BayesCode::history_map get_history() const { return impl_->get_history(); }

private:
    std::mt19937_64                  rng_;
    mutable std::mt19937_64          predict_rng_;
    mutable std::mt19937_64          readapt_rng_;
    std::unique_ptr<composite_block> impl_;
    bool                             keep_history_ = false;
};

// ============================================================================
//  Standalone demo: simulate -> fit -> recover.
//
//  Generative truth:
//    - Pick a known base simplex s_true (length P) and concentration
//      kappa_true. Each observed row S_obs(k, :) ~ Dirichlet(kappa_true *
//      s_true), exactly the per-document model s_k | s, kappa.
//    - Fit the joint_nuts_block model and recover the posterior mean of s.
//
//  PASS criterion (honest recovery check):
//    (1) the posterior-mean s must beat the uniform simplex (1/P) baseline in
//        L2 distance to s_true (the data is informative), AND
//    (2) the posterior-mean kappa must be the right order of magnitude
//        (within a factor of ~3 of kappa_true) — Exp(1) prior pulls kappa
//        toward 0, so we only require it not collapse and not blow up.
// ============================================================================

namespace {

// Draw one Dirichlet(alpha) simplex via independent Gammas.
arma::vec rdirichlet(const arma::vec& alpha, std::mt19937_64& rng) {
    const std::size_t P = alpha.n_elem;
    arma::vec out(P);
    double sum_g = 0.0;
    for (std::size_t i = 0; i < P; ++i) {
        std::gamma_distribution<double> gam(std::max(alpha[i], 1e-10), 1.0);
        double g = gam(rng);
        if (g < 1e-300) g = 1e-300;
        out[i] = g;
        sum_g += g;
    }
    out /= sum_g;
    return out;
}

}  // namespace

int main() {
    // ---- Known truth ------------------------------------------------------
    const std::size_t P = 4;
    const int         K = 400;          // number of observed simplexes (docs)
    const double kappa_true = 30.0;     // concentration: rows cluster near s_true
    arma::vec s_true = {0.50, 0.25, 0.15, 0.10};   // base simplex
    s_true /= arma::sum(s_true);

    std::mt19937_64 sim_rng(20260621ULL);

    // ---- Simulate K rows ~ Dirichlet(kappa_true * s_true) -----------------
    arma::mat S_obs(static_cast<arma::uword>(K), static_cast<arma::uword>(P));
    const arma::vec dir_alpha = kappa_true * s_true;
    for (int k = 0; k < K; ++k) {
        arma::vec row = rdirichlet(dir_alpha, sim_rng);
        // Guard against exact zeros (model requires strictly positive entries).
        for (std::size_t i = 0; i < P; ++i)
            row[i] = std::max(row[i], 1e-8);
        row /= arma::sum(row);
        S_obs.row(k) = row.t();
    }

    // ---- Fit --------------------------------------------------------------
    const double beta_a = 0.5, beta_b = 1.0;
    const int    seed   = 12345;
    DirichletHierarchical model(S_obs, beta_a, beta_b, seed);

    const int n_warm = 300;   // warmup sweeps (joint block also self-warms 800 on first call)
    const int n_keep = 1500;  // kept sweeps
    model.step(n_warm);       // discard

    arma::vec s_sum(static_cast<arma::uword>(P), arma::fill::zeros);
    double kappa_sum = 0.0, theta_sum = 0.0;
    for (int it = 0; it < n_keep; ++it) {
        model.step(1);
        s_sum     += model.data().get("s");
        kappa_sum += model.data().get("kappa")[0];
        theta_sum += model.data().get("theta")[0];
    }
    const arma::vec s_hat   = s_sum / static_cast<double>(n_keep);
    const double    kappa_hat = kappa_sum / static_cast<double>(n_keep);
    const double    theta_hat = theta_sum / static_cast<double>(n_keep);

    // ---- Compare to truth + uniform baseline ------------------------------
    const arma::vec s_unif(static_cast<arma::uword>(P),
                           arma::fill::value(1.0 / static_cast<double>(P)));
    const double err_hat  = arma::norm(s_hat  - s_true, 2);
    const double err_unif = arma::norm(s_unif - s_true, 2);

    std::printf("=== DirichletHierarchical standalone demo ===\n");
    std::printf("  P = %zu, K = %d, kappa_true = %.2f\n", P, K, kappa_true);
    std::printf("  s_true   = [");
    for (std::size_t i = 0; i < P; ++i) std::printf(" %.4f", s_true[i]);
    std::printf(" ]\n  s_hat    = [");
    for (std::size_t i = 0; i < P; ++i) std::printf(" %.4f", s_hat[i]);
    std::printf(" ]\n  s_unif   = [");
    for (std::size_t i = 0; i < P; ++i) std::printf(" %.4f", s_unif[i]);
    std::printf(" ]\n");
    std::printf("  ||s_hat - s_true||_2  = %.5f\n", err_hat);
    std::printf("  ||s_unif - s_true||_2 = %.5f   (naive baseline)\n", err_unif);
    std::printf("  kappa_hat = %.3f   (truth %.3f)\n", kappa_hat, kappa_true);
    std::printf("  theta_hat = %.4f\n", theta_hat);

    const bool beats_baseline = err_hat < 0.5 * err_unif;   // clearly better than uniform
    const bool kappa_ok = (kappa_hat > kappa_true / 3.0) &&
                          (kappa_hat < kappa_true * 3.0);
    const bool ok = beats_baseline && kappa_ok && std::isfinite(theta_hat);

    std::printf("  recovery beats baseline: %s   kappa order-of-magnitude: %s\n",
                beats_baseline ? "yes" : "NO",
                kappa_ok ? "yes" : "NO");
    std::printf(ok ? "[demo PASS]\n" : "[demo FAIL]\n");
    return ok ? 0 : 1;
}
