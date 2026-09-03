// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v3.0 or later
// (GPL-3.0-or-later; see https://www.gnu.org/licenses/gpl-3.0.html).
// ============================================================================
//  DPGammaMixture.cpp -- truncated Dirichlet process mixture of GAMMA
//  densities, for strictly positive data (waiting times, claim sizes,
//  rainfall). The reference use of `cluster_atom_block`.
//
//  Model
//  -----
//      alpha        ~ Gamma(1, 1)                       [shape, RATE]
//      V_k          ~ Beta(1, alpha),  k = 1..K-1;  V_K = 1
//      w            = stick-breaking(V)                 (Ishwaran-James)
//      z_i          ~ Categorical(w),        i = 1..N
//      shape_k      ~ Gamma(a_shape, b_shape)           k = 1..K
//      rate_k       ~ Gamma(a_rate,  b_rate)            k = 1..K
//      y_i | z_i=k  ~ Gamma(shape_k, rate_k)            y_i > 0
//
//  WHY THIS MODEL IS THE RIGHT DEMONSTRATION
//  -----------------------------------------
//  The component parameters have NO conjugate closed form. The Gamma
//  log-likelihood contributes  -n_k * lgamma(shape_k)  to component k's
//  conditional, so `shape_k` is not a member of any standard family no matter
//  what prior you put on it. (`rate_k` alone IS conjugate given `shape_k`, but
//  the PAIR is not, and the pair is what the atom step must draw.) That rules
//  out `normal_gamma_cluster_gibbs_block` and `niw_cluster_gibbs_block`, which
//  are the conjugate members of this family -- prefer those whenever they DO
//  apply, they draw exactly and are cheaper.
//
//  The block that fills the gap is `cluster_atom_block`: it needs only the
//  per-component conditional as a plain log-density, and it samples each
//  component with a univariate slice kernel, so no gradient and no conjugacy
//  are required. Both atom slices here are POSITIVE; the block owns the log
//  transform and its Jacobian.
//
//  WHY NOT ONE joint_nuts_block OVER ALL COMPONENTS
//  ------------------------------------------------
//  Given z the components are conditionally independent (Ishwaran & James
//  2001, blocked Gibbs step (a), Eq. 18), and an UNOCCUPIED component's
//  conditional collapses to its prior -- for a truncation of K with only a few
//  occupied components, most of the state is prior-wide while the occupied
//  ones are data-narrow, and which is which changes every sweep as z moves.
//  One NUTS trajectory over the whole 2K vector has a single step size that
//  must be frozen after warmup to stay valid (Check #20), so it cannot track
//  that. `cluster_atom_block` gives every component its own update, and the
//  empty-component case needs no special code: with n_k = 0 the likelihood
//  terms drop out of the expression below and the target IS the prior.
//
//  Block decomposition
//  -------------------
//      child(0) z            categorical_gibbs_block   (exact, closed form)
//      child(1) atoms        cluster_atom_block        (shape, rate) per k
//      child(2) w            stick_breaking_block      Beta(1 + n_k, alpha + tail)
//      child(3) alpha        univariate_slice_sampling_block on the Antoniak (k, n) marginal
//
//  Component labels are exchangeable, so summarise label-INVARIANT functionals
//  (the fitted density on a grid, the number of occupied components) rather
//  than shape_3 by name.
//
// @example:R
//   library(AI4BayesCode)
//   ai4bayescode_example("DPGammaMixture")
//   set.seed(7); n <- 200L
//   grp <- sample(1:2, n, TRUE, c(0.6, 0.4))            # 2 well-separated groups
//   y   <- ifelse(grp == 1, rgamma(n, 9, 3), rgamma(n, 40, 2))
//   run <- ai4bayescode_run_chains(
//       function(seed) new(DPGammaMixture, y, 12L, as.integer(seed), TRUE),
//       n_chains = 4, n_burn = 1000, n_keep = 2000)
//   print(ai4bayescode_rhat_summary(run))
//   ai4bayescode_diagnose(run$histories[[1]])
// @example:python
//   import numpy as np, AI4BayesCode
//   rng = np.random.default_rng(7); n = 200
//   grp = rng.choice(2, n, p=[0.6, 0.4])
//   y = np.where(grp == 0, rng.gamma(9.0, 1/3.0, n), rng.gamma(40.0, 1/2.0, n))
//   Mod = AI4BayesCode.example("DPGammaMixture")
//   chains = AI4BayesCode.run_chains(
//       lambda seed: Mod.DPGammaMixture(y, 12, rng_seed=int(seed), keep_history=True),
//       seeds=[11, 22, 33, 44], n_burn=1000, n_keep=2000)
//   AI4BayesCode.diagnose(chains[0]["hist"])
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

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "AI4BayesCode/backend_neutral.hpp"
#include "AI4BayesCode/categorical_gibbs_block.hpp"
#include "AI4BayesCode/cluster_atom_block.hpp"
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/constraints.hpp"
#include "AI4BayesCode/bnp_utils.hpp"
#include "AI4BayesCode/kernel_control_mixin.hpp"
#include "AI4BayesCode/nuts_block.hpp"
#include "AI4BayesCode/univariate_slice_sampling_block.hpp"
#include "AI4BayesCode/stick_breaking_block.hpp"

using AI4BayesCode::block_context;
using AI4BayesCode::categorical_gibbs_block;
using AI4BayesCode::categorical_gibbs_block_config;
using AI4BayesCode::cluster_atom_block;
using AI4BayesCode::cluster_atom_block_config;
using AI4BayesCode::composite_block;
using AI4BayesCode::joint_constraint;
using AI4BayesCode::joint_nuts_sub_param;
using AI4BayesCode::nuts_block;
using AI4BayesCode::nuts_block_config;
using AI4BayesCode::univariate_slice_sampling_block;
using AI4BayesCode::univariate_slice_sampling_block_config;
using AI4BayesCode::stick_breaking_block;
using AI4BayesCode::stick_breaking_block_config;
namespace constraints = AI4BayesCode::constraints;
namespace bnp = AI4BayesCode::bnp;

namespace {

/// log Gamma(y | shape, rate), for y > 0.
inline double gamma_log_density(double y, double shape, double rate) {
    return shape * std::log(rate) - std::lgamma(shape)
         + (shape - 1.0) * std::log(y) - rate * y;
}

/// Component k's conditional, NATURAL scale, up to a constant. cluster_atom_block
/// adds the POSITIVE Jacobians for both slices itself.
///
///   log p(shape_k, rate_k | z, y)
///     = (a_shape - 1) log shape - b_shape * shape        [Gamma prior]
///     + (a_rate  - 1) log rate  - b_rate  * rate         [Gamma prior]
///     + sum_{i: z_i = k+1} log Gamma(y_i | shape, rate)
///
/// With n_k = 0 the sum is empty and this IS the prior -- Ishwaran & James
/// step (a) for an unoccupied component, with no branch to write.
double atom_log_density(const arma::vec& th, std::size_t k,
                        const block_context& ctx) {
    const double shape = th[0], rate = th[1];
    if (!(shape > 0.0) || !(rate > 0.0))
        return -std::numeric_limits<double>::infinity();

    const double a_shape = ctx.at("a_shape")[0], b_shape = ctx.at("b_shape")[0];
    const double a_rate  = ctx.at("a_rate")[0],  b_rate  = ctx.at("b_rate")[0];
    const arma::vec& y = ctx.at("y");
    const arma::vec& z = ctx.at("z");

    double lp = (a_shape - 1.0) * std::log(shape) - b_shape * shape
              + (a_rate  - 1.0) * std::log(rate)  - b_rate  * rate;
    for (std::size_t i = 0; i < z.n_elem; ++i) {
        if (static_cast<std::size_t>(std::llround(z[i])) != k + 1) continue;
        lp += gamma_log_density(y[i], shape, rate);
    }
    return lp;
}

/// log p(alpha | k, n), the Antoniak (1974) (k, n) marginal on the NATURAL
/// scale -- prior-agnostic slice update, matching the other DP examples.
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
    const double lp = (a_prior - 1.0) * std::log(a) - b_prior * a
                    + kd * std::log(a)
                    + std::lgamma(a) - std::lgamma(a + n);
    if (grad_nat) {
        grad_nat->set_size(1);
        (*grad_nat)[0] = (a_prior - 1.0 + kd) / a - b_prior
                       + ai4b::digamma(a) - ai4b::digamma(a + n);
    }
    return lp;
}

}  // namespace

// ============================================================================
//  Wrapper class
// ============================================================================
class DPGammaMixture
    : public AI4BayesCode::kernel_control_mixin<DPGammaMixture> {
    friend class AI4BayesCode::kernel_control_mixin<DPGammaMixture>;
public:
    /// SHORT constructor -- data + seed. Hyperparameters take their
    /// defaults: K_trunc 20. Use the full constructor to change them.
    /// Rcpp ignores C++ default arguments, hence a separate ctor.
    DPGammaMixture(const arma::vec& y,
                   int  rng_seed,
                   bool keep_history = false)
        : DPGammaMixture(y, /*K_trunc=*/20, rng_seed, keep_history) {}

    // rng_seed carries no C++ default: with one, DPGammaMixture(y, n) would
    // be ambiguous between this ctor (n = K_trunc) and the short one above
    // (n = rng_seed).
    DPGammaMixture(const arma::vec& y, int K_trunc,
                   int rng_seed, bool keep_history = false)
        : impl_(std::make_unique<composite_block>("DPGammaMixture")),
          rng_(rng_seed == 0 ? std::random_device{}()
                             : static_cast<std::mt19937_64::result_type>(rng_seed)),
          predict_rng_(rng_seed == 0 ? std::random_device{}()
                                     : static_cast<std::mt19937_64::result_type>(rng_seed) + 7919u) {
        if (y.n_elem < 2)      ai4b::stop("DPGammaMixture: need at least 2 observations");
        if (!y.is_finite())    ai4b::stop("DPGammaMixture: y must be finite");
        if (y.min() <= 0.0)    ai4b::stop("DPGammaMixture: Gamma data must be strictly positive");
        if (K_trunc < 2)       ai4b::stop("DPGammaMixture: K_trunc must be >= 2");
        N_ = y.n_elem;
        K_ = static_cast<std::size_t>(K_trunc);

        // ---- data + hyperparameters -------------------------------------
        impl_->data().set("y", y);
        impl_->data().set("a_shape", arma::vec{2.0});
        impl_->data().set("b_shape", arma::vec{1.0});
        impl_->data().set("a_rate",  arma::vec{2.0});
        impl_->data().set("b_rate",  arma::vec{1.0});
        impl_->data().set("a_alpha", arma::vec{1.0});
        impl_->data().set("b_alpha", arma::vec{1.0});

        // ---- initial values: split y at its quantiles so the components
        //      start apart instead of on top of each other ----------------
        arma::vec y_sorted = arma::sort(y);
        arma::vec z_init(N_), atoms_init(2u * K_);
        for (std::size_t i = 0; i < N_; ++i)
            z_init[i] = double(1 + (i * std::min<std::size_t>(K_, 3u)) / N_);
        for (std::size_t k = 0; k < K_; ++k) {
            const std::size_t q = std::min(N_ - 1, (k * N_) / std::max<std::size_t>(K_, 1u));
            const double m = std::max(y_sorted[q], 1e-6);
            const double v = std::max(arma::var(y), 1e-6);
            atoms_init[2 * k]     = std::max(m * m / v, 1e-2);   // moment-matched shape
            atoms_init[2 * k + 1] = std::max(m / v,     1e-2);   // moment-matched rate
        }
        impl_->data().set("z", z_init);
        impl_->data().set("shape", atoms_init.elem(arma::regspace<arma::uvec>(0, 2, 2 * K_ - 2)));
        impl_->data().set("rate",  atoms_init.elem(arma::regspace<arma::uvec>(1, 2, 2 * K_ - 1)));
        impl_->data().set("alpha", arma::vec{1.0});
        arma::vec w_init(K_, arma::fill::value(1.0 / double(K_)));
        impl_->data().set("pi", w_init);
        impl_->data().set("stick_V", arma::vec(K_, arma::fill::value(1.0 / double(K_))));

        impl_->data().set("cluster_counts", arma::vec(K_, arma::fill::zeros));
        impl_->data().register_refresher("cluster_counts",
            [K = K_](const AI4BayesCode::shared_data_t& d) -> arma::vec {
                return bnp::counts_from_z(d.get("z"), K);
            });

        // ---- Gibbs DAG ---------------------------------------------------
        impl_->data().declare_dependencies("z",     {"y", "pi", "shape", "rate"});
        impl_->data().declare_dependencies("cluster_params", {"z", "y", "cluster_counts",
                                                     "a_shape", "b_shape",
                                                     "a_rate", "b_rate"});
        impl_->data().declare_dependencies("pi",    {"cluster_counts", "alpha"});
        impl_->data().declare_dependencies("alpha", {"cluster_counts", "a_alpha", "b_alpha"});
        impl_->data().declare_invalidates("z", {"cluster_counts"});

        // ---- child(0) z : exact categorical draw --------------------------
        // Sampling note: z is a discrete cluster label, drawn exactly from its
        // closed-form conditional.
        {
            categorical_gibbs_block_config cfg;
            cfg.name           = "z";
            cfg.n_obs          = N_;
            cfg.n_categories   = K_;
            cfg.initial_labels = z_init;
            const std::size_t N_cap = N_, K_cap = K_;
            cfg.log_probs_fn = [N_cap, K_cap](const block_context& ctx) -> arma::mat {
                const arma::vec& y   = ctx.at("y");
                const arma::vec& pi  = ctx.at("pi");
                const arma::vec& sh  = ctx.at("shape");
                const arma::vec& rt  = ctx.at("rate");
                arma::mat lp(N_cap, K_cap);
                for (std::size_t i = 0; i < N_cap; ++i)
                    for (std::size_t k = 0; k < K_cap; ++k)
                        lp(i, k) = std::log(pi[k] + 1e-300)
                                 + gamma_log_density(y[i], sh[k], rt[k]);
                return lp;
            };
            impl_->add_child(std::make_unique<categorical_gibbs_block>(std::move(cfg)));
        }

        // ---- child(1) atoms : the non-conjugate per-component update -------
        // Sampling note: each component's shape and rate are updated on their
        // own, from the data currently assigned to that component; a component
        // with no data assigned falls back to its prior.
        {
            cluster_atom_block_config cfg;
            cfg.name    = "cluster_params";
            cfg.K_trunc = K_;
            cfg.sub_params.push_back(joint_nuts_sub_param{"shape", 1u, joint_constraint::POSITIVE});
            cfg.sub_params.push_back(joint_nuts_sub_param{"rate",  1u, joint_constraint::POSITIVE});
            cfg.initial_cat = atoms_init;
            cfg.log_density   = &atom_log_density;
            impl_->add_child(std::make_unique<cluster_atom_block>(std::move(cfg)));
        }

        // ---- child(2) w : stick-breaking weights ---------------------------
        {
            stick_breaking_block_config cfg;
            cfg.name       = "pi";
            cfg.K_trunc    = K_;
            cfg.counts_key = "cluster_counts";
            cfg.v_name     = "stick_V";
            cfg.a_fn = [](std::size_t k, const arma::vec& counts,
                          const block_context&) { return 1.0 + counts[k]; };
            cfg.b_fn = [](std::size_t k, const arma::vec& counts,
                          const block_context& ctx) {
                double tail = 0.0;
                for (std::size_t j = k + 1; j < counts.n_elem; ++j) tail += counts[j];
                return ctx.at("alpha")[0] + tail;
            };
            cfg.initial_pi = w_init;
            impl_->add_child(std::make_unique<stick_breaking_block>(std::move(cfg)));
        }

        // ---- child(3) alpha ------------------------------------------------
        {
            univariate_slice_sampling_block_config cfg;
            cfg.name        = "alpha";
            cfg.initial_unc = arma::vec{0.0};
            cfg.constrain   = constraints::positive::constrain;
            cfg.unconstrain = constraints::positive::unconstrain;
            cfg.log_density =
                [](const arma::vec& t_unc, const block_context& ctx) -> double {
                    return constraints::positive::wrap(t_unc, nullptr,
                        [&](const arma::vec& t_nat, arma::vec* g_nat) {
                            return alpha_natural_log_density(t_nat, ctx, g_nat);
                        });
                };
            impl_->add_child(std::make_unique<univariate_slice_sampling_block>(std::move(cfg)));
        }

        // ---- Predict DAG + y_rep stochastic refresher --------------------
        // y is an observed terminal, not a replaceable covariate, so there is
        // no declare_data_input here. The refresher reads pi/shape/rate only.
        impl_->data().declare_predict_edges("pi",    {"y_rep"});
        impl_->data().declare_predict_edges("shape", {"y_rep"});
        impl_->data().declare_predict_edges("rate",  {"y_rep"});

        impl_->data().set("y_rep", arma::vec(N_, arma::fill::zeros));
        impl_->data().register_stochastic_refresher("y_rep",
            [N = N_, K = K_](const AI4BayesCode::shared_data_t& dat,
                             std::mt19937_64& rng) -> arma::vec {
                // Marginal posterior predictive: z_rep_i ~ Cat(pi),
                // y_rep_i ~ Gamma(shape_{z_rep_i}, rate_{z_rep_i}).
                const arma::vec& pi = dat.get("pi");
                const arma::vec& sh = dat.get("shape");
                const arma::vec& rt = dat.get("rate");
                std::uniform_real_distribution<double> uniform(0.0, 1.0);
                arma::vec out(N);
                for (std::size_t i = 0; i < N; ++i) {
                    const double u = uniform(rng);
                    double cumul = 0.0;
                    std::size_t z_i = K - 1;
                    for (std::size_t k = 0; k < K; ++k) {
                        cumul += pi[k];
                        if (u < cumul) { z_i = k; break; }
                    }
                    // std::gamma_distribution is (shape, SCALE) = (shape, 1/rate).
                    std::gamma_distribution<double> gd(sh[z_i], 1.0 / rt[z_i]);
                    out[i] = gd(rng);
                }
                return out;
            });

        keep_history_ = keep_history;
        if (keep_history) impl_->set_keep_history(true);
    }

    void step(int n_steps) {
        if (n_steps < 0) ai4b::stop("n_steps must be >= 0");
        for (int i = 0; i < n_steps; ++i) impl_->step(rng_);
    }

    AI4BayesCode::state_map get_current() const {
        AI4BayesCode::state_map out;
        out["z"]       = impl_->data().get("z");
        out["pi"]      = impl_->data().get("pi");
        out["shape"]   = impl_->data().get("shape");
        out["rate"]    = impl_->data().get("rate");
        out["alpha"]   = impl_->data().get("alpha");
        out["K_trunc"] = arma::vec{static_cast<double>(K_)};
        return out;
    }

    /// Accepts any subset of (y, z, pi, shape, rate, alpha). y is the data
    /// inflow an outer sampler uses to push a revised response inward
    /// (imputation, a working response from a sibling block, newly arrived
    /// observations); the remaining keys inject parameter state.
    void set_current(const AI4BayesCode::state_map& params) {
        for (const auto& kv : params) {
            const std::string& k = kv.first;
            if (k != "y" && k != "z" && k != "pi" && k != "shape" &&
                k != "rate" && k != "alpha")
                ai4b::stop("DPGammaMixture::set_current: unknown key '%s'. "
                           "Valid: y, z, pi, shape, rate, alpha.", k.c_str());
        }
        if (params.count("y")) {
            const arma::vec& y_new = params.at("y");
            // STRICT-N: z has length N and the per-component sufficient
            // statistics are sized for N observations, so N is fixed at
            // construction. To change N, reconstruct the wrapper.
            if (y_new.n_elem != N_)
                ai4b::stop("set_current: DPGammaMixture fixes N at "
                           "construction (z is length-N). Supplied y has %zu "
                           "elements but requires N = %zu. To change N, "
                           "reconstruct.",
                           static_cast<std::size_t>(y_new.n_elem), N_);
            if (!y_new.is_finite()) ai4b::stop("set_current: y must be finite");
            if (y_new.min() <= 0.0)
                ai4b::stop("set_current: Gamma data must be strictly positive");
            impl_->data().set("y", y_new);
        }
        if (params.count("z")) {
            const arma::vec& znew = params.at("z");
            if (znew.n_elem != N_)
                ai4b::stop("set_current: z length must equal N");
            for (std::size_t i = 0; i < N_; ++i) {
                const long lab = static_cast<long>(std::llround(znew[i]));
                if (lab < 1 || static_cast<std::size_t>(lab) > K_)
                    ai4b::stop("set_current: z[i] out of {1, ..., K_trunc}");
            }
            dynamic_cast<categorical_gibbs_block&>(impl_->child(0))
                .set_current(znew);
            impl_->data().set("z", znew);
            impl_->data().refresh_derived_for("z");
        }
        if (params.count("pi")) {
            const arma::vec& pinew = params.at("pi");
            if (pinew.n_elem != K_)
                ai4b::stop("set_current: pi length must equal K_trunc");
            dynamic_cast<stick_breaking_block&>(impl_->child(2))
                .set_current(pinew);
            impl_->data().set("pi", pinew);
        }
        // shape and rate are the two sub-parameters of the single
        // cluster_atom_block, whose state is one atom-major vector
        // [shape_0, rate_0, shape_1, rate_1, ...]; setting either one alone
        // rebuilds that vector from the other's current value.
        if (params.count("shape") || params.count("rate")) {
            arma::vec sh = impl_->data().get("shape");
            arma::vec rt = impl_->data().get("rate");
            if (params.count("shape")) sh = params.at("shape");
            if (params.count("rate"))  rt = params.at("rate");
            if (sh.n_elem != K_ || rt.n_elem != K_)
                ai4b::stop("set_current: shape and rate must each have "
                           "length K_trunc");
            for (std::size_t k = 0; k < K_; ++k) {
                if (!(sh[k] > 0.0)) ai4b::stop("set_current: shape must be > 0");
                if (!(rt[k] > 0.0)) ai4b::stop("set_current: rate must be > 0");
            }
            arma::vec atoms(2u * K_);
            for (std::size_t k = 0; k < K_; ++k) {
                atoms[2 * k]     = sh[k];
                atoms[2 * k + 1] = rt[k];
            }
            dynamic_cast<cluster_atom_block&>(impl_->child(1))
                .set_current(atoms);
            impl_->data().set("shape", sh);
            impl_->data().set("rate",  rt);
        }
        if (params.count("alpha")) {
            const double a_new = params.at("alpha")[0];
            if (!(a_new > 0.0)) ai4b::stop("set_current: alpha must be > 0");
            dynamic_cast<univariate_slice_sampling_block&>(impl_->child(3))
                .set_current(arma::vec{a_new});
            impl_->data().set("alpha", arma::vec{a_new});
        }
    }

    /// Posterior-predictive y_rep at the training data. An EMPTY map is the
    /// only supported argument: the mixture carries no covariates, so there
    /// is no new design to predict at.
    AI4BayesCode::history_map predict_at(
            const AI4BayesCode::state_map& new_data) const {
        if (!new_data.empty())
            ai4b::stop("DPGammaMixture: predict_at takes an empty list/map -- "
                       "the model has no covariates. Pass list() to draw "
                       "y_rep at the training y.");

        AI4BayesCode::history_map out;

        if (!keep_history_) {
            // No-history mode: one posterior-predictive draw, emitted as a
            // 1 x N row so both modes share get_history()'s shape.
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

        // History mode: draw y_rep per retained posterior draw. shape and
        // rate are sub-outputs of the "cluster_params" block; pi is its own.
        AI4BayesCode::history_map hist = impl_->get_history();
        const arma::mat& pi_hist = hist.at("pi");     // n_draws x K
        const arma::mat& sh_hist = hist.at("shape");  // n_draws x K
        const arma::mat& rt_hist = hist.at("rate");   // n_draws x K
        const std::size_t n_draws = pi_hist.n_rows;

        arma::mat yrep_mat(n_draws, N_);
        std::uniform_real_distribution<double> unif(0.0, 1.0);
        for (std::size_t draw = 0; draw < n_draws; ++draw) {
            for (std::size_t i = 0; i < N_; ++i) {
                const double u = unif(predict_rng_);
                double cumul = 0.0;
                std::size_t z_i = K_ - 1;
                for (std::size_t k = 0; k < K_; ++k) {
                    cumul += pi_hist(draw, k);
                    if (u < cumul) { z_i = k; break; }
                }
                // std::gamma_distribution is (shape, SCALE) = (shape, 1/rate).
                std::gamma_distribution<double> gd(
                    sh_hist(draw, z_i), 1.0 / rt_hist(draw, z_i));
                yrep_mat(draw, i) = gd(predict_rng_);
            }
        }
        out.emplace("y_rep", std::move(yrep_mat));
        return out;
    }

    AI4BayesCode::history_map get_history() const { return impl_->get_history(); }
    AI4BayesCode::dag_info    get_dag()     const { return impl_->get_dag(); }

private:
    std::unique_ptr<composite_block> impl_;
    std::mt19937_64 rng_;
    mutable std::mt19937_64 predict_rng_;   // predict_at() const advances it
    std::size_t N_ = 0, K_ = 0;
    bool keep_history_ = false;
};

// ============================================================================
//  Standalone demo: simulate -> fit -> recover
// ============================================================================
#if !defined(AI4BAYESCODE_RCPP_MODULE) && !defined(AI4BAYESCODE_PYBIND_MODULE)
int main() {
    std::mt19937_64 gen(7u);
    const std::size_t n = 200;
    // Two well-separated Gamma groups: (shape 9, rate 3) mean 3, and
    // (shape 40, rate 2) mean 20.
    arma::vec y(n);
    std::bernoulli_distribution grp(0.4);
    std::gamma_distribution<double> g1(9.0, 1.0 / 3.0), g2(40.0, 1.0 / 2.0);
    for (std::size_t i = 0; i < n; ++i) y[i] = grp(gen) ? g2(gen) : g1(gen);

    DPGammaMixture m(y, 12, 1, true);
    m.step(2000);
    m.step(4000);

    const auto cur = m.get_current();
    const arma::vec& pi = cur.at("pi");
    const arma::vec& sh = cur.at("shape");
    const arma::vec& rt = cur.at("rate");
    arma::uvec ord = arma::sort_index(pi, "descend");
    std::printf("DPGammaMixture demo -- truth: mean 3 (shape 9, rate 3) and mean 20 (shape 40, rate 2)\n");
    int shown = 0;
    for (std::size_t j = 0; j < ord.n_elem && shown < 3; ++j) {
        const std::size_t k = ord[j];
        if (pi[k] < 0.02) break;
        std::printf("  w=%.3f  shape=%7.3f  rate=%6.3f  -> mean %6.3f\n",
                    pi[k], sh[k], rt[k], sh[k] / rt[k]);
        ++shown;
    }
    std::printf("  alpha = %.3f\n", cur.at("alpha")[0]);
    const bool ok = shown >= 2;
    std::printf("%s\n", ok ? "[demo OK] at least two components carry weight"
                           : "[demo FAIL] the mixture collapsed");
    return ok ? 0 : 1;
}
#endif

// ============================================================================
//  R binding
// ============================================================================
#ifdef AI4BAYESCODE_RCPP_MODULE
#include "AI4BayesCode/rcpp_wrap.hpp"
RCPP_MODULE(DPGammaMixture) {
    Rcpp::class_<DPGammaMixture>("DPGammaMixture")
        .constructor<arma::vec, int>(
            "Minimal: data + seed. Hyperparameters default (K_trunc 20).")
        .constructor<arma::vec, int, bool>(
            "Minimal + keep_history.")
        .constructor<arma::vec, int, int, bool>()
        .method("step",        &DPGammaMixture::step)
        .method("get_current", &DPGammaMixture::get_current)
        .method("set_current", &DPGammaMixture::set_current)
        .method("predict_at",  &DPGammaMixture::predict_at)
        .method("get_history", &DPGammaMixture::get_history)
        .method("get_dag",     &DPGammaMixture::get_dag)
        AI4BAYESCODE_BIND_KERNEL_CONTROL(DPGammaMixture)
        ;
}
#endif

// ============================================================================
//  Python binding
// ============================================================================
#ifdef AI4BAYESCODE_PYBIND_MODULE
#include "AI4BayesCode/pybind_casters.hpp"
PYBIND11_MODULE(DPGammaMixture, m) {
    AI4BayesCode::register_ai4bayescode_types(m);
    pybind11::class_<DPGammaMixture>(m, "DPGammaMixture")
        .def(pybind11::init<arma::vec, int, int, bool>(),
             pybind11::arg("y"),
             pybind11::arg("K_trunc") = 20,
             pybind11::arg("rng_seed")     = 1,
             pybind11::arg("keep_history") = false)
        .def("step",        &DPGammaMixture::step)
        .def("get_current", &DPGammaMixture::get_current)
        .def("set_current", &DPGammaMixture::set_current, pybind11::arg("params"))
        .def("predict_at",  &DPGammaMixture::predict_at, pybind11::arg("new_data"))
        .def("get_history", &DPGammaMixture::get_history)
        .def("get_dag",     &DPGammaMixture::get_dag)
        AI4BAYESCODE_PYBIND_KERNEL_CONTROL(DPGammaMixture)
        ;
}
#endif
