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
//      child(3) alpha        nuts_block on the Antoniak (k, n) marginal
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
/// scale -- prior-agnostic NUTS update, matching the other DP examples.
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
    DPGammaMixture(const arma::vec& y, int K_trunc,
                   int rng_seed = 1, bool keep_history = false)
        : impl_(std::make_unique<composite_block>("DPGammaMixture")),
          rng_(rng_seed == 0 ? std::random_device{}()
                             : static_cast<std::mt19937_64::result_type>(rng_seed)),
          predict_rng_(rng_seed == 0 ? std::random_device{}()
                                     : static_cast<std::mt19937_64::result_type>(rng_seed) + 7919u),
          readapt_rng_(rng_seed == 0 ? std::random_device{}()
                                     : static_cast<std::mt19937_64::result_type>(rng_seed) + 104729u) {
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
            nuts_block_config cfg;
            cfg.name        = "alpha";
            cfg.initial_unc = arma::vec{0.0};
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
            impl_->add_child(std::make_unique<nuts_block>(std::move(cfg)));
        }

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

    AI4BayesCode::history_map get_history() const { return impl_->get_history(); }
    AI4BayesCode::dag_info    get_dag()     const { return impl_->get_dag(); }

private:
    std::unique_ptr<composite_block> impl_;
    std::mt19937_64 rng_, predict_rng_, readapt_rng_;
    std::size_t N_ = 0, K_ = 0;
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
        .constructor<arma::vec, int, int, bool>()
        .method("step",        &DPGammaMixture::step)
        .method("get_current", &DPGammaMixture::get_current)
        .method("get_history", &DPGammaMixture::get_history)
        .method("get_dag",     &DPGammaMixture::get_dag)
        AI4BAYESCODE_BIND_KERNEL_CONTROL(DPGammaMixture)
        AI4BAYESCODE_BIND_READAPT_NUTS(DPGammaMixture)
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
             pybind11::arg("K_trunc"),
             pybind11::arg("rng_seed")     = 1,
             pybind11::arg("keep_history") = false)
        .def("step",        &DPGammaMixture::step)
        .def("get_current", &DPGammaMixture::get_current)
        .def("get_history", &DPGammaMixture::get_history)
        .def("get_dag",     &DPGammaMixture::get_dag)
        AI4BAYESCODE_PYBIND_KERNEL_CONTROL(DPGammaMixture)
        AI4BAYESCODE_PYBIND_READAPT_NUTS(DPGammaMixture)
        ;
}
#endif
