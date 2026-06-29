// Copyright (C) 2026 AI4BayesCode contributors. GPL-3.0-or-later. See LICENSE at the repository root.
/*================================================================================
 *  nuts_block_v2.hpp  --  drop-in replacement for nuts_block backed by the
 *                          new NutsKernel (v1 mass-precision fix, Mahalanobis
 *                          U-turn, biased Metropolis, 3-phase warmup).
 *
 *  USE: same construction as nuts_block; just swap class name.
 *
 *  KEY DIFFERENCES vs mcmclib-based nuts_block:
 *    - mass matrix is ADAPTED online via Welford during phase II warmup;
 *      user does NOT need to call set_precond_matrix
 *    - step-size dual averaging uses the same Hoffman 2014 rule but with
 *      a 3-phase Stan-style schedule
 *    - subsequent step() calls (after the first) do NO further adaptation;
 *      the kernel's state persists. This matches nuts_block's
 *      n_warmup_per_step = 0 default (which is the mandatory default).
 *================================================================================*/

#ifndef AI4BAYESCODE_NUTS_BLOCK_V2_HPP
#define AI4BAYESCODE_NUTS_BLOCK_V2_HPP

#include "block_sampler.hpp"

#include "nuts_kernel_v1/nuts/nuts_kernel.hpp"

#include <armadillo>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace AI4BayesCode {

using log_density_gradient_fn_v2 = std::function<
    double(const arma::vec& theta_unc,
           const block_context& ctx,
           arma::vec* grad_out)>;

using transform_fn_v2 = std::function<arma::vec(const arma::vec&)>;

struct nuts_block_v2_config {
    std::string name;
    arma::vec   initial_unc;
    log_density_gradient_fn_v2 log_density_grad;
    transform_fn_v2 constrain;    // unconstrained -> natural
    transform_fn_v2 unconstrain;  // natural -> unconstrained

    /// Warmup iterations on the FIRST call to step(). Subsequent calls
    /// do zero further adaptation (n_warmup_per_step = 0, the mandatory
    /// default). NOTE: this default is 1000, NOT mcmclib nuts_block's 200
    /// — empirically the NutsKernel multinomial sampler needs the longer
    /// warmup to reach mcmclib step-size parity on tight time-series
    /// conditionals (arma11: 1000 → 4/20 fail = mcmclib parity; 200 →
    /// 13/20 fail). Validated 2026-06-11.
    std::size_t n_warmup_first_call = 1000;

    /// Number of draws kept per step(). In a tight Gibbs sweep this is 1.
    std::size_t n_draws_per_step = 1;

    /// Optional override for the initial step size.
    double initial_step_size = 0.0;

    /// NutsKernel configuration. The block constructor injects the warmup
    /// schedule based on n_warmup_first_call.
    NutsKernelConfig kernel_cfg;

    /// RNG seed for the kernel. If left at 1u (the default), the kernel
    /// uses seed = 1u (deterministic across blocks). For multi-block
    /// composites users may want to set this to a unique value per block,
    /// or set it to 0 to draw from the outer composite RNG instead.
    std::uint64_t rng_seed_value = 1u;

    /// Whether to run online diagonal Welford mass-matrix adaptation
    /// during warmup. Default OFF — matches legacy mcmclib nuts_block,
    /// safe for multi-block Gibbs composites where the per-block
    /// conditional scale shifts as other blocks update. Turn ON for
    /// joint blocks (single coherent target) or known-stationary
    /// conditionals — empirically gives 10-100× ESS on hierarchical
    /// posteriors. The mass-precision fix from the kernel applies
    /// regardless.
    bool adapt_mass = false;

    /// Use a fixed IDENTITY mass matrix (opt-in). When true, the kernel
    /// never adapts a diagonal mass during warmup — matching mcmclib's
    /// nuts_block, which fixes its precond matrix at construction. This
    /// avoids the per-block-Gibbs funnel freeze caused by a diagonal mass
    /// estimated from the unrepresentative frozen-sibling first-call
    /// warmup variance (which amplifies the effective step). Distinct from
    /// adapt_mass: adapt_mass=false still installs a single marginal-
    /// variance mass at warmup end; use_identity_metric suppresses that
    /// too. See NutsKernelConfig::use_identity_metric.
    bool use_identity_metric = false;

    // Compatibility shims for sim1 models that hard-code mcmclib field
    // names — accepted but currently no-ops.
    bool use_dense_metric         = false;
    bool use_three_phase_warmup   = false;
    std::size_t n_warmup_per_step = 0;
    std::size_t dense_metric_pilot_iters = 200;
    std::size_t dense_metric_adapt_iters = 500;
    std::size_t tp_phase1_iters   = 75;
    std::vector<std::size_t> tp_phase2_windows = {25, 50, 100, 200, 500};
    std::size_t tp_phase3_iters   = 50;
};

class nuts_block_v2 : public block_sampler {
public:
    explicit nuts_block_v2(nuts_block_v2_config cfg)
        : cfg_(std::move(cfg))
    {
        if (cfg_.initial_unc.n_elem == 0) {
            throw std::invalid_argument(
                "nuts_block_v2: initial_unc must be non-empty");
        }
        if (!cfg_.log_density_grad) {
            throw std::invalid_argument(
                "nuts_block_v2: log_density_grad oracle is required");
        }
        if (!cfg_.constrain) {
            cfg_.constrain = [](const arma::vec& x) { return x; };
        }
        if (!cfg_.unconstrain) {
            cfg_.unconstrain = [](const arma::vec& x) { return x; };
        }
        if (cfg_.n_draws_per_step == 0) cfg_.n_draws_per_step = 1;

        // Warmup schedule. If cfg_.adapt_mass = true (opt-in), use the
        // Stan-style 3-phase split with mass-adaptation windows so the
        // posterior marginal scales drive the metric. Otherwise stay on
        // identity mass (compat with legacy mcmclib nuts_block default).
        std::size_t total = cfg_.n_warmup_first_call;
        if (total < 100) total = 100;
        if (cfg_.adapt_mass) {
            // 10% phase1, 80% phase2 (5 windows: 5/10/20/40/the rest),
            // 10% phase3. Min 200 total to allow at least one
            // meaningful mass-adapt window.
            if (total < 200) total = 200;
            std::size_t phase1 = std::max<std::size_t>(20, total / 10);
            std::size_t phase3 = std::max<std::size_t>(20, total / 10);
            std::size_t phase2_total = total - phase1 - phase3;
            cfg_.kernel_cfg.warmup_phase1_iters = phase1;
            cfg_.kernel_cfg.warmup_phase3_iters = phase3;
            cfg_.kernel_cfg.warmup_phase2_windows =
                split_phase2_windows(phase2_total);
        } else {
            cfg_.kernel_cfg.warmup_phase1_iters = total;
            cfg_.kernel_cfg.warmup_phase2_windows = {};
            cfg_.kernel_cfg.warmup_phase3_iters = 0;
        }

        // Propagate the identity-metric flag (mcmclib-style fixed identity
        // mass). When ON, the kernel skips diagonal mass adaptation.
        cfg_.kernel_cfg.use_identity_metric = cfg_.use_identity_metric;

        theta_unc_     = cfg_.initial_unc;
        theta_natural_ = cfg_.constrain(theta_unc_);
        first_call_    = true;
    }

    void set_context(const block_context& ctx) override {
        context_ = ctx;
    }

    void step(std::mt19937_64& rng) override {
        // Lazily build the kernel on the FIRST call, when context_ has
        // been filled by the outer composite. The log_density lambda
        // closes over `this` and reads context_ on every kernel.step().
        if (!kernel_) {
            // Draw a kernel seed from caller's rng for forward progress.
            const std::uint64_t seed =
                cfg_.rng_seed_value
                    ? cfg_.rng_seed_value
                    : static_cast<std::uint64_t>(rng());
            auto target = [this](const arma::vec& theta_in,
                                  arma::vec* grad_out) -> double {
                return cfg_.log_density_grad(theta_in, context_, grad_out);
            };
            kernel_ = std::make_unique<NutsKernel>(
                theta_unc_.n_elem, target, cfg_.kernel_cfg, seed);
            kernel_->set_current_position(theta_unc_);
        }

        if (first_call_) {
            // Run the full warmup budget before any kept draws. The
            // kernel's warmup_remaining_ counter handles adaptation
            // internally.
            for (std::size_t i = 0; i < cfg_.n_warmup_first_call; ++i) {
                kernel_->step();
            }
            first_call_ = false;
#ifndef NDEBUG_OFF
            if (std::getenv("NUTS_V2_DIAG")) {
                const auto& s = kernel_->stats();
                const arma::vec md = kernel_->mass_diagonal();
                const double m0 = md.n_elem ? md[0] : 1.0;
                const double eps = kernel_->step_size();
                // Effective natural-scale position step ~ eps * sqrt(1/mass)
                // = eps * sqrt(var). This is what actually overshoots a
                // tight conditional, regardless of the raw eps value.
                const double eff = eps / std::sqrt(m0 > 0 ? m0 : 1.0);
                std::fprintf(stderr,
                    "[v2 %-10s] dim=%zu eps=%.4g mass0=%.4g eff_step=%.4g "
                    "div=%zu uturn=%zu treemax=%zu lastdepth=%.0f\n",
                    cfg_.name.c_str(), (size_t) theta_unc_.n_elem,
                    eps, m0, eff, s.divergences,
                    s.subtree_uturn_stops, s.tree_depth_max_hit,
                    s.last_tree_depth_d);
            }
#endif
        }

        // Kept draws. Just record the last one as theta_unc_.
        for (std::size_t i = 0; i < cfg_.n_draws_per_step; ++i) {
            kernel_->step();
        }
        theta_unc_     = kernel_->current_position();
        theta_natural_ = cfg_.constrain(theta_unc_);

        if (keep_history_) {
            history_buf_.push_back(theta_natural_);
        }
    }

    const arma::vec& current() const override {
        return theta_natural_;
    }

    void set_current(const arma::vec& theta_natural) override {
        theta_natural_ = theta_natural;
        theta_unc_     = cfg_.unconstrain(theta_natural);
        if (kernel_) {
            kernel_->set_current_position(theta_unc_);
        }
    }

    const std::string& name() const noexcept override { return cfg_.name; }

    std::size_t dim() const noexcept override { return theta_natural_.n_elem; }

    // --- diagnostics ------------------------------------------------------

    double current_step_size() const noexcept {
        return kernel_ ? kernel_->step_size() : 0.0;
    }

    arma::vec current_mass_diagonal() const {
        return kernel_ ? kernel_->mass_diagonal() : arma::vec();
    }

    // --- readapt ----------------------------------------------------------

    bool supports_readapt() const noexcept override { return true; }

    void readapt(std::size_t n,
                 bool reset,
                 std::mt19937_64& rng) override {
        if (n == 0 || !kernel_) return;
        kernel_->readapt(n, reset, /*adapt_mass=*/ true);
        // chain state is restored by kernel.readapt itself
        theta_unc_     = kernel_->current_position();
        theta_natural_ = cfg_.constrain(theta_unc_);
    }

    history_map get_history() const override {
        return detail::make_history_map(cfg_.name, history_buf_,
                                        theta_natural_);
    }

    std::size_t history_size() const noexcept override {
        return history_buf_.empty() ? 1 : history_buf_.size();
    }

    void clear_history() override {
        history_buf_.clear();
    }

private:
    static std::vector<std::size_t>
    split_phase2_windows(std::size_t total) {
        // Stan's geometric window schedule: 25, 50, 100, 200, then the
        // remainder. Falls through to a single-window when total is small.
        std::vector<std::size_t> windows;
        if (total < 25) {
            if (total > 0) windows.push_back(total);
            return windows;
        }
        std::size_t remaining = total;
        for (std::size_t w : {std::size_t(25), std::size_t(50),
                              std::size_t(100), std::size_t(200)}) {
            if (remaining >= 2 * w) {
                windows.push_back(w);
                remaining -= w;
            } else {
                break;
            }
        }
        if (remaining > 0) windows.push_back(remaining);
        return windows;
    }

    nuts_block_v2_config       cfg_;
    arma::vec                  theta_unc_;
    arma::vec                  theta_natural_;
    block_context              context_;
    bool                       first_call_ = true;
    std::unique_ptr<NutsKernel> kernel_;
    std::vector<arma::vec>     history_buf_;
};

} // namespace AI4BayesCode

#endif // AI4BAYESCODE_NUTS_BLOCK_V2_HPP
