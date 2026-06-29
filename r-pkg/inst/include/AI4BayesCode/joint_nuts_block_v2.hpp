// Copyright (C) 2026 AI4BayesCode contributors. GPL-3.0-or-later. See LICENSE at the repository root.
/*================================================================================
 *  joint_nuts_block_v2.hpp  --  drop-in replacement for joint_nuts_block
 *                                backed by the new NutsKernel (v1 mass-precision
 *                                fix, Mahalanobis U-turn, biased Metropolis).
 *================================================================================*/

#ifndef AI4BAYESCODE_JOINT_NUTS_BLOCK_V2_HPP
#define AI4BAYESCODE_JOINT_NUTS_BLOCK_V2_HPP

#include "block_sampler.hpp"
#include "nuts_block_v2.hpp"  // for log_density_gradient_fn_v2

#include <armadillo>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace AI4BayesCode {

struct joint_nuts_sub_param_v2 {
    std::string  name;
    std::size_t  dim;
};

struct joint_nuts_block_v2_config {
    std::string name;
    std::vector<joint_nuts_sub_param_v2> sub_params;
    log_density_gradient_fn_v2 log_density_grad;
    arma::vec initial_cat;
    std::size_t n_warmup_first_call = 1000;
    std::size_t n_draws_per_step    = 1;
    double      initial_step_size   = 0.0;
    NutsKernelConfig kernel_cfg;
    std::uint64_t rng_seed_value    = 1u;
    /// Joint blocks default to mass-adaptation ON — that's the reason
    /// for using a joint block. See nuts_block_v2 for the per-block
    /// safety rationale of OFF in Gibbs composites.
    bool adapt_mass = true;

    // Compatibility shims — accepted but currently no-ops (diagonal mass
    // is online-adapted automatically by NutsKernel; dense metric path
    // is a v2.1 follow-up).
    bool use_dense_metric             = false;
    bool use_three_phase_warmup       = false;
    std::size_t dense_metric_pilot_iters = 200;
    std::size_t dense_metric_adapt_iters = 500;
    std::size_t n_warmup_per_step     = 0;
    std::size_t tp_phase1_iters       = 75;
    std::vector<std::size_t> tp_phase2_windows = {25, 50, 100, 200, 500};
    std::size_t tp_phase3_iters       = 50;
};

class joint_nuts_block_v2 : public block_sampler {
public:
    explicit joint_nuts_block_v2(joint_nuts_block_v2_config cfg)
        : cfg_(std::move(cfg))
    {
        if (cfg_.sub_params.empty()) {
            throw std::invalid_argument(
                "joint_nuts_block_v2: sub_params must be non-empty");
        }
        std::unordered_set<std::string> seen;
        std::size_t sum_dims = 0;
        offsets_.reserve(cfg_.sub_params.size());
        for (const auto& sp : cfg_.sub_params) {
            if (sp.name.empty())
                throw std::invalid_argument(
                    "joint_nuts_block_v2: sub_param.name must be non-empty");
            if (sp.dim == 0)
                throw std::invalid_argument(
                    "joint_nuts_block_v2: sub_param.dim must be > 0 "
                    "(sub_param '" + sp.name + "')");
            if (!seen.insert(sp.name).second)
                throw std::invalid_argument(
                    "joint_nuts_block_v2: duplicate sub_param name '" +
                    sp.name + "'");
            offsets_.push_back(sum_dims);
            sum_dims += sp.dim;
        }
        if (cfg_.initial_cat.n_elem != sum_dims) {
            throw std::invalid_argument(
                "joint_nuts_block_v2: initial_cat size (" +
                std::to_string(cfg_.initial_cat.n_elem) +
                ") does not match sum of sub_param dims (" +
                std::to_string(sum_dims) + ")");
        }
        if (!cfg_.log_density_grad) {
            throw std::invalid_argument(
                "joint_nuts_block_v2: log_density_grad oracle is required");
        }
        if (cfg_.n_draws_per_step == 0) cfg_.n_draws_per_step = 1;

        // Joint blocks own a coupled slice — defaults to mass adaptation
        // ON, because the within-block correlation structure is the
        // whole point of using a joint block. For composites where the
        // joint conditional is non-stationary across whole-sweep
        // updates, set cfg_.adapt_mass = false to use identity metric.
        std::size_t total = cfg_.n_warmup_first_call;
        if (total < 100) total = 100;
        if (cfg_.adapt_mass) {
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

        theta_cat_ = cfg_.initial_cat;
        rebuild_named_outputs_cache_();
        first_call_ = true;
    }

    void set_context(const block_context& ctx) override {
        context_ = ctx;
    }

    void step(std::mt19937_64& rng) override {
        if (!kernel_) {
            const std::uint64_t seed =
                cfg_.rng_seed_value
                    ? cfg_.rng_seed_value
                    : static_cast<std::uint64_t>(rng());
            auto target = [this](const arma::vec& theta_in,
                                  arma::vec* grad_out) -> double {
                return cfg_.log_density_grad(theta_in, context_, grad_out);
            };
            kernel_ = std::make_unique<NutsKernel>(
                theta_cat_.n_elem, target, cfg_.kernel_cfg, seed);
            kernel_->set_current_position(theta_cat_);
        }
        if (first_call_) {
            for (std::size_t i = 0; i < cfg_.n_warmup_first_call; ++i) {
                kernel_->step();
            }
            first_call_ = false;
        }
        for (std::size_t i = 0; i < cfg_.n_draws_per_step; ++i) {
            kernel_->step();
        }
        theta_cat_ = kernel_->current_position();
        rebuild_named_outputs_cache_();
        if (keep_history_) {
            history_buf_.push_back(theta_cat_);
        }
    }

    const arma::vec& current() const override {
        return theta_cat_;
    }

    void set_current(const arma::vec& theta) override {
        if (theta.n_elem != theta_cat_.n_elem) {
            throw std::invalid_argument(
                "joint_nuts_block_v2::set_current: size mismatch");
        }
        theta_cat_ = theta;
        rebuild_named_outputs_cache_();
        if (kernel_) kernel_->set_current_position(theta_cat_);
    }

    const std::string& name() const noexcept override { return cfg_.name; }
    std::size_t dim() const noexcept override { return theta_cat_.n_elem; }

    std::unordered_map<std::string, arma::vec>
    current_named_outputs() const override {
        return named_outputs_cache_;
    }

    bool supports_readapt() const noexcept override { return true; }

    void readapt(std::size_t n, bool reset, std::mt19937_64& rng) override {
        if (n == 0 || !kernel_) return;
        kernel_->readapt(n, reset, /*adapt_mass=*/ true);
        theta_cat_ = kernel_->current_position();
        rebuild_named_outputs_cache_();
    }

    history_map get_history() const override {
        history_map out;
        if (history_buf_.empty()) {
            for (std::size_t s = 0; s < cfg_.sub_params.size(); ++s) {
                const auto& sp = cfg_.sub_params[s];
                const arma::vec slice = theta_cat_.subvec(
                    offsets_[s], offsets_[s] + sp.dim - 1);
                arma::mat m(1, sp.dim);
                for (std::size_t j = 0; j < sp.dim; ++j) m(0, j) = slice[j];
                out.emplace(sp.name, std::move(m));
            }
            return out;
        }
        const std::size_t n = history_buf_.size();
        for (std::size_t s = 0; s < cfg_.sub_params.size(); ++s) {
            const auto& sp = cfg_.sub_params[s];
            arma::mat m(n, sp.dim);
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t j = 0; j < sp.dim; ++j) {
                    m(i, j) = history_buf_[i][offsets_[s] + j];
                }
            }
            out.emplace(sp.name, std::move(m));
        }
        return out;
    }

    std::size_t history_size() const noexcept override {
        return history_buf_.empty() ? 1 : history_buf_.size();
    }

    void clear_history() override { history_buf_.clear(); }

    double current_step_size() const noexcept {
        return kernel_ ? kernel_->step_size() : 0.0;
    }

    /// Return the dimension of the named sub-parameter, or 0 if not found.
    /// Sim1 models call this when they need to know how many indices a
    /// sub-parameter occupies (e.g. theta-irt where dim depends on data).
    std::size_t sub_param_dim(const std::string& name) const noexcept {
        for (const auto& sp : cfg_.sub_params) {
            if (sp.name == name) return sp.dim;
        }
        return 0;
    }

private:
    static std::vector<std::size_t>
    split_phase2_windows(std::size_t total) {
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

    void rebuild_named_outputs_cache_() {
        named_outputs_cache_.clear();
        for (std::size_t s = 0; s < cfg_.sub_params.size(); ++s) {
            const auto& sp = cfg_.sub_params[s];
            arma::vec slice = theta_cat_.subvec(offsets_[s],
                                                 offsets_[s] + sp.dim - 1);
            named_outputs_cache_.emplace(sp.name, std::move(slice));
        }
    }

    joint_nuts_block_v2_config cfg_;
    std::vector<std::size_t>   offsets_;
    arma::vec                  theta_cat_;
    block_context              context_;
    bool                       first_call_ = true;
    std::unique_ptr<NutsKernel> kernel_;
    std::unordered_map<std::string, arma::vec> named_outputs_cache_;
    std::vector<arma::vec>     history_buf_;
};

} // namespace AI4BayesCode

#endif // AI4BAYESCODE_JOINT_NUTS_BLOCK_V2_HPP
