// Copyright (C) 2026 AI4BayesCode contributors. GPL-3.0-or-later. See LICENSE at the repository root.
// dual_averaging.hpp — Stan-style dual-averaging step-size adapter
//
// Algorithm: Hoffman & Gelman 2014 Algorithm 6, dual-averaging update
// (Nesterov 2009). Single hyperparameter: target_accept (default 0.8).
// Returns smoothed log(eps_bar) at end of warmup (Polyak-Ruppert
// averaging). This is the value used during sampling phase, NOT raw
// log(eps).
//
// References (read during Phase 0):
//   - mcmclib nuts.hpp lines 422-430 (mcmclib's variant)
//   - Stan stan/services/util/run_adaptive_sampler.hpp
//   - walnuts walnuts/include/walnuts/adapt.hpp
//
// State semantics:
//   - reset(): zero out all dual-averaging counters (call at start of
//              fresh warmup window OR at start of readapt(reset=true)).
//   - update(actual_accept): one dual-averaging step; call after each
//              MCMC iteration during warmup.
//   - current_step_size(): the RAW eps used by the next iteration
//              (driver during warmup).
//   - sampling_step_size(): the SMOOTHED eps_bar to use during
//              sampling phase (frozen after warmup).
//
// The decomposition lets the kernel use eps during warmup (responsive
// to recent acceptance) and eps_bar after warmup (Polyak-averaged,
// stable).

#pragma once

#include <cmath>
#include <stdexcept>

namespace AI4BayesCode {
namespace internal {

// Stan default Hoffman 2014 hyperparameters.
struct DualAveragingConfig {
    double target_accept = 0.8;
    double gamma = 0.05;   // dual-averaging regularization
    double kappa = 0.75;   // Polyak-Ruppert averaging decay rate
    double t0 = 10.0;      // Polyak-Ruppert warmup offset
    // Maximum allowed step size. The dual-averaging update is clamped so eps
    // cannot run away on loose/drifting per-block conditionals — the funnel
    // freeze: eps grew to 6.84 → shallow trees → spuriously high accept stat
    // → even bigger eps → frozen chain. nuts-rs caps at pi by default
    // (src/stepsize/dual_avg.rs:59); we follow it. The cap also BREAKS that
    // feedback loop: bounded eps → deeper trees → accurate accept stat → eps
    // converges DOWN toward the correct value.
    double max_step_size = 3.14159265358979323846;  // pi (nuts-rs default)
};

class DualAveragingAdapter {
public:
    explicit DualAveragingAdapter(const DualAveragingConfig& cfg = {})
        : cfg_(cfg), m_(0), h_bar_(0.0),
          log_eps_(0.0), log_eps_bar_(0.0), mu_(0.0),
          have_initial_(false) {}

    // Initialize from a user-provided initial step size.
    // Called at the start of warmup. Per Hoffman 2014 §3.4,
    // mu = log(10 * initial_step_size) is the recommended anchor.
    void init(double initial_step_size) {
        if (!(initial_step_size > 0.0) ||
            !std::isfinite(initial_step_size)) {
            throw std::invalid_argument(
                "DualAveragingAdapter::init requires positive finite eps");
        }
        log_eps_ = std::log(initial_step_size);
        log_eps_bar_ = 0.0;       // Hoffman 2014 §3.4: eps_bar_0 = 1
        mu_ = std::log(10.0 * initial_step_size);
        m_ = 0;
        h_bar_ = 0.0;
        have_initial_ = true;
    }

    // One adapt step. Call AFTER each warmup MCMC iteration with the
    // accept probability of the just-completed proposal (clipped to
    // [0, 1] by caller).
    void update(double actual_accept) {
        if (!have_initial_) {
            throw std::runtime_error(
                "DualAveragingAdapter::update called before init()");
        }
        if (actual_accept < 0.0) actual_accept = 0.0;
        if (actual_accept > 1.0) actual_accept = 1.0;

        ++m_;  // m starts at 1 for first observed accept
        const double m_d = static_cast<double>(m_);

        // Step (a): update running mean of (target - actual) deviation.
        // Hoffman 2014 eq (5): H_bar_{m+1} = (1 - 1/(m + t0)) H_bar_m
        //                                    + (1/(m + t0)) (target - alpha_m)
        h_bar_ = (1.0 - 1.0 / (m_d + cfg_.t0)) * h_bar_
               + (1.0 / (m_d + cfg_.t0)) * (cfg_.target_accept - actual_accept);

        // Step (b): set log(eps) directly from H_bar (raw signal).
        // log(eps_m) = mu - sqrt(m)/gamma * H_bar_m
        log_eps_ = mu_ - std::sqrt(m_d) / cfg_.gamma * h_bar_;

        // Clamp eps to max_step_size (nuts-rs dual_avg.rs:59). Without this,
        // eps runs away on loose/drifting conditionals (the funnel-freeze
        // root cause). Bounding it also keeps trees deep enough for an
        // accurate accept stat so dual-averaging can pull eps back down.
        const double log_max = std::log(cfg_.max_step_size);
        if (log_eps_ > log_max) log_eps_ = log_max;

        // Step (c): Polyak-Ruppert average eps into eps_bar.
        // log(eps_bar_m) = m^{-kappa} log(eps_m) + (1 - m^{-kappa}) log(eps_bar_{m-1})
        const double weight = std::pow(m_d, -cfg_.kappa);
        log_eps_bar_ = weight * log_eps_ + (1.0 - weight) * log_eps_bar_;
    }

    // The "raw" eps to use for the NEXT warmup leapfrog. Updated
    // continuously during warmup.
    double current_step_size() const noexcept {
        return std::exp(log_eps_);
    }

    // The Polyak-averaged eps to use during sampling phase (frozen
    // after warmup).
    double sampling_step_size() const noexcept {
        return std::exp(log_eps_bar_);
    }

    // Reset all counters (called at start of fresh warmup window or
    // readapt(reset=true)).
    void reset() {
        m_ = 0;
        h_bar_ = 0.0;
        log_eps_ = 0.0;
        log_eps_bar_ = 0.0;
        mu_ = 0.0;
        have_initial_ = false;
    }

    // Diagnostics.
    std::size_t iter_count() const noexcept { return m_; }
    double h_bar() const noexcept { return h_bar_; }
    double mu() const noexcept { return mu_; }
    const DualAveragingConfig& config() const noexcept { return cfg_; }

    // Bitwise snapshot/restore for readapt_NUTS state preservation.
    // Snapshot is a POD struct serializable without aliasing.
    struct Snapshot {
        std::size_t m;
        double h_bar;
        double log_eps;
        double log_eps_bar;
        double mu;
        bool have_initial;
    };

    Snapshot snapshot() const noexcept {
        return {m_, h_bar_, log_eps_, log_eps_bar_, mu_, have_initial_};
    }

    void restore(const Snapshot& s) noexcept {
        m_ = s.m;
        h_bar_ = s.h_bar;
        log_eps_ = s.log_eps;
        log_eps_bar_ = s.log_eps_bar;
        mu_ = s.mu;
        have_initial_ = s.have_initial;
    }

private:
    DualAveragingConfig cfg_;

    std::size_t m_;      // step count m (Hoffman 2014 notation)
    double h_bar_;       // running deviation from target accept
    double log_eps_;     // raw log step size
    double log_eps_bar_; // Polyak-Ruppert smoothed log step size
    double mu_;          // anchor = log(10 * initial_eps)
    bool have_initial_;
};

// -----------------------------------------------------------------------
// Heuristic initial step size search (Hoffman 2014 Algorithm 4).
// Returns an eps such that the acceptance ratio is roughly 0.5 after
// one leapfrog step from a random momentum.
//
// Caller provides:
//   - theta_init: starting position
//   - mass_inv_diag: diagonal of M^{-1} (used to sample momentum)
//   - leapfrog_one_step: lambda (theta, p, grad, eps) -> (theta_new, p_new, grad_new, lp_new)
//   - log_density: lambda (theta) -> (lp, grad)
//
// Returns eps that approximately matches target_accept = 0.5 (or fails
// after max_doubles iterations and returns 1e-3 as a conservative
// fallback).
// -----------------------------------------------------------------------
// This function template is implemented in nuts_kernel.hpp where the
// leapfrog and log_density functor types are visible. The header only
// declares the interface.

}  // namespace internal
}  // namespace AI4BayesCode
