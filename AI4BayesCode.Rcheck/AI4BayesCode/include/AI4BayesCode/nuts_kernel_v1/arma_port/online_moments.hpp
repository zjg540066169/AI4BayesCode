// Copyright (C) 2026 AI4BayesCode contributors. GPL-3.0-or-later. See LICENSE at the repository root.
// online_moments.hpp — Continuous online Welford with exponential decay
//
// Algorithm ported from
//   walnuts/include/walnuts/online_moments.hpp (MIT, Bou-Rabee, Carpenter,
//   Kleppe, Liu 2025).
// Re-implemented in armadillo (different LA library); not a verbatim copy.
// Original MIT license attribution carried into THIRD_PARTY_LICENSES.md.
//
// Purpose:
//   Track an online estimate of mean + diagonal variance with an
//   exponential discount factor (nutpie-style). Used by the NUTS kernel's
//   mass matrix adaptation during 3-phase warmup (Phase II) and during
//   readapt_NUTS(adapt_mass=true) mini-windows.
//
// The estimator handles:
//   - Scalar Welford (mean + sample variance) for step-size acceptance rate
//   - Diagonal vector Welford (mean + diagonal variance) for mass matrix
//
// Detail-balance contract: mass matrix MUST be frozen during steady-state
// sampling. Caller responsible for not calling observe() outside adaptation
// windows.

#pragma once

#include <armadillo>
#include <stdexcept>
#include <cmath>

namespace AI4BayesCode {
namespace internal {

// -----------------------------------------------------------------------
// WelfordAccumulator — scalar online Welford (no discount).
// Used for acceptance-rate tracking during dual averaging.
// -----------------------------------------------------------------------
class WelfordAccumulator {
public:
    WelfordAccumulator() : n_(0), mean_(0.0), m2_(0.0) {}

    void observe(double x) {
        ++n_;
        const double delta = x - mean_;
        mean_ += delta / static_cast<double>(n_);
        const double delta2 = x - mean_;
        m2_ += delta * delta2;
    }

    std::size_t count() const noexcept { return n_; }
    double mean() const noexcept { return mean_; }

    double sample_variance() const noexcept {
        return (n_ > 1)
            ? m2_ / static_cast<double>(n_ - 1)
            : std::numeric_limits<double>::quiet_NaN();
    }

    void reset() {
        n_ = 0;
        mean_ = 0.0;
        m2_ = 0.0;
    }

private:
    std::size_t n_;
    double mean_;
    double m2_;
};

// -----------------------------------------------------------------------
// OnlineMoments — discounted vector Welford for mass matrix adaptation.
//
// API mirrors walnuts::OnlineMoments. discount_factor ∈ (0, 1]:
//   1 = no discount (standard Welford)
//   0 = complete forget (only most recent observation matters)
// Typical use: discount_factor close to 1 (e.g., 0.99 per draw).
//
// After N observations y[0], ..., y[N-1]:
//   weight[n] = discount_factor^(N - n - 1)
//   mean = sum(weight * y) / sum(weight)
//   variance = sum(weight * (y - mean)^2) / sum(weight)
// -----------------------------------------------------------------------
class OnlineMoments {
public:
    OnlineMoments() : discount_factor_(std::numeric_limits<double>::quiet_NaN()),
                       weight_(0.0) {}

    // Seed-from-prior constructor. init_weight is the effective sample
    // count this prior represents (positive).
    OnlineMoments(double init_weight,
                  const arma::vec& init_mean,
                  const arma::vec& init_variance)
        : discount_factor_(std::numeric_limits<double>::quiet_NaN()),
          weight_(init_weight),
          mean_(init_mean),
          sum_sq_dev_(init_weight * init_variance) {
        if (!(init_weight > 0)) {
            throw std::invalid_argument(
                "OnlineMoments: init_weight must be positive");
        }
        if (init_mean.n_elem != init_variance.n_elem) {
            throw std::invalid_argument(
                "OnlineMoments: init_mean and init_variance size mismatch");
        }
    }

    // Set the discount factor for the next observe() call.
    void set_discount_factor(double df) {
        if (!(df > 0.0 && df <= 1.0)) {
            throw std::invalid_argument(
                "OnlineMoments: discount_factor must be in (0, 1]");
        }
        discount_factor_ = df;
    }

    // Update with observation y. Discount factor must have been set.
    void observe(const arma::vec& y) {
        if (std::isnan(discount_factor_)) {
            throw std::runtime_error(
                "OnlineMoments::observe called before set_discount_factor");
        }
        // First observation initializes mean to y itself.
        if (mean_.n_elem == 0) {
            mean_ = arma::vec(y.n_elem, arma::fill::zeros);
            sum_sq_dev_ = arma::vec(y.n_elem, arma::fill::zeros);
        }
        if (y.n_elem != mean_.n_elem) {
            throw std::invalid_argument(
                "OnlineMoments::observe size mismatch");
        }
        const arma::vec delta = y - mean_;
        weight_ = discount_factor_ * weight_ + 1.0;
        mean_ += delta / weight_;
        // sum_sq_dev update uses (y - new_mean), not (y - old_mean)
        // — this is what makes Welford numerically stable.
        sum_sq_dev_ = discount_factor_ * sum_sq_dev_ + delta % (y - mean_);
    }

    // Convenience: set discount factor and observe in one call.
    void discount_observe(double df, const arma::vec& y) {
        set_discount_factor(df);
        observe(y);
    }

    // Current discounted mean estimate.
    const arma::vec& mean() const noexcept { return mean_; }

    // Current discounted variance estimate.
    // Returns vector of 1s if no observations yet (safe identity metric).
    arma::vec variance() const {
        if (weight_ <= 0.0) {
            // Safe identity default for an empty estimator.
            return arma::vec(mean_.n_elem > 0 ? mean_.n_elem : 0,
                             arma::fill::ones);
        }
        return sum_sq_dev_ / weight_;
    }

    // Reset to empty state (caller must re-seed via constructor if needed).
    void reset() {
        weight_ = 0.0;
        mean_.reset();
        sum_sq_dev_.reset();
    }

    // Diagnostics.
    double current_weight() const noexcept { return weight_; }
    std::size_t dim() const noexcept { return mean_.n_elem; }

private:
    double discount_factor_;
    double weight_;
    arma::vec mean_;
    arma::vec sum_sq_dev_;
};

}  // namespace internal
}  // namespace AI4BayesCode
