/*================================================================================
 *  block_mcmc: stateful modular MCMC for composable Gibbs samplers
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under the GNU General Public License v2.0 or later
 *  (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
 *================================================================================
 *
 *  elliptical_slice_sampling_block.hpp -- generic Elliptical Slice
 *  Sampling (Murray, Adams, MacKay 2010) for any latent-Gaussian model.
 *
 *  NAME DISAMBIGUATION
 *  ===================
 *  We use the full name `elliptical_slice_sampling` instead of the more
 *  common "ESS" abbreviation to avoid collision with "Effective Sample
 *  Size" which pervades MCMC diagnostics vocabulary. ESS-the-algorithm
 *  and ESS-the-diagnostic are fundamentally different things; the
 *  explicit name makes code + logs unambiguous.
 *
 *  WHAT THIS BLOCK DOES
 *  ====================
 *  Given:
 *    - a latent-Gaussian prior on f: f ~ N(0, Sigma), where the Cholesky
 *      L (Sigma = L L^T) is provided via block_context (key configurable
 *      via config.L_chol_key);
 *    - an arbitrary likelihood log p(y | f, ...) supplied as a user
 *      lambda config.log_lik(f, ctx);
 *  the block's step() produces one Elliptical Slice Sampling update of
 *  f. No gradient information needed. No step-size tuning. Handles
 *  arbitrary cross-correlation in Sigma (which naive NUTS struggles with).
 *
 *  Murray-Adams-MacKay 2010 algorithm:
 *    1. Draw nu ~ N(0, Sigma) via nu = L z where z ~ N(0, I)
 *    2. Slice height: log_y = log U(0,1) + log_lik(f)
 *    3. Propose angle theta ~ Uniform(0, 2pi); initial bracket [theta_min, theta_max]
 *       = [theta - 2pi, theta]
 *    4. Loop:
 *         f' = f cos(theta) + nu sin(theta)
 *         if log_lik(f') > log_y: accept, return f'
 *         else: shrink bracket (theta > 0 => theta_max=theta; theta<0 => theta_min=theta)
 *               draw new theta ~ Uniform(theta_min, theta_max)
 *       Guaranteed to accept (shrinking interval).
 *
 *  REUSE SCOPE
 *  -----------
 *  Works for any latent-Gaussian-with-arbitrary-likelihood model, not
 *  just GPs. Reference use cases:
 *    - GP regression / classification / Poisson regression (L from a
 *      kernel Cholesky refresher -- see examples/GPRegression.cpp)
 *    - CAR / ICAR / GMRF spatial models (L from graph Laplacian)
 *    - Intrinsic GMRF temporal smoothing (L from random-walk penalty)
 *    - Any latent Gaussian model in the Rue-Held 2005 book
 *
 *  This block is DELIBERATELY LIBGP-AGNOSTIC: it does not include any
 *  kernel library. The Cholesky L is supplied via shared_data; the
 *  user's wrapper (Tier A) is responsible for refreshing L when
 *  covariance hyperparameters change (via a deterministic refresher in
 *  shared_data).
 *
 *  SHARED_DATA CONVENTION FOR L
 *  ----------------------------
 *  Sigma is N x N; L is N x N lower-triangular. Stored in shared_data as
 *  a flat arma::vec of length N^2 in COLUMN-MAJOR order (matching arma +
 *  Eigen default layouts). The block reshapes via arma::mat(ptr, N, N,
 *  copy_aux_mem=false, strict=true) for zero-copy view access.
 *
 *  JUSTIFICATION (Check #16): Exception 1 from codegen.md section 2b --
 *  specialized latent-Gaussian sampler; NUTS on f with Sigma-correlated
 *  posterior suffers from step-size collapse (known issue).
 *  Library-level parity test: tests_autodiff/block_tests/
 *  test_elliptical_slice_sampling_block.cpp (fix L = I, Gaussian
 *  likelihood y = f + N(0, sigma^2); ESS draws of f should match analytical
 *  N((I+sigma^2I)^{-1} y, sigma^2/(1+sigma^2) I) mean and covariance at 10k draws).
 *================================================================================*/

#ifndef AI4BAYESCODE_ELLIPTICAL_SLICE_SAMPLING_BLOCK_HPP
#define AI4BAYESCODE_ELLIPTICAL_SLICE_SAMPLING_BLOCK_HPP

#include "block_sampler.hpp"

#include <cmath>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace AI4BayesCode {

/**
 * @brief Configuration bundle for elliptical_slice_sampling_block.
 */
struct elliptical_slice_sampling_block_config {
    /// Unique name for this block within its composite. Also the
    /// shared_data key under which this block's current f is written.
    std::string name;

    /// Dimensionality of the latent f.
    std::size_t N = 0;

    /// shared_data key for the prior-covariance Cholesky L (lower
    /// triangular), stored as a flat arma::vec of length N^2 in
    /// column-major order. The user's wrapper is responsible for
    /// refreshing this key when covariance hyperparameters change.
    std::string L_chol_key = "L_chol";

    /// User-supplied log-likelihood: log p(y | f, ctx). May read any
    /// declared dependency from ctx (e.g. noise sigma, data y, design
    /// matrix X if the likelihood involves a design). Returns a finite
    /// double; non-finite is treated as -Inf (proposal rejected inside
    /// the slice shrink loop).
    std::function<double(const arma::vec& f, const block_context& ctx)> log_lik;

    /// Initial f. Length N. Default = zeros.
    arma::vec initial_f;

    /// Safety cap on the slice-shrink inner loop. ESS is guaranteed to
    /// accept eventually, but a pathological log_lik (returning -Inf on
    /// every proposal) could in principle loop forever. We abort and
    /// return the current f unchanged after this many shrink iterations.
    std::size_t max_shrink_iter = 100;
};

/**
 * @brief Elliptical Slice Sampling block for latent Gaussian models.
 *        See file header for motivation and usage.
 */
class elliptical_slice_sampling_block : public block_sampler {
public:
    explicit elliptical_slice_sampling_block(
        elliptical_slice_sampling_block_config cfg)
        : cfg_(std::move(cfg))
    {
        if (cfg_.name.empty()) {
            throw std::invalid_argument(
                "elliptical_slice_sampling_block: name must be non-empty");
        }
        if (cfg_.N == 0) {
            throw std::invalid_argument(
                "elliptical_slice_sampling_block: N must be > 0");
        }
        if (!cfg_.log_lik) {
            throw std::invalid_argument(
                "elliptical_slice_sampling_block: log_lik is required");
        }
        // Initialize f.
        if (cfg_.initial_f.n_elem == 0) {
            f_.set_size(cfg_.N);
            f_.zeros();
        } else if (cfg_.initial_f.n_elem == cfg_.N) {
            f_ = cfg_.initial_f;
        } else {
            throw std::invalid_argument(
                "elliptical_slice_sampling_block: initial_f must be empty "
                "or length N");
        }
    }

    // ---- block_sampler interface --------------------------------------

    void set_context(const block_context& ctx) override {
        // Copy the context for step() to use. Per the design contract
        // (block_sampler.hpp), we must not retain pointers into ctx.
        context_ = ctx;
    }

    void step(std::mt19937_64& rng) override {
        // Locate L (the Cholesky of the prior covariance) in ctx.
        auto lit = context_.find(cfg_.L_chol_key);
        if (lit == context_.end()) {
            throw std::runtime_error(
                "elliptical_slice_sampling_block '" + cfg_.name +
                "': missing L_chol key '" + cfg_.L_chol_key +
                "' in context");
        }
        const arma::vec& L_flat = lit->second;
        if (L_flat.n_elem != cfg_.N * cfg_.N) {
            throw std::runtime_error(
                "elliptical_slice_sampling_block '" + cfg_.name +
                "': L_chol length " + std::to_string(L_flat.n_elem) +
                " does not match N^2 = " +
                std::to_string(cfg_.N * cfg_.N));
        }
        // Zero-copy view of L as NxN matrix (column-major).
        arma::mat L(const_cast<double*>(L_flat.memptr()), cfg_.N, cfg_.N,
                    /*copy_aux_mem=*/false, /*strict=*/true);

        std::normal_distribution<double>       std_norm(0.0, 1.0);
        std::uniform_real_distribution<double> unif(0.0, 1.0);

        // 1. Prior sample nu = L z
        arma::vec z(cfg_.N);
        for (std::size_t i = 0; i < cfg_.N; ++i) z[i] = std_norm(rng);
        arma::vec nu = L * z;

        // 2. Slice threshold
        double log_lik_cur = cfg_.log_lik(f_, context_);
        if (!std::isfinite(log_lik_cur)) {
            // If current state has log_lik = -Inf we cannot shrink
            // meaningfully. Return without updating and let the outer
            // chain recover on the next sweep.
            record_history_();
            return;
        }
        double log_y = std::log(unif(rng)) + log_lik_cur;

        // 3. Initial angle + bracket
        const double TWO_PI = 2.0 * M_PI;
        double theta     = unif(rng) * TWO_PI;
        double theta_min = theta - TWO_PI;
        double theta_max = theta;

        // 4. Shrink loop
        arma::vec f_prime(cfg_.N);
        for (std::size_t iter = 0; iter < cfg_.max_shrink_iter; ++iter) {
            const double ct = std::cos(theta);
            const double st = std::sin(theta);
            f_prime = f_ * ct + nu * st;

            const double lp = cfg_.log_lik(f_prime, context_);
            if (std::isfinite(lp) && lp > log_y) {
                // Accept
                f_ = f_prime;
                record_history_();
                return;
            }
            // Shrink
            if (theta < 0.0) theta_min = theta;
            else             theta_max = theta;
            std::uniform_real_distribution<double>
                unif_iv(theta_min, theta_max);
            theta = unif_iv(rng);
            // Small safety: if bracket collapses to machine zero we
            // accept the current f (no move). This is extraordinarily
            // rare; ESS mathematically guarantees acceptance.
            if (theta_max - theta_min < 1e-12) {
                record_history_();
                return;
            }
        }
        // Shrink cap reached -- leave f unchanged.
        record_history_();
    }

    const arma::vec& current() const override { return f_; }

    void set_current(const arma::vec& theta) override {
        if (theta.n_elem != cfg_.N) {
            throw std::invalid_argument(
                "elliptical_slice_sampling_block::set_current: wrong length");
        }
        f_ = theta;
    }

    const std::string& name() const noexcept override { return cfg_.name; }

    std::size_t dim() const noexcept override { return cfg_.N; }

    // ---- History ------------------------------------------------------

    history_map get_history() const override {
        return detail::make_history_map(cfg_.name, history_, f_);
    }

    std::size_t history_size() const noexcept override {
        return history_.empty() ? 1 : history_.size();
    }

    void clear_history() override { history_.clear(); }

private:
    void record_history_() {
        if (keep_history_) history_.push_back(f_);
    }

    elliptical_slice_sampling_block_config cfg_;
    arma::vec                              f_;
    block_context                          context_;
    std::vector<arma::vec>                 history_;
};

} // namespace AI4BayesCode

#endif // AI4BAYESCODE_ELLIPTICAL_SLICE_SAMPLING_BLOCK_HPP
