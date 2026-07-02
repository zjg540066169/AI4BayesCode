/*================================================================================
 *  block_mcmc: stateful modular MCMC for composable Gibbs samplers
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under the GNU General Public License v2.0 or later
 *  (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
 *================================================================================
 *
 *  probit_aug_block.hpp -- closed-form Gibbs leaf for the Albert-Chib
 *                          (1993) data-augmentation latent z in any
 *                          probit-link binary model:
 *
 *      y_i        ~ Bernoulli(p_i),  p_i = Phi(mu_i + offset_i)
 *      z_i | rest ~ N(mu_i + offset_i, 1) truncated to:
 *                       (0, +inf)   if y_i = 1
 *                       (-inf, 0)   if y_i = 0
 *
 *  Composing this block with any Gaussian-likelihood downstream block
 *  (bart_block / nuts_block over a linear predictor / GP latent / ...)
 *  recovers the standard Bayesian probit sampler. The downstream block
 *  reads the latent z (or z minus the offset) as its working response.
 *
 *  WHY THIS BLOCK EXISTS
 *  =====================
 *  Albert-Chib data augmentation is *the* textbook closed-form Gibbs
 *  step for probit binary likelihoods (Albert & Chib 1993, JASA). Before
 *  this block, every probit example in the AI4BayesCode tree had to
 *  inline the truncated-normal step in its wrapper class -- which works,
 *  but loses uniformity and makes the Gibbs DAG harder to declare. This
 *  block provides a reusable, library-blessed Gibbs leaf so that probit
 *  models compose cleanly:
 *
 *      composite "ProbitWhatever":
 *        child(0) z      probit_aug_block (this block)
 *        child(1) <mean> nuts_block / bart_block / linear_block / ...
 *
 *  Use cases:
 *    - Probit linear regression: Albert-Chib + NUTS on beta with
 *      Normal prior. See `examples/ProbitRegression.cpp` for the
 *      reference template.
 *    - Probit BART: Albert-Chib + bart_block (with cfg.binary = true
 *      so the leaf prior tau matches BART::pbart's 3 / (k * sqrt(ntree))
 *      formula). See `examples/probit_BART_model.cpp`.
 *    - Probit GLM with shrinkage: Albert-Chib + nuts_block on beta
 *      under horseshoe / regularised horseshoe.
 *    - Hierarchical probit: Albert-Chib + nuts_block over a multilevel
 *      mean structure.
 *
 *  DESIGN
 *  ======
 *  - Inputs (read from block_context per `set_context`):
 *      y_key      : binary y in {0.0, 1.0}, length n_obs (REQUIRED).
 *      mu_key     : linear predictor on the probit (z) scale, length
 *                   n_obs (REQUIRED). Written by the sibling Gaussian
 *                   block whose output represents E[z_i | rest].
 *      offset_key : OPTIONAL additive offset on the same scale as mu.
 *                   Length 1 (scalar broadcast) OR length n_obs
 *                   (per-observation). When the key is absent or
 *                   `offset_key` is empty (default), no offset.
 *  - Output (written under cfg.name in shared_data after each step()):
 *      z          : length n_obs, the freshly drawn truncated-normal
 *                   latent. Conditionally independent across i, so the
 *                   draw is fully vectorised (no inner loop dependency).
 *  - sigma is FIXED at 1.0 by probit identifiability and is NOT a
 *    config knob. Anyone wanting sigma != 1 is targeting a different
 *    likelihood (Tobit / censored Gaussian) and should write a
 *    different block.
 *
 *  NUMERICAL STABILITY
 *  -------------------
 *  Truncated-normal sampler uses Robert (1995) algorithm for TN(0, 1,
 *  [b, +inf)):
 *    - For b < 0.5: plain rejection from N(0, 1).
 *    - For b >= 0.5: exponential rejection with optimal rate
 *      alpha = (b + sqrt(b^2 + 4)) / 2  (Robert 1995 Section 2.2).
 *  Both regimes have well-bounded acceptance rate; no "get stuck"
 *  pathology possible. A safety guard (kMaxTrials) returns b + epsilon
 *  in the astronomically rare case the rejection loop fails, so the
 *  block never returns a non-finite value.
 *
 *  CONDITIONAL INDEPENDENCE (no sequential update needed)
 *  ------------------------------------------------------
 *  Unlike binary_gibbs_block (where z_i's may be dependent through
 *  shared parameters they all read), the Albert-Chib step is over
 *  CONDITIONALLY-INDEPENDENT z_i's given (y, mu, offset). Each z_i
 *  depends only on (y_i, mu_i, offset_i). We therefore sample all
 *  z_i's in a SINGLE PASS without re-reading the context — the order
 *  is irrelevant to the stationary distribution.
 *
 *  CHECK #15 (library parity test)
 *  -------------------------------
 *  See `tests_autodiff/block_tests/test_probit_aug_block.cpp`. Tests
 *  empirical mean and variance of the drawn z's against the closed-
 *  form TN(mu, 1, [bound]) moments at three regimes:
 *    (1) y = 1, mu = 0    : E[z] = phi(0)/Phi(0) = sqrt(2/pi),
 *                           Var[z] = 1 - 2*phi(0)*0/Phi(0) - (E[z])^2
 *    (2) y = 1, mu = 2    : moderately positive shifted TN
 *    (3) y = 0, mu = -1   : moderately negative shifted TN
 *  Tolerance 5% relative on E[z] at 10000 draws (Check #15 standard).
 *
 *  CHECK #17 (no hand-written distribution samplers)
 *  -------------------------------------------------
 *  This block IS the library-blessed Gibbs sampler for probit data
 *  augmentation. Any wrapper that does inline truncated-normal sampling
 *  for an Albert-Chib step should be refactored to use this block
 *  instead. Whitelisted under codegen_priors.md Exception 3 (textbook
 *  closed-form vector conjugate sample).
 *================================================================================*/

#ifndef AI4BAYESCODE_PROBIT_AUG_BLOCK_HPP
#define AI4BAYESCODE_PROBIT_AUG_BLOCK_HPP

#include "block_sampler.hpp"

#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace AI4BayesCode {

/**
 * @brief Configuration bundle for probit_aug_block construction.
 */
struct probit_aug_block_config {
    /// Unique name for this block within its composite; also the key
    /// under which shared_data_t stores the drawn latent z.
    std::string name;

    /// Number of observations N (length of y, mu, and z).
    std::size_t n_obs = 0;

    /// shared_data key for the binary outcome vector y (length n_obs,
    /// values 0.0 or 1.0). REQUIRED.
    std::string y_key;

    /// shared_data key for the linear predictor mu on the probit (z)
    /// scale, length n_obs. REQUIRED. Written by the sibling Gaussian
    /// block whose output represents E[z_i | rest] (= f_bart for BART,
    /// X*beta for linear regression, etc.).
    std::string mu_key;

    /// OPTIONAL shared_data key for an additive offset on the same
    /// scale as mu. May be length 1 (scalar broadcast) or length n_obs
    /// (per-observation). Empty (default) = no offset.
    std::string offset_key;

    /// Initial z values (length n_obs). If empty, the constructor
    /// initialises z to the sign-of-(2y-1) (a finite valid first
    /// draw); the very first step() will overwrite this.
    arma::vec initial_z;
};

namespace detail {

// Sample W ~ TN(0, 1, [b, +inf)) using Robert (1995) algorithm.
//   - For b < 0.5: plain rejection from N(0, 1).
//   - For b >= 0.5: exponential rejection (Robert 1995 Section 2.2).
// Returns a finite value w with w > b on success; falls back to b + eps
// on pathological non-convergence (extremely unlikely; safety guard).
inline double sample_std_truncnorm_lower(double b, std::mt19937_64& rng) {
    constexpr int kMaxTrials = 10000;
    if (b < 0.5) {
        std::normal_distribution<double> norm(0.0, 1.0);
        for (int trial = 0; trial < kMaxTrials; ++trial) {
            const double w = norm(rng);
            if (w > b) return w;
        }
        return b + 1e-12;
    }
    // b >= 0.5 : exponential proposal at optimal rate.
    const double alpha = 0.5 * (b + std::sqrt(b * b + 4.0));
    std::uniform_real_distribution<double> uniform(
        std::numeric_limits<double>::min(), 1.0);
    for (int trial = 0; trial < kMaxTrials; ++trial) {
        const double u = uniform(rng);
        const double z = b - std::log(u) / alpha;          // z = b + Exp(alpha)
        const double v = uniform(rng);
        const double rho = std::exp(-0.5 * (z - alpha) * (z - alpha));
        if (v < rho) return z;
    }
    return b + 1e-12;
}

// Sample z ~ N(mu, 1) truncated to:
//   (0, +inf)  if y == 1
//   (-inf, 0)  if y == 0
// Implementation:
//   y = 1 -> z = mu + W,  W ~ TN(0, 1, [-mu, +inf))
//   y = 0 -> z = mu - V,  V ~ TN(0, 1, [mu,  +inf))
inline double sample_truncnorm_for_y(double mu, double y,
                                      std::mt19937_64& rng) {
    if (y >= 0.5) {
        return mu + sample_std_truncnorm_lower(-mu, rng);
    }
    return mu - sample_std_truncnorm_lower(mu, rng);
}

}  // namespace detail

/**
 * @brief Closed-form Gibbs leaf for the Albert-Chib (1993) latent z in
 *        a probit binary model. See the file header for design /
 *        composition / usage details.
 */
class probit_aug_block : public block_sampler {
public:
    explicit probit_aug_block(probit_aug_block_config cfg)
        : cfg_(std::move(cfg))
    {
        if (cfg_.n_obs == 0) {
            throw std::invalid_argument(
                "probit_aug_block: n_obs must be > 0");
        }
        if (cfg_.name.empty()) {
            throw std::invalid_argument(
                "probit_aug_block: name must be non-empty");
        }
        if (cfg_.y_key.empty()) {
            throw std::invalid_argument(
                "probit_aug_block: y_key must be non-empty");
        }
        if (cfg_.mu_key.empty()) {
            throw std::invalid_argument(
                "probit_aug_block: mu_key must be non-empty");
        }
        if (!cfg_.initial_z.is_empty() &&
            cfg_.initial_z.n_elem != cfg_.n_obs) {
            throw std::invalid_argument(
                "probit_aug_block: initial_z length must equal n_obs");
        }
        values_.set_size(cfg_.n_obs);
        if (cfg_.initial_z.is_empty()) {
            values_.zeros();   // sign-of-(2y-1) snap on first step;
                               // safe finite default until then.
        } else {
            for (std::size_t i = 0; i < cfg_.n_obs; ++i) {
                values_[i] = cfg_.initial_z[i];
            }
        }
    }

    // ---- block_sampler interface ----------------------------------------

    void set_context(const block_context& ctx) override {
        // COPY (per block_sampler.hpp design contract).
        context_ = ctx;
    }

    void step(std::mt19937_64& rng) override {
        // Read y and mu from context. They MUST have length n_obs.
        const auto it_y = context_.find(cfg_.y_key);
        if (it_y == context_.end()) {
            throw std::runtime_error(
                "probit_aug_block '" + cfg_.name
                + "': y_key '" + cfg_.y_key + "' not found in context");
        }
        const auto it_mu = context_.find(cfg_.mu_key);
        if (it_mu == context_.end()) {
            throw std::runtime_error(
                "probit_aug_block '" + cfg_.name
                + "': mu_key '" + cfg_.mu_key + "' not found in context");
        }
        const arma::vec& y  = it_y->second;
        const arma::vec& mu = it_mu->second;
        if (y.n_elem != cfg_.n_obs) {
            throw std::runtime_error(
                "probit_aug_block '" + cfg_.name
                + "': y length = " + std::to_string(y.n_elem)
                + " but n_obs = " + std::to_string(cfg_.n_obs));
        }
        if (mu.n_elem != cfg_.n_obs) {
            throw std::runtime_error(
                "probit_aug_block '" + cfg_.name
                + "': mu length = " + std::to_string(mu.n_elem)
                + " but n_obs = " + std::to_string(cfg_.n_obs));
        }

        // Optional offset: scalar broadcast or per-observation.
        const arma::vec* offset_ptr = nullptr;
        bool offset_scalar = false;
        if (!cfg_.offset_key.empty()) {
            const auto it_off = context_.find(cfg_.offset_key);
            if (it_off != context_.end()) {
                if (it_off->second.n_elem == 1) {
                    offset_scalar = true;
                } else if (it_off->second.n_elem == cfg_.n_obs) {
                    offset_scalar = false;
                } else {
                    throw std::runtime_error(
                        "probit_aug_block '" + cfg_.name
                        + "': offset_key '" + cfg_.offset_key
                        + "' must have length 1 or n_obs (got "
                        + std::to_string(it_off->second.n_elem) + ")");
                }
                offset_ptr = &it_off->second;
            }
        }
        const double offset_scalar_val =
            (offset_ptr && offset_scalar) ? (*offset_ptr)[0] : 0.0;

        // Vectorised draw (z_i conditionally independent given y, mu, offset).
        for (std::size_t i = 0; i < cfg_.n_obs; ++i) {
            const double off_i = offset_ptr
                ? (offset_scalar ? offset_scalar_val : (*offset_ptr)[i])
                : 0.0;
            const double mean_i = mu[i] + off_i;
            // Validate y_i in {0, 1} (defensive — bad input would
            // produce silently-wrong z otherwise).
            const double y_i = y[i];
            if (!(y_i == 0.0 || y_i == 1.0)) {
                throw std::runtime_error(
                    "probit_aug_block '" + cfg_.name
                    + "': y[" + std::to_string(i) + "] = "
                    + std::to_string(y_i) + " is not 0/1");
            }
            values_[i] = detail::sample_truncnorm_for_y(mean_i, y_i, rng);
        }

        if (keep_history_) {
            history_buf_.push_back(values_);
        }
    }

    const arma::vec& current() const override { return values_; }

    void set_current(const arma::vec& theta) override {
        if (theta.n_elem != cfg_.n_obs) {
            throw std::invalid_argument(
                "probit_aug_block::set_current: wrong length");
        }
        values_ = theta;
    }

    const std::string& name() const noexcept override { return cfg_.name; }

    std::size_t dim() const noexcept override { return cfg_.n_obs; }

    // ---- History overrides ----------------------------------------------

    history_map get_history() const override {
        return detail::make_history_map(cfg_.name, history_buf_, values_);
    }

    std::size_t history_size() const noexcept override {
        return history_buf_.empty() ? 1 : history_buf_.size();
    }

    void clear_history() override {
        history_buf_.clear();
    }

private:
    probit_aug_block_config cfg_;
    arma::vec               values_;
    block_context           context_;
    std::vector<arma::vec>  history_buf_;
};

}  // namespace AI4BayesCode

#endif  // AI4BAYESCODE_PROBIT_AUG_BLOCK_HPP
