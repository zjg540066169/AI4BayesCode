/*================================================================================
 *  AI4BayesCode/poisson_multinomial_aug_block.hpp
 *      Poisson-multinomial gamma-augmentation block for multinomial
 *      logistic BART regression via genBART, under the REFERENCE-
 *      CATEGORY IDENTIFIED parameterization (category 0 fixed as
 *      reference, f^(0)(x) := 1).
 *
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under the GNU General Public License v2.0 or later
 *  (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
 *================================================================================
 *
 *  MODEL CONTEXT
 *  =============
 *  For n observations at covariate values x_i, each falling in one of
 *  C categories indexed j = 0, ..., C-1 (category 0 is the reference),
 *  the identified multinomial logistic likelihood is
 *
 *      p(y_i = j | x_i, f^(1), ..., f^(C-1))
 *          = f^(j)(x_i) / (1 + sum_{l=1..C-1} f^(l)(x_i)),   j >= 1
 *          = 1          / (1 + sum_{l=1..C-1} f^(l)(x_i)),   j == 0
 *
 *  where each f^(j)(x) = exp(r_j(x)) > 0 is the exponential of the
 *  corresponding category's genBART linear-predictor output.
 *
 *  GAMMA AUGMENTATION
 *  ==================
 *  Introduce a latent phi_i > 0 per observation with full conditional
 *
 *      phi_i | y_i, f  ~  Gamma(n_i, 1 + sum_{l=1..C-1} exp(r_l(x_i))).
 *
 *  For the common single-observation case (n_i = 1) this reduces to
 *
 *      phi_i | y_i, f  ~  Exp(1 + sum_{l=1..C-1} exp(r_l(x_i))).
 *
 *  Given phi_i, the augmented likelihood factorises into C-1
 *  conditionally independent Poisson-like likelihoods (one per non-
 *  reference category j) of the form
 *
 *      likelihood_j  ~  prod_i  [exp(r_j(x_i))]^{u_{i,j}}
 *                              * exp[-phi_i * exp(r_j(x_i))]
 *                    = prod_i  exp(u_{i,j} * r_j(x_i) - phi_i * exp(r_j(x_i)))
 *
 *  with u_{i,j} = [y_i == j]. Taking logs, the per-observation log-
 *  likelihood contribution for r_j evaluated at x_i is
 *
 *      u_{i,j} * r_j(x_i) - phi_i * exp(r_j(x_i)).
 *
 *  Rewriting with an additive offset o_i := log(phi_i) and lambda_i :=
 *  r_j(x_i) + o_i gives
 *
 *      u_{i,j} * (lambda_i - o_i) - exp(lambda_i)
 *    = u_{i,j} * lambda_i - exp(lambda_i) - u_{i,j} * o_i
 *
 *  The final -u_{i,j} * o_i term is constant in the r_j tree parameters,
 *  so it drops from the BART tree-update acceptance ratio. What remains
 *  is EXACTLY genBART's `poisson_lik::log_f(y=u_{i,j}, lambda) =
 *  y*lambda - exp(lambda)`. Conclusion: the block feeds
 *
 *      genbart_block(poisson_lik,
 *                    Y_key = "u_j",          (category indicator)
 *                    offset_key = "log_phi")
 *
 *  and this wrapper writes both u_j indicators and log_phi into
 *  shared_data before the genbart children step. The C-1 genbart_block
 *  children then run independent poisson RJMCMC sweeps on their trees,
 *  which jointly compose the identified multinomial logistic BART.
 *
 *  For the C = 2 binary logistic case there is exactly ONE non-
 *  reference category (j = 1), and this block reduces to a single
 *  auxiliary variable per observation. Users who prefer the direct
 *  sigmoid parameterisation can bypass this block entirely and use
 *  genbart_block(logistic_lik) instead; the two approaches yield the
 *  same marginal posterior on pi(x) = P(y=1|x) but the logistic_lik
 *  path is simpler and has no augmentation latent, while this block's
 *  path scales to C >= 3 without code changes.
 *
 *  HISTORY & ATTRIBUTION
 *  =====================
 *  The Poisson-multinomial transformation (a.k.a. "gamma trick") is
 *  classical, appearing independently in
 *    - Baker (1994) "The multinomial-Poisson transformation", The
 *      Statistician 43(4):495-504.
 *    - Forster (2010) "Bayesian inference for Poisson and multinomial
 *      log-linear models", Statistical Methodology 7(3):210-224.
 *    - Walker (2011) "Posterior sampling when the normalizing constant
 *      is unknown", Comm. Stat. Sim. Comput. 40(5):784-792.
 *    - Caron & Doucet (2012) "Efficient Bayesian inference for
 *      generalized Bradley-Terry models", J. Comp. Graph. Stat.
 *      21(1):174-196.
 *
 *  Murray (2021) "Log-Linear Bayesian Additive Regression Trees for
 *  Multinomial Logistic and Count Regression Models", JASA 116(534):
 *  756-769, adapted this augmentation to BART via a C-1 coupled log-
 *  linear ensemble architecture with a symmetric conjugate leaf prior
 *  (the GIG mixture P_lambda(c, d)) and Bayesian backfitting.
 *
 *  THIS implementation keeps Murray's C-1 ensemble architecture but
 *  replaces the log-linear backfitting kernel with Linero (2022)
 *  "Generalized Bayesian Additive Regression Trees Models: Beyond
 *  Conditional Conjugacy" (arXiv:2202.09924) reversible-jump MCMC +
 *  Laplace leaf proposals, as shipped in AI4BayesCode's genBART
 *  kernel (include/AI4BayesCode/genbart_block.hpp, genbart/src/).
 *
 *  WHAT THIS BLOCK DOES
 *  ====================
 *  One step() of poisson_multinomial_aug_block:
 *    1. Reads y and each r_j (j = 1 .. C-1) from block_context.
 *    2. Computes rate_i = 1 + sum_{j=1..C-1} exp(r_j(x_i)).
 *    3. Samples phi_i ~ Gamma(n_i, rate_i) for i = 1 .. N
 *       (reducing to Exp(rate_i) when n_i = 1).
 *    4. Derives u_{i,j} = [y_i == j] from the current y.
 *    5. Writes log_phi under this block's name (so downstream
 *       genbart_blocks can reference it via offset_key = <this block's
 *       name>), and writes each u_j vector under a user-configured
 *       key u_keys[j-1].
 *
 *  The block's current() returns the length-N log_phi vector.
 *  Additional outputs (u_1, ..., u_{C-1}) are exposed via
 *  current_named_outputs() so the composite writes them back to
 *  shared_data after every step().
 *
 *  DAG WIRING (example for C = 3, non-reference classes {1, 2})
 *  ------------------------------------------------------------
 *  In the composite constructor:
 *
 *      impl_->data().declare_dependencies(
 *          "log_phi_aug", {"y", "r_1", "r_2"});
 *
 *      impl_->data().declare_dependencies("r_1", {"u_1", "log_phi_aug"});
 *      impl_->data().declare_dependencies("r_2", {"u_2", "log_phi_aug"});
 *
 *  In each genbart_block_config for the non-reference category j:
 *
 *      bart_cfg.name       = "r_" + std::to_string(j);
 *      bart_cfg.y_key      = "u_" + std::to_string(j);
 *      bart_cfg.offset_key = "log_phi_aug";
 *
 *  Gibbs order inside composite: poisson_multinomial_aug_block FIRST,
 *  then each genbart child.
 *
 *  JUSTIFICATION (Check #16): Exception 1 from codegen_priors.md §2b --
 *    discrete / latent augmentation; NUTS cannot target the continuous
 *    phi directly (the full conditional is Exp/Gamma with a rate that
 *    mixes continuous r and discrete y).
 *  Check #15 parity test at
 *    tests_autodiff/block_tests/test_poisson_multinomial_aug_block.cpp
 *  (to be added in Phase 7 verification).
 *
 *  NUMERICAL STABILITY
 *  -------------------
 *  The rate term 1 + sum exp(r_j) is computed directly. genBART's
 *  leaf prior keeps |r_j| bounded in practice (approximately
 *  N(0, a_0^2) marginal with a_0 ~ 1/sqrt(ntrees), so |r| < ~10 in
 *  realistic posterior draws, hence exp(r) < ~2e4). The block still
 *  applies defensive clamps: rate is floored at 1e-12 and capped at
 *  1e300; phi output is floored at 1e-300 before log (to avoid log(0)
 *  under pathological inputs). Under those clamps log_phi stays in
 *  (-690, 300).
 *================================================================================*/

#ifndef AI4BAYESCODE_POISSON_MULTINOMIAL_AUG_BLOCK_HPP
#define AI4BAYESCODE_POISSON_MULTINOMIAL_AUG_BLOCK_HPP

#include "block_sampler.hpp"

#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace AI4BayesCode {

/**
 * @brief Configuration bundle for poisson_multinomial_aug_block.
 *
 * The block's own name() key is also the shared_data key under which
 * log_phi is published. Downstream genbart_blocks should set their
 * offset_key to this block's name.
 */
struct poisson_multinomial_aug_block_config {
    /// Block name. Also the shared_data key under which log_phi is
    /// written. Choose a unique name per composite (e.g. "log_phi_aug").
    /// Downstream genbart_block's offset_key should be set to this
    /// name.
    std::string name = "log_phi_aug";

    /// Number of observations.
    std::size_t N = 0;

    /// Number of classes INCLUDING the reference (category 0). C >= 2.
    /// For binary logistic (used with C-1 = 1 non-reference ensemble):
    /// C = 2.
    std::size_t C = 2;

    /// shared_data key for the class-label vector y. Values are read
    /// as doubles but MUST be integer-valued in {0, 1, ..., C-1}.
    std::string y_key = "y";

    /// Optional shared_data key for the per-observation count vector
    /// n_i. If empty, all n_i = 1 (the common single-observation case,
    /// which reduces the Gamma augmentation to Exp). If non-empty, the
    /// block reads n_i from ctx and uses Gamma(n_i, rate).
    std::string n_key;

    /// shared_data keys for the r_j linear-predictor vectors,
    /// j = 1 .. C-1. Must have length C-1. Indexing: r_keys[j-1] is
    /// the key for r_j(x), the output of the genbart_block handling
    /// non-reference category j.
    std::vector<std::string> r_keys;

    /// shared_data keys where u_j indicators are written,
    /// j = 1 .. C-1. Must have length C-1. u_keys[j-1] will be
    /// populated with the length-N vector u_{i,j-1} = [y_i == j].
    /// These keys are what downstream genbart_block's y_key should
    /// reference.
    std::vector<std::string> u_keys;

    /// Initial y vector (integer labels in {0..C-1}, stored as
    /// doubles). Length N. Used (i) to seed the u_j indicators into
    /// shared_data at composite add_child time (before any step()
    /// runs), and (ii) as a fallback when set_context is called
    /// without a y_key entry.
    arma::vec initial_y;

    /// Initial log_phi vector. Length N. If empty, log_phi is
    /// initialized to zero (phi = 1, matching Exp(1) mean scale when
    /// r_j starts at zero).
    arma::vec initial_log_phi;
};

/**
 * @brief Gamma-augmentation block for multinomial / binary logistic
 *        BART under the reference-category identified parameterization
 *        when paired with genBART Poisson likelihoods. See file header
 *        for the derivation and DAG wiring.
 */
class poisson_multinomial_aug_block : public block_sampler {
public:
    explicit poisson_multinomial_aug_block(
        poisson_multinomial_aug_block_config cfg)
        : cfg_(std::move(cfg))
    {
        // ---- Validate config --------------------------------------------
        if (cfg_.name.empty()) {
            throw std::invalid_argument(
                "poisson_multinomial_aug_block: cfg.name must be non-empty");
        }
        if (cfg_.N == 0) {
            throw std::invalid_argument(
                "poisson_multinomial_aug_block: cfg.N must be > 0");
        }
        if (cfg_.C < 2) {
            throw std::invalid_argument(
                "poisson_multinomial_aug_block: cfg.C must be >= 2 "
                "(2 = binary logistic; C-1 non-reference categories)");
        }
        if (cfg_.r_keys.size() != cfg_.C - 1) {
            throw std::invalid_argument(
                "poisson_multinomial_aug_block: r_keys.size() must "
                "equal C-1");
        }
        if (cfg_.u_keys.size() != cfg_.C - 1) {
            throw std::invalid_argument(
                "poisson_multinomial_aug_block: u_keys.size() must "
                "equal C-1");
        }
        if (cfg_.initial_y.n_elem != cfg_.N) {
            throw std::invalid_argument(
                "poisson_multinomial_aug_block: initial_y length must "
                "equal N");
        }
        for (std::size_t i = 0; i < cfg_.N; ++i) {
            const double yi = cfg_.initial_y[i];
            if (!(yi >= 0.0) || !(yi <= static_cast<double>(cfg_.C - 1))
                || yi != std::floor(yi))
            {
                throw std::invalid_argument(
                    "poisson_multinomial_aug_block: initial_y entries "
                    "must be integer-valued in {0, 1, ..., C-1}");
            }
        }

        // ---- Cache y internally -----------------------------------------
        y_cache_ = cfg_.initial_y;

        // ---- Initialize log_phi -----------------------------------------
        if (cfg_.initial_log_phi.n_elem == 0) {
            log_phi_.set_size(cfg_.N);
            log_phi_.zeros();   // phi = 1, neutral start
        } else if (cfg_.initial_log_phi.n_elem == cfg_.N) {
            log_phi_ = cfg_.initial_log_phi;
        } else {
            throw std::invalid_argument(
                "poisson_multinomial_aug_block: initial_log_phi must "
                "be empty or of length N");
        }

        // ---- Compute initial u_j indicators from initial_y --------------
        u_cache_.resize(cfg_.C - 1);
        for (std::size_t j = 0; j < cfg_.C - 1; ++j) {
            u_cache_[j].set_size(cfg_.N);
            const double jval = static_cast<double>(j + 1);
            for (std::size_t i = 0; i < cfg_.N; ++i) {
                u_cache_[j][i] =
                    (y_cache_[i] == jval) ? 1.0 : 0.0;
            }
        }
    }

    // ---- block_sampler interface --------------------------------------

    void set_context(const block_context& ctx) override {
        // Copy; block contract forbids retaining pointers into ctx
        // beyond this call.
        context_ = ctx;

        // Refresh y from ctx if present.
        auto yit = context_.find(cfg_.y_key);
        if (yit != context_.end()) {
            if (yit->second.n_elem != cfg_.N) {
                throw std::runtime_error(
                    "poisson_multinomial_aug_block '" + cfg_.name
                    + "': y context length "
                    + std::to_string(yit->second.n_elem)
                    + " does not match N "
                    + std::to_string(cfg_.N));
            }
            y_cache_ = yit->second;
            recompute_u_indicators_();
        }
    }

    void step(std::mt19937_64& rng) override {
        // Gather the length-(C-1) r_j references from current context.
        std::vector<const arma::vec*> r_ptrs(cfg_.C - 1, nullptr);
        for (std::size_t j = 0; j < cfg_.C - 1; ++j) {
            auto it = context_.find(cfg_.r_keys[j]);
            if (it == context_.end()) {
                throw std::runtime_error(
                    "poisson_multinomial_aug_block '" + cfg_.name
                    + "': missing r key '" + cfg_.r_keys[j]
                    + "' in context");
            }
            if (it->second.n_elem != cfg_.N) {
                throw std::runtime_error(
                    "poisson_multinomial_aug_block '" + cfg_.name
                    + "': r key '" + cfg_.r_keys[j]
                    + "' has length "
                    + std::to_string(it->second.n_elem)
                    + "; expected N = " + std::to_string(cfg_.N));
            }
            r_ptrs[j] = &it->second;
        }

        // Optional n_i vector.
        const arma::vec* n_ptr = nullptr;
        if (!cfg_.n_key.empty()) {
            auto it = context_.find(cfg_.n_key);
            if (it != context_.end()) {
                if (it->second.n_elem != cfg_.N) {
                    throw std::runtime_error(
                        "poisson_multinomial_aug_block '" + cfg_.name
                        + "': n context length "
                        + std::to_string(it->second.n_elem)
                        + " does not match N "
                        + std::to_string(cfg_.N));
                }
                n_ptr = &it->second;
            }
        }

        // Sample phi_i ~ Gamma(n_i, rate_i), then write log_phi_i.
        //
        // C++ std parameterization: std::gamma_distribution<double>(shape, scale)
        //   shape = n_i,  scale = 1.0 / rate_i
        //   E[phi_i] = shape * scale = n_i / rate_i.
        // Matches the full conditional
        //   phi_i ~ Gamma(n_i, 1 + sum_{j=1..C-1} exp(r_j(x_i))),
        // where the second argument of Gamma is the RATE.
        std::exponential_distribution<double> exp_dist(1.0);
        constexpr double PHI_FLOOR = 1e-300;   // log(1e-300) = -690.78
        for (std::size_t i = 0; i < cfg_.N; ++i) {
            double rate = 1.0;   // reference-category f^(0) = 1
            for (std::size_t j = 0; j < cfg_.C - 1; ++j) {
                const double rij = (*r_ptrs[j])[i];
                rate += std::exp(rij);
            }
            // Defensive clamp.
            if (!std::isfinite(rate) || rate > 1e300) rate = 1e300;
            if (rate < 1e-12)                          rate = 1e-12;

            double phi_i;
            if (n_ptr == nullptr) {
                phi_i = exp_dist(rng) / rate;      // n_i = 1 default
            } else {
                const double ni = (*n_ptr)[i];
                if (ni == 1.0) {
                    phi_i = exp_dist(rng) / rate;
                } else {
                    std::gamma_distribution<double> gam(ni, 1.0 / rate);
                    phi_i = gam(rng);
                }
            }
            if (!std::isfinite(phi_i) || phi_i < PHI_FLOOR) {
                phi_i = PHI_FLOOR;
            }
            log_phi_[i] = std::log(phi_i);
        }

        // u_j indicators already reflect y_cache_ (refreshed in
        // set_context or set_y). No recomputation needed here.

        // Append to history if enabled.
        if (keep_history_) {
            history_log_phi_.push_back(log_phi_);
        }
    }

    const arma::vec& current() const override { return log_phi_; }

    /**
     * Overwrite log_phi. Used for warm-starting from a previous run.
     * The u_j indicators are derived from y and are NOT overwritten
     * here; call set_y to refresh y (and thereby the u_j cache).
     */
    void set_current(const arma::vec& theta) override {
        if (theta.n_elem != cfg_.N) {
            throw std::invalid_argument(
                "poisson_multinomial_aug_block::set_current: wrong "
                "length (expected N = " + std::to_string(cfg_.N)
                + ", got " + std::to_string(theta.n_elem) + ")");
        }
        log_phi_ = theta;
    }

    const std::string& name() const noexcept override { return cfg_.name; }

    std::size_t dim() const noexcept override { return cfg_.N; }

    /**
     * Named outputs: {name -> log_phi, u_keys[j-1] -> u_j}. The
     * composite writes all of these into shared_data after every
     * step(). This is how downstream genbart_blocks see u_j (via
     * their y_key) and log_phi (via offset_key = this block's name).
     */
    std::unordered_map<std::string, arma::vec>
    current_named_outputs() const override {
        std::unordered_map<std::string, arma::vec> out;
        out[cfg_.name] = log_phi_;
        for (std::size_t j = 0; j < cfg_.C - 1; ++j) {
            out[cfg_.u_keys[j]] = u_cache_[j];
        }
        return out;
    }

    // ---- Tier-B setters for outer-MCMC integration ------------------

    /**
     * Push a new y vector (integer labels). Refreshes the u_j cache.
     * Useful when the outer sampler imputes y between sweeps and wants
     * to reflect the update eagerly (before the next set_context
     * call).
     */
    void set_y(const arma::vec& new_y) {
        if (new_y.n_elem != cfg_.N) {
            throw std::invalid_argument(
                "poisson_multinomial_aug_block::set_y: length mismatch "
                "(expected N = " + std::to_string(cfg_.N) + ", got "
                + std::to_string(new_y.n_elem) + ")");
        }
        for (std::size_t i = 0; i < cfg_.N; ++i) {
            const double yi = new_y[i];
            if (!(yi >= 0.0) || !(yi <= static_cast<double>(cfg_.C - 1))
                || yi != std::floor(yi))
            {
                throw std::invalid_argument(
                    "poisson_multinomial_aug_block::set_y: entries "
                    "must be integer-valued in {0, 1, ..., C-1}");
            }
        }
        y_cache_ = new_y;
        recompute_u_indicators_();
    }

    /// Read-only access to the internal y cache (post any imputation).
    const arma::vec& current_y() const noexcept { return y_cache_; }

    /// Number of non-reference categories (C-1).
    std::size_t n_nonref() const noexcept { return cfg_.C - 1; }

    /// Number of observations (N).
    std::size_t n_obs() const noexcept { return cfg_.N; }

    // ---- History --------------------------------------------------------

    history_map get_history() const override {
        return detail::make_history_map(cfg_.name, history_log_phi_,
                                        log_phi_);
    }

    std::size_t history_size() const noexcept override {
        return history_log_phi_.empty() ? 1 : history_log_phi_.size();
    }

    void clear_history() override { history_log_phi_.clear(); }

private:
    void recompute_u_indicators_() {
        for (std::size_t j = 0; j < cfg_.C - 1; ++j) {
            const double jval = static_cast<double>(j + 1);
            for (std::size_t i = 0; i < cfg_.N; ++i) {
                u_cache_[j][i] =
                    (y_cache_[i] == jval) ? 1.0 : 0.0;
            }
        }
    }

    poisson_multinomial_aug_block_config cfg_;
    block_context                         context_;
    arma::vec                             y_cache_;
    std::vector<arma::vec>                u_cache_;  // length C-1
    arma::vec                             log_phi_;
    std::vector<arma::vec>                history_log_phi_;
};

} // namespace AI4BayesCode

#endif // AI4BAYESCODE_POISSON_MULTINOMIAL_AUG_BLOCK_HPP
