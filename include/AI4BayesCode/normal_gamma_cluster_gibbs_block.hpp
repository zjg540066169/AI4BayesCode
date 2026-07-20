/*================================================================================
 *  AI4BayesCode: stateful modular MCMC for composable Gibbs samplers
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under the GNU General Public License v2.0 or later
 *  (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
 *================================================================================
 *
 *  normal_gamma_cluster_gibbs_block.hpp -- a closed-form Gibbs leaf that
 *      samples per-cluster diagonal-Gaussian parameters (mu_k, lambda_k)
 *      jointly across K_trunc clusters under conjugate Normal-Gamma
 *      priors. Used as the cluster-emission block in truncated stick-
 *      breaking BNP mixture models (DPGaussianMixture, PYGaussian-
 *      Mixture, ...).
 *
 *  WHY THIS BLOCK EXISTS
 *  =====================
 *  In a truncated SBP Gaussian mixture model
 *
 *      y_i  ~  N(mu_{z_i}, diag(1 / lambda_{z_i}))     i = 1, ..., N
 *      z_i  ~  Categorical(pi)
 *      pi   ~  truncated stick-breaking
 *      (mu_k, lambda_k) ~ NormalGamma(mu_0, kappa_0, a_lambda_0, b_lambda_0)
 *
 *  the conditional distribution of (mu_k, lambda_k) given z, y is
 *  exactly NormalGamma per dimension. This block performs the standard
 *  conjugate update for ALL K_trunc clusters in one O(K_trunc * d + N)
 *  sweep, drawing populated clusters from the data-driven posterior and
 *  EMPTY clusters from the prior (matches Ishwaran & James 2001 §3.2).
 *
 *  WHY NOT joint_nuts_block_mixed
 *  ------------------------------
 *  Two reasons:
 *    1. Empty clusters have a flat-prior posterior; NUTS gets stuck on
 *       flat regions while exact Gibbs samples directly from the prior.
 *    2. NUTS warmup over a 2 * K_trunc * d-dimensional joint state with
 *       block-diagonal Hessian is wasteful when K_trunc * d * 2 conjugate
 *       draws are O(K_trunc * d) each and exact.
 *  The 5-10x speedup matters for the 4-chain audit budget.
 *
 *  PRIOR (per dimension d, identical across clusters)
 *  --------------------------------------------------
 *      lambda_d  ~  Gamma(shape = a_lambda_0, rate = b_lambda_0)
 *      mu_d | lambda_d  ~  N(mu_0_d, 1 / (kappa_0 * lambda_d))
 *
 *  This is the standard NormalGamma conjugate to a univariate Normal
 *  observation with unknown mean and precision. Diagonal across d
 *  (Bishop PRML §2.3.6); for full-covariance NIW use a future
 *  niw_cluster_gibbs_block.
 *
 *  POSTERIOR (per cluster k, dimension d)
 *  --------------------------------------
 *  Let n_k = #{i : z_i = k+1} (1-indexed labels), and y^k_d the values
 *  of dimension d among observations assigned to cluster k.
 *
 *    bar_y_d   = mean(y^k_d)              (only if n_k > 0)
 *    s2_d      = sum((y^k_d - bar_y_d)^2) (only if n_k > 0)
 *    kappa_n   = kappa_0 + n_k
 *    mu_n_d    = (kappa_0 * mu_0_d + n_k * bar_y_d) / kappa_n
 *    a_n       = a_lambda_0 + n_k / 2
 *    b_n_d     = b_lambda_0
 *                + 0.5 * s2_d
 *                + 0.5 * kappa_0 * n_k / kappa_n * (bar_y_d - mu_0_d)^2
 *
 *    lambda_kd ~ Gamma(shape = a_n, rate = b_n_d)
 *    mu_kd     ~ N(mu_n_d, 1 / (kappa_n * lambda_kd))
 *
 *  When n_k == 0 the block draws from the prior:
 *    lambda_kd ~ Gamma(a_lambda_0, rate = b_lambda_0)
 *    mu_kd     ~ N(mu_0_d, 1 / (kappa_0 * lambda_kd))
 *
 *  Reference: Murphy 2007, "Conjugate Bayesian analysis of the Gaussian
 *  distribution", §4 (equations 4-9 in the technical report).
 *
 *  STORAGE / OUTPUT
 *  ----------------
 *  Two named outputs (current_named_outputs):
 *    - cfg.mu_name: length K_trunc * d, cluster-major row order
 *                   (mu_1[0], ..., mu_1[d-1], mu_2[0], ..., mu_K[d-1]).
 *    - cfg.lambda_name: length K_trunc * d, same order, holds PRECISIONS
 *                       (lambda_kd, NOT variance 1/lambda_kd).
 *
 *  current() returns the concatenated [mu; lambda] of length 2 * K * d.
 *  set_current(theta) accepts the same concatenated vector.
 *
 *  COMMENT-PARAMETERIZATION DISCIPLINE (rcpp_api.md §11)
 *  -----------------------------------------------------
 *  - std::gamma_distribution<double>(shape, scale) is shape-SCALE,
 *    E[X] = shape * scale.
 *  - To draw Gamma(shape=a, rate=b), use scale = 1 / b:
 *      std::gamma_distribution<double> gam(a, 1.0 / b);
 *      // E[X] = a / b
 *  - std::normal_distribution<double>(mean, stddev) — stddev, NOT
 *    variance.
 *
 *  Both transformations are explicitly commented at every usage site.
 *
 *  DEPENDENCIES (declare these in the composite)
 *  ---------------------------------------------
 *      declare_dependencies(cfg.name, {z_key, y_key, ...prior_keys...})
 *      declare_invalidates(cfg.name, {derived_keys_using_mu_or_lambda})
 *
 *  cfg.name is used as the BLOCK label only (for declare_*); the
 *  shared_data WRITES happen under cfg.mu_name and cfg.lambda_name.
 *================================================================================*/

#ifndef AI4BAYESCODE_NORMAL_GAMMA_CLUSTER_GIBBS_BLOCK_HPP
#define AI4BAYESCODE_NORMAL_GAMMA_CLUSTER_GIBBS_BLOCK_HPP

#include "block_sampler.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace AI4BayesCode {

/// Configuration bundle for normal_gamma_cluster_gibbs_block.
struct normal_gamma_cluster_gibbs_block_config {
    /// Block label (used by composite for declare_dependencies /
    /// declare_invalidates). NOT used as a shared_data key — the block
    /// writes under cfg.mu_name and cfg.lambda_name.
    std::string name;

    /// Truncation level (number of clusters).
    std::size_t K_trunc = 0;

    /// Observation dimension.
    std::size_t d = 0;

    /// Number of observations.
    std::size_t N = 0;

    /// Shared_data key for cluster assignments z (length N, values in
    /// {1, ..., K_trunc}, stored as arma::vec).
    std::string z_key = "z";

    /// Shared_data key for the observed data y. Stored as a flat
    /// arma::vec of length N * d in row-major order:
    ///   y_flat[i * d + dim] == y_i_dim.
    std::string y_key = "y";

    /// Shared_data key under which the K_trunc * d flat mu vector is
    /// exposed.
    std::string mu_name = "mu";

    /// Shared_data key under which the K_trunc * d flat lambda vector
    /// (precisions, NOT variances) is exposed.
    std::string lambda_name = "lambda";

    /// Prior hyperparameters (per dimension; replicated across
    /// dimensions and across clusters).
    arma::vec mu_0;            // length d, prior mean of mu
    double    kappa_0      = 0.0;  // prior precision multiplier on mu
    double    a_lambda_0   = 0.0;  // Gamma shape parameter on lambda
    double    b_lambda_0   = 0.0;  // Gamma RATE parameter on lambda

    /// Initial values. Each must have length K_trunc * d (cluster-major
    /// row order). All entries of initial_lambda must be > 0.
    arma::vec initial_mu;
    arma::vec initial_lambda;
};

class normal_gamma_cluster_gibbs_block : public block_sampler {
public:
    explicit normal_gamma_cluster_gibbs_block(
            normal_gamma_cluster_gibbs_block_config cfg)
        : cfg_(std::move(cfg))
    {
        if (cfg_.name.empty()) {
            throw std::invalid_argument(
                "normal_gamma_cluster_gibbs_block: name must be non-empty");
        }
        if (cfg_.K_trunc < 1) {
            throw std::invalid_argument(
                "normal_gamma_cluster_gibbs_block: K_trunc must be >= 1");
        }
        if (cfg_.d == 0) {
            throw std::invalid_argument(
                "normal_gamma_cluster_gibbs_block: d must be > 0");
        }
        if (cfg_.N == 0) {
            throw std::invalid_argument(
                "normal_gamma_cluster_gibbs_block: N must be > 0");
        }
        if (cfg_.mu_0.n_elem != cfg_.d) {
            throw std::invalid_argument(
                "normal_gamma_cluster_gibbs_block: mu_0 length must equal d");
        }
        if (!(cfg_.kappa_0 > 0.0)) {
            throw std::invalid_argument(
                "normal_gamma_cluster_gibbs_block: kappa_0 must be > 0");
        }
        if (!(cfg_.a_lambda_0 > 0.0)) {
            throw std::invalid_argument(
                "normal_gamma_cluster_gibbs_block: a_lambda_0 must be > 0");
        }
        if (!(cfg_.b_lambda_0 > 0.0)) {
            throw std::invalid_argument(
                "normal_gamma_cluster_gibbs_block: b_lambda_0 must be > 0");
        }
        const std::size_t flat = cfg_.K_trunc * cfg_.d;
        if (cfg_.initial_mu.n_elem != flat) {
            throw std::invalid_argument(
                "normal_gamma_cluster_gibbs_block: initial_mu length must "
                "equal K_trunc * d");
        }
        if (cfg_.initial_lambda.n_elem != flat) {
            throw std::invalid_argument(
                "normal_gamma_cluster_gibbs_block: initial_lambda length "
                "must equal K_trunc * d");
        }
        for (std::size_t i = 0; i < flat; ++i) {
            if (!(cfg_.initial_lambda[i] > 0.0)) {
                throw std::invalid_argument(
                    "normal_gamma_cluster_gibbs_block: initial_lambda must "
                    "be strictly positive");
            }
        }
        if (cfg_.mu_name.empty() || cfg_.lambda_name.empty()) {
            throw std::invalid_argument(
                "normal_gamma_cluster_gibbs_block: mu_name and lambda_name "
                "must be non-empty");
        }
        if (cfg_.mu_name == cfg_.lambda_name) {
            throw std::invalid_argument(
                "normal_gamma_cluster_gibbs_block: mu_name and lambda_name "
                "must differ");
        }
        mu_     = cfg_.initial_mu;
        lambda_ = cfg_.initial_lambda;
        // current_ = concat(mu, lambda) of length 2 * flat.
        rebuild_current_();
    }

    // ---- block_sampler interface ---------------------------------------

    void set_context(const block_context& ctx) override {
        // COPY (per design contract — no pointer survival across step).
        context_ = ctx;
    }

    void step(std::mt19937_64& rng) override {
        const std::size_t K = cfg_.K_trunc;
        const std::size_t d = cfg_.d;
        const std::size_t N = cfg_.N;

        // Pull z, y from context.
        auto it_z = context_.find(cfg_.z_key);
        if (it_z == context_.end()) {
            throw std::runtime_error(
                "normal_gamma_cluster_gibbs_block '" + cfg_.name +
                "': z_key '" + cfg_.z_key + "' not in context");
        }
        const arma::vec& z = it_z->second;
        if (z.n_elem != N) {
            throw std::runtime_error(
                "normal_gamma_cluster_gibbs_block '" + cfg_.name +
                "': z length " + std::to_string(z.n_elem) +
                " != N " + std::to_string(N));
        }
        auto it_y = context_.find(cfg_.y_key);
        if (it_y == context_.end()) {
            throw std::runtime_error(
                "normal_gamma_cluster_gibbs_block '" + cfg_.name +
                "': y_key '" + cfg_.y_key + "' not in context");
        }
        const arma::vec& y_flat = it_y->second;
        if (y_flat.n_elem != N * d) {
            throw std::runtime_error(
                "normal_gamma_cluster_gibbs_block '" + cfg_.name +
                "': y length " + std::to_string(y_flat.n_elem) +
                " != N * d " + std::to_string(N * d));
        }

        // Per-cluster sufficient statistics.
        // sum_y_kd  = sum of y_i_d among i with z_i = k+1
        // sum_y2_kd = sum of (y_i_d)^2 among i with z_i = k+1
        std::vector<std::size_t> n_k(K, 0);
        std::vector<double> sum_y(K * d, 0.0);
        std::vector<double> sum_y2(K * d, 0.0);
        for (std::size_t i = 0; i < N; ++i) {
            const long lab = static_cast<long>(std::llround(z[i]));
            if (lab < 1 || static_cast<std::size_t>(lab) > K) {
                throw std::runtime_error(
                    "normal_gamma_cluster_gibbs_block '" + cfg_.name +
                    "': z[" + std::to_string(i) +
                    "] out of {1, ..., K_trunc}");
            }
            const std::size_t k = static_cast<std::size_t>(lab) - 1;
            n_k[k] += 1;
            for (std::size_t j = 0; j < d; ++j) {
                const double v = y_flat[i * d + j];
                sum_y[k * d + j]  += v;
                sum_y2[k * d + j] += v * v;
            }
        }

        // Per-cluster, per-dimension Normal-Gamma posterior draw.
        std::normal_distribution<double> stdnorm(0.0, 1.0);
        for (std::size_t k = 0; k < K; ++k) {
            const std::size_t nk = n_k[k];
            for (std::size_t j = 0; j < d; ++j) {
                const std::size_t idx = k * d + j;
                if (nk == 0) {
                    // Empty cluster: prior draw.
                    // lambda_kd ~ Gamma(shape = a_lambda_0, rate = b_lambda_0)
                    // E[lambda_kd] = a_lambda_0 / b_lambda_0
                    std::gamma_distribution<double> gam(
                        cfg_.a_lambda_0, 1.0 / cfg_.b_lambda_0);
                    double lam = gam(rng);
                    if (!(lam > 0.0)) lam = 1e-300;
                    // Kernel-control sub-key freeze (DESIGN_NOTES Sec.10.a):
                    // if lambda_frozen_, skip lambda update; use existing value.
                    if (!lambda_frozen_) lambda_[idx] = lam;
                    const double lam_use = lambda_frozen_ ? lambda_[idx] : lam;
                    // mu_kd | lambda_kd ~ N(mu_0_d, 1 / (kappa_0 * lambda_kd))
                    // sd = 1 / sqrt(kappa_0 * lambda_kd)
                    const double sd = 1.0 /
                        std::sqrt(cfg_.kappa_0 * lam_use);
                    if (!mu_frozen_)
                        mu_[idx] = cfg_.mu_0[j] + sd * stdnorm(rng);
                } else {
                    // Posterior draw with sufficient statistics.
                    const double mean_y =
                        sum_y[idx] / static_cast<double>(nk);
                    // s2 = sum (y - mean_y)^2 = sum y^2 - n * mean_y^2
                    double s2 = sum_y2[idx]
                              - static_cast<double>(nk) * mean_y * mean_y;
                    if (s2 < 0.0) s2 = 0.0;  // floating-point safety
                    const double dev    = mean_y - cfg_.mu_0[j];
                    const double kappa_n =
                        cfg_.kappa_0 + static_cast<double>(nk);
                    const double a_n     =
                        cfg_.a_lambda_0 + static_cast<double>(nk) / 2.0;
                    const double b_n     =
                        cfg_.b_lambda_0
                        + 0.5 * s2
                        + 0.5 * cfg_.kappa_0 *
                                static_cast<double>(nk) / kappa_n
                              * dev * dev;
                    const double mu_n    =
                        (cfg_.kappa_0 * cfg_.mu_0[j]
                         + static_cast<double>(nk) * mean_y) / kappa_n;

                    // lambda_kd ~ Gamma(shape = a_n, rate = b_n)
                    // std uses shape-SCALE; convert: scale = 1 / rate
                    // E[lambda_kd] = a_n / b_n
                    std::gamma_distribution<double> gam(a_n, 1.0 / b_n);
                    double lam = gam(rng);
                    if (!(lam > 0.0)) lam = 1e-300;
                    // Kernel-control sub-key freeze (DESIGN_NOTES Sec.10.a):
                    // if lambda_frozen_, skip lambda update; use existing value.
                    if (!lambda_frozen_) lambda_[idx] = lam;
                    const double lam_use = lambda_frozen_ ? lambda_[idx] : lam;
                    // mu_kd | lambda_kd ~ N(mu_n, 1 / (kappa_n * lambda_kd))
                    const double sd = 1.0 / std::sqrt(kappa_n * lam_use);
                    if (!mu_frozen_)
                        mu_[idx] = mu_n + sd * stdnorm(rng);
                }
            }
        }

        rebuild_current_();
        if (keep_history_) {
            mu_history_buf_.push_back(mu_);
            lambda_history_buf_.push_back(lambda_);
        }
    }

    const arma::vec& current() const override { return current_; }

    void set_current(const arma::vec& theta) override {
        const std::size_t flat = cfg_.K_trunc * cfg_.d;
        if (theta.n_elem != 2 * flat) {
            throw std::invalid_argument(
                "normal_gamma_cluster_gibbs_block::set_current: expected "
                "length " + std::to_string(2 * flat));
        }
        // First half is mu, second half is lambda.
        for (std::size_t i = 0; i < flat; ++i) {
            mu_[i]     = theta[i];
            lambda_[i] = theta[flat + i];
            if (!(lambda_[i] > 0.0)) {
                throw std::invalid_argument(
                    "normal_gamma_cluster_gibbs_block::set_current: lambda "
                    "entries must be strictly positive");
            }
        }
        rebuild_current_();
    }

    const std::string& name() const noexcept override { return cfg_.name; }

    // Kernel-control sub-key freeze API (DESIGN_NOTES Sec.10 + subagent-B
    // block-family audit). Two sub-names -- the block's cfg_.mu_name and
    // cfg_.lambda_name. Freezing "<mu_name>" holds cluster means fixed while
    // precisions sample from their (data-dependent) conditional; freezing
    // "<lambda_name>" holds cluster precisions fixed while means sample.
    // Composite dot-path form: "<block_name>.<mu_name>" also works.
    std::vector<std::string> subnames() const override {
        return {cfg_.mu_name, cfg_.lambda_name};
    }
    void freeze_sub(const std::string& sub) override {
        if (sub == cfg_.mu_name)          mu_frozen_ = true;
        else if (sub == cfg_.lambda_name) lambda_frozen_ = true;
        else throw std::runtime_error(
            "normal_gamma_cluster_gibbs_block '" + cfg_.name +
            "': freeze_sub unknown sub-name '" + sub +
            "'; valid: '" + cfg_.mu_name + "', '" + cfg_.lambda_name + "'");
    }
    void unfreeze_sub(const std::string& sub) override {
        if (sub == cfg_.mu_name)          mu_frozen_ = false;
        else if (sub == cfg_.lambda_name) lambda_frozen_ = false;
        // unknown = no-op (permissive)
    }
    std::vector<std::string> frozen_subnames() const override {
        std::vector<std::string> out;
        if (mu_frozen_)     out.push_back(cfg_.mu_name);
        if (lambda_frozen_) out.push_back(cfg_.lambda_name);
        return out;
    }
    void unfreeze() override {
        block_sampler::unfreeze();
        mu_frozen_ = false;
        lambda_frozen_ = false;
    }

    std::size_t dim() const noexcept override {
        return 2 * cfg_.K_trunc * cfg_.d;
    }

    /// Two named outputs: mu (length K*d) and lambda (length K*d).
    state_map current_named_outputs() const override {
        state_map out;
        out.emplace(cfg_.mu_name,     mu_);
        out.emplace(cfg_.lambda_name, lambda_);
        return out;
    }

    // ---- History overrides -----------------------------------------------

    history_map get_history() const override {
        history_map out =
            detail::make_history_map(cfg_.mu_name, mu_history_buf_, mu_);
        history_map lam =
            detail::make_history_map(cfg_.lambda_name,
                                     lambda_history_buf_, lambda_);
        for (auto& kv : lam) {
            out.emplace(kv.first, std::move(kv.second));
        }
        return out;
    }

    std::size_t history_size() const noexcept override {
        return mu_history_buf_.empty() ? 1 : mu_history_buf_.size();
    }

    void clear_history() override {
        mu_history_buf_.clear();
        lambda_history_buf_.clear();
    }

private:
    void rebuild_current_() {
        const std::size_t flat = cfg_.K_trunc * cfg_.d;
        current_.set_size(2 * flat);
        for (std::size_t i = 0; i < flat; ++i) {
            current_[i]        = mu_[i];
            current_[flat + i] = lambda_[i];
        }
    }

    // Kernel-control sub-key freeze flags (DESIGN_NOTES Sec.10 + subagent-B).
    // Whole-block freeze goes via is_frozen_ in base class; these additionally
    // gate the mu-update and lambda-update inside step().
    bool                                    mu_frozen_     = false;
    bool                                    lambda_frozen_ = false;

    normal_gamma_cluster_gibbs_block_config cfg_;
    arma::vec                               mu_;
    arma::vec                               lambda_;
    arma::vec                               current_;  // cached concat
    block_context                           context_;
    std::vector<arma::vec>                  mu_history_buf_;
    std::vector<arma::vec>                  lambda_history_buf_;
};

}  // namespace AI4BayesCode

#endif  // AI4BAYESCODE_NORMAL_GAMMA_CLUSTER_GIBBS_BLOCK_HPP
