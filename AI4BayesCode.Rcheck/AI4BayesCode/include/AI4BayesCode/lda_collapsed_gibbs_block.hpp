/*================================================================================
 *  AI4BayesCode: stateful modular MCMC for composable Gibbs samplers
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under the GNU General Public License v2.0 or later
 *  (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
 *================================================================================
 *
 *  lda_collapsed_gibbs_block.hpp -- Griffiths & Steyvers (2004) collapsed
 *                                   Gibbs sampler for Latent Dirichlet
 *                                   Allocation, packaged as one block.
 *
 *  WHY A SEPARATE BLOCK
 *  ====================
 *  LDA's joint state (z, theta, phi) has the property that, for every
 *  token n, theta_{doc_n} and phi_{z_n,*} are coupled to z through
 *  Dirichlet-multinomial conjugacy. A naive composition
 *
 *      categorical_gibbs_block(z) + dirichlet_gibbs_block(theta_d) (M of)
 *                                 + dirichlet_gibbs_block(phi_k)   (K of)
 *
 *  is correct in the limit but mixes catastrophically because (i) z's
 *  full conditional reads explicit theta and phi from shared_data, so
 *  the chain alternates between "sample z given (theta, phi) snapshot"
 *  and "draw fresh (theta, phi) given z" with strong cross-block
 *  correlation; (ii) the simulation builds many small Dirichlet draws
 *  where most posterior mass is in mid-K-vector entries, putting the
 *  chain in a high-dimensional flat region. system_design.md §11.2(b)
 *  specifically classes this as a "discrete target with strong local
 *  dependence" requiring a specialized sampler.
 *
 *  Griffiths & Steyvers 2004 PNAS 101(suppl 1):5228-5235 collapse out
 *  theta and phi analytically and sample only z, with
 *
 *      P(z_n = k | z_{-n}, w, alpha, beta)
 *           = (n_{d,k}^{-n} + alpha_k)
 *             * (n_{k,w_n}^{-n} + beta_{w_n})
 *             / (n_{k,*}^{-n} + sum(beta)).
 *
 *  Each token update is O(K) and uses three running count tables
 *  n_{d,k}, n_{k,v}, n_{k,*} which we maintain incrementally
 *  (O(1) per token: decrement old z_n, sample new, increment).
 *
 *  After the z sweep, theta and phi are sampled once each from their
 *  Dirichlet conjugate posteriors via gamma-normalization (same
 *  mechanism as dirichlet_gibbs_block). This makes the block produce
 *  three named outputs per step (z, theta, phi) so downstream blocks
 *  / generated quantities / posterior-predictive refreshers can read
 *  any of them via shared_data.
 *
 *  TARGET DISTRIBUTION (system_design.md §11.4 #4)
 *  ===============================================
 *  This block samples a §11.1 #2 fixed-dim discrete target (z in
 *  {1..K}^N with α, β fixed in advance), augmented with conjugate
 *  Dirichlet draws for the §11.1 #1 simplex parameters theta and phi.
 *  K is fixed at construction. NOT a §11.2(a) trans-dimensional
 *  target. The block is a specialized sampler for a §11.2(b)-class
 *  discrete-with-strong-local-dependence target; the collapsed-Gibbs
 *  marginalization of (theta, phi) is what produces fast mixing
 *  despite the local dependence.
 *
 *  CHECK #16 JUSTIFICATION
 *  =======================
 *  Exception 1 from codegen_priors.md §2b: the latent z is discrete,
 *  so NUTS cannot target it. Specialized Gibbs sampler is the only
 *  correct option. Library-level parity test is at
 *      tests_autodiff/block_tests/test_lda_collapsed_gibbs_block.cpp
 *  (Check #15) -- compares against an independent reference
 *  implementation of the same algorithm on a fixed seed.
 *
 *  CONSTRAINTS / JACOBIANS
 *  =======================
 *  None. z is integer-valued, theta and phi are sampled directly on
 *  the simplex via gamma-normalization; no unconstrained transform,
 *  no Jacobian. system_design.md §10's universal rule "users never
 *  write a Jacobian formula" holds vacuously.
 *
 *  RNG DISCIPLINE
 *  ==============
 *  step(rng) advances the supplied mt19937. The block does NOT own a
 *  predict_rng; if the wrapper class wants to expose predict_at /
 *  posterior-predictive, it owns predict_rng_ separately
 *  (system_design.md §8). Token order randomization within a sweep
 *  also draws from the supplied rng.
 *
 *  HISTORY STORAGE
 *  ===============
 *  Per get_history() override: returns three named entries
 *      "<z_out_key>"     : (n_draws x N)
 *      "<theta_out_key>" : (n_draws x M*K) col-major flat per draw
 *      "<phi_out_key>"   : (n_draws x K*V) col-major flat per draw
 *  When keep_history is OFF, returns 1-row matrices of the current
 *  state.
 *
 *  CANONICALIZATION
 *  ================
 *  The block does NOT permute topics internally. Topic labels at
 *  draw boundary are exchangeable in the posterior (label switching).
 *  Cross-implementation alignment is the caller's responsibility:
 *  apply Stephens 2000 in R post-hoc on the n_draws x M x K theta
 *  array (see skills/label_switching.md §4 LDA recipe).
 *================================================================================*/

#ifndef AI4BAYESCODE_LDA_COLLAPSED_GIBBS_BLOCK_HPP
#define AI4BAYESCODE_LDA_COLLAPSED_GIBBS_BLOCK_HPP

#include "block_sampler.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace AI4BayesCode {

/// Configuration bundle for lda_collapsed_gibbs_block construction.
///
/// All inputs that vary per token (w, doc) are read from the block
/// context via the keys w_key / doc_key on the FIRST set_context call.
/// This block expects them to be fixed for the lifetime of the chain
/// (they are observed data); subsequent set_context calls are
/// ignored from the data side.
struct lda_collapsed_gibbs_block_config {
    /// Unique label for this block in its composite. NOT used as a
    /// shared_data key directly -- z, theta, phi go to z_out_key /
    /// theta_out_key / phi_out_key respectively (see joint-block
    /// pattern in system_design.md §3 / §13).
    std::string name = "lda";

    /// Number of documents.
    std::size_t M = 0;

    /// Vocabulary size.
    std::size_t V = 0;

    /// Number of topics.
    std::size_t K = 0;

    /// Length-K Dirichlet hyperparameter on theta_d.
    /// Default (1, 1, ..., 1) corresponds to a uniform Dirichlet.
    arma::vec alpha;

    /// Length-V Dirichlet hyperparameter on phi_k.
    /// Default (1, 1, ..., 1) corresponds to a uniform Dirichlet.
    arma::vec beta;

    /// Block-context keys for the data inputs.
    /// w_key : vector of length N, entries in {1, ..., V}
    /// doc_key : vector of length N, entries in {1, ..., M}
    /// Stored as arma::vec of doubles (existing convention shared
    /// with categorical_gibbs_block).
    std::string w_key   = "w";
    std::string doc_key = "doc";

    /// Output shared_data keys exposed via current_named_outputs().
    /// z is the length-N integer-valued vector of topic assignments.
    /// theta is the M x K simplex matrix, flattened COLUMN-MAJOR
    ///   (i.e., theta_flat[d + m*M] is theta_{d, m+1} ... CAREFUL:
    ///   actually since arma is column-major and we lay out the
    ///   matrix as M rows by K columns, index is [m + k*M]).
    ///   See current_named_outputs() comment.
    /// phi   is the K x V simplex matrix, flattened column-major
    ///   (entry [k + v*K] is phi_{k+1, v+1}).
    std::string z_out_key     = "z";
    std::string theta_out_key = "theta";
    std::string phi_out_key   = "phi";

    /// Optional initial topic assignments. Must be empty (random
    /// uniform init) or length-N with entries in {1..K}.
    arma::vec z_init;
};

/**
 * @brief Griffiths & Steyvers 2004 collapsed Gibbs LDA, packaged as
 *        one self-contained block. Samples z (token-level topic
 *        assignment) by collapsed Gibbs, then theta and phi from
 *        their Dirichlet conjugate posteriors.
 *
 * State carried between steps:
 *   z_       : length-N (current topic assignments, 1-indexed)
 *   theta_   : length M*K, col-major; current Dirichlet draw per doc
 *   phi_     : length K*V, col-major; current Dirichlet draw per topic
 *   n_dk_    : M x K count matrix (z-induced)
 *   n_kv_    : K x V count matrix
 *   n_k_    : length K, n_kv_ row sums
 *
 * Counts are derived from z_ on every set_current and on the first
 * set_context, so the block is self-consistent regardless of init.
 */
class lda_collapsed_gibbs_block : public block_sampler {
public:
    explicit lda_collapsed_gibbs_block(lda_collapsed_gibbs_block_config cfg)
        : cfg_(std::move(cfg))
    {
        if (cfg_.M < 1) {
            throw std::invalid_argument(
                "lda_collapsed_gibbs_block: M (number of documents) "
                "must be >= 1");
        }
        if (cfg_.V < 2) {
            throw std::invalid_argument(
                "lda_collapsed_gibbs_block: V (vocabulary size) "
                "must be >= 2");
        }
        if (cfg_.K < 2) {
            throw std::invalid_argument(
                "lda_collapsed_gibbs_block: K (number of topics) "
                "must be >= 2");
        }
        if (cfg_.alpha.n_elem == 0) {
            cfg_.alpha = arma::vec(cfg_.K, arma::fill::ones);
        } else if (cfg_.alpha.n_elem != cfg_.K) {
            throw std::invalid_argument(
                "lda_collapsed_gibbs_block: alpha length must equal K");
        }
        if (cfg_.beta.n_elem == 0) {
            cfg_.beta = arma::vec(cfg_.V, arma::fill::ones);
        } else if (cfg_.beta.n_elem != cfg_.V) {
            throw std::invalid_argument(
                "lda_collapsed_gibbs_block: beta length must equal V");
        }
        for (std::size_t k = 0; k < cfg_.K; ++k) {
            if (!(cfg_.alpha[k] > 0.0)) {
                throw std::invalid_argument(
                    "lda_collapsed_gibbs_block: alpha entries must be "
                    "strictly positive");
            }
        }
        for (std::size_t v = 0; v < cfg_.V; ++v) {
            if (!(cfg_.beta[v] > 0.0)) {
                throw std::invalid_argument(
                    "lda_collapsed_gibbs_block: beta entries must be "
                    "strictly positive");
            }
        }

        alpha_sum_ = arma::sum(cfg_.alpha);
        beta_sum_  = arma::sum(cfg_.beta);

        // Empty until set_context plants w / doc and we know N.
        N_ = 0;
        z_.set_size(0);
        theta_.set_size(0);
        phi_.set_size(0);
        named_outputs_cache_.clear();
        history_z_.clear();
        history_theta_.clear();
        history_phi_.clear();
        data_initialized_ = false;
    }

    // ---- block_sampler interface ---------------------------------------

    /// On the first call, copy w, doc, build counts, init z if needed.
    /// Subsequent calls are no-ops on the data side (w, doc are
    /// observed and fixed). The context can still be (and is)
    /// re-installed without harm.
    void set_context(const block_context& ctx) override {
        context_ = ctx;
        if (!data_initialized_) {
            initialize_from_context_();
            data_initialized_ = true;
        }
    }

    /// One MCMC sweep: collapsed Gibbs over all N tokens (in random
    /// order), followed by a fresh Dirichlet conjugate draw of theta
    /// and phi from the updated counts.
    void step(std::mt19937_64& rng) override {
        if (!data_initialized_) {
            throw std::runtime_error(
                "lda_collapsed_gibbs_block::step: set_context must be "
                "called before step (need w, doc to initialize)");
        }

        // 1. Random token order.
        std::vector<std::size_t> order(N_);
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), rng);

        // 2. Collapsed Gibbs sweep.
        std::vector<double> probs(cfg_.K);
        for (std::size_t t : order) {
            const std::size_t d = doc_int_[t];     // 0-indexed
            const std::size_t v = w_int_[t];       // 0-indexed
            const std::size_t k_old = static_cast<std::size_t>(
                std::lround(z_[t])) - 1;            // 0-indexed

            // Decrement
            n_dk_(d, k_old) -= 1.0;
            n_kv_(k_old, v) -= 1.0;
            n_k_[k_old]     -= 1.0;

            // Build conditional p(z_t = k+1 | ...) for k = 0..K-1.
            // Unnormalized:
            //   (n_dk[d,k] + alpha_k) * (n_kv[k,v] + beta_v)
            //                          / (n_k[k]   + beta_sum)
            double total = 0.0;
            for (std::size_t k = 0; k < cfg_.K; ++k) {
                const double doc_term =
                    n_dk_(d, k) + cfg_.alpha[k];
                const double topic_term =
                    (n_kv_(k, v) + cfg_.beta[v])
                    / (n_k_[k]   + beta_sum_);
                const double w = doc_term * topic_term;
                probs[k] = w;
                total   += w;
            }
            if (!(total > 0.0) || !std::isfinite(total)) {
                // Should be impossible given alpha, beta > 0 and counts
                // are non-negative integers; defensive only.
                throw std::runtime_error(
                    "lda_collapsed_gibbs_block::step: invalid total "
                    "probability mass during token update");
            }

            // 3. Sample new k.
            std::uniform_real_distribution<double> u(0.0, total);
            const double draw = u(rng);
            double cum = 0.0;
            std::size_t k_new = cfg_.K - 1;  // safe fallback
            for (std::size_t k = 0; k < cfg_.K; ++k) {
                cum += probs[k];
                if (draw <= cum) { k_new = k; break; }
            }

            // Increment with new label, write z.
            n_dk_(d, k_new) += 1.0;
            n_kv_(k_new, v) += 1.0;
            n_k_[k_new]     += 1.0;
            z_[t] = static_cast<double>(k_new + 1);
        }

        // 4. Conjugate Dirichlet draws for theta and phi.
        sample_theta_(rng);
        sample_phi_(rng);

        // 5. Refresh named outputs cache.
        rebuild_named_outputs_();

        // 6. History.
        if (keep_history_) {
            history_z_.push_back(z_);
            history_theta_.push_back(theta_);
            history_phi_.push_back(phi_);
        }
    }

    /// Returns the current z vector (length N). For multi-output
    /// access, prefer current_named_outputs() which also exposes
    /// theta and phi.
    const arma::vec& current() const override { return z_; }

    /// Force-set the topic assignments. Counts are rebuilt from z.
    /// theta and phi are NOT touched here (they are sampled in
    /// step()); call step() once after set_current to get a
    /// consistent (z, theta, phi) triple.
    void set_current(const arma::vec& z_new) override {
        if (!data_initialized_) {
            throw std::runtime_error(
                "lda_collapsed_gibbs_block::set_current: data not yet "
                "initialized; call set_context first");
        }
        if (z_new.n_elem != N_) {
            throw std::invalid_argument(
                "lda_collapsed_gibbs_block::set_current: length must "
                "equal N (number of tokens)");
        }
        for (std::size_t t = 0; t < N_; ++t) {
            const long zk = std::lround(z_new[t]);
            if (zk < 1 || zk > static_cast<long>(cfg_.K)) {
                throw std::invalid_argument(
                    "lda_collapsed_gibbs_block::set_current: z entries "
                    "must be integers in {1, ..., K}");
            }
        }
        z_ = z_new;
        rebuild_counts_();
        rebuild_named_outputs_();
    }

    const std::string& name() const noexcept override { return cfg_.name; }

    /// Block's "primary" dimension is the length of z (length-N).
    /// theta and phi are auxiliary outputs accessed via
    /// current_named_outputs() and get_history().
    std::size_t dim() const noexcept override { return N_; }

    /// Three named outputs: z (length-N), theta (length M*K), phi
    /// (length K*V). Layouts:
    ///   theta : column-major flatten of an M x K matrix; entry
    ///           theta[m + k*M] = theta_{d=m+1, k=k+1}.
    ///   phi   : column-major flatten of a K x V matrix; entry
    ///           phi[k + v*K] = phi_{k+1, v=v+1}.
    /// (Both follow Armadillo's column-major default; downstream R
    ///  code can `matrix(theta, M, K)` and `matrix(phi, K, V)` to
    ///  recover.)
    state_map current_named_outputs() const override {
        return named_outputs_cache_;
    }

    // ---- History overrides -----------------------------------------------

    history_map get_history() const override {
        history_map out;
        out.emplace(cfg_.z_out_key,
                    detail::make_history_map(cfg_.z_out_key,
                                             history_z_, z_)
                        .at(cfg_.z_out_key));
        out.emplace(cfg_.theta_out_key,
                    detail::make_history_map(cfg_.theta_out_key,
                                             history_theta_, theta_)
                        .at(cfg_.theta_out_key));
        out.emplace(cfg_.phi_out_key,
                    detail::make_history_map(cfg_.phi_out_key,
                                             history_phi_, phi_)
                        .at(cfg_.phi_out_key));
        return out;
    }

    std::size_t history_size() const noexcept override {
        return history_z_.empty() ? 1 : history_z_.size();
    }

    void clear_history() override {
        history_z_.clear();
        history_theta_.clear();
        history_phi_.clear();
    }

    // ---- Introspection helpers (test use) --------------------------------

    /// Read-only access to the current (n_dk, n_kv, n_k) counts.
    /// Used by Check #15 parity test.
    const arma::mat& counts_dk() const { return n_dk_; }
    const arma::mat& counts_kv() const { return n_kv_; }
    const arma::vec& counts_k()  const { return n_k_;  }
    const arma::vec& current_theta() const { return theta_; }
    const arma::vec& current_phi()   const { return phi_;   }
    std::size_t      n_tokens() const { return N_; }

private:
    // ----- Initialization helpers --------------------------------------

    void initialize_from_context_() {
        // Pull w and doc from the context. They are observed and fixed.
        auto it_w   = context_.find(cfg_.w_key);
        auto it_doc = context_.find(cfg_.doc_key);
        if (it_w == context_.end()) {
            throw std::runtime_error(
                "lda_collapsed_gibbs_block: context missing required "
                "key '" + cfg_.w_key + "'");
        }
        if (it_doc == context_.end()) {
            throw std::runtime_error(
                "lda_collapsed_gibbs_block: context missing required "
                "key '" + cfg_.doc_key + "'");
        }
        const arma::vec& w_vec   = it_w->second;
        const arma::vec& doc_vec = it_doc->second;
        if (w_vec.n_elem != doc_vec.n_elem) {
            throw std::runtime_error(
                "lda_collapsed_gibbs_block: w and doc must have the "
                "same length");
        }
        if (w_vec.n_elem == 0) {
            throw std::runtime_error(
                "lda_collapsed_gibbs_block: at least one token "
                "required (w / doc empty)");
        }

        N_ = w_vec.n_elem;
        w_int_.resize(N_);
        doc_int_.resize(N_);
        for (std::size_t t = 0; t < N_; ++t) {
            const long wt = std::lround(w_vec[t]);
            const long dt = std::lround(doc_vec[t]);
            if (wt < 1 || wt > static_cast<long>(cfg_.V)) {
                throw std::runtime_error(
                    "lda_collapsed_gibbs_block: w entry out of range "
                    "{1, ..., V}");
            }
            if (dt < 1 || dt > static_cast<long>(cfg_.M)) {
                throw std::runtime_error(
                    "lda_collapsed_gibbs_block: doc entry out of range "
                    "{1, ..., M}");
            }
            w_int_[t]   = static_cast<std::size_t>(wt - 1);
            doc_int_[t] = static_cast<std::size_t>(dt - 1);
        }

        // Initialize z. Random uniform if user did not supply.
        z_.set_size(N_);
        if (cfg_.z_init.n_elem == 0) {
            // Deterministic init pattern z[t] = (t mod K) + 1.
            // Random init would draw from the rng, but rng is a
            // step-time argument, not a constructor argument; the
            // first step() will randomize via its own sweep. The
            // deterministic init keeps construction reproducible
            // regardless of rng plumbing.
            for (std::size_t t = 0; t < N_; ++t) {
                z_[t] = static_cast<double>((t % cfg_.K) + 1);
            }
        } else if (cfg_.z_init.n_elem == N_) {
            for (std::size_t t = 0; t < N_; ++t) {
                const long zk = std::lround(cfg_.z_init[t]);
                if (zk < 1 || zk > static_cast<long>(cfg_.K)) {
                    throw std::runtime_error(
                        "lda_collapsed_gibbs_block: z_init entry out of "
                        "range {1, ..., K}");
                }
                z_[t] = static_cast<double>(zk);
            }
        } else {
            throw std::runtime_error(
                "lda_collapsed_gibbs_block: z_init must be empty or "
                "length-N");
        }

        // Build counts and produce initial (theta, phi) from priors
        // (alpha, beta) so named_outputs is non-empty before step.
        rebuild_counts_();

        // Initial theta, phi: prior means (won't be used until step()
        // overwrites them). Use Dirichlet posterior means with
        // current counts so they are at least likelihood-aware.
        theta_.set_size(cfg_.M * cfg_.K);
        for (std::size_t k = 0; k < cfg_.K; ++k) {
            for (std::size_t d = 0; d < cfg_.M; ++d) {
                const double a = cfg_.alpha[k] + n_dk_(d, k);
                const double s = alpha_sum_
                    + arma::accu(n_dk_.row(d));
                theta_[d + k * cfg_.M] = (s > 0.0) ? (a / s) : (1.0 / cfg_.K);
            }
        }
        phi_.set_size(cfg_.K * cfg_.V);
        for (std::size_t v = 0; v < cfg_.V; ++v) {
            for (std::size_t k = 0; k < cfg_.K; ++k) {
                const double b = cfg_.beta[v] + n_kv_(k, v);
                const double s = beta_sum_ + n_k_[k];
                phi_[k + v * cfg_.K] = (s > 0.0) ? (b / s) : (1.0 / cfg_.V);
            }
        }
        rebuild_named_outputs_();
    }

    void rebuild_counts_() {
        n_dk_.set_size(cfg_.M, cfg_.K);
        n_dk_.zeros();
        n_kv_.set_size(cfg_.K, cfg_.V);
        n_kv_.zeros();
        n_k_.set_size(cfg_.K);
        n_k_.zeros();
        for (std::size_t t = 0; t < N_; ++t) {
            const std::size_t d = doc_int_[t];
            const std::size_t v = w_int_[t];
            const std::size_t k =
                static_cast<std::size_t>(std::lround(z_[t])) - 1;
            n_dk_(d, k) += 1.0;
            n_kv_(k, v) += 1.0;
            n_k_[k]     += 1.0;
        }
    }

    /// Sample theta_d ~ Dir(alpha + n_{d,*}) for each doc d via
    /// gamma-normalization. Stores the M x K result column-major in
    /// theta_ (entry [d + k*M] = theta_{d, k}).
    void sample_theta_(std::mt19937_64& rng) {
        theta_.set_size(cfg_.M * cfg_.K);
        std::vector<double> g(cfg_.K);
        for (std::size_t d = 0; d < cfg_.M; ++d) {
            double total = 0.0;
            for (std::size_t k = 0; k < cfg_.K; ++k) {
                const double shape = cfg_.alpha[k] + n_dk_(d, k);
                std::gamma_distribution<double> gam(shape, 1.0);
                g[k] = gam(rng);
                total += g[k];
            }
            if (!(total > 0.0)) {
                // Underflow defensive: fall back to prior mean.
                for (std::size_t k = 0; k < cfg_.K; ++k) {
                    theta_[d + k * cfg_.M] =
                        cfg_.alpha[k] / alpha_sum_;
                }
                continue;
            }
            for (std::size_t k = 0; k < cfg_.K; ++k) {
                theta_[d + k * cfg_.M] = g[k] / total;
            }
        }
    }

    /// Sample phi_k ~ Dir(beta + n_{k,*}) for each topic k. Stores
    /// K x V column-major in phi_ (entry [k + v*K] = phi_{k, v}).
    void sample_phi_(std::mt19937_64& rng) {
        phi_.set_size(cfg_.K * cfg_.V);
        std::vector<double> g(cfg_.V);
        for (std::size_t k = 0; k < cfg_.K; ++k) {
            double total = 0.0;
            for (std::size_t v = 0; v < cfg_.V; ++v) {
                const double shape = cfg_.beta[v] + n_kv_(k, v);
                std::gamma_distribution<double> gam(shape, 1.0);
                g[v] = gam(rng);
                total += g[v];
            }
            if (!(total > 0.0)) {
                for (std::size_t v = 0; v < cfg_.V; ++v) {
                    phi_[k + v * cfg_.K] =
                        cfg_.beta[v] / beta_sum_;
                }
                continue;
            }
            for (std::size_t v = 0; v < cfg_.V; ++v) {
                phi_[k + v * cfg_.K] = g[v] / total;
            }
        }
    }

    void rebuild_named_outputs_() {
        named_outputs_cache_.clear();
        named_outputs_cache_.emplace(cfg_.z_out_key,     z_);
        named_outputs_cache_.emplace(cfg_.theta_out_key, theta_);
        named_outputs_cache_.emplace(cfg_.phi_out_key,   phi_);
    }

    // ----- Member data -------------------------------------------------

    lda_collapsed_gibbs_block_config cfg_;
    double alpha_sum_ = 0.0;
    double beta_sum_  = 0.0;

    block_context context_;
    bool data_initialized_ = false;

    // Token data.
    std::size_t N_ = 0;
    std::vector<std::size_t> w_int_;     // 0-indexed word ids
    std::vector<std::size_t> doc_int_;   // 0-indexed doc ids

    // Sampled state.
    arma::vec z_;       // length N, 1-indexed topic labels
    arma::vec theta_;   // M*K col-major
    arma::vec phi_;     // K*V col-major

    // Counts (z-derived).
    arma::mat n_dk_;
    arma::mat n_kv_;
    arma::vec n_k_;

    // Output cache.
    state_map named_outputs_cache_;

    // History buffers (one per output).
    std::vector<arma::vec> history_z_;
    std::vector<arma::vec> history_theta_;
    std::vector<arma::vec> history_phi_;
};

} // namespace AI4BayesCode

#endif // AI4BAYESCODE_LDA_COLLAPSED_GIBBS_BLOCK_HPP
