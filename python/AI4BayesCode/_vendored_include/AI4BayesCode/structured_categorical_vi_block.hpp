/*================================================================================
 *  AI4BayesCode: stateful modular MCMC for composable Gibbs samplers
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under the GNU General Public License v2.0 or later
 *  (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
 *================================================================================
 *
 *  structured_categorical_vi_block.hpp  -- Saul-Jordan 1996 partial-
 *                                         factorisation mean-field VI
 *                                         for discrete categorical latents.
 *
 *  VARIATIONAL FAMILY (Saul-Jordan 1996 §2.2)
 *  ==========================================
 *      q(z_1, ..., z_n) = prod_C q_C(z_C),
 *      where {C_1, ..., C_M} is a USER-SUPPLIED PARTITION of {0,..,n-1}
 *      and each q_C is a joint Categorical over the clique's joint
 *      state space (size prod_{i in C} K_i).
 *
 *  This refines the fully-factorised mean field of Block 4:
 *    - Block 4: q(z) = prod_i Cat(z_i; phi_i)
 *               (independent per variable)
 *    - Block 5: q(z) = prod_C Cat(z_C; phi_C)
 *               (independent ACROSS cliques, joint WITHIN cliques)
 *
 *  Saul-Jordan's motivating example was coupled HMMs: chains as tractable
 *  cliques, inter-chain couplings handled by mean field. Here we
 *  generalise the SAME machinery to user-defined clique partitions on
 *  any discrete MRF.
 *
 *  Each q_C is internally an anchored softmax over the clique's joint
 *  state space:
 *      For clique C with S_C = prod_{i in C} K_i joint states:
 *        eta_C in R^{S_C - 1}, anchored eta_C[s=0] := 0
 *        phi_C(s) = exp(eta_C[s]) / Z_C
 *        Z_C    = 1 + sum_{l>=1} exp(eta_C[l])
 *  Total free parameters = sum_C (S_C - 1).
 *
 *  GRADIENT — analytical sum-over-clique-state + MC marginalisation
 *  ================================================================
 *  Following the SAME pattern as Block 4 but at the clique level:
 *      G_C(s) := E_{q\C}[ log p~(z, z_C = decode_C(s)) ]
 *  Then
 *      dELBO/dphi_C(s) = G_C(s) - log phi_C(s) - 1
 *  and the chain rule through the anchored softmax gives
 *      dELBO/deta_C(l) = phi_C(l) * [
 *           (G_C(l) - log phi_C(l))
 *         - sum_s phi_C(s) (G_C(s) - log phi_C(s))
 *      ]   for l = 1, ..., S_C - 1
 *
 *  Computing G_C(s) requires marginalising over z_{-C} (the other
 *  cliques) under the current q. Two modes:
 *    (a) EXACT: enumerate the full joint state prod_i K_i — capped
 *        at exact_state_cap (default 4096) — and compute G_C(s)
 *        by direct sum.
 *    (b) MC (default): sample z_s ~ q for s = 1..S, then for each
 *        clique C and each clique-joint-state q_idx, replace z[i ∈ C]
 *        with decode_C(q_idx), evaluate log p~ and average over S.
 *
 *  Sampling z ~ q: for each clique, draw one joint state from
 *  Categorical(phi_C) and set the per-node values via decode_C.
 *
 *  DEGENERATE CASES (correctness invariants)
 *  =========================================
 *    Singleton cliques (each clique = one node)
 *      => Block 5 reduces to Block 4 (fully factorised MF).
 *    Single grand-clique containing all nodes
 *      => phi_C is the full joint posterior;
 *         exact mode = exact inference.
 *  These are both unit-tested.
 *
 *  TARGET-SHAPE CATEGORY (system_design.md §11.2(b))
 *  =================================================
 *  Same as Block 4: discrete latents with strong local dependence.
 *  Structured MF captures within-clique correlation that Block 4
 *  factorises away, giving tighter marginal & joint approximation
 *  for the same n. The textbook benefit is when the true posterior
 *  has STRONG intra-clique coupling but WEAK inter-clique coupling —
 *  exactly the Saul-Jordan setting.
 *
 *  ENGINE FAMILY
 *  =============
 *  engine_kind() = VI. composite_block writes q-samples (NOT q-mean)
 *  to shared_data per §18.4. The shared_data write key is cfg_.name;
 *  value is a length-n arma::vec of doubles (integer-encoded z_i in
 *  {0, ..., K_i - 1}). Same R-side semantic as Block 4.
 *
 *  STATE EXPOSED VIA current()
 *  ===========================
 *  current() returns the concatenated phi_C vectors (length
 *  total_S = sum_C S_C) — the q-mean point estimate over CLIQUE
 *  joint states. R-level get_current() consumers can pass each
 *  slice to a per-clique marginalisation helper to obtain per-node
 *  marginals.
 *
 *  HISTORY
 *  =======
 *  When keep_history=true:
 *      vi_history_t.elbo   : per-step ELBO
 *      vi_history_t.mu     : per-step eta concat (length total_D = sum_C (S_C-1))
 *      vi_history_t.log_sd : per-step phi concat (length total_S = sum_C S_C)
 *      vi_history_t.gamma, .epoch, .final_khat: standard
 *
 *  VALIDATOR
 *  =========
 *  - Check #15 (parity): structured MF degeneracy + finite-diff
 *      gradient checks in tests/test_structured_categorical_vi_block.cpp.
 *  - Check #21 (vi_block contract): inherits from vi_block.
 *  - Check #22 (optimizer = RAABBVI): vi_optimizer::avgAdam_step.
 *  - Check #23 (Layer-3 PSIS-k̂): joint k̂ at SKL termination.
 *================================================================================*/

#ifndef AI4BAYESCODE_STRUCTURED_CATEGORICAL_VI_BLOCK_HPP
#define AI4BAYESCODE_STRUCTURED_CATEGORICAL_VI_BLOCK_HPP

#include "vi_block.hpp"
#include "vi_optimizer.hpp"

#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef MCMC_USE_RCPP_ARMADILLO
# include <armadillo>
#else
# include <RcppArmadillo.h>
#endif

namespace AI4BayesCode {

/// Same callback signature as Block 4: log p~(z, ctx) evaluated on an
/// integer-encoded z vector.
using structured_log_density_fn =
    std::function<double(const arma::uvec& z, const block_context& ctx)>;

/**
 * @brief Configuration for structured_categorical_vi_block.
 *
 * Required fields: name, cardinalities, clique_partition, log_density.
 *
 * `clique_partition` must be a PARTITION of {0, 1, ..., n-1}:
 *   - Every node index appears in exactly one clique.
 *   - No empty cliques.
 *   - No duplicates within a clique.
 *
 * Singleton cliques are valid (block reduces to Block 4).
 * Single all-node clique is valid (block is exact inference via
 * enumeration over the full joint state).
 */
struct structured_categorical_vi_block_config {
    /// Block name; becomes the shared_data key holding the per-step
    /// q-sample (length-n arma::vec of integer indices).
    std::string name;

    /// Cardinality K_i of each latent. Length n. K_i >= 2.
    arma::uvec cardinalities;

    /// Clique partition. Inner vectors hold node indices; together they
    /// must cover {0, ..., n-1} exactly once.
    std::vector<std::vector<std::size_t>> clique_partition;

    /// log p~(z, ctx) on integer-encoded z, same as Block 4.
    structured_log_density_fn log_density;

    /// shared_data keys this block reads via ctx in step().
    std::vector<std::string> dependencies;

    /// Number of MC samples per gradient evaluation in MC mode. Default 16.
    std::size_t n_mc_samples = 16;

    /// If true (and joint state prod_i K_i <= exact_state_cap),
    /// gradient is computed via full enumeration. EXACT, zero variance.
    bool exact_enumeration = false;

    /// Cap on full joint state count for exact_enumeration. Default 4096.
    /// Beyond this the block warns (via warning_fn if set) and falls
    /// back to MC.
    std::size_t exact_state_cap = 4096;

    /// Optional non-fatal-warning callback.
    std::function<void(const std::string& msg)> warning_fn;

    /// Initial perturbation magnitude on eta. Breaks symmetry for
    /// genuinely coupled posteriors. Default 0 (pure uniform start).
    double init_random_eps = 0.0;

    /// RAABBVI optimizer hyperparams.
    vi_optimizer::raabbvi_config optimizer;

    /// Seed for the init perturbation; independent of the step() rng.
    std::uint64_t init_rng_seed = 0;
};

/**
 * @brief Saul-Jordan 1996 structured (partially factorised) mean-field
 *        VI for discrete categorical latents.
 *
 * See file header for the math, gradient, and contract details.
 */
class structured_categorical_vi_block : public vi_block {
public:
    explicit structured_categorical_vi_block(
        structured_categorical_vi_block_config cfg)
        : cfg_(std::move(cfg)) {

        if (cfg_.name.empty()) {
            throw std::invalid_argument(
                "structured_categorical_vi_block: cfg.name required");
        }
        if (cfg_.cardinalities.is_empty()) {
            throw std::invalid_argument(
                "structured_categorical_vi_block: cfg.cardinalities required");
        }
        if (cfg_.clique_partition.empty()) {
            throw std::invalid_argument(
                "structured_categorical_vi_block: cfg.clique_partition required");
        }
        if (!cfg_.log_density) {
            throw std::invalid_argument(
                "structured_categorical_vi_block: cfg.log_density required");
        }
        for (std::size_t i = 0; i < cfg_.cardinalities.n_elem; ++i) {
            if (cfg_.cardinalities[i] < 2u) {
                throw std::invalid_argument(
                    "structured_categorical_vi_block: every K_i must be >= 2");
            }
        }

        n_ = cfg_.cardinalities.n_elem;

        // ---- Validate clique partition + build lookup tables --------
        var_to_clique_.assign(n_, std::numeric_limits<std::size_t>::max());
        var_pos_in_clique_.assign(n_, 0);
        for (std::size_t c = 0; c < cfg_.clique_partition.size(); ++c) {
            const auto& C = cfg_.clique_partition[c];
            if (C.empty()) {
                throw std::invalid_argument(
                    "structured_categorical_vi_block: empty clique " + std::to_string(c));
            }
            for (std::size_t pos = 0; pos < C.size(); ++pos) {
                const std::size_t i = C[pos];
                if (i >= n_) {
                    throw std::invalid_argument(
                        "structured_categorical_vi_block: clique " + std::to_string(c)
                        + " contains out-of-range index " + std::to_string(i));
                }
                if (var_to_clique_[i] != std::numeric_limits<std::size_t>::max()) {
                    throw std::invalid_argument(
                        "structured_categorical_vi_block: node " + std::to_string(i)
                        + " appears in more than one clique");
                }
                var_to_clique_[i] = c;
                var_pos_in_clique_[i] = pos;
            }
        }
        for (std::size_t i = 0; i < n_; ++i) {
            if (var_to_clique_[i] == std::numeric_limits<std::size_t>::max()) {
                throw std::invalid_argument(
                    "structured_categorical_vi_block: node " + std::to_string(i)
                    + " not in any clique (partition not a cover)");
            }
        }
        M_ = cfg_.clique_partition.size();

        // ---- Compute clique state counts S_C = prod K_i for i in C --
        clique_state_counts_.assign(M_, 1);
        for (std::size_t c = 0; c < M_; ++c) {
            std::size_t S = 1;
            for (std::size_t i : cfg_.clique_partition[c]) {
                const std::size_t prev = S;
                S *= cfg_.cardinalities[i];
                if (S < prev) {
                    throw std::invalid_argument(
                        "structured_categorical_vi_block: clique " + std::to_string(c)
                        + " joint state count overflows size_t");
                }
            }
            clique_state_counts_[c] = S;
        }

        // ---- Build offsets ------------------------------------------
        offsets_eta_.assign(M_ + 1, 0);
        offsets_phi_.assign(M_ + 1, 0);
        for (std::size_t c = 0; c < M_; ++c) {
            const std::size_t S = clique_state_counts_[c];
            offsets_eta_[c + 1] = offsets_eta_[c] + (S - 1);
            offsets_phi_[c + 1] = offsets_phi_[c] + S;
        }
        total_D_ = offsets_eta_[M_];
        total_S_ = offsets_phi_[M_];

        // ---- Check exact-mode tractability --------------------------
        if (cfg_.exact_enumeration) {
            std::size_t total = 1;
            for (std::size_t i = 0; i < n_; ++i) {
                const std::size_t prev = total;
                total *= cfg_.cardinalities[i];
                if (total < prev) {
                    total = std::numeric_limits<std::size_t>::max();
                    break;
                }
            }
            if (total > cfg_.exact_state_cap) {
                if (cfg_.warning_fn) {
                    cfg_.warning_fn(
                        "structured_categorical_vi_block: joint state " +
                        std::to_string(total) + " > exact_state_cap; "
                        "falling back to MC.");
                }
                exact_active_ = false;
            } else {
                exact_active_ = true;
                exact_total_states_ = total;
            }
        }

        // ---- Initialise eta = 0 (phi uniform within each clique) ----
        eta_.zeros(total_D_);
        if (cfg_.init_random_eps > 0.0) {
            std::mt19937_64 init_rng(cfg_.init_rng_seed
                                      ^ 0xC2B2AE3D27D4EB4FULL);
            std::normal_distribution<double> N(0.0, cfg_.init_random_eps);
            for (std::size_t i = 0; i < total_D_; ++i) eta_[i] = N(init_rng);
        }
        phi_.zeros(total_S_);
        update_phi_from_eta_();

        // ---- Initial epoch state ------------------------------------
        epoch_eta_avg_history_.push_back(eta_);
        opt_state_.reset_epoch(total_D_);
        gamma_current_  = cfg_.optimizer.gamma_0;
        epoch_index_    = 0;
        steps_in_epoch_ = 0;
        converged_      = false;
        last_elbo_      = std::numeric_limits<double>::quiet_NaN();
    }

    // ---- block_sampler interface ----------------------------------------

    void set_context(const block_context& ctx) override {
        ctx_.clear();
        for (const auto& key : cfg_.dependencies) {
            auto it = ctx.find(key);
            if (it != ctx.end()) ctx_.emplace(key, it->second);
        }
    }

    void step(std::mt19937_64& rng) override {
        if (converged_) return;

        arma::vec grad_eta(total_D_, arma::fill::zeros);
        const double elbo = compute_elbo_and_grad_(rng, &grad_eta);
        last_elbo_ = elbo;

        // RAABBVI minimises -ELBO; we computed +ELBO gradient, negate.
        arma::vec neg_grad = -grad_eta;
        vi_optimizer::avgAdam_step(eta_, opt_state_, neg_grad,
                                    gamma_current_, cfg_.optimizer);
        update_phi_from_eta_();

        steps_in_epoch_ += 1;

        if (keep_history()) {
            history_.elbo.push_back(elbo);
            history_.mu.push_back(eta_);
            history_.log_sd.push_back(phi_);
            history_.gamma.push_back(gamma_current_);
            history_.epoch.push_back(static_cast<int>(epoch_index_));
        }

        if (steps_in_epoch_ >= cfg_.optimizer.inner_iter_per_epoch) {
            close_epoch_and_maybe_terminate_(rng);
        }
    }

    const arma::vec& current() const override { return phi_; }

    void set_current(const arma::vec& phi_new) override {
        if (phi_new.n_elem != total_S_) {
            throw std::invalid_argument(
                "structured_categorical_vi_block::set_current: expected length "
                + std::to_string(total_S_) + ", got "
                + std::to_string(phi_new.n_elem));
        }
        for (std::size_t c = 0; c < M_; ++c) {
            const std::size_t S = clique_state_counts_[c];
            double sum = 0.0;
            for (std::size_t s = 0; s < S; ++s) {
                const double v = phi_new[offsets_phi_[c] + s];
                if (!(v >= 0.0) || !std::isfinite(v)) {
                    throw std::invalid_argument(
                        "structured_categorical_vi_block::set_current: phi "
                        "must be non-negative and finite");
                }
                sum += v;
            }
            if (std::abs(sum - 1.0) > 1e-6) {
                throw std::invalid_argument(
                    "structured_categorical_vi_block::set_current: phi for clique "
                    + std::to_string(c) + " must sum to 1 (got "
                    + std::to_string(sum) + ")");
            }
        }
        const double eps = 1e-12;
        for (std::size_t c = 0; c < M_; ++c) {
            const std::size_t S = clique_state_counts_[c];
            const double phi_0 = std::max(phi_new[offsets_phi_[c]], eps);
            for (std::size_t l = 1; l < S; ++l) {
                const double phi_l = std::max(phi_new[offsets_phi_[c] + l], eps);
                eta_[offsets_eta_[c] + (l - 1)] =
                    std::log(phi_l) - std::log(phi_0);
            }
        }
        update_phi_from_eta_();
    }

    const std::string& name() const noexcept override { return cfg_.name; }
    std::size_t dim() const noexcept override { return n_; }

    history_map get_history() const override {
        history_map out;
        const std::size_t T = history_.elbo.size();
        if (T == 0) {
            arma::mat m(1, total_S_);
            for (std::size_t j = 0; j < total_S_; ++j) m(0, j) = phi_[j];
            out.emplace(cfg_.name, std::move(m));
            return out;
        }
        const std::size_t cols = 3 + total_D_ + total_S_ + 1;
        arma::mat m(T, cols);
        for (std::size_t t = 0; t < T; ++t) {
            m(t, 0) = history_.elbo[t];
            m(t, 1) = history_.gamma[t];
            m(t, 2) = static_cast<double>(history_.epoch[t]);
            for (std::size_t j = 0; j < total_D_; ++j) {
                m(t, 3 + j) = history_.mu[t][j];
            }
            for (std::size_t j = 0; j < total_S_; ++j) {
                m(t, 3 + total_D_ + j) = history_.log_sd[t][j];
            }
            m(t, cols - 1) = history_.final_khat;
        }
        out.emplace(cfg_.name + "__vi_history", std::move(m));
        return out;
    }

    std::size_t history_size() const noexcept override {
        return history_.elbo.size();
    }

    void clear_history() override { history_ = vi_history_t{}; }

    // ---- vi_block interface ---------------------------------------------

    /// Draw z ~ q. For each clique, draw one joint state from
    /// Categorical(phi_C); decode to per-node integers in clique nodes.
    /// Returns length-n arma::vec of doubles.
    arma::vec current_sample(std::mt19937_64& rng) const override {
        arma::vec z(n_, arma::fill::zeros);
        std::uniform_real_distribution<double> U(0.0, 1.0);
        for (std::size_t c = 0; c < M_; ++c) {
            const std::size_t S = clique_state_counts_[c];
            const double u = U(rng);
            double acc = 0.0;
            std::size_t pick = S - 1;
            for (std::size_t s = 0; s < S; ++s) {
                acc += phi_[offsets_phi_[c] + s];
                if (u <= acc) { pick = s; break; }
            }
            // Decode pick → per-node values in this clique.
            std::vector<std::size_t> vals(cfg_.clique_partition[c].size());
            decode_clique_state_(c, pick, vals);
            for (std::size_t p = 0; p < vals.size(); ++p) {
                const std::size_t i = cfg_.clique_partition[c][p];
                z[i] = static_cast<double>(vals[p]);
            }
        }
        return z;
    }

    arma::vec get_log_sd() const override { return eta_; }

    void set_variational_state(const arma::vec& mu_or_phi,
                                const arma::vec& log_sd_or_eta) override {
        if (log_sd_or_eta.n_elem == total_D_) {
            eta_ = log_sd_or_eta;
            update_phi_from_eta_();
        } else if (mu_or_phi.n_elem == total_S_) {
            set_current(mu_or_phi);
        } else {
            throw std::invalid_argument(
                "structured_categorical_vi_block::set_variational_state: "
                "neither phi (length total_S) nor eta (length total_D) "
                "matches expected dimensions");
        }
    }

    double current_elbo() const override { return last_elbo_; }
    const vi_history_t& vi_history() const override { return history_; }

    // ---- Public diagnostics ---------------------------------------------

    bool converged() const noexcept { return converged_; }
    std::size_t epoch() const noexcept { return epoch_index_; }
    double gamma() const noexcept { return gamma_current_; }
    std::size_t n_cliques() const noexcept { return M_; }
    std::size_t total_S() const noexcept { return total_S_; }
    std::size_t total_D() const noexcept { return total_D_; }
    bool exact_active() const noexcept { return exact_active_; }
    std::size_t clique_state_count(std::size_t c) const {
        return clique_state_counts_[c];
    }

    /// Per-node marginal q_i(z_i = k) computed by marginalising the
    /// clique joint phi. Returns n × K_max matrix; columns past K_i
    /// for variable i are 0.
    arma::mat per_node_marginals() const {
        std::size_t K_max = 0;
        for (std::size_t i = 0; i < n_; ++i)
            if (cfg_.cardinalities[i] > K_max) K_max = cfg_.cardinalities[i];
        arma::mat M(n_, K_max, arma::fill::zeros);
        for (std::size_t c = 0; c < M_; ++c) {
            const auto& C = cfg_.clique_partition[c];
            const std::size_t S = clique_state_counts_[c];
            std::vector<std::size_t> vals(C.size());
            for (std::size_t s = 0; s < S; ++s) {
                decode_clique_state_(c, s, vals);
                const double p = phi_[offsets_phi_[c] + s];
                for (std::size_t pos = 0; pos < C.size(); ++pos) {
                    M(C[pos], vals[pos]) += p;
                }
            }
        }
        return M;
    }

private:
    // ---- Encoding / decoding helpers -----------------------------------

    /// Decode an integer joint-state index for clique c into per-position
    /// values. vals has length |clique c|.
    void decode_clique_state_(std::size_t c, std::size_t s,
                               std::vector<std::size_t>& vals) const {
        const auto& C = cfg_.clique_partition[c];
        for (std::size_t pos = 0; pos < C.size(); ++pos) {
            const std::size_t Ki = cfg_.cardinalities[C[pos]];
            vals[pos] = s % Ki;
            s /= Ki;
        }
    }

    /// Encode per-node values (provided for nodes in clique c, in clique
    /// order) into an integer joint state.
    std::size_t encode_clique_state_(std::size_t c,
                                       const std::vector<std::size_t>& vals)
        const {
        const auto& C = cfg_.clique_partition[c];
        std::size_t s = 0;
        std::size_t mult = 1;
        for (std::size_t pos = 0; pos < C.size(); ++pos) {
            s += vals[pos] * mult;
            mult *= cfg_.cardinalities[C[pos]];
        }
        return s;
    }

    /// Apply per-clique anchored softmax: phi from eta.
    void update_phi_from_eta_() {
        for (std::size_t c = 0; c < M_; ++c) {
            const std::size_t S = clique_state_counts_[c];
            const std::size_t eb = offsets_eta_[c];
            const std::size_t pb = offsets_phi_[c];
            double max_eta = 0.0;  // eta_C(0) = 0 implicit
            for (std::size_t l = 1; l < S; ++l) {
                const double e = eta_[eb + (l - 1)];
                if (e > max_eta) max_eta = e;
            }
            double Z = std::exp(0.0 - max_eta);
            for (std::size_t l = 1; l < S; ++l) {
                Z += std::exp(eta_[eb + (l - 1)] - max_eta);
            }
            phi_[pb + 0] = std::exp(0.0 - max_eta) / Z;
            for (std::size_t l = 1; l < S; ++l) {
                phi_[pb + l] = std::exp(eta_[eb + (l - 1)] - max_eta) / Z;
            }
        }
    }

    /// Compute ELBO and gradient w.r.t. eta in one pass.
    double compute_elbo_and_grad_(std::mt19937_64& rng,
                                    arma::vec* g_eta) const {
        arma::vec G(total_S_, arma::fill::zeros);

        if (exact_active_) compute_G_exact_(G);
        else               compute_G_mc_(rng, G, cfg_.n_mc_samples);

        // g_phi_C(s) = G_C(s) - log phi_C(s). The -1 cancels in chain rule.
        arma::vec g_phi(total_S_);
        for (std::size_t c = 0; c < M_; ++c) {
            const std::size_t S = clique_state_counts_[c];
            const std::size_t pb = offsets_phi_[c];
            for (std::size_t s = 0; s < S; ++s) {
                const double phi_cs = phi_[pb + s];
                const double log_phi = (phi_cs > 0.0)
                                         ? std::log(phi_cs)
                                         : -std::numeric_limits<double>::infinity();
                g_phi[pb + s] = G[pb + s] - log_phi;
            }
        }

        // Chain through anchored softmax per clique:
        //   dELBO/deta_C(l) = phi_C(l) * [(g_phi_C(l)) - sum_s phi_C(s) g_phi_C(s)]
        for (std::size_t c = 0; c < M_; ++c) {
            const std::size_t S = clique_state_counts_[c];
            const std::size_t pb = offsets_phi_[c];
            const std::size_t eb = offsets_eta_[c];
            double centre = 0.0;
            for (std::size_t s = 0; s < S; ++s)
                centre += phi_[pb + s] * g_phi[pb + s];
            for (std::size_t l = 1; l < S; ++l) {
                const double phi_l = phi_[pb + l];
                (*g_eta)[eb + (l - 1)] = phi_l * (g_phi[pb + l] - centre);
            }
        }

        // ELBO Rao-Blackwellised via clique c=0:
        //   E_q[log p~] ≈ sum_s phi_C0(s) * G_C0(s)
        const std::size_t S0 = clique_state_counts_[0];
        double e_log_p = 0.0;
        for (std::size_t s = 0; s < S0; ++s) {
            e_log_p += phi_[s] * G[s];
        }
        // Entropy: sum_C [ -sum_s phi_C(s) log phi_C(s) ]
        double H_total = 0.0;
        for (std::size_t c = 0; c < M_; ++c) {
            const std::size_t S = clique_state_counts_[c];
            const std::size_t pb = offsets_phi_[c];
            for (std::size_t s = 0; s < S; ++s) {
                const double p = phi_[pb + s];
                if (p > 0.0) H_total -= p * std::log(p);
            }
        }
        return e_log_p + H_total;
    }

    /// Exact enumeration of the FULL joint state space.
    void compute_G_exact_(arma::vec& G_out) const {
        // Joint state count.
        std::size_t total = 1;
        for (std::size_t i = 0; i < n_; ++i) total *= cfg_.cardinalities[i];

        arma::uvec z(n_);
        // q(z) = prod_C phi_C(state of clique C induced by z).
        // For each global z, induce per-clique state s_c via encoding,
        // accumulate G_C(s_c) += w_marg(c) * log_density.
        // w_marg(c) = q(z) / phi_C(s_c) = prod_{c' != c} phi_{c'}(s_{c'}).
        std::vector<std::size_t> nbr_vals;
        for (std::size_t s = 0; s < total; ++s) {
            std::size_t r = s;
            for (std::size_t i = 0; i < n_; ++i) {
                z[i] = r % cfg_.cardinalities[i];
                r /= cfg_.cardinalities[i];
            }
            const double lp = cfg_.log_density(z, ctx_);

            // Determine each clique's induced state.
            std::vector<std::size_t> s_c(M_);
            double w_full = 1.0;
            for (std::size_t c = 0; c < M_; ++c) {
                const auto& C = cfg_.clique_partition[c];
                nbr_vals.assign(C.size(), 0);
                for (std::size_t pos = 0; pos < C.size(); ++pos) {
                    nbr_vals[pos] = z[C[pos]];
                }
                s_c[c] = encode_clique_state_(c, nbr_vals);
                w_full *= phi_[offsets_phi_[c] + s_c[c]];
            }
            if (!(w_full > 0.0)) continue;

            for (std::size_t c = 0; c < M_; ++c) {
                const double phi_cs = phi_[offsets_phi_[c] + s_c[c]];
                if (!(phi_cs > 0.0)) continue;
                const double w_marg = w_full / phi_cs;
                G_out[offsets_phi_[c] + s_c[c]] += w_marg * lp;
            }
        }
    }

    /// MC estimator: sample z_s ~ q, then for each clique C and each
    /// clique joint state q_idx, substitute z[C] with decode_C(q_idx)
    /// and evaluate log_density.
    void compute_G_mc_(std::mt19937_64& rng, arma::vec& G_out,
                       std::size_t S) const {
        if (S == 0) S = 1;
        arma::uvec z(n_);
        std::uniform_real_distribution<double> U(0.0, 1.0);
        const double inv_S = 1.0 / static_cast<double>(S);

        std::vector<std::size_t> vals;

        for (std::size_t s = 0; s < S; ++s) {
            // Sample z ~ q.
            for (std::size_t c = 0; c < M_; ++c) {
                const std::size_t Sc = clique_state_counts_[c];
                const double u = U(rng);
                double acc = 0.0;
                std::size_t pick = Sc - 1;
                for (std::size_t q = 0; q < Sc; ++q) {
                    acc += phi_[offsets_phi_[c] + q];
                    if (u <= acc) { pick = q; break; }
                }
                vals.assign(cfg_.clique_partition[c].size(), 0);
                decode_clique_state_(c, pick, vals);
                for (std::size_t p = 0; p < vals.size(); ++p) {
                    z[cfg_.clique_partition[c][p]] = vals[p];
                }
            }

            // For each clique and each clique joint state, substitute
            // and evaluate. Save originals to restore.
            for (std::size_t c = 0; c < M_; ++c) {
                const auto& C = cfg_.clique_partition[c];
                const std::size_t Sc = clique_state_counts_[c];

                // Save original values for this clique.
                std::vector<std::size_t> saved(C.size());
                for (std::size_t p = 0; p < C.size(); ++p) saved[p] = z[C[p]];

                vals.assign(C.size(), 0);
                for (std::size_t q = 0; q < Sc; ++q) {
                    decode_clique_state_(c, q, vals);
                    for (std::size_t p = 0; p < C.size(); ++p) z[C[p]] = vals[p];
                    const double lp = cfg_.log_density(z, ctx_);
                    G_out[offsets_phi_[c] + q] += inv_S * lp;
                }

                // Restore.
                for (std::size_t p = 0; p < C.size(); ++p) z[C[p]] = saved[p];
            }
        }
    }

    /// SKL termination check + epoch shrink.
    void close_epoch_and_maybe_terminate_(std::mt19937_64& rng) {
        const arma::vec eta_bar = opt_state_.lambda_bar;
        epoch_eta_avg_history_.push_back(eta_bar);
        const std::size_t H = epoch_eta_avg_history_.size();

        if (H >= 3) {
            const arma::vec& eta_prev = epoch_eta_avg_history_[H - 2];
            const arma::vec& eta_init = epoch_eta_avg_history_[0];
            const double skl_c = skl_eta_(eta_bar, eta_prev);
            const double skl_i = skl_eta_(eta_bar, eta_init);
            if (vi_optimizer::skl_terminate(skl_c, skl_i,
                                              cfg_.optimizer.tau)) {
                terminate_(eta_bar, rng);
                return;
            }
        }
        if (epoch_index_ + 1 >= cfg_.optimizer.max_epochs) {
            terminate_(eta_bar, rng);
            return;
        }
        eta_ = eta_bar;
        update_phi_from_eta_();
        gamma_current_ *= cfg_.optimizer.rho;
        epoch_index_  += 1;
        steps_in_epoch_ = 0;
        opt_state_.reset_epoch(total_D_);
    }

    /// SKL across all cliques: sum of per-clique SKL of two Categoricals.
    double skl_eta_(const arma::vec& eta_a, const arma::vec& eta_b) const {
        if (eta_a.n_elem != total_D_ || eta_b.n_elem != total_D_) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        double total = 0.0;
        const double eps = 1e-30;
        for (std::size_t c = 0; c < M_; ++c) {
            const std::size_t S = clique_state_counts_[c];
            const std::size_t eb = offsets_eta_[c];
            // softmax a, b.
            double max_a = 0.0, max_b = 0.0;
            for (std::size_t l = 1; l < S; ++l) {
                const double ea = eta_a[eb + (l - 1)];
                const double eb_ = eta_b[eb + (l - 1)];
                if (ea > max_a) max_a = ea;
                if (eb_ > max_b) max_b = eb_;
            }
            std::vector<double> pa(S), pb(S);
            double Za = std::exp(-max_a), Zb = std::exp(-max_b);
            for (std::size_t l = 1; l < S; ++l) {
                Za += std::exp(eta_a[eb + (l - 1)] - max_a);
                Zb += std::exp(eta_b[eb + (l - 1)] - max_b);
            }
            pa[0] = std::exp(-max_a) / Za;
            pb[0] = std::exp(-max_b) / Zb;
            for (std::size_t l = 1; l < S; ++l) {
                pa[l] = std::exp(eta_a[eb + (l - 1)] - max_a) / Za;
                pb[l] = std::exp(eta_b[eb + (l - 1)] - max_b) / Zb;
            }
            for (std::size_t s = 0; s < S; ++s) {
                const double a = std::max(pa[s], eps);
                const double b = std::max(pb[s], eps);
                total += pa[s] * (std::log(a) - std::log(b))
                       + pb[s] * (std::log(b) - std::log(a));
            }
        }
        return total;
    }

    /// Final termination: lock eta, compute joint PSIS-k̂ over S samples.
    void terminate_(const arma::vec& eta_final, std::mt19937_64& rng) {
        eta_ = eta_final;
        update_phi_from_eta_();
        converged_ = true;

        const std::size_t S = cfg_.optimizer.S_khat;
        arma::vec log_weights(S);
        arma::uvec z(n_);
        std::uniform_real_distribution<double> U(0.0, 1.0);
        std::vector<std::size_t> vals;

        for (std::size_t s = 0; s < S; ++s) {
            double log_q = 0.0;
            for (std::size_t c = 0; c < M_; ++c) {
                const std::size_t Sc = clique_state_counts_[c];
                const double u = U(rng);
                double acc = 0.0;
                std::size_t pick = Sc - 1;
                for (std::size_t q = 0; q < Sc; ++q) {
                    acc += phi_[offsets_phi_[c] + q];
                    if (u <= acc) { pick = q; break; }
                }
                vals.assign(cfg_.clique_partition[c].size(), 0);
                decode_clique_state_(c, pick, vals);
                for (std::size_t p = 0; p < vals.size(); ++p) {
                    z[cfg_.clique_partition[c][p]] = vals[p];
                }
                const double phi_pick = std::max(phi_[offsets_phi_[c] + pick],
                                                   1e-300);
                log_q += std::log(phi_pick);
            }
            const double log_p = cfg_.log_density(z, ctx_);
            log_weights[s] = log_p - log_q;
        }
        history_.final_khat = vi_optimizer::psis_khat(log_weights);
    }

    // ---- State ----------------------------------------------------------

    structured_categorical_vi_block_config cfg_;
    std::size_t n_       = 0;     // number of latents
    std::size_t M_       = 0;     // number of cliques
    std::size_t total_S_ = 0;     // sum_C S_C
    std::size_t total_D_ = 0;     // sum_C (S_C - 1)

    std::vector<std::size_t> clique_state_counts_;       // length M
    std::vector<std::size_t> offsets_eta_;               // length M+1
    std::vector<std::size_t> offsets_phi_;               // length M+1
    std::vector<std::size_t> var_to_clique_;             // length n
    std::vector<std::size_t> var_pos_in_clique_;         // length n

    arma::vec eta_;
    arma::vec phi_;

    block_context ctx_;

    vi_optimizer::avgAdam_state opt_state_;
    std::vector<arma::vec> epoch_eta_avg_history_;
    double gamma_current_  = 0.1;
    std::size_t epoch_index_     = 0;
    std::size_t steps_in_epoch_  = 0;
    bool converged_        = false;
    double last_elbo_      = std::numeric_limits<double>::quiet_NaN();

    bool exact_active_        = false;
    std::size_t exact_total_states_ = 0;

    vi_history_t history_;
};

} // namespace AI4BayesCode

#endif // AI4BAYESCODE_STRUCTURED_CATEGORICAL_VI_BLOCK_HPP
