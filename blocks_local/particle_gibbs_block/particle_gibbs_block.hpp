/*================================================================================
 *  particle_gibbs_block.hpp -- Particle Gibbs (conditional SMC) sampler for the
 *                              latent state path of a general nonlinear /
 *                              non-Gaussian state-space model.
 *
 *  Copyright (C) 2026 AI4BayesCode contributors.
 *  Licensed under the GNU General Public License v3.0 or later
 *  (GPL-3.0-or-later). See COPYING / LICENSE at the repo root.
 *================================================================================
 *
 *  TARGET (geometry.md SS11.1, shape 1 -- fixed-dim absolutely continuous).
 *  ---------------------------------------------------------------------------
 *  Given a state-space model with parameters theta (read from context),
 *
 *      x_1         ~ mu_theta(.)                       (initial)
 *      x_t|x_{t-1} ~ f_theta(. | x_{t-1}),  t=2..T     (transition)
 *      y_t|x_t     ~ g_theta(. | x_t),      t=1..T     (observation)
 *
 *  this block samples the latent path x_{1:T} from its full conditional
 *
 *      p(x_{1:T} | y_{1:T}, theta)
 *        ~ mu(x_1) g(y_1|x_1) prod_{t=2}^T f(x_t|x_{t-1}) g(y_t|x_t).
 *
 *  Each x_t lives in R^d (d = state_dim); the path is a fixed-dim (d*T)
 *  absolutely-continuous vector. The model-specific densities are injected as
 *  callbacks, so the SAME block serves stochastic volatility, nonlinear dynamic,
 *  and state-space count models. theta is NOT sampled here -- it is read from the
 *  context each sweep (a sibling block samples it, or it is held fixed), making
 *  this the path-update half of a full particle Gibbs scheme.
 *
 *  ALGORITHM -- Particle Gibbs / conditional SMC (Andrieu, Doucet & Holenstein
 *  2010, JRSS-B 72(3):269-342, doi:10.1111/j.1467-9868.2009.00736.x), with
 *  optional ancestor sampling (PGAS; Lindsten, Jordan & Schon 2014, JMLR
 *  15:2145-2184).
 *
 *  One step() = one conditional SMC sweep that retains the current reference
 *  trajectory x' in particle slot 0 and resamples a new path:
 *
 *    t=1: slot 0 := x'_1; slots 1..N-1 ~ mu_theta; log-weights = log g(y_1|.).
 *    t=2..T:
 *      - resample ancestors of slots 1..N-1 from the normalized weights W_{t-1};
 *      - reference ancestor a^0_t: vanilla PG -> 0; PGAS -> j with prob.
 *        proportional to W_{t-1}^j f(x'_t | x_{t-1}^j) (breaks path degeneracy);
 *      - propagate slots 1..N-1 ~ f(. | x_{t-1}^{anc}); slot 0 := x'_t;
 *      - log-weights = log g(y_t|.).
 *    Draw terminal index k ~ Categorical(W_T); trace its lineage back to t=1;
 *    the resulting path replaces the reference.
 *
 *  CORRECTNESS. The conditional-SMC kernel leaves p(x_{1:T}|y,theta) exactly
 *  invariant for ANY N >= 2 (ADH 2010, Thm 5); N is purely a mixing/cost knob,
 *  never a correctness knob. Vanilla PG mixes poorly on long series (path
 *  degeneracy: early states collapse to one ancestor); ancestor sampling fixes
 *  this while preserving invariance, so it is the default.
 *
 *  PROPOSAL. Bootstrap (propagate via the transition prior, weight by the
 *  observation density). Only init_sample / transition_sample / obs_loglik are
 *  required; transition_logpdf is needed only when ancestor_sampling=true.
 *  Guided / locally-optimal proposals are a future extension.
 *
 *  JUSTIFICATION (Check #17): Exception 4 (codegen_priors.md SS2b -- AI-authored
 *  custom block) -- no blessed block fits because the target is a continuous
 *  latent path of a nonlinear/non-Gaussian state-space model with strong
 *  sequential (Markov) dependence: hmm_block is discrete-state forward-backward,
 *  the gmrf_* blocks are Gaussian-Markov random fields, and no conjugate-Gibbs
 *  full conditional exists for a nonlinear/non-Gaussian observation. Joint/single
 *  NUTS structurally inapplicable because the generic class allows
 *  non-differentiable / intractable f,g (no gradient oracle), and even when
 *  differentiable the high-dim strongly-coupled path is better served by a
 *  structured sequential-Monte-Carlo move than by a gradient sampler. Custom
 *  scheme = conditional SMC with ancestor sampling (the continuous, non-Gaussian
 *  generalization of HMM forward-backward); it targets the correct posterior
 *  because the cSMC kernel is invariant for any N>=2 (ADH 2010, Thm 5) and
 *  ancestor sampling preserves that invariance (Lindsten 2014).
 *
 *  STATEFUL CONTRACT.
 *    set_context(ctx)  copies ctx; the callbacks read theta / y_t from it.
 *    step(rng)         advances the reference path by one cSMC/PGAS sweep,
 *                      threading rng through every stochastic op (no hidden RNG).
 *    current()         the current reference path, flat length d*T.
 *    set_current(p)    force the reference path (warm start / external injection).
 *    keep_history      appends a copy of the path each step.
 *  First sweep with no reference (and no initial_path): seeded by one
 *  unconditional particle filter.
 *================================================================================*/

#ifndef AI4BAYESCODE_PARTICLE_GIBBS_BLOCK_HPP
#define AI4BAYESCODE_PARTICLE_GIBBS_BLOCK_HPP

#include "AI4BayesCode/block_sampler.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace AI4BayesCode {

/// Resampling scheme used inside each conditional-SMC sweep.
enum class pg_resampling_scheme {
    SYSTEMATIC,   ///< single uniform offset; lowest resampling variance (default)
    STRATIFIED,   ///< one uniform per stratum
    MULTINOMIAL   ///< N i.i.d. categorical draws (highest variance; reference baseline)
};

namespace detail_particle_gibbs {

/// log-sum-exp of log-weights (entries may be -inf). Returns -inf if all -inf.
inline double log_sum_exp(const arma::vec& logw) {
    double m = -std::numeric_limits<double>::infinity();
    for (arma::uword i = 0; i < logw.n_elem; ++i)
        if (logw[i] > m) m = logw[i];
    if (!std::isfinite(m)) return m;
    double s = 0.0;
    for (arma::uword i = 0; i < logw.n_elem; ++i)
        s += std::exp(logw[i] - m);
    return m + std::log(s);
}

/// Normalize log-weights to probabilities. All-(-inf) -> uniform (degenerate
/// but valid: no information to discriminate the particles).
inline arma::vec normalize_log_weights(const arma::vec& logw_in) {
    const arma::uword N = logw_in.n_elem;
    arma::vec logw = logw_in;
    for (arma::uword i = 0; i < N; ++i)
        if (std::isnan(logw[i])) logw[i] = -std::numeric_limits<double>::infinity();
    const double lse = log_sum_exp(logw);
    arma::vec W(N);
    if (!std::isfinite(lse)) { W.fill(1.0 / static_cast<double>(N)); return W; }
    for (arma::uword i = 0; i < N; ++i) W[i] = std::exp(logw[i] - lse);
    const double s = arma::accu(W);
    if (s > 0.0) W /= s; else W.fill(1.0 / static_cast<double>(N));
    return W;
}

/// Single categorical index from normalized weights W given u in [0,1).
inline arma::uword categorical_from_W(const arma::vec& W, double u) {
    const arma::uword N = W.n_elem;
    double c = 0.0;
    for (arma::uword i = 0; i < N; ++i) { c += W[i]; if (u < c) return i; }
    return N - 1;  // round-off fallback
}

/// Draw m ancestor indices from normalized weights W under the chosen scheme.
/// Systematic/stratified use ordered uniforms through the CDF (both monotone in
/// k, so a single forward pointer is correct).
inline std::vector<arma::uword> resample(const arma::vec& W,
                                         std::size_t m,
                                         pg_resampling_scheme scheme,
                                         std::mt19937_64& rng) {
    const arma::uword N = W.n_elem;
    std::vector<arma::uword> out(m);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    if (scheme == pg_resampling_scheme::MULTINOMIAL) {
        for (std::size_t k = 0; k < m; ++k) out[k] = categorical_from_W(W, unif(rng));
        return out;
    }
    const arma::vec cum = arma::cumsum(W);
    const double md = static_cast<double>(m);
    double u0 = 0.0;
    if (scheme == pg_resampling_scheme::SYSTEMATIC) u0 = unif(rng) / md;
    arma::uword i = 0;
    for (std::size_t k = 0; k < m; ++k) {
        const double uk = (scheme == pg_resampling_scheme::SYSTEMATIC)
                            ? (u0 + static_cast<double>(k) / md)
                            : ((static_cast<double>(k) + unif(rng)) / md);
        while (i + 1 < N && cum[i] < uk) ++i;
        out[k] = i;
    }
    return out;
}

}  // namespace detail_particle_gibbs

// ============================================================================
//  Config
// ============================================================================

struct particle_gibbs_block_config {
    /// Unique block name; also the shared_data key the sampled path is published
    /// under.
    std::string name = "x";

    /// Number of time steps T (> 0).
    int T = 0;

    /// Dimension d of each state x_t (>= 1). 1 for scalar SV; >1 for vector states.
    int state_dim = 1;

    /// Number of particles N (>= 2: the reference plus at least one free
    /// particle). Invariance holds for any N >= 2; this is purely a mixing/cost
    /// knob. Tunable post-construction by rebuilding (no metric state to carry).
    int n_particles = 64;

    /// Ancestor sampling (PGAS). Breaks path degeneracy; far better mixing on
    /// long series. Requires transition_logpdf. Default on.
    bool ancestor_sampling = true;

    /// Resampling scheme used each sweep.
    pg_resampling_scheme resampling = pg_resampling_scheme::SYSTEMATIC;

    /// Optional warm-start reference path (length T*state_dim). If empty, the
    /// first sweep is seeded by one unconditional particle filter.
    arma::vec initial_path;

    // ---- model-specific plug-ins (read theta & y_t from ctx themselves) ----

    /// Draw x_1 ~ mu_theta(.). Length-state_dim vector.
    std::function<arma::vec(const block_context& ctx, std::mt19937_64& rng)>
        init_sample;

    /// Draw x_t ~ f_theta(. | x_prev). t is the 0-based target time index.
    std::function<arma::vec(int t, const arma::vec& x_prev,
                            const block_context& ctx, std::mt19937_64& rng)>
        transition_sample;

    /// log g_theta(y_t | x_t). Non-finite return is treated as -inf (the
    /// particle gets zero weight). t is the 0-based time index.
    std::function<double(int t, const arma::vec& x_t, const block_context& ctx)>
        obs_loglik;

    /// log f_theta(x_t | x_prev). Used ONLY when ancestor_sampling = true.
    std::function<double(int t, const arma::vec& x_t, const arma::vec& x_prev,
                         const block_context& ctx)>
        transition_logpdf;
};

// ============================================================================
//  Block
// ============================================================================

class particle_gibbs_block : public block_sampler {
public:
    explicit particle_gibbs_block(particle_gibbs_block_config cfg)
        : cfg_(std::move(cfg))
    {
        if (cfg_.T <= 0)
            throw std::invalid_argument("particle_gibbs_block '" + cfg_.name +
                "': T must be > 0");
        if (cfg_.state_dim <= 0)
            throw std::invalid_argument("particle_gibbs_block '" + cfg_.name +
                "': state_dim must be > 0");
        if (cfg_.n_particles < 2)
            throw std::invalid_argument("particle_gibbs_block '" + cfg_.name +
                "': n_particles must be >= 2 (conditional SMC needs the "
                "reference plus at least one free particle)");
        if (!cfg_.init_sample)
            throw std::invalid_argument("particle_gibbs_block '" + cfg_.name +
                "': init_sample is required");
        if (!cfg_.transition_sample)
            throw std::invalid_argument("particle_gibbs_block '" + cfg_.name +
                "': transition_sample is required");
        if (!cfg_.obs_loglik)
            throw std::invalid_argument("particle_gibbs_block '" + cfg_.name +
                "': obs_loglik is required");
        if (cfg_.ancestor_sampling && !cfg_.transition_logpdf)
            throw std::invalid_argument("particle_gibbs_block '" + cfg_.name +
                "': transition_logpdf is required when ancestor_sampling is true "
                "(PGAS)");

        path_len_ = static_cast<std::size_t>(cfg_.T) *
                    static_cast<std::size_t>(cfg_.state_dim);
        ref_.set_size(path_len_);
        if (cfg_.initial_path.n_elem == path_len_) {
            ref_ = cfg_.initial_path;
            has_reference_ = true;
        } else if (cfg_.initial_path.n_elem == 0) {
            ref_.zeros();
            has_reference_ = false;
        } else {
            throw std::invalid_argument("particle_gibbs_block '" + cfg_.name +
                "': initial_path length must be T*state_dim or 0");
        }
    }

    // ---- block_sampler interface --------------------------------------

    void set_context(const block_context& ctx) override { context_ = ctx; }

    void step(std::mt19937_64& rng) override {
        // First sweep without a reference runs an UNCONDITIONAL particle filter
        // to seed it; every later sweep runs conditional SMC on the reference.
        arma::vec new_path = run_csmc_(rng, /*conditional=*/has_reference_);
        ref_ = std::move(new_path);
        has_reference_ = true;
        if (keep_history_) history_buf_.push_back(ref_);
    }

    const arma::vec& current() const override { return ref_; }

    void set_current(const arma::vec& path) override {
        if (path.n_elem != path_len_)
            throw std::invalid_argument("particle_gibbs_block '" + cfg_.name +
                "': set_current length must equal T*state_dim");
        ref_ = path;
        has_reference_ = true;
    }

    const std::string& name() const noexcept override { return cfg_.name; }
    std::size_t dim() const noexcept override { return path_len_; }

    state_map current_named_outputs() const override {
        return { { cfg_.name, ref_ } };
    }

    history_map get_history() const override {
        return detail::make_history_map(cfg_.name, history_buf_, ref_);
    }
    std::size_t history_size() const noexcept override {
        return history_buf_.empty() ? 1 : history_buf_.size();
    }
    void clear_history() override { history_buf_.clear(); }

private:
    /// state at time t extracted from a flat path vector.
    arma::vec state_at_(const arma::vec& path, std::size_t t) const {
        const std::size_t d = static_cast<std::size_t>(cfg_.state_dim);
        return path.subvec(t * d, (t + 1) * d - 1);
    }

    double obs_logw_(int t, const arma::vec& x) const {
        const double lw = cfg_.obs_loglik(t, x, context_);
        return std::isnan(lw) ? -std::numeric_limits<double>::infinity() : lw;
    }

    /// One conditional-SMC sweep; returns the new path (flat length T*state_dim).
    /// If `conditional`, slot 0 is pinned to the reference ref_ and (if PGAS) its
    /// ancestor is resampled; otherwise an ordinary unconditional PF runs.
    arma::vec run_csmc_(std::mt19937_64& rng, bool conditional) {
        namespace dpg = detail_particle_gibbs;
        const std::size_t T = static_cast<std::size_t>(cfg_.T);
        const std::size_t d = static_cast<std::size_t>(cfg_.state_dim);
        const std::size_t N = static_cast<std::size_t>(cfg_.n_particles);
        const std::size_t first_free = conditional ? 1u : 0u;
        std::uniform_real_distribution<double> unif(0.0, 1.0);

        std::vector<arma::mat> X(T);                  // X[t]: d x N particles
        std::vector<std::vector<arma::uword>> A(T);   // A[t][i]: ancestor of i at t
        arma::vec logw(N), W(N);

        // ---- t = 0 ----
        X[0].set_size(d, N);
        if (conditional) X[0].col(0) = state_at_(ref_, 0);
        for (std::size_t i = first_free; i < N; ++i)
            X[0].col(i) = cfg_.init_sample(context_, rng);
        for (std::size_t i = 0; i < N; ++i)
            logw[i] = obs_logw_(0, X[0].col(i));
        W = dpg::normalize_log_weights(logw);

        // ---- t = 1 .. T-1 ----
        for (std::size_t t = 1; t < T; ++t) {
            A[t].assign(N, 0);
            // free-particle ancestors from W_{t-1}
            std::vector<arma::uword> anc =
                dpg::resample(W, N - first_free, cfg_.resampling, rng);

            arma::vec ref_t;
            if (conditional) {
                ref_t = state_at_(ref_, t);
                if (cfg_.ancestor_sampling) {
                    // PGAS reference ancestor ~ W_{t-1}^j f(ref_t | x_{t-1}^j)
                    arma::vec log_aw(N);
                    for (std::size_t j = 0; j < N; ++j) {
                        const double lw = (W[j] > 0.0)
                            ? std::log(W[j])
                            : -std::numeric_limits<double>::infinity();
                        double lf = cfg_.transition_logpdf(
                            static_cast<int>(t), ref_t, X[t - 1].col(j), context_);
                        if (std::isnan(lf))
                            lf = -std::numeric_limits<double>::infinity();
                        log_aw[j] = lw + lf;
                    }
                    const arma::vec aW = dpg::normalize_log_weights(log_aw);
                    A[t][0] = dpg::categorical_from_W(aW, unif(rng));
                } else {
                    A[t][0] = 0;  // reference keeps its own lineage
                }
                for (std::size_t i = 1; i < N; ++i) A[t][i] = anc[i - 1];
            } else {
                for (std::size_t i = 0; i < N; ++i) A[t][i] = anc[i];
            }

            // propagate
            X[t].set_size(d, N);
            if (conditional) X[t].col(0) = ref_t;
            for (std::size_t i = first_free; i < N; ++i)
                X[t].col(i) = cfg_.transition_sample(
                    static_cast<int>(t), X[t - 1].col(A[t][i]), context_, rng);

            // weights (bootstrap: incremental weight = observation density)
            for (std::size_t i = 0; i < N; ++i)
                logw[i] = obs_logw_(static_cast<int>(t), X[t].col(i));
            W = dpg::normalize_log_weights(logw);
        }

        // ---- terminal index + ancestral trace-back ----
        arma::uword b = dpg::categorical_from_W(W, unif(rng));
        arma::vec path(path_len_);
        for (std::size_t tt = T; tt-- > 0; ) {
            path.subvec(tt * d, (tt + 1) * d - 1) = X[tt].col(b);
            if (tt > 0) b = A[tt][b];
        }
        return path;
    }

    particle_gibbs_block_config cfg_;
    std::size_t                 path_len_ = 0;
    arma::vec                   ref_;
    bool                        has_reference_ = false;
    block_context               context_;
    std::vector<arma::vec>      history_buf_;
};

}  // namespace AI4BayesCode

#endif  // AI4BAYESCODE_PARTICLE_GIBBS_BLOCK_HPP
