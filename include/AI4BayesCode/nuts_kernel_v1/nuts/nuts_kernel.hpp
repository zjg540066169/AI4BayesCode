// Copyright (C) 2026 AI4BayesCode contributors. GPL-3.0-or-later. See LICENSE at the repository root.
// nuts_kernel.hpp — NUTS kernel main entry, stateful single-chain
//
// Integrates Algorithm 6 (multinomial) tree building, dual-averaging
// step-size adaptation, and continuous-online Welford diagonal mass
// matrix adaptation. Designed to drop into AI4BayesCode block_sampler
// interface (Tier B nuts_block / joint_nuts_block).
//
// Stateful semantics — preserves the §1 invariant:
//   - chain state (theta_current, grad_current, lp_current) advanced by step()
//   - internal RNG advanced by step() / readapt()
//   - readapt(n, reset, adapt_mass) snapshots chain state, runs n
//     internal adaptation iters, restores chain state
//   - mass matrix is FROZEN during sampling; updated only during warmup
//     or readapt(adapt_mass=true) windows
//
// API summary:
//   class NutsKernel {
//     NutsKernel(dim, log_density, cfg, rng_seed)
//     void step()                              -> advance chain one transition
//     const arma::vec& current_position()      -> read theta
//     void set_current_position(theta)          -> seed/override
//     double current_log_density()             -> read lp
//     void readapt(n, reset, adapt_mass)        -> re-tune step + (opt) mass
//     SnapshotState snapshot()                  -> bitwise save for readapt
//     void restore(snapshot)                    -> restore (caller's responsibility)
//     // Configuration & diagnostics:
//     double step_size()                        -> current eps
//     const arma::vec& mass_diagonal()          -> current diagonal mass
//     ChainStats stats()                        -> divergences, tree depths, etc.
//   };

#pragma once

#include "leapfrog.hpp"
#include "tree_building.hpp"
#include "../adapt/dual_averaging.hpp"
#include "../arma_port/online_moments.hpp"

#include <armadillo>
#include <random>
#include <functional>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace AI4BayesCode {

// Public configuration for v1 kernel.
struct NutsKernelConfig {
    // NUTS parameters
    std::size_t max_tree_depth = 10;
    double max_energy_error = 1000.0;  // divergence threshold (delta_max)

    // Step-size adaptation
    internal::DualAveragingConfig dual_avg;

    // Mass matrix adaptation
    bool use_dense_metric = false;
    double mass_discount_factor = 0.99;       // nutpie-style decay
    double mass_init_count = 1.1;             // walnuts default
    double mass_additive_smoothing = 1e-5;    // walnuts default

    // Outer-combine policy: how the streaming sampler picks between the
    // current accumulated span and the newly-built subtree at each
    // doubling. MULTINOMIAL (default) matches Stan / PyMC and is robust
    // across single-block standalone NUTS AND multi-block Gibbs
    // composites. METROPOLIS_BIASED (Betancourt 2017 §3) gives ~4× more
    // ESS on hierarchical joint targets (M2 NCR) but regressed sim1 blr
    // R-hat 1.0003 → 1.10. Opt in only when the block is known to be
    // joint NUTS on a single coherent target.
    enum class OuterCombine { MULTINOMIAL, METROPOLIS_BIASED };
    OuterCombine outer_combine = OuterCombine::MULTINOMIAL;

    // Warmup schedule (Stan-style 5-window in Phase II)
    std::size_t warmup_phase1_iters = 75;     // step-size only
    std::vector<std::size_t> warmup_phase2_windows = {25, 50, 100, 200, 500};
    std::size_t warmup_phase3_iters = 50;     // step-size only with frozen mass

    // NOTE: a `persistent_step_adapt` mode (continue dual-averaging after
    // warmup to track a drifting per-block conditional) was prototyped here
    // 2026-06-14 and REMOVED: it death-spirals coupled 1-D Gibbs blocks
    // (every block chasing siblings that are themselves adapting -> step
    // collapses -> arma11 froze, cross-impl R-hat 2.23). The spec's premise
    // that mcmclib relies on persistent adaptation was incorrect (mcmclib's
    // shipped config freezes the step after a one-shot warmup, like the
    // joint-NUTS references). See FINDINGS_2026-06-14.md.

    // Use a fixed IDENTITY mass matrix throughout (opt-in). When true, the
    // kernel does NOT observe Welford moments or install a diagonal mass
    // during warmup — the metric stays at identity, matching mcmclib's
    // nuts_block (which fixes its precond matrix at construction and never
    // adapts it).
    //
    // Why this matters for per-block Gibbs funnels: the existing default
    // installs a diagonal mass = 1/Var(theta) estimated DURING the
    // first-call warmup, when the sibling blocks are FROZEN at their init
    // values. For a centered hierarchical block whose component starts at
    // spread-out data (eight_schools theta_j init = y_j) the warmup
    // variance is huge and unrepresentative (Var ~ 45), so the installed
    // mass is tiny (~0.02) and the EFFECTIVE step eps*sqrt(Var) blows up
    // (~21) even though the raw eps is fine — the chain then freezes once
    // the real conditional tightens. Instrumented comparison (2026-06-14)
    // on eight_schools_centered:
    //     block      our eff_step (adapted mass)   mcmclib eff_step (identity)
    //     theta(8)   21.2  (eps=pi, Var=45)         5.70  (eps=5.70)
    //     mu(1)       7.97                          4.50
    // mcmclib samples it correctly (cross-impl R-hat 1.007) with identity
    // mass; our raw eps (pi) is actually SMALLER than mcmclib's (5.70), so
    // with identity mass our eff_step would be ~pi < 5.70 and should also
    // sample correctly. The diagonal-mass amplification is the entire bug.
    bool use_identity_metric = false;

    // Upper bound on the per-dimension variance used to build the diagonal
    // mass matrix (default +inf = no bound = current behavior). The
    // installed mass is M = 1/Var; the effective natural-scale step is
    // eps*sqrt(Var). When Var > 1 the metric AMPLIFIES the step beyond what
    // identity mass would give — which is exactly the per-block funnel
    // freeze: a frozen-sibling warmup produces an unrepresentative large
    // Var (eight_schools theta Var ~ 45), the mass amplifies eff_step to
    // ~21, and the chain freezes once the real conditional tightens.
    // Bounding Var at 1.0 caps the metric at identity from above (the mass
    // may only TIGHTEN the step, never loosen it past identity), which
    // removes the amplification for funnel blocks while still letting tight
    // conditionals (arma11, Var << 1) keep their beneficial rescaling.
    //
    // OPT-IN (default +inf = legacy uncapped behavior, bit-identical to
    // baseline). Set to 1.0 to enable the funnel-freeze fix.
    //
    // Validated 2026-06-14 (full 115-model sim1 sweep). Setting this to 1.0
    // fixes the MASS-AMPLIFICATION funnel subclass — blocks whose
    // frozen-sibling first-call warmup produces an unrepresentative large
    // Var (eight_schools_centered theta Var~45) that the diagonal mass then
    // amplifies into a step that freezes the chain. Cap=1.0 fixed 6 sim1
    // models (eight_schools_centered 2.23->1.00, dogs, kidscore_mom_work,
    // kidscore_interaction_z, seeds_centered, logearn_logheight_male) and is
    // surgical for well-conditioned blocks (Var<<1, untouched). It is NOT a
    // complete funnel fix: ~19 hierarchical funnels (radon_*, seeds_model,
    // irt_2pl, Mh, ...) freeze from a SEPARATE mechanism — seed-dependent
    // funnel NON-TRAVERSAL with reasonable small steps (mcmclib's slice
    // sampler handles it; our multinomial does not) — which this cap does
    // NOT address, and 1 borderline model regressed (radon_variable_
    // intercept_noncentered 1.042->1.072). Therefore DEFAULT OFF until the
    // deeper funnel-traversal issue is resolved. See
    // PERSISTENT_ADAPT_FIX/FINDINGS_2026-06-14.md.
    double max_mass_variance = std::numeric_limits<double>::infinity();
};

// Chain diagnostics tracked per step.
//
// divergences: true energy divergences only (lp/H non-finite or |ΔH|
//              exceeded delta_max). These indicate sampler pathology.
// subtree_uturn_stops: subtree terminated by Mahalanobis U-turn before
//              completing. Normal NUTS behavior, not a pathology.
struct ChainStats {
    std::size_t total_steps = 0;
    std::size_t divergences = 0;
    std::size_t subtree_uturn_stops = 0;
    std::size_t tree_depth_max_hit = 0;
    double last_tree_depth_d = 0.0;
    double last_accept_prob = 0.0;
    double last_energy_error = 0.0;
};

// Functor type for the user-provided log-density. Signature:
//   double log_density(const arma::vec& theta, arma::vec* grad_out)
//   - returns log p(theta) (up to constant)
//   - if grad_out != nullptr, writes ∇log p(theta) there
//
// CRITICAL: this is the same convention as mcmclib's
// target_log_kernel — backward-compatible from the user's lambda
// perspective.
using LogDensityFn = std::function<double(const arma::vec&, arma::vec*)>;

class NutsKernel {
public:
    NutsKernel(std::size_t dim,
                LogDensityFn log_density,
                NutsKernelConfig cfg,
                std::uint64_t rng_seed)
        : dim_(dim), log_density_(std::move(log_density)),
          cfg_(std::move(cfg)),
          rng_(rng_seed),
          theta_current_(arma::vec(dim, arma::fill::zeros)),
          grad_current_(arma::vec(dim, arma::fill::zeros)),
          lp_current_(0.0),
          step_adapter_(cfg_.dual_avg),
          mass_diag_(arma::vec(dim, arma::fill::ones)),
          mass_inv_diag_(arma::vec(dim, arma::fill::ones)),
          welford_(cfg_.mass_init_count,
                   arma::vec(dim, arma::fill::zeros),
                   arma::vec(dim, arma::fill::ones)),
          warmup_remaining_(cfg_.warmup_phase1_iters
                            + sum_window_iters()
                            + cfg_.warmup_phase3_iters),
          step_size_(1.0) {
        if (dim == 0) {
            throw std::invalid_argument(
                "NutsKernel: dim must be positive");
        }
    }

    // Seed the chain state. Must be called before first step() unless
    // the default zero-initialization is intended. After seeding, this
    // ALSO finds a reasonable initial step size via Hoffman 2014
    // Algorithm 4 (double/halve until accept prob ~ 0.5).
    void set_current_position(const arma::vec& theta) {
        if (theta.n_elem != dim_) {
            throw std::invalid_argument(
                "NutsKernel::set_current_position: size mismatch");
        }
        theta_current_ = theta;
        lp_current_ = log_density_(theta_current_, &grad_current_);
        if (!std::isfinite(lp_current_)) {
            throw std::runtime_error(
                "NutsKernel: initial log_density is non-finite at given position");
        }
        // Find initial step size (Hoffman 2014 Algorithm 4).
        // Sample one momentum, take one leapfrog step at eps=1, check
        // energy ratio. Double or halve eps until accept prob ~ 0.5.
        step_size_ = find_initial_step_size_();
        // Initialize the step adapter with the found eps so the first
        // warmup step uses the right starting point (NOT 1.0).
        step_adapter_.init(step_size_);
    }

    const arma::vec& current_position() const noexcept { return theta_current_; }
    double current_log_density() const noexcept { return lp_current_; }
    double step_size() const noexcept { return step_size_; }
    const arma::vec& mass_diagonal() const noexcept { return mass_diag_; }
    const ChainStats& stats() const noexcept { return stats_; }
    // Diagnostic: the dual-averaging running deviation (target - actual
    // accept). Near 0 means adaptation converged to the target rate.
    double adapt_h_bar() const noexcept { return step_adapter_.h_bar(); }
    double last_accept_prob() const noexcept { return stats_.last_accept_prob; }

    // Advance the chain by one NUTS transition. Handles warmup phase
    // adaptation internally (no separate burnin / sampling caller logic).
    void step() {
        const bool in_warmup = warmup_remaining_ > 0;
        const bool in_phase2 = in_warmup
                              && warmup_remaining_ > cfg_.warmup_phase3_iters
                              && warmup_remaining_ <= cfg_.warmup_phase1_iters + sum_window_iters();
        const bool in_phase1 = in_warmup
                              && warmup_remaining_ > cfg_.warmup_phase1_iters + sum_window_iters() - 0;  // earliest part
        (void) in_phase1;  // currently we use phase1 implicitly via step_adapter init

        // Pick effective step size for THIS step:
        //   - during warmup: raw eps (responsive)
        //   - after warmup: eps_bar (Polyak-averaged)
        step_size_ = in_warmup ? step_adapter_.current_step_size()
                                : step_adapter_.sampling_step_size();
        if (!std::isfinite(step_size_) || step_size_ <= 0.0) {
            step_size_ = 1e-3;  // conservative fallback
        }

        // Sample momentum from N(0, M) (diagonal metric).
        arma::vec p_init = internal::sample_momentum_diagonal(mass_diag_, rng_);
        internal::DiagonalMetric M{mass_inv_diag_};

        // Initial energy.
        double K_init = internal::kinetic_energy(p_init, M);
        double H_init = -lp_current_ + K_init;

        // Build the trajectory by progressive doubling.
        // Initial leaf is the current state — alpha at init = 1 (vs itself).
        // H_init IS the global trajectory H0; pass it as both the leaf
        // energy and the baseline so the initial leaf has dH=0, alpha=1,
        // and is never spuriously flagged divergent.
        internal::SpanW current = internal::SpanW::from_single(
            theta_current_, p_init, grad_current_, lp_current_,
            H_init, H_init, cfg_.max_energy_error);

        // NOTE on the acceptance statistic baseline: Hoffman 2014 Alg 6,
        // Stan, and mcmclib all EXCLUDE the trajectory start point from the
        // dual-averaging accept average (it is not a BuildTree leaf). We
        // tried zeroing the start leaf's alpha/n_alpha to match that letter
        // (2026-06-11) but it REGRESSED arma11's 1-D blocks 4/20 → 9/20
        // fail (rhat 1.88 → 2.23), breaking the exact mcmclib parity that
        // fixes 1+2 (divergence threshold + global-H0 alpha) achieve. For a
        // deep tree the start leaf is 1/2^depth — negligible — so excluding
        // it is correct and neutral there (clean 40-D Gaussian: identical
        // eps/recovery either way). But for the very SHORT trees our
        // multinomial sampler builds on tight 1-D conditionals, including
        // the start leaf empirically stabilizes the accept stat and the
        // resulting step size. We therefore KEEP the start leaf counted
        // (n_alpha=1, alpha=1 from from_single). The capture-recapture
        // funnel models (Mh/Mth) are a separate Gibbs-funnel issue (see
        // project_nutskernel_bugs_2026-06-11.md) that excluding the start
        // leaf does NOT fix.

        std::size_t depth = 0;
        bool stop = false;
        std::uniform_int_distribution<int> coin(0, 1);
        std::uniform_real_distribution<double> uniform(0.0, 1.0);

        while (!stop && depth < cfg_.max_tree_depth) {
            int direction = coin(rng_) == 0 ? -1 : 1;
            // build_subtree takes its seed args by const& -- no need to
            // copy `current`'s endpoint into local seed_* vecs first.
            // After the call, `current` is still untouched (we move it
            // into combine_subtrees below, but only AFTER build_subtree
            // has returned).
            const bool fwd = (direction > 0);
            const arma::vec& seed_theta = fwd ? current.theta_right : current.theta_left;
            const arma::vec& seed_p     = fwd ? current.p_right     : current.p_left;
            const arma::vec& seed_grad  = fwd ? current.grad_right  : current.grad_left;
            const double     seed_lp    = fwd ? current.lp_right    : current.lp_left;
            internal::SpanW new_span = internal::build_subtree(
                seed_theta, seed_p, seed_grad, seed_lp,
                direction, depth, step_size_, M,
                H_init, cfg_.max_energy_error, log_density_, rng_);
            if (new_span.stop) {
                // Subtree stopped. Classify it for reporting:
                // - energy divergence: lp / H non-finite anywhere in subtree
                //   → sampler pathology (e.g. step size too large)
                // - U-turn termination: normal NUTS geometric stop
                if (new_span.energy_divergent) {
                    ++stats_.divergences;
                } else {
                    ++stats_.subtree_uturn_stops;
                }
                // CRITICAL for dual averaging robustness: even when the
                // subtree is rejected, fold its leaf-level alpha into the
                // accumulator so the dual-averaging signal reflects the
                // failed leapfrog. Without this, every divergent step
                // collapses accept_p back to the initial leaf's alpha=1
                // (since current was seeded from from_single(self) where
                // dH=0 ⇒ alpha=1). Dual averaging then misreads "100%
                // accept" and DOUBLES eps each step — observed on arma11
                // (BridgeStan) where eps grew to 7e49 before the first
                // sane backoff. Folding new_span's alpha — even after
                // a divergence — gives the adapter a true picture of
                // accept rate and brings eps back down.
                current.alpha_sum += new_span.alpha_sum;
                current.n_alpha   += new_span.n_alpha;
                break;
            }
            // OUTER combine: MULTINOMIAL. Matches Stan / PyMC default
            // (Hoffman 2014 Algorithm 3 weighting). Empirical sweep over
            // sim1 models (blr, GLM_Poisson, eight_schools) showed
            // METROPOLIS_BIASED at this layer regressed R-hat on
            // moderate-dim composites: blr 1.0003 → 1.10, GLM_Poisson
            // 1.001 → 1.61. The BIASED policy is correct for the inner
            // recursion (still in build_subtree default) but at the
            // outer layer it overweights the accumulator span and
            // induces sticky regions on multi-block Gibbs sweeps.
            internal::SpanW combined =
                internal::combine_subtrees(std::move(current),
                                            std::move(new_span),
                                            direction, M, rng_,
                                            internal::CombineUpdate::MULTINOMIAL);
            current = std::move(combined);
            ++depth;
            stop = current.stop;
        }
        if (depth >= cfg_.max_tree_depth) {
            ++stats_.tree_depth_max_hit;
        }
        stats_.last_tree_depth_d = static_cast<double>(depth);

        // Accept the proposal sampled by the streaming multinomial.
        // SpanW now threads grad_proposal alongside theta_proposal, so
        // we can adopt the cached grad directly — eliminating the one
        // redundant log_density(theta_current, &grad) call that v1
        // originally performed after each accept. Saves ~1% wall on
        // typical sim1 models (1 extra grad call per outer step on top
        // of the ~2^depth leapfrog grads, which were already cached).
        theta_current_ = std::move(current.theta_proposal);
        grad_current_  = std::move(current.grad_proposal);
        lp_current_    = current.lp_proposal;

        // Compute average accept probability over the trajectory's leaves
        // for dual averaging update (Hoffman 2014 alpha_bar = alpha_sum / n_alpha).
        double accept_p = (current.n_alpha > 0)
                          ? current.alpha_sum / static_cast<double>(current.n_alpha)
                          : 0.0;
        if (accept_p > 1.0) accept_p = 1.0;
        if (accept_p < 0.0) accept_p = 0.0;
        stats_.last_accept_prob = accept_p;
        stats_.last_energy_error = std::abs(-current.lp_proposal - H_init);

        // Adaptation updates (warmup only). Step adapter is already
        // initialized in set_current_position() with the found eps.
        // NOTE: mass adaptation runs in EVERY warmup configuration —
        // even when warmup_phase2_windows is empty, the in_phase2 check
        // remains true (because warmup_remaining_ stays > phase3_iters
        // throughout phase1). Sim1 results (M0/Mt/Mb) showed this is the
        // correct behavior: Welford-observed marginal scaling beats pure
        // identity for these models. Earlier attempt to gate on empty
        // phase2_windows regressed M0 R-hat 1.014 → 1.56.
        if (in_warmup) {
            step_adapter_.update(accept_p);

            // Mass adaptation. Skipped entirely when use_identity_metric is
            // set — the metric then stays at its identity initialization,
            // matching mcmclib's fixed-precond behavior and avoiding the
            // frozen-sibling-warmup variance amplification (see
            // NutsKernelConfig::use_identity_metric).
            if (in_phase2 && !cfg_.use_identity_metric) {
                welford_.discount_observe(cfg_.mass_discount_factor,
                                            theta_current_);
            }
            --warmup_remaining_;
            if (warmup_remaining_ == cfg_.warmup_phase3_iters
                && !cfg_.use_identity_metric) {
                install_welford_mass();
            }
        }
        // (No post-warmup step adaptation: the step freezes at eps_bar after
        // warmup, matching Stan / nuts-rs / nutpie / walnuts / mcmclib. A
        // persistent post-warmup adaptation mode was prototyped and removed
        // — see the NutsKernelConfig note above.)
        ++stats_.total_steps;
    }

    // readapt_NUTS — re-tune step size and (optionally) mass matrix at
    // current state. Snapshots and restores chain state.
    //
    // reset=false: continue dual-averaging from current state.
    // reset=true: re-init dual-averaging from scratch (using current eps_bar as seed).
    // adapt_mass=true: also run continuous Welford on mass matrix
    //                  during the n iterations (then freeze).
    void readapt(std::size_t n, bool reset, bool adapt_mass) {
        // Snapshot chain state.
        arma::vec theta_snap = theta_current_;
        arma::vec grad_snap = grad_current_;
        double lp_snap = lp_current_;
        ChainStats stats_snap = stats_;
        std::size_t warmup_snap = warmup_remaining_;

        if (reset) {
            step_adapter_.reset();
            step_adapter_.init(step_adapter_.sampling_step_size() > 0
                               ? step_adapter_.sampling_step_size()
                               : 1.0);
            if (adapt_mass) {
                welford_ = internal::OnlineMoments(
                    cfg_.mass_init_count,
                    arma::vec(dim_, arma::fill::zeros),
                    arma::vec(dim_, arma::fill::ones));
            }
        }
        // Set warmup_remaining_ so step() enters warmup logic for n iters.
        warmup_remaining_ = n;
        for (std::size_t i = 0; i < n; ++i) {
            step();
        }
        if (adapt_mass) {
            install_welford_mass();
        }
        // Restore chain state (kernel state — eps, mass — is kept).
        theta_current_ = theta_snap;
        grad_current_ = grad_snap;
        lp_current_ = lp_snap;
        // stats_: keep updated stats (count adaptation iters)? Per v1
        // plan §4.6 / §13 NUTS-family, the ghost iterations are NOT
        // recorded in history but ARE counted in stats. Keep the diff.
        // (caller may diff to inspect adapt iteration accept rates).
        (void) stats_snap;
        warmup_remaining_ = warmup_snap;
    }

private:
    // Hoffman & Gelman 2014 Algorithm 4: find a reasonable initial step
    // size by leapfrogging once at eps=1, then doubling or halving eps
    // until the energy-change probability crosses 0.5.
    double find_initial_step_size_() {
        double eps = 1.0;
        arma::vec p_init = internal::sample_momentum_diagonal(mass_diag_, rng_);
        internal::DiagonalMetric M{mass_inv_diag_};
        double H_init = -lp_current_ + internal::kinetic_energy(p_init, M);

        auto one_step_log_accept = [&](double eps_try) -> double {
            auto step = internal::leapfrog_step(
                theta_current_, p_init, grad_current_, eps_try, M, log_density_);
            if (!std::isfinite(step.lp_new)) {
                return -std::numeric_limits<double>::infinity();
            }
            double K_new = internal::kinetic_energy(step.p_new, M);
            double H_new = -step.lp_new + K_new;
            // log accept = -ΔH (clamped to ≤ 0)
            double dH = H_new - H_init;
            return -dH;
        };

        double log_accept = one_step_log_accept(eps);
        // If the first leapfrog already diverges (energy NaN), the target
        // gradient is too stiff at the initial position for eps=1.
        // Force-halve regardless of "direction" until we get a finite energy.
        // Without this, the loop below (which only halves when direction<0)
        // can get stuck doubling on -inf log_accept. Caught on arma11 via
        // BridgeStan: eps grew to 7×10^49 before any sane backoff.
        if (!std::isfinite(log_accept)) {
            for (std::size_t i = 0; i < 30; ++i) {
                eps *= 0.5;
                if (eps < 1e-6) { eps = 1e-3; break; }
                log_accept = one_step_log_accept(eps);
                if (std::isfinite(log_accept)) break;
            }
        }
        // Decide direction: if accept prob > 0.5, eps too small → double;
        //                   if accept prob < 0.5, eps too big → halve.
        double direction = (log_accept > std::log(0.5)) ? 1.0 : -1.0;
        const std::size_t max_iter = 50;          // was 100
        const double eps_upper_cap = 5.0;          // was 1e10
        const double eps_lower_cap = 1e-6;         // was 1e-10
        for (std::size_t i = 0; i < max_iter; ++i) {
            if ((direction > 0 && log_accept <= std::log(0.5))
                || (direction < 0 && log_accept >= std::log(0.5))) {
                break;
            }
            eps *= (direction > 0) ? 2.0 : 0.5;
            if (eps < eps_lower_cap || eps > eps_upper_cap) break;
            log_accept = one_step_log_accept(eps);
            // If a doubling-attempt drove energy non-finite, that's a
            // sign we overshot — back off to the last good eps.
            if (direction > 0 && !std::isfinite(log_accept)) {
                eps *= 0.5;
                break;
            }
        }
        // Guard against pathological cases.
        if (eps < 1e-6) eps = 1e-3;
        if (eps > eps_upper_cap) eps = eps_upper_cap;
        if (!std::isfinite(eps) || eps <= 0.0) eps = 0.1;
        return eps;
    }

    std::size_t sum_window_iters() const {
        std::size_t s = 0;
        for (auto w : cfg_.warmup_phase2_windows) s += w;
        return s;
    }

    void install_welford_mass() {
        // Stan/walnuts convention: M = Σ⁻¹ (precision). The mass matrix
        // is the *inverse* of the posterior covariance, so that:
        //   - p ~ N(0, M) has standardized scale (variance = precision)
        //   - leapfrog drift eps · M⁻¹ · p has Var = eps² · Σ → matches
        //     posterior covariance per step, enabling efficient mixing.
        // Earlier draft mistakenly set mass = variance directly; this
        // makes M⁻¹ = 1/var, which inverts the geometry and produces
        // tiny position steps (≈ eps/σ instead of eps·σ). The Gaussian
        // tests didn't catch this because Σ = I → var = 1/var = 1.
        // Verified against walnuts adaptive_walnuts.hpp: inv_mass_estimate
        // returns the variance of draws (= Σ), mass = 1/inv_mass.
        arma::vec var = welford_.variance() + cfg_.mass_additive_smoothing;
        // Upper-bound the variance at cfg_.max_mass_variance (default +inf).
        // Capping at 1.0 prevents the metric from amplifying the effective
        // step beyond identity — the per-block funnel-freeze fix. The lower
        // bound 1e-12 guards against degenerate (zero-variance) dimensions.
        mass_inv_diag_ = arma::clamp(var, 1e-12, cfg_.max_mass_variance);
        mass_diag_ = 1.0 / mass_inv_diag_;
    }

    std::size_t dim_;
    LogDensityFn log_density_;
    NutsKernelConfig cfg_;

    std::mt19937_64 rng_;

    arma::vec theta_current_;
    arma::vec grad_current_;
    double   lp_current_;

    internal::DualAveragingAdapter step_adapter_;

    arma::vec mass_diag_;
    arma::vec mass_inv_diag_;
    internal::OnlineMoments welford_;

    std::size_t warmup_remaining_;
    double step_size_;

    ChainStats stats_;
};

}  // namespace AI4BayesCode
