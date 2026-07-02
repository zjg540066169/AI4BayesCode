// Copyright (C) 2026 AI4BayesCode contributors. GPL-3.0-or-later. See LICENSE at the repository root.
// tree_building.hpp — NUTS Algorithm 6 (Hoffman 2014, multinomial sampling)
//
// Streaming tree-building per walnuts SpanW pattern:
//   - Only store left/right ENDPOINTS (theta, p, grad)
//   - Running multinomial-weighted proposal
//   - Running log-sum-of-weights for biased progressive sampling
//   - O(dim) memory per leapfrog step, NOT O(dim * 2^depth)
//
// Reference: walnuts/include/walnuts/walnuts.hpp combine() + build_tree*

#pragma once

#include "leapfrog.hpp"
#include <armadillo>
#include <cmath>
#include <random>
#include <utility>

namespace AI4BayesCode {
namespace internal {

// SpanW — a subtree representation. Holds endpoints, proposal, and
// log-weight statistics. Used by combine() and build_tree().
struct SpanW {
    // Trajectory endpoints
    arma::vec theta_left;
    arma::vec p_left;
    arma::vec grad_left;
    double   lp_left;

    arma::vec theta_right;
    arma::vec p_right;
    arma::vec grad_right;
    double   lp_right;

    // Sampled proposal from this subtree (multinomial weighted by
    // exp(lp - K)).
    //
    // grad_proposal is the gradient at theta_proposal, threaded through
    // combine_subtrees so the outer caller (NutsKernel::step()) can
    // adopt it directly as the new chain state's grad — eliminating the
    // one redundant log_density(theta_current, &grad) call that v1
    // originally performed after each accept.
    arma::vec theta_proposal;
    arma::vec grad_proposal;
    double   lp_proposal;

    // Log of sum of exp(lp - K) over all states in subtree
    double   log_sum_weights;

    // Number of leapfrog states in the subtree (2^depth)
    std::size_t n_states;

    // Hoffman 2014 alpha/n_alpha for dual averaging acceptance tracking.
    // alpha_sum = sum over leaves of min(1, exp(-ΔH_l)); n_alpha = leaf count.
    // Avg accept = alpha_sum / n_alpha.
    double      alpha_sum;
    std::size_t n_alpha;

    // Stop flag: either a true energy divergence (NaN lp / energy) OR
    // a Mahalanobis U-turn detected within this subtree.
    // `energy_divergent` separates the true divergence case so the
    // caller can report it distinctly — U-turns are normal NUTS
    // termination, not a sampler pathology.
    bool stop = false;
    bool energy_divergent = false;

    // Construct from a single leapfrog state (subtree depth 0).
    //
    // CRITICAL (2026-06-11 bugfix): `H0` MUST be the GLOBAL trajectory
    // initial Hamiltonian (the energy of the chain's current state +
    // sampled momentum), NOT the per-leaf seed energy. Both the
    // acceptance statistic alpha AND the divergence test are defined
    // relative to the global trajectory start, per Hoffman & Gelman
    // 2014 Algorithm 6 (alpha = min{1, exp(L(θ')−½r'·r' − L(θ⁰)+½r⁰·r⁰)})
    // and Stan base_nuts.hpp:262 ((h - H0) > max_deltaH_ ⇒ divergent).
    //
    // The earlier code passed the per-leaf seed energy as the baseline,
    // which (a) made the dual-averaging target the single-step error
    // rather than the trajectory-start error, and (b) never enforced the
    // finite Δ_max divergence threshold at all (only non-finite energy
    // stopped the tree). On stiff geometries (capture-recapture Mh/Mth,
    // ARMA) this let the trajectory integrate through high-error regions
    // without stopping, breaking reversibility and producing badly
    // under-covering posteriors (Mh coverage 0.38).
    //
    // @param max_energy_error  Δ_max divergence threshold (default 1000,
    //   matching Stan/nutpie/nuts-rs). A leaf with H_proposal − H0 >
    //   max_energy_error is flagged divergent and stops tree expansion.
    static SpanW from_single(const arma::vec& theta, const arma::vec& p,
                              const arma::vec& grad, double lp,
                              double total_energy_proposal,
                              double H0,
                              double max_energy_error) {
        SpanW s;
        s.theta_left = s.theta_right = theta;
        s.p_left = s.p_right = p;
        s.grad_left = s.grad_right = grad;
        s.lp_left = s.lp_right = lp;
        s.theta_proposal = theta;
        s.grad_proposal = grad;
        s.lp_proposal = lp;
        // Multinomial leaf weight: Stan uses log w = H0 − h (≡ −ΔH).
        // The global H0 offset is constant across all leaves in a
        // trajectory, so it cancels in the relative multinomial
        // selection, but subtracting it keeps the log-weights near 0
        // for numerical stability (vs the previous −H_proposal, which
        // could be a large magnitude).
        const double dH = total_energy_proposal - H0;  // = h − H0
        s.log_sum_weights = -dH;                        // = H0 − h
        s.n_states = 1;
        // Per-leaf acceptance statistic for dual averaging:
        //   alpha = min(1, exp(H0 − h)) = min(1, exp(−ΔH))
        // relative to the GLOBAL trajectory start H0.
        s.alpha_sum = (dH <= 0.0) ? 1.0 : std::exp(-dH);
        if (!std::isfinite(s.alpha_sum)) s.alpha_sum = 0.0;
        s.n_alpha = 1;
        // Divergence: non-finite energy OR finite-but-excessive energy
        // error relative to the global start (Hoffman Δ_max / Stan
        // max_deltaH_). The one-sided test (h − H0 > Δ_max, not |·|)
        // matches Stan base_nuts.hpp:262 and nuts-rs (Euclidean path).
        s.energy_divergent =
            !std::isfinite(lp) || !std::isfinite(total_energy_proposal)
            || (dH > max_energy_error);
        s.stop = s.energy_divergent;
        return s;
    }
};

// log_sum_exp helper for numerically stable combine.
inline double log_sum_exp(double a, double b) {
    if (a == -std::numeric_limits<double>::infinity()) return b;
    if (b == -std::numeric_limits<double>::infinity()) return a;
    if (a >= b) {
        return a + std::log1p(std::exp(b - a));
    } else {
        return b + std::log1p(std::exp(a - b));
    }
}

// Mahalanobis U-turn check using the inverse mass matrix.
// Walnuts: scaled_diff = inv_mass .* (theta_R - theta_L); check
// dot(p_endpoint, scaled_diff) ≥ 0 at both endpoints.
//
// CRITICAL FIX (2026-06-08): without inv_mass scaling, U-turn check
// only works correctly when mass = identity. For adapted diagonal mass
// (logistic regression, etc.), the dim with larger variance dominates
// the un-scaled dot product, causing trajectory to terminate too early.
inline bool combined_no_uturn_diag(const arma::vec& delta,
                                    const arma::vec& p_left,
                                    const arma::vec& p_right,
                                    const arma::vec& inv_mass_diag) {
    arma::vec delta_scaled = inv_mass_diag % delta;  // Mahalanobis-scaled
    bool no_uturn_left  = arma::dot(p_left,  delta_scaled) >= 0.0;
    bool no_uturn_right = arma::dot(p_right, delta_scaled) >= 0.0;
    return no_uturn_left && no_uturn_right;
}

inline bool combined_no_uturn_dense(const arma::vec& delta,
                                     const arma::vec& p_left,
                                     const arma::vec& p_right,
                                     const arma::mat& inv_mass_dense) {
    arma::vec delta_scaled = inv_mass_dense * delta;
    bool no_uturn_left  = arma::dot(p_left,  delta_scaled) >= 0.0;
    bool no_uturn_right = arma::dot(p_right, delta_scaled) >= 0.0;
    return no_uturn_left && no_uturn_right;
}

// Combine policy: how to choose proposal between two adjacent subtrees.
//
// MULTINOMIAL (Barker): equal-weight, used INSIDE build_subtree recursion
//                       where both subtrees have the same depth (size).
//                       prob_select_new = weight_new / (weight_old + weight_new)
//
// METROPOLIS_BIASED:    bias toward NEW subtree, used in OUTER step() loop
//                       where span_accum has grown larger than new_span.
//                       prob_select_new = min(1, weight_new / weight_old)
//                       This is the Betancourt 2017 biased progressive sampling
//                       that ensures the new (farther) states get a fair chance.
enum class CombineUpdate { MULTINOMIAL, METROPOLIS_BIASED };

// Combine two subtrees into one, choosing proposal via biased
// progressive sampling (Betancourt 2017 refinement of Hoffman 2014).
//
// Direction: span_old is the "current" tree, span_new is the newly-built
// adjacent subtree. Caller ensures the left endpoint of one matches the
// right endpoint of the other (NOT enforced here for speed).
//
// extend_direction: +1 means new tree extends FORWARD from old, -1
// means BACKWARD.
//
// Metric: passed for the U-turn check (Mahalanobis scaled by inv_mass).
//
// update_policy: MULTINOMIAL (equal-weight, inner) or METROPOLIS_BIASED
//                (favor new, outer). Default MULTINOMIAL for backward compat.
template <typename Rng, typename Metric>
SpanW combine_subtrees(SpanW old_span, SpanW new_span,
                        int extend_direction, Metric M, Rng& rng,
                        CombineUpdate update_policy = CombineUpdate::MULTINOMIAL) {
    if (old_span.stop || new_span.stop) {
        old_span.stop = true;
        // Propagate energy-divergent flag so the caller can distinguish
        // U-turn termination from a true divergence.
        old_span.energy_divergent =
            old_span.energy_divergent || new_span.energy_divergent;
        return old_span;
    }

    SpanW combined;
    combined.n_states = old_span.n_states + new_span.n_states;
    combined.log_sum_weights = log_sum_exp(old_span.log_sum_weights,
                                            new_span.log_sum_weights);
    combined.alpha_sum = old_span.alpha_sum + new_span.alpha_sum;
    combined.n_alpha = old_span.n_alpha + new_span.n_alpha;

    // Two combine policies (Betancourt 2017):
    //   - MULTINOMIAL: prob_new = w_new / (w_old + w_new)
    //   - METROPOLIS_BIASED: prob_new = min(1, w_new / w_old)
    // In log-space: log_prob_select_new differs between the two.
    double log_prob_select_new;
    if (update_policy == CombineUpdate::METROPOLIS_BIASED) {
        // Biased: favor new. log_prob = min(0, log_w_new - log_w_old)
        // If log_w_new >= log_w_old, prob=1 (always accept new).
        log_prob_select_new = std::min(0.0,
            new_span.log_sum_weights - old_span.log_sum_weights);
    } else {
        // Multinomial (equal weight by state count)
        log_prob_select_new =
            new_span.log_sum_weights - combined.log_sum_weights;
    }
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    const double u = uniform(rng);
    if (std::log(u) < log_prob_select_new) {
        combined.theta_proposal = std::move(new_span.theta_proposal);
        combined.grad_proposal  = std::move(new_span.grad_proposal);
        combined.lp_proposal    = new_span.lp_proposal;
    } else {
        combined.theta_proposal = std::move(old_span.theta_proposal);
        combined.grad_proposal  = std::move(old_span.grad_proposal);
        combined.lp_proposal    = old_span.lp_proposal;
    }

    // Endpoints: depends on extend_direction. Move-construct from the
    // (rvalue) old_span/new_span — the unselected endpoints will be
    // destroyed when the function returns, so move is safe and saves
    // ~4 arma::vec copies per combine (each ~dim*8 bytes + alloc churn).
    if (extend_direction > 0) {
        // Forward extend: old.left stays, new.right becomes new combined.right
        combined.theta_left = std::move(old_span.theta_left);
        combined.p_left     = std::move(old_span.p_left);
        combined.grad_left  = std::move(old_span.grad_left);
        combined.lp_left    = old_span.lp_left;

        combined.theta_right = std::move(new_span.theta_right);
        combined.p_right     = std::move(new_span.p_right);
        combined.grad_right  = std::move(new_span.grad_right);
        combined.lp_right    = new_span.lp_right;
    } else {
        // Backward extend: new.left becomes combined.left, old.right stays
        combined.theta_left = std::move(new_span.theta_left);
        combined.p_left     = std::move(new_span.p_left);
        combined.grad_left  = std::move(new_span.grad_left);
        combined.lp_left    = new_span.lp_left;

        combined.theta_right = std::move(old_span.theta_right);
        combined.p_right     = std::move(old_span.p_right);
        combined.grad_right  = std::move(old_span.grad_right);
        combined.lp_right    = old_span.lp_right;
    }

    // U-turn termination: combined subtree is "stop" if the trajectory
    // turns back on itself in the Mahalanobis metric defined by M^{-1}.
    // (Critical fix: was previously unscaled — only worked for M = I.)
    arma::vec delta = combined.theta_right - combined.theta_left;
    bool no_uturn = combined_no_uturn_metric(delta, combined.p_left,
                                              combined.p_right, M);
    combined.stop = !no_uturn;
    // U-turn ≠ energy divergence; the latter only propagates from a
    // divergent leaf (set in from_single).
    combined.energy_divergent =
        old_span.energy_divergent || new_span.energy_divergent;

    return combined;
}

// Dispatch helpers for the metric kind.
inline bool combined_no_uturn_metric(const arma::vec& delta,
                                      const arma::vec& p_left,
                                      const arma::vec& p_right,
                                      const DiagonalMetric& M) {
    return combined_no_uturn_diag(delta, p_left, p_right, M.inv_diag);
}
inline bool combined_no_uturn_metric(const arma::vec& delta,
                                      const arma::vec& p_left,
                                      const arma::vec& p_right,
                                      const DenseMetric& M) {
    return combined_no_uturn_dense(delta, p_left, p_right, M.inv_mat);
}

// Build a subtree of depth d in the given direction. Recursive
// implementation following Hoffman 2014 Algorithm 6.
//
// state_in: the endpoint of the current trajectory at the side we're
//            extending FROM. For forward (dir = +1), this is the right
//            endpoint; for backward (dir = -1), this is the left
//            endpoint.
//
// H0:        the GLOBAL trajectory-initial Hamiltonian. Threaded through
//            unchanged (matching Hoffman 2014's θ⁰,r⁰ parameters that
//            stay fixed across the whole recursion) and used in the base
//            case for both the acceptance statistic and the divergence
//            test.
// max_energy_error: Δ_max divergence threshold (Stan max_deltaH_, default
//            1000). A leaf with H_proposal − H0 > max_energy_error stops
//            the tree.
//
// Returns a new SpanW representing the built subtree.
template <typename LogDensityFn, typename Rng, typename Metric>
SpanW build_subtree(const arma::vec& theta_seed,
                     const arma::vec& p_seed,
                     const arma::vec& grad_seed,
                     double lp_seed,
                     int direction,    // +1 or -1
                     std::size_t depth,
                     double eps,
                     Metric M,
                     double H0,
                     double max_energy_error,
                     LogDensityFn& log_density,
                     Rng& rng) {
    if (depth == 0) {
        // Base case: take one leapfrog step.
        double effective_eps = (direction > 0) ? eps : -eps;
        auto step = leapfrog_step(theta_seed, p_seed, grad_seed,
                                   effective_eps, M, log_density);
        double K_proposal = kinetic_energy(step.p_new, M);
        double H_proposal = -step.lp_new + K_proposal;
        // Acceptance stat AND divergence are relative to the GLOBAL H0,
        // NOT the per-leaf seed energy (Hoffman 2014 Alg 6 / Stan).
        (void) lp_seed;  // seed energy no longer used for the baseline
        return SpanW::from_single(step.theta_new, step.p_new,
                                    step.grad_new, step.lp_new,
                                    H_proposal, H0, max_energy_error);
    }
    // Recursive case: build two subtrees of depth-1.
    SpanW first = build_subtree(theta_seed, p_seed, grad_seed, lp_seed,
                                  direction, depth - 1, eps, M,
                                  H0, max_energy_error,
                                  log_density, rng);
    if (first.stop) return first;

    // Seed the second subtree from the far endpoint of the first
    // (in the direction of extension).
    //
    // build_subtree takes its seed args by const& and only reads them
    // (passing them down to leapfrog_step which also takes const&), so
    // we can hand it the first sub-tree's endpoint vec by reference --
    // no need to copy into local seed_* variables. After build_subtree
    // returns, `first` is still untouched, so the subsequent
    // std::move(first) into combine_subtrees is still valid.
    const bool fwd = (direction > 0);
    const arma::vec& seed_theta = fwd ? first.theta_right : first.theta_left;
    const arma::vec& seed_p     = fwd ? first.p_right     : first.p_left;
    const arma::vec& seed_grad  = fwd ? first.grad_right  : first.grad_left;
    const double     seed_lp    = fwd ? first.lp_right    : first.lp_left;
    SpanW second = build_subtree(seed_theta, seed_p, seed_grad, seed_lp,
                                   direction, depth - 1, eps, M,
                                   H0, max_energy_error,
                                   log_density, rng);
    return combine_subtrees(std::move(first), std::move(second),
                             direction, M, rng);
}

}  // namespace internal
}  // namespace AI4BayesCode
