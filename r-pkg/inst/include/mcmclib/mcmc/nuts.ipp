/*################################################################################
  ##
  ##   Copyright (C) 2011-2023 Keith O'Hara
  ##
  ##   This file is part of the MCMC C++ library.
  ##
  ##   Licensed under the Apache License, Version 2.0 (the "License");
  ##   you may not use this file except in compliance with the License.
  ##   You may obtain a copy of the License at
  ##
  ##       http://www.apache.org/licenses/LICENSE-2.0
  ##
  ##   Unless required by applicable law or agreed to in writing, software
  ##   distributed under the License is distributed on an "AS IS" BASIS,
  ##   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  ##   See the License for the specific language governing permissions and
  ##   limitations under the License.
  ##
  ################################################################################*/

/*################################################################################
  ##
  ##   Modifications Copyright (C) 2025-2026 Jungang Zou
  ##
  ##   This file has been modified from the original MCMClib source as part of
  ##   the MTBART and block_mcmc projects. Modifications are licensed under the
  ##   Apache License, Version 2.0 (same as the original file).
  ##
  ##   Summary of modifications:
  ##     - Added an overflow / underflow guard inside
  ##       nuts_find_initial_step_size: the doubling/halving loop is now
  ##       capped at max_steps = 50 iterations and step_size is clamped
  ##       to the range [1e-10, 1e10] if it becomes non-finite or leaves
  ##       that range. This prevents infinite loops / NaN step sizes when
  ##       the conditional target has extreme curvature (observed when
  ##       kappa is initialized at a large value in DP_DART / RE_DART).
  ##     - Fixed the search direction of nuts_find_initial_step_size.
  ##       The original implementation only increased the step size:
  ##       the while condition was the single-sided test "L > -log 2"
  ##       (equivalent to "p(theta',r')/p(theta,r) > 1/2"), which only
  ##       matches Hoffman & Gelman 2014 Algorithm 4 in the doubling
  ##       branch. When the initial epsilon = 1 was already too large
  ##       (e.g. a tight conditional posterior with a large gradient at
  ##       the starting point), the loop was never entered and the
  ##       function returned step_size = 1, causing the downstream dual
  ##       averaging loop to diverge. The condition is now the two-sided
  ##       test  a * (L + log 2) > 0  with a frozen outside the loop, as
  ##       specified in Algorithm 4.
  ##     - Phase-1 speedup (2026-06-08, GZ): leapfrog gradient cache.
  ##       leap_frog_fn now caches the gradient computed for the second
  ##       half-kick of one step and reuses it as the first half-kick
  ##       gradient of the next step. This halves the number of
  ##       target_log_kernel-with-grad calls per leapfrog step.
  ##       nuts_build_tree maintains DUAL caches per call --
  ##       cached_at_new_draw_pos_out and cached_at_new_draw_neg_out.
  ##     - CORRECTNESS (2026-08-17, JZ): the second sub-tree call had its
  ##       new_draw_pos / new_draw_neg output slots TRANSPOSED in both
  ##       direction branches, so the caller kept the NEAR boundary of the
  ##       new sub-tree and discarded the far one. Upstream mcmclib has the
  ##       same transposition; the outer doubling loop in nuts.hpp does NOT,
  ##       which is how the two came to disagree.
  ##       This is what the 253 cache-invariant violations reported above
  ##       were really pointing at: the assumption that "new_draw_pos always
  ##       tracks the far forward leaf" is CORRECT -- Hoffman & Gelman 2014
  ##       Algorithm 3 says so -- and the dual-cache patch was aligned to the
  ##       broken mapping instead of to the algorithm. Both are now fixed
  ##       together, so -DCACHE_CHECK stays clean.
  ##       Effect of the bug: from depth 2 the U-turn check compared a
  ##       mid-tree state rather than a trajectory endpoint; from depth 3 the
  ##       next sub-tree started from a non-extremal state and re-walked
  ##       states already in the trajectory, so n_val double-counted and both
  ##       acceptance ratios consumed corrupted counts. Measured on
  ##       Dirichlet(2,3,4,5) through a SIMPLEX at fixed step size, 240
  ##       independent chains started from EXACT draws: E[p_0] was +2.9% at
  ##       eps 0.05, +3.5% at 0.10, +2.7% at 0.30, and clean at 1.00 (where
  ##       trajectories end at depth 0-1 and the transposition cannot show).
  ##       Flat in chain length from 500 to 200000 draws, i.e. a wrong
  ##       invariant distribution, not burn-in.
  ##       Gate: tests/test_nuts_small_step_invariance.cpp.
  ##       See project_mcmclib_nuts_cache_investigation.md for the earlier
  ##       cache work this supersedes.
  ##     - SAFE-SPEEDUP (2026-07-26, JZ) Fix #3: per-depth ColVec_t scratch
  ##       pool for nuts_build_tree, eliminating ~154 ColVec_t stack
  ##       constructions per nuts() call at max_tree_depth=10. See top of
  ##       file for the build_tree_scratch_t struct and alias-safety
  ##       argument. Byte-identical to the pre-pool version.
  ##
  ################################################################################*/

/*
 * No-U-Turn Sampler (NUTS) (with Dual Averaging)
 */

#ifndef _mcmc_nuts_IPP
#define _mcmc_nuts_IPP

// -------------------------------------------------------------------
// FORK MARKER (2026-07-26, JZ): SAFE-SPEEDUP Fix #3, per-depth ColVec_t pool.
//
// nuts_build_tree previously stack-constructed up to 14 ColVec_t per
// recursion frame; at max_tree_depth=10 that's ~154 heap allocations plus
// O(n) memcpy work per slot per nuts() call. The pool is a
// std::vector<build_tree_scratch_t> sized max_tree_depth+1, indexed by
// tree_depth. Alias-safety: a caller at depth d uses pool[d]; the callee
// at depth d-1 uses pool[d-1]; recursion is single-threaded depth-first,
// so at most one frame is active per depth. Within a single frame the
// leaf slots and recursive slots are mutually exclusive (only ONE of the
// two branches runs), so they can share the same slot layout without
// conflict. Slots are pre-sized to n_vals in nuts_impl so subsequent
// arma operator= calls reuse memory instead of resizing.
//
// BYTE-IDENTICAL to the pre-pool stack version: no arithmetic changed,
// only WHERE temporaries live. Signature of nuts_build_tree gains a
// `std::vector<build_tree_scratch_t>&` trailing parameter;
// nuts_find_initial_step_size is untouched (single call per nuts()).
//
// REBASE NOTE: delete this struct + the scratch_pool parameter and revert
// to stack-local ColVec_t declarations to restore upstream mcmclib.
// -------------------------------------------------------------------

// ===================================================================
// FORK MARKER (2026-07-27, JZ) [DIAG-vector Fix]
//
// When the preconditioner (mass) matrix supplied by the caller is a
// DIAGONAL matrix stored as a full n*n dense matrix -- the shape that
// joint_nuts_block::build_precond_ produces under use_diagonal_metric
// (M = diagmat(1/var)) -- the DENSE branch pays a full O(n^2) SGEMV per
// leapfrog drift and per leaf kinetic-energy evaluation, even though
// every off-diagonal entry is exactly 0. This block adds the machinery
// to detect that case ONCE per nuts() call (O(n^2), amortized over the
// thousands of O(n^2) matvecs it replaces) and fall onto O(n) elementwise
// ops.
//
// BIT-IDENTITY CONTRACT (see DIAGVEC probe results in the deliverable):
//   * For a diagonal D, a plain (alpha=1) matvec (D*v)_i computed by BLAS
//     gemv reduces to the single non-zero product D(i,i)*v_i with all
//     other terms 0*v_j == 0 added exactly. So the elementwise form
//     inv_diag_i * v_i is BIT-IDENTICAL to `inv_precond_matrix * v`, and
//     the quadratic form dot(v, inv_precond*v) is bit-identical when the
//     matvec temporary is built elementwise and fed to the SAME dot().
//     This holds for the KINETIC-ENERGY and MOMENTUM sites (alpha=1).
//   * The DRIFT site is `new_draw += step_size * inv_precond_matrix * v`,
//     which armadillo lowers to a FUSED gemv(alpha=step_size, beta=1).
//     BLAS applies the non-unit alpha/beta with implementation-specific
//     per-element rounding that NO elementwise grouping reproduces
//     bit-for-bit (~1 ULP at every size incl. n=1605). v2: the drift diag
//     path is ENABLED BY DEFAULT (runtime metric_is_diag) -- the ~1 ULP
//     does NOT change the stationary distribution, and gating it off left
//     high-dim diagonal metrics ~500x too slow. The AI4BAYESCODE_DIAGVEC_DRIFT
//     macro below is legacy/unused (the compile gates were removed).
//
// v2 (nuts.hpp): detection reads the INPUT precond_mat (built as diagmat(...),
// bit-exact 0 off-diagonals) and builds inv_diag/sqrt_diag DIRECTLY, so a
// diagonal metric also SKIPS the dense O(n^3) inv/chol setup entirely. (v1 read
// the dense inv/chol OUTPUT, whose ~1e-16 off-diagonal noise made detection
// ALWAYS fail -> dense O(n^2) SGEMV every leapfrog.) mat_is_exactly_diagonal is
// still used on the input. REBASE NOTE: delete this block + the metric_is_diag/
// inv_diag/sqrt_diag params + branches (search "DIAG-vector Fix") to restore the
// pre-fork dense-only path.
// ===================================================================
#ifndef AI4BAYESCODE_DIAGVEC_DRIFT
    #define AI4BAYESCODE_DIAGVEC_DRIFT 0
#endif

// True iff every off-diagonal entry of the n*n matrix M is exactly 0.0.
inline
bool
mat_is_exactly_diagonal(const Mat_t& M, const size_t n)
{
    for (size_t j = 0; j < n; ++j) {
        for (size_t i = 0; i < n; ++i) {
            if (i != j && M(i, j) != fp_t(0)) return false;
        }
    }
    return true;
}

// dot(v, D*v) for a diagonal D given by its diagonal inv_diag, computed as
// BMO_MATOPS_DOT_PROD(v, w) with w_i = inv_diag_i * v_i. Bit-identical to
// dot(v, inv_precond_matrix * v) for an exactly-diagonal inv_precond_matrix
// (the elementwise w equals the alpha=1 gemv result bit-for-bit, and the
// dot is the SAME reduction). w_scratch is caller-owned to avoid per-call
// allocation on the hot path; a set_size to the same length is a no-op.
inline
fp_t
diag_quad_form(const ColVec_t& v, const ColVec_t& inv_diag, ColVec_t& w_scratch)
{
    const size_t n = BMO_MATOPS_SIZE(v);
    w_scratch.set_size(n);
    for (size_t i = 0; i < n; ++i) w_scratch[i] = inv_diag[i] * v[i];
    return BMO_MATOPS_DOT_PROD(v, w_scratch);
}

struct build_tree_scratch_t {
    // Leaf case (tree_depth == 0). Mutually exclusive with recursive case;
    // kept as distinct slots for readability -- memory cost is trivial.
    ColVec_t leaf_new_mntm;
    ColVec_t leaf_local_raw_grad;
    // FORK MARKER (2026-07-27, JZ) [DIAG-vector Fix]: elementwise matvec
    // temporary w_i = inv_diag_i * new_mntm_i for the leaf kinetic energy.
    ColVec_t leaf_metric_w;

    // First-subtree outputs (must survive across the SECOND subtree call).
    // The callee writes into pool[d-1]; these live in pool[d] so aliasing is
    // impossible.
    ColVec_t new_draw_p;
    ColVec_t cached_at_first_pos;
    ColVec_t cached_at_first_neg;
    ColVec_t prop_grad_first;

    // Second-subtree output holders (only touched inside the second-subtree
    // branch when the first subtree returned s_p_val == 1).
    ColVec_t new_draw_pp;
    ColVec_t cached_at_second_pos;
    ColVec_t cached_at_second_neg;
    ColVec_t prop_grad_second;

    // Direction-specific temporaries. Only ONE of direction_val == -1 / +1
    // runs per frame, so a single set of four slots serves both. dir_draw
    // / dir_mntm hold the copy of draw_neg/mntm_neg (or draw_pos/mntm_pos)
    // that is passed as the second subtree's starting position; dir_dummy_*
    // receive the "lost" boundary written by the callee.
    ColVec_t dir_dummy_draw;
    ColVec_t dir_dummy_mntm;
    ColVec_t dir_draw;
    ColVec_t dir_mntm;
};

//

inline
fp_t
nuts_find_initial_step_size(
    const ColVec_t& draw_vec,
    const ColVec_t& mntm_vec,
    const Mat_t& inv_precond_matrix,
    // FORK MARKER (2026-07-26, JZ) [IDENTITY-only Fix #1]: identity metric
    // fast-path flag. When true, skip the two O(n^2) matvecs and use plain
    // dot(p,p)/2. Same result byte-wise as inv_precond_matrix == I; only
    // the flops differ.
    const bool is_identity_metric,
    // FORK MARKER (2026-07-27, JZ) [DIAG-vector Fix]: diagonal-metric flag +
    // its diagonal. When set, the O(n^2) K matvecs use the O(n) elementwise
    // quadratic form (bit-identical to the dense dot(p, inv*p)).
    const bool metric_is_diag,
    const ColVec_t& inv_diag,
    std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* target_data)> box_log_kernel_fn,
    std::function<fp_t (const ColVec_t& pos_inp, ColVec_t& raw_grad_out, void* target_data)> raw_grad_fn,
    std::function<void (const fp_t step_size, const size_t n_leap_steps, ColVec_t& new_draw, ColVec_t& new_mntm, ColVec_t& cached_raw_grad, fp_t& cached_box_U_out, void* target_data)> leap_frog_fn,
    void* target_data
)
{
    fp_t step_size = fp_t(1); // initial value
    fp_t leap_U_unused = fp_t(0);  // value-cache out param (warmup: unused)

    // FORK MARKER (2026-07-27, JZ) [DIAG-vector Fix]: elementwise matvec temp
    // for the diagonal K quadratic form (not on the hot per-leaf path).
    ColVec_t metric_w;

    //

    fp_t prev_U = - box_log_kernel_fn(draw_vec, nullptr, target_data);

    if (!std::isfinite(prev_U)) {
        prev_U = posinf;
    }

    // FORK MARKER (2026-07-26, JZ) [IDENTITY-only Fix #1]
    // FORK MARKER (2026-07-27, JZ) [DIAG-vector Fix]: diag branch (bit-identical)
    fp_t prev_K = is_identity_metric
        ? BMO_MATOPS_DOT_PROD(mntm_vec, mntm_vec) / fp_t(2)
        : metric_is_diag
            ? diag_quad_form(mntm_vec, inv_diag, metric_w) / fp_t(2)
            : BMO_MATOPS_DOT_PROD(mntm_vec, inv_precond_matrix * mntm_vec) / fp_t(2);

    //

    // Bootstrap the cache at draw_vec.
    const size_t n_vals_init = BMO_MATOPS_SIZE(draw_vec);
    ColVec_t initial_raw_grad(n_vals_init);
    raw_grad_fn(draw_vec, initial_raw_grad, target_data);

    ColVec_t new_draw = draw_vec;
    ColVec_t new_mntm = mntm_vec;
    ColVec_t local_raw_grad = initial_raw_grad;

    leap_frog_fn(step_size, 1, new_draw, new_mntm, local_raw_grad, leap_U_unused, target_data);

    fp_t prop_U = - box_log_kernel_fn(new_draw, nullptr, target_data);

    if (!std::isfinite(prop_U)) {
        prop_U = posinf;
    }

    // FORK MARKER (2026-07-26, JZ) [IDENTITY-only Fix #1]
    // FORK MARKER (2026-07-27, JZ) [DIAG-vector Fix]: diag branch (bit-identical)
    fp_t prop_K = is_identity_metric
        ? BMO_MATOPS_DOT_PROD(new_mntm, new_mntm) / fp_t(2)
        : metric_is_diag
            ? diag_quad_form(new_mntm, inv_diag, metric_w) / fp_t(2)
            : BMO_MATOPS_DOT_PROD(new_mntm, inv_precond_matrix * new_mntm) / fp_t(2);

    //

    // Hoffman & Gelman (2014), Algorithm 4 (see prior comment block for
    // search-direction fix history).

    const auto log_accept_ratio = [&]() -> fp_t {
        return (-(prop_U + prop_K)) - (-(prev_U + prev_K));
    };

    const int a_val = (log_accept_ratio() > std::log(fp_t(0.5))) ? 1 : -1;

    auto check_cond = [&]() -> bool {
        return static_cast<fp_t>(a_val)
             * (log_accept_ratio() + std::log(fp_t(2))) > fp_t(0);
    };

    int max_steps = 50;  // guard against pathological overflow / underflow
    while (check_cond() && max_steps > 0) {
        step_size *= std::pow(fp_t(2), a_val);

        if (!std::isfinite(step_size) ||
            step_size > fp_t(1e10) || step_size < fp_t(1e-10)) {
            step_size = std::max(std::min(step_size, fp_t(1e10)),
                                 fp_t(1e-10));
            break;
        }

        // Hoffman & Gelman (2014) Algorithm 4: re-leapfrog from the ORIGINAL
        // (draw_vec, mntm_vec) with the candidate step_size each iteration —
        // NOT a cumulative leap. The old upstream mcmclib leapt cumulatively
        // (testing a multi-leap trajectory rather than a SINGLE leap of the
        // candidate step), returning a wrong "reasonable" epsilon. That
        // mis-anchors dual-averaging (mu = log(10*eps)) and badly slows
        // step-size convergence for non-identity (diagonal/dense) metrics.
        new_draw       = draw_vec;
        new_mntm       = mntm_vec;
        local_raw_grad = initial_raw_grad;

        leap_frog_fn(step_size, 1, new_draw, new_mntm, local_raw_grad, leap_U_unused, target_data);

        prop_U = - box_log_kernel_fn(new_draw, nullptr, target_data);

        if (!std::isfinite(prop_U)) {
            prop_U = posinf;
        }

        // FORK MARKER (2026-07-26, JZ) [IDENTITY-only Fix #1]
        // FORK MARKER (2026-07-27, JZ) [DIAG-vector Fix]: diag branch (bit-identical)
        prop_K = is_identity_metric
            ? BMO_MATOPS_DOT_PROD(new_mntm, new_mntm) / fp_t(2)
            : metric_is_diag
                ? diag_quad_form(new_mntm, inv_diag, metric_w) / fp_t(2)
                : BMO_MATOPS_DOT_PROD(new_mntm, inv_precond_matrix * new_mntm)
                  / fp_t(2);

        max_steps--;
    }

    return step_size;
}

//

inline
void
nuts_build_tree(
    const int direction_val,
    const fp_t step_size,
    const fp_t log_rand_val,
    const fp_t prev_U,
    const fp_t prev_K,
    const ColVec_t& draw_vec,
    const ColVec_t& mntm_vec,
    const Mat_t& inv_precond_matrix,
    std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* target_data)> box_log_kernel_fn,
    std::function<void (const fp_t step_size, const size_t n_leap_steps, ColVec_t& new_draw, ColVec_t& new_mntm, ColVec_t& cached_raw_grad, fp_t& cached_box_U_out, void* target_data)> leap_frog_fn,
    const size_t tree_depth,
    ColVec_t& new_draw,
    ColVec_t& new_draw_pos,
    ColVec_t& new_draw_neg,
    ColVec_t& new_mntm_pos,
    ColVec_t& new_mntm_neg,
    size_t& n_val,
    size_t& s_val,
    fp_t& alpha_val,
    size_t& n_alpha_val,
    rand_engine_t& rand_engine,
    void* target_data,
    // Dual-cache parameters (Phase-1 cache).
    //   cached_raw_grad_in : grad@draw_vec, satisfied by caller.
    //   cached_at_new_draw_pos_out : on return, grad@new_draw_pos.
    //   cached_at_new_draw_neg_out : on return, grad@new_draw_neg.
    const ColVec_t& cached_raw_grad_in,
    ColVec_t& cached_at_new_draw_pos_out,
    ColVec_t& cached_at_new_draw_neg_out,
    // PERF (2026-06-14): selected proposal's box log-density VALUE and
    // natural GRAD, threaded out so the main loop can reuse them on accept
    // instead of recomputing (box_log_kernel_fn + raw_grad_fn at new_draw).
    // new_draw is a leaf created here; its value+grad were already computed.
    fp_t& prop_box_U_out,
    ColVec_t& prop_grad_out,
    // FORK MARKER (2026-07-26, JZ) [Fix #3]: per-depth ColVec_t pool.
    // Owned + pre-sized by nuts_impl. This frame touches
    // scratch_pool[tree_depth] exclusively; the recursive call at depth-1
    // touches scratch_pool[tree_depth-1]. See build_tree_scratch_t at top
    // of file for the alias-safety argument. If rebasing onto upstream,
    // drop this parameter.
    std::vector<build_tree_scratch_t>& scratch_pool,
    // FORK MARKER (2026-07-26, JZ) [IDENTITY-only Fix #1]: identity metric
    // fast-path flag. When true, the leaf's prop_K uses dot(p,p)/2 instead
    // of a full O(n^2) matvec through inv_precond_matrix. Rebase: drop.
    const bool is_identity_metric,
    // FORK MARKER (2026-07-27, JZ) [DIAG-vector Fix]: diagonal-metric flag +
    // its diagonal. When set, the leaf's prop_K uses the O(n) elementwise
    // quadratic form (bit-identical to dense dot(p, inv*p)). Rebase: drop.
    const bool metric_is_diag,
    const ColVec_t& inv_diag
)
{
    const fp_t max_tuning_par = 1000;
    // FORK MARKER (2026-07-26, JZ) [Fix #3]: bind our depth's slot once.
    build_tree_scratch_t& scratch = scratch_pool[tree_depth];

    if (tree_depth == size_t(0)) {
        new_draw = draw_vec;
        // FORK MARKER (2026-07-26, JZ) [Fix #3]: pool-backed (was:
        //   ColVec_t new_mntm = mntm_vec;
        // ). Byte-preserved assignment; only storage location changed.
        ColVec_t& new_mntm = scratch.leaf_new_mntm;
        new_mntm = mntm_vec;

        // FORK MARKER (2026-07-26, JZ) [Fix #3]: pool-backed (was:
        //   ColVec_t local_raw_grad = cached_raw_grad_in;
        // ).
        ColVec_t& local_raw_grad = scratch.leaf_local_raw_grad;
        local_raw_grad = cached_raw_grad_in;
        fp_t leaf_box_U = fp_t(0);
        leap_frog_fn(direction_val * step_size, 1, new_draw, new_mntm, local_raw_grad, leaf_box_U, target_data);
        // After leap_frog: local_raw_grad = grad(new_draw); leaf_box_U =
        // box log-density at new_draw, captured from the SAME grad eval.

        // PERF (2026-06-14): use the cached value instead of re-evaluating
        // box_log_kernel_fn(new_draw, nullptr) — one fewer full target eval
        // per leaf. leaf_box_U == box_log_kernel_fn(new_draw,nullptr) by
        // construction (raw_grad_fn's value + log_jacobian when vals_bound).
        fp_t prop_U = - leaf_box_U;
        if (!std::isfinite(prop_U)) prop_U = posinf;

        // FORK MARKER (2026-07-26, JZ) [IDENTITY-only Fix #1]
        // FORK MARKER (2026-07-27, JZ) [DIAG-vector Fix]: diag branch. Uses the
        // pool-backed scratch.leaf_metric_w so the hot leaf path allocates
        // nothing. Bit-identical to the dense dot(p, inv*p) for exactly-diagonal
        // inv_precond_matrix.
        fp_t prop_K = is_identity_metric
            ? BMO_MATOPS_DOT_PROD(new_mntm, new_mntm) / fp_t(2)
            : metric_is_diag
                ? diag_quad_form(new_mntm, inv_diag, scratch.leaf_metric_w) / fp_t(2)
                : BMO_MATOPS_DOT_PROD(new_mntm, inv_precond_matrix * new_mntm) / fp_t(2);

        n_val = (log_rand_val <= - prop_U - prop_K);
        s_val = (log_rand_val < max_tuning_par - prop_U - prop_K);

        // At depth 0, both boundaries collapse onto the leaf.
        new_draw_pos = new_draw;
        new_draw_neg = new_draw;
        new_mntm_pos = new_mntm;
        new_mntm_neg = new_mntm;
        // Both caches = grad(leaf) -- consistent with the boundary
        // setting above. Invariant: cached_at_*_out matches new_draw_*.
        cached_at_new_draw_pos_out = local_raw_grad;
        cached_at_new_draw_neg_out = local_raw_grad;

        alpha_val = std::exp( std::min( fp_t(0), - (prop_U + prop_K) + (prev_U + prev_K)) );
        n_alpha_val = 1;

        // The depth-0 proposal IS this leaf; its value+grad are already known.
        prop_box_U_out = leaf_box_U;
        prop_grad_out  = local_raw_grad;
    } else {
        size_t n_p_val;
        size_t s_p_val;
        fp_t alpha_p_val;
        size_t n_alpha_p_val;
        // FORK MARKER (2026-07-26, JZ) [Fix #3]: pool-backed first-subtree
        // slots. Callee (recursive call below) touches pool[d-1], disjoint
        // from these pool[d] slots -- safe.
        ColVec_t& new_draw_p          = scratch.new_draw_p;
        ColVec_t& cached_at_first_pos = scratch.cached_at_first_pos;
        ColVec_t& cached_at_first_neg = scratch.cached_at_first_neg;
        ColVec_t& prop_grad_first     = scratch.prop_grad_first;
        // First sub-tree's selected-proposal value+grad (PERF threading).
        fp_t prop_box_U_first = fp_t(0);

        nuts_build_tree(
            direction_val, step_size, log_rand_val, prev_U, prev_K,
            draw_vec, mntm_vec, inv_precond_matrix,
            box_log_kernel_fn, leap_frog_fn, tree_depth - 1,
            new_draw_p, new_draw_pos, new_draw_neg, new_mntm_pos, new_mntm_neg,
            n_p_val, s_p_val, alpha_p_val, n_alpha_p_val, rand_engine, target_data,
            cached_raw_grad_in, cached_at_first_pos, cached_at_first_neg,
            prop_box_U_first, prop_grad_first,
            // FORK MARKER (2026-07-26, JZ) [Fix #3]: thread scratch pool.
            scratch_pool,
            // FORK MARKER (2026-07-26, JZ) [IDENTITY-only Fix #1]: thread flag.
            is_identity_metric, metric_is_diag, inv_diag);

        // After first sub-tree: new_draw_pos/neg are the first sub-tree's
        // boundaries; cached_at_first_pos/neg are aligned with them.

        if (s_p_val == size_t(1)) {
            size_t n_pp_val;
            size_t s_pp_val;
            fp_t alpha_pp_val;
            size_t n_alpha_pp_val;
            // FORK MARKER (2026-07-26, JZ) [Fix #3]: pool-backed second-subtree.
            ColVec_t& new_draw_pp          = scratch.new_draw_pp;
            ColVec_t& cached_at_second_pos = scratch.cached_at_second_pos;
            ColVec_t& cached_at_second_neg = scratch.cached_at_second_neg;
            ColVec_t& prop_grad_second     = scratch.prop_grad_second;
            // Second sub-tree's selected-proposal value+grad (PERF threading).
            fp_t prop_box_U_second = fp_t(0);

            if (direction_val == -1) {
                // Second sub-tree extends backward from new_draw_neg.
                // Input cache for it = cached_at_first_neg
                //   (= grad@new_draw_neg, which is the second sub-tree's
                //    starting position).
                // FORK MARKER (2026-07-26, JZ) [Fix #3]: pool-backed direction
                // temporaries (only ONE branch runs per frame).
                ColVec_t& dummy_draw = scratch.dir_dummy_draw;
                ColVec_t& dummy_mntm = scratch.dir_dummy_mntm;
                ColVec_t& draw_neg   = scratch.dir_draw;
                ColVec_t& mntm_neg   = scratch.dir_mntm;
                dummy_draw = new_draw_pos;
                dummy_mntm = new_mntm_pos;
                draw_neg   = new_draw_neg;
                mntm_neg   = new_mntm_neg;

                nuts_build_tree(
                    direction_val, step_size, log_rand_val, prev_U, prev_K,
                    draw_neg, mntm_neg, inv_precond_matrix,
                    box_log_kernel_fn, leap_frog_fn, tree_depth - 1,
                    new_draw_pp, dummy_draw, new_draw_neg, dummy_mntm, new_mntm_neg,
                    n_pp_val, s_pp_val, alpha_pp_val, n_alpha_pp_val, rand_engine, target_data,
                    cached_at_first_neg, cached_at_second_pos, cached_at_second_neg,
                    prop_box_U_second, prop_grad_second,
                    // FORK MARKER (2026-07-26, JZ) [Fix #3]: thread scratch pool.
                    scratch_pool,
                    // FORK MARKER (2026-07-26, JZ) [IDENTITY-only Fix #1]: thread flag.
                    is_identity_metric, metric_is_diag, inv_diag);
                // FIXED (2026-08-17): the two output slots were TRANSPOSED
                // here, and upstream mcmclib has the same transposition. Going
                // backwards, the new far end is the callee's NEG boundary, so
                // that is what the caller's new_draw_neg must receive (Hoffman
                // & Gelman 2014 Alg 3, BuildTree else-branch, v = -1: keep the
                // callee's minus, discard its plus). The old code bound the
                // callee's POS here and threw its NEG away, so from depth 2 on
                // the U-turn check compared a mid-tree state instead of an
                // endpoint, and from depth 3 on the next sub-tree started from
                // a NON-extremal state -- the trajectory re-walked states it
                // already held, n_val double-counted them, and the acceptance
                // ratios n'/n and n''/(n'+n'') consumed the corrupted counts.
                // Measured on Dirichlet(2,3,4,5) via SIMPLEX at fixed eps:
                // +3.5% on E[p_0] at eps 0.10, flat in chain length out to
                // 200k draws (+324 se); clean at eps 1.00, where trajectories
                // stop at depth 0-1 and the transposition is invisible.
                //   boundary: callee's new_draw_neg -> caller's new_draw_neg
                //             callee's new_draw_pos -> dummy_draw (discarded)
                // Cache mapping MUST match the boundary mapping:
                //   caller's new_draw_neg cache = callee's new_draw_NEG cache
                //   caller's new_draw_pos cache = (unchanged from first sub-tree)
                cached_at_new_draw_neg_out = cached_at_second_neg;
                cached_at_new_draw_pos_out = cached_at_first_pos;
            } else {
                // Second sub-tree extends forward from new_draw_pos.
                // FORK MARKER (2026-07-26, JZ) [Fix #3]: pool-backed
                // direction temporaries (only ONE branch runs per frame).
                ColVec_t& dummy_draw = scratch.dir_dummy_draw;
                ColVec_t& dummy_mntm = scratch.dir_dummy_mntm;
                ColVec_t& draw_pos   = scratch.dir_draw;
                ColVec_t& mntm_pos   = scratch.dir_mntm;
                dummy_draw = new_draw_neg;
                dummy_mntm = new_mntm_neg;
                draw_pos   = new_draw_pos;
                mntm_pos   = new_mntm_pos;

                nuts_build_tree(
                    direction_val, step_size, log_rand_val, prev_U, prev_K,
                    draw_pos, mntm_pos, inv_precond_matrix,
                    box_log_kernel_fn, leap_frog_fn, tree_depth - 1,
                    new_draw_pp, new_draw_pos, dummy_draw, new_mntm_pos, dummy_mntm,
                    n_pp_val, s_pp_val, alpha_pp_val, n_alpha_pp_val, rand_engine, target_data,
                    cached_at_first_pos, cached_at_second_pos, cached_at_second_neg,
                    prop_box_U_second, prop_grad_second,
                    // FORK MARKER (2026-07-26, JZ) [Fix #3]: thread scratch pool.
                    scratch_pool,
                    // FORK MARKER (2026-07-26, JZ) [IDENTITY-only Fix #1]: thread flag.
                    is_identity_metric, metric_is_diag, inv_diag);
                // FIXED (2026-08-17): mirror of the v = -1 branch above --
                // see the explanation there. Going forwards the new far end is
                // the callee's POS boundary (Alg 3, else-branch: keep the
                // callee's plus, discard its minus).
                //   boundary: callee's new_draw_pos -> caller's new_draw_pos
                //             callee's new_draw_neg -> dummy_draw (discarded)
                // Cache mapping MUST match the boundary mapping:
                //   caller's new_draw_pos cache = callee's new_draw_POS cache
                //   caller's new_draw_neg cache = (unchanged from first sub-tree)
                cached_at_new_draw_pos_out = cached_at_second_pos;
                cached_at_new_draw_neg_out = cached_at_first_neg;
            }

            const fp_t prob_val = fp_t(n_pp_val) / fp_t( n_p_val + n_pp_val );
            const fp_t z = bmo::stats::runif<fp_t>(rand_engine);
            if (z < prob_val) {
                new_draw_p = new_draw_pp;
                prop_box_U_out = prop_box_U_second;   // proposal -> second sub-tree
                prop_grad_out  = prop_grad_second;
            } else {
                prop_box_U_out = prop_box_U_first;    // proposal stays first sub-tree
                prop_grad_out  = prop_grad_first;
            }

            n_p_val += n_pp_val;
            alpha_p_val += alpha_pp_val;
            n_alpha_p_val += n_alpha_pp_val;

            int check_val_1 = BMO_MATOPS_DOT_PROD(new_draw_pos - new_draw_neg, new_mntm_neg) >= fp_t(0);
            int check_val_2 = BMO_MATOPS_DOT_PROD(new_draw_pos - new_draw_neg, new_mntm_pos) >= fp_t(0);
            s_p_val = s_pp_val * check_val_1 * check_val_2;
        } else {
            // s_p_val == 0 from first sub-tree; second sub-tree not run.
            // Output caches = first sub-tree's caches (boundaries already
            // reflect first sub-tree's result). Proposal = first sub-tree's.
            cached_at_new_draw_pos_out = cached_at_first_pos;
            cached_at_new_draw_neg_out = cached_at_first_neg;
            prop_box_U_out = prop_box_U_first;
            prop_grad_out  = prop_grad_first;
        }

        n_val = n_p_val;
        s_val = s_p_val;
        alpha_val = alpha_p_val;
        n_alpha_val = n_alpha_p_val;
        new_draw = new_draw_p;
    }
}

#endif
