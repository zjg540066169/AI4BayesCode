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
  ##       cached_at_new_draw_pos_out and cached_at_new_draw_neg_out --
  ##       to honor the cache invariant under mcmclib's boundary swap
  ##       convention (the second sub-tree call for direction=+1
  ##       maps callee's new_draw_neg to caller's new_draw_pos, etc.).
  ##       Single-cache versions of this patch produced 253 invariant
  ##       violations on a 4-dim toy test because they assumed
  ##       new_draw_pos always tracks the "far forward leaf", which is
  ##       NOT what mcmclib's convention produces at depth >= 2.
  ##       See project_mcmclib_nuts_cache_investigation.md.
  ##     - AI4BayesCode fork (2026-07-25, JZ): metric-kind dispatch.
  ##       Every leap_frog drift + kinetic-energy dot on
  ##       inv_precond_matrix * v was rerouted through nuts_metric_state_t
  ##       so IDENTITY skips the matvec entirely (raw daxpy) and DIAGONAL
  ##       uses the arma element-wise "%" product. DENSE keeps the
  ##       upstream matvec byte-identical. The struct is a plain
  ##       dispatch tag + non-owning pointers; nuts.hpp owns the storage
  ##       (or a cached copy on nuts_settings_t). Signatures of
  ##       nuts_find_initial_step_size and nuts_build_tree now take a
  ##       const nuts_metric_state_t& in place of const Mat_t&
  ##       inv_precond_matrix. If rebasing onto upstream mcmclib,
  ##       re-apply this signature change and revert to the Mat_t path.
  ##     - AI4BayesCode fork (2026-07-25, JZ) [Fix #3]: per-depth
  ##       ColVec_t scratch pool inside nuts_build_tree. Each recursive
  ##       frame previously stack-constructed up to 12 ColVec_t via the
  ##       recursive-case + one direction branch (14 including the two
  ##       leaf-case locals); on max_tree_depth=10 this is 154 arma
  ##       ColVec_t heap allocations per nuts() call and O(dim) memcpy
  ##       work per slot on entry. The pool is a
  ##       std::vector<build_tree_scratch_t> sized max_tree_depth+1 and
  ##       indexed by tree_depth. Alias-safe by construction: a caller
  ##       at depth d uses pool[d]; the callee at depth d-1 uses
  ##       pool[d-1]. Recursion is single-threaded depth-first, so at
  ##       most one frame is active per depth. Each pool slot is
  ##       pre-sized to n_vals in nuts_impl so subsequent arma
  ##       assignments reuse memory instead of resizing. Bit-identical
  ##       math: no arithmetic changed, only WHERE the temporaries
  ##       live. Signature of nuts_build_tree gains a
  ##       std::vector<build_tree_scratch_t>& scratch_pool trailing
  ##       parameter; find_initial_step_size is untouched (called once
  ##       per nuts() call, ROI negligible). If rebasing onto upstream
  ##       mcmclib, delete build_tree_scratch_t + the scratch_pool
  ##       parameter and revert to stack-local ColVec_t.
  ##
  ################################################################################*/

/*
 * No-U-Turn Sampler (NUTS) (with Dual Averaging)
 */

#ifndef _mcmc_nuts_IPP
#define _mcmc_nuts_IPP

// ---- AI4BayesCode fork (2026-07-25): metric-kind dispatch struct ----------
// nuts_metric_state_t is a plain dispatch tag + non-owning pointers into the
// preconditioner cache owned by nuts_impl (or nuts_settings_t). It replaces
// the `const Mat_t& inv_precond_matrix` parameter throughout the NUTS
// internals so IDENTITY and DIAGONAL can skip the dense matvec entirely.
// Correctness invariant:
//   IDENTITY : inv(I) * v == v exactly (floating-point identity).
//   DIAGONAL : inv(diag(d)) * v == (1/d) elementwise * v exactly.
//   DENSE    : inv_mat * v -- byte-identical to upstream inv_precond_matrix * v.
// If rebasing onto upstream mcmclib, revert callers to pass Mat_t and drop
// this struct.
// --------------------------------------------------------------------------
struct nuts_metric_state_t {
    metric_kind_t   kind      = metric_kind_t::IDENTITY;
    const ColVec_t* inv_diag  = nullptr;  // used iff kind == DIAGONAL
    const ColVec_t* sqrt_diag = nullptr;  // used iff kind == DIAGONAL
    const Mat_t*    inv_mat   = nullptr;  // used iff kind == DENSE
    const Mat_t*    sqrt_mat  = nullptr;  // used iff kind == DENSE
};

// ---- AI4BayesCode fork (2026-07-25) [Fix #3]: per-depth scratch pool -------
// Replaces the stack-local ColVec_t constructions inside nuts_build_tree with
// slots on a caller-owned per-depth pool.
//
// Alias-safety: nuts_build_tree at depth d touches ONLY scratch_pool[d]. The
// two recursive calls inside the depth-d body invoke nuts_build_tree(depth-1),
// which touches scratch_pool[d-1]. Recursion is single-threaded and
// depth-first, so at most one frame is active at each depth at any moment;
// callee's pool[d-1] slots and caller's pool[d] slots therefore never overlap
// in memory. The two calls at depth d never overlap either: the first call
// returns before the second is invoked, and both use pool[d-1].
//
// Within a single frame:
//   * The leaf case (tree_depth == 0) uses only leaf_new_mntm and
//     leaf_local_raw_grad.
//   * The recursive case (tree_depth > 0) uses the "first-subtree" slots
//     (which must survive across the second subtree call), the "second-subtree
//     output" slots (only touched inside the second subtree branch), and the
//     "direction" slots (only one branch runs per frame). The leaf slots are
//     not touched in the recursive path, so leaf/recursive live cleanly in
//     the same pool slot without aliasing (they are mutually exclusive).
//
// Slots are pre-sized to n_vals in nuts_impl (see nuts.hpp) so subsequent arma
// operator= calls reuse existing storage rather than reallocating. All math
// remains bit-identical to the pre-Fix#3 stack-local version: only the
// storage location of the temporaries changes.
//
// If rebasing onto upstream mcmclib, delete this struct + the scratch_pool
// parameter on nuts_build_tree and revert to stack-local ColVec_t.
// ---------------------------------------------------------------------------
struct build_tree_scratch_t {
    // Leaf case (tree_depth == 0). Mutually exclusive with the recursive
    // case; kept as separate slots for readability -- the memory cost is
    // trivial (2 * n_vals doubles per depth).
    ColVec_t leaf_new_mntm;
    ColVec_t leaf_local_raw_grad;

    // First-subtree outputs (survive across the SECOND subtree call).
    ColVec_t new_draw_p;
    ColVec_t cached_at_first_pos;
    ColVec_t cached_at_first_neg;
    ColVec_t prop_grad_first;

    // Second-subtree output holders (only touched in the recursive-case
    // second-subtree branch).
    ColVec_t new_draw_pp;
    ColVec_t cached_at_second_pos;
    ColVec_t cached_at_second_neg;
    ColVec_t prop_grad_second;

    // Direction-specific temporaries. Only ONE branch (direction_val == -1
    // OR == +1) runs per frame, so a single set of four slots serves both.
    // dir_draw / dir_mntm hold the copy of draw_neg / mntm_neg (or
    // draw_pos / mntm_pos) that is passed as the second subtree's input
    // starting position; dir_dummy_draw / dir_dummy_mntm receive the
    // "lost" boundary written by the callee.
    ColVec_t dir_dummy_draw;
    ColVec_t dir_dummy_mntm;
    ColVec_t dir_draw;
    ColVec_t dir_mntm;
};

// K = m^T inv(M) m / 2 with metric dispatch.
inline
fp_t
nuts_metric_K(const nuts_metric_state_t& metric, const ColVec_t& mntm)
{
    switch (metric.kind) {
        case metric_kind_t::IDENTITY:
            // K = m^T m / 2
            return BMO_MATOPS_DOT_PROD(mntm, mntm) / fp_t(2);
        case metric_kind_t::DIAGONAL:
            // K = m^T (inv_diag % m) / 2
            return BMO_MATOPS_DOT_PROD(mntm, (*metric.inv_diag) % mntm) / fp_t(2);
        case metric_kind_t::DENSE:
        default:
            return BMO_MATOPS_DOT_PROD(mntm, (*metric.inv_mat) * mntm) / fp_t(2);
    }
}

//

inline
fp_t
nuts_find_initial_step_size(
    const ColVec_t& draw_vec,
    const ColVec_t& mntm_vec,
    const nuts_metric_state_t& metric,
    std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* target_data)> box_log_kernel_fn,
    std::function<fp_t (const ColVec_t& pos_inp, ColVec_t& raw_grad_out, void* target_data)> raw_grad_fn,
    std::function<void (const fp_t step_size, const size_t n_leap_steps, ColVec_t& new_draw, ColVec_t& new_mntm, ColVec_t& cached_raw_grad, fp_t& cached_box_U_out, void* target_data)> leap_frog_fn,
    void* target_data
)
{
    fp_t step_size = fp_t(1); // initial value
    fp_t leap_U_unused = fp_t(0);  // value-cache out param (warmup: unused)

    //

    fp_t prev_U = - box_log_kernel_fn(draw_vec, nullptr, target_data);

    if (!std::isfinite(prev_U)) {
        prev_U = posinf;
    }

    // AI4BayesCode fork (2026-07-25): metric dispatch (was:
    //   prev_K = DOT(mntm_vec, inv_precond_matrix * mntm_vec) / 2 ;
    // ).
    fp_t prev_K = nuts_metric_K(metric, mntm_vec);

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

    // AI4BayesCode fork (2026-07-25): metric dispatch.
    fp_t prop_K = nuts_metric_K(metric, new_mntm);

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

        // AI4BayesCode fork (2026-07-25): metric dispatch (was:
        //   prop_K = DOT(new_mntm, inv_precond_matrix * new_mntm) / 2 ;
        // ).
        prop_K = nuts_metric_K(metric, new_mntm);

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
    // AI4BayesCode fork (2026-07-25): metric dispatch struct in place of
    // const Mat_t& inv_precond_matrix. Byte-identical dispatch on the DENSE
    // path; O(n) leap on IDENTITY / DIAGONAL. See top of file.
    const nuts_metric_state_t& metric,
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
    // AI4BayesCode fork (2026-07-25) [Fix #3]: per-depth ColVec_t pool.
    // Owned + pre-sized by nuts_impl. This frame touches scratch_pool[tree_depth]
    // exclusively; the recursive call at depth-1 touches scratch_pool[tree_depth-1].
    // See build_tree_scratch_t comment above for the alias-safety argument.
    std::vector<build_tree_scratch_t>& scratch_pool
)
{
    const fp_t max_tuning_par = 1000;
    // AI4BayesCode fork (2026-07-25) [Fix #3]: bind our depth's slot ONCE.
    build_tree_scratch_t& scratch = scratch_pool[tree_depth];

    if (tree_depth == size_t(0)) {
        new_draw = draw_vec;
        // AI4BayesCode fork (2026-07-25) [Fix #3]: pool-backed (was stack
        // ColVec_t new_mntm = mntm_vec).
        ColVec_t& new_mntm = scratch.leaf_new_mntm;
        new_mntm = mntm_vec;

        // AI4BayesCode fork (2026-07-25) [Fix #3]: pool-backed (was stack
        // ColVec_t local_raw_grad = cached_raw_grad_in).
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

        // AI4BayesCode fork (2026-07-25): metric dispatch (was:
        //   prop_K = DOT(new_mntm, inv_precond_matrix * new_mntm) / 2 ;
        // ).
        fp_t prop_K = nuts_metric_K(metric, new_mntm);

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
        // AI4BayesCode fork (2026-07-25) [Fix #3]: pool-backed. First-subtree
        // slots must survive across the second subtree call and must NOT be
        // written by the callee (which uses pool[d-1], disjoint from pool[d]).
        ColVec_t& new_draw_p          = scratch.new_draw_p;
        ColVec_t& cached_at_first_pos = scratch.cached_at_first_pos;
        ColVec_t& cached_at_first_neg = scratch.cached_at_first_neg;
        ColVec_t& prop_grad_first     = scratch.prop_grad_first;
        // First sub-tree's selected-proposal value+grad (PERF threading).
        fp_t prop_box_U_first = fp_t(0);

        nuts_build_tree(
            direction_val, step_size, log_rand_val, prev_U, prev_K,
            draw_vec, mntm_vec, metric,
            box_log_kernel_fn, leap_frog_fn, tree_depth - 1,
            new_draw_p, new_draw_pos, new_draw_neg, new_mntm_pos, new_mntm_neg,
            n_p_val, s_p_val, alpha_p_val, n_alpha_p_val, rand_engine, target_data,
            cached_raw_grad_in, cached_at_first_pos, cached_at_first_neg,
            prop_box_U_first, prop_grad_first, scratch_pool);

        // After first sub-tree: new_draw_pos/neg are the first sub-tree's
        // boundaries; cached_at_first_pos/neg are aligned with them.

        if (s_p_val == size_t(1)) {
            size_t n_pp_val;
            size_t s_pp_val;
            fp_t alpha_pp_val;
            size_t n_alpha_pp_val;
            // AI4BayesCode fork (2026-07-25) [Fix #3]: pool-backed.
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
                // AI4BayesCode fork (2026-07-25) [Fix #3]: pool-backed
                // direction-branch temporaries (only one branch runs).
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
                    draw_neg, mntm_neg, metric,
                    box_log_kernel_fn, leap_frog_fn, tree_depth - 1,
                    new_draw_pp, new_draw_neg, dummy_draw, new_mntm_neg, dummy_mntm,
                    n_pp_val, s_pp_val, alpha_pp_val, n_alpha_pp_val, rand_engine, target_data,
                    cached_at_first_neg, cached_at_second_pos, cached_at_second_neg,
                    prop_box_U_second, prop_grad_second, scratch_pool);
                // mcmclib boundary mapping for direction=-1 second sub-tree:
                //   arg 13 = callee's new_draw_pos -> caller's new_draw_neg (UPDATED via swap)
                //   arg 14 = callee's new_draw_neg -> dummy_draw (lost)
                // Cache mapping (must MATCH boundary mapping):
                //   caller's new_draw_neg cache = callee's new_draw_POS cache
                //   caller's new_draw_pos cache = (unchanged from first sub-tree)
                cached_at_new_draw_neg_out = cached_at_second_pos;
                cached_at_new_draw_pos_out = cached_at_first_pos;
            } else {
                // Second sub-tree extends forward from new_draw_pos.
                // AI4BayesCode fork (2026-07-25) [Fix #3]: pool-backed.
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
                    draw_pos, mntm_pos, metric,
                    box_log_kernel_fn, leap_frog_fn, tree_depth - 1,
                    new_draw_pp, dummy_draw, new_draw_pos, dummy_mntm, new_mntm_pos,
                    n_pp_val, s_pp_val, alpha_pp_val, n_alpha_pp_val, rand_engine, target_data,
                    cached_at_first_pos, cached_at_second_pos, cached_at_second_neg,
                    prop_box_U_second, prop_grad_second, scratch_pool);
                // mcmclib boundary mapping for direction=+1 second sub-tree:
                //   callee's new_draw_pos -> dummy_draw (lost)
                //   callee's new_draw_neg -> caller's new_draw_pos (updated)
                // Cache mapping (matches boundary mapping):
                //   caller's new_draw_pos's cache = callee's new_draw_neg's cache
                //   caller's new_draw_neg's cache = (unchanged from first sub-tree)
                cached_at_new_draw_pos_out = cached_at_second_neg;
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
