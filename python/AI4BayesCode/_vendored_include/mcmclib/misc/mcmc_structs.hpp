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
  ##   the MTBART project. Modifications are licensed under the Apache License,
  ##   Version 2.0 (same as the original file).
  ##
  ##   Summary of modifications:
  ##     - Added persistent dual-averaging adaptation fields to nuts_settings_t
  ##       (use_persistent_adapt, epsilon_bar_persist, h_val_persist,
  ##        mu_val_persist, adapt_iter_persist) to support mini-warmup
  ##        adaptation across Gibbs iterations in DP_DART / RE_DART.
  ##     - AI4BayesCode fork (2026-07-25, JZ): metric-kind dispatch and
  ##       preconditioner cache. See the AI4BayesCode fork block above
  ##       `metric_kind_t` below for the full design + correctness argument.
  ##
  ################################################################################*/

#ifndef mcmc_structs_HPP
#define mcmc_structs_HPP

// AEES

struct aees_settings_t
{
    size_t n_initial_draws = 1E03;
    size_t n_burnin_draws = 1E03;
    size_t n_keep_draws = 1E03;

    int omp_n_threads = -1; // numbers of threads to use

    fp_t par_scale = 1.0;
    Mat_t cov_mat;

    size_t n_rings = 5; // number of energy rings
    fp_t ee_prob_par = 0.10; // equi-energy probability parameter
    ColVec_t temper_vec; // temperature vector
};

// DE

struct de_settings_t
{
    bool jumps = false;

    size_t n_pop = 100;
    size_t n_burnin_draws = 1E03;
    size_t n_keep_draws = 1E03;

    int omp_n_threads = -1; // numbers of threads to use

    fp_t par_b = 1E-04;
    fp_t par_gamma = 1.0;
    fp_t par_gamma_jump = 2.0;

    ColVec_t initial_lb; // this will default to -0.5
    ColVec_t initial_ub; // this will default to  0.5

    size_t n_accept_draws; // will be returned by the algorithm
};

// HMC

struct hmc_settings_t
{
    size_t n_burnin_draws = 1E03;
    size_t n_keep_draws = 1E03;

    int omp_n_threads = -1; // numbers of threads to use

    size_t n_leap_steps = 1; // number of leap frog steps
    fp_t step_size = 1.0;
    Mat_t precond_mat;

    size_t n_accept_draws; // will be returned by the function
};

// ---- AI4BayesCode fork (2026-07-25): metric-kind dispatch ----------------
// Upstream mcmclib treats every preconditioner as a general dense positive-
// definite matrix and dispatches every leapfrog drift + kinetic-energy
// evaluation through a dense (BMO_MATOPS_INV(M) * v) matvec, which is O(n^2)
// per leap and dominates the cost for large n. This fork tags the metric so
// nuts() can route the identity + diagonal cases through their exact-arithmetic
// O(n) reductions:
//   IDENTITY : inv(I) * v == v                 (a raw daxpy)
//   DIAGONAL : inv(diag(d)) * v == (1/d) % v   (arma element-wise product)
//   DENSE    : inv(M) * v                      (unchanged upstream path)
// AUTO auto-detects from precond_mat contents on the first nuts() call and is
// the default (byte-identical dispatch to the upstream behaviour it replaces).
// See `nuts_metric_state_t` in nuts.hpp for the internal state used by the
// leap_frog / build_tree dispatch.
// If rebasing onto upstream mcmclib, re-apply this enum + the fields below.
// --------------------------------------------------------------------------
enum class metric_kind_t {
    AUTO     = 0,  // detect from precond_mat at first call (default)
    IDENTITY = 1,
    DIAGONAL = 2,
    DENSE    = 3
};

// NUTS

struct nuts_settings_t
{
    size_t n_burnin_draws = 1E03;
    size_t n_keep_draws = 1E03;

    int omp_n_threads = -1; // numbers of threads to use

    size_t n_adapt_draws = 1E03;
    fp_t target_accept_rate = 0.55;

    size_t max_tree_depth = size_t(10);

    fp_t step_size = 1.0; // \bar{\epsilon}_0
    fp_t gamma_val = 0.05;
    fp_t t0_val = 10;
    fp_t kappa_val = 0.75;
    Mat_t precond_mat;

    size_t n_accept_draws; // will be returned by the function

    // Persistent dual averaging state for continuous adaptation across calls.
    // If use_persistent_adapt is true, nuts() will load/save these fields
    // instead of resetting dual averaging at each call.
    bool use_persistent_adapt = false;
    fp_t epsilon_bar_persist = 1.0;
    fp_t h_val_persist = 0.0;
    fp_t mu_val_persist = 0.0;  // log(10 * step_size_init)
    size_t adapt_iter_persist = 0;  // cumulative iteration count across calls

    // ---- AI4BayesCode fork (2026-07-25): metric + preconditioner cache ----
    // Upstream mcmclib rebuilds inv(precond_mat) + chol_lower(precond_mat) on
    // every nuts() call (O(n^3)). This fork memoizes those derivatives on the
    // caller's settings so a stateful sampler that calls nuts() many times
    // (nuts_block / joint_nuts_block) pays the O(n^3) setup once per metric
    // change, not once per step(). Correctness: the cache is a pure function
    // of (precond_mat, resolved metric_kind, n_vals); user code that mutates
    // precond_mat MUST set precond_cache_valid = false to trigger a rebuild.
    // If rebasing onto upstream mcmclib, re-apply this block.
    metric_kind_t metric_kind          = metric_kind_t::AUTO;
    bool          precond_cache_valid  = false;
    metric_kind_t precond_cache_kind   = metric_kind_t::AUTO;
    size_t        precond_cache_n_vals = 0;
    Mat_t         precond_inv_cache;   // used only when precond_cache_kind==DENSE
    Mat_t         precond_sqrt_cache;  // used only when precond_cache_kind==DENSE
    ColVec_t      precond_inv_diag;    // used only when precond_cache_kind==DIAGONAL
    ColVec_t      precond_sqrt_diag;   // used only when precond_cache_kind==DIAGONAL
    // ----------------------------------------------------------------------
};

// RM-HMC

struct rmhmc_settings_t
{
    size_t n_burnin_draws = 1E03;
    size_t n_keep_draws = 1E03;

    int omp_n_threads = -1; // numbers of threads to use

    size_t n_leap_steps = 1; // number of leap frog steps
    fp_t step_size = 1.0;
    Mat_t precond_mat;

    size_t n_fp_steps = 5; // number of fixed point iteration steps

    size_t n_accept_draws; // will be returned by the function
};

// MALA

struct mala_settings_t
{
    size_t n_burnin_draws = 1E03;
    size_t n_keep_draws = 1E03;

    int omp_n_threads = -1; // numbers of threads to use

    fp_t step_size = 1.0;
    Mat_t precond_mat;

    size_t n_accept_draws; // will be returned by the function
};

// RWMH

struct rwmh_settings_t
{
    size_t n_burnin_draws = 1E03;
    size_t n_keep_draws = 1E03;

    int omp_n_threads = -1; // numbers of threads to use

    fp_t par_scale = 1.0;
    Mat_t cov_mat;

    size_t n_accept_draws; // will be returned by the function
};

struct algo_settings_t
{
    // RNG seeding

    size_t rng_seed_value = std::random_device{}();

    // bounds 
    
    bool vals_bound = false;

    ColVec_t lower_bounds;
    ColVec_t upper_bounds;

    // AEES
    aees_settings_t aees_settings;

    // DE
    de_settings_t de_settings;

    // HMC
    hmc_settings_t hmc_settings;

    // NUTS
    nuts_settings_t nuts_settings;

    // RM-HMC
    rmhmc_settings_t rmhmc_settings;

    // MALA
    mala_settings_t mala_settings;

    // RWMH
    rwmh_settings_t rwmh_settings;
};

#endif