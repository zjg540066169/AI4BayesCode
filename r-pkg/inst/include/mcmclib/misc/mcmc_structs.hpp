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
  ##   Modifications Copyright (C) 2025 Jungang Zou
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
  ##     - SAFE-SPEEDUP (2026-07-26, JZ): Added precond_mat setup-cache fields
  ##       (precond_cache_valid, precond_inv_cache, precond_sqrt_cache,
  ##        precond_cache_n_vals) to nuts_settings_t. See nuts.hpp Fix #2 and
  ##       joint_nuts_block.hpp mutation-site invalidation for the contract.
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

    // -------------------------------------------------------------------
    // FORK MARKER (2026-07-26, JZ): SAFE-SPEEDUP SUBSET, Fix #2 setup cache.
    //
    // Cache the precomputed inverse and lower-Cholesky of precond_mat across
    // successive nuts() calls that reuse the SAME mass matrix (identity or
    // adapted). nuts_impl checks precond_cache_valid on entry; if true, reuses
    // the cached inv/sqrt; else rebuilds and stamps precond_cache_valid=true.
    //
    // CACHE COHERENCE CONTRACT: any code that MUTATES nuts_settings_t::precond_mat
    // (this file's field + all AI4BayesCode wiring) MUST set precond_cache_valid
    // = false before the next nuts() call. joint_nuts_block.hpp wires this at
    // the constructor, set_precond_matrix, set_adaptation, adapt_dense_metric_
    // install, three_phase Phase-I save/restore, three_phase Phase-II install,
    // and the escape-guard fallback.
    //
    // BYTE-PRESERVED to no-cache when precond_cache_valid == false at entry
    // (the rebuild path takes the same code — including the identical
    // BMO_MATOPS_INV / BMO_MATOPS_CHOL_LOWER of the same input).
    //
    // REBASE NOTE: If mcmclib upstream is bumped, keep this block AND the
    // matching read/write in nuts.hpp. The cache is a pure optimization; if
    // ever in doubt, set precond_cache_valid = false unconditionally in the
    // AI4BayesCode wiring and correctness is preserved.
    // -------------------------------------------------------------------
    bool   precond_cache_valid  = false;
    Mat_t  precond_inv_cache;
    Mat_t  precond_sqrt_cache;
    size_t precond_cache_n_vals = 0;
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