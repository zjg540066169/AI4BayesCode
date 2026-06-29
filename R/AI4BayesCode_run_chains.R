# ----------------------------------------------------------------------------
# AI4BayesCode_run_chains.R (T13)
#
# R-level multi-chain runner for any AI4BayesCode wrapper class. Spawns
# n_chains chains (parallel or sequential), runs warmup + keep on each,
# and returns the list of get_history() results plus (optionally) the
# combined posterior::rhat / ess summaries.
#
# Usage
# -----
#   source("path/to/AI4BayesCode_run_chains.R")
#   model_ctor <- function(seed) {
#       new(SpikeSlabRJMCMC, X, y, a_pi, b_pi, as.integer(seed), TRUE)
#   }
#   results <- AI4BayesCode_run_chains(model_ctor, n_chains = 4,
#                                    n_burn = 2000, n_keep = 10000,
#                                    parallel = TRUE, seeds = c(101, 202, 303, 404))
#   # results$histories : list of n_chains $get_history() returns
#   # results$seeds     : the seeds used
#   # results$wall      : wall time per chain (seconds)
#
# Parallel mode uses parallel::mclapply on POSIX (Linux/macOS). On
# Windows it silently falls back to sequential. Pass parallel=FALSE to
# force sequential anywhere.
#
# NOTE: each chain instantiates a NEW Rcpp module object in its own
# worker, so Rcpp object copying / ownership is handled cleanly. The
# wrapper constructor MUST be deterministic in `seed` — i.e., given
# (seed, other args), the wrapper produces identical state.
# ----------------------------------------------------------------------------

AI4BayesCode_run_chains <- function(model_ctor,
                                 n_chains  = 4,
                                 n_burn    = 2000,
                                 n_keep    = 10000,
                                 seeds     = NULL,
                                 parallel  = TRUE,
                                 mc.cores  = NULL,
                                 verbose   = TRUE) {
    stopifnot(is.function(model_ctor),
              n_chains >= 1, n_burn >= 0, n_keep >= 1)

    if (is.null(seeds)) {
        seeds <- as.integer(101L + (seq_len(n_chains) - 1L) * 101L)
    }
    if (length(seeds) != n_chains) {
        stop("length(seeds) must equal n_chains")
    }

    one_chain <- function(seed_val) {
        t0 <- Sys.time()
        m <- model_ctor(seed = seed_val)
        m$step(as.integer(n_burn))
        m$step(as.integer(n_keep))
        t1 <- Sys.time()
        list(history = m$get_history(),
             seed    = seed_val,
             wall    = as.numeric(difftime(t1, t0, units = "secs")))
    }

    use_parallel <- parallel && .Platform$OS.type != "windows" &&
                    requireNamespace("parallel", quietly = TRUE)

    if (is.null(mc.cores)) {
        mc.cores <- if (use_parallel)
            min(n_chains, max(1, parallel::detectCores() - 1))
        else 1
    }

    if (use_parallel && mc.cores > 1) {
        if (verbose)
            message("AI4BayesCode_run_chains: running ", n_chains,
                    " chains on ", mc.cores, " cores (parallel)")
        results <- parallel::mclapply(seeds, one_chain, mc.cores = mc.cores,
                                      mc.set.seed = TRUE)
    } else {
        if (verbose)
            message("AI4BayesCode_run_chains: running ", n_chains,
                    " chains sequentially")
        results <- lapply(seeds, one_chain)
    }

    # Detect failures (mclapply may return try-error on child crash).
    for (i in seq_along(results)) {
        if (inherits(results[[i]], "try-error") ||
            is.null(results[[i]]$history)) {
            stop("AI4BayesCode_run_chains: chain ", i, " failed")
        }
    }

    list(
        histories = lapply(results, `[[`, "history"),
        seeds     = sapply(results, `[[`, "seed"),
        wall      = sapply(results, `[[`, "wall")
    )
}

# ----------------------------------------------------------------------------
# Helper: compute R-hat / ESS for scalar-history keys across chains.
# Requires `posterior` package.
# ----------------------------------------------------------------------------
AI4BayesCode_rhat_summary <- function(run, keys = NULL, drop_burn = 0) {
    if (!requireNamespace("posterior", quietly = TRUE)) {
        stop("posterior package required for R-hat summary")
    }
    histories <- run$histories
    if (length(histories) < 2L) {
        stop("need at least 2 chains for R-hat")
    }
    all_keys <- names(histories[[1]])
    if (is.null(keys)) keys <- all_keys

    out <- list()
    for (k in keys) {
        if (!k %in% all_keys) next
        vals <- lapply(histories, function(h) h[[k]])
        if (is.null(dim(vals[[1]]))) {
            # Scalar history per chain.
            n <- length(vals[[1]])
            if (drop_burn > 0 && drop_burn < n)
                vals <- lapply(vals, function(v) v[(drop_burn + 1):n])
            arr <- array(unlist(vals),
                          dim = c(length(vals[[1]]), length(vals), 1))
            rh <- tryCatch(posterior::rhat(arr[,,1]), error = function(e) NA)
            eb <- tryCatch(posterior::ess_bulk(arr[,,1]),
                           error = function(e) NA)
            out[[k]] <- list(rhat = rh, ess_bulk = eb)
        } else {
            # Matrix history (n_draws × dim per chain).
            p <- ncol(vals[[1]])
            n <- nrow(vals[[1]])
            if (drop_burn > 0 && drop_burn < n)
                vals <- lapply(vals, function(m) m[(drop_burn + 1):n, , drop=FALSE])
            rhats <- numeric(p); ess   <- numeric(p)
            for (j in seq_len(p)) {
                per_chain <- lapply(vals, function(m) m[, j])
                arr_j <- array(unlist(per_chain),
                               dim = c(length(per_chain[[1]]),
                                       length(per_chain), 1))
                rhats[j] <- tryCatch(posterior::rhat(arr_j[,,1]),
                                      error = function(e) NA)
                ess[j]   <- tryCatch(posterior::ess_bulk(arr_j[,,1]),
                                      error = function(e) NA)
            }
            out[[k]] <- list(rhat = rhats, ess_bulk = ess,
                              max_rhat = max(rhats, na.rm = TRUE),
                              min_ess  = min(ess,   na.rm = TRUE))
        }
    }
    out
}
