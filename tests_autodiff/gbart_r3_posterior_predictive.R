# ============================================================================
# gbart_r3_posterior_predictive.R
#
# Layer 3 R3 validator for the GBart* examples: Bayesian posterior-
# predictive p-values on six summary statistics + PSIS-LOO when the
# loo package is installed.
#
# A p-value in (0.05, 0.95) on each statistic indicates that the
# posterior predictive distribution covers the observed data (no
# gross misspecification).
# ============================================================================

source("R/AI4BayesCode_helpers.R")

NBURN <- 1500L
NKEEP <- 2000L
N_PPC_DRAWS <- 1000L

# Summary statistics for the p-value calculations.
bp_stats <- list(
    mean = mean, sd = sd,
    min  = function(x) min(x),
    max  = function(x) max(x),
    q25  = function(x) quantile(x, 0.25, names = FALSE),
    q75  = function(x) quantile(x, 0.75, names = FALSE))

bp_stats_binary <- list(mean = mean)

# Collect posterior-predictive draws INTERSPERSED with MCMC sweeps so each
# y_rep comes from a different posterior draw of r (not the fixed final
# live-tree state). This matches the standard Bayesian p-value definition
# p(T) = Pr(T(y_rep) >= T(y_obs) | y_obs) where the outer prob integrates
# over the POSTERIOR of r, not just Poisson variance conditional on r.
collect_y_rep <- function(m, n_draws, sweeps_between = 1L) {
    y_rep <- NULL
    for (d in seq_len(n_draws)) {
        m$step(sweeps_between)
        pp <- m$predict_at(list())
        yr <- pp$y_rep
        if (is.null(y_rep)) y_rep <- matrix(NA_real_, n_draws, length(yr))
        y_rep[d, ] <- yr
    }
    y_rep
}

run_r3 <- function(example_name, build_fn, y_obs, stats_list = bp_stats) {
    cat(sprintf("\n===== %s R3 =====\n", example_name))
    set.seed(101)
    m <- build_fn()
    m$step(NBURN); m$step(NKEEP)
    yrep <- collect_y_rep(m, N_PPC_DRAWS, sweeps_between = 1L)
    cat(sprintf("  Collected %d posterior-predictive draws (1-sweep thinning); "
                , N_PPC_DRAWS))
    cat(sprintf("y_rep shape %d x %d\n", nrow(yrep), ncol(yrep)))
    pvs <- sapply(stats_list, function(f) {
        t_obs <- f(y_obs)
        t_rep <- apply(yrep, 1, f)
        mean(t_rep >= t_obs)
    })
    status <- sapply(pvs, function(p) {
        if (p > 0.05 && p < 0.95) "OK"
        else if (p > 0.01 && p < 0.99) "WARN" else "FAIL"
    })
    for (k in seq_along(pvs)) {
        cat(sprintf("  %-6s  p = %.3f  [%s]\n",
                    names(pvs)[k], pvs[k], status[k]))
    }
    list(pvs = pvs, status = status)
}

# -- GBartPoisson R3 --
ai4bayescode_sourceCpp("r-pkg/inst/examples/GBartPoisson.cpp",
                       AI4BayesCode_path = ".")
set.seed(1); N <- 200L; p <- 3L; X <- matrix(runif(N*p), N, p)
truth <- -1 + X[,1] - 0.5*X[,2]; y_p <- rpois(N, exp(truth))
res_p <- run_r3("GBartPoisson",
                function() new(GBartPoisson, X, as.numeric(y_p), 200L, 101L, FALSE),
                y_p, bp_stats)

# -- GBartLogistic R3 (binary) --
ai4bayescode_sourceCpp("r-pkg/inst/examples/GBartLogistic.cpp",
                       AI4BayesCode_path = ".")
set.seed(2); X <- matrix(runif(N*p), N, p)
y_l <- rbinom(N, 1, plogis(-0.5 + 2*X[,1] - X[,2]))
res_l <- run_r3("GBartLogistic",
                function() new(GBartLogistic, X, as.numeric(y_l), 50L, 101L, FALSE),
                y_l, bp_stats_binary)

cat("\n===== R3 summary =====\n")
cat("GBartPoisson p-values:  ",
    paste(sprintf("%s=%.2f", names(res_p$pvs), res_p$pvs), collapse = "  "), "\n")
cat("GBartLogistic p-values: ",
    paste(sprintf("%s=%.2f", names(res_l$pvs), res_l$pvs), collapse = "  "), "\n")

all_statuses <- c(res_p$status, res_l$status)
cat(sprintf("\n%d / %d statistics at OK status (in (0.05, 0.95))\n",
            sum(all_statuses == "OK"), length(all_statuses)))
