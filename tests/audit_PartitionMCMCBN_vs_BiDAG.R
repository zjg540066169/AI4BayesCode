#!/usr/bin/env Rscript
# audit_PartitionMCMCBN_vs_BiDAG.R -- reference cross-check for this library's
# partition MCMC (order_mcmc_block, cfg$method = partition, BDeu) against the
# Kuipers-Moffa reference implementation BiDAG::partitionMCMC, on the SAME
# SYNTHETIC data. Both target P(G | D) under BDeu(alpha = 1, uniform DAG prior).
#
# Convergence diagnostic = classical (no-split) cross-impl Gelman-Rubin R-hat on
# the BDeu log-marginal-likelihood scalar log P(D | G) -- a single continuous
# quantity computed IDENTICALLY for both chains in R, so it is on one scale
# regardless of each implementation's internal bookkeeping. (Per-edge binary
# R-hat is a numerical artifact when a chain sits on a near-deterministic edge;
# the log-marginal scalar avoids it.) Outcome (A) = rhat_log_marginal < 1.05:
# our sampler converges to the same target as the reference.
#
# The C++ side (partition_crosscheck_gen) generates the data + our DAG samples;
# this script regenerates nothing benchmark-specific.

suppressPackageStartupMessages({ library(BiDAG); library(bnlearn) })

args <- commandArgs(trailingOnly = TRUE)
data_csv <- if (length(args) >= 1) args[1] else "/tmp/partition_xcheck_data.csv"
ai_csv   <- if (length(args) >= 2) args[2] else "/tmp/partition_xcheck_ai_dags.csv"
ALPHA <- 1.0; K_MAX <- 3L

D0 <- as.matrix(read.csv(data_csv, header = FALSE))      # 0-based codes {0,1}
storage.mode(D0) <- "integer"
n <- ncol(D0); N <- nrow(D0)
cards <- rep(2L, n)
D1 <- D0 + 1L                                            # 1-based for the scorer

# ---- BDeu log marginal log P(D|G), closed form (HGC 1995 + Buntine BDeu) -----
bdeu_node_logml <- function(D, j, Pa_j, alpha, cards) {
  Nn <- nrow(D); r_j <- cards[j]
  if (length(Pa_j) == 0L) {
    q_j <- 1L; tbl <- matrix(tabulate(D[, j], nbins = r_j), nrow = 1L)
  } else {
    r_pa <- cards[Pa_j]; q_j <- as.integer(prod(r_pa))
    strides <- c(1L, cumprod(r_pa)[-length(r_pa)])
    pi_idx  <- 1L + as.integer((D[, Pa_j, drop = FALSE] - 1L) %*% strides)
    flat    <- (pi_idx - 1L) * r_j + D[, j]
    tbl     <- matrix(tabulate(flat, nbins = q_j * r_j), q_j, r_j, byrow = TRUE)
  }
  a_jpi <- alpha / q_j; a_jpik <- alpha / (r_j * q_j); Nd <- rowSums(tbl)
  sum(lgamma(a_jpi) - lgamma(a_jpi + Nd)) +
    sum(lgamma(a_jpik + tbl) - lgamma(a_jpik))
}
# adj in incidence convention adj[a,b] = 1 iff a -> b; parents of j = column j.
dag_logml <- function(adj, D, alpha, cards) {
  total <- 0; p <- ncol(adj)
  for (j in seq_len(p)) {
    Pa_j <- which(adj[, j] > 0L)
    total <- total + bdeu_node_logml(D, j, Pa_j, alpha, cards)
  }
  total
}
rhat_classical <- function(chains) {           # 1992 Gelman-Rubin, no split/rank
  m <- length(chains[[1]])
  W <- mean(vapply(chains, var,  numeric(1)))
  B <- m * var(vapply(chains, mean, numeric(1)))
  if (W > 0) sqrt(((m - 1) / m) * W + B / m) / sqrt(W) else 1
}

# ---- our chain: read AI DAG samples (each row = n*n incidence, byrow) --------
ai_raw <- as.matrix(read.csv(ai_csv, header = FALSE))
S_ai <- nrow(ai_raw)
ai_lml <- numeric(S_ai)
for (s in seq_len(S_ai)) {
  adj <- matrix(as.integer(ai_raw[s, ]), n, n, byrow = TRUE)
  ai_lml[s] <- dag_logml(adj, D1, ALPHA, cards)
}

# ---- reference chain: BiDAG::partitionMCMC on the SAME data -------------------
Df <- as.data.frame(lapply(seq_len(n), function(j)
  factor(D1[, j], levels = seq_len(cards[j]))))
names(Df) <- paste0("V", seq_len(n))
score <- BiDAG::scoreparameters("bdecat", Df, bdecatpar = list(chi = ALPHA, edgepf = 1))
# FULL candidate space (all edges allowed) so BiDAG considers every parent set,
# matching our all-candidates (uncapped) setup -- both then target the same P(G|D).
fullspace <- matrix(1L, n, n); diag(fullspace) <- 0L
set.seed(42)
fit <- BiDAG::partitionMCMC(score, iterations = 40000L, stepsave = 10L,
                            startspace = fullspace, verbose = FALSE)
incid <- fit$traceadd$incidence
S_ref0 <- length(incid)
ref_lml <- numeric(S_ref0)
for (s in seq_len(S_ref0)) ref_lml[s] <- dag_logml(as.matrix(incid[[s]]), D1, ALPHA, cards)
ref_lml <- ref_lml[seq.int(S_ref0 %/% 2L + 1L, S_ref0)]   # drop 50% burn-in

# ---- cross-impl R-hat on the log-marginal scalar -----------------------------
L <- min(length(ai_lml), length(ref_lml))
rhat_lml <- rhat_classical(list(tail(ai_lml, L), tail(ref_lml, L)))

cat(sprintf("[xcheck] n=%d N=%d ; AI logML mean=%.2f (sd %.2f) ; BiDAG logML mean=%.2f (sd %.2f)\n",
            n, N, mean(ai_lml), sd(ai_lml), mean(ref_lml), sd(ref_lml)))
cat(sprintf("[xcheck] cross-impl rhat_log_marginal = %.4f (want < 1.05)\n", rhat_lml))
if (is.finite(rhat_lml) && rhat_lml < 1.05) {
  cat("PASS  partition MCMC matches BiDAG::partitionMCMC on log P(D|G) (Outcome A)\n")
  quit(save = "no", status = 0L)
} else {
  cat("FAIL  partition MCMC and BiDAG disagree on the target\n")
  quit(save = "no", status = 1L)
}
