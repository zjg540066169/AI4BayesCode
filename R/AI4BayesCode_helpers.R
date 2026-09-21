# ----------------------------------------------------------------------------
# AI4BayesCode_helpers.R
#
# CHECKOUT-ONLY helpers. Use these ONLY when the R package is not installed.
#
# If you installed the package --
#     remotes::install_github("zjg540066169/AI4BayesCode", subdir = "r-pkg")
# -- then do NOT source this file. The package ships its own headers, so
#
#     library(AI4BayesCode)
#     ai4bayescode_source("./MyModel.cpp")
#
# works from any directory with no path to configure. Sourcing this file on
# top of the installed package used to define a same-named compiler with a
# different signature that required a local ./AI4BayesCode tree; R masks
# silently, so the packaged call started failing with
# "path[1]=\"./AI4BayesCode\": No such file or directory". The compiler here
# is now named ai4bayescode_source_checkout() so it can never shadow the
# packaged API.
#
# User-facing convenience helpers for using AI4BayesCode from an R script.
# Source this file once at the top of your analysis script and then call
# ai4bayescode_source_checkout("MyModel.cpp", AI4BayesCode_path = "./AI4BayesCode").
#
# Why this file exists
# --------------------
# Rcpp's sourceCpp does not honor Sys.setenv("PKG_CPPFLAGS" = ...) directly
# (at least through Rcpp 1.1.1). The canonical way to add custom include
# paths to a sourceCpp compile is via a Makevars file pointed at by the
# R_MAKEVARS_USER environment variable. Doing that by hand is clunky
# (temp files, absolute paths, env var juggling), so this helper wraps
# the whole dance into a single call.
#
# Usage
# -----
#     source("path/to/AI4BayesCode/R/AI4BayesCode_helpers.R")
#     ai4bayescode_source_checkout("MyModel.cpp", AI4BayesCode_path = "./AI4BayesCode")
#     model <- new(MyModel, ...)
#
# Requirements
# ------------
#     - R with Rcpp and RcppArmadillo installed
#     - The AI4BayesCode folder reachable on disk at `AI4BayesCode_path`
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Pin the native BLAS to one thread, as early as possible.
#
# ai4bayescode_run_chains() parallelises with parallel::mclapply(), i.e.
# fork(). A multithreaded BLAS -- macOS Accelerate/vecLib above all, but also
# OpenBLAS with pthreads and MKL -- keeps a worker-thread pool; fork() copies
# only the calling thread, so the child inherits a pool whose threads do not
# exist and the first heavy BLAS call there dies. On macOS the crash is inside
# libdispatch (`_dispatch_root_queue_push` under a LAPACK entry point,
# reported as "crashed on child side of fork pre-exec"), and it aborts the
# whole R session -- it cannot be caught and retried.
#
# The cap has to be in place BEFORE the first BLAS call, because that is when
# Accelerate decides whether to use its dispatch queues. Setting it later --
# inside ai4bayescode_run_chains(), say -- has no effect whatsoever. Hence:
# here, at load.
#
# Only variables the caller has not already set are touched.
#
# The cap costs nothing in practice: the chains are separate processes
# already, so mc.cores workers x k BLAS threads each would only oversubscribe
# the machine. Measured on the GP examples, per-chain wall time is the same or
# slightly better at one thread, and the parallel speedup is a clean 4.2x on
# four cores.
.ai4b_pin_blas_threads <- function() {
    vars <- c("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS",
              "OMP_NUM_THREADS", "MKL_NUM_THREADS")
    unset <- vars[is.na(Sys.getenv(vars, unset = NA))]
    if (length(unset)) {
        do.call(Sys.setenv, as.list(stats::setNames(rep("1", length(unset)), unset)))
    }
    invisible(NULL)
}
.ai4b_pin_blas_threads()

ai4bayescode_source_checkout <- function(cpp_file,
                                AI4BayesCode_path = "./AI4BayesCode",
                                rebuild        = TRUE,
                                verbose        = FALSE,
                                extra_cppflags = character(),
                                extra_libs     = character()) {
    if (!requireNamespace("Rcpp", quietly = TRUE)) {
        stop("Rcpp is required but not installed.")
    }
    if (!requireNamespace("RcppArmadillo", quietly = TRUE)) {
        stop("RcppArmadillo is required but not installed.")
    }

    cpp_file <- normalizePath(cpp_file, mustWork = TRUE)
    pkg_root <- normalizePath(AI4BayesCode_path, mustWork = TRUE)

    inc_root  <- file.path(pkg_root, "include")
    inc_mc    <- file.path(pkg_root, "include", "mcmclib")
    inc_bmo   <- file.path(pkg_root, "include", "mcmclib",
                           "BaseMatrixOps", "include")
    # autodiff is at inc_root/autodiff/ -> users include as
    # `<autodiff/reverse/var.hpp>`. No separate -I needed (inc_root covers it).
    # Eigen is at inc_root/eigen/Eigen/ -> needs a dedicated -I for
    # `<Eigen/Dense>` to resolve.
    inc_eigen <- file.path(pkg_root, "include", "eigen")
    # libgp kernel subsystem (vendored, BSD-3) lives at pkg_root/libgp_kernels/.
    # Its internal headers use bare `#include "cov.h"` etc., so we add the
    # directory directly to the include path.
    inc_libgp <- file.path(pkg_root, "libgp_kernels")
    # celerite (vendored, MIT) for O(N) 1-D time-series GP. Header-only
    # under pkg_root/celerite/include/. Headers include as
    # `<celerite/celerite.h>` so we add pkg_root/celerite/include to -I.
    inc_celerite <- file.path(pkg_root, "celerite", "include")
    for (p in c(inc_root, inc_mc, inc_bmo)) {
        if (!dir.exists(p)) {
            stop("Could not find expected AI4BayesCode directory: ", p,
                 "\nDid you pass the right AI4BayesCode_path?")
        }
    }

    cppflags <- c(
        paste0("-I", shQuote(inc_root)),
        paste0("-I", shQuote(inc_mc)),
        paste0("-I", shQuote(inc_bmo)),
        if (dir.exists(inc_eigen)) paste0("-I", shQuote(inc_eigen)) else character(),
        if (dir.exists(inc_libgp)) paste0("-I", shQuote(inc_libgp)) else character(),
        if (dir.exists(inc_celerite)) paste0("-I", shQuote(inc_celerite)) else character(),
        "-DMCMC_ENABLE_ARMA_WRAPPERS",
        # Trailing "=" makes this an EMPTY replacement, matching
        # RcppArmadillo's own `#define ARMA_DONT_USE_WRAPPER`. A bare -D
        # expands to "1", a different token sequence, and every single user
        # compile then opens with a macro-redefinition warning.
        "-DARMA_DONT_USE_WRAPPER=",
        "-DAI4BAYESCODE_RCPP_MODULE",
        extra_cppflags
    )

    libs <- if (Sys.info()[["sysname"]] == "Darwin") {
        c("-framework Accelerate", extra_libs)
    } else {
        c("$(BLAS_LIBS)", "$(LAPACK_LIBS)", extra_libs)
    }

    mk_file <- tempfile(pattern = "AI4BayesCode_Makevars_", fileext = "")
    writeLines(c(
        paste("PKG_CPPFLAGS =", paste(cppflags, collapse = " ")),
        paste("PKG_LIBS =",     paste(libs,     collapse = " "))
    ), mk_file)
    on.exit(unlink(mk_file), add = TRUE)

    old_makevars <- Sys.getenv("R_MAKEVARS_USER", unset = NA)
    Sys.setenv(R_MAKEVARS_USER = mk_file)
    on.exit({
        if (is.na(old_makevars)) {
            Sys.unsetenv("R_MAKEVARS_USER")
        } else {
            Sys.setenv(R_MAKEVARS_USER = old_makevars)
        }
    }, add = TRUE)

    Rcpp::sourceCpp(cpp_file,
                    rebuild = rebuild,
                    verbose = verbose,
                    env     = globalenv())
    # Give every exposed class a name-matching constructor, so that
    # new(<Class>, y = y, rng_seed = 7L) means what it reads as and every
    # argument carrying a C++ default may simply be omitted.
    tryCatch(.ai4b_install_named_ctors(cpp_file, globalenv()),
             error = function(e) character(0))
    invisible(NULL)
}

# ----------------------------------------------------------------------------
# ai4bayescode_plot_dag -- render the model prediction DAG.
#
# Default behavior: draws to the current R graphics device (screen /
# RStudio plot pane). This is what end users almost always want.
# Pass `out_path = "my_dag.png"` to write a PNG instead.
#
# Features:
#   1. Sugiyama hierarchical layout (DAG-aware; minimizes edge crossings)
#   2. Three node categories, each with a distinct color/shape:
#        data input       (green  circle) -- declared via declare_data_input
#        sampled param    (blue   circle) -- root non-data node (has a prior,
#                                            sampled by MCMC)
#        predictive value (orange circle) -- any downstream node (incoming
#                                            predict edges), covers both
#                                            deterministic transforms
#                                            (linear predictor, f(X), f_bart)
#                                            and stochastic posterior-
#                                            predictive draws (y_rep)
#   3. Plate detection: names sharing a `<prefix>_<digit>+` suffix collapse
#        into a single node labeled `<prefix>_i [n=K]`
#
# Usage:
#     ai4bayescode_plot_dag(model)                                   # interactive (default)
#     ai4bayescode_plot_dag(model, out_path = "generated/my_dag.png") # PNG to file
#     ai4bayescode_plot_dag(model, width = 1800, height = 1200, res = 160)
#     ai4bayescode_plot_dag(model, plate = FALSE)                    # disable plate collapse
#
# Returns: invisibly NULL (interactive) or the PNG path (if out_path given).
# ----------------------------------------------------------------------------

ai4bayescode_plot_dag <- function(model,
                     out_path = NULL,
                     main     = NULL,
                     width    = 1600,
                     height   = 1100,
                     res      = 150,
                     plate    = TRUE,
                     ...) {
    if (!requireNamespace("igraph", quietly = TRUE)) {
        stop("ai4bayescode_plot_dag requires the igraph package.\n",
             "Install with: install.packages('igraph')")
    }

    dag <- model$get_dag()

    # --- Back-compat: legacy flat form (predict-only, no named sub-lists) ---
    if (is.null(dag$gibbs_reads) && is.null(dag$predict_edges)) {
        predict_edges_only <- dag[setdiff(names(dag), "_data_inputs")]
        dag <- list(
            gibbs_reads       = list(),
            gibbs_invalidates = list(),
            predict_edges     = predict_edges_only,
            context_edges     = list(),
            data_inputs       = dag[["_data_inputs"]])
    }

    data_inputs <- as.character(dag$data_inputs)

    # --- Collect edges (by type) --------------------------------------------
    collect_edges <- function(spec, type) {
        rows <- list()
        for (from in names(spec)) {
            to_nodes <- as.character(spec[[from]])
            if (length(to_nodes) == 0) next
            rows[[length(rows) + 1L]] <- data.frame(
                from = rep(from, length(to_nodes)),
                to   = to_nodes,
                type = type,
                stringsAsFactors = FALSE)
        }
        if (length(rows) == 0) {
            data.frame(from = character(0), to = character(0),
                       type = character(0), stringsAsFactors = FALSE)
        } else do.call(rbind, rows)
    }

    # PREDICTION-ONLY DAG -- Gibbs / refresh edges intentionally hidden.
    # Per codegen.md Sec.2(b): user-facing DAG visualization shows the
    # generative / causal data flow, not the internal Gibbs full-
    # conditional graph. Conflating the two is a common source of
    # confusion. If you need to debug Gibbs dependencies, call
    # `model$get_dag()$gibbs_reads` directly.
    gibbs_reads_edges <- data.frame(from = character(0),
                                     to   = character(0),
                                     type = character(0),
                                     stringsAsFactors = FALSE)
    refresh_edges    <- data.frame(from = character(0),
                                    to   = character(0),
                                    type = character(0),
                                    stringsAsFactors = FALSE)
    predict_edges <- collect_edges(dag$predict_edges, "predict")
    # VIZ-ONLY prior / hyperprior context (faded). These edges are NEVER
    # traversed by predict_at; they exist so the rendered DAG shows the
    # generative origin of each sampled-parameter root (priors / hyper-
    # priors, BART tree prior, etc.). See shared_data::declare_context_edges.
    context_edges <- collect_edges(dag$context_edges, "context")

    all_edges <- rbind(gibbs_reads_edges, refresh_edges,
                       predict_edges, context_edges)
    all_nodes <- unique(c(all_edges$from, all_edges$to, data_inputs))

    if (length(all_nodes) == 0) {
        if (is.null(out_path)) {
            plot.new()
            title(main = if (is.null(main)) "Model DAG (empty)" else main)
            text(0.5, 0.5, "No nodes declared", cex = 1.2, col = "grey50")
        }
        return(invisible(NULL))
    }

    # --- Plate detection: collapse `prefix_0, prefix_1, ...` into one node --
    plate_map <- NULL
    plate_sizes <- integer(0)
    if (isTRUE(plate)) {
        # strip trailing _<digits> from names
        plate_re <- "^(.*)_([0-9]+)$"
        with_suffix <- grepl(plate_re, all_nodes)
        if (any(with_suffix)) {
            prefixes <- sub(plate_re, "\\1", all_nodes[with_suffix])
            tab <- table(prefixes)
            repeated <- names(tab)[tab >= 2]
            if (length(repeated) > 0) {
                plate_map <- setNames(rep(NA_character_, length(all_nodes)),
                                      all_nodes)
                for (p in repeated) plate_sizes[p] <- as.integer(tab[p])
                for (i in seq_along(all_nodes)) {
                    nm <- all_nodes[i]
                    if (grepl(plate_re, nm)) {
                        pr <- sub(plate_re, "\\1", nm)
                        if (pr %in% repeated) plate_map[nm] <- pr
                    }
                }
            }
        }
    }

    # If plate detection grouped any nodes, rewrite edges + node list.
    if (!is.null(plate_map) && any(!is.na(plate_map))) {
        collapse <- function(nm) {
            pr <- plate_map[nm]
            if (is.na(pr)) nm else sprintf("%s_i [n=%d]", pr, plate_sizes[pr])
        }
        all_edges$from <- vapply(all_edges$from, collapse, character(1))
        all_edges$to   <- vapply(all_edges$to,   collapse, character(1))
        # drop self-loops created by the collapse and de-duplicate
        keep <- all_edges$from != all_edges$to
        all_edges <- unique(all_edges[keep, , drop = FALSE])
        # rebuild node list from the collapsed mapping
        collapse_vec <- vapply(all_nodes, collapse, character(1))
        all_nodes <- unique(c(collapse_vec,
                              all_edges$from, all_edges$to))
        # rewrite data_inputs by the same mapping
        data_inputs <- unique(vapply(data_inputs, collapse, character(1)))
    }

    g <- igraph::graph_from_data_frame(
        d        = all_edges,
        directed = TRUE,
        vertices = data.frame(name = all_nodes, stringsAsFactors = FALSE))

    # --- Edge aesthetics -----------------------------------------------------
    edge_colors <- c(gibbs = "#333333", refresh = "#8A2BE2",
                     predict = "#D62728", context = "#AAAAAA")
    edge_ltys   <- c(gibbs = 1,         refresh = 2,
                     predict = 1,        context = 2)
    igraph::E(g)$color <- edge_colors[igraph::E(g)$type]
    igraph::E(g)$lty   <- edge_ltys  [igraph::E(g)$type]

    # --- Classify nodes into 4 roles ----------------------------------------
    # Categories (graph-structural). The sampled-vs-predictive split is
    # computed from the PREDICT subgraph only -- context (prior) edges are
    # excluded so a sampled parameter with an incoming prior edge
    # (e.g. sigma_rw2 -> beta) is NOT mis-classified as predictive.
    #   data       -- explicitly declared via declare_data_input
    #   context    -- appears ONLY in context edges (prior / hyperprior /
    #                  BART tree prior); rendered faded, NOT traversed by
    #                  predict_at
    #   sampled    -- in the predict subgraph with predict-in-deg == 0:
    #                  a posterior-draw-fed parameter (may have incoming
    #                  context edges showing its prior)
    #   predictive -- predict-in-deg > 0: any predicted quantity, merging
    #                  deterministic transforms (f_bart, spline,
    #                  theta_mean) AND stochastic draws (theta_pred, y_rep)
    vnames <- igraph::V(g)$name
    predict_rows  <- all_edges[all_edges$type == "predict", , drop = FALSE]
    predict_nodes <- unique(c(predict_rows$from, predict_rows$to))
    predict_in    <- table(predict_rows$to)

    classify <- function(nm) {
        if (nm %in% data_inputs)        return("data")
        if (!(nm %in% predict_nodes))   return("context")
        cnt <- predict_in[nm]
        if (!is.na(cnt) && cnt > 0)     return("predictive")
        "sampled"
    }
    roles <- vapply(vnames, classify, character(1))

    color_for <- c(data       = "#9FDF9F",   # green
                   sampled    = "#A6CEE3",   # light blue
                   predictive = "#FDBF6F",   # orange
                   context    = "#DDDDDD")   # faded grey (prior context)
    shape_for <- c(data       = "circle",
                   sampled    = "circle",
                   predictive = "circle",
                   context    = "circle")

    node_col   <- color_for[roles]
    node_shape <- shape_for[roles]

    # --- Layout: rank by the PREDICT subgraph ONLY --------------------------
    # Context (prior / hyperprior) edges must NOT influence ranking: a long
    # prior chain (a_smooth -> log_sigma_rw2 -> beta -> spline) would push
    # its targets to deep ranks while a shallow predict node (f_bart) stays
    # near the top, making the long f_bart -> theta_mean edge visually
    # sweep across the unrelated `spline` node (looks like f_bart->spline).
    # We Sugiyama-rank the predict subgraph only, then drop each
    # context-only node in as a satellite just above its first target.
    pred_only_edges <- all_edges[all_edges$type == "predict", , drop = FALSE]
    g_layout <- igraph::graph_from_data_frame(
        d        = pred_only_edges[, c("from", "to"), drop = FALSE],
        directed = TRUE,
        vertices = data.frame(name = vnames, stringsAsFactors = FALSE))
    sug <- tryCatch(igraph::layout_with_sugiyama(g_layout, hgap = 1.5,
                                                  vgap = 1.2, maxiter = 200),
                    error = function(e) NULL)
    layout <- if (!is.null(sug)) sug$layout else
        tryCatch(
            igraph::layout_as_tree(
                g_layout,
                root = which(igraph::degree(g_layout, mode = "in") == 0),
                mode = "out"),
            error = function(e) igraph::layout_with_fr(g_layout))

    # Place context-only nodes in DEPTH-LAYERED shelves stacked ABOVE the
    # predict graph (Sugiyama gives roots high y, terminal y_rep low y).
    #   depth 1 = points directly at a predict node (sits one shelf up)
    #   depth d = points at a depth-(d-1) context node (d shelves up)
    # Shelves are spaced > node size so they never overlap the predict
    # path; siblings sharing an anchor are spread horizontally.
    if (nrow(context_edges) > 0L) {
        idx_of   <- setNames(seq_along(vnames), vnames)
        pred_in  <- intersect(predict_nodes, vnames)
        ctx_only <- intersect(setdiff(unique(context_edges$from),
                                       predict_nodes), vnames)

        if (length(ctx_only) > 0L && length(pred_in) > 0L) {
            ctx_tgts <- function(cn)
                intersect(context_edges$to[context_edges$from == cn], vnames)

            # --- context depth via fixpoint relaxation ---
            depth <- setNames(rep(NA_integer_, length(ctx_only)), ctx_only)
            for (it in seq_len(length(ctx_only) + 1L)) {
                changed <- FALSE
                for (cn in ctx_only) {
                    tg <- ctx_tgts(cn); cand <- integer(0)
                    for (t in tg) {
                        if (t %in% pred_in) cand <- c(cand, 1L)
                        else if (t %in% ctx_only && !is.na(depth[t]))
                            cand <- c(cand, depth[t] + 1L)
                    }
                    if (length(cand)) {
                        nd <- max(cand)
                        if (is.na(depth[cn]) || depth[cn] != nd) {
                            depth[cn] <- nd; changed <- TRUE
                        }
                    }
                }
                if (!changed) break
            }
            depth[is.na(depth)] <- 1L

            py        <- sort(unique(layout[idx_of[pred_in], 2]))
            rank_gap  <- if (length(py) >= 2L) stats::median(diff(py)) else 1.0
            row_gap   <- 1.6 * rank_gap
            pred_y_top<- max(layout[idx_of[pred_in], 2])
            px        <- layout[idx_of[pred_in], 1]
            x_unit    <- if (length(px) >= 2L)
                             max(rank_gap,
                                 diff(range(px)) / max(1L, length(pred_in)))
                         else rank_gap

            # place shallow shelves first (targets already positioned)
            for (d in sort(unique(depth))) {
                nodes_d <- names(depth)[depth == d]
                for (cn in nodes_d) {
                    tg <- ctx_tgts(cn)
                    layout[idx_of[cn], 1] <- mean(layout[idx_of[tg], 1])
                    layout[idx_of[cn], 2] <- pred_y_top + d * row_gap
                }
                # spread same-anchor siblings on this shelf
                axr <- round(layout[idx_of[nodes_d], 1], 6)
                for (av in unique(axr)) {
                    grp <- nodes_d[axr == av]
                    if (length(grp) >= 2L) {
                        offs <- seq(-1, 1, length.out = length(grp)) *
                                (0.95 * x_unit)
                        for (j in seq_along(grp))
                            layout[idx_of[grp[j]], 1] <- av + offs[j]
                    }
                }
            }
        }
    }

    if (is.null(main))
        main <- "Generative DAG (predict path solid; prior context faded)"

    draw <- function() {
        # Reserve extra right-side space for the two legends.
        op <- par(mar = c(1, 1, 3, 11))
        on.exit(par(op), add = TRUE)

        # Auto-scale vertex size + label size by node count so short names
        # fit INSIDE the circles. Longer names (K_matrix, L_chol,
        # prefix_i [n=K]) get smaller labels.
        n <- length(vnames)
        v_size <- if (n <= 8) 26 else if (n <= 20) 22 else 18
        max_label_len <- max(nchar(vnames))
        v_label_cex <- if (max_label_len <= 6) 0.75
                       else if (max_label_len <= 12) 0.65
                       else 0.55

        igraph::plot.igraph(
            g,
            layout             = layout,
            vertex.label       = vnames,
            vertex.color       = node_col,
            vertex.shape       = node_shape,
            vertex.frame.color = "grey30",
            vertex.size        = v_size,
            vertex.label.cex   = v_label_cex,
            vertex.label.color = "black",
            vertex.label.font  = 2,
            vertex.label.dist  = 0,   # label INSIDE the circle
            edge.arrow.size    = 0.5,
            edge.width         = 1.4,
            edge.curved        = 0.12,
            asp                = 0,
            main               = main,
            ...)

        # Legends in the RIGHT margin (never overlap the graph).
        par(xpd = NA)
        legend("topright", inset = c(-0.24, 0),
               legend = c("data input", "sampled param",
                          "predicted (det. / stochastic)",
                          "prior context (not predicted)"),
               fill   = c(color_for[["data"]],
                          color_for[["sampled"]],
                          color_for[["predictive"]],
                          color_for[["context"]]),
               border = "grey30", cex = 0.78, bty = "n", title = "nodes")
        # Two edge types: solid red = predict_at-traversed (generative);
        # dashed grey = prior/hyperprior context, NOT traversed.
        legend("bottomright", inset = c(-0.28, 0.04),
               legend = c("predict (generative)", "prior context"),
               col    = c("#D62728", "#AAAAAA"),
               lty    = c(1, 2), lwd = c(2, 1.5),
               cex = 0.78, bty = "n", title = "edges")
    }

    if (is.null(out_path)) {
        # Default: interactive on-screen render to current graphics device.
        # Users in R / RStudio see the DAG in their plot pane immediately.
        draw()
        return(invisible(NULL))
    }

    # Explicit out_path: PNG to file.
    grDevices::png(out_path, width = width, height = height, res = res)
    draw()
    grDevices::dev.off()
    message("Wrote DAG PNG to ", out_path)
    invisible(out_path)
}


# ----------------------------------------------------------------------------
# ai4bayescode_perf_hint
#
# Emit a friendly post-run performance message. The generated R runner
# template calls this once at the end so users who default to modular
# NUTS and find it too slow get a clear escape-hatch suggestion, along
# with the appropriate warning that joint_nuts_block has higher
# semantic-bug risk (see validator skill, Check #11).
#
# Arguments
#   wall_sec         : total wall-clock time across ALL chains (seconds)
#   n_sweeps_total   : total number of sweeps across ALL chains (including
#                      warmup). Used to compute per-sweep time.
#   uses_joint_nuts  : logical; set TRUE when the runner already uses
#                      joint_nuts_block. Suppresses the "switch to joint"
#                      hint and replaces it with validator Check #11
#                      reminder instead.
#   thresholds       : optional list with `slow_sweep_sec` (default 0.5).
#                      Hints are emitted only when per-sweep time exceeds
#                      this. Absolute threshold; the alternative -- a
#                      relative speedup number -- would need a theoretical
#                      lower bound we cannot estimate generically.
#
# Returns nothing; prints to stderr via message().
# ----------------------------------------------------------------------------
ai4bayescode_perf_hint <- function(wall_sec,
                                n_sweeps_total,
                                uses_joint_nuts = FALSE,
                                thresholds = list(slow_sweep_sec = 0.5)) {
    if (n_sweeps_total <= 0) return(invisible(NULL))
    per_sweep <- wall_sec / n_sweeps_total
    message(sprintf(
        "[AI4BayesCode perf] total %.1fs across %d sweeps (%.3fs / sweep)",
        wall_sec, n_sweeps_total, per_sweep))

    if (per_sweep <= thresholds$slow_sweep_sec) {
        message("[AI4BayesCode perf] per-sweep time looks OK.")
        return(invisible(NULL))
    }

    if (uses_joint_nuts) {
        message(
            "[AI4BayesCode perf] per-sweep time is high even with joint_nuts_block.\n",
            "  Possible causes: (a) N * J grad eval is genuinely expensive,\n",
            "  (b) NUTS tree depth maxing out -> try raising n_warmup_first_call\n",
            "      or seeding cfg.initial_step_size,\n",
            "  (c) mass-matrix adaptation not yet converged -> longer warmup.\n",
            "  Validator reminder: joint_nuts_block usage requires Check #11.")
        return(invisible(NULL))
    }

    message(
        "[AI4BayesCode perf] per-sweep time is high.\n",
        "  If this sampler has tightly-coupled continuous parameters in the\n",
        "  likelihood (e.g. additive linear mean, shift invariance,\n",
        "  fixed+random effects sharing mean), consider regenerating with\n",
        "  joint_nuts_block over the coupled parameters:\n",
        "    -> see skills/codegen.md Section 4a (Coupling analysis)\n",
        "    -> see examples/IRT1PL_joint.cpp for a reference\n",
        "  WARNING: joint_nuts_block has a higher semantic-bug surface than\n",
        "  modular NUTS (concatenate-and-slice code is easier to get\n",
        "  subtly wrong). Any joint sampler you generate must pass\n",
        "  validator Check #11 (grad slicing, prior completeness, scale\n",
        "  consistency, Jacobian, write-back offsets, dim asserts).")
    invisible(NULL)
}

# Flags a history key whose RAW rank-Rhat exceeds `hi` while its
# order-statistic rank-Rhat is already converged -- the signature of
# label switching rather than of a mixing failure. Kept identical to
# the package tree's copy in r-pkg/R/helpers.R.
.ai4b_label_switch_scan <- function(hist, hi = 1.05, converged = 1.05) {
    if (!requireNamespace("posterior", quietly = TRUE)) return(list())
    out <- list()
    rh <- function(m) suppressWarnings(max(apply(m, 2L, posterior::rhat), na.rm = TRUE))
    for (nm in names(hist)) {
        x <- hist[[nm]]
        if (is.null(dim(x)) || ncol(x) < 2L || nrow(x) < 8L) next
        raw <- rh(x); ord <- rh(t(apply(x, 1L, sort)))
        if (is.finite(raw) && is.finite(ord) && raw > hi && ord < converged)
            out[[nm]] <- list(raw = raw, ordered = ord)
    }
    out
}


# ----------------------------------------------------------------------------
# ai4bayescode_diagnose() -- model-independent posterior diagnostics + plot.
#
# Shipped so generated run_chain_<Model>(diagnosis = TRUE) runners CALL one
# function instead of re-emitting it. Identical to AI4BayesCode::ai4bayescode_diagnose()
# in the R package -- defined HERE too so the source()-the-helpers workflow (no
# package install) also has diagnostics. Per parameter: split-R-hat, bulk/tail
# ESS, MCSE, mean/sd/median/90% CI (via posterior), plus a combined trace +
# autocorrelation + density plot (via bayesplot, base-R fallback). No PSIS-LOO
# (that needs a model-specific pointwise log-likelihood).
# ----------------------------------------------------------------------------
ai4bayescode_diagnose <- function(hist, n_burn = 0, plot = TRUE, order_components = FALSE) {
    # A model object is accepted in place of its history.
    if (!is.list(hist) &&
        is.function(tryCatch(hist$get_history, error = function(e) NULL)))
        hist <- hist$get_history()
    if (!requireNamespace("posterior", quietly = TRUE)) {
        stop("ai4bayescode_diagnose() needs the 'posterior' package. ",
             "Install it with install.packages('posterior').", call. = FALSE)
    }
    if (!is.list(hist) || is.null(names(hist)) || !all(nzchar(names(hist)))) {
        stop("`hist` must be a named list of posterior draws or a model with get_history() ",
             "(scalars as vectors, vector parameters as matrices).",
             call. = FALSE)
    }
    # Drop the first n_burn draws of EVERY key (scalars: leading elements; matrices:
    # leading rows). get_history() returns burn-in + keep, so callers pass the
    # burn-in length; n_burn = 0 leaves hist untouched. Mirrors Python diagnose()'s
    # `np.asarray(x)[n_burn:]` + the "no post-burn-in draws" ValueError.
    n_burn <- as.integer(n_burn)
    if (is.na(n_burn) || n_burn < 0L)
        stop("`n_burn` must be a non-negative integer.", call. = FALSE)
    if (n_burn > 0L) {
        drop_burn_key <- function(x) {
            if (is.null(dim(x))) {
                n <- length(x)
                if (n_burn >= n) return(x[integer(0)])
                x[(n_burn + 1L):n]
            } else {
                n <- nrow(x)
                if (n_burn >= n) return(x[integer(0), , drop = FALSE])
                x[(n_burn + 1L):n, , drop = FALSE]
            }
        }
        hist <- lapply(hist, drop_burn_key)
        lens <- vapply(hist, function(x)
            if (is.null(dim(x))) length(x) else nrow(x), integer(1))
        if (length(lens) && min(lens) == 0L)
            stop(sprintf(
"ai4bayescode_diagnose: n_burn=%d leaves no post-burn-in draws (the shortest history key has <= %d draws). Reduce n_burn.",
                n_burn, n_burn), call. = FALSE)
    }
    # Detect label switching on the RAW draws (before any ordering).
    label_switch <- .ai4b_label_switch_scan(hist)
    if (length(label_switch) && !isTRUE(order_components)) {
        ex <- label_switch[[1L]]
        message(sprintf(
"ai4bayescode_diagnose: %s MIGHT have LABEL SWITCHING -- ordering components within each\n  draw brings R-hat down to a converged level (e.g. %s: %.2f -> %.2f), so the high raw\n  R-hat MAY be a labelling artefact rather than non-convergence. Pass order_components =\n  TRUE for a label-invariant summary, or canonicalize the labels in the sampler.",
            paste(names(label_switch), collapse = ", "),
            names(label_switch)[1L], ex$raw, ex$ordered))
    }
    # order_components: sort each draw's components within every matrix key (order
    # statistics) -> a label-invariant per-component summary for exchangeable params.
    sort_key <- function(x) if (is.null(dim(x)) || ncol(x) < 2L) x else t(apply(x, 1L, sort))
    hist_use <- if (isTRUE(order_components)) lapply(hist, sort_key) else hist
    cols <- list()
    for (nm in names(hist_use)) {
        x <- hist_use[[nm]]
        if (is.null(dim(x))) cols[[nm]] <- as.numeric(x)
        else {
            # Flatten anything with more than 2 dimensions down to
            # (n_draws x everything-else), as the Python side does. A
            # covariance history arrives as n x r x c, and `x[, j]` on a 3-D
            # array raises "incorrect number of dimensions" -- a shape
            # ai4bayescode_run_chains's own documentation promises to support.
            if (length(dim(x)) > 2L) x <- matrix(x, nrow = dim(x)[1L])
            for (j in seq_len(ncol(x))) cols[[sprintf("%s[%d]", nm, j)]] <- x[, j]
        }
    }
    M <- do.call(cbind, cols); colnames(M) <- names(cols)
    drw <- posterior::as_draws_matrix(M)
    summary <- posterior::summarise_draws(drw)
    plt <- NULL
    if (isTRUE(plot)) {
        if (requireNamespace("bayesplot", quietly = TRUE)) {
            # bayesplot's ACF panel can ABORT the whole plot (e.g. "Too few
            # iterations for lags=20") on degenerate / near-constant columns --
            # empty mixture components in DP/PY/HDP models, constant params. The
            # diagnostic plot is secondary; never let it crash the diagnose (the
            # summary + R-hat/ESS are the point). Fall back to trace+dens, then none.
            plt <- tryCatch(
                bayesplot::mcmc_combo(drw, combo = c("trace", "acf", "dens")),
                error = function(e) tryCatch(
                    bayesplot::mcmc_combo(drw, combo = c("trace", "dens")),
                    error = function(e2) {
                        warning("ai4bayescode_diagnose: diagnostic plot skipped (",
                                conditionMessage(e2), ").", call. = FALSE)
                        NULL
                    }))
        } else {
            message("Install 'bayesplot' for a ready-to-print diagnostic plot; ",
                    "returning a base-R plotting function instead.")
            plt <- function() {
                op <- graphics::par(mfrow = c(min(ncol(M), 4L), 3L),
                                    mar = c(3, 3, 2, 1))
                on.exit(graphics::par(op))
                for (nm in colnames(M)) {
                    graphics::plot(M[, nm], type = "l",
                                   main = paste("trace:", nm), xlab = "", ylab = "")
                    stats::acf(M[, nm], main = paste("ACF:", nm))
                    graphics::plot(stats::density(M[, nm]),
                                   main = paste("density:", nm))
                }
            }
        }
    }
    list(summary = summary, plot = plt, label_switch = label_switch)
}

# ----------------------------------------------------------------------------
# ai4bayescode_new_frozen
#
# Ctor helper: construct a model and immediately set + freeze the components
# listed in `fixed`, in one call. Equivalent to the three-step form:
#     m <- new(module_class, ...)
#     m$set_current(fixed)
#     m$freeze(names(fixed), quiet = quiet_freeze)
#
# Scope: `fixed` keys must be FLAT (a top-level child block name, or a
# component of a joint_nuts_block -- whatever set_current() accepts).
# Dot-path names (nested composite descent, rjmcmc sub-key) are REJECTED
# because set_current does not route dot-path keys: freezing a value that
# was never set would give a wrong posterior. To freeze at a dot path, call
# set_current() at the matching composite level and then freeze().
#
# Args
# ----
# module_class   : Rcpp module class name (e.g. MyGaussianReg)
# ...            : wrapper ctor arguments forwarded to new(module_class, ...)
# fixed          : named list of (name = value) pairs. Empty list is a no-op.
# quiet_freeze   : passed as `quiet=` to m$freeze() -- default TRUE, since
#                  this is a ctor-time call where redundant-refreeze is
#                  impossible (nothing was frozen before).
# ----------------------------------------------------------------------------
ai4bayescode_new_frozen <- function(module_class, ...,
                                    fixed = list(),
                                    quiet_freeze = TRUE) {
    if (length(fixed) > 0L) {
        nms <- names(fixed)
        if (is.null(nms) || any(nms == "")) {
            stop("ai4bayescode_new_frozen(): `fixed` must be a NAMED list")
        }
        dot_names <- grepl("\\.", nms, fixed = TRUE)
        if (any(dot_names)) {
            stop(sprintf(
                "ai4bayescode_new_frozen(): dot-path names not allowed in `fixed` (%s). Use post-construction m$set_current(...) at the correct composite level + m$freeze(<dot.path>) instead.",
                paste(nms[dot_names], collapse = ", ")))
        }
    }
    m <- new(module_class, ...)
    if (length(fixed) > 0L) {
        m$set_current(fixed)
        m$freeze(names(fixed), quiet = isTRUE(quiet_freeze))
    }
    m
}


# ---------------------------------------------------------------------------
# Multi-chain driver + cross-chain convergence summary.
#
# Ported verbatim from the installed package (r-pkg/R/helpers.R) so that a
# checkout-mode user -- who sources THIS file instead of library(AI4BayesCode)
# -- gets the identical generated-example flow:
#     run <- ai4bayescode_run_chains(function(seed) new(<Class>, ..., seed, TRUE),
#                                    n_chains = 4L, n_burn = 4000L, n_keep = 4000L)
#     ai4bayescode_rhat_summary(run)
#     ai4bayescode_diagnose(run$histories[[1]])
# Keep in sync with r-pkg/R/helpers.R (single source of truth: the package).
#
# ai4bayescode_run_chains(): returns list(histories, seeds, wall). Histories are
#   KEEP-ONLY -- the first n_burn draws of every entry are stripped along its
#   first axis (vectors, matrices, and higher-dimensional arrays alike).
# ai4bayescode_rhat_summary(): per-key rhat + ess_bulk across chains; needs the
#   'posterior' package.
# ---------------------------------------------------------------------------

ai4bayescode_run_chains <- function(model_ctor,
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

    # Drop the first `nb` draws along the FIRST axis of one history entry.
    # Entries can be plain vectors (n), matrices (n x d), or higher-D
    # arrays (n x r x c, ...); the slice runs along axis 1 only and keeps
    # every other axis (and names/dimnames) intact. An entry with <= nb
    # rows is returned unchanged: with keep_history disabled get_history()
    # holds a single, already post-warmup, current draw -- emptying it
    # would be worse than keeping it.
    strip_burn_1 <- function(x, nb) {
        d   <- dim(x)
        len <- if (is.null(d)) length(x) else d[1L]
        if (nb <= 0L || len <= nb) return(x)
        keep <- (nb + 1L):len
        if (is.null(d)) return(x[keep])
        args <- c(list(x, keep),
                  rep(list(quote(expr = )), length(d) - 1L),
                  list(drop = FALSE))
        do.call(`[`, args)
    }

    one_chain <- function(seed_val) {
        t0 <- Sys.time()
        m <- model_ctor(seed_val)
        m$step(as.integer(n_burn))
        m$step(as.integer(n_keep))
        t1 <- Sys.time()
        # get_history() spans EVERY stepped iteration (warmup + keep);
        # return keep-only draws, matching this function's documentation.
        h <- lapply(m$get_history(), strip_burn_1, nb = as.integer(n_burn))
        list(history = h,
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

    chain_ok <- function(r) !inherits(r, "try-error") && !is.null(r$history)

    # The BLAS is pinned to one thread at load (.ai4b_pin_blas_threads above),
    # which is the only place it can work -- Accelerate decides whether to use
    # its dispatch queues at the FIRST BLAS call, so a cap applied here would
    # change nothing. If a variable is not "1" by now the caller overrode it,
    # and forking a BLAS-heavy model may abort the session.
    warn_if_blas_unpinned <- function() {
        vars <- c("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        cur  <- Sys.getenv(vars, unset = NA)
        bad  <- vars[!is.na(cur) & cur != "1"]
        if (length(bad))
            warning("ai4bayescode_run_chains: ",
                    paste(sprintf("%s=%s", bad, cur[bad]), collapse = ", "),
                    " -- the native BLAS may be multithreaded. fork() and a ",
                    "multithreaded BLAS do not mix; on macOS Accelerate this ",
                    "aborts the whole R session rather than failing a chain. ",
                    "Set these to \"1\" before starting R, or pass ",
                    "parallel = FALSE.", call. = FALSE, immediate. = TRUE)
    }

    if (use_parallel && mc.cores > 1) {
        if (verbose)
            message("ai4bayescode_run_chains: running ", n_chains,
                    " chains on ", mc.cores, " cores (parallel)")
        warn_if_blas_unpinned()
        results <- parallel::mclapply(seeds, one_chain, mc.cores = mc.cores,
                                      mc.set.seed = TRUE)
        if (!all(vapply(results, chain_ok, logical(1)))) {
            # Second line of defence: a forked worker returned no history.
            # With the BLAS pinned above the fork-unsafety is gone, so this is
            # now most likely a genuine model or data error -- but re-running
            # sequentially costs one run and rules out anything fork-specific.
            warning("ai4bayescode_run_chains: a parallel chain failed; ",
                    "re-running all chains sequentially to rule out a ",
                    "fork-specific cause. Pass parallel = FALSE to skip the ",
                    "parallel attempt entirely.",
                    call. = FALSE, immediate. = TRUE)
            results <- lapply(seeds, one_chain)
        }
    } else {
        if (verbose)
            message("ai4bayescode_run_chains: running ", n_chains,
                    " chains sequentially")
        results <- lapply(seeds, one_chain)
    }

    # Detect failures (after any sequential fallback -- a real model/data error).
    for (i in seq_along(results)) {
        if (!chain_ok(results[[i]])) {
            stop("ai4bayescode_run_chains: chain ", i, " failed")
        }
    }

    list(
        histories = lapply(results, `[[`, "history"),
        seeds     = sapply(results, `[[`, "seed"),
        wall      = sapply(results, `[[`, "wall")
    )
}

ai4bayescode_rhat_summary <- function(run, keys = NULL, drop_burn = 0,
                                      order_components = FALSE) {
    if (!requireNamespace("posterior", quietly = TRUE)) {
        stop("posterior package required for R-hat summary")
    }
    histories <- run$histories
    if (length(histories) < 1L) {
        stop("run$histories is empty -- nothing to summarise")
    }
    if (length(histories) == 1L) {
        # Single chain: posterior::rhat() splits the one chain in half and
        # returns the rank-normalized SPLIT-R-hat (a valid within-chain
        # convergence check). Between-chain R-hat needs >= 2 chains.
        message("ai4bayescode_rhat_summary: only 1 chain -- reporting split-R-hat ",
                "(each chain split in half); use >= 2 chains for the standard ",
                "between-chain R-hat.")
    }
    all_keys <- names(histories[[1]])
    if (is.null(keys)) keys <- all_keys

    # Per-column split-R-hat + bulk-ESS across chains for a list of per-chain
    # matrices (draws in rows). Returns list(rhat = <p>, ess = <p>).
    col_diag <- function(mats) {
        p <- ncol(mats[[1]])
        rh <- numeric(p); eb <- numeric(p)
        for (j in seq_len(p)) {
            per_chain <- lapply(mats, function(m) m[, j])
            arr_j <- array(unlist(per_chain),
                           dim = c(length(per_chain[[1]]), length(per_chain), 1))
            rh[j] <- tryCatch(posterior::rhat(arr_j[,,1]), error = function(e) NA)
            eb[j] <- tryCatch(posterior::ess_bulk(arr_j[,,1]), error = function(e) NA)
        }
        list(rhat = rh, ess = eb)
    }
    # Sort each draw's components within a chain (order statistics are invariant to
    # relabelling); leaves scalars / single-column matrices unchanged.
    sort_rows <- function(m) if (is.null(dim(m)) || ncol(m) < 2L) m else t(apply(m, 1L, sort))

    out <- list(); label_switch <- list()
    for (k in keys) {
        if (!k %in% all_keys) next
        vals <- lapply(histories, function(h) h[[k]])
        if (is.null(dim(vals[[1]]))) {
            # Scalar history per chain (never label-switches).
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
            # Matrix history (n_draws x dim per chain).
            n <- nrow(vals[[1]])
            if (drop_burn > 0 && drop_burn < n)
                vals <- lapply(vals, function(m) m[(drop_burn + 1):n, , drop=FALSE])
            raw <- col_diag(vals)
            ord <- if (ncol(vals[[1]]) >= 2L) col_diag(lapply(vals, sort_rows)) else raw
            # Flag label switching ONLY when the high raw max R-hat drops BELOW a
            # converged level after ordering components within each draw. If it
            # stays high, the non-convergence is genuine (bad sampler / wrong
            # model / slow mixing) -- NOT a labelling artefact -- so do not flag.
            mr <- max(raw$rhat, na.rm = TRUE); mo <- max(ord$rhat, na.rm = TRUE)
            if (is.finite(mr) && is.finite(mo) && mr > 1.05 && mo < 1.05)
                label_switch[[k]] <- list(raw = mr, ordered = mo)
            chosen <- if (isTRUE(order_components)) ord else raw
            out[[k]] <- list(rhat = chosen$rhat, ess_bulk = chosen$ess,
                              max_rhat = max(chosen$rhat, na.rm = TRUE),
                              min_ess  = min(chosen$ess,  na.rm = TRUE))
        }
    }
    if (length(label_switch)) {
        attr(out, "label_switch") <- label_switch
        if (!isTRUE(order_components)) {
            ex <- label_switch[[1L]]
            message(sprintf(
"ai4bayescode_rhat_summary: %s MIGHT have LABEL SWITCHING -- ordering components within\n  each draw brings R-hat down to a converged level (e.g. %s: %.2f -> %.2f), so the high\n  raw R-hat MAY be a labelling artefact rather than non-convergence. Pass order_components\n  = TRUE for a label-invariant summary, or canonicalize the labels in the sampler.",
                paste(names(label_switch), collapse = ", "),
                names(label_switch)[1L], ex$raw, ex$ordered))
        }
    }
    out
}

# ---------------------------------------------------------------------------
# Named-argument constructors. Mirrors r-pkg/R/named_ctor.R plus the three
# parsing helpers it uses from r-pkg/R/doc.R; the package tree is the
# source of truth, this file is the checkout-path copy.
# ---------------------------------------------------------------------------

# Map a C++ type to a short R-friendly hint.
#' @keywords internal
#' @noRd
.ai4b_type_hint <- function(type) {
    t <- gsub("const|&|\\s+", " ", type); t <- trimws(gsub("\\s+", " ", t))
    dplc <- function(p) grepl(p, t, perl = TRUE)
    if (dplc("arma::mat|Mat<")) return("numeric matrix")
    if (dplc("arma::ivec|arma::uvec|Col<arma::(sword|uword)")) return("integer vector")
    if (dplc("arma::vec|Col<double|NumericVector")) return("numeric vector")
    if (dplc("NumericMatrix|IntegerMatrix")) return("numeric matrix")
    if (dplc("IntegerVector")) return("integer vector")
    if (dplc("\\bbool\\b")) return("logical (TRUE/FALSE)")
    if (dplc("\\bint\\b|size_t|uint")) return("integer (e.g. 1L)")
    if (dplc("double|float")) return("numeric")
    # Rcpp reports a bound constructor's types demangled from typeid, where
    # std::string appears as basic_string<...> (libc++ and libstdc++ spell the
    # namespace differently), so match the template name too.
    if (dplc("std::string|basic_string|CharacterVector|String")) return("character")
    trimws(type)
}

# Given the index of an opening '(' in src, balance-match its ')' and parse
# the parameter list into args (name, type, hint, default). Returns NULL on
# imbalance, list() for an empty list.
#' @keywords internal
#' @noRd
.ai4b_args_after <- function(src, open) {
    chars <- strsplit(substring(src, open), "")[[1]]
    depth <- 0L; end <- NA_integer_
    for (i in seq_along(chars)) {
        if (chars[i] == "(") depth <- depth + 1L
        else if (chars[i] == ")") { depth <- depth - 1L; if (depth == 0L) { end <- i; break } }
    }
    if (is.na(end)) return(NULL)
    inside <- gsub("\\s+", " ", trimws(substring(src, open + 1L, open + end - 2L)))
    if (!nzchar(inside)) return(list())
    parts <- character(0); buf <- ""; ang <- 0L            # split top-level commas
    for (ch in strsplit(inside, "")[[1]]) {
        if (ch == "<") ang <- ang + 1L else if (ch == ">") ang <- ang - 1L
        if (ch == "," && ang == 0L) { parts <- c(parts, buf); buf <- "" } else buf <- paste0(buf, ch)
    }
    parts <- c(parts, buf)
    parts <- parts[nzchar(trimws(parts))]   # drop empty parts (e.g. trailing comma)
    lapply(parts, function(p) {
        p <- trimws(p); default <- NA_character_
        if (grepl("=", p)) { sp <- strsplit(p, "=", fixed = TRUE)[[1]]
            default <- trimws(paste(sp[-1], collapse = "=")); p <- trimws(sp[1]) }
        toks <- strsplit(p, "\\s+")[[1]]
        last <- toks[length(toks)]
        name <- gsub("[&*]", "", last)
        # Strip the trailing name token by LENGTH, not as a regex (a name with
        # regex metachars like '(' would otherwise crash sub()).
        type <- trimws(substring(p, 1L, nchar(p) - nchar(last)))
        list(name = name, type = type, hint = .ai4b_type_hint(type), default = default)
    })
}

# Exposed class names from `Rcpp::class_<X>("Name")` or `class_<X>("Name")`.
#' @keywords internal
#' @noRd
.ai4b_exposed_class_names <- function(src) {
    m <- gregexpr('class_<[^>]+>\\s*\\(\\s*"([^"]+)"', src, perl = TRUE)
    hit <- regmatches(src, m)[[1]]
    if (!length(hit)) return(character(0))
    unique(sub('.*"([^"]+)".*', "\\1", hit))
}

# named_ctor.R
#
# Named-argument constructors for Rcpp module classes.
#
# Rcpp dispatches a module constructor on ARITY ALONE -- its default validator
# is `yes_arity<N>`, whose whole body is `return nargs == n`. Argument NAMES
# never reach the dispatcher, and C++ default arguments are invisible to R.
# So for a stock module class `new(Model, X = X, y = y, rng_seed = 7L)` does
# not mean what it reads as: the three values are passed POSITIONALLY into
# whichever constructor happens to have arity 3, and the names are dropped in
# silence.
#
# The names are still intact one level up, in the reference-class `initialize`
# method Rcpp installs on each exposed class:
#
#     if (nargs()) Rcpp::cpp_object_initializer(.self, .refClassDef, ...)
#     else         Rcpp::cpp_object_dummy(.self, .refClassDef)
#
# This file replaces that method, per class, with one that matches arguments
# R-style (exact then unique-partial names, then positional fill), supplies
# the C++ default for anything omitted, rejects unknown or duplicated names,
# and only then calls the original initializer with a COMPLETE positional
# list. Nothing in Rcpp is patched globally; only classes loaded through
# ai4bayescode_source() are affected.

# Convert a C++ default-argument literal to an R value, guided by the declared
# type. Returns a one-element list on success and NULL when the literal is not
# something R can reconstruct (in which case the argument stays REQUIRED --
# guessing a value the user did not write is the failure this file exists to
# prevent).
#' @keywords internal
#' @noRd
.ai4b_cpp_default_to_r <- function(default, type) {
    if (is.null(default) || is.na(default)) return(NULL)
    d <- trimws(default)
    d <- sub("^\\{(.*)\\}$", "\\1", d)          # brace-init: `= {3}`
    d <- trimws(sub("[uUlLfF]+$", "", d))       # literal suffixes: 3u, 1.0f
    if (!nzchar(d)) return(NULL)
    if (identical(d, "true"))  return(list(TRUE))
    if (identical(d, "false")) return(list(FALSE))
    # A default-constructed armadillo container is EMPTY, and an empty R
    # vector/matrix converts back to exactly that -- so this default is
    # reconstructible, unlike an arbitrary expression.
    if (grepl("^arma::(vec|colvec|rowvec|uvec|ivec)\\(\\)$", d))
        return(list(numeric(0)))
    if (grepl("^arma::(mat|umat|imat)\\(\\)$", d))
        return(list(matrix(numeric(0), 0L, 0L)))
    if (!grepl("^[+-]?([0-9]+\\.?[0-9]*|\\.[0-9]+)([eE][+-]?[0-9]+)?$", d))
        return(NULL)
    hint <- .ai4b_type_hint(type)
    if (identical(hint, "logical (TRUE/FALSE)")) return(list(as.numeric(d) != 0))
    if (identical(hint, "integer (e.g. 1L)"))    return(list(as.integer(as.numeric(d))))
    list(as.numeric(d))
}

# Canonical form of a C++ type, so a declared parameter type and the type
# Rcpp reports for a bound constructor compare equal.
#' @keywords internal
#' @noRd
.ai4b_norm_type <- function(t) {
    # The .cpp says `const arma::vec&`; Rcpp reports the resolved
    # `arma::Col<double>` for the same parameter. Comparing the spellings
    # directly would never match, so both sides go through the same coarse
    # type hint, which maps every spelling of a kind onto one label.
    vapply(t, function(x) .ai4b_type_hint(gsub("[&*]", "", x)),
           character(1), USE.NAMES = FALSE)
}

# Every constructor a class declares, in declaration order. A class may
# declare more than one -- the mixtures pair a simple data-driven constructor
# with an advanced one taking explicit hyperparameters -- and which of them a
# call means is decided by the arguments themselves, not up front.
#' @keywords internal
#' @noRd
.ai4b_parse_all_constructors <- function(src, class_name) {
    if (is.null(class_name) || is.na(class_name) || !nzchar(class_name))
        return(list())
    src <- gsub("//[^\n]*", "", src)
    src <- gsub("(?s)/\\*.*?\\*/", "", src, perl = TRUE)
    locs <- gregexpr(sprintf("(?m)^[ \\t]*%s[ \\t]*\\(", class_name), src,
                     perl = TRUE)[[1]]
    if (locs[1L] < 0) return(list())
    lens <- attr(locs, "match.length")
    out <- list()
    for (i in seq_along(locs)) {
        a <- .ai4b_args_after(src, locs[i] + lens[i] - 1L)
        if (!is.null(a) && length(a)) out[[length(out) + 1L]] <- a
    }
    out
}

# Constructor signatures for one class: for each declared constructor, the
# argument names, canonical types, R-side defaults, and a flag per argument
# saying whether a usable default exists.
#' @keywords internal
#' @noRd
.ai4b_ctor_signatures <- function(cpp_file, class_name) {
    src <- tryCatch(paste(readLines(cpp_file, warn = FALSE), collapse = "\n"),
                    error = function(e) NULL)
    if (is.null(src)) return(list())
    all <- tryCatch(.ai4b_parse_all_constructors(src, class_name),
                    error = function(e) list())
    out <- list()
    for (args in all) {
        nms <- vapply(args, function(a) a$name, character(1))
        if (anyNA(nms) || !all(nzchar(nms)) || anyDuplicated(nms)) next
        defs <- vector("list", length(args)); has <- logical(length(args))
        for (i in seq_along(args)) {
            v <- .ai4b_cpp_default_to_r(args[[i]]$default, args[[i]]$type)
            if (!is.null(v)) { defs[[i]] <- v[[1L]]; has[i] <- TRUE }
        }
        out[[length(out) + 1L]] <- list(
            names = nms,
            types = vapply(args, function(a) .ai4b_norm_type(a$type), character(1)),
            defaults = defs, has_default = has)
    }
    out
}

# The constructors Rcpp actually bound, as arity plus canonical type list,
# read from each C++Constructor's reported signature.
#' @keywords internal
#' @noRd
.ai4b_bound_ctors <- function(gen_holder) {
    ks <- tryCatch(gen_holder@constructors, error = function(e) list())
    out <- list()
    for (k in ks) {
        n <- tryCatch(as.integer(k$nargs), error = function(e) NA_integer_)
        sg <- tryCatch(as.character(k$signature), error = function(e) NA_character_)
        ty <- character(0)
        if (!is.na(sg)) {
            inner <- sub("^[^(]*\\((.*)\\)\\s*$", "\\1", sg)
            if (nzchar(trimws(inner)))
                ty <- .ai4b_norm_type(.ai4b_split_types(inner))
        }
        if (!is.na(n)) out[[length(out) + 1L]] <- list(nargs = n, types = ty)
    }
    out
}

# Split a comma-separated C++ parameter/type list at TOP level only. A
# templated type carries commas of its own (a map, a nested vector), and
# splitting on every comma would report more types than the constructor has
# parameters -- which silently disqualified the class from named arguments.
#' @keywords internal
#' @noRd
.ai4b_split_types <- function(s) {
    chars <- strsplit(s, "")[[1]]
    out <- character(0); buf <- ""; depth <- 0L
    for (ch in chars) {
        if (ch == "<" || ch == "(") depth <- depth + 1L
        else if (ch == ">" || ch == ")") depth <- depth - 1L
        if (ch == "," && depth == 0L) { out <- c(out, buf); buf <- "" }
        else buf <- paste0(buf, ch)
    }
    trimws(c(out, buf))
}

# Match a call's arguments against a constructor signature, R-style. Returns
# list(values, filled); `filled` marks which slots the CALLER supplied, so the
# installer can pick the narrowest bound arity that still covers them.
#' @keywords internal
#' @noRd
.ai4b_match_ctor_args <- function(nms, defaults, has_default, args, class_name) {
    n  <- length(nms)
    an <- names(args); if (is.null(an)) an <- rep("", length(args))
    values <- vector("list", n); filled <- logical(n)

    # Assigning NULL through [[<- DELETES the element and shifts every later
    # one down, which would move already-matched values onto the wrong
    # parameters. Single-bracket assignment of a one-element list stores NULL
    # in place.
    put <- function(j, v) { values[j] <<- list(v); filled[j] <<- TRUE }

    named <- which(nzchar(an))
    # Pass 1: exact names, ALL of them, before any partial matching -- this is
    # base R's order, and doing it per-argument instead would let a partial
    # match claim a formal that a later exact name owns.
    partial <- integer(0)
    for (i in named) {
        j <- match(an[i], nms)
        if (is.na(j)) { partial <- c(partial, i); next }
        if (filled[j])
            stop(sprintf("%s: argument \"%s\" supplied more than once.",
                         class_name, nms[j]), call. = FALSE)
        put(j, args[[i]])
    }
    # Pass 2: unique partial matches, considered only against the formals that
    # pass 1 left unclaimed.
    for (i in partial) {
        cand <- which(!filled & startsWith(nms, an[i]))
        if (length(cand) > 1L)
            stop(sprintf("%s: argument \"%s\" matches several arguments: %s.",
                         class_name, an[i], paste(nms[cand], collapse = ", ")),
                 call. = FALSE)
        if (!length(cand))
            stop(sprintf("%s: unknown argument \"%s\". Valid arguments: %s.",
                         class_name, an[i], paste(nms, collapse = ", ")),
                 call. = FALSE)
        put(cand, args[[i]])
    }
    # Pass 3: positional arguments fill the slots still open, left to right.
    k <- 1L
    for (i in which(!nzchar(an))) {
        while (k <= n && filled[k]) k <- k + 1L
        if (k > n)
            stop(sprintf("%s: too many arguments; the constructor takes %d (%s).",
                         class_name, n, paste(nms, collapse = ", ")), call. = FALSE)
        put(k, args[[i]])
    }
    missing <- which(!filled & !has_default)
    if (length(missing))
        stop(sprintf("%s: missing required argument(s): %s.",
                     class_name, paste(nms[missing], collapse = ", ")),
             call. = FALSE)
    for (j in which(!filled)) values[j] <- list(defaults[[j]])
    list(values = values, filled = filled)
}

# Resolve one call against a class's constructors. Each declared constructor
# is tried in declaration order and the first that ACCEPTS the arguments wins
# -- argument names disambiguate on their own, so a call naming an advanced
# hyperparameter cannot land on the simple constructor and vice versa.
#
# Having chosen a constructor, pick which bound registration to call: the
# narrowest one whose type list is a PREFIX of that constructor's parameter
# types and that still covers every argument the caller supplied. Matching on
# types, not on arity alone, is what keeps a call off a same-arity
# registration belonging to a DIFFERENT constructor.
#' @keywords internal
#' @noRd
.ai4b_resolve_ctor_call <- function(sigs, bound, args, class_name) {
    # When a class declares several constructors and none accepts the call,
    # the useful complaint comes from the one the caller evidently MEANT --
    # the signature recognising the most of the names they supplied -- not
    # from whichever happens to be declared first.
    supplied <- names(args); if (is.null(supplied)) supplied <- character(0)
    supplied <- supplied[nzchar(supplied)]
    score <- vapply(sigs, function(sg)
        sum(vapply(supplied, function(nm)
            nm %in% sg$names || length(which(startsWith(sg$names, nm))) == 1L,
            logical(1))), integer(1))
    best_err <- NULL; best_score <- -1L
    keep_err <- function(e, i) {
        if (score[i] > best_score) { best_err <<- e; best_score <<- score[i] }
    }
    for (si in seq_along(sigs)) {
        sig <- sigs[[si]]
        m <- tryCatch(
            .ai4b_match_ctor_args(sig$names, sig$defaults, sig$has_default,
                                  args, class_name),
            error = function(e) e)
        if (inherits(m, "error")) { keep_err(m, si); next }
        need <- if (any(m$filled)) max(which(m$filled)) else 0L
        best <- NA_integer_
        for (b in bound) {
            if (b$nargs < need || b$nargs > length(sig$names)) next
            if (length(b$types) != b$nargs) next
            if (!identical(b$types, unname(sig$types[seq_len(b$nargs)]))) next
            if (is.na(best) || b$nargs < best) best <- b$nargs
        }
        if (!is.na(best)) return(m$values[seq_len(best)])
        keep_err(simpleError(sprintf(
                paste0("%s: no bound constructor accepts %d argument(s). ",
                       "Bound arities: %s."),
                class_name, need,
                paste(sort(vapply(bound, function(b) b$nargs, integer(1))),
                      collapse = ", "))), si)
    }
    stop(if (is.null(best_err))
             simpleError(sprintf("%s: no constructor matches this call.", class_name))
         else best_err)
}

# Replace one exposed class's reference-class `initialize` with the
# name-matching version. Silently does nothing when the signatures cannot be
# recovered -- the class then keeps stock Rcpp behaviour rather than acquiring
# a half-informed one.
#' @keywords internal
#' @noRd
.ai4b_install_named_ctor <- function(class_name, cpp_file, env) {
    holder <- tryCatch(get(class_name, envir = env), error = function(e) NULL)
    if (is.null(holder) || !methods::is(holder, "C++Class")) return(FALSE)
    gen <- tryCatch(holder@generator, error = function(e) NULL)
    if (is.null(gen)) return(FALSE)

    sigs <- .ai4b_ctor_signatures(cpp_file, class_name)
    if (!length(sigs)) return(FALSE)
    bound <- .ai4b_bound_ctors(holder)
    if (!length(bound)) return(FALSE)
    # Every bound registration must be reachable from some declared
    # constructor; if one is not, the parse disagrees with the module and the
    # class is left alone rather than served a guess.
    reachable <- vapply(bound, function(b) {
        any(vapply(sigs, function(sg)
            b$nargs <= length(sg$names) && length(b$types) == b$nargs &&
            identical(b$types, unname(sg$types[seq_len(b$nargs)])), logical(1)))
    }, logical(1))
    if (!all(reachable)) return(FALSE)

    # The replacement runs as a reference-class method, which is re-environed
    # onto the object, so it cannot close over anything here. Everything it
    # needs is baked into the body as a literal instead.
    f <- function(...) NULL
    body(f) <- bquote({
        if (!nargs()) return(Rcpp::cpp_object_dummy(.self, .refClassDef))
        .args <- list(...)
        # C++ handing an already-constructed object back to R
        # (Rcpp::cpp_object_maker) must pass through untouched.
        if (".object_pointer" %in% names(.args))
            return(Rcpp::cpp_object_initializer(.self, .refClassDef, ...))
        do.call(Rcpp::cpp_object_initializer,
                c(list(.self, .refClassDef),
                  .(.ai4b_resolve_ctor_call)(.(sigs), .(bound), .args,
                                             .(class_name))))
    })
    tryCatch({ gen$methods(initialize = f); TRUE }, error = function(e) FALSE)
}

# Install on every class the file exposes. Returns the names it handled.
#' @keywords internal
#' @noRd
.ai4b_install_named_ctors <- function(cpp_file, env) {
    src <- tryCatch(paste(readLines(cpp_file, warn = FALSE), collapse = "\n"),
                    error = function(e) "")
    done <- character(0)
    for (nm in .ai4b_exposed_class_names(src))
        if (isTRUE(.ai4b_install_named_ctor(nm, cpp_file, env))) done <- c(done, nm)
    done
}
