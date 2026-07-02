# ----------------------------------------------------------------------------
# AI4BayesCode_helpers.R
#
# User-facing convenience helpers for using AI4BayesCode from an R script.
# Source this file once at the top of your analysis script and then call
# ai4bayescode_sourceCpp("MyModel.cpp", AI4BayesCode_path = "./AI4BayesCode").
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
#     ai4bayescode_sourceCpp("MyModel.cpp", AI4BayesCode_path = "./AI4BayesCode")
#     model <- new(MyModel, ...)
#
# Requirements
# ------------
#     - R with Rcpp and RcppArmadillo installed
#     - The AI4BayesCode folder reachable on disk at `AI4BayesCode_path`
# ----------------------------------------------------------------------------

AI4BayesCode_generated_dir <- function(AI4BayesCode_path = "./AI4BayesCode") {
    gen_dir <- file.path(dirname(normalizePath(AI4BayesCode_path, mustWork = TRUE)),
                         "generated")
    if (!dir.exists(gen_dir)) {
        dir.create(gen_dir, recursive = TRUE)
        message("Created ", gen_dir)
    }
    gen_dir
}

ai4bayescode_sourceCpp <- function(cpp_file,
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
    # autodiff is at inc_root/autodiff/ → users include as
    # `<autodiff/reverse/var.hpp>`. No separate -I needed (inc_root covers it).
    # Eigen is at inc_root/eigen/Eigen/ → needs a dedicated -I for
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
        "-DARMA_DONT_USE_WRAPPER",
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

    # PREDICTION-ONLY DAG — Gibbs / refresh edges intentionally hidden.
    # Per codegen.md §2(b): user-facing DAG visualization shows the
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
    # computed from the PREDICT subgraph only — context (prior) edges are
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
#                      this. Absolute threshold; the alternative — a
#                      relative speedup number — would need a theoretical
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
ai4bayescode_diagnose <- function(hist, plot = TRUE) {
    if (!requireNamespace("posterior", quietly = TRUE)) {
        stop("ai4bayescode_diagnose() needs the 'posterior' package. ",
             "Install it with install.packages('posterior').", call. = FALSE)
    }
    if (!is.list(hist) || is.null(names(hist)) || !all(nzchar(names(hist)))) {
        stop("`hist` must be a named list of posterior draws ",
             "(scalars as vectors, vector parameters as matrices).",
             call. = FALSE)
    }
    cols <- list()
    for (nm in names(hist)) {
        x <- hist[[nm]]
        if (is.null(dim(x))) cols[[nm]] <- as.numeric(x)
        else for (j in seq_len(ncol(x))) cols[[sprintf("%s[%d]", nm, j)]] <- x[, j]
    }
    M <- do.call(cbind, cols); colnames(M) <- names(cols)
    drw <- posterior::as_draws_matrix(M)
    summary <- posterior::summarise_draws(drw)
    plt <- NULL
    if (isTRUE(plot)) {
        if (requireNamespace("bayesplot", quietly = TRUE)) {
            plt <- bayesplot::mcmc_combo(drw, combo = c("trace", "acf", "dens"))
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
    list(summary = summary, plot = plt)
}
