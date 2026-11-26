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
# ai4bayescode_run_chains() parallelises with fork(): worker processes
# forked with parallel::mcparallel(), or parallel::mclapply() when
# keep_models = FALSE. A multithreaded BLAS -- macOS Accelerate/vecLib above all, but also
# OpenBLAS with pthreads and MKL -- keeps a worker-thread pool; fork() copies
# only the calling thread, so the child inherits a pool whose threads do not
# exist and the first heavy BLAS call there dies. On macOS the crash is inside
# libdispatch (`_dispatch_root_queue_push` under a LAPACK entry point,
# reported as "crashed on child side of fork pre-exec"), and it aborts the
# whole R session -- it cannot be caught and retried.
#
# The cap has to be in place BEFORE the first BLAS call, because that call
# sets whether Accelerate uses its dispatch queues. Setting it later --
# inside ai4bayescode_run_chains(), say -- has no effect whatsoever. Hence:
# here, at load.
#
# Only variables the caller has not already set are touched.
#
# The cap costs nothing in practice: the chains are separate processes
# already, so n_cores workers x k BLAS threads each would only oversubscribe
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
    align_hdr <- file.path(inc_root, "AI4BayesCode", "arma_aligned_alloc.hpp")
    if (!file.exists(align_hdr)) {
        stop("Could not find ", align_hdr,
             "\nDid you pass the right AI4BayesCode_path?")
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
        # Armadillo allocates its heap arrays on 256-byte boundaries (see
        # include/AI4BayesCode/arma_aligned_alloc.hpp for the Accelerate
        # routines whose results depend on the operand addresses modulo 256
        # bytes). With -include the preprocessor processes the header before
        # the first line of the source, so before <RcppArmadillo.h>. The
        # relative name needs no quoting: the compiler resolves a relative
        # -include name against its working directory (Rcpp's build folder)
        # first, then R's include folder, then the -I folders in order; of
        # these, only the checkout's include root, the first -I of
        # PKG_CPPFLAGS, contains a folder AI4BayesCode.
        "-include AI4BayesCode/arma_aligned_alloc.hpp",
        extra_cppflags
    )

    libs <- if (Sys.info()[["sysname"]] == "Darwin") {
        c("-framework Accelerate", extra_libs)
    } else {
        c("$(BLAS_LIBS)", "$(LAPACK_LIBS)", extra_libs)
    }

    # Rcpp::sourceCpp keys its build cache on the source file alone, so with
    # rebuild = FALSE a module built earlier with other flags would be loaded
    # again. The build goes to a subfolder of the cache folder named after an
    # MD5 of the flags and of the force-included header.
    fp_file <- tempfile("ai4b_fp_")
    writeLines(c(cppflags, "\x1e", libs, "\x1e", unname(tools::md5sum(align_hdr))),
               fp_file, useBytes = TRUE)
    cache_dir <- file.path(getOption("rcpp.cache.dir", tempdir()),
                           paste0("ai4b-", substr(unname(tools::md5sum(fp_file)), 1L, 12L)))
    unlink(fp_file)

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
                    rebuild  = rebuild,
                    verbose  = verbose,
                    env      = globalenv(),
                    cacheDir = cache_dir)
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
    data_inputs <- data_inputs[!is.na(data_inputs) & nzchar(data_inputs)]

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
    # Per codegen.md, the user-facing DAG visualization shows the
    # generative / causal data flow, not the internal Gibbs full-
    # conditional graph. If you need to debug Gibbs dependencies, call
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
    # generative origin of each sampled-parameter root.
    context_edges <- collect_edges(dag$context_edges, "context")

    all_edges <- rbind(gibbs_reads_edges, refresh_edges,
                       predict_edges, context_edges)
    # Drop any edge with a missing/empty endpoint. A malformed DAG export
    # (e.g. an NA/"" node name) must not crash rendering with "missing value
    # where TRUE/FALSE needed" at the plate-detection any(), nor trip
    # graph_from_data_frame() with an unlisted vertex name.
    ok_edge <- !is.na(all_edges$from) & nzchar(all_edges$from) &
               !is.na(all_edges$to)   & nzchar(all_edges$to)
    all_edges <- all_edges[ok_edge, , drop = FALSE]
    all_nodes <- unique(c(all_edges$from, all_edges$to, data_inputs))
    all_nodes <- all_nodes[!is.na(all_nodes) & nzchar(all_nodes)]

    if (length(all_nodes) == 0) {
        if (is.null(out_path)) {
            graphics::plot.new()
            graphics::title(main = if (is.null(main)) "Model DAG (empty)" else main)
            graphics::text(0.5, 0.5, "No nodes declared", cex = 1.2, col = "grey50")
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
                plate_map <- stats::setNames(rep(NA_character_, length(all_nodes)),
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
        # Keep the STANDALONE context_edges table (used by the layout code
        # below) consistent with the collapsed graph. Without this, a context
        # edge onto a plate member -- e.g. `BART -> beta_2` while beta_0/1/2
        # collapse to a single `beta_i` node -- can no longer find its target,
        # so ctx_tgts() returns nothing and the layout computes
        # mean(numeric(0)) = NaN, which crashes downstream with "missing value
        # where TRUE/FALSE needed". (This is the VaryingCoefBART trigger.)
        if (nrow(context_edges) > 0L) {
            context_edges$from <- vapply(context_edges$from, collapse, character(1))
            context_edges$to   <- vapply(context_edges$to,   collapse, character(1))
            context_edges <- unique(
                context_edges[context_edges$from != context_edges$to, , drop = FALSE])
        }
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
    # The sampled-vs-predictive split is computed from the PREDICT subgraph
    # only -- context (prior) edges are excluded so a sampled parameter with an
    # incoming prior edge is NOT mis-classified as predictive.
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
    # Context (prior / hyperprior) edges must NOT influence ranking. We
    # Sugiyama-rank the predict subgraph only, then drop each context-only
    # node in as a satellite just above its first target.
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
    if (nrow(context_edges) > 0L) {
        idx_of   <- stats::setNames(seq_along(vnames), vnames)
        pred_in  <- intersect(predict_nodes, vnames)
        ctx_only <- intersect(setdiff(unique(context_edges$from),
                                       predict_nodes), vnames)

        if (length(ctx_only) > 0L && length(pred_in) > 0L) {
            ctx_tgts <- function(cn)
                intersect(context_edges$to[context_edges$from == cn], vnames)

            # --- context depth via fixpoint relaxation ---
            depth <- stats::setNames(rep(NA_integer_, length(ctx_only)), ctx_only)
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
                    # If a context node's targets are absent (e.g. mismatch
                    # after a collapse), keep its existing x rather than
                    # mean(numeric(0)) = NaN, which would crash rendering.
                    xs <- layout[idx_of[tg], 1]
                    if (length(xs)) layout[idx_of[cn], 1] <- mean(xs)
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
        op <- graphics::par(mar = c(1, 1, 3, 11))
        on.exit(graphics::par(op), add = TRUE)

        # Auto-scale vertex size + label size by node count so short names
        # fit INSIDE the circles.
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
        graphics::par(xpd = NA)
        graphics::legend("topright", inset = c(-0.24, 0),
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
        graphics::legend("bottomright", inset = c(-0.28, 0.04),
               legend = c("predict (generative)", "prior context"),
               col    = c("#D62728", "#AAAAAA"),
               lty    = c(1, 2), lwd = c(2, 1.5),
               cex = 0.78, bty = "n", title = "edges")
    }

    if (is.null(out_path)) {
        # Default: interactive on-screen render to current graphics device.
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
# ai4bayescode_perf_hint (deprecated; the timing is in run$wall)
#
# Emits a post-run per-sweep performance message. When the modular-NUTS
# per-sweep time is high it suggests the joint_nuts_block escape hatch; when
# the runner already uses joint NUTS it prints tuning hints instead.
#
# Arguments
#   wall_sec         : total wall-clock time across ALL chains (seconds)
#   n_sweeps_total   : total number of sweeps across ALL chains (including
#                      warmup). Used to compute per-sweep time.
#   uses_joint_nuts  : logical; set TRUE when the runner already uses
#                      joint_nuts_block. Suppresses the "switch to joint"
#                      hint.
#   thresholds       : optional list with `slow_sweep_sec` (default 0.5).
#                      Hints are emitted only when per-sweep time exceeds
#                      this.
#
# Returns nothing; prints to stderr via message().
# ----------------------------------------------------------------------------
ai4bayescode_perf_hint <- function(wall_sec,
                                n_sweeps_total,
                                uses_joint_nuts = FALSE,
                                thresholds = list(slow_sweep_sec = 0.5)) {
    .Deprecated(msg = paste0(
        "ai4bayescode_perf_hint() is deprecated and will be removed.\n",
        "  It asks you to compute and pass two numbers that ",
        "ai4bayescode_run_chains()\n",
        "  already returns -- `wall` per chain, and the sweep count from ",
        "n_burn + n_keep --\n",
        "  and its 0.5s/sweep threshold is a fixed number that ignores the ",
        "size of the\n",
        "  model. Read the timing off the run object instead:\n",
        "      run <- ai4bayescode_run_chains(...)\n",
        "      sum(run$wall)"))
    if (n_sweeps_total <= 0) return(invisible(NULL))
    per_sweep <- wall_sec / n_sweeps_total
    message(sprintf(
        "[AI4BayesCode perf] total %.1fs across %d sweeps (%.3fs / sweep)",
        wall_sec, n_sweeps_total, per_sweep))

    # thresholds = list() is a legal value of a list-typed argument, and
    # `list()$slow_sweep_sec` is NULL, so the comparison below became
    # `argument is of length zero` AFTER the timing line had already printed.
    slow_sweep_sec <- thresholds$slow_sweep_sec
    if (is.null(slow_sweep_sec)) slow_sweep_sec <- 0.5
    if (per_sweep <= slow_sweep_sec) {
        message("[AI4BayesCode perf] per-sweep time looks OK.")
        return(invisible(NULL))
    }

    if (uses_joint_nuts) {
        message(
            "[AI4BayesCode perf] per-sweep time is high even with joint_nuts_block.\n",
            "  Possible causes: (a) N * J grad eval is genuinely expensive,\n",
            "  (b) the sampler is still tuning -> run a longer warmup: raise\n",
            "      n_burn in ai4bayescode_run_chains(), or call\n",
            "      m$readapt_NUTS(2000L, FALSE, -1L) before the sampling steps,\n",
            "  (c) mass-matrix adaptation not yet converged -> longer warmup.")
        return(invisible(NULL))
    }

    message(
        "[AI4BayesCode perf] per-sweep time is high.\n",
        "  If this sampler has tightly-coupled continuous parameters in the\n",
        "  likelihood (e.g. additive linear mean, shift invariance,\n",
        "  fixed+random effects sharing mean), consider regenerating with\n",
        "  joint_nuts_block over the coupled parameters. For a reference\n",
        "  model, see ai4bayescode_example(\"IRT1PL_joint\").")
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
    # leading rows). A model's get_history() includes the warmup draws unless
    # clear_history() was called after them; for such a history callers pass
    # the warmup length, and n_burn = 0 leaves hist untouched. Same as Python
    # diagnose()'s `np.asarray(x)[n_burn:]` + the "no post-burn-in draws" ValueError.
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
            # covariance history is n x r x c, and `x[, j]` on a 3-D array
            # raises "incorrect number of dimensions" -- a shape that the
            # documentation of ai4bayescode_run_chains lists as supported.
            if (length(dim(x)) > 2L) x <- matrix(x, nrow = dim(x)[1L])
            for (j in seq_len(ncol(x))) cols[[sprintf("%s[%d]", nm, j)]] <- x[, j]
        }
    }
    M <- do.call(cbind, cols); colnames(M) <- names(cols)
    # posterior::summarise_draws happily returns an R-hat near 1 for a column
    # containing Inf/NaN, so a non-finite draw reads as "converged". Say so.
    .bad <- colnames(M)[apply(M, 2L, function(v) any(!is.finite(v)))]
    if (length(.bad))
        warning("non-finite draws in: ", paste(.bad, collapse = ", "),
                ". The diagnostics below are not meaningful for those.",
                call. = FALSE)
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
        # fixed = TRUE makes the pattern a LITERAL string, so "\\." searched
        # for a backslash followed by a dot and never matched anything --
        # the guard below was dead and a dot-path silently froze the child
        # at whatever value it happened to hold. The literal is ".".
        dot_names <- grepl(".", nms, fixed = TRUE)
        if (any(dot_names)) {
            stop(sprintf(
                "ai4bayescode_new_frozen(): dot-path names not allowed in `fixed` (%s). Use post-construction m$set_current(...) at the correct composite level + m$freeze(<dot.path>) instead.",
                paste(nms[dot_names], collapse = ", ")))
        }
    }
    m <- methods::new(module_class, ...)
    if (length(fixed) > 0L) {
        m$set_current(fixed)
        m$freeze(names(fixed), quiet = isTRUE(quiet_freeze))
    }
    m
}


# ---------------------------------------------------------------------------
# Multi-chain driver + cross-chain convergence summary.
#
# The functions of r-pkg/R/helpers.R for a checkout-mode user, who sources
# THIS file instead of library(AI4BayesCode), so that the generated-example
# flow is the same:
#     run <- ai4bayescode_run_chains(function(seed) new(<Class>, ..., seed, TRUE),
#                                    n_chains = 4L, n_burn = 4000L, n_keep = 4000L)
#     ai4bayescode_rhat_summary(run)
#     ai4bayescode_diagnose(run$histories[[1]])
# Keep every function in sync with r-pkg/R/helpers.R (single source of
# truth: the package); their documentation is there (?ai4bayescode_run_chains,
# ?ai4bayescode_predict, ?ai4bayescode_close_models, ...).
#
# ai4bayescode_run_chains(): with keep_models = TRUE (default) returns an
#   "ai4bayescode_run" list with histories, seeds, wall, models, n_burn,
#   n_keep and history_cleared; with keep_models = FALSE a plain list with
#   histories, seeds, wall, n_burn, n_keep and history_cleared. Histories
#   are KEEP-ONLY: each chain calls clear_history() after the warmup when the
#   model has that method, and for a model without it the first n_burn draws
#   of every entry are stripped along its first axis. run$models holds
#   handles to persistent forked worker processes after a parallel run (at
#   most max_workers of them over all kept runs; the least recently used
#   runs are closed first), the objects themselves after a sequential one.
#   For chain i, set.seed(seeds[i]) is called immediately before
#   model_ctor(seeds[i]) in every path, and the caller's .Random.seed (or
#   its absence) is restored after the run. An R error in a chain stops the
#   run with "chain i (seed s): <message>"; a worker process that exits
#   during the run makes every chain run again sequentially.
# ai4bayescode_predict() / predict(run): predict_at() over all chains.
# ai4bayescode_close_models(): terminates the worker processes of a run.
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
                                 verbose   = TRUE,
                                 n_cores   = NULL,
                                 keep_models = TRUE) {
    stopifnot(is.function(model_ctor),
              n_chains >= 1, n_burn >= 0, n_keep >= 1)

    # .ai4b_rc_one_chain() calls set.seed(seeds[i]) before it constructs
    # chain i. The caller's .Random.seed (NULL when it does not exist) is
    # saved here and restored on exit in every path: after the sequential
    # runs, after parallel::mclapply() (which with RNGkind "L'Ecuyer-CMRG"
    # creates .Random.seed when none exists), and after the gc() that
    # precedes the start of worker processes (a BART model of the caller that
    # it destroys writes R's generator state to .Random.seed through its
    # Rcpp::RNGScope).
    rng_caller <- .ai4b_rc_rng_get()
    on.exit(.ai4b_rc_rng_set(rng_caller), add = TRUE)

    if (is.null(seeds)) {
        seeds <- as.integer(101L + (seq_len(n_chains) - 1L) * 101L)
    }
    if (length(seeds) != n_chains) {
        stop("length(seeds) must equal n_chains")
    }

    # mc.cores is the earlier name of n_cores and is still accepted.
    if (!is.null(n_cores) && !is.null(mc.cores) &&
        !identical(as.numeric(n_cores), as.numeric(mc.cores)))
        stop("ai4bayescode_run_chains(): n_cores = ", n_cores,
             " and mc.cores = ", mc.cores, " differ; mc.cores is an alias ",
             "of n_cores, so pass only one of them.", call. = FALSE)
    if (is.null(n_cores)) n_cores <- mc.cores
    n_burn <- as.integer(n_burn)
    n_keep <- as.integer(n_keep)

    # keep = FALSE drops the model object from the result, so that it is not
    # serialized back from a forked mclapply worker.
    one_chain <- function(seed_val, keep = FALSE) {
        r <- .ai4b_rc_one_chain(model_ctor, seed_val, n_burn, n_keep)
        if (!keep) r$model <- NULL
        r
    }

    # An R error in chain i is raised as "chain i (seed s): <message>", with
    # the classes of the original condition.
    chain_error <- function(i, message, class)
        stop(structure(class = unique(c(class, "error", "condition")),
                       list(message = sprintf(
                                "ai4bayescode_run_chains: chain %d (seed %s): %s",
                                i, format(seeds[[i]]), message),
                            call = NULL)))
    run_sequential <- function(keep)
        lapply(seq_len(n_chains), function(i)
            tryCatch(one_chain(seeds[[i]], keep), error = function(e)
                chain_error(i, conditionMessage(e), class(e))))

    use_parallel <- parallel && .Platform$OS.type != "windows" &&
                    requireNamespace("parallel", quietly = TRUE)

    if (is.null(n_cores)) {
        n_cores <- if (use_parallel)
            min(n_chains, max(1, parallel::detectCores() - 1))
        else 1
    }

    chain_ok <- function(r) !inherits(r, "try-error") && !is.null(r$history)

    # The BLAS is pinned to one thread in .onLoad (zzz.R), which is the only
    # place it can work -- the FIRST BLAS call sets whether Accelerate uses
    # its dispatch queues, so a cap applied here would change nothing. If a variable is not "1" by now the caller overrode it, and
    # forking a BLAS-heavy model may abort the session.
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

    models <- NULL
    if (keep_models && use_parallel && n_cores >= 2 && n_chains > 1) {
        # Persistent forked workers: the model of each chain is kept in its
        # worker process, and run$models holds handles to them.
        n_workers <- as.integer(min(n_cores, n_chains))
        if (verbose)
            message("ai4bayescode_run_chains: running ", n_chains,
                    " chains on ", n_workers, " cores (parallel)")
        warn_if_blas_unpinned()
        pr <- .ai4b_rc_pool_run(model_ctor, seeds, n_burn, n_keep, n_workers)
        if (identical(pr$kind, "chain"))
            chain_error(pr$chain, pr$message, pr$class)
        if (identical(pr$kind, "done")) {
            results <- pr$results
            models  <- pr$handles
        } else {
            # The worker processes could not be started, or one of them
            # exited during the run (a fork-unsafe multithreaded BLAS --
            # notably macOS Accelerate -- makes a forked chain segfault). The
            # pool is closed by now; every chain runs sequentially here.
            warning("ai4bayescode_run_chains: ", switch(pr$kind,
                lost = paste0(pr$message, "; re-running all chains ",
                              "sequentially in this session. If a fork-unsafe ",
                              "multithreaded BLAS (e.g. macOS Accelerate) is ",
                              "the cause, set VECLIB_MAXIMUM_THREADS=1 before ",
                              "starting R to keep parallelism, or pass ",
                              "parallel = FALSE."),
                start = paste0("the worker processes could not be started (",
                               pr$message, "); the chains run sequentially ",
                               "in this session."),
                pr$message), call. = FALSE, immediate. = TRUE)
            results <- run_sequential(keep = TRUE)
            models  <- lapply(results, `[[`, "model")
        }
    } else if (!keep_models && use_parallel && n_cores >= 2 && n_chains > 1) {
        if (verbose)
            message("ai4bayescode_run_chains: running ", n_chains,
                    " chains on ", n_cores, " cores (parallel)")
        warn_if_blas_unpinned()
        results <- parallel::mclapply(seq_len(n_chains), function(i)
            tryCatch(one_chain(seeds[[i]]), error = function(e)
                structure(list(message = conditionMessage(e), class = class(e)),
                          class = "ai4b_rc_chain_error")),
            mc.cores = n_cores, mc.set.seed = TRUE)
        err <- which(vapply(results, inherits, logical(1), "ai4b_rc_chain_error"))
        if (length(err))
            chain_error(err[1L], results[[err[1L]]]$message,
                        results[[err[1L]]]$class)
        if (!all(vapply(results, chain_ok, logical(1)))) {
            # A forked worker exited before returning a history. The usual
            # cause is a fork-unsafe multithreaded BLAS -- notably macOS
            # Accelerate (vecLib), whose GCD worker threads are not copied by
            # fork(), so a chain doing heavier linear algebra segfaults under
            # mclapply. Every chain is re-run sequentially (no fork).
            warning("ai4bayescode_run_chains: a forked worker process exited ",
                    "before it returned its chains (likely a fork-unsafe ",
                    "multithreaded BLAS, e.g. macOS Accelerate); re-running ",
                    "all chains sequentially. To keep parallelism set ",
                    "VECLIB_MAXIMUM_THREADS=1 before starting R, or pass ",
                    "parallel = FALSE.", call. = FALSE, immediate. = TRUE)
            results <- run_sequential(keep = FALSE)
        }
    } else {
        if (verbose)
            message("ai4bayescode_run_chains: running ", n_chains,
                    " chains sequentially")
        results <- run_sequential(keep = keep_models)
        if (keep_models) models <- lapply(results, `[[`, "model")
    }

    for (i in seq_along(results)) {
        if (!chain_ok(results[[i]])) {
            stop("ai4bayescode_run_chains: chain ", i, " failed")
        }
    }
    histories <- lapply(results, `[[`, "history")
    seeds_out <- sapply(results, `[[`, "seed")
    wall      <- sapply(results, `[[`, "wall")
    cleared   <- vapply(results, function(r) isTRUE(r$cleared), logical(1))
    if (!keep_models)
        return(list(histories = histories, seeds = seeds_out, wall = wall,
                    n_burn = n_burn, n_keep = n_keep,
                    history_cleared = cleared))
    .ai4b_rc_history_notes(results, n_burn, n_keep)
    structure(list(
        histories       = histories,
        seeds           = seeds_out,
        wall            = wall,
        models          = models,
        n_burn          = n_burn,
        n_keep          = n_keep,
        history_cleared = cleared
    ), class = "ai4bayescode_run")
}


# ai4bayescode_predict(run, newdata = list(), pool = TRUE): predict_at(newdata)
# on the model of every chain of a keep_models = TRUE run; pool = TRUE stacks
# the chains along the first axis per output key, pool = FALSE returns one
# list per chain.
ai4bayescode_predict <- function(run, newdata = list(), pool = TRUE) {
    models <- if (is.list(run)) run$models
    if (is.null(models))
        stop("ai4bayescode_predict(): the run holds no models (run$models is ",
             "NULL). Rerun ai4bayescode_run_chains() with keep_models = TRUE.",
             call. = FALSE)
    if (!isTRUE(pool) && !isFALSE(pool))
        stop("ai4bayescode_predict(): pool must be TRUE or FALSE.", call. = FALSE)
    newdata <- .ai4b_rc_check_newdata(newdata)
    n_chains <- length(models)
    preds <- vector("list", n_chains)
    is_handle <- vapply(models, inherits, logical(1),
                        what = "ai4bayescode_chain_model")
    if (any(is_handle)) {
        # One request per worker process for all of its chains.
        hidx  <- which(is_handle)
        pools <- lapply(models[hidx], .subset2, "pool")
        for (p in unique(pools)) {
            sel  <- hidx[vapply(pools, identical, logical(1), p)]
            caps <- .ai4b_rc_remote(p, vapply(models[sel], .subset2, integer(1),
                                              "chain"),
                                    "predict_at", list(newdata))
            for (k in seq_along(sel))
                preds[sel[k]] <- list(.ai4b_rc_relay(caps[[k]], sel[k]))
        }
    }
    for (i in which(!is_handle)) {
        preds[i] <- list(tryCatch(models[[i]]$predict_at(newdata),
            error = function(e) stop(structure(
                class = unique(c(class(e), "error", "condition")),
                list(message = sprintf("chain %d: %s", i, conditionMessage(e)),
                     call = NULL)))))
    }
    for (i in seq_len(n_chains)) .ai4b_rc_check_prediction(preds[[i]], i)

    # An entry with n_burn + n_keep rows holds the warmup draws too: from a
    # model without clear_history(), or from per-sweep values that a model
    # stores outside the buffers clear_history() empties. Its first n_burn
    # rows are dropped; for a model that has clear_history(), with a warning.
    n_burn  <- if (is.null(run$n_burn)) 0L else as.integer(run$n_burn)
    n_keep  <- if (is.null(run$n_keep)) 0L else as.integer(run$n_keep)
    cleared <- run$history_cleared
    if (length(cleared) != n_chains) cleared <- rep(FALSE, n_chains)
    if (n_burn > 0L) {
        for (i in seq_len(n_chains)) {
            full <- vapply(preds[[i]], function(x)
                .ai4b_rc_n_rows(x) == n_burn + n_keep, logical(1))
            if (!any(full)) next
            preds[[i]][full] <- lapply(preds[[i]][full], .ai4b_rc_strip_burn,
                                       nb = n_burn)
            if (isTRUE(cleared[i]))
                warning(sprintf(paste0(
                    "chain %d: predict_at() returned n_burn + n_keep = %d rows ",
                    "for %s although clear_history() was called after the ",
                    "warmup; the first n_burn = %d rows (the warmup draws) were ",
                    "dropped. The model stores these draws outside the buffers ",
                    "that clear_history() empties."),
                    i, n_burn + n_keep,
                    paste(sQuote(names(preds[[i]])[full], FALSE), collapse = ", "),
                    n_burn), call. = FALSE)
        }
    }
    if (isFALSE(pool)) return(preds)
    if (n_keep > 0L) {
        odd <- unique(unlist(lapply(preds, function(p)
            names(p)[vapply(p, function(x) !(.ai4b_rc_n_rows(x) %in% c(1L, n_keep)),
                            logical(1))])))
        if (length(odd))
            warning(sprintf(paste0(
                "ai4bayescode_predict(): the first dimension of %s is neither ",
                "n_keep = %d nor 1, so it is not one row per stored draw; ",
                "pool = TRUE stacks the chains along it. Use pool = FALSE to get ",
                "the entries of each chain unchanged."),
                paste(sQuote(odd, FALSE), collapse = ", "), n_keep), call. = FALSE)
    }
    .ai4b_rc_pool_predictions(preds)
}

# predict(run, newdata, pool) is the same call; any other argument raises an
# error.
predict.ai4bayescode_run <- function(object, newdata = list(), pool = TRUE, ...) {
    if (...length()) {
        nm <- names(list(...))
        if (is.null(nm)) nm <- character(...length())
        nm[!nzchar(nm)] <- "(unnamed)"
        stop(sprintf(paste0("predict(): unused argument%s %s; predict() for an ",
                            "ai4bayescode_run takes newdata and pool."),
                     if (length(nm) == 1L) "" else "s",
                     paste(sprintf("'%s'", nm), collapse = ", ")),
             call. = FALSE)
    }
    ai4bayescode_predict(object, newdata = newdata, pool = pool)
}


# ai4bayescode_close_models(run): terminate the worker processes that hold the
# models of a parallel keep_models = TRUE run (idempotent; a no-op otherwise).
ai4bayescode_close_models <- function(run) {
    for (p in .ai4b_rc_run_pools(run))
        .ai4b_rc_pool_close(p, .ai4b_rc_reason("close_models"))
    invisible(NULL)
}


# Compact print of a run: settings, history entry dimensions, wall time, and
# where the models are.
print.ai4bayescode_run <- function(x, ...) {
    n_chains <- length(x$histories)
    cat(sprintf("AI4BayesCode run: %d chain%s, n_burn = %s warmup + n_keep = %s kept sweeps per chain\n",
                n_chains, if (n_chains == 1L) "" else "s",
                format(x$n_burn), format(x$n_keep)))
    h <- if (n_chains > 0L) x$histories[[1L]] else list()
    if (length(h)) {
        nm <- names(h)
        if (is.null(nm)) nm <- rep("", length(h))
        nm[!nzchar(nm)] <- sprintf("[[%d]]", which(!nzchar(nm)))
        dims <- vapply(h, function(e)
            paste(if (is.null(dim(e))) length(e) else dim(e), collapse = " x "),
            character(1))
        show <- seq_len(min(length(h), 10L))
        cat("History entries (draws along the first axis):\n")
        cat(sprintf("  %s  %s\n", formatC(nm[show], width = -max(nchar(nm[show]))),
                    dims[show]), sep = "")
        if (length(h) > 10L) cat(sprintf("  ... and %d more\n", length(h) - 10L))
    }
    if (length(x$wall))
        cat("Wall time per chain: ", format(signif(min(x$wall), 3)), " to ",
            format(signif(max(x$wall), 3)), " s (sum ",
            format(signif(sum(x$wall), 3)), " s)\n", sep = "")
    cat("Models: ", .ai4b_rc_models_where(x$models), "\n", sep = "")
    if (!is.null(x$models) && isTRUE(x$n_burn > 0) &&
        !all(x$history_cleared))
        cat("  (models without clear_history(): their get_history() and",
            "predict_at() include the warmup draws)\n")
    invisible(x)
}


# Handle to the model of one chain in a worker process: h$method(...) calls
# the method there and returns its result (output, messages, warnings and
# errors are shown here); assignment through a handle raises an error.
`$.ai4bayescode_chain_model` <- function(x, name) {
    force(name)
    function(...) .ai4b_rc_handle_call(x, name, list(...))
}

`$<-.ai4bayescode_chain_model` <- function(x, name, value)
    stop("assignment through an ai4bayescode_chain_model handle is not ",
         "supported; call a method of the model instead, e.g. ",
         "h$set_current(list(...)).", call. = FALSE)

.DollarNames.ai4bayescode_chain_model <- function(x, pattern = "") {
    nms <- .subset2(x, "methods")
    tryCatch(grep(pattern, nms, value = TRUE), error = function(e) nms)
}

print.ai4bayescode_chain_model <- function(x, ...) {
    pool  <- .subset2(x, "pool")
    chain <- .subset2(x, "chain")
    cat(sprintf("<ai4bayescode_chain_model: chain %d of %d, %s>\n", chain,
                pool$n_chains, .subset2(x, "class_name")))
    w <- pool$node_of[chain]
    txt <- switch(.ai4b_rc_pool_state(pool),
        open = {
            entry <- get(pool$token, envir = .ai4b_rc_registry(), inherits = FALSE)
            if (is.na(entry$lost[w]))
                sprintf(paste0("The model is in worker process %d of this R ",
                               "session; x$<method>(...) is evaluated there."),
                        pool$pids[w])
            else .ai4b_rc_sentence(entry$lost[w])
        },
        closed = .ai4b_rc_sentence(.ai4b_rc_closed_reason(pool)$message),
        child  = sprintf(paste0("Created in R process %d; a handle cannot be ",
                                "called in a forked child process of it."),
                         pool$owner_pid),
        other  = paste0("Created in another R session or process; the model ",
                        "is not available here."))
    cat("  ", txt, "\n", sep = "")
    invisible(x)
}


# ---------------------------------------------------------------------------
# Internals of ai4bayescode_run_chains() / ai4bayescode_predict().
#
# A parallel run with keep_models = TRUE forks its worker processes with
# parallel::mcparallel(detached = TRUE). The constructor reaches the workers
# through fork() (it is stored in .ai4b_rc_env immediately before forking and
# removed from the parent right after), never through serialization: its
# environment can hold an Rcpp class generator, whose external pointer would
# be NULL after unserialize(). Each worker connects to a listening socket of
# the parent and then serves the requests of the parent on that one
# connection (.ai4b_rc_worker_main()); the models of its chains are stored in
# its own copy of .ai4b_rc_env$models. Requests and replies hold data only
# (seeds, chain indices, method names, arguments, values), so a worker runs
# the copy of these functions that fork() made, wherever this file was
# sourced.
#
# In the parent, .ai4b_rc_env$pools maps the token of every open pool to its
# connections, worker pids and start times, request counters and the
# messages of lost workers. A handle refers to a "pool" environment holding
# the token, the session id and the chain-to-worker map, but not the
# connections, so a handle restored with readRDS() in another session
# contains no connection. A finalizer on the pool environment (onexit =
# TRUE) closes the pool when the run and its handles are garbage-collected,
# or when R exits. The reason for which a pool was closed is recorded in
# .ai4b_rc_env$closed under its token, where every copy of the run in this
# session finds it.
#
# Messages between the parent and a worker are frames: the bytes "A4B1", the
# length of the body (4 bytes, little-endian) and the body, a list
# serialized with xdr = FALSE that holds the number `seq` of its request.
# The parent sends a request to a worker only after it has read the replies
# to all earlier requests of that worker, and a worker writes a reply only
# after it has read the request, so each direction of a connection holds at
# most one frame. A frame has at most 4096 bytes: it is written with one
# send() into an empty socket buffer and read with one recv() (R reads a
# socket in blocks of 4096 bytes). R raises an interrupt in a socket
# operation only in the select() before the send() or recv(); the frame
# operations run under suspendInterrupts(), so an interrupt leaves either
# none or all of a frame in the connection, with the request counters
# updated accordingly. A message whose serialization is longer is written
# to a file in tempdir(), and its frame carries the file name; the reader
# deletes the file. Each request names the file for a long reply. When a
# worker process is terminated or found lost, the parent deletes the files
# of its last request after the process has exited, so the reply of an
# interrupted call that was never read leaves no file behind.
# ---------------------------------------------------------------------------
# A second source() of this file leaves the registry of open pools in place.
if (!exists(".ai4b_rc_env", inherits = FALSE))
    .ai4b_rc_env <- new.env(parent = emptyenv())

# Id of this R process: pid, temporary directory and creation time. A forked
# child gets a new id at its first call, because its pid differs.
.ai4b_rc_session <- function() {
    s <- .ai4b_rc_env$session
    if (is.null(s) || !identical(s$pid, Sys.getpid())) {
        s <- list(pid = Sys.getpid(),
                  id  = paste(Sys.getpid(), tempdir(),
                              format(Sys.time(), "%Y%m%d%H%M%OS6"), sep = "|"))
        .ai4b_rc_env$session <- s
    }
    s$id
}

.ai4b_rc_registry <- function() {
    r <- .ai4b_rc_env$pools
    if (is.null(r)) {
        r <- new.env(parent = emptyenv())
        .ai4b_rc_env$pools <- r
    }
    r
}

# Number of draws of one history entry: the length of its first axis.
.ai4b_rc_n_rows <- function(x) {
    d <- dim(x)
    if (is.null(d)) length(x) else d[1L]
}

# Drop the first `nb` draws along the FIRST axis of one history entry.
# Entries can be plain vectors (n), matrices (n x d), or higher-D
# arrays (n x r x c, ...); the slice runs along axis 1 only and keeps
# every other axis (and names/dimnames) intact. An entry with <= nb
# rows is returned unchanged: with keep_history disabled get_history()
# holds a single, already post-warmup, current draw -- emptying it
# would be worse than keeping it.
.ai4b_rc_strip_burn <- function(x, nb) {
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

# TRUE when `name` is a method of `m`. The lookup uses `$`, which finds the
# methods of an Rcpp module object as well as the functions of a list or an
# environment.
.ai4b_rc_has_method <- function(m, name)
    is.function(tryCatch(eval(call("$", m, name)), error = function(e) NULL))

.ai4b_rc_class_name <- function(m) sub("^Rcpp_", "", class(m)[1L])

# One chain: seed R's random-number generator, construct, warm up, clear the
# warmup draws, sample. set.seed(seed_val) is called immediately before
# model_ctor(seed_val) in every path (worker process, mclapply, this
# session), so the random numbers that a model draws from R's generator
# (bart_block, genbart_block and softbart_block under Rcpp) are determined by
# seed_val and the current RNGkind(). clear_history() is called only when the
# model has that method. The history returned is keep-only; `model` is the
# model object itself.
.ai4b_rc_one_chain <- function(model_ctor, seed_val, n_burn, n_keep) {
    t0 <- Sys.time()
    set.seed(seed_val)
    m <- model_ctor(seed_val)
    m$step(n_burn)
    has_clear <- .ai4b_rc_has_method(m, "clear_history")
    cleared   <- n_burn > 0L && has_clear
    if (cleared) m$clear_history()
    m$step(n_keep)
    t1 <- Sys.time()
    h <- lapply(m$get_history(), identity)
    extra <- character(0)
    if (cleared) {
        # After clear_history() every entry holds n_keep draws (1 when the
        # model stores a single draw). An entry with more rows was not
        # emptied: drop its warmup draws here and report it.
        idx <- which(vapply(h, .ai4b_rc_n_rows, numeric(1)) > n_keep)
        h[idx] <- lapply(h[idx], .ai4b_rc_strip_burn, nb = n_burn)
        extra  <- if (is.null(names(h))) sprintf("[[%d]]", idx) else names(h)[idx]
    } else {
        # get_history() spans EVERY stepped iteration (warmup + keep);
        # return keep-only draws, matching this function's documentation.
        h <- lapply(h, .ai4b_rc_strip_burn, nb = n_burn)
    }
    list(history   = h,
         seed      = seed_val,
         wall      = as.numeric(difftime(t1, t0, units = "secs")),
         cleared   = cleared,
         has_clear = has_clear,
         extra     = extra,
         class     = .ai4b_rc_class_name(m),
         model     = m)
}

# The message for models without clear_history() and the warning for entries
# that clear_history() left non-empty, each emitted once per run. Called only
# for a run that returns its models (keep_models = TRUE).
.ai4b_rc_history_notes <- function(results, n_burn, n_keep) {
    cls <- function(rs) paste(unique(vapply(rs, `[[`, character(1), "class")),
                              collapse = ", ")
    no_clear <- Filter(function(r) !isTRUE(r$has_clear), results)
    if (n_burn > 0L && length(no_clear))
        message("ai4bayescode_run_chains: ", cls(no_clear), " has no ",
                "clear_history() method, so the model's own get_history() and ",
                "predict_at() include the n_burn = ", n_burn, " warmup draws ",
                "(run$histories exclude them, and ai4bayescode_predict() drops ",
                "them). A compiled model has clear_history() when its Rcpp ",
                "module binds AI4BAYESCODE_BIND_KERNEL_CONTROL() and it is ",
                "compiled against AI4BayesCode >= 1.1.3.")
    with_extra <- Filter(function(r) length(r$extra) > 0L, results)
    if (length(with_extra))
        warning("ai4bayescode_run_chains: after clear_history(), the history ",
                "entries ", paste(unique(unlist(lapply(with_extra, `[[`, "extra"))),
                                  collapse = ", "),
                " of ", cls(with_extra), " have more than n_keep = ", n_keep,
                " rows. Their first n_burn = ", n_burn, " rows are dropped in ",
                "run$histories, but the model's own get_history() and ",
                "predict_at() still contain the warmup draws for them: the ",
                "class has no clear_extra_history_() hook that empties these ",
                "entries.", call. = FALSE)
    invisible(NULL)
}

# .Random.seed of the global environment, or NULL when it does not exist.
.ai4b_rc_rng_get <- function()
    get0(".Random.seed", envir = globalenv(), inherits = FALSE)

# Restore a value of .ai4b_rc_rng_get(): assign .Random.seed, or remove it
# when the value is NULL, then call RNGkind(), which loads .Random.seed into
# R's generator (GetRNGstate; with no .Random.seed the generator is seeded
# from the clock). Without the RNGkind() call the generator state after the
# run is the state at the end of the last chain, and the Rcpp::RNGScope
# member of the `arn` generator of bart_block and genbart_block copies that
# state to .Random.seed (PutRNGstate) each time a model containing one is
# garbage-collected.
.ai4b_rc_rng_set <- function(seed) {
    if (is.null(seed)) {
        if (exists(".Random.seed", envir = globalenv(), inherits = FALSE))
            rm(".Random.seed", envir = globalenv(), inherits = FALSE)
    } else {
        assign(".Random.seed", seed, envir = globalenv())
    }
    RNGkind()
    invisible(NULL)
}

# Evaluate fun() and return its value with the printed output, messages,
# warnings and error it produced, so that the parent can show them. An
# interrupt (SIGINT) raised during fun() ends it with an error.
.ai4b_rc_capture <- function(fun) {
    value <- NULL; visible <- FALSE; err <- NULL
    warns <- character(0); msgs <- character(0)
    out <- utils::capture.output(invisible(withCallingHandlers(
        tryCatch({
            v <- withVisible(fun())
            value   <- v$value
            visible <- v$visible
        }, error = function(e)
            err <<- list(message = conditionMessage(e), class = class(e)),
        interrupt = function(e)
            err <<- list(message = "interrupted in the worker process (SIGINT)",
                         class = c("simpleError", "error", "condition"))),
        warning = function(w) {
            warns <<- c(warns, conditionMessage(w))
            invokeRestart("muffleWarning")
        },
        message = function(m) {
            msgs <<- c(msgs, conditionMessage(m))
            invokeRestart("muffleMessage")
        })))
    list(value = value, visible = visible, error = err,
         warnings = warns, messages = msgs, output = out)
}

# Show the output, messages and warnings of a captured call of chain `chain`,
# then raise its error (with the error's classes) or return its value.
.ai4b_rc_relay <- function(cap, chain) {
    if (length(cap$output)) cat(cap$output, sep = "\n")
    for (m in cap$messages) message(m, appendLF = FALSE)
    for (w in cap$warnings) warning(w, call. = FALSE)
    if (!is.null(cap$error))
        stop(structure(class = unique(c(cap$error$class, "error", "condition")),
                       list(message = sprintf("chain %d: %s", chain,
                                              cap$error$message),
                            call = NULL)))
    if (isTRUE(cap$visible)) cap$value else invisible(cap$value)
}

# A captured call that failed with `message`.
.ai4b_rc_failed <- function(message)
    list(error = list(message = message,
                      class = c("simpleError", "error", "condition")))

# "a matrix", "NULL", ...: the kind of an R value, for error messages.
.ai4b_rc_describe <- function(x) {
    if (is.null(x)) return("NULL")
    k <- if (is.data.frame(x)) "data.frame" else if (is.matrix(x)) "matrix"
         else if (is.array(x)) "array" else class(x)[1L]
    paste(if (grepl("^[aeiou]", k)) "an" else "a", k)
}

# A message as a sentence: capital first letter and a final period.
.ai4b_rc_sentence <- function(s)
    paste0(toupper(substr(s, 1L, 1L)), substring(s, 2L), ".")

# ---- frames ----------------------------------------------------------------

.ai4b_rc_frame_magic <- as.raw(c(0x41, 0x34, 0x42, 0x31))

# The frame of message `msg` (a list with element `seq`), as
# list(bytes = <raw>, file = <path of the file holding msg, or NULL>). A
# long message is written to `file`, or to a new file in tempdir() when
# `file` is NULL.
.ai4b_rc_frame <- function(msg, file = NULL) {
    body <- if (utils::object.size(msg) < 1e6) serialize(msg, NULL, xdr = FALSE)
    if (!is.null(body) && length(body) <= 4088L) file <- NULL
    else {
        if (is.null(file)) file <- tempfile("ai4b_rc_", fileext = ".bin")
        con  <- file(file, "wb")
        done <- FALSE
        on.exit({
            close(con)
            if (!done) unlink(file)
        }, add = TRUE)
        if (is.null(body)) serialize(msg, con, xdr = FALSE) else writeBin(body, con)
        done <- TRUE
        body <- serialize(list(seq = msg$seq, spill = file), NULL, xdr = FALSE)
    }
    list(bytes = c(.ai4b_rc_frame_magic,
                   writeBin(length(body), raw(), size = 4L, endian = "little"),
                   body),
         file = file)
}

# Read one frame from `con` and return its list; NULL at the end of the
# connection. A malformed frame raises an error.
.ai4b_rc_read_frame <- function(con) {
    hdr <- readBin(con, "raw", 8L)
    if (length(hdr) < 8L) return(NULL)
    if (!identical(hdr[1:4], .ai4b_rc_frame_magic))
        stop("a malformed message was read from the connection")
    n <- readBin(hdr[5:8], "integer", size = 4L, endian = "little")
    body <- readBin(con, "raw", n)
    if (length(body) < n) return(NULL)
    unserialize(body)
}

# The message of a frame's list: the list itself, or the message in the file
# it names (the file is deleted).
.ai4b_rc_unpack <- function(m) {
    file <- m[["spill"]]
    if (is.null(file)) return(m)
    on.exit(unlink(file), add = TRUE)
    con <- file(file, "rb")
    on.exit(close(con), add = TRUE, after = FALSE)
    unserialize(con)
}

# Delete the file of a frame's list, if it has one.
.ai4b_rc_discard <- function(m) {
    if (!is.null(m[["spill"]])) unlink(m[["spill"]])
    invisible(NULL)
}

# ---- worker side -----------------------------------------------------------

# Body of a worker process forked by parallel::mcparallel(). It closes its
# copies of the listening socket, of the parent's connections to the workers
# started before it (`siblings`) and of the connections of other pools,
# sends its output to the null device, connects to the parent at `port`,
# sends `key`, its number `w` and its pid, and serves requests until a
# "close" request or the end of the connection: "run" runs the chains of the
# worker and stores their models, "call" calls a method of the model of each
# requested chain. The loop runs under suspendInterrupts(): R raises an
# interrupt (SIGINT, also the one of Ctrl+C in a terminal, which reaches the
# whole process group) only in a select(): in the select() before the next
# request is read, the interrupt is caught and the select() repeated, and in
# a method call it ends that call with an error; every request gets one
# reply.
.ai4b_rc_worker_main <- function(port, key, w, listener, siblings) {
    try(close(listener), silent = TRUE)
    for (cn in siblings) if (!is.null(cn)) try(close(cn), silent = TRUE)
    reg <- .ai4b_rc_env$pools
    if (is.environment(reg))
        for (tok in ls(reg))
            for (cn in get(tok, envir = reg, inherits = FALSE)$cons)
                if (!is.null(cn)) try(close(cn), silent = TRUE)
    .ai4b_rc_env$pools <- NULL
    nul <- file(nullfile(), open = "w")
    sink(nul)
    sink(nul, type = "message")
    con <- socketConnection("localhost", port = port, blocking = TRUE,
                            open = "a+b", timeout = 60)
    suspendInterrupts(.ai4b_rc_worker_loop(con, key, w))
    invisible(NULL)
}

.ai4b_rc_worker_loop <- function(con, key, w) {
    if (.ai4b_rc_worker_send(con, list(seq = 0L, key = key, worker = w,
                                       pid = Sys.getpid())))
        repeat {
            req <- .ai4b_rc_worker_recv(con)
            if (is.null(req) || identical(req$op, "close")) break
            out <- tryCatch(
                if (!is.null(req$error)) stop(req$error)
                else if (identical(req$op, "run")) .ai4b_rc_worker_run(req)
                else if (identical(req$op, "call")) .ai4b_rc_worker_call(req)
                else stop("unknown request"),
                error = function(e) e)
            reply <- if (inherits(out, "error"))
                list(seq = req$seq, error = conditionMessage(out))
            else list(seq = req$seq, out = out)
            file <- req$reply_file
            if (!is.character(file) || length(file) != 1L) file <- NULL
            if (!.ai4b_rc_worker_send(con, reply, file)) break
        }
    try(close(con), silent = TRUE)
    invisible(NULL)
}

# Read the next request (socketSelect() until the connection is readable);
# NULL at the end of the connection.
.ai4b_rc_worker_recv <- function(con) {
    repeat {
        m <- tryCatch(
            if (isTRUE(socketSelect(list(con), timeout = 3600)))
                .ai4b_rc_read_frame(con) else "again",
            interrupt = function(e) "again",
            error = function(e) NULL)
        if (!identical(m, "again")) break
    }
    if (is.null(m)) return(NULL)
    tryCatch(.ai4b_rc_unpack(m), error = function(e)
        list(seq = m$seq, error = paste0("the request could not be read (",
                                         conditionMessage(e), ")")))
}

# Write one reply, a long one to `file` (see .ai4b_rc_frame()); FALSE when
# the connection failed.
.ai4b_rc_worker_send <- function(con, msg, file = NULL) {
    fr <- tryCatch(.ai4b_rc_frame(msg, file), error = function(e)
        .ai4b_rc_frame(list(seq = msg$seq, error = paste0(
            "the reply could not be written (", conditionMessage(e), ")"))))
    repeat {
        ok <- tryCatch({
            writeBin(fr$bytes, con)
            TRUE
        }, interrupt = function(e) NA, error = function(e) FALSE)
        if (!is.na(ok)) break
    }
    if (!ok && !is.null(fr$file)) unlink(fr$file)
    ok
}

# Run the chains of this worker and store their models. The chains after
# the first one that fails are not run.
.ai4b_rc_worker_run <- function(req) {
    ctor <- get0("ctor", envir = .ai4b_rc_env, inherits = FALSE)
    if (exists("ctor", envir = .ai4b_rc_env, inherits = FALSE))
        rm(list = "ctor", envir = .ai4b_rc_env)
    models <- list()
    out <- list()
    for (j in seq_along(req$chains)) {
        cap <- .ai4b_rc_capture(function() {
            if (!is.function(ctor))
                stop("the model constructor is not available in this worker ",
                     "process")
            .ai4b_rc_one_chain(ctor, req$seeds[[j]], req$n_burn, req$n_keep)
        })
        if (is.null(cap$error)) {
            m <- cap$value$model
            models[[as.character(req$chains[[j]])]] <- m
            cap$value$model   <- NULL
            cap$value$methods <- tryCatch(as.character(utils::.DollarNames(m, "")),
                                          error = function(e) character(0))
        }
        out[[j]] <- cap
        if (!is.null(cap$error)) break
    }
    .ai4b_rc_env$models <- models
    out
}

# Call `req$method` with `req$args` on the model of each chain in
# `req$chains`.
.ai4b_rc_worker_call <- function(req) {
    models <- .ai4b_rc_env$models
    lapply(req$chains, function(i)
        .ai4b_rc_capture(function() {
            m <- models[[as.character(i)]]
            if (is.null(m))
                stop("this worker process holds no model of chain ", i)
            f <- eval(call("$", m, req$method))
            if (!is.function(f))
                stop("'", req$method, "' is not a method of ",
                     .ai4b_rc_class_name(m))
            do.call(f, req$args, quote = TRUE)
        }))
}

# ---- parent side: processes and resources ----------------------------------

# Parent pid and start time of each process in `pids` ("<ppid> <start>"),
# NA for a process that does not exist or has exited: fields 4 and 22 of
# /proc/<pid>/stat where /proc exists, else `ps -o ppid=,lstart=`. A pid that
# the system has given to another process after the worker exited has
# another parent or a later start time.
.ai4b_rc_proc_start <- function(pids) {
    pids <- as.integer(pids)
    out  <- rep(NA_character_, length(pids))
    if (!length(pids)) return(out)
    if (file.exists("/proc/self/stat")) {
        for (k in seq_along(pids)) {
            x <- tryCatch(readLines(sprintf("/proc/%d/stat", pids[k]), n = 1L,
                                    warn = FALSE),
                          error = function(e) character(0),
                          warning = function(w) character(0))
            if (!length(x)) next
            f <- strsplit(sub("^.*\\) ", "", x), " ", fixed = TRUE)[[1L]]
            if (length(f) >= 20L && f[1L] != "Z")
                out[k] <- paste(f[2L], f[20L])
        }
        return(out)
    }
    o <- tryCatch(suppressWarnings(system2("ps", c("-o", "pid=,ppid=,stat=,lstart=",
                                                   "-p", paste(pids, collapse = ",")),
                                           stdout = TRUE, stderr = FALSE)),
                  error = function(e) character(0))
    for (line in trimws(o)) {
        f <- strsplit(line, "[[:space:]]+")[[1L]]
        k <- match(suppressWarnings(as.integer(f[1L])), pids)
        if (!is.na(k) && length(f) >= 4L && !startsWith(f[3L], "Z"))
            out[k] <- paste(c(f[2L], f[-(1:3)]), collapse = " ")
    }
    out
}

# TRUE when process `pid` still has the parent and start time `start`,
# FALSE when it has exited or the pid now belongs to another process, NA
# when they were not known.
.ai4b_rc_same_process <- function(pid, start) {
    if (is.na(start)) return(NA)
    identical(.ai4b_rc_proc_start(pid), start)
}

# Return when none of the processes `pids` exists any more (parallel reaps
# the worker processes it forked), or after `timeout` seconds. An interrupt
# during this time ends the polling only.
.ai4b_rc_wait_exit <- function(pids, timeout = 2) {
    t0 <- Sys.time()
    while (length(pids) && any(tools::pskill(pids, 0L)) &&
           difftime(Sys.time(), t0, units = "secs") < timeout)
        if (identical(tryCatch(Sys.sleep(0.01), interrupt = function(e) "int"),
                      "int")) break
    invisible(NULL)
}

# Number of R connections that can still be opened, counted up to `need`
# by opening that many and closing them again.
.ai4b_rc_free_slots <- function(need) {
    held <- list()
    on.exit(for (cn in held) try(close(cn), silent = TRUE), add = TRUE)
    for (i in seq_len(need)) {
        cn <- tryCatch(textConnection(NULL, "w"), error = function(e) NULL)
        if (is.null(cn)) break
        held[[i]] <- cn
    }
    length(held)
}

# Upper bound on the worker processes that the kept runs of this session
# hold: getOption("ai4bayescode.max_workers"), else the environment variable
# AI4BAYESCODE_MAX_WORKERS, else max(8, number of cores).
.ai4b_rc_max_workers <- function() {
    v <- getOption("ai4bayescode.max_workers")
    src <- "getOption(\"ai4bayescode.max_workers\")"
    if (is.null(v) && nzchar(Sys.getenv("AI4BAYESCODE_MAX_WORKERS"))) {
        v <- Sys.getenv("AI4BAYESCODE_MAX_WORKERS")
        src <- "the environment variable AI4BAYESCODE_MAX_WORKERS"
    }
    if (is.null(v)) {
        nc <- suppressWarnings(parallel::detectCores())
        if (length(nc) != 1L || is.na(nc) || nc < 1) nc <- 1L
        return(as.integer(max(8, nc)))
    }
    n <- suppressWarnings(as.numeric(v))
    if (length(n) != 1L || is.na(n) || n < 1 || n != floor(n))
        stop("ai4bayescode_run_chains(): ", src, " must be a positive whole ",
             "number; it is ", paste(deparse(v), collapse = ""), ".",
             call. = FALSE)
    as.integer(min(n, .Machine$integer.max))
}

# The next value of a counter that orders the uses of the pools.
.ai4b_rc_tick <- function() {
    n <- if (is.null(.ai4b_rc_env$clock)) 1 else .ai4b_rc_env$clock + 1
    .ai4b_rc_env$clock <- n
    n
}

# Why the worker processes of a pool were closed: list(message = the error of
# a call through a handle, short = the text of print(run)).
.ai4b_rc_reason <- function(kind, detail = "") {
    pre <- "the worker processes holding the models of this run were closed"
    switch(kind,
        close_models = list(message = paste(pre, "(ai4bayescode_close_models)"),
                            short   = "closed (ai4bayescode_close_models)"),
        gc = list(
            message = paste0(pre, " when the run and its handles were ",
                             "garbage-collected; a copy of the run restored ",
                             "with readRDS() or unserialize() in this session ",
                             "reaches the worker processes only while the ",
                             "original run or one of its handles is referenced"),
            short   = "closed (the run that started them was garbage-collected)"),
        evicted = list(
            message = paste0(pre, " to start the worker processes of a later ",
                             "run: ", detail),
            short   = paste0("closed to start the worker processes of a later ",
                             "run (", detail, ")")),
        list(message = paste(pre, "after the run that started them failed"),
             short   = "closed (the run failed)"))
}

.ai4b_rc_closed_record <- function(token, reason) {
    cl <- .ai4b_rc_env$closed
    if (is.null(cl)) cl <- list()
    cl[[token]] <- reason
    .ai4b_rc_env$closed <- cl
    invisible(NULL)
}

.ai4b_rc_closed_reason <- function(pool) {
    r <- .ai4b_rc_env$closed[[pool$token]]
    if (is.null(r)) r <- list(
        message = "the worker processes holding the models of this run were closed",
        short   = "closed")
    r
}

# ---- parent side: pools ----------------------------------------------------

# A listening socket on a free port: R_PARALLEL_PORT when it is set, else
# ports in 11000-30999 derived from the pid and the clock (no draw from R's
# random-number generator).
.ai4b_rc_listen <- function() {
    env  <- suppressWarnings(as.integer(Sys.getenv("R_PARALLEL_PORT")))
    base <- (as.integer(Sys.getpid()) %% 20000L) * 7L +
        as.integer((as.numeric(Sys.time()) * 1000) %% 20000)
    ports <- 11000L + (base + 7919L * (0:24)) %% 20000L
    if (!is.na(env)) ports <- c(env, ports)
    for (p in ports) {
        s <- tryCatch(serverSocket(p), error = function(e) NULL)
        if (!is.null(s)) return(list(socket = s, port = p))
    }
    stop("no free port for the connections of the worker processes (",
         length(ports), " ports tried)")
}

# Fork the worker processes of a pool, accept their connections and register
# the pool.
.ai4b_rc_pool_open <- function(model_ctor, n_workers, n_chains) {
    lis <- .ai4b_rc_listen()
    on.exit(try(close(lis$socket), silent = TRUE), add = TRUE)
    key <- paste(c(as.character(tryCatch(readBin("/dev/urandom", "raw", 16L),
                                         error = function(e) raw(0),
                                         warning = function(w) raw(0))),
                   Sys.getpid(), format(Sys.time(), "%OS6")), collapse = "")
    pids   <- integer(0)
    start  <- character(0)
    cons   <- vector("list", n_workers)
    opened <- FALSE
    on.exit(if (!opened) {
        for (cn in cons) if (!is.null(cn)) try(close(cn), silent = TRUE)
        for (k in seq_along(pids))
            if (!isFALSE(.ai4b_rc_same_process(pids[k], start[k])))
                tools::pskill(pids[k], tools::SIGTERM)
    }, add = TRUE)
    assign("ctor", model_ctor, envir = .ai4b_rc_env)
    on.exit(rm(list = "ctor", envir = .ai4b_rc_env), add = TRUE)
    # One worker at a time: fork it, accept its connection and read its
    # first message (key, number, pid), then fork the next one, which closes
    # its copies of the connections accepted so far. parallel exports
    # mcparallel() on POSIX only, and this code runs on POSIX only.
    mcparallel <- getExportedValue("parallel", "mcparallel")
    for (w in seq_len(n_workers)) {
        job <- mcparallel(
            .ai4b_rc_worker_main(lis$port, key, w, lis$socket, cons),
            mc.set.seed = FALSE, mc.interactive = NA, detached = TRUE)
        pids[w] <- as.integer(job$pid)
        for (attempt in 1:5) {
            con <- tryCatch(suppressWarnings(socketAccept(
                       lis$socket, blocking = TRUE, open = "a+b", timeout = 60)),
                   error = function(e) NULL)
            if (is.null(con))
                stop("worker process ", pids[w], " did not connect within 60 s")
            hello <- if (isTRUE(tryCatch(socketSelect(list(con), timeout = 60),
                                         error = function(e) FALSE)))
                tryCatch(.ai4b_rc_read_frame(con), error = function(e) NULL)
            if (is.list(hello) && identical(hello$key, key) &&
                identical(as.integer(hello$worker), w) &&
                identical(as.integer(hello$pid), pids[w])) {
                cons[[w]] <- con
                break
            }
            try(close(con), silent = TRUE)
        }
        if (is.null(cons[[w]]))
            stop("the first message of worker process ", pids[w],
                 " was not received or not valid")
    }
    try(close(lis$socket), silent = TRUE)
    start <- .ai4b_rc_proc_start(pids)
    if (exists("socketTimeout", envir = baseenv()))
        for (cn in cons) socketTimeout(cn, 2592000)
    node_of <- ((seq_len(n_chains) - 1L) %% n_workers) + 1L
    n <- if (is.null(.ai4b_rc_env$n_pools)) 1L else .ai4b_rc_env$n_pools + 1L
    .ai4b_rc_env$n_pools <- n
    token <- paste0("pool", n)
    entry <- new.env(parent = emptyenv())
    entry$cons      <- cons
    entry$pids      <- pids
    entry$start     <- start
    entry$node_of   <- node_of
    entry$seq       <- 0L                           # number of the last request
    entry$sent      <- integer(n_workers)           # last request sent to each worker
    entry$recv      <- integer(n_workers)           # last request whose reply was read
    entry$partial   <- logical(n_workers)           # a frame read was cut short
    entry$files     <- vector("list", n_workers)    # files of the last request
    entry$lost      <- rep(NA_character_, n_workers)  # message of a lost worker
    entry$last_used <- .ai4b_rc_tick()
    assign(token, entry, envir = .ai4b_rc_registry())
    pool <- new.env(parent = emptyenv())
    pool$token     <- token
    pool$session   <- .ai4b_rc_session()
    pool$owner_pid <- Sys.getpid()
    pool$owner_tmp <- tempdir()
    pool$pids      <- pids
    pool$n_workers <- n_workers
    pool$n_chains  <- n_chains
    pool$node_of   <- node_of
    reg.finalizer(pool, .ai4b_rc_pool_finalize, onexit = TRUE)
    opened <- TRUE
    pool
}

# "open", "closed", "child" (called in a forked child process of the R
# process that created the pool) or "other" (created in another R session).
.ai4b_rc_pool_state <- function(pool) {
    if (!identical(pool$session, .ai4b_rc_session())) {
        if (!is.null(pool$owner_pid) && !identical(pool$owner_pid, Sys.getpid()) &&
            identical(pool$owner_tmp, tempdir()))
            return("child")
        return("other")
    }
    if (exists(pool$token, envir = .ai4b_rc_registry(), inherits = FALSE))
        "open" else "closed"
}

# The registry entry of an open pool of this session, or an error.
.ai4b_rc_pool_entry <- function(pool) {
    switch(.ai4b_rc_pool_state(pool),
        other = stop("this ai4bayescode_chain_model handle was created in ",
                     "another R session or process. The models of a parallel ",
                     "run exist only in the worker processes of the R session ",
                     "that ran ai4bayescode_run_chains(), and saveRDS() / ",
                     "readRDS() does not transfer them; rerun ",
                     "ai4bayescode_run_chains() in this session.", call. = FALSE),
        child = stop("an ai4bayescode_chain_model handle can be called only in ",
                     "the R process that ran ai4bayescode_run_chains() (pid ",
                     pool$owner_pid, "), not in a forked child process of it ",
                     "(parallel::mclapply(), or model_ctor of another parallel ",
                     "ai4bayescode_run_chains() run). Read the values in that ",
                     "process first, e.g. inits <- lapply(pilot$models, ",
                     "function(h) h$get_current()), and use inits[[i]] in ",
                     "model_ctor.", call. = FALSE),
        closed = stop(.ai4b_rc_closed_reason(pool)$message, call. = FALSE),
        get(pool$token, envir = .ai4b_rc_registry(), inherits = FALSE))
}

# Mark worker w of `entry` as lost: a call on one of its chains raises
# `entry$lost[w]` from now on, and the other workers are not affected. A
# worker process that still runs (the same pid and start time) is sent
# SIGTERM; after it has exited, the files of its last request are deleted.
.ai4b_rc_lose <- function(entry, w, cause) {
    if (!is.na(entry$lost[w])) return(invisible(NULL))
    suspendInterrupts({
        same <- .ai4b_rc_same_process(entry$pids[w], entry$start[w])
        if (!isFALSE(same)) tools::pskill(entry$pids[w], tools::SIGTERM)
        chains <- which(entry$node_of == w)
        entry$lost[w] <- sprintf(
            "the worker process %d holding chain%s %s %s (%s); the model%s of %s lost",
            entry$pids[w], if (length(chains) == 1L) "" else "s",
            paste(chains, collapse = ", "),
            if (isTRUE(same)) "was terminated" else "exited unexpectedly", cause,
            if (length(chains) == 1L) "" else "s",
            if (length(chains) == 1L) "this chain is" else "these chains are")
        con <- entry$cons[[w]]
        entry$cons[w] <- list(NULL)
        if (!is.null(con)) try(close(con), silent = TRUE)
        if (!isFALSE(same)) .ai4b_rc_wait_exit(entry$pids[w])
        unlink(entry$files[[w]])
        entry$files[w] <- list(NULL)
    })
    invisible(NULL)
}

# Close the pool registered under `token`, recording `reason`: an idle worker
# gets a "close" request and exits; a worker whose reply to the last request
# was not read (an interrupted call) is sent SIGTERM when its pid and start
# time are still those of the worker, and the files of that request are
# deleted after it has exited. An interrupt does not stop the closing.
.ai4b_rc_close_token <- function(token, reason) {
    reg   <- .ai4b_rc_registry()
    entry <- get0(token, envir = reg, inherits = FALSE)
    if (is.null(entry)) return(invisible(FALSE))
    rm(list = token, envir = reg)
    .ai4b_rc_closed_record(token, reason)
    suspendInterrupts({
        termed <- integer(0)
        for (w in seq_along(entry$cons)) {
            con <- entry$cons[[w]]
            if (is.null(con)) next
            idle  <- entry$sent[w] == entry$recv[w] && !entry$partial[w]
            asked <- idle && isTRUE(tryCatch({
                writeBin(.ai4b_rc_frame(list(seq = 0L, op = "close"))$bytes, con)
                TRUE
            }, error = function(e) FALSE, interrupt = function(e) FALSE))
            if (!asked && !isFALSE(.ai4b_rc_same_process(entry$pids[w],
                                                         entry$start[w]))) {
                tools::pskill(entry$pids[w], tools::SIGTERM)
                termed <- c(termed, entry$pids[w])
            }
            entry$cons[w] <- list(NULL)
            try(close(con), silent = TRUE)
        }
        .ai4b_rc_wait_exit(termed)
        unlink(unlist(entry$files))
        entry$files <- vector("list", length(entry$files))
    })
    invisible(TRUE)
}

# Terminate the worker processes of a pool; a no-op for a pool that is
# closed, of another session, or seen from a forked child process.
.ai4b_rc_pool_close <- function(pool, reason) {
    if (!identical(.ai4b_rc_pool_state(pool), "open")) return(invisible(FALSE))
    .ai4b_rc_close_token(pool$token, reason)
}

.ai4b_rc_pool_finalize <- function(pool)
    .ai4b_rc_pool_close(pool, .ai4b_rc_reason("gc"))

# Before the worker processes of a new run are started: close the pools of
# earlier runs, least recently used first, until the n_workers new worker
# processes fit under max_workers and n_workers + 2 connections are free,
# with one warning. Returns NULL, or the text of a warning when even closing
# every earlier pool is not enough (nothing is closed then).
.ai4b_rc_make_room <- function(n_workers) {
    max_w <- .ai4b_rc_max_workers()
    need  <- n_workers + 2L
    invisible(gc())             # finalizers of pools that are not referenced
    reg  <- .ai4b_rc_registry()
    live <- function(e) sum(is.na(e$lost))
    held <- sum(vapply(mget(ls(reg), envir = reg), live, numeric(1)))
    if (n_workers > max_w)
        return(sprintf(paste0(
            "the %d worker processes of this run ",
            "exceed max_workers = %d (getOption(\"ai4bayescode.max_workers\"), ",
            "the environment variable AI4BAYESCODE_MAX_WORKERS, or ",
            "max(8, number of cores)); the chains run sequentially in this ",
            "session. Raise that bound, or pass a smaller n_cores."),
            n_workers, max_w))
    if (.ai4b_rc_free_slots(need) + held < need)
        return(sprintf(paste0(
            "R has fewer than %d free connections, ",
            "which the %d worker processes of this run require; the chains run ",
            "sequentially in this session. Close connections that are no ",
            "longer needed (see showConnections()), or start R with ",
            "--max-connections=N."), need, n_workers))
    n_runs <- 0L
    n_proc <- 0L
    by_w   <- FALSE
    by_c   <- FALSE
    repeat {
        # A pool whose worker processes were all lost holds no process and
        # no connection, so closing it would not help.
        entries <- Filter(function(e) live(e) > 0,
                          mget(ls(reg), envir = reg))
        over_w  <- sum(vapply(entries, live, numeric(1))) + n_workers > max_w
        over_c  <- .ai4b_rc_free_slots(need) < need
        if ((!over_w && !over_c) || !length(entries)) break
        lru <- names(entries)[which.min(vapply(entries, function(e) e$last_used,
                                               numeric(1)))]
        n_runs <- n_runs + 1L
        n_proc <- n_proc + live(entries[[lru]])
        by_w   <- by_w || over_w
        by_c   <- by_c || over_c
        why <- paste(c(if (over_w) sprintf(paste0(
                           "at most max_workers = %d worker processes of kept ",
                           "runs exist at a time"), max_w),
                       if (over_c) "R had too few free connections"),
                     collapse = ", and ")
        .ai4b_rc_close_token(lru, .ai4b_rc_reason("evicted", why))
    }
    if (n_runs)
        warning(sprintf(paste0(
            "ai4bayescode_run_chains: the worker processes of %d earlier ",
            "run%s (%d process%s) were closed, least recently used first, to ",
            "start the %d worker processes of this run: %s. A call on a model ",
            "handle of %s raises an error. To keep the models of more runs, ",
            "close runs that are no longer needed with ",
            "ai4bayescode_close_models(run)%s; a run made with keep_models = ",
            "FALSE holds no worker processes."),
            n_runs, if (n_runs == 1L) "" else "s", n_proc,
            if (n_proc == 1L) "" else "es", n_workers,
            paste(c(if (by_w) sprintf(paste0(
                        "at most max_workers = %d worker processes of kept runs ",
                        "exist at a time"), max_w),
                    if (by_c) sprintf("R had fewer than %d free connections",
                                      need)),
                  collapse = ", and "),
            if (n_runs == 1L) "that run" else "those runs",
            paste0(if (by_w) paste0(", or raise options(ai4bayescode.max_workers",
                                    " = ...) or the environment variable ",
                                    "AI4BAYESCODE_MAX_WORKERS"),
                   if (by_c) ", or start R with --max-connections=N")),
            call. = FALSE, immediate. = TRUE)
    NULL
}

# ---- parent side: requests -------------------------------------------------

# Write request `msg` to worker w of `entry` and record its number and its
# files (the file of a long request, and the file named for a long reply);
# FALSE when the worker was found lost.
.ai4b_rc_send <- function(entry, w, msg) {
    msg$reply_file <- tempfile("ai4b_rc_", fileext = ".bin")
    fr   <- .ai4b_rc_frame(msg)
    sent <- FALSE
    on.exit(if (!sent && !is.null(fr$file)) unlink(fr$file), add = TRUE)
    r <- suspendInterrupts(tryCatch({
        writeBin(fr$bytes, entry$cons[[w]])
        entry$sent[w] <- as.integer(msg$seq)
        entry$files[w] <- list(c(fr$file, msg$reply_file))
        sent <- TRUE
        NULL
    }, error = function(e) e))
    if (inherits(r, "error")) {
        .ai4b_rc_lose(entry, w, paste0("writing to its connection failed: ",
                                       conditionMessage(r)))
        return(FALSE)
    }
    TRUE
}

# Read one frame from worker w of `entry` and record its request number;
# NULL when the worker was found lost. entry$partial[w] is set to TRUE after
# the first recv() of a frame and to FALSE after its last byte is read; an
# interrupt between the two, which is possible only for a frame whose bytes
# were received in more than one segment, leaves it TRUE.
.ai4b_rc_recv <- function(entry, w) {
    con <- entry$cons[[w]]
    m <- suspendInterrupts(tryCatch({
        hdr <- readBin(con, "raw", 8L)
        entry$partial[w] <- TRUE
        if (length(hdr) < 8L) stop("its connection reached its end")
        if (!identical(hdr[1:4], .ai4b_rc_frame_magic))
            stop("a malformed message was read from its connection")
        n <- readBin(hdr[5:8], "integer", size = 4L, endian = "little")
        body <- readBin(con, "raw", n)
        if (length(body) < n) stop("its connection reached its end")
        msg <- unserialize(body)
        entry$partial[w] <- FALSE
        entry$recv[w] <- as.integer(msg$seq)
        msg
    }, error = function(e) e))
    if (inherits(m, "error")) {
        .ai4b_rc_lose(entry, w, conditionMessage(m))
        return(NULL)
    }
    m
}

# TRUE when the connection of worker w has a frame (or its end) to read
# within `timeout` seconds; FALSE otherwise or when the worker was found
# lost.
.ai4b_rc_ready <- function(entry, w, timeout) {
    r <- tryCatch(socketSelect(entry$cons[w], timeout = timeout),
                  error = function(e) e)
    if (inherits(r, "error")) {
        .ai4b_rc_lose(entry, w, conditionMessage(r))
        return(FALSE)
    }
    isTRUE(r)
}

# Read and discard the replies of worker w to requests whose callers were
# interrupted before they read them.
.ai4b_rc_drain <- function(entry, w) {
    while (is.na(entry$lost[w]) && entry$recv[w] < entry$sent[w]) {
        if (!.ai4b_rc_ready(entry, w, 1)) next
        m <- .ai4b_rc_recv(entry, w)
        if (is.null(m)) next
        .ai4b_rc_discard(m)
        if (entry$recv[w] == entry$sent[w]) entry$files[w] <- list(NULL)
    }
    invisible(NULL)
}

# Send jobs[[k]] to worker nodes[k] of `pool`; the workers compute their jobs
# concurrently. Returns, for each node, its reply list (with `out`, or
# `error` when the worker could not complete the request) or
# list(lost = <message>) for a lost worker. The replies to earlier requests
# whose callers were interrupted are read and discarded first.
.ai4b_rc_exchange <- function(pool, nodes, jobs) {
    entry <- .ai4b_rc_pool_entry(pool)
    entry$last_used <- .ai4b_rc_tick()
    for (w in nodes)
        if (is.na(entry$lost[w]) && entry$partial[w])
            .ai4b_rc_lose(entry, w, paste0(
                "an earlier call was interrupted while it read a message of ",
                "this worker process, and the rest of that message was left ",
                "in the connection"))
    for (w in nodes) .ai4b_rc_drain(entry, w)
    s <- entry$seq + 1L
    entry$seq <- s
    for (k in seq_along(nodes))
        if (is.na(entry$lost[nodes[k]]))
            .ai4b_rc_send(entry, nodes[k], c(jobs[[k]], list(seq = s)))
    out <- vector("list", length(nodes))
    repeat {
        pending <- nodes[is.na(entry$lost[nodes]) & entry$sent[nodes] == s &
                         entry$recv[nodes] < s]
        if (!length(pending)) break
        ready <- tryCatch(socketSelect(entry$cons[pending], timeout = 1),
                          error = function(e) NULL)
        if (is.null(ready))
            ready <- vapply(pending, function(w) .ai4b_rc_ready(entry, w, 0),
                            logical(1))
        for (w in pending[ready]) {
            m <- .ai4b_rc_recv(entry, w)
            if (is.null(m)) next
            if (!identical(as.integer(m$seq), s)) {
                .ai4b_rc_discard(m)
                .ai4b_rc_lose(entry, w, "it replied to another request")
                next
            }
            out[[match(w, nodes)]] <- tryCatch(.ai4b_rc_unpack(m),
                error = function(e) list(error = paste0(
                    "its reply could not be read (", conditionMessage(e), ")")))
            entry$files[w] <- list(NULL)
        }
    }
    for (k in seq_along(nodes))
        if (!is.na(entry$lost[nodes[k]]))
            out[k] <- list(list(lost = entry$lost[nodes[k]]))
    out
}

# Run all chains in a new pool. Returns list(kind = "done", results,
# handles), or, with the pool closed: list(kind = "chain", chain, message,
# class) for an R error in a chain, list(kind = "lost", message) when a
# worker process exited during the run, list(kind = "start", message) when
# the worker processes could not be started, or list(kind = "room", message)
# when max_workers or the free connections do not allow them.
.ai4b_rc_pool_run <- function(model_ctor, seeds, n_burn, n_keep, n_workers) {
    n_chains <- length(seeds)
    room <- .ai4b_rc_make_room(n_workers)
    if (!is.null(room)) return(list(kind = "room", message = room))
    pool <- tryCatch(.ai4b_rc_pool_open(model_ctor, n_workers, n_chains),
                     error = function(e) e)
    if (inherits(pool, "error"))
        return(list(kind = "start", message = conditionMessage(pool)))
    finished <- FALSE
    on.exit(if (!finished) .ai4b_rc_pool_close(pool, .ai4b_rc_reason("failed")),
            add = TRUE)
    jobs <- lapply(seq_len(n_workers), function(w) {
        idx <- which(pool$node_of == w)
        list(op = "run", chains = idx, seeds = seeds[idx], n_burn = n_burn,
             n_keep = n_keep)
    })
    out  <- .ai4b_rc_exchange(pool, seq_len(n_workers), jobs)
    caps <- vector("list", n_chains)
    for (w in seq_len(n_workers)) {
        o <- out[[w]]
        if (!is.null(o[["lost"]]))
            return(list(kind = "lost", message = sub(
                "; the models? of (this chain is|these chains are) lost$", "",
                o[["lost"]])))
        if (!is.null(o[["error"]]))
            return(list(kind = "start", message = sprintf(
                "worker process %d: %s", pool$pids[w], o[["error"]])))
        caps[jobs[[w]]$chains[seq_along(o$out)]] <- o$out
    }
    bad <- which(vapply(caps, function(cp) !is.null(cp$error), logical(1)))
    if (length(bad))
        return(list(kind = "chain", chain = bad[1L],
                    message = caps[[bad[1L]]]$error$message,
                    class = caps[[bad[1L]]]$error$class))
    results <- lapply(seq_len(n_chains), function(i) .ai4b_rc_relay(caps[[i]], i))
    handles <- lapply(seq_len(n_chains), function(i)
        .ai4b_rc_handle(pool, i, results[[i]]$class, results[[i]]$methods))
    finished <- TRUE
    list(kind = "done", results = results, handles = handles)
}

.ai4b_rc_handle <- function(pool, chain, class_name, methods) {
    h <- new.env(parent = emptyenv())
    h$pool       <- pool
    h$chain      <- as.integer(chain)
    h$class_name <- class_name
    h$methods    <- methods
    lockEnvironment(h, bindings = TRUE)
    class(h) <- "ai4bayescode_chain_model"
    h
}

# Call `method` with `args` on the models of `chains` of a pool; returns the
# captured results in the order of `chains`. The chains of a lost worker get
# its message as their error.
.ai4b_rc_remote <- function(pool, chains, method, args) {
    nodes <- unique(pool$node_of[chains])
    # Positions in `chains` of the chains of each node (a chain can occur
    # more than once).
    at    <- lapply(nodes, function(w) which(pool$node_of[chains] == w))
    jobs  <- lapply(at, function(idx)
        list(op = "call", chains = chains[idx], method = method, args = args))
    out  <- .ai4b_rc_exchange(pool, nodes, jobs)
    caps <- vector("list", length(chains))
    for (k in seq_along(nodes)) {
        o   <- out[[k]]
        idx <- at[[k]]
        msg <- if (!is.null(o[["lost"]])) o[["lost"]]
               else if (!is.null(o[["error"]]))
                   sprintf("worker process %d: %s",
                           pool$pids[nodes[k]], o[["error"]])
        caps[idx] <- if (is.null(msg)) o$out
                     else rep(list(.ai4b_rc_failed(msg)), length(idx))
    }
    caps
}

.ai4b_rc_handle_call <- function(h, method, args) {
    chain <- .subset2(h, "chain")
    .ai4b_rc_relay(.ai4b_rc_remote(.subset2(h, "pool"), chain, method,
                                   args)[[1L]], chain)
}

# The pools of the handles in run$models.
.ai4b_rc_run_pools <- function(run) {
    models <- if (is.list(run)) run$models
    hs <- Filter(function(m) inherits(m, "ai4bayescode_chain_model"), models)
    unique(lapply(hs, .subset2, "pool"))
}

.ai4b_rc_models_where <- function(models) {
    if (is.null(models)) return("not kept (keep_models = FALSE)")
    pools <- .ai4b_rc_run_pools(list(models = models))
    if (!length(pools)) return("in this R session (run$models[[i]])")
    parts <- vapply(pools, function(p) switch(.ai4b_rc_pool_state(p),
        open = {
            e    <- get(p$token, envir = .ai4b_rc_registry(), inherits = FALSE)
            ok   <- is.na(e$lost)
            txt  <- if (any(ok)) sprintf(paste0(
                "in %d worker process%s of this R session (pid %s); ",
                "run$models[[i]]$<method>(...) is evaluated there, and ",
                "ai4bayescode_close_models(run) terminates them"),
                sum(ok), if (sum(ok) == 1L) "" else "es",
                paste(e$pids[ok], collapse = ", "))
            paste(c(txt, e$lost[!ok]), collapse = "; ")
        },
        closed = .ai4b_rc_closed_reason(p)$short,
        child  = paste0("in worker processes of the parent R process (a handle ",
                        "cannot be called in a forked child process)"),
        other  = "in worker processes of another R session (not available here)"),
        character(1))
    paste(unique(parts), collapse = "; ")
}

# ---- predictions -----------------------------------------------------------

# newdata of ai4bayescode_predict(): NULL means list(); anything other than a
# list with a name for every element raises an error.
.ai4b_rc_check_newdata <- function(newdata) {
    if (is.null(newdata)) return(list())
    nm <- names(newdata)
    what <- if (is.data.frame(newdata))
        paste0("a data.frame, whose columns would reach predict_at() as ",
               "separate inputs; e.g. pass list(X = as.matrix(newdata))")
    else if (!is.list(newdata)) .ai4b_rc_describe(newdata)
    else if (length(newdata) &&
             (is.null(nm) || anyNA(nm) || !all(nzchar(nm))))
        "a list with an unnamed element"
    if (!is.null(what))
        stop("ai4bayescode_predict(): newdata must be a named list of the ",
             "inputs of the model's predict_at(), e.g. predict(run, ",
             "list(X = X_new)), or list() for the training data; got ", what,
             ".", call. = FALSE)
    newdata
}

# The value of predict_at() of chain i must be a list with unique non-empty
# names.
.ai4b_rc_check_prediction <- function(p, i) {
    nm <- names(p)
    if (!is.list(p) || (length(p) && (is.null(nm) || anyNA(nm) ||
                                      !all(nzchar(nm)) || anyDuplicated(nm))))
        stop(sprintf(paste0("ai4bayescode_predict(): predict_at() of chain %d ",
                            "returned %s; it must return a list with unique ",
                            "non-empty names."), i,
                     if (is.list(p) && !is.data.frame(p))
                         "a list without unique non-empty names"
                     else .ai4b_rc_describe(p)), call. = FALSE)
    invisible(NULL)
}

# Stack the predictions of the chains per output key along the first axis.
# The pooled entry has the number of dimensions of the per-chain entries: a
# vector (also a 1-D array) gives a vector of length n_chains * n, a matrix a
# matrix, a k-D array a k-D array.
.ai4b_rc_pool_predictions <- function(preds) {
    keys <- names(preds[[1L]])
    for (i in seq_along(preds)[-1L]) {
        if (!setequal(names(preds[[i]]), keys))
            stop(sprintf(paste0("ai4bayescode_predict(): chain 1 gives the keys ",
                                "(%s) and chain %d the keys (%s); use pool = FALSE."),
                         paste(keys, collapse = ", "), i,
                         paste(names(preds[[i]]), collapse = ", ")), call. = FALSE)
    }
    out <- lapply(keys, function(k) {
        parts <- lapply(preds, `[[`, k)
        for (i in seq_along(parts))
            if (is.null(parts[[i]]) || !is.atomic(parts[[i]]))
                stop(sprintf(paste0("ai4bayescode_predict(): '%s' of chain %d is ",
                                    "%s, not a vector, matrix or array; use ",
                                    "pool = FALSE."),
                             k, i, .ai4b_rc_describe(parts[[i]])), call. = FALSE)
        nd  <- vapply(parts, function(x) max(1L, length(dim(x))), integer(1))
        bad <- which(nd != nd[1L])
        if (length(bad))
            stop(sprintf(paste0("ai4bayescode_predict(): '%s' has %d dimension%s ",
                                "in chain 1 and %d in chain %d; use pool = FALSE."),
                         k, nd[1L], if (nd[1L] == 1L) "" else "s", nd[bad[1L]],
                         bad[1L]), call. = FALSE)
        if (nd[1L] == 1L)
            return(unlist(lapply(parts, as.vector), use.names = FALSE))
        cols <- lapply(parts, function(x) dim(x)[-1L])
        same <- vapply(cols, identical, logical(1), cols[[1L]])
        if (!all(same))
            stop(sprintf(paste0("ai4bayescode_predict(): '%s' has %s columns in ",
                                "chain 1 and %s in chain %d; use pool = FALSE."),
                         k, paste(cols[[1L]], collapse = " x "),
                         paste(cols[[which(!same)[1L]]], collapse = " x "),
                         which(!same)[1L]), call. = FALSE)
        .ai4b_rc_bind_rows(parts)
    })
    names(out) <- keys
    out
}

# Stack per-chain entries of 2 or more dimensions along the first axis.
.ai4b_rc_bind_rows <- function(parts) {
    if (length(dim(parts[[1L]])) <= 2L) return(do.call(rbind, parts))
    d    <- dim(parts[[1L]])
    perm <- c(seq_along(d)[-1L], 1L)
    n    <- sum(vapply(parts, function(x) dim(x)[1L], integer(1)))
    arr  <- array(unlist(lapply(parts, aperm, perm)), dim = c(d[-1L], n))
    aperm(arr, c(length(d), seq_len(length(d) - 1L)))
}


# ai4bayescode_rhat_summary(): per-key rhat + ess_bulk across chains; needs the
# 'posterior' package.
ai4bayescode_rhat_summary <- function(run, keys = NULL, drop_burn = 0,
                                      order_components = FALSE) {
    if (!requireNamespace("posterior", quietly = TRUE)) {
        stop("posterior package required for R-hat summary")
    }
    histories <- run$histories
    if (length(histories) < 1L) {
        stop("run$histories is empty -- nothing to summarise")
    }
    # Validate drop_burn the way ai4bayescode_diagnose() validates n_burn. It
    # used to be applied only when `0 < drop_burn < n`, so drop_burn = n, a
    # negative, or a string silently returned the drop_burn = 0 answer -- and
    # drop_burn = n - 1 left a single draw, whose R-hat is NA, whose max() over
    # an all-NA set is -Inf, and -Inf < 1.01 PASSES a convergence gate.
    if (!is.numeric(drop_burn) || length(drop_burn) != 1L || is.na(drop_burn) ||
        drop_burn < 0 || drop_burn != as.integer(drop_burn)) {
        stop("ai4bayescode_rhat_summary(): `drop_burn` must be a single ",
             "non-negative integer; got ", paste(deparse(drop_burn), collapse = ""),
             call. = FALSE)
    }
    .n_draws <- function(h) if (is.null(dim(h))) length(h) else nrow(h)
    .shortest <- min(vapply(histories,
                            function(h) min(vapply(h, .n_draws, numeric(1))),
                            numeric(1)))
    if (drop_burn >= .shortest - 1) {
        stop("ai4bayescode_rhat_summary(): `drop_burn` = ", drop_burn,
             " leaves ", max(0, .shortest - drop_burn),
             " draw(s) of the shortest history (", .shortest,
             "); at least 2 are needed for a split R-hat. Reduce drop_burn.",
             call. = FALSE)
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

# ---------------------------------------------------------------------------
# S3 methods of this file. The package registers them in its NAMESPACE; here
# they are registered when the file is sourced, so that print(run),
# predict(run) and run$models[[i]]$f() dispatch to them also when the file is
# sourced into an environment other than the global one (sys.source(),
# source(local = TRUE), a knitr chunk). The generics are looked up from the
# search path; a generic that is not found there is skipped.
# ---------------------------------------------------------------------------
.ai4b_register_s3 <- function() {
    m <- list(
        list("print",        "ai4bayescode_run",         print.ai4bayescode_run),
        list("predict",      "ai4bayescode_run",         predict.ai4bayescode_run),
        list("print",        "ai4bayescode_chain_model", print.ai4bayescode_chain_model),
        list("$",            "ai4bayescode_chain_model", `$.ai4bayescode_chain_model`),
        list("$<-",          "ai4bayescode_chain_model", `$<-.ai4bayescode_chain_model`),
        list(".DollarNames", "ai4bayescode_chain_model", .DollarNames.ai4bayescode_chain_model))
    for (x in m)
        try(registerS3method(x[[1L]], x[[2L]], x[[3L]], envir = globalenv()),
            silent = TRUE)
    invisible(NULL)
}
.ai4b_register_s3()
