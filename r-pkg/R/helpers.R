# ----------------------------------------------------------------------------
# helpers.R
#
# Runtime helpers a generated R runner (or an end user) calls AFTER a sampler
# wrapper has been compiled. These are shipped so that generated runners can
# call the SAME functions by name instead of re-emitting them:
#
#   * ai4bayescode_diagnose()          -- model-independent posterior diagnostics + plot
#                                 (codegen_r_runner.md mandates the runner call
#                                  the SHIPPED ai4bayescode_diagnose()).
#   * ai4bayescode_perf_hint() -- post-run per-sweep performance hint.
#   * ai4bayescode_plot_dag()               -- render the model prediction DAG.
#   * ai4bayescode_run_chains() -- multi-chain runner for any wrapper class.
#   * ai4bayescode_rhat_summary() -- R-hat / ESS summary across chains.
#
# Ported verbatim (behavior-preserving) from the standalone scripts
# AI4BayesCode/R/AI4BayesCode_helpers.R and AI4BayesCode/R/ai4bayescode_run_chains.R
# so that `library(AI4BayesCode)` provides them too.
# ----------------------------------------------------------------------------


#' Locate (and create) the AI4BayesCode generated-code directory
#'
#' @description Returns the path to a sibling `generated/` directory next to
#'   the AI4BayesCode checkout, creating it if it does not exist.
#' @param AI4BayesCode_path Path to the AI4BayesCode folder on disk.
#' @return Absolute path to the `generated/` directory (created if missing).
#' @noRd
AI4BayesCode_generated_dir <- function(AI4BayesCode_path = "./AI4BayesCode") {
    gen_dir <- file.path(dirname(normalizePath(AI4BayesCode_path, mustWork = TRUE)),
                         "generated")
    if (!dir.exists(gen_dir)) {
        dir.create(gen_dir, recursive = TRUE)
        message("Created ", gen_dir)
    }
    gen_dir
}



#' Render the model prediction (generative) DAG
#'
#' @description Draws the prediction DAG returned by `model$get_dag()` using a
#'   Sugiyama hierarchical layout. Nodes are coloured by role (data input,
#'   sampled parameter, predicted value, prior context) and repeated
#'   `prefix_0, prefix_1, ...` names are optionally collapsed into a single
#'   plate node. By default it draws to the current graphics device; pass
#'   `out_path` to write a PNG instead.
#' @param model A compiled AI4BayesCode wrapper exposing `get_dag()`.
#' @param out_path Optional path to a PNG file. If `NULL` (default), draws to
#'   the active graphics device.
#' @param main Optional plot title.
#' @param width,height,res PNG device dimensions / resolution (only used when
#'   `out_path` is given).
#' @param plate Logical; collapse `prefix_<digits>` siblings into one plate
#'   node. Defaults to `TRUE`.
#' @param ... Forwarded to [igraph::plot.igraph()].
#' @return Invisibly `NULL` when drawing interactively, or the PNG path when
#'   `out_path` is supplied.
#' @export
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


#' Post-run performance hint for AI4BayesCode samplers
#'
#' @description Emits a friendly per-sweep performance message after a run. When
#'   the modular-NUTS per-sweep time is high it suggests the `joint_nuts_block`
#'   escape hatch (with the validator Check #11 caveat); when the runner already
#'   uses joint NUTS it instead prints tuning hints. The generated R runner
#'   calls this once at the end.
#' @param wall_sec Total wall-clock time across all chains, in seconds.
#' @param n_sweeps_total Total number of sweeps across all chains (warmup
#'   included); used to compute per-sweep time.
#' @param uses_joint_nuts Logical; `TRUE` when the runner already uses
#'   `joint_nuts_block`. Suppresses the "switch to joint" hint.
#' @param thresholds List with `slow_sweep_sec` (default 0.5); hints fire only
#'   when per-sweep time exceeds it.
#' @return Invisibly `NULL`. Side effect: prints to stderr via `message()`.
#' @export
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


# Scan matrix-valued history keys for LABEL SWITCHING: a key is flagged when its
# raw max split-R-hat is high but collapses after ordering the components within
# each draw (order statistics are invariant to relabelling).
#' @keywords internal
#' @noRd
# A key is flagged as label switching ONLY when the raw max split-R-hat is high
# (> hi) AND ordering components within each draw brings it BELOW a converged
# level (< converged). If ordering does NOT bring R-hat down to convergence, the
# high R-hat is genuine non-convergence, not a labelling artefact -- do not flag.
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

#' Model-independent posterior diagnostics and plot
#'
#' @description Computes per-parameter convergence diagnostics (split-R-hat,
#'   bulk/tail ESS, MCSE, mean/sd/median/90% CI via the \pkg{posterior} package)
#'   from a named list of posterior draws, and returns a combined
#'   trace + autocorrelation + density plot (via \pkg{bayesplot}, with a base-R
#'   fallback when bayesplot is absent). Shipped so generated
#'   `run_chain_<Model>(diagnosis = TRUE)` runners call ONE function instead of
#'   re-emitting it. PSIS-LOO is intentionally excluded (it needs a
#'   model-specific pointwise log-likelihood).
#' @param hist Named list of posterior draws: scalars as numeric vectors,
#'   vector parameters as matrices (draws in rows).
#' @param n_burn Integer; number of leading draws to drop from EVERY key before
#'   summarising. `get_history()` includes burn-in, so pass the burn-in length
#'   here; use `0` (default) when `hist` is already burn-in-stripped (no change
#'   for existing callers). Errors if `n_burn` leaves no post-burn-in draws.
#' @param plot Logical; build the diagnostic plot. Defaults to `TRUE`.
#' @return A list with `summary` (a `posterior::summarise_draws` table), `plot`
#'   (a \pkg{bayesplot} object, a base-R plotting closure, or `NULL`), and
#'   `label_switch` (named list of keys flagged as label-switching, each with the
#'   `raw` and `ordered` max split-R-hat; empty if none).
#' @param order_components Logical; if `TRUE`, sort each draw's components within
#'   every matrix-valued key (order statistics) before summarising -> a LABEL-
#'   INVARIANT per-component summary for exchangeable (mixture/cluster) params.
#'   Defaults to `FALSE`, in which case `ai4bayescode_diagnose` still scans for label
#'   switching and warns when a high R-hat is a labelling artefact.
#' @examples
#' \dontrun{
#' hist <- list(mu = rnorm(2000), sigma = abs(rnorm(2000)))
#' ai4bayescode_diagnose(hist)                 # all draws
#' ai4bayescode_diagnose(hist, n_burn = 1000)  # drop the first 1000 draws first
#' }
#' @export
ai4bayescode_diagnose <- function(hist, n_burn = 0, plot = TRUE, order_components = FALSE) {
    if (!requireNamespace("posterior", quietly = TRUE)) {
        stop("ai4bayescode_diagnose() needs the 'posterior' package. ",
             "Install it with install.packages('posterior').", call. = FALSE)
    }
    if (!is.list(hist) || is.null(names(hist)) || !all(nzchar(names(hist)))) {
        stop("`hist` must be a named list of posterior draws ",
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
    list(summary = summary, plot = plt, label_switch = label_switch)
}


#' Run multiple MCMC chains for an AI4BayesCode wrapper class
#'
#' @description Spawns `n_chains` chains (parallel via `parallel::mclapply` on
#'   POSIX, or sequential) from a deterministic constructor closure, runs warmup
#'   + keep on each, and returns the per-chain `get_history()` results, the seeds
#'   used, and the per-chain wall time. Each chain instantiates a fresh Rcpp
#'   module object in its own worker, so object ownership is handled cleanly.
#' @param model_ctor Function of one argument (the integer seed, passed positionally; the parameter name is irrelevant) returning a fresh wrapper
#'   object exposing `step()` and `get_history()`. Must be deterministic in
#'   `seed`.
#' @param n_chains Number of chains. Defaults to 4.
#' @param n_burn Warmup iterations per chain. Defaults to 2000.
#' @param n_keep Kept iterations per chain. Defaults to 10000.
#' @param seeds Optional integer vector of length `n_chains`; auto-generated if
#'   `NULL`.
#' @param parallel Logical; use `parallel::mclapply` on POSIX. Defaults to
#'   `TRUE` (silently sequential on Windows).
#' @param mc.cores Number of cores; auto-chosen if `NULL`.
#' @param verbose Logical; print a progress message. Defaults to `TRUE`.
#' @return A list with `histories` (list of per-chain `get_history()` returns),
#'   `seeds`, and `wall` (per-chain seconds).
#' @export
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

    one_chain <- function(seed_val) {
        t0 <- Sys.time()
        m <- model_ctor(seed_val)
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

    chain_ok <- function(r) !inherits(r, "try-error") && !is.null(r$history)

    if (use_parallel && mc.cores > 1) {
        if (verbose)
            message("ai4bayescode_run_chains: running ", n_chains,
                    " chains on ", mc.cores, " cores (parallel)")
        results <- parallel::mclapply(seeds, one_chain, mc.cores = mc.cores,
                                      mc.set.seed = TRUE)
        if (!all(vapply(results, chain_ok, logical(1)))) {
            # A forked worker died before returning a history. The usual cause
            # is a fork-unsafe multithreaded BLAS -- notably macOS Accelerate
            # (vecLib), whose GCD worker threads do not survive fork(), so a
            # chain doing heavier linear algebra segfaults under mclapply.
            # Recover by re-running every chain sequentially (no fork).
            warning("ai4bayescode_run_chains: a parallel chain failed (likely a ",
                    "fork-unsafe multithreaded BLAS, e.g. macOS Accelerate); ",
                    "re-running all chains sequentially. To keep parallelism set ",
                    "VECLIB_MAXIMUM_THREADS=1 before starting R, or pass ",
                    "parallel = FALSE.", call. = FALSE, immediate. = TRUE)
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

#' R-hat / ESS summary across chains
#'
#' @description Computes split-R-hat and bulk-ESS (via the \pkg{posterior}
#'   package) for each scalar or matrix history key across the chains returned by
#'   [ai4bayescode_run_chains()].
#' @param run The list returned by [ai4bayescode_run_chains()] (needs at least
#'   two chains).
#' @param keys Optional character vector of history keys to summarise; defaults
#'   to all keys.
#' @param drop_burn Number of leading draws to drop per chain before computing
#'   diagnostics. Defaults to 0.
#' @param order_components Logical; if `TRUE`, sort each draw's components within
#'   every matrix-valued key (order statistics) before computing R-hat/ESS -> a
#'   LABEL-INVARIANT summary for exchangeable (mixture/cluster) parameters.
#'   Defaults to `FALSE`, in which case the RAW R-hat is reported (unchanged
#'   behaviour) but the function still scans for label switching and warns when a
#'   high R-hat is only a labelling artefact.
#' @return A named list, one entry per key, each with `rhat` and `ess_bulk`
#'   (plus `max_rhat` / `min_ess` for matrix-valued keys). When label switching is
#'   detected, the affected keys (each with their `raw` and `ordered` max R-hat)
#'   are also attached as `attr(<result>, "label_switch")`.
#' @export
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


