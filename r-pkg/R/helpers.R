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
#   * ai4bayescode_predict() / ai4bayescode_close_models() -- prediction from,
#                                 and shutdown of, the models of a run.
#   * ai4bayescode_rhat_summary() -- R-hat / ESS summary across chains.
#
# Ported verbatim (behavior-preserving) from the standalone scripts
# AI4BayesCode/R/AI4BayesCode_helpers.R and AI4BayesCode/R/ai4bayescode_run_chains.R
# so that `library(AI4BayesCode)` provides them too.
# ----------------------------------------------------------------------------


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
#'   escape hatch; when the runner already
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
#' @param hist Named list of posterior draws (scalars as numeric vectors,
#'   vector parameters as matrices, draws in rows), or a model object with a
#'   `get_history()` method, whose history is then used.
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
#' # or pass the model itself (same as m$get_history()):
#' # ai4bayescode_diagnose(m, n_burn = 1000)
#' }
#' @export
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


#' Run multiple MCMC chains for an AI4BayesCode wrapper class
#'
#' @description Runs `n_chains` chains from a deterministic constructor
#'   closure and returns the per-chain histories, the models, the seeds used,
#'   and the per-chain wall time. Each chain builds a fresh model with
#'   `model_ctor(seed)`, runs `n_burn` warmup sweeps, discards the stored draws
#'   with the model's `clear_history()`, and runs `n_keep` sweeps, so the
#'   model's own `get_history()` and `predict_at()` cover the `n_keep`
#'   post-warmup draws only. The histories in `run$histories` are KEEP-ONLY,
#'   so downstream `ai4bayescode_rhat_summary()` / `ai4bayescode_diagnose()`
#'   need no `drop_burn` / `n_burn` bookkeeping.
#'
#'   The chains run in forked worker processes on POSIX when
#'   `parallel = TRUE`, `n_cores > 1` and `n_chains > 1`, and sequentially in
#'   this session otherwise. With `keep_models = TRUE` (default) a parallel
#'   run uses persistent worker processes (`parallel::makeForkCluster`):
#'   chain `i` runs in worker `((i - 1) %% n_workers) + 1`, and its model is
#'   kept in that process.
#'   `run$models[[i]]` is then a handle of class `"ai4bayescode_chain_model"`:
#'   `run$models[[i]]$predict_at(newdata)`, or any other method call, is
#'   evaluated on chain `i`'s model in its worker process and returns the
#'   result (an error there is raised here with the worker's message). The
#'   worker processes are terminated by `ai4bayescode_close_models(run)`,
#'   when the run and every handle taken from it are garbage-collected, or
#'   when R exits. A handle restored with `readRDS()` in another R session
#'   raises an error. A sequential run returns the model objects themselves in
#'   `run$models`. With `keep_models = FALSE` a parallel run uses
#'   `parallel::mclapply`, which evaluates the chains in this session when
#'   `as.integer(n_cores) < 2`, and `run$models` is `NULL`.
#'
#'   A model without `clear_history()` (compiled against AI4BayesCode older
#'   than 1.1.3, or a custom object) gives the same `run$histories`, with the
#'   first `n_burn` draws dropped here, but its own `get_history()` and
#'   `predict_at()` include the warmup draws; a message says so, and
#'   `ai4bayescode_predict()` drops them.
#' @param model_ctor Function of one argument (the integer seed, passed positionally; the parameter name is irrelevant) returning a fresh wrapper
#'   object exposing `step()` and `get_history()`. Must be deterministic in
#'   `seed`.
#' @param n_chains Number of chains. Defaults to 4.
#' @param n_burn Warmup iterations per chain. Defaults to 2000.
#' @param n_keep Kept iterations per chain. Defaults to 10000.
#' @param seeds Optional integer vector of length `n_chains`; auto-generated if
#'   `NULL`. For chain `i`, `set.seed(seeds[i])` (R's random-number generator,
#'   current `RNGkind()`) is called immediately before `model_ctor(seeds[i])`,
#'   in every path (worker processes, `parallel::mclapply`, sequential). The
#'   draws of a model that uses R's generator (`bart_block`, `genbart_block`
#'   and `softbart_block` under Rcpp) are therefore determined by `seeds`, and
#'   chains with different seeds use different streams. The caller's
#'   `.Random.seed` (or its absence) is saved before the run and restored
#'   after a sequential run and after the `parallel::mclapply` call, which
#'   evaluates the chains in this session when `as.integer(n_cores) < 2`, so
#'   the caller's random-number stream is unchanged by the run.
#' @param parallel Logical; run the chains in forked worker processes on
#'   POSIX. Defaults to `TRUE` (sequential on Windows).
#' @param mc.cores Alias of `n_cores`; an error is raised when both are given
#'   and differ.
#' @param verbose Logical; print a progress message. Defaults to `TRUE`.
#' @param n_cores Number of worker processes; defaults to
#'   `min(n_chains, max(1, parallel::detectCores() - 1))` for a parallel run.
#' @param keep_models Logical; return the models in `run$models`. Defaults to
#'   `TRUE`.
#' @return An object of class `"ai4bayescode_run"` (a list) with `histories`
#'   (list of per-chain histories, keep-only: each entry -- vector, matrix, or
#'   higher-dimensional array -- holds the `n_keep` post-warmup draws along
#'   its first axis; an entry with a single stored draw, i.e.
#'   `keep_history = FALSE`, is returned unchanged), `models` (list of the
#'   `n_chains` models -- handles after a parallel run, model objects after a
#'   sequential one -- or `NULL` when `keep_models = FALSE`), `seeds`, `wall`
#'   (per-chain seconds), `n_burn`, `n_keep`, and `history_cleared` (logical,
#'   one per chain: `TRUE` when `clear_history()` was called after the
#'   warmup). See `ai4bayescode_predict()` for prediction from all chains and
#'   `ai4bayescode_close_models()` to terminate the worker processes.
#' @export
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

    use_parallel <- parallel && .Platform$OS.type != "windows" &&
                    requireNamespace("parallel", quietly = TRUE)

    if (is.null(n_cores)) {
        n_cores <- if (use_parallel)
            min(n_chains, max(1, parallel::detectCores() - 1))
        else 1
    }

    chain_ok <- function(r) !inherits(r, "try-error") && !is.null(r$history)

    # The BLAS is pinned to one thread in .onLoad (zzz.R), which is the only
    # place it can work -- Accelerate decides whether to use its dispatch
    # queues at the FIRST BLAS call, so a cap applied here would change
    # nothing. If a variable is not "1" by now the caller overrode it, and
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

    # .ai4b_rc_one_chain() calls set.seed(seeds[i]) before it constructs
    # chain i. The caller's .Random.seed (NULL when it does not exist) is
    # saved here, before the run, and restored on exit from run_sequential()
    # and from run_mclapply(): parallel::mclapply() evaluates the chains in
    # this R session, without forking, when as.integer(mc.cores) < 2, and
    # with RNGkind "L'Ecuyer-CMRG" it creates .Random.seed when none exists
    # (mc.reset.stream()).
    rng_caller <- .ai4b_rc_rng_get()
    run_sequential <- function(keep) {
        on.exit(.ai4b_rc_rng_set(rng_caller))
        lapply(seeds, one_chain, keep = keep)
    }
    run_mclapply <- function() {
        on.exit(.ai4b_rc_rng_set(rng_caller))
        parallel::mclapply(seeds, one_chain, mc.cores = n_cores,
                           mc.set.seed = TRUE)
    }

    models <- NULL
    if (keep_models && use_parallel && n_cores > 1 && n_chains > 1) {
        # Persistent forked workers: the model of each chain is kept in its
        # worker process, and run$models holds handles to them.
        n_workers <- as.integer(min(n_cores, n_chains))
        if (verbose)
            message("ai4bayescode_run_chains: running ", n_chains,
                    " chains on ", n_workers, " cores (parallel)")
        warn_if_blas_unpinned()
        pr <- .ai4b_rc_pool_run(model_ctor, seeds, n_burn, n_keep, n_workers)
        if (is.null(pr$error)) {
            results <- pr$results
            models  <- pr$handles
        } else {
            # The pool could not be started, or a chain or a worker process
            # failed (a fork-unsafe multithreaded BLAS -- notably macOS
            # Accelerate -- makes a forked chain segfault). The pool is closed
            # by now; every chain is re-run sequentially in this session.
            warning("ai4bayescode_run_chains: ", pr$error, "; re-running all ",
                    "chains sequentially in this session. If a fork-unsafe ",
                    "multithreaded BLAS (e.g. macOS Accelerate) is the cause, ",
                    "set VECLIB_MAXIMUM_THREADS=1 before starting R to keep ",
                    "parallelism, or pass parallel = FALSE.",
                    call. = FALSE, immediate. = TRUE)
            results <- run_sequential(keep = TRUE)
            models  <- lapply(results, `[[`, "model")
        }
    } else if (!keep_models && use_parallel && n_cores > 1 && n_chains > 1) {
        if (verbose)
            message("ai4bayescode_run_chains: running ", n_chains,
                    " chains on ", n_cores, " cores (parallel)")
        warn_if_blas_unpinned()
        results <- run_mclapply()
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
            results <- run_sequential(keep = FALSE)
        }
    } else {
        if (verbose)
            message("ai4bayescode_run_chains: running ", n_chains,
                    " chains sequentially")
        results <- run_sequential(keep = keep_models)
        if (keep_models) models <- lapply(results, `[[`, "model")
    }

    # Detect failures (after any sequential fallback -- a real model/data error).
    for (i in seq_along(results)) {
        if (!chain_ok(results[[i]])) {
            stop("ai4bayescode_run_chains: chain ", i, " failed")
        }
    }
    .ai4b_rc_history_notes(results, n_burn, n_keep)

    structure(list(
        histories       = lapply(results, `[[`, "history"),
        models          = models,
        seeds           = sapply(results, `[[`, "seed"),
        wall            = sapply(results, `[[`, "wall"),
        n_burn          = n_burn,
        n_keep          = n_keep,
        history_cleared = vapply(results, function(r) isTRUE(r$cleared),
                                 logical(1))
    ), class = "ai4bayescode_run")
}


#' Posterior prediction from every chain of a run
#'
#' @description Calls `predict_at(newdata)` on the model of every chain of a
#'   run made by `ai4bayescode_run_chains(..., keep_models = TRUE)` and
#'   returns the draws of all chains stacked (`pool = TRUE`) or per chain.
#'   After a parallel run, one request goes to each worker process and the
#'   worker processes compute their chains concurrently. Each chain gives the
#'   draws its model's `predict_at()` returns, i.e. one row per post-warmup
#'   draw; for a chain whose model has no `clear_history()`
#'   (`run$history_cleared[i]` is `FALSE`) the first `n_burn` rows of every
#'   entry with `n_burn + n_keep` rows are dropped. `predict(run, newdata)`
#'   is the same call.
#' @param run An object returned by `ai4bayescode_run_chains()`.
#' @param newdata Named list passed to each model's `predict_at()`. The
#'   default `list()` gives the posterior predictive at the training data.
#' @param pool Logical. `TRUE` (default) returns, for each output key, one
#'   matrix with the rows of all chains in chain order (an error is raised
#'   when the chains give different keys or numbers of columns); `FALSE`
#'   returns a list with one named list per chain.
#' @param object An object returned by `ai4bayescode_run_chains()`.
#' @param ... Unused.
#' @return With `pool = TRUE`, a named list with one matrix per output key
#'   (`n_chains * n_keep` rows for models built with keep_history = TRUE); with
#'   `pool = FALSE`, a list of `n_chains` named lists as returned by
#'   `predict_at()`.
#' @examples
#' \dontrun{
#' run  <- ai4bayescode_run_chains(function(seed) new(MyModel, X, y, seed, TRUE),
#'                                 n_chains = 4, n_burn = 1000, n_keep = 2000)
#' yrep <- predict(run)                    # posterior predictive at the data
#' pred <- predict(run, list(X = X_test))  # at new covariates
#' by_chain <- ai4bayescode_predict(run, list(X = X_test), pool = FALSE)
#' ai4bayescode_close_models(run)          # terminate the worker processes
#' }
#' @export
ai4bayescode_predict <- function(run, newdata = list(), pool = TRUE) {
    models <- if (is.list(run)) run$models
    if (is.null(models))
        stop("ai4bayescode_predict(): the run holds no models (run$models is ",
             "NULL). Rerun ai4bayescode_run_chains() with keep_models = TRUE.",
             call. = FALSE)
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

    # A model without clear_history() predicts from its warmup draws too.
    n_burn  <- if (is.null(run$n_burn)) 0L else as.integer(run$n_burn)
    n_keep  <- if (is.null(run$n_keep)) 0L else as.integer(run$n_keep)
    cleared <- run$history_cleared
    if (n_burn > 0L && length(cleared) == n_chains) {
        for (i in which(!cleared)) {
            preds[[i]] <- lapply(preds[[i]], function(x)
                if (.ai4b_rc_n_rows(x) == n_burn + n_keep)
                    .ai4b_rc_strip_burn(x, n_burn) else x)
        }
    }
    if (!isTRUE(pool)) return(preds)

    keys <- names(preds[[1L]])
    for (i in seq_len(n_chains)[-1L]) {
        if (!setequal(names(preds[[i]]), keys))
            stop(sprintf(paste0("ai4bayescode_predict(): chain 1 gives the keys ",
                                "(%s) and chain %d the keys (%s); use pool = FALSE."),
                         paste(keys, collapse = ", "), i,
                         paste(names(preds[[i]]), collapse = ", ")), call. = FALSE)
    }
    out <- lapply(keys, function(k) {
        parts <- lapply(preds, function(p) .ai4b_rc_as_rows(p[[k]]))
        cols  <- lapply(parts, function(x) dim(x)[-1L])
        same  <- vapply(cols, identical, logical(1), cols[[1L]])
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

#' @rdname ai4bayescode_predict
#' @exportS3Method stats::predict
predict.ai4bayescode_run <- function(object, newdata = list(), pool = TRUE, ...)
    ai4bayescode_predict(object, newdata = newdata, pool = pool)


#' Terminate the worker processes that hold the models of a run
#'
#' @description After a parallel `ai4bayescode_run_chains()` with
#'   `keep_models = TRUE`, the model of each chain is held in a worker
#'   process, and the worker processes are terminated only by this function,
#'   by garbage collection of the run and of every handle taken from it, or
#'   when R exits. A method call through a handle of the run afterwards raises
#'   the error "the worker processes holding the models of this run were
#'   closed (ai4bayescode_close_models)". A second call does nothing, and so
#'   does a call on a run whose models are objects in this session or that was
#'   made with `keep_models = FALSE`.
#' @param run An object returned by `ai4bayescode_run_chains()`.
#' @return Invisibly `NULL`.
#' @export
ai4bayescode_close_models <- function(run) {
    for (p in .ai4b_rc_run_pools(run)) .ai4b_rc_pool_close(p)
    invisible(NULL)
}


#' Compact print for an ai4bayescode_run result
#'
#' Shows the run settings, the dimensions of the history entries, the wall
#' time, and where the models are, instead of the draws. The draws are
#' accessible by name (`x$histories`, `x$models`).
#'
#' @param x An `ai4bayescode_run` result (from `ai4bayescode_run_chains()`).
#' @param ... Unused.
#' @return `x`, invisibly.
#' @exportS3Method base::print
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


#' Handle to the model of one chain in a worker process
#'
#' After a parallel `ai4bayescode_run_chains()` with `keep_models = TRUE`,
#' `run$models[[i]]` is a handle of class `"ai4bayescode_chain_model"`.
#' `h$method(...)` calls `method` on chain `i`'s model in the worker process
#' that holds it, and returns the result; output, messages and warnings of the
#' call are shown here, and an error is raised here with the worker's message.
#' Calls are evaluated in order and change the model as the same call on the
#' model object would (`h$step(5)` adds five draws to its history).
#'
#' @param x A handle (an element of `run$models`).
#' @param name Name of the model's method.
#' @param value Unused; assignment through a handle raises an error.
#' @param pattern Regular expression for tab completion.
#' @param ... Unused.
#' @return `$` returns a function that calls the method in the worker process;
#'   `print` returns `x` invisibly.
#' @name ai4bayescode_chain_model
NULL

#' @rdname ai4bayescode_chain_model
#' @export
`$.ai4bayescode_chain_model` <- function(x, name) {
    force(name)
    function(...) .ai4b_rc_handle_call(x, name, list(...))
}

#' @rdname ai4bayescode_chain_model
#' @export
`$<-.ai4bayescode_chain_model` <- function(x, name, value)
    stop("assignment through an ai4bayescode_chain_model handle is not ",
         "supported; call a method of the model instead, e.g. ",
         "h$set_current(list(...)).", call. = FALSE)

#' @rdname ai4bayescode_chain_model
#' @exportS3Method utils::.DollarNames
.DollarNames.ai4bayescode_chain_model <- function(x, pattern = "") {
    nms <- .subset2(x, "methods")
    tryCatch(grep(pattern, nms, value = TRUE), error = function(e) nms)
}

#' @rdname ai4bayescode_chain_model
#' @exportS3Method base::print
print.ai4bayescode_chain_model <- function(x, ...) {
    pool  <- .subset2(x, "pool")
    chain <- .subset2(x, "chain")
    cat(sprintf("<ai4bayescode_chain_model: chain %d of %d, %s>\n", chain,
                pool$n_chains, .subset2(x, "class_name")))
    cat(switch(.ai4b_rc_pool_state(pool),
        open   = sprintf(paste0("  The model is in worker process %d of this R ",
                                "session; x$<method>(...) is evaluated there.\n"),
                         pool$pids[pool$node_of[chain]]),
        closed = paste0("  The worker processes holding the models of this run ",
                        "were closed (ai4bayescode_close_models).\n"),
        other  = paste0("  Created in another R session or process; the model is ",
                        "not available here.\n")))
    invisible(x)
}


# ---------------------------------------------------------------------------
# Internals of ai4bayescode_run_chains() / ai4bayescode_predict().
#
# A parallel run with keep_models = TRUE forks one pool of worker processes
# with parallel::makeForkCluster(). The constructor is passed to the workers
# through fork() (it is stored in .ai4b_rc_env immediately before forking and
# removed from the parent right after), never through serialization: its
# environment can hold an Rcpp class generator, whose external pointer would
# be NULL after unserialize(). Only seeds, chain indices, method names and
# arguments are sent to the workers; the models of a worker's chains are
# stored in that worker's own copy of .ai4b_rc_env$models.
#
# In the parent, .ai4b_rc_env$pools maps the token of every open pool to its
# connections, worker pids and request counters. A handle refers to a "pool"
# environment holding the token, the session id and the chain-to-worker map,
# but not the connections, so a handle restored with readRDS() in another
# session contains no connection. A finalizer on the pool environment
# (onexit = TRUE) closes the pool when the run and its handles are
# garbage-collected, or when R exits.
# ---------------------------------------------------------------------------
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
# seed_val and the current RNGkind(). The history returned is keep-only;
# `model` is the model object itself.
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
# that clear_history() left non-empty, each emitted once per run.
.ai4b_rc_history_notes <- function(results, n_burn, n_keep) {
    cls <- function(rs) paste(unique(vapply(rs, `[[`, character(1), "class")),
                              collapse = ", ")
    no_clear <- Filter(function(r) !isTRUE(r$has_clear), results)
    if (n_burn > 0L && length(no_clear))
        message("ai4bayescode_run_chains: ", cls(no_clear), " has no ",
                "clear_history() method, so the model's own get_history() and ",
                "predict_at() include the n_burn = ", n_burn, " warmup draws ",
                "(run$histories exclude them, and ai4bayescode_predict() drops ",
                "them). Recompiling the model with AI4BayesCode >= 1.1.3 adds ",
                "clear_history().")
    with_extra <- Filter(function(r) length(r$extra) > 0L, results)
    if (length(with_extra))
        warning("ai4bayescode_run_chains: after clear_history(), the history ",
                "entries ", paste(unique(unlist(lapply(with_extra, `[[`, "extra"))),
                                  collapse = ", "),
                " of ", cls(with_extra), " have more than n_keep = ", n_keep,
                " rows. Their first n_burn = ", n_burn, " rows are dropped in ",
                "run$histories, but the model's own get_history() and ",
                "predict_at() still contain the warmup draws for them: the ",
                "class needs a clear_extra_history_() hook that empties them.",
                call. = FALSE)
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
# state to .Random.seed (PutRNGstate) when the last model containing one is
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
# warnings and error it produced, so that the parent can show them.
.ai4b_rc_capture <- function(fun) {
    value <- NULL; visible <- FALSE; err <- NULL
    warns <- character(0); msgs <- character(0)
    out <- utils::capture.output(invisible(withCallingHandlers(
        tryCatch({
            v <- withVisible(fun())
            value   <- v$value
            visible <- v$visible
        }, error = function(e)
            err <<- list(message = conditionMessage(e), class = class(e))),
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

# Worker side: run the chains of one worker and store their models.
.ai4b_rc_worker_run <- function(job, n_burn, n_keep) {
    ctor <- get0("ctor", envir = .ai4b_rc_env, inherits = FALSE)
    models <- list()
    out <- list()
    for (j in seq_along(job$chains)) {
        cap <- .ai4b_rc_capture(function() {
            if (!is.function(ctor))
                stop("the model constructor is not available in this worker ",
                     "process")
            .ai4b_rc_one_chain(ctor, job$seeds[[j]], n_burn, n_keep)
        })
        if (is.null(cap$error)) {
            m <- cap$value$model
            models[[as.character(job$chains[[j]])]] <- m
            cap$value$model   <- NULL
            cap$value$methods <- tryCatch(as.character(utils::.DollarNames(m, "")),
                                          error = function(e) character(0))
        }
        out[[j]] <- cap
        if (!is.null(cap$error)) break   # the run falls back to sequential
    }
    .ai4b_rc_env$models <- models
    list(seq = job$seq, out = out)
}

# Worker side: call `job$method` with `job$args` on the model of each chain
# in `job$chains`.
.ai4b_rc_worker_call <- function(job) {
    models <- .ai4b_rc_env$models
    list(seq = job$seq, out = lapply(job$chains, function(i)
        .ai4b_rc_capture(function() {
            m <- models[[as.character(i)]]
            if (is.null(m))
                stop("this worker process holds no model of chain ", i)
            f <- eval(call("$", m, job$method))
            if (!is.function(f))
                stop("'", job$method, "' is not a method of ",
                     .ai4b_rc_class_name(m))
            do.call(f, job$args, quote = TRUE)
        })))
}

# Fork the worker processes of a pool and register the pool.
.ai4b_rc_pool_open <- function(model_ctor, n_workers, n_chains) {
    # Pools of earlier runs that are no longer reachable are closed by their
    # finalizers; collect them first, so that their connections are free (R
    # allows about 128 open connections, and each worker process uses one).
    gc()
    assign("ctor", model_ctor, envir = .ai4b_rc_env)
    on.exit(rm(list = "ctor", envir = .ai4b_rc_env), add = TRUE)
    cl <- parallel::makeForkCluster(n_workers)
    pids <- tryCatch(as.integer(unlist(parallel::clusterCall(cl, Sys.getpid))),
                     error = function(e) {
                         try(parallel::stopCluster(cl), silent = TRUE)
                         stop(e)
                     })
    n <- if (is.null(.ai4b_rc_env$n_pools)) 1L else .ai4b_rc_env$n_pools + 1L
    .ai4b_rc_env$n_pools <- n
    token <- paste0("pool", n)
    entry <- new.env(parent = emptyenv())
    entry$cl   <- cl
    entry$pids <- pids
    entry$seq  <- 0L                   # number of the last request
    entry$sent <- integer(n_workers)   # last request sent to each worker
    entry$recv <- integer(n_workers)   # last request whose result was read
    assign(token, entry, envir = .ai4b_rc_registry())
    pool <- new.env(parent = emptyenv())
    pool$token     <- token
    pool$session   <- .ai4b_rc_session()
    pool$pids      <- pids
    pool$n_workers <- n_workers
    pool$n_chains  <- n_chains
    pool$node_of   <- ((seq_len(n_chains) - 1L) %% n_workers) + 1L
    reg.finalizer(pool, .ai4b_rc_pool_close, onexit = TRUE)
    pool
}

# "open", "closed", or "other" (created in another R session or process).
.ai4b_rc_pool_state <- function(pool) {
    if (!identical(pool$session, .ai4b_rc_session())) return("other")
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
        closed = stop(if (is.null(pool$closed_reason))
                          paste0("the worker processes holding the models of ",
                                 "this run were closed (ai4bayescode_close_models)")
                      else pool$closed_reason, call. = FALSE),
        get(pool$token, envir = .ai4b_rc_registry(), inherits = FALSE))
}

# Terminate the worker processes of a pool; a no-op for a closed pool or a
# pool of another session. Idle workers exit on the "DONE" message of
# parallel::stopCluster(); a worker whose result of the last request was not
# read (an interrupted call), or every worker when force = TRUE, is sent
# SIGTERM.
.ai4b_rc_pool_close <- function(pool, force = FALSE, reason = NULL) {
    if (!identical(.ai4b_rc_pool_state(pool), "open")) return(invisible(FALSE))
    reg   <- .ai4b_rc_registry()
    entry <- get(pool$token, envir = reg, inherits = FALSE)
    rm(list = pool$token, envir = reg)
    if (!is.null(reason)) pool$closed_reason <- reason
    busy <- force | entry$sent > entry$recv
    for (w in seq_along(entry$cl)) {
        if (busy[w]) tools::pskill(entry$pids[w], tools::SIGTERM)
        done <- !busy[w] && !inherits(try(parallel::stopCluster(entry$cl[w]),
                                          silent = TRUE), "try-error")
        if (!done) try(close(entry$cl[[w]]$con), silent = TRUE)
    }
    invisible(TRUE)
}

# Send one job to each worker in `nodes` (they compute concurrently) and
# return their results in the order of `nodes`. Every request carries a
# number that the worker returns with its result. A call interrupted before
# its results arrived leaves those results unread; a result with an earlier
# number is such a result and is discarded, and the result of this request
# is the next one on that connection. A failed connection closes the pool.
.ai4b_rc_exchange <- function(pool, nodes, jobs, fun, ...) {
    entry <- .ai4b_rc_pool_entry(pool)
    s <- entry$seq + 1L
    entry$seq <- s
    entry$sent[nodes] <- s
    for (k in seq_along(jobs)) jobs[[k]]$seq <- s
    fail <- function(e) {
        msg <- paste0("the connection to a worker process of this run failed (",
                      conditionMessage(e), ")")
        .ai4b_rc_pool_close(pool, force = TRUE, reason = paste0(
            "the worker processes holding the models of this run were closed ",
            "after ", msg))
        stop(msg, "; the worker processes of this run were closed.",
             call. = FALSE)
    }
    out <- tryCatch(parallel::clusterApply(entry$cl[nodes], jobs, fun, ...),
                    error = function(e) e)
    if (inherits(out, "error")) fail(out)
    seq_of <- function(r) if (is.list(r)) r$seq else NA
    for (k in seq_along(nodes)) {
        r <- out[[k]]
        while (!identical(seq_of(r), s)) {
            # A worker's reply on a socket node is one serialized list whose
            # `value` is the result of the function sent.
            r <- tryCatch(unserialize(entry$cl[[nodes[k]]]$con)$value,
                          error = function(e) e)
            if (inherits(r, "error")) fail(r)
        }
        out[[k]] <- r$out
    }
    entry$recv[nodes] <- s
    out
}

# Run all chains in a new pool. Returns list(results, handles) or, when the
# pool cannot be started or a chain fails, list(error = <message>) with the
# pool closed.
.ai4b_rc_pool_run <- function(model_ctor, seeds, n_burn, n_keep, n_workers) {
    n_chains <- length(seeds)
    pool <- tryCatch(.ai4b_rc_pool_open(model_ctor, n_workers, n_chains),
                     error = function(e) e)
    if (inherits(pool, "error"))
        return(list(error = paste0("the worker processes could not be started (",
                                   conditionMessage(pool), ")")))
    finished <- FALSE
    on.exit(if (!finished) .ai4b_rc_pool_close(pool, force = TRUE), add = TRUE)
    jobs <- lapply(seq_len(n_workers), function(w) {
        idx <- which(pool$node_of == w)
        list(chains = idx, seeds = seeds[idx])
    })
    out <- tryCatch(.ai4b_rc_exchange(pool, seq_len(n_workers), jobs,
                                      .ai4b_rc_worker_run,
                                      n_burn = n_burn, n_keep = n_keep),
                    error = function(e) e)
    if (inherits(out, "error"))
        return(list(error = conditionMessage(out)))
    caps <- vector("list", n_chains)
    for (w in seq_len(n_workers))
        caps[jobs[[w]]$chains[seq_along(out[[w]])]] <- out[[w]]
    bad <- which(vapply(caps, function(cp) is.null(cp) || !is.null(cp$error),
                        logical(1)))
    if (length(bad))
        return(list(error = sprintf("a parallel chain failed (chain %d: %s)",
                                    bad[1L], caps[[bad[1L]]]$error$message)))
    results <- lapply(seq_len(n_chains), function(i) .ai4b_rc_relay(caps[[i]], i))
    handles <- lapply(seq_len(n_chains), function(i)
        .ai4b_rc_handle(pool, i, results[[i]]$class, results[[i]]$methods))
    finished <- TRUE
    list(results = results, handles = handles, error = NULL)
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

# Call `method` on the models of `chains` of a pool; returns the captured
# results in the order of `chains`.
.ai4b_rc_remote <- function(pool, chains, method, args) {
    nodes <- unique(pool$node_of[chains])
    jobs  <- lapply(nodes, function(w)
        list(chains = chains[pool$node_of[chains] == w], method = method,
             args = args))
    out  <- .ai4b_rc_exchange(pool, nodes, jobs, .ai4b_rc_worker_call)
    caps <- vector("list", length(chains))
    for (k in seq_along(nodes)) caps[match(jobs[[k]]$chains, chains)] <- out[[k]]
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
    st <- vapply(pools, .ai4b_rc_pool_state, character(1))
    if (all(st == "open")) {
        pids <- unlist(lapply(pools, function(p) p$pids))
        return(sprintf(paste0("in %d worker process%s of this R session (pid %s); ",
                              "run$models[[i]]$<method>(...) is evaluated there, ",
                              "and ai4bayescode_close_models(run) terminates them"),
                       length(pids), if (length(pids) == 1L) "" else "es",
                       paste(pids, collapse = ", ")))
    }
    if (all(st == "closed")) return("closed (ai4bayescode_close_models)")
    if (all(st == "other"))
        return("in worker processes of another R session (not available here)")
    "partly closed or in another R session"
}

# One prediction entry as rows = draws: a vector becomes one column.
.ai4b_rc_as_rows <- function(x) if (is.null(dim(x))) matrix(x, ncol = 1L) else x

# Stack per-chain entries along the first axis.
.ai4b_rc_bind_rows <- function(parts) {
    if (length(dim(parts[[1L]])) <= 2L) return(do.call(rbind, parts))
    d    <- dim(parts[[1L]])
    perm <- c(seq_along(d)[-1L], 1L)
    n    <- sum(vapply(parts, function(x) dim(x)[1L], integer(1)))
    arr  <- array(unlist(lapply(parts, aperm, perm)), dim = c(d[-1L], n))
    aperm(arr, c(length(d), seq_len(length(d) - 1L)))
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

#' Construct a model with some components fixed
#'
#' @description Builds a generated model and immediately sets and freezes the
#'   components listed in `fixed`, in one call. The frozen components are held
#'   at the supplied values for every subsequent `step()`, so the sampler
#'   targets the conditional posterior of the remaining components. Equivalent
#'   to the three-step form:
#'   ```
#'   m <- new(module_class, ...)
#'   m$set_current(fixed)
#'   m$freeze(names(fixed), quiet = quiet_freeze)
#'   ```
#' @param module_class The generated model's Rcpp module class (for example
#'   `MyGaussianReg`).
#' @param ... Constructor arguments forwarded to `new(module_class, ...)`.
#' @param fixed Named list of `component = value` pairs to set and freeze. An
#'   empty list constructs the model without freezing anything. Names must be
#'   FLAT -- a top-level child block name, or a component of a
#'   `joint_nuts_block`; whatever `set_current()` accepts. Dot-path names
#'   (nested composite descent, `rjmcmc_block` sub-keys) are rejected, because
#'   `set_current()` does not route dot-path keys: freezing a value that was
#'   never set would give a wrong posterior. To freeze at a dot path, call
#'   `set_current()` at the matching composite level and then `freeze()`.
#' @param quiet_freeze Passed as `quiet=` to `m$freeze()`. Defaults to `TRUE`:
#'   nothing can be frozen yet at construction time, so the redundant-refreeze
#'   warning has nothing to report.
#' @return The constructed model, with the `fixed` components set and frozen.
#' @importFrom methods new
#' @export
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


