# generate.R
#
# NL -> validated sampler code, front doors:
#   ai4bayescode_prompt(model_description, ...)   -- PURE prompt builder (no network).
#   ai4bayescode_generate(model_description, ...)  -- drive an LLM (Claude today) to
#                                                 generate a sampler, console-
#                                                 interactive for missing params
#                                                 and for PRIOR elicitation.
#   ai4bayescode_models()                          -- supported LLM registry.
#
# Design (2026-06-21):
#   * LLM-AGNOSTIC by design: the AI4BayesCode skill corpus works with any LLM.
#     `LLM` resolves through ai4bayescode_models() to a (provider, model id). The
#     Anthropic (Claude) provider is fully implemented as the illustration; other
#     providers are a clean extension point (an informative error until added).
#   * Model ids are bare Messages-API ids (e.g. `claude-opus-4-8`).
#   * `effort` is a SEPARATE knob (default "high") -- low effort yields low-quality
#     samplers, so quality-sensitive codegen defaults high.
#   * Interactive-if-missing: with interactive = TRUE, any NULL/empty parameter is
#     asked in the console (readline / menu). PRIORS are always elicited
#     interactively, by the model via an `ask_user` tool, one parameter at a time.
#   * Validation is MANDATORY and runs a validate -> repair-to-convergence loop:
#     each generation attempt is compiled (ai4bayescode_source) and its emitted
#     runner is executed; the runner MUST print `AI4BAYES_VALIDATE: PASS` as its
#     final line (max rank-normalized R-hat < 1.01 across two chains). On any
#     failure the compile / run output is fed back to the model for repair, up to
#     `max_attempts` times.

# ---------------------------------------------------------------------------
# Skill corpus helpers
# ---------------------------------------------------------------------------
#' @keywords internal
#' @noRd
.ai4b_skill <- function(name) {
    p <- ai4bayescode_skills_path(name)
    if (!nzchar(p) || !file.exists(p)) return("")
    paste(readLines(p, warn = FALSE), collapse = "\n")
}

#' @keywords internal
#' @noRd
.ai4b_default_skill_set <- function(backend) {
    base <- c("start.md", "codegen.md", "codegen_cpp.md", "block_catalogue.md",
              "constraints.md", "codegen_priors.md", "rcpp_api.md", "validator.md")
    runner <- switch(backend,
                     "R"      = "codegen_r_runner.md",
                     "Python" = "codegen_python_runner.md",
                     "both"   = c("codegen_r_runner.md", "codegen_python_runner.md"))
    unique(c(base, runner))
}

#' @keywords internal
#' @noRd
.ai4b_derive_class_name <- function(description) {
    words <- regmatches(description, gregexpr("[A-Za-z][A-Za-z0-9]*", description))[[1]]
    words <- words[nchar(words) >= 3L]          # drop noise tokens (y, i, X, ...)
    words <- head(words, 3L)
    if (!length(words)) return("GeneratedModel")
    nm <- paste(paste0(toupper(substring(words, 1, 1)), substring(words, 2)), collapse = "")
    if (!grepl("^[A-Za-z_]", nm)) nm <- paste0("M_", nm)
    substr(nm, 1, 60L)
}

# Prior policy block. "interactive" -> instruct the model to elicit via ask_user.
#' @keywords internal
#' @noRd
.ai4b_prior_block <- function(priors) {
    if (identical(priors, "interactive")) {
        return(paste0(
"  - Priors: FIRST use every prior the model description ALREADY specifies\n",
"    -- e.g. `p(beta) propto 1`, `p(sigma) propto 1/sigma`, `beta ~ N(0, 10)`,\n",
"    `sigma ~ Half-Normal(2.5)` -- EXACTLY, and do NOT ask about them (a written\n",
"    prior IS the user's decision). ELICIT (via the `ask_user` tool, ONE\n",
"    parameter at a time: non-informative (default) / weakly-informative /\n",
"    literature-informed / fixed value / custom) ONLY the priors that are\n",
"    MISSING or genuinely AMBIGUOUS (e.g. Gamma(2,3) rate-vs-scale). If the\n",
"    description already specifies EVERY parameter's prior, ask NOTHING and go\n",
"    straight to code (codegen.md §2: 'Spec already COMPLETE -> skip the entire\n",
"    elicitation'). (See codegen_priors.md.)"))
    }
    if (is.list(priors)) {
        lines <- vapply(seq_along(priors),
                        function(i) sprintf("    - %s: %s", names(priors)[i], priors[[i]]), "")
        return(paste0(
            "  - Priors: use these EXACTLY where given; for any parameter not listed,\n",
            "    use a strictly NON-INFORMATIVE prior (see table below).\n",
            paste(lines, collapse = "\n")))
    }
    if (identical(priors, "weakly")) {
        return("  - Priors: weakly-informative defaults are acceptable where the model\n    description does not specify a prior.")
    }
    paste0(
"  - Priors: where the model description specifies a prior, use it EXACTLY.\n",
"    For any parameter whose prior is NOT specified, use a strictly\n",
"    NON-INFORMATIVE prior (do NOT substitute a weakly-informative default):\n",
"      * Real location / regression coef (mu, alpha, beta): FLAT IMPROPER p(.)\u221d1\n",
"        (omit the prior term in the log-density; likelihood only).\n",
"      * Positive scale (sigma, tau): Jeffreys p(sigma)\u221d1/sigma.\n",
"      * Probability on [0,1] (p): Beta(1,1) = uniform.\n",
"      * Simplex (theta_1..theta_K): Dirichlet(1,...,1) = uniform simplex.\n",
"      * Correlation matrix R: LKJ(eta=1) = uniform over correlation matrices.\n",
"      * Count rate (lambda): Jeffreys p(lambda)\u221d1/sqrt(lambda) via log+Jacobian.\n",
"      * Concentration (kappa): Jeffreys p(kappa)\u221d1/kappa.\n",
"    Apply the non-informative default SILENTLY; do not stop to ask.")
}

# Case-insensitive backend: a user may write "python"/"r"/"BOTH"; map to the
# canonical "R"/"Python"/"both" before match.arg. The default (an unmatched
# multi-value vector) passes through untouched.
#' @keywords internal
#' @noRd
.ai4b_norm_backend <- function(backend) {
    if (length(backend) != 1L) return(backend)
    nb <- unname(c(r = "R", python = "Python", both = "both")[tolower(trimws(backend))])
    if (is.na(nb)) backend else nb
}

#' @keywords internal
#' @noRd
.ai4b_build_user <- function(description, backend, output_path, classname,
                             priors, max_attempts, confirm_model = FALSE) {
    confirm_block <- if (isTRUE(confirm_model)) paste0(
"  - PRE-GENERATION MODEL CONFIRMATION -- do NOT skip it (codegen.md §3). After\n",
"    eliciting the priors and BEFORE writing ANY code, present the model AS YOU\n",
"    UNDERSTOOD IT for the user to verify, via the `ask_user` tool. ALWAYS include both:\n",
"      (a) the FULL model as display-math formulas (codegen.md §0b: `$$ ... $$`, never\n",
"          inline `$...$`) -- the likelihood AND every prior; and\n",
"      (b) a parameter summary table -- name / role / support / prior.\n",
"    (A prediction DAG is OPTIONAL -- include it only if it clarifies a complex model.)\n",
"    Ask the user to sign off, offering exactly: 'Correct -- generate the sampler' /\n",
"    'Not quite -- I will correct it'. If they do NOT confirm, ask what to change, apply\n",
"    it, and re-confirm. Only AFTER sign-off do you write any code; then emit the\n",
"    COMPLETE files in that next message (do not defer).\n")
    else paste0(
"  - SKIP the model-confirmation step (no LaTeX summary / DAG image). After any prior\n",
"    elicitation, emit the COMPLETE files immediately -- do not pause to summarize.\n")
    paste0(
"You are deploying the AI4BayesCode library to GENERATE A SAMPLER for the model\n",
"described at the end of this message.\n\n",
"TOOLS -- you have `read_file`, `grep`, and `glob` over the INSTALLED AI4BayesCode\n",
"package (examples/, skills/, include/). The skills reference worked reference\n",
"implementations ('see examples/GaussianLocationScale.cpp') and headers: READ the\n",
"relevant example with `read_file` BEFORE writing the .cpp (its class shape,\n",
"set_current, predict_at, get_current, the block config are all there), and read\n",
"the on-demand skills (skills/system_design.md, skills/joint_nuts_failure.md,\n",
"skills/hierarchical_re.md, skills/label_switching.md) when the model calls for\n",
"them. Do NOT invent an API you can read -- `grep` the headers/examples first.\n\n",
"Settings:\n",
sprintf("  - Runtime backend: %s\n", backend),
sprintf("  - Output folder:   %s/\n", output_path),
sprintf("  - Class name:      %s (valid C++ identifier; use as-is)\n", classname),
.ai4b_prior_block(priors), "\n",
confirm_block,
sprintf("  - Up to %d total generation attempts; iterate on failures.\n", max_attempts),
"\nDeliverables (emit as fenced code blocks labelled `// path: <out>/<file>`):\n",
sprintf("  - %s/%s.cpp                  (the sampler)\n", output_path, classname),
if (backend %in% c("R","both"))
  sprintf("  - %s/%s_runner.R             (R runner / Layer-3 harness)\n", output_path, classname) else "",
if (backend %in% c("Python","both"))
  sprintf("  - %s/%s_runner.py            (Python runner)\n", output_path, classname) else "",
"  - Register a y_rep stochastic refresher for the observation likelihood\n",
"    (MANDATORY -- Layer-3 R3 Bayesian p-values need posterior-predictive draws).\n",
"  - Code comments in English only.\n",
"\nCRITICAL COMPILE PITFALLS (these EXACT errors recur -- avoid them):\n",
"  - set_current: `composite_block::set_current` AND every block's `set_current`\n",
"    take a CONCATENATED `arma::vec` (sub_params order), NOT a `state_map` /\n",
"    `Rcpp::List`. `impl_->set_current(<map>)` does NOT compile ('no viable\n",
"    conversion from state_map to const arma::vec'). Build the arma::vec from the\n",
"    named entries (read `jblk.current()`, overwrite the named slots) and call\n",
"    `<block>.set_current(cat)` -- see `examples/GaussianLocationScale.cpp`.\n",
"  - `shared_data_t` has `declare_data_input(key)` and `data_input_keys()` (a set),\n",
"    but NO `is_data_input(key)`. To test membership: `data().data_input_keys().count(key)`.\n",
"  - Any method with C++ default arguments (readapt_NUTS, ...) loses them through\n",
"    the Rcpp module -- the R caller must pass ALL arguments explicitly.\n",
"\nMANDATORY VALIDATION (this is how the generator detects convergence):\n",
"  - You MUST emit a self-contained runner (the R runner shown above) that:\n",
"      1. simulates data from the model at known parameter values,\n",
"      2. compiles the sampler so the class is callable by name. NOTE:\n",
"         `ai4bayescode_sourceCpp(cpp)` binds the class in the CALLER's frame\n",
"         (`env = parent.frame()`). If you wrap the compile in a helper\n",
"         FUNCTION you MUST pass `env = globalenv()`, else `new(<Class>, ...)`\n",
"         fails with 'object <Class> not found'. Simplest: compile at top level.\n",
"      3. runs TWO independent chains from over-dispersed inits. For a NUTS /\n",
"         joint_nuts model, each chain MUST do:\n",
"           m$step(n_burnin); m$readapt_NUTS(N, FALSE, -1L); m$step(n_keep)\n",
"         The first-call warmup adapts the metric from the over-dispersed init,\n",
"         so WITHOUT re-adapting at the mode R-hat commonly stays > 1.01. Pass\n",
"         ALL 3 args to readapt_NUTS (Rcpp modules ignore C++ default arguments;\n",
"         calling with fewer errors 'could not find valid method'). Keep the\n",
"         LAST n_keep draws (readapt does not add history rows).\n",
"      4. computes the rank-normalized R-hat (`posterior::rhat`, Vehtari 2021)\n",
"         for EVERY model parameter across the two chains, and\n",
"      5. prints, as its VERY LAST line, EXACTLY one of:\n",
"           AI4BAYES_VALIDATE: PASS                    (if max rank-R-hat < 1.01)\n",
"           AI4BAYES_VALIDATE: FAIL maxRhat=<value>    (otherwise)\n",
"  - The generator runs this runner with Rscript and greps for\n",
"    `AI4BAYES_VALIDATE: PASS`. If it does not see PASS (compile error, runner\n",
"    error, or non-convergence) it will feed the compile / run output back to you\n",
"    and ask you to FIX and RE-EMIT the FULL .cpp and runner. Make the runner\n",
"    print that final line unconditionally on every successful run.\n",
"\n---\nModel description:\n", description, "\n")
}

# ---------------------------------------------------------------------------
# Validation: compile + run the emitted runner, detect convergence
# ---------------------------------------------------------------------------
# Compile the emitted .cpp, then run its runner and grep for the convergence
# sentinel. Returns list(ok, stage, detail) where stage is one of "compile",
# "no_runner", "convergence", or "converged".
#' @keywords internal
#' @noRd
.ai4b_validate <- function(cpp_path, runner_path, classname, verbose = FALSE) {
    # ---- 1. compile ----
    comp <- tryCatch({
        ai4bayescode_source(cpp_path, env = new.env()); NULL
    }, error = function(e) conditionMessage(e))
    if (!is.null(comp)) {
        if (verbose) message("  [validate] compile failed")
        return(list(ok = FALSE, stage = "compile", detail = comp))
    }

    # ---- 2. runner present? ----
    if (is.null(runner_path) || is.na(runner_path) || !file.exists(runner_path)) {
        if (verbose) message("  [validate] no runner emitted")
        return(list(ok = FALSE, stage = "no_runner", detail = "no runner emitted"))
    }

    # ---- 3. run the runner; capture stdout+stderr ----
    # Run it FROM the output dir: the runner compiles its .cpp via a RELATIVE
    # `ai4bayescode_sourceCpp("<Class>.cpp")` (the start.md convention), so it must
    # execute with the .cpp in its working dir or it errors "... does not exist"
    # (which the sentinel logic below would otherwise mislabel a "runtime crash").
    owd <- getwd()
    out <- tryCatch({
        setwd(dirname(runner_path))
        suppressWarnings(system2("Rscript", shQuote(runner_path),
                                 stdout = TRUE, stderr = TRUE))
    }, error = function(e) paste("runner error:", conditionMessage(e)))
    setwd(owd)
    out_txt <- paste(out, collapse = "\n")

    # ---- 4. convergence sentinel ----
    if (grepl("AI4BAYES_VALIDATE:\\s*PASS", out_txt)) {
        if (verbose) message("  [validate] converged")
        tail_lines <- tail(strsplit(out_txt, "\n", fixed = TRUE)[[1]], 15L)
        return(list(ok = TRUE, stage = "converged",
                    detail = paste(tail_lines, collapse = "\n")))
    }
    # No PASS sentinel -> distinguish WHY, rather than always blaming "convergence".
    tail_lines <- tail(strsplit(out_txt, "\n", fixed = TRUE)[[1]], 40L)
    detail <- paste(tail_lines, collapse = "\n")
    if (grepl("AI4BAYES_VALIDATE:\\s*FAIL", out_txt)) {       # ran to the end; R-hat too high
        if (verbose) message("  [validate] ran but did not converge")
        return(list(ok = FALSE, stage = "convergence", detail = detail))
    }
    if (grepl("Execution halted|Error in |Error:|Segmentation fault|terminate called|Abort trap|Traceback \\(most recent",
              out_txt)) {                                     # compiled but crashed at RUNTIME
        if (verbose) message("  [validate] runner crashed at runtime")
        return(list(ok = FALSE, stage = "runtime", detail = detail))
    }
    if (verbose) message("  [validate] runner did not reach the validate line")
    list(ok = FALSE, stage = "incomplete", detail = detail)   # no PASS, no FAIL, no error
}

# Build the repair user-message fed back to the model after a failed attempt.
#' @keywords internal
#' @noRd
.ai4b_repair_msg <- function(result) {
    if (identical(result$stage, "no_code")) {
        return(paste0(
"No sampler code was emitted. ", result$detail, "\n\n",
"Output the COMPLETE files NOW, and NOTHING else:\n",
"  - the full `.cpp` sampler, and\n",
"  - the full runner.\n",
"Emit EACH as its own fenced code block whose FIRST line is exactly\n",
"`// path: <out>/<file>` (e.g. `// path: ./generated/Model.cpp`). Do NOT write\n",
"any prose, plan, or confirmation before, between, or after the blocks -- output\n",
"ONLY the fenced code blocks. Keep the code complete but concise so it is not cut\n",
"off. The runner MUST print `AI4BAYES_VALIDATE: PASS` (or\n",
"`AI4BAYES_VALIDATE: FAIL maxRhat=<v>`) as its very last line.\n"))
    }
    hint <- switch(result$stage %||% "",
        compile     = "The C++ did NOT COMPILE. Fix the compile error below.",
        runtime     = "The sampler COMPILED but the runner CRASHED at RUNTIME -- this is NOT a convergence problem; fix the C++/runner bug below.",
        incomplete  = "The runner did NOT reach the `AI4BAYES_VALIDATE` line (it stopped early) -- find why it never finished.",
        convergence = "The runner ran to completion but rank-R-hat was too high (true non-convergence) -- fix the sampler/model.",
        "The generated sampler did not pass automated validation.")
    paste0(
"Validation FAILED at stage `", result$stage, "`. ", hint, "\n",
"Details below:\n\n",
"-----\n", result$detail, "\n-----\n\n",
"Please FIX the problem and RE-EMIT the FULL corrected files (do NOT send a diff\n",
"or only the changed lines). Emit the complete `.cpp` sampler AND the complete\n",
"R runner, each as a fenced code block whose first line is\n",
"`// path: <out>/<file>` (the same paths as before). The runner MUST still print\n",
"`AI4BAYES_VALIDATE: PASS` (or `AI4BAYES_VALIDATE: FAIL maxRhat=<v>`) as its very\n",
"last line so the generator can re-check convergence.\n")
}

# ---------------------------------------------------------------------------
# ai4bayescode_prompt(): pure prompt builder
# ---------------------------------------------------------------------------
#' Assemble the AI4BayesCode codegen prompt (no network)
#'
#' Builds the system + user prompt for natural-language -> sampler code, from the
#' bundled skill corpus and a set of settings. Pure and offline.
#'
#' @param model_description Model description (string) or a `.txt`/`.md` path.
#' @param backend `"R"` (default), `"Python"`, or `"both"`.
#' @param output_path Output folder the generated files target.
#' @param classname C++/Rcpp class name; default derived from the description.
#' @param priors `"noninformative"` (default), `"weakly"`, `"interactive"`, or a
#'   named list of parameter -> prior.
#' @param max_attempts Max generation attempts to instruct (default 5).
#' @param include_skills If `TRUE`, inline the backend skill subset into the system
#'   prompt (for a single-shot completion). If `FALSE` (default), `start.md`-anchored.
#' @param skills Optional skill file names to inline (overrides the default subset).
#' @param confirm_model If `TRUE`, instruct the model to confirm its understanding of
#'   the model (likelihood, parameters, priors) via `ask_user` before writing any code.
#' @return A list with `system`, `user`, `classname`, `backend`, `skills`.
#' @export
ai4bayescode_prompt <- function(model_description,
                            backend        = c("R", "Python", "both"),
                            output_path    = "./generated",
                            classname      = NULL,
                            priors         = "noninformative",
                            max_attempts   = 5L,
                            include_skills = FALSE,
                            skills         = NULL,
                            confirm_model  = FALSE) {
    backend <- .ai4b_norm_backend(backend)
    backend <- match.arg(backend)
    if (is.character(model_description) && length(model_description) == 1L &&
        file.exists(model_description) && grepl("\\.(txt|md)$", model_description)) {
        model_description <- paste(readLines(model_description, warn = FALSE), collapse = "\n")
    }
    if (!is.character(model_description) || length(model_description) != 1L ||
        !nzchar(model_description)) {
        stop("`model_description` must be a non-empty model description.", call. = FALSE)
    }
    if (is.null(classname)) classname <- .ai4b_derive_class_name(model_description)

    framing <- paste0(
"You are an expert Bayesian statistician + C++ engineer using the AI4BayesCode\n",
"header-only library to generate a stateful, modular MCMC sampler. Follow the\n",
"AI4BayesCode workflow exactly.\n\n")

    if (include_skills) {
        used <- if (is.null(skills)) .ai4b_default_skill_set(backend) else skills
        blocks <- vapply(used, function(s) {
            txt <- .ai4b_skill(s)
            if (!nzchar(txt)) "" else paste0("\n\n===== SKILL: ", s, " =====\n", txt)
        }, "")
        system <- paste0(framing, "Skill corpus (read in order):", paste(blocks, collapse = ""))
    } else {
        used <- "start.md"
        system <- paste0(framing,
            "Entry point (read FIRST; load further skills on demand):\n\n",
            "===== SKILL: start.md =====\n", .ai4b_skill("start.md"))
    }
    list(system = system,
         user = .ai4b_build_user(model_description, backend, output_path, classname,
                                 priors, max_attempts, confirm_model),
         classname = classname, backend = backend, skills = used)
}

# ---------------------------------------------------------------------------
# Offline emit + code extraction
# ---------------------------------------------------------------------------
#' @keywords internal
#' @noRd
.ai4b_offline_emit <- function(prompt, output_path, verbose) {
    dir.create(output_path, showWarnings = FALSE, recursive = TRUE)
    pf <- file.path(output_path, "PROMPT.txt")
    writeLines(c("===== SYSTEM =====", prompt$system, "",
                 "===== USER =====", prompt$user), pf)
    rf <- file.path(output_path, "README_generate.txt")
    writeLines(c(
        "No API key was available, so ai4bayescode_generate() emitted the prompt.",
        "  (A) Claude Code: \"Read AI4BayesCode/start.md first, then <model description>.\"",
        "  (B) Online: set a key once with ai4bayescode_set_key(\"sk-YOUR-KEY-HERE\", \"anthropic\")",
        "      (or pass API_key= / LLM=), then re-run ai4bayescode_generate(...).",
        sprintf("Target class: %s   backend: %s", prompt$classname, prompt$backend)
    ), rf)
    if (verbose) message("No API key -- wrote prompt to:\n  ", pf, "\n  ", rf)
    invisible(list(prompt_path = pf, readme_path = rf, prompt = prompt, called_api = FALSE))
}

#' @keywords internal
#' @noRd
.ai4b_extract_code <- function(text) {
    fences <- gregexpr("(?s)```[A-Za-z]*\\n.*?```", text, perl = TRUE)
    blocks <- regmatches(text, fences)[[1]]
    out <- list()
    for (b in blocks) {
        body <- sub("(?s)^```[A-Za-z]*\\n", "", b, perl = TRUE)
        body <- sub("(?s)```$", "", body, perl = TRUE)
        lines <- strsplit(body, "\n", fixed = TRUE)[[1]]
        ph <- NA_character_; marker_idx <- NA_integer_
        for (i in head(seq_along(lines), 3L)) {
            hl <- lines[i]
            m <- regmatches(hl, regexec("path:\\s*([^[:space:]]+)", hl))[[1]]
            if (length(m) == 2L) {
                ph <- m[2]
                # Strip the routing-marker line so it never leaks into the written
                # file -- ONLY when it is a comment-style directive (`// path:` or
                # `# path:`), never a genuine code line. A `// path:` header is a
                # harmless C++ comment but an R/Python SYNTAX ERROR ("unexpected
                # '/'"), so writing it verbatim crashes any .R/.py runner whose
                # marker used `//` (which the model does slip into under repair).
                if (grepl("^\\s*(//+|#+)\\s*path:", hl)) marker_idx <- i
                break
            }
        }
        if (!is.na(marker_idx)) lines <- lines[-marker_idx]
        out[[length(out) + 1L]] <- list(path = ph, code = paste(lines, collapse = "\n"))
    }
    out
}

# ---------------------------------------------------------------------------
# LLM registry + resolution (LLM-agnostic by design)
# ---------------------------------------------------------------------------
#' Supported LLMs for ai4bayescode_generate()
#'
#' The AI4BayesCode skill corpus is LLM-agnostic. `LLM` is resolved through this
#' registry to a (provider, model id). The Anthropic (Claude) provider is fully
#' implemented; others are an extension point.
#' @return A data.frame with `name`, `provider`, `model_id`, `implemented`, and
#'   `effort_levels` (comma-separated valid effort/reasoning levels for that model;
#'   empty = the model has no effort knob, e.g. Haiku).
#' @export
ai4bayescode_models <- function() {
    data.frame(
        name = c("claude-opus-4-8", "claude-opus-4-7", "claude-sonnet-4-6",
                 "claude-haiku-4-5", "gpt-5.5", "codex"),
        provider = c("anthropic", "anthropic", "anthropic", "anthropic",
                     "openai", "openai"),
        model_id = c("claude-opus-4-8", "claude-opus-4-7", "claude-sonnet-4-6",
                     "claude-haiku-4-5", "gpt-5.5", "gpt-5.5-codex"),
        implemented = c(TRUE, TRUE, TRUE, TRUE, TRUE, TRUE),
        # Valid effort levels differ PER MODEL: xhigh is Opus 4.7+; max not on
        # Sonnet 4.5/Haiku; Haiku 4.5 has no effort knob; OpenAI uses minimal..high.
        effort_levels = c("low,medium,high,xhigh,max", "low,medium,high,xhigh,max",
                          "low,medium,high,max", "",
                          "minimal,low,medium,high", "minimal,low,medium,high"),
        stringsAsFactors = FALSE)
}

# Valid effort levels for a model: NULL = unknown model (cannot validate);
# character(0) = known model with NO effort knob; else the valid levels.
#' @keywords internal
#' @noRd
.ai4b_model_effort_levels <- function(model) {
    m <- ai4bayescode_models()
    hit <- which(m$model_id == model | m$name == model)
    if (!length(hit)) return(NULL)
    lv <- m$effort_levels[hit[1]]
    if (!nzchar(lv)) return(character(0))
    trimws(strsplit(lv, ",", fixed = TRUE)[[1]])
}

#' @keywords internal
#' @noRd
.ai4b_resolve_llm <- function(LLM) {
    m <- ai4bayescode_models()
    key <- tolower(trimws(LLM))
    key <- sub("^claude[ -]", "claude-", key)
    key <- gsub("[ ]+", "-", key)
    key <- sub("-(max|xhigh|high|medium|low|fast)$", "", key)   # strip effort/speed suffix
    cand <- unique(c(key, gsub("\\.", "-", key)))
    for (k in cand) {
        hit <- which(tolower(m$name) == k | tolower(m$model_id) == k)
        if (length(hit)) return(list(provider = m$provider[hit[1]],
                                     model = m$model_id[hit[1]],
                                     implemented = m$implemented[hit[1]]))
    }
    prov <- if (grepl("^(gpt|codex|openai)", key)) "openai"
            else if (grepl("^(gemini|google)", key)) "google"
            else "anthropic"
    list(provider = prov, model = LLM, implemented = (prov %in% c("anthropic", "openai")))
}

#' @keywords internal
#' @noRd
.ai4b_provider_key <- function(provider) {
    if (provider == "anthropic") {
        # API key (pay-per-token) OR a Claude subscription OAuth token (Bearer):
        # ANTHROPIC_API_KEY first, else ANTHROPIC_AUTH_TOKEN (e.g. from
        # `claude setup-token` -- billed to your subscription, not per-token).
        k <- Sys.getenv("ANTHROPIC_API_KEY", unset = "")
        if (!nzchar(k)) k <- Sys.getenv("ANTHROPIC_AUTH_TOKEN", unset = "")
        return(k)
    }
    env <- switch(provider, openai = "OPENAI_API_KEY", google = "GOOGLE_API_KEY",
                  "ANTHROPIC_API_KEY")
    Sys.getenv(env, unset = "")
}

# Mask a key for display: a short prefix + the last 4 chars, never the full
# secret. Used by the key helpers and the verbose billing line.
#' @keywords internal
#' @noRd
.ai4b_mask_key <- function(key) {
    if (is.null(key) || !nzchar(key)) return("<empty>")
    n <- nchar(key)
    if (n <= 10L) return(paste0(substr(key, 1L, 2L), "..."))
    paste0(substr(key, 1L, 6L), "...", substr(key, n - 3L, n))
}

#' Set an LLM provider API key for this session
#'
#' Stores `key` in the provider's environment variable for the CURRENT R
#' session only (via [base::Sys.setenv]). It is **NOT written to disk** and does
#' not persist across sessions; the key is never printed in full. After this,
#' [ai4bayescode_generate()] picks the key up automatically whenever its
#' `API_key` argument is left `NULL` -- you no longer have to pass it on every
#' call (you still can, to override per call).
#'
#' @param key Non-empty API-key string (e.g. `"sk-ant-api..."`, `"sk-YOUR-KEY-HERE"`).
#'   An Anthropic subscription token (`"sk-ant-oat..."` from
#'   `claude setup-token`) works too -- the Bearer header is detected from the
#'   `sk-ant-oat` prefix.
#' @param provider One of `"anthropic"`, `"openai"`, `"google"`.
#' @return Invisibly, the provider name.
#' @seealso [ai4bayescode_key_status()], [ai4bayescode_generate()]
#' @examples
#' \dontrun{
#' ai4bayescode_set_key("sk-ant-api-...", "anthropic")
#' ai4bayescode_set_key("sk-YOUR-KEY-HERE",         "openai")
#' ai4bayescode_generate("Linear regression.", LLM = "gpt-5.5")  # key picked up
#' }
#' @export
ai4bayescode_set_key <- function(key, provider = "anthropic", check = TRUE) {
    provider <- match.arg(provider, c("anthropic", "openai", "google"))
    if (!is.character(key) || length(key) != 1L || is.na(key) || !nzchar(key))
        stop("`key` must be a single non-empty string.", call. = FALSE)
    if (grepl("\\.\\.\\.|[<>]|YOUR.?KEY", key, ignore.case = TRUE))
        stop("'", key, "' is a PLACEHOLDER from the examples, not a real key.\n",
             "  Replace it with YOUR ", provider, " key",
             switch(provider,
                    anthropic = " -- it starts with 'sk-ant-'; get one at https://console.anthropic.com/settings/keys",
                    openai    = " -- get one at https://platform.openai.com/api-keys",
                    google    = " -- get one at https://aistudio.google.com/apikey",
                    ""), ".", call. = FALSE)
    env <- switch(provider, anthropic = "ANTHROPIC_API_KEY",
                  openai = "OPENAI_API_KEY", google = "GOOGLE_API_KEY")
    args <- list(key); names(args) <- env
    do.call(Sys.setenv, args)
    message(sprintf("Set %s key for this session (%s). Not saved to disk.",
                    provider, .ai4b_mask_key(key)))
    if (isTRUE(check) && identical(provider, "anthropic"))
        tryCatch(ai4bayescode_stream_check(API_key = key),
                 error = function(e)
                     message("[warning] streaming self-check failed: ", conditionMessage(e)))
    invisible(provider)
}

#' Report which LLM provider keys are set for this session
#'
#' Prints, for each known provider, whether a key is currently visible to
#' [ai4bayescode_generate()] -- from [ai4bayescode_set_key()] or a pre-existing
#' environment variable -- shown masked. The full key is never printed.
#'
#' @return Invisibly, a named logical vector (`provider -> key present`).
#' @seealso [ai4bayescode_set_key()]
#' @export
ai4bayescode_key_status <- function() {
    provs <- c("anthropic", "openai", "google")
    present <- logical(length(provs)); names(present) <- provs
    for (p in provs) {
        k <- .ai4b_provider_key(p)
        present[[p]] <- nzchar(k)
        message(sprintf("  %-10s %s", p,
                        if (nzchar(k)) paste0("set (", .ai4b_mask_key(k), ")") else "not set"))
    }
    invisible(present)
}

# Auth headers for the Anthropic Messages API. An OAuth token (`sk-ant-oat...`,
# from `claude setup-token`) uses Bearer + the oauth beta header and bills to the
# subscription; an API key (`sk-ant-api...`) uses x-api-key (pay-per-token).
#' @keywords internal
#' @noRd
.ai4b_anthropic_headers <- function(token) {
    if (grepl("^sk-ant-oat", token)) {
        c(Authorization = paste("Bearer", token),
          `anthropic-version` = "2023-06-01",
          `anthropic-beta` = "oauth-2025-04-20",
          `content-type` = "application/json")
    } else {
        c(`x-api-key` = token, `anthropic-version` = "2023-06-01",
          `content-type` = "application/json")
    }
}

# ---------------------------------------------------------------------------
# Console prompting (interactive-if-missing + ask_user rendering)
# ---------------------------------------------------------------------------
#' @keywords internal
#' @noRd
.ai4b_latex_map <- c(
  "\\alpha"="α","\\beta"="β","\\gamma"="γ","\\delta"="δ",
  "\\epsilon"="ε","\\varepsilon"="ε","\\zeta"="ζ","\\eta"="η",
  "\\theta"="θ","\\vartheta"="θ","\\iota"="ι","\\kappa"="κ",
  "\\lambda"="λ","\\mu"="μ","\\nu"="ν","\\xi"="ξ","\\pi"="π",
  "\\rho"="ρ","\\sigma"="σ","\\tau"="τ","\\upsilon"="υ",
  "\\phi"="φ","\\varphi"="φ","\\chi"="χ","\\psi"="ψ","\\omega"="ω",
  "\\Gamma"="Γ","\\Delta"="Δ","\\Theta"="Θ","\\Lambda"="Λ",
  "\\Xi"="Ξ","\\Pi"="Π","\\Sigma"="Σ","\\Phi"="Φ",
  "\\Psi"="Ψ","\\Omega"="Ω",
  "\\sim"="~","\\mid"="|","\\in"="∈","\\notin"="∉",
  "\\leq"="≤","\\le"="≤","\\geq"="≥","\\ge"="≥",
  "\\neq"="≠","\\ne"="≠","\\approx"="≈","\\equiv"="≡",
  "\\propto"="∝","\\times"="×","\\cdot"="·","\\pm"="±",
  "\\to"="→","\\rightarrow"="→","\\Rightarrow"="⇒","\\mapsto"="↦",
  "\\infty"="∞","\\partial"="∂","\\nabla"="∇",
  "\\sum"="Σ","\\prod"="∏","\\int"="∫",
  "\\forall"="∀","\\exists"="∃","\\cup"="∪","\\cap"="∩",
  "\\subseteq"="⊆","\\subset"="⊂","\\langle"="⟨","\\rangle"="⟩",
  "\\ldots"="...","\\cdots"="...","\\dotsc"="...","\\dots"="...",
  "\\lVert"="‖","\\rVert"="‖","\\Vert"="‖","\\lvert"="|","\\rvert"="|",
  "\\quad"="  ","\\qquad"="    ")
.ai4b_latex_sup <- c("^{\\top}"="ᵀ","^\\top"="ᵀ","^{-1}"="⁻¹",
  "^{2}"="²","^{3}"="³","^{T}"="ᵀ","^2"="²","^3"="³","^T"="ᵀ")

# Render the model's LaTeX (display math) as plain text a bare console can show.
# Not a LaTeX engine -- just the stats notation the elicitation prompts use.
# Non-LaTeX text passes through unchanged.
#' @keywords internal
#' @noRd
.ai4b_latex_to_console <- function(s) {
    if (length(s) != 1L || is.na(s)) return(s)
    if (!grepl("\\", s, fixed = TRUE) && !grepl("$", s, fixed = TRUE)) return(s)
    t <- s
    t <- gsub("\\\\(?:begin|end)\\{[^}]*\\}", "", t, perl = TRUE)
    t <- gsub("\\\\\\\\\\[[^][]*\\]", "\n", t, perl = TRUE)  # \\[2pt] line break + spacing
    t <- gsub("$$", "\n", t, fixed = TRUE)
    t <- gsub("\\[", "\n", t, fixed = TRUE)
    t <- gsub("\\]", "\n", t, fixed = TRUE)
    t <- gsub("\\(", "", t, fixed = TRUE); t <- gsub("\\)", "", t, fixed = TRUE)
    t <- gsub("$", "", t, fixed = TRUE)
    t <- gsub("\\\\", "\n", t, fixed = TRUE)        # LaTeX line break
    t <- gsub("&", "", t, fixed = TRUE)             # alignment
    t <- gsub("\\\\(?:left|right|bigg|Bigg|big|Big)\\b", "", t, perl = TRUE)
    t <- gsub("\\\\(?:text|mathrm|mathbf|mathsf|mathtt|operatorname|hat|bar|tilde|widehat|widetilde|vec|boldsymbol|overline|underline)\\{([^{}]*)\\}", "\\1", t, perl = TRUE)
    t <- gsub("\\mathbb{R}", "ℝ", t, fixed = TRUE)
    t <- gsub("\\mathbb{N}", "ℕ", t, fixed = TRUE)
    t <- gsub("\\mathbb{Z}", "ℤ", t, fixed = TRUE)
    t <- gsub("\\mathbb{E}", "E", t, fixed = TRUE)
    t <- gsub("\\\\math(?:cal|bb|frak|scr)\\{([^{}]*)\\}", "\\1", t, perl = TRUE)
    t <- gsub("\\\\frac\\{([^{}]*)\\}\\{([^{}]*)\\}", "(\\1)/(\\2)", t, perl = TRUE)
    t <- gsub("\\\\sqrt\\{([^{}]*)\\}", "sqrt(\\1)", t, perl = TRUE)
    ks <- names(.ai4b_latex_sup); ks <- ks[order(nchar(ks), decreasing = TRUE)]
    for (k in ks) t <- gsub(k, .ai4b_latex_sup[[k]], t, fixed = TRUE)
    t <- gsub("\\top", "ᵀ", t, fixed = TRUE)
    km <- names(.ai4b_latex_map); km <- km[order(nchar(km), decreasing = TRUE)]
    for (k in km) t <- gsub(k, .ai4b_latex_map[[k]], t, fixed = TRUE)
    t <- gsub("\\\\[,;:! ]", " ", t, perl = TRUE)        # thin/medium spaces
    t <- gsub("\\\\([A-Za-z]+)", "\\1", t, perl = TRUE)  # residual \cmd -> word
    t <- gsub("\\", "", t, fixed = TRUE)
    t <- gsub("[{}]", "", t, perl = TRUE)
    ln <- strsplit(t, "\n", fixed = TRUE)[[1]]
    ln <- sub("[ \t]+$", "", gsub("[ \t]+", " ", ln))
    t <- paste(ln, collapse = "\n")
    t <- gsub("\n{3,}", "\n\n", t, perl = TRUE)
    sub("^\n+|\n+$", "", t)
}

#' @keywords internal
#' @noRd
.ai4b_console_ask <- function(prompt, options = NULL, default = NULL) {
    prompt <- .ai4b_latex_to_console(prompt)
    if (!is.null(options) && length(options)) {
        cat(prompt, "\n", sep = "")
        ch <- utils::menu(as.character(options))
        if (ch == 0L) return(if (!is.null(default)) default else options[[1]])
        return(options[[ch]])
    }
    dtxt <- if (!is.null(default) && nzchar(default)) sprintf(" [%s]", default) else ""
    ans <- readline(paste0(prompt, dtxt, ": "))
    if (!nzchar(ans) && !is.null(default)) default else ans
}

# The ask_user tool the model calls to elicit priors / clarify in the console.
#' @keywords internal
#' @noRd
.ai4b_ask_user_tool <- function() {
    list(list(
        name = "ask_user",
        description = paste(
            "Ask the user a clarifying question in their console, especially to ELICIT",
            "the PRIOR for each model parameter. Provide `question`; optionally",
            "`options` (allowed answers, shown as a menu)."),
        input_schema = list(
            type = "object",
            properties = list(
                question = list(type = "string"),
                options  = list(type = "array", items = list(type = "string"))),
            required = list("question"))))
}

# Read-only file tools -- let the model read the installed package's canonical
# reference material ON DEMAND (like Claude Code's Read / Grep / Glob), so the
# skills' "see examples/GaussianLocationScale.cpp" actually works instead of the
# model guessing the API. Restricted to the AI4BayesCode tree (examples/, skills/,
# include/, blocks_local/). Returned alongside ask_user in the tools list.
#' @keywords internal
#' @noRd
.ai4b_agent_tools <- function() {
    c(.ai4b_ask_user_tool(), list(
        list(name = "read_file",
             description = paste(
                 "Read a file from the installed AI4BayesCode package -- its reference",
                 "`examples/*.cpp` (worked, compiling samplers), `skills/*.md`, and",
                 "`include/AI4BayesCode/*.hpp` headers. READ the reference example a skill",
                 "points to (e.g. 'examples/GaussianLocationScale.cpp') BEFORE writing code;",
                 "do not guess an API you can read."),
             input_schema = list(type = "object",
                 properties = list(path = list(type = "string",
                     description = "package-relative, e.g. 'examples/GaussianLocationScale.cpp'")),
                 required = list("path"))),
        list(name = "grep",
             description = paste(
                 "Regex-search file CONTENTS across the installed AI4BayesCode package.",
                 "Returns `path:line: text`. Use to find which example/header uses a",
                 "symbol or block (e.g. pattern='joint_nuts_block', glob='examples/*.cpp')."),
             input_schema = list(type = "object",
                 properties = list(pattern = list(type = "string"),
                                   glob = list(type = "string",
                     description = "restrict search, default 'examples/*.cpp'")),
                 required = list("pattern"))),
        list(name = "glob",
             description = paste(
                 "List files in the installed AI4BayesCode package matching a glob, e.g.",
                 "'examples/*.cpp', 'skills/*.md', 'include/AI4BayesCode/*.hpp'."),
             input_schema = list(type = "object",
                 properties = list(pattern = list(type = "string")),
                 required = list("pattern")))))
}

#' @keywords internal
#' @noRd
.ai4b_pkg_root <- function() {
    r <- tryCatch(system.file(package = "AI4BayesCode"), error = function(e) "")
    if (is.null(r) || !nzchar(r)) NA_character_ else normalizePath(r, mustWork = FALSE)
}

# Whitelist + escape guard for a package-relative path.
#' @keywords internal
#' @noRd
.ai4b_safe_pkg_path <- function(rel, root) {
    rel <- gsub("^[./]+", "", rel %||% "")
    if (grepl("\\.\\.", rel) || !nzchar(rel)) return(NULL)
    if (!grepl("^(examples|skills|include|blocks_local)(/|$)", rel)) return(NULL)
    p <- normalizePath(file.path(root, rel), mustWork = FALSE)
    if (!startsWith(p, root)) return(NULL)
    p
}

# Glob within the whitelisted package dirs (supports a single ** for recursion).
#' @keywords internal
#' @noRd
.ai4b_glob_pkg <- function(pattern, root) {
    pattern <- gsub("^[./]+", "", pattern %||% "")
    if (grepl("\\.\\.", pattern) ||
        !grepl("^(examples|skills|include|blocks_local)(/|$)", pattern)) return(character(0))
    if (grepl("\\*\\*", pattern)) {
        base <- sub("/?\\*\\*.*$", "", pattern)
        ext  <- sub("^.*\\*\\*/?", "", pattern)
        list.files(file.path(root, base),
                   pattern = utils::glob2rx(if (nzchar(ext)) ext else "*"),
                   recursive = TRUE, full.names = TRUE)
    } else {
        Sys.glob(file.path(root, pattern))
    }
}

# Execute one read_file / grep / glob tool_use; returns the tool_result string.
#' @keywords internal
#' @noRd
.ai4b_exec_readonly_tool <- function(name, input, root) {
    if (is.na(root))
        return("(AI4BayesCode package not installed; read_file/grep/glob unavailable.)")
    rel <- function(f) sub(paste0("^", root, "/?"), "", f)
    if (identical(name, "read_file")) {
        p <- .ai4b_safe_pkg_path(input$path, root)
        if (is.null(p) || !file.exists(p))
            return(sprintf("read_file: '%s' not found or not allowed (only examples/, skills/, include/, blocks_local/).",
                           input$path %||% ""))
        txt <- tryCatch(readLines(p, warn = FALSE), error = function(e) NULL)
        if (is.null(txt)) return(sprintf("read_file: could not read '%s'.", input$path))
        if (length(txt) > 1400L)
            txt <- c(txt[1:1400], sprintf("... [truncated; %d lines total -- grep for a specific part]",
                                          length(txt)))
        paste(txt, collapse = "\n")
    } else if (identical(name, "grep")) {
        pat <- input$pattern %||% ""
        g   <- input$glob %||% "examples/*.cpp"
        hits <- character(0)
        for (f in .ai4b_glob_pkg(g, root)) {
            ls <- tryCatch(readLines(f, warn = FALSE), error = function(e) character(0))
            m  <- tryCatch(grep(pat, ls, perl = TRUE), error = function(e) integer(0))
            for (k in m) {
                hits <- c(hits, sprintf("%s:%d: %s", rel(f), k, ls[k]))
                if (length(hits) >= 200L) break
            }
            if (length(hits) >= 200L) break
        }
        if (!length(hits)) sprintf("grep: no match for /%s/ in %s", pat, g)
        else paste(c(hits, if (length(hits) >= 200L) "... [capped at 200 hits]"), collapse = "\n")
    } else if (identical(name, "glob")) {
        fs <- vapply(.ai4b_glob_pkg(input$pattern, root), rel, "")
        if (!length(fs)) sprintf("glob: no files match '%s'", input$pattern %||% "")
        else paste(fs, collapse = "\n")
    } else {
        sprintf("(unknown tool '%s')", name)
    }
}

# Flush one finished streamed text block to the console. In the per-block
# (`live = FALSE`) path, prose / model-confirmations are rendered through the
# LaTeX->console map so `$$ ... $$` math is readable instead of raw source;
# code-bearing blocks (``` fences / #include / module macros) print verbatim so
# C++/R source is never mangled by the math substitutions. Idempotent via a
# per-block `flushed` flag; a no-op when `live = TRUE` (already streamed raw).
#' @keywords internal
#' @noRd
.ai4b_flush_text_block <- function(state, i, progress, live = FALSE) {
    if (isTRUE(live)) return(state)
    b <- state$blocks[[i]]
    if (is.null(b) || !identical(b$type, "text") || isTRUE(b$flushed)) return(state)
    state$blocks[[i]]$flushed <- TRUE
    txt <- b$text %||% ""
    if (progress && nzchar(txt)) {
        if (grepl("```", txt, fixed = TRUE)) {
            # Mixed prose + fenced code: render the PROSE (so $$...$$ math is readable)
            # but keep CODE verbatim (never mangle C++/R). Split on ``` -> odd segments
            # are prose, even segments are code; re-wrap code in its fences.
            parts <- strsplit(txt, "```", fixed = TRUE)[[1]]
            cat("\n")
            for (k in seq_along(parts)) {
                if (k %% 2L == 1L) cat(.ai4b_latex_to_console(parts[k]))   # prose
                else               cat("```", parts[k], "```", sep = "")   # code, verbatim
            }
            cat("\n")
        } else {
            cat("\n", .ai4b_latex_to_console(txt), "\n", sep = "")          # pure prose
        }
    }
    state
}

# ---------------------------------------------------------------------------
# Anthropic provider: one request + the agentic ask_user loop
# ---------------------------------------------------------------------------
# Fold ONE parsed Anthropic streaming event into `state` (content blocks +
# stop_reason). Pure except optional live-progress printing. Unit-testable.
#' @keywords internal
#' @noRd
.ai4b_sse_step <- function(state, ev, progress = TRUE, live = FALSE) {
    type <- ev$type
    if (identical(type, "content_block_start")) {
        i <- ev$index + 1L
        cb <- ev$content_block
        state$blocks[[i]] <- list(type = cb$type %||% "text",
                                  text = cb$text %||% "", thinking = cb$thinking %||% "",
                                  signature = cb$signature %||% "", data = cb$data %||% "",
                                  id = cb$id, name = cb$name, partial = "")
        if (progress && identical(cb$type, "thinking")) cat("\n  [thinking] ")
        if (progress && identical(cb$type, "text"))     cat("\n")
        if (progress && identical(cb$type, "tool_use")) cat("\n  [preparing a question] ")
    } else if (identical(type, "content_block_delta")) {
        i <- ev$index + 1L
        d <- ev$delta
        if (identical(d$type, "text_delta")) {
            state$blocks[[i]]$text <- paste0(state$blocks[[i]]$text %||% "", d$text)
            if (progress && live) {
                cat(d$text)                                 # raw token stream (stream_check)
            } else if (progress) {                          # buffer; render at block stop
                state$text_chars <- (state$text_chars %||% 0L) + nchar(d$text %||% "")
                if (state$text_chars >= (state$next_text_dot %||% 0L)) {
                    cat("."); state$next_text_dot <- state$text_chars + 600L
                }
            }
        } else if (identical(d$type, "thinking_delta")) {
            state$blocks[[i]]$thinking <- paste0(state$blocks[[i]]$thinking %||% "", d$thinking)
            state$think_chars <- (state$think_chars %||% 0L) + nchar(d$thinking)
            if (progress && state$think_chars >= (state$next_dot %||% 0L)) {
                cat("."); state$next_dot <- state$think_chars + 400L   # heartbeat
            }
        } else if (identical(d$type, "signature_delta")) {
            # REQUIRED: the thinking block's cryptographic signature. Anthropic
            # rejects (HTTP 400) a follow-up turn whose tool_use-bearing assistant
            # message carries a thinking block stripped of its signature, so it
            # MUST be captured here and re-emitted by .ai4b_sse_finalize().
            state$blocks[[i]]$signature <- paste0(state$blocks[[i]]$signature %||% "", d$signature %||% "")
        } else if (identical(d$type, "input_json_delta")) {
            state$blocks[[i]]$partial <- paste0(state$blocks[[i]]$partial %||% "", d$partial_json)
        }
    } else if (identical(type, "message_delta")) {
        if (!is.null(ev$delta$stop_reason)) state$stop_reason <- ev$delta$stop_reason
    } else if (identical(type, "content_block_stop")) {
        state <- .ai4b_flush_text_block(state, ev$index + 1L, progress, live)
    }
    state
}

# Convert accumulated streaming state into the SAME list(content, stop_reason)
# shape the buffered path returns, so the agent loop is unchanged.
#' @keywords internal
#' @noRd
.ai4b_sse_finalize <- function(state) {
    content <- Filter(Negate(is.null), lapply(state$blocks, function(b) {
        if (is.null(b)) return(NULL)
        if (identical(b$type, "text"))     return(list(type = "text", text = b$text %||% ""))
        if (identical(b$type, "thinking")) {
            tb <- list(type = "thinking", thinking = b$thinking %||% "")
            # Carry the signature so a tool_use turn survives the next request.
            if (nzchar(b$signature %||% "")) tb$signature <- b$signature
            return(tb)
        }
        if (identical(b$type, "redacted_thinking"))
            return(list(type = "redacted_thinking", data = b$data %||% ""))
        if (identical(b$type, "tool_use")) {
            pj  <- if (nzchar(b$partial %||% "")) b$partial else "{}"
            inp <- tryCatch(jsonlite::fromJSON(pj, simplifyVector = FALSE),
                            error = function(e) list())
            return(list(type = "tool_use", id = b$id, name = b$name, input = inp))
        }
        NULL
    }))
    list(content = content, stop_reason = state$stop_reason %||% "end_turn")
}

#' @keywords internal
#' @noRd
.ai4b_anthropic_request <- function(system, messages, model, effort, tools,
                                    api_key, max_tokens, timeout,
                                    stream = TRUE, progress = TRUE, live = FALSE) {
    # Anthropic REQUIRES max_tokens. NULL = "no cap I chose" -> send the model's
    # maximum (64000 for Claude 4.x = effectively unlimited).
    body <- list(
        model = model,
        max_tokens = if (is.null(max_tokens) || is.na(max_tokens)) 64000L else as.integer(max_tokens),
        thinking = list(type = "adaptive"),
        system = list(list(type = "text", text = system,
                           cache_control = list(type = "ephemeral"))),
        messages = messages)
    if (!is.null(effort) && !is.na(effort) && nzchar(effort))
        body$output_config <- list(effort = effort)   # omitted for no-effort models
    if (!is.null(tools)) body$tools <- tools
    build_req <- function(streaming) {
        b <- body; if (streaming) b$stream <- TRUE
        req <- httr2::request("https://api.anthropic.com/v1/messages")
        req <- httr2::req_headers(req, !!!as.list(.ai4b_anthropic_headers(api_key)))
        req <- httr2::req_body_json(req, b, auto_unbox = TRUE)
        # Streaming keeps the socket alive (token + ping events), so a generous
        # cap instead of the buffered "0 bytes for `timeout`s" death that killed
        # long effort=max generations.
        req <- httr2::req_timeout(req, if (streaming) max(timeout, 1800) else timeout)
        # Surface the API's actual error message on 4xx/5xx instead of an opaque
        # "HTTP 400 Bad Request" (e.g. a malformed message or an invalid param).
        req <- httr2::req_error(req, body = function(resp) {
            b <- tryCatch(httr2::resp_body_json(resp), error = function(...) NULL)
            msg <- b$error$message %||% b$message
            if (!is.null(msg)) paste0("anthropic: ", msg) else NULL
        })
        httr2::req_retry(req, max_tries = 4,
            is_transient = function(r) httr2::resp_status(r) %in% c(429, 500, 529))
    }
    if (!isTRUE(stream))
        return(httr2::resp_body_json(httr2::req_perform(build_req(FALSE))))
    # Wait hint: the model can sit silent for many seconds before the first
    # token (it is thinking/generating server-side, longer at higher effort).
    # Fill that gap so the console does not look frozen. Skipped for the live
    # stream-check (which prints its own banner).
    if (progress && !isTRUE(live)) {
        eff <- if (!is.null(effort) && !is.na(effort) && nzchar(effort))
                   sprintf(" (effort=%s)", effort) else ""
        cat(sprintf(
            "\n  [contacting %s%s -- the model is working; this can take a while, please wait ...]\n",
            model, eff))
    }
    tryCatch({
        conn <- httr2::req_perform_connection(build_req(TRUE))
        on.exit(try(close(conn), silent = TRUE), add = TRUE)
        state <- list(blocks = list(), stop_reason = NULL, think_chars = 0L, next_dot = 0L)
        repeat {
            ev <- httr2::resp_stream_sse(conn)
            if (is.null(ev)) break                              # connection closed
            if (is.null(ev$data) || !nzchar(ev$data)) next      # ping / keep-alive
            pd <- tryCatch(jsonlite::fromJSON(ev$data, simplifyVector = FALSE),
                           error = function(e) NULL)
            if (is.null(pd)) next
            if (identical(pd$type, "error"))
                stop(pd$error$message %||% "anthropic stream error", call. = FALSE)
            if (identical(pd$type, "message_stop")) break
            state <- .ai4b_sse_step(state, pd, progress, live)
        }
        # Safety: flush any text block that never received an explicit
        # content_block_stop event (per-block render in the !live path).
        for (j in seq_along(state$blocks))
            state <- .ai4b_flush_text_block(state, j, progress, live)
        if (progress) cat("\n")
        .ai4b_sse_finalize(state)
    }, error = function(e) {
        if (progress) message("[stream interrupted: ", conditionMessage(e),
                              " -- retrying without streaming]")
        httr2::resp_body_json(httr2::req_perform(build_req(FALSE)))
    })
}

#' Quick streaming self-check
#'
#' Sends a tiny one-sentence prompt to the configured LLM with streaming ON and
#' prints the reply token-by-token, so you can confirm the streaming transport
#' works before paying for a full [ai4bayescode_generate()] run. Minimal cost.
#'
#' @param LLM Model id (default `"claude-opus-4-8"`); see [ai4bayescode_models()].
#' @param API_key Optional; defaults to the session key from [ai4bayescode_set_key()].
#' @param effort Optional thinking level; default `NULL` (fast + cheap).
#' @return Invisibly, the parsed `list(content, stop_reason)`.
#' @export
ai4bayescode_stream_check <- function(LLM = "claude-opus-4-8", API_key = NULL,
                                      effort = NULL) {
    if (!requireNamespace("httr2", quietly = TRUE))
        stop("'httr2' is required for the streaming check.", call. = FALSE)
    llm <- .ai4b_resolve_llm(LLM)
    if (!identical(llm$provider, "anthropic"))
        stop("ai4bayescode_stream_check verifies the Anthropic streaming path; '",
             llm$provider, "' is not supported here.", call. = FALSE)
    key <- if (!is.null(API_key) && nzchar(API_key)) API_key
           else .ai4b_provider_key(llm$provider)
    if (is.null(key) || !nzchar(key))
        stop("No API key. Pass API_key= or call ai4bayescode_set_key().", call. = FALSE)
    message(sprintf("Streaming check -> %s. Reply should appear token-by-token:", llm$model))
    sys  <- "You are a connectivity test. Answer in ONE short sentence."
    msgs <- list(list(role = "user",
                      content = "In one short sentence, confirm you are streaming this reply."))
    t0 <- Sys.time()
    parsed <- .ai4b_anthropic_request(sys, msgs, llm$model, effort, tools = NULL,
                                      api_key = key, max_tokens = 128L, timeout = 60,
                                      stream = TRUE, progress = TRUE, live = TRUE)
    txt <- paste(vapply(Filter(function(b) identical(b$type, "text"), parsed$content),
                        function(b) b$text, ""), collapse = "")
    message(sprintf("[streaming OK] %d chars, stop=%s, %.1fs",
                    nchar(txt), parsed$stop_reason %||% "?",
                    as.numeric(difftime(Sys.time(), t0, units = "secs"))))
    invisible(parsed)
}

# ---------------------------------------------------------------------------
# OpenAI provider: Anthropic<->OpenAI shape converters + one request
# ---------------------------------------------------------------------------
# Convert an Anthropic-shaped conversation (the loop's internal representation)
# to the OpenAI Chat Completions `messages` array. The system prompt is prepended
# as a {role:"system"} message. Plain user/assistant strings pass through. An
# assistant turn carrying a tool_use block becomes {role:"assistant", tool_calls}
# and a user turn carrying a tool_result block becomes {role:"tool"}.
#' @keywords internal
#' @noRd
.ai4b_to_openai_msgs <- function(messages, system) {
    out <- list(list(role = "system", content = system))
    for (m in messages) {
        ct <- m$content
        if (is.character(ct)) {                       # plain string passes through
            out[[length(out) + 1L]] <- list(role = m$role, content = ct)
            next
        }
        # content is a list of typed blocks. Detect tool_use / tool_result.
        tu <- Filter(function(b) identical(b$type, "tool_use"), ct)
        tr <- Filter(function(b) identical(b$type, "tool_result"), ct)
        if (length(tr)) {                             # user turn -> role:"tool" message(s)
            for (b in tr) {
                txt <- b$content
                if (!is.character(txt)) txt <- paste(vapply(txt, function(x)
                    if (is.list(x) && !is.null(x$text)) x$text else as.character(x), ""),
                    collapse = "\n")
                out[[length(out) + 1L]] <- list(role = "tool",
                                                tool_call_id = b$tool_use_id,
                                                content = txt)
            }
            next
        }
        # gather any text blocks into a single content string
        text <- paste(vapply(ct, function(b)
            if (identical(b$type, "text")) b$text else "", ""), collapse = "")
        if (length(tu)) {                             # assistant turn with tool_calls
            tool_calls <- lapply(tu, function(b) list(
                id = b$id, type = "function",
                `function` = list(
                    name = b$name,
                    arguments = jsonlite::toJSON(b$input, auto_unbox = TRUE))))
            out[[length(out) + 1L]] <- list(role = m$role,
                                            content = text,   # text or "" alongside calls
                                            tool_calls = tool_calls)
        } else {                                      # plain typed-text turn
            out[[length(out) + 1L]] <- list(role = m$role, content = text)
        }
    }
    out
}

# Convert an OpenAI Chat Completions response (parsed JSON) back to the
# Anthropic-shaped `list(stop_reason, content)` the agentic loop expects. A
# non-empty assistant text -> a {type:"text"} block; each tool_call -> a
# {type:"tool_use"} block; stop_reason is "tool_use" iff there are tool_calls.
#' @keywords internal
#' @noRd
.ai4b_from_openai_response <- function(resp) {
    msg <- resp$choices[[1]]$message
    content <- list()
    if (!is.null(msg$content) && is.character(msg$content) && nzchar(msg$content))
        content[[length(content) + 1L]] <- list(type = "text", text = msg$content)
    tcs <- msg$tool_calls
    if (!is.null(tcs) && length(tcs)) {
        for (tc in tcs) {
            args <- tc[["function"]]$arguments
            input <- if (is.character(args) && nzchar(args))
                jsonlite::fromJSON(args, simplifyVector = FALSE) else list()
            content[[length(content) + 1L]] <- list(
                type = "tool_use", id = tc$id,
                name = tc[["function"]]$name, input = input)
        }
    }
    stop_reason <- if (!is.null(tcs) && length(tcs)) "tool_use" else "end_turn"
    list(stop_reason = stop_reason, content = content)
}

# One request to the OpenAI Chat Completions API. ALL Anthropic-shaped tools
# (ask_user, read_file, grep, glob) are converted generically to the OpenAI
# {type:"function", function:{...}} shape, so the agent's file-reading tools work
# through this path too; `reasoning_effort` is sent for reasoning models when
# `effort` is a non-empty string. Mirrors `.ai4b_anthropic_request`'s timeout +
# transient retry. Returns the Anthropic-shaped parsed response.
#' @keywords internal
#' @noRd
.ai4b_openai_request <- function(system, messages, model, effort, tools,
                                 api_key, max_tokens, timeout) {
    openai_tools <- if (!is.null(tools)) lapply(tools, function(t) list(
        type = "function",
        `function` = list(name = t$name, description = t$description,
                          parameters = t$input_schema))) else NULL
    body <- list(
        model = model,
        messages = .ai4b_to_openai_msgs(messages, system))
    # NULL max_tokens -> omit the cap so the model uses its full output budget.
    if (!is.null(max_tokens) && !is.na(max_tokens))
        body$max_completion_tokens <- as.integer(max_tokens)
    if (!is.null(openai_tools)) {
        body$tools <- openai_tools
        body$tool_choice <- "auto"
    }
    if (!is.null(effort) && !is.na(effort) && nzchar(effort))
        body$reasoning_effort <- effort
    req <- httr2::request("https://api.openai.com/v1/chat/completions")
    req <- httr2::req_headers(req,
        Authorization = paste("Bearer", api_key),
        `content-type` = "application/json")
    req <- httr2::req_body_json(req, body, auto_unbox = TRUE)
    req <- httr2::req_timeout(req, timeout)
    req <- httr2::req_retry(req, max_tries = 4,
        is_transient = function(r) httr2::resp_status(r) %in% c(429, 500, 503))
    .ai4b_from_openai_response(httr2::resp_body_json(httr2::req_perform(req)))
}

# `claude -p` CLI transport: render system + conversation as a single prompt,
# shell out to the `claude` CLI, and wrap stdout as a parsed Messages-shaped
# response. Used when no API key is available but the CLI is on PATH (or when the
# caller forces it via use_cli = TRUE). The CLI does not support the ask_user
# tool round-trip, so the returned response is always a terminal end_turn.
#' @keywords internal
#' @noRd
.ai4b_claude_p_call <- function(system, msgs, model) {
    rendered <- vapply(msgs, function(m) {
        ct <- m$content
        if (!is.character(ct)) ct <- paste(vapply(ct, function(b)
            if (is.list(b) && !is.null(b$text)) b$text else "", ""), collapse = "\n")
        paste0(m$role, ": ", ct)
    }, "")
    prompt_text <- paste0(system, "\n\n", paste(rendered, collapse = "\n\n"))
    args <- c("-p", if (!is.null(model) && !is.na(model)) c("--model", model))
    out <- system2("claude", args, input = prompt_text, stdout = TRUE, stderr = TRUE)
    list(stop_reason = "end_turn",
         content = list(list(type = "text", text = paste(out, collapse = "\n"))))
}

# One agentic turn: drive the ask_user sub-loop from `msgs` until the model emits
# terminal text (no tool_use). `call` is the transport `function(messages) ->
# parsed`. Returns list(text = <final assistant text>, msgs = <msgs incl this
# turn's assistant/tool exchanges>) so the caller can append a repair message and
# call again for the next attempt.
#' @keywords internal
#' @noRd
.ai4b_anthropic_turn <- function(system, msgs, model, effort, api_key, max_tokens,
                                 timeout, ask, verbose, call, max_subturns = 30L) {
    for (i in seq_len(max_subturns)) {
        parsed  <- call(msgs)
        content <- parsed$content
        tu <- Filter(function(b) identical(b$type, "tool_use"), content)
        if (!length(tu)) {
            txt <- paste(vapply(content, function(b)
                if (identical(b$type, "text")) b$text else "", ""), collapse = "")
            return(list(text = txt, msgs = msgs,
                        truncated = identical(parsed$stop_reason, "max_tokens")))
        }
        root <- .ai4b_pkg_root()
        results <- lapply(tu, function(t) {
            if (identical(t$name, "ask_user")) {
                q <- t$input$question
                opts <- if (!is.null(t$input$options)) unlist(t$input$options) else NULL
                if (verbose) message("[model asks]")
                list(type = "tool_result", tool_use_id = t$id, content = ask(q, options = opts))
            } else {
                if (verbose) message("  [model ", t$name, ": ",
                                     t$input$path %||% t$input$pattern %||% "", "]")
                list(type = "tool_result", tool_use_id = t$id,
                     content = .ai4b_exec_readonly_tool(t$name, t$input, root))
            }
        })
        msgs <- c(msgs, list(list(role = "assistant", content = content),
                             list(role = "user", content = results)))
    }
    stop("ai4bayescode_generate: exceeded ", max_subturns, " conversation turns.", call. = FALSE)
}

# Factor the emitted-text -> files step out of the generate loop so each attempt
# can rewrite files and locate the runner. Returns list(files, cpp_path,
# runner_path); runner_path = first emitted file matching `_runner\.R$`, else
# `\.R$`, else NA.
#' @keywords internal
#' @noRd
.ai4b_write_emitted <- function(txt, output_path, classname) {
    dir.create(output_path, showWarnings = FALSE, recursive = TRUE)
    files <- character(0); cpp_path <- NA_character_
    for (blk in .ai4b_extract_code(txt)) {
        path <- blk$path
        if (is.na(path))
            path <- if (grepl("RCPP_MODULE|PYBIND11_MODULE|#include", blk$code))
                file.path(output_path, paste0(classname, ".cpp")) else NA
        if (is.na(path)) next
        if (!grepl("^(/|\\.)", path)) path <- file.path(output_path, basename(path))
        dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)
        writeLines(blk$code, path)
        files <- c(files, path)
        if (grepl("\\.cpp$", path)) cpp_path <- path
    }
    runner_path <- {
        r <- files[grepl("_runner\\.R$", files)]
        if (!length(r)) r <- files[grepl("\\.R$", files)]
        if (length(r)) r[1] else NA_character_
    }
    list(files = files, cpp_path = cpp_path, runner_path = runner_path)
}

# ---------------------------------------------------------------------------
# ai4bayescode_generate(): NL -> code, console-interactive
# ---------------------------------------------------------------------------
#' Generate a validated AI4BayesCode sampler from a natural-language description
#'
#' LLM-agnostic NL -> code front door. With `interactive = TRUE` (default in an
#' interactive session), any NULL/empty argument is asked in the console, and the
#' PRIOR for each model parameter is elicited interactively (the LLM asks via an
#' `ask_user` tool). With a provider API key it drives the LLM (Claude today),
#' writes the `.cpp` (+ runner) to `output_path`, then runs a MANDATORY
#' validate -> repair-to-convergence loop: each attempt is compiled via
#' [ai4bayescode_source()] and its runner executed; the runner must print
#' `AI4BAYES_VALIDATE: PASS` (max rank-normalized R-hat < 1.01). On any failure the
#' compile / run output is fed back to the model for repair, up to `max_attempts`
#' times. With no key it writes the prompt to `PROMPT.txt`.
#'
#' @section Billing & security:
#' A key bills pay-per-token; it is read from the provider env var (e.g.
#' `ANTHROPIC_API_KEY`), sent only to the provider over TLS, and never logged.
#'
#' @param model_description Model description (string or `.txt`/`.md` path). `NULL`
#'   (default) -> asked interactively.
#' @param classname C++ class name. `NULL` (default) -> derived / asked.
#' @param LLM Friendly model name resolved via [ai4bayescode_models()] (default
#'   `"claude-opus-4-8"`). LLM-agnostic; Claude is the implemented provider.
#' @param effort Reasoning/effort level (default `"high"`; low yields lower-quality
#'   samplers). Validated against the chosen model's own valid levels (see
#'   [ai4bayescode_models()] -- e.g. Opus has `xhigh`/`max`, Haiku has no effort knob,
#'   OpenAI uses `minimal..high`). If `effort` is not a valid level for the model,
#'   you are asked to pick from that model's levels (interactive) or it errors.
#' @param output_path Output folder (default `"./generated"`).
#' @param backend `"R"`, `"Python"`, `"both"`. `NULL` (default) -> asked / `"R"`.
#' @param API_key Provider API key. `NULL` (default) -> provider env var, then
#'   asked interactively, else offline prompt emit.
#' @param interactive If `TRUE` (default = [interactive()]), ask for missing
#'   arguments and elicit priors in the console.
#' @param use_cli If `TRUE`, drive generation through the local `claude` CLI
#'   (`claude -p`) instead of the Messages API. Default `FALSE`; the CLI is also
#'   used automatically when no API key is available but `claude` is on the PATH.
#' @param max_attempts Max validate -> repair iterations before an honest failure
#'   (default 5L). Lower it (e.g. 2L) to cap a costly run.
#' @param priors `NULL` (default -> `"interactive"` when interactive, else
#'   `"noninformative"`), `"noninformative"`, `"weakly"`, or a named list.
#' @param confirm_model `NULL` (default -> follows `interactive`). When `TRUE`, the
#'   model must state its understanding (likelihood, parameters, priors) via `ask_user`
#'   and get your approval BEFORE it writes any code; `FALSE` goes straight to code.
#' @param max_tokens Max output tokens per model reply. `NULL` (default) imposes
#'   NO cap you chose -- the model produces up to its maximum (OpenAI: the cap is
#'   omitted entirely; Anthropic requires a number, so the model's max -- 64000 for
#'   Claude 4.x -- is sent). Pass an integer to cap it. Samplers can be long and
#'   hard to size up front, so the default avoids an arbitrary limit; if a reply is
#'   still truncated the generator re-asks for the code for FREE (no attempt spent).
#' @param timeout Request timeout, seconds (default 600).
#' @param verbose Print progress (default `TRUE`). Never prints the key.
#' @param ... Advanced/testing hooks: `.responder` (an injectable transport
#'   `function(messages)` returning a parsed response, used to exercise the loop
#'   offline), `.ask` (a console-prompt function override), and `.validate` (an
#'   injectable `function(cpp_path, runner_path, classname, verbose)` returning
#'   `list(ok, stage, detail)`, used to exercise the validate -> repair loop
#'   offline without real compilation).
#' @return Invisibly: `cpp_path`, `files`, `prompt`, `called_api`, `transcript`,
#'   `validated` (did the final attempt pass), `attempts` (number used), and
#'   `validation` (the final `list(ok, stage, detail)`).
#' @seealso [ai4bayescode_prompt()], [ai4bayescode_models()], [ai4bayescode_source()].
#' @export
ai4bayescode_generate <- function(model_description = NULL,
                              classname    = NULL,
                              LLM          = "claude-opus-4-8",
                              effort       = "high",
                              output_path  = "./generated",
                              backend      = NULL,
                              API_key      = NULL,
                              interactive  = base::interactive(),
                              use_cli      = FALSE,
                              max_attempts = 5L,
                              priors       = NULL,
                              confirm_model = NULL,
                              max_tokens   = NULL,
                              timeout      = 600,
                              verbose      = TRUE,
                              verify_stream = TRUE,
                              ...) {
    dots   <- list(...)
    ask    <- if (!is.null(dots$.ask)) dots$.ask else .ai4b_console_ask
    responder <- dots$.responder    # injectable transport for offline testing

    llm <- .ai4b_resolve_llm(LLM)

    # ---- interactive-if-missing fill ----
    if (interactive) {
        if (is.null(model_description) || !nzchar(model_description))
            model_description <- ask("Model description (text, or path to a .txt)")
        # Pick the LLM model, then its thinking/effort level -- kept consistent
        # with the per-model effort check below (only that model's levels offered).
        LLM <- ask("LLM model?", options = ai4bayescode_models()$name, default = LLM)
        llm <- .ai4b_resolve_llm(LLM)
        lv  <- .ai4b_model_effort_levels(llm$model)
        if (length(lv))
            effort <- ask("Thinking / effort level?", options = lv,
                          default = if (!is.null(effort) && !is.na(effort) && effort %in% lv) effort
                                    else if ("high" %in% lv) "high" else lv[length(lv)])
        if (is.null(backend))
            backend <- ask("Backend?", options = c("R", "Python", "both"), default = "R")
        if (is.null(output_path) || !nzchar(output_path))
            output_path <- ask("Output folder", default = "./generated")
        if (is.null(classname) || !nzchar(classname)) {
            dflt <- .ai4b_derive_class_name(model_description %||% "GeneratedModel")
            classname <- ask("Class name", default = dflt)
        }
    } else {
        if (is.null(model_description) || !nzchar(model_description))
            stop("`model_description` is required when interactive = FALSE.", call. = FALSE)
        if (is.null(backend)) backend <- "R"
        if (is.null(output_path)) output_path <- "./generated"
        if (is.null(classname)) classname <- .ai4b_derive_class_name(model_description)
    }
    backend <- match.arg(.ai4b_norm_backend(backend), c("R", "Python", "both"))
    if (is.null(priors)) priors <- if (interactive) "interactive" else "noninformative"
    if (is.null(confirm_model)) confirm_model <- isTRUE(interactive)

    # ---- effort: match the chosen model's valid levels; ask if no match ----
    lv <- .ai4b_model_effort_levels(llm$model)
    if (is.null(lv)) {
        # unknown model -> cannot validate; pass `effort` through as-is.
    } else if (!length(lv)) {
        effort <- NA_character_                       # model has no effort knob
    } else if (is.null(effort) || !(effort %in% lv)) {
        if (interactive) {
            if (!is.null(effort) && nzchar(effort))
                message("effort '", effort, "' is not a valid level for ", llm$model, ".")
            dflt <- if ("high" %in% lv) "high" else lv[length(lv)]
            effort <- ask(sprintf("Effort / reasoning level for %s?", llm$model),
                          options = lv, default = dflt)
        } else {
            stop("effort '", effort, "' is not valid for ", llm$model,
                 "; valid levels: ", paste(lv, collapse = ", "), ".", call. = FALSE)
        }
    }

    # ---- API key resolution ----
    if (is.null(API_key)) API_key <- .ai4b_provider_key(llm$provider)
    if (!nzchar(API_key) && interactive && is.null(responder)) {
        API_key <- ask(sprintf(paste0("%s API key (Enter to skip -> writes offline PROMPT.txt; ",
                                       "or set once via ai4bayescode_set_key(); input is echoed)"),
                               llm$provider), default = "")
    }
    # CLI (claude -p) is OPT-IN only, via use_cli = TRUE. A blank key no longer
    # silently routes to the local CLI -- it lands on the predictable offline
    # PROMPT.txt path. Use ai4bayescode_set_key() or API_key = to generate online.
    cli_available <- isTRUE(use_cli) && nzchar(Sys.which("claude"))
    if (isTRUE(use_cli) && !nzchar(Sys.which("claude")))
        warning("use_cli = TRUE but `claude` is not on the PATH; ",
                "using the API key / offline path instead.", call. = FALSE)
    online <- nzchar(API_key) || !is.null(responder) || cli_available

    # ---- build prompt; online path inlines the skill subset ----
    prompt <- ai4bayescode_prompt(model_description, backend = backend,
                              output_path = output_path, classname = classname,
                              priors = priors, max_attempts = max_attempts,
                              include_skills = online, confirm_model = confirm_model)

    if (!online) return(.ai4b_offline_emit(prompt, output_path, verbose))

    # ---- provider dispatch (LLM-agnostic: anthropic + openai implemented) ----
    if (!llm$provider %in% c("anthropic", "openai")) {
        stop("LLM provider '", llm$provider, "' is not yet implemented. Anthropic (Claude) ",
             "and OpenAI are implemented; see ai4bayescode_models(). The design is ",
             "LLM-agnostic -- add a provider by implementing its request function.",
             call. = FALSE)
    }

    # ---- transport `call`: function(messages) -> parsed (Anthropic-shaped) ----
    # `.responder` overrides the transport for BOTH providers (offline testing).
    # The CLI transport is Anthropic-only (Claude CLI); OpenAI uses the API only.
    use_cli_transport <- is.null(responder) && cli_available &&
                         identical(llm$provider, "anthropic")
    if (!is.null(responder)) {
        call <- responder
    } else if (use_cli_transport) {
        call <- function(messages) .ai4b_claude_p_call(prompt$system, messages, llm$model)
    } else {
        if (!requireNamespace("httr2", quietly = TRUE)) {
            warning("'httr2' not installed; emitting the prompt offline instead.",
                    call. = FALSE)
            return(.ai4b_offline_emit(prompt, output_path, verbose))
        }
        tools <- .ai4b_agent_tools()
        if (identical(llm$provider, "openai")) {
            call <- function(messages)
                .ai4b_openai_request(prompt$system, messages, llm$model, effort, tools,
                                     API_key, max_tokens, timeout)
        } else {
            call <- function(messages)
                .ai4b_anthropic_request(prompt$system, messages, llm$model, effort, tools,
                                        API_key, max_tokens, timeout)
        }
    }
    validate_fn <- dots$.validate %||% .ai4b_validate

    if (verbose) {
        bill <- if (use_cli_transport) "via the local `claude` CLI"
        else if (identical(llm$provider, "openai")) "billed pay-per-token (OpenAI API key)"
        else if (grepl("^sk-ant-oat", API_key))
            "via your Claude subscription (OAuth token; no per-token charge)"
        else "billed pay-per-token (API key)"
        message("Generating via ", llm$model, " (effort=",
                if (is.na(effort)) "n/a" else effort, ") -- ", bill,
                ". Priors are elicited interactively.")
    }

    # ---- streaming pre-flight: fail fast if the live stream is broken --------
    if (isTRUE(verify_stream) && identical(llm$provider, "anthropic") &&
        !use_cli_transport && is.null(responder)) {
        tryCatch(
            ai4bayescode_stream_check(LLM = llm$model, API_key = API_key, effort = NULL),
            error = function(e)
                stop("streaming pre-flight check failed (", conditionMessage(e),
                     "); aborting before the paid generation. Fix connectivity/key and ",
                     "retry, or pass verify_stream = FALSE.", call. = FALSE))
    }

    # ---- validate -> repair-to-convergence loop ----
    # `attempt` counts ONLY replies that produced code and went to the validator;
    # a reply with no code block (the model deferred, or was truncated) is re-asked
    # on a SEPARATE budget (`max_code_retries`) so it never burns a validation
    # attempt -- max_attempts is reserved for real compile/convergence failures.
    max_attempts     <- as.integer(max(1L, max_attempts))
    max_code_retries <- 6L
    msgs <- list(list(role = "user", content = prompt$user))
    em <- NULL; txt <- ""; result <- NULL; attempt <- 0L; code_retries <- 0L
    repeat {
        turn <- .ai4b_anthropic_turn(prompt$system, msgs, llm$model, effort, API_key,
                                     max_tokens, timeout, ask, verbose, call = call)
        txt  <- turn$text
        msgs <- turn$msgs
        em   <- .ai4b_write_emitted(txt, output_path, classname)
        writeLines(txt, file.path(output_path, paste0(classname, "_transcript.md")))

        if (!length(em$files)) {
            # No usable code block -- the model deferred the code or was truncated.
            # Re-ask for it WITHOUT spending a validation attempt.
            code_retries <- code_retries + 1L
            result <- list(ok = FALSE, stage = "no_code", detail = paste0(
                "Your response contained no fenced code block, so no file was written",
                if (isTRUE(turn$truncated))
                    " (your output was TRUNCATED at the token limit -- be more concise, or raise max_tokens)"
                else "", "."))
            if (verbose) message("No code in the reply",
                                 if (isTRUE(turn$truncated)) " (truncated at token limit)" else "",
                                 " -- re-asking (free retry ", code_retries, "/", max_code_retries,
                                 ", does not use a validation attempt).")
            if (code_retries > max_code_retries) {
                if (verbose) message("Gave up after ", max_code_retries,
                                     " code-less replies (no validation attempt was spent).")
                break
            }
            msgs <- c(msgs, list(list(role = "assistant", content = txt),
                                 list(role = "user", content = .ai4b_repair_msg(result))))
            next
        }

        # Code was emitted -> this is a real validation attempt.
        attempt <- attempt + 1L
        if (verbose) message("Attempt ", attempt, ": wrote ", length(em$files),
                             " file(s) to ", output_path, "; validating ...")
        result <- validate_fn(em$cpp_path, em$runner_path, classname, verbose)
        if (isTRUE(result$ok)) {
            if (verbose) message("Attempt ", attempt, ": validation PASSED.")
            break
        }
        if (attempt >= max_attempts) {
            if (verbose) message("Attempt ", attempt, ": validation FAILED at stage `",
                                 result$stage %||% "?", "`; out of attempts.")
            break
        }
        if (verbose) message("Attempt ", attempt, ": validation FAILED at stage `",
                             result$stage %||% "?", "`; repairing.")
        msgs <- c(msgs, list(list(role = "assistant", content = txt),
                             list(role = "user", content = .ai4b_repair_msg(result))))
    }

    invisible(list(cpp_path   = em$cpp_path,
                   files      = em$files,
                   prompt     = prompt,
                   called_api = TRUE,
                   transcript = txt,
                   validated  = isTRUE(result$ok),
                   attempts   = attempt,
                   validation = result))
}

#' @keywords internal
#' @noRd
`%||%` <- function(a, b) if (is.null(a) || (is.character(a) && !nzchar(a))) b else a

# ---------------------------------------------------------------------------
# Back-compat aliases (pre-0.9.4 names). Canonical names are ai4bayescode_*.
# ---------------------------------------------------------------------------
#' @rdname ai4bayescode_prompt
#' @export
ai4bayes_prompt <- ai4bayescode_prompt

#' @rdname ai4bayescode_models
#' @export
ai4bayes_models <- ai4bayescode_models

#' @rdname ai4bayescode_generate
#' @export
ai4bayes_generate <- ai4bayescode_generate
