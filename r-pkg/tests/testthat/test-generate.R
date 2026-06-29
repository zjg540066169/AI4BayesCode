# Offline tests for the NL->code front doors. No network / no API key.
# The agentic ask_user loop is tested with an injected mock transport (.responder)
# and a stub console (.ask), so the loop logic is exercised without a live call.
# The validate->repair loop is exercised with an injected .validate stub so no
# real compilation / Rscript run happens here.

test_that("ai4bayescode_prompt assembles a start.md-anchored prompt", {
    skip_if(!nzchar(ai4bayescode_skills_path("start.md")), "skills not bundled")
    p <- ai4bayescode_prompt("Logistic regression: y ~ Bernoulli(sigmoid(X beta)).",
                         backend = "R", output_path = "./gen")
    expect_true(grepl("START HERE", p$system))
    expect_true(grepl("NON-INFORMATIVE", p$user))
    expect_true(nzchar(p$classname))
    expect_identical(p$skills, "start.md")
})

test_that("prompt mandates the AI4BAYES_VALIDATE sentinel", {
    p <- ai4bayescode_prompt("Poisson GLM", backend = "R")
    expect_true(grepl("AI4BAYES_VALIDATE: PASS", p$user))
    expect_true(grepl("rank-normalized R-hat|posterior::rhat", p$user))
})

test_that("priors = 'interactive' instructs ask_user elicitation", {
    p <- ai4bayescode_prompt("Poisson GLM", priors = "interactive")
    expect_true(grepl("ask_user", p$user))
    expect_true(grepl("ELICIT the prior", p$user))
})

test_that("ai4bayescode_models + LLM resolution", {
    m <- ai4bayescode_models()
    expect_true(all(c("name", "provider", "model_id", "implemented") %in% names(m)))
    expect_identical(AI4BayesCode:::.ai4b_resolve_llm("claude-opus-4-8")$provider, "anthropic")
    # friendly name + effort suffix strip
    r <- AI4BayesCode:::.ai4b_resolve_llm("Claude Opus 4.8 Max")
    expect_identical(r$model, "claude-opus-4-8")
    expect_true(r$implemented)
    g <- AI4BayesCode:::.ai4b_resolve_llm("gpt-5.5")
    expect_identical(g$provider, "openai")
    expect_true(g$implemented)                       # OpenAI provider now implemented
    # an unrecognized provider (google/gemini) is still an extension point
    gg <- AI4BayesCode:::.ai4b_resolve_llm("gemini-2.5-pro")
    expect_identical(gg$provider, "google")
    expect_false(gg$implemented)
})

test_that("ai4bayescode_generate offline (no key, no CLI) writes PROMPT.txt", {
    td <- file.path(tempdir(), "ai4b_gen_off")
    unlink(td, recursive = TRUE)
    # The offline branch fires only when there is no key AND no `claude` CLI on the
    # PATH. A CLI may exist on this machine, so temporarily blank PATH to hide it.
    old_path <- Sys.getenv("PATH", unset = NA)
    on.exit(if (is.na(old_path)) Sys.unsetenv("PATH") else Sys.setenv(PATH = old_path),
            add = TRUE)
    Sys.setenv(PATH = "")
    res <- ai4bayescode_generate("Gaussian location-scale.", classname = "GLS",
                             output_path = td, backend = "R", interactive = FALSE,
                             API_key = "", verbose = FALSE)
    expect_false(res$called_api)
    expect_true(file.exists(file.path(td, "PROMPT.txt")))
})

test_that("agentic ask_user loop: model asks a prior, host answers, code extracted", {
    td <- file.path(tempdir(), "ai4b_gen_loop")
    unlink(td, recursive = TRUE)
    asked <- character(0)
    # mock transport: turn 1 calls ask_user; turn 2 emits the .cpp
    turn <- 0L
    responder <- function(messages) {
        turn <<- turn + 1L
        if (turn == 1L)
            list(stop_reason = "tool_use", content = list(
                list(type = "text", text = "Eliciting priors."),
                list(type = "tool_use", id = "t1", name = "ask_user",
                     input = list(question = "Prior for beta?",
                                  options = list("non-informative", "weakly")))))
        else
            list(stop_reason = "end_turn", content = list(list(type = "text",
                text = "```cpp\n// path: GenFoo.cpp\n#include <RcppArmadillo.h>\nclass GenFoo {};\n```")))
    }
    ask_stub <- function(prompt, options = NULL, default = NULL) {
        asked <<- c(asked, prompt); "non-informative"
    }
    res <- ai4bayescode_generate("Logistic regression with coefficient beta.",
                             classname = "GenFoo", output_path = td, backend = "R",
                             interactive = FALSE, verbose = FALSE,
                             .responder = responder, .ask = ask_stub,
                             .validate = function(...) list(ok = TRUE))
    expect_true(res$called_api)
    expect_match(res$cpp_path, "GenFoo\\.cpp$")
    expect_true(file.exists(res$cpp_path))
    expect_match(asked[1], "Prior for beta")     # the model's question reached the console
    expect_equal(turn, 2L)                        # exactly two transport calls
})

test_that("validate-repair loop repairs on attempt 2", {
    td <- file.path(tempdir(), "ai4b_gen_repair")
    unlink(td, recursive = TRUE)
    # transport: each turn emits a .cpp + a runner (terminal end_turn).
    responder <- function(messages) {
        list(stop_reason = "end_turn", content = list(list(type = "text", text = paste0(
            "```cpp\n// path: Foo.cpp\n#include <RcppArmadillo.h>\nclass Foo {};\n```\n",
            "```r\n// path: Foo_runner.R\ncat('AI4BAYES_VALIDATE: PASS\\n')\n```"))))
    }
    # .validate: FAIL (compile) on its first call, PASS on its second.
    vcount <- 0L
    validate_stub <- function(cpp_path, runner_path, classname, verbose = FALSE) {
        vcount <<- vcount + 1L
        if (vcount == 1L) list(ok = FALSE, stage = "compile", detail = "boom")
        else list(ok = TRUE)
    }
    res <- ai4bayescode_generate("Model with parameter mu.",
                             classname = "Foo", output_path = td, backend = "R",
                             interactive = FALSE, verbose = FALSE,
                             .responder = responder, .ask = function(...) "",
                             .validate = validate_stub)
    expect_equal(res$attempts, 2L)
    expect_true(isTRUE(res$validated))
    expect_equal(vcount, 2L)
})

test_that("validates on attempt 1", {
    td <- file.path(tempdir(), "ai4b_gen_pass1")
    unlink(td, recursive = TRUE)
    responder <- function(messages) list(stop_reason = "end_turn", content = list(
        list(type = "text",
             text = "```cpp\n// path: Bar.cpp\n#include <x>\nclass Bar{};\n```")))
    res <- ai4bayescode_generate("Model with parameter mu.",
                             classname = "Bar", output_path = td, backend = "R",
                             interactive = FALSE, verbose = FALSE,
                             .responder = responder, .ask = function(...) "",
                             .validate = function(...) list(ok = TRUE))
    expect_equal(res$attempts, 1L)
    expect_true(res$validated)
})

test_that("unimplemented provider errors clearly", {
    # gpt-5.5 (openai) now works, so use a still-unimplemented provider (google).
    expect_error(
        ai4bayescode_generate("X", classname = "X", LLM = "gemini-2.5-pro", interactive = FALSE,
                          .responder = function(m) NULL, .ask = function(...) ""),
        regexp = "not yet implemented")
})

test_that("openai (gpt-5.5) is dispatched, no longer errors", {
    td <- file.path(tempdir(), "ai4b_gen_openai"); unlink(td, recursive = TRUE)
    # .responder overrides the transport, so NO live OpenAI call is made.
    responder <- function(messages) list(stop_reason = "end_turn", content = list(
        list(type = "text",
             text = "```cpp\n// path: LR.cpp\n#include <RcppArmadillo.h>\nclass LR{};\n```")))
    res <- ai4bayescode_generate("Linear regression.", classname = "LR", LLM = "gpt-5.5",
                             API_key = "sk-test", effort = "high", backend = "R",
                             interactive = FALSE, verbose = FALSE,
                             .responder = responder,
                             .validate = function(...) list(ok = TRUE))
    expect_true(res$called_api)
    expect_match(res$cpp_path, "LR\\.cpp$")
    expect_true(file.exists(res$cpp_path))
})

test_that("openai converters round-trip", {
    # ---- to_openai: Anthropic assistant-tool_use + a tool_result ----
    msgs <- list(
        list(role = "user", content = "Describe the model."),
        list(role = "assistant", content = list(
            list(type = "text", text = "Eliciting."),
            list(type = "tool_use", id = "call_1", name = "ask_user",
                 input = list(question = "Prior for beta?",
                              options = list("non-informative", "weakly"))))),
        list(role = "user", content = list(
            list(type = "tool_result", tool_use_id = "call_1",
                 content = "non-informative"))))
    oa <- AI4BayesCode:::.ai4b_to_openai_msgs(msgs, "SYSTEM PROMPT")
    expect_identical(oa[[1]]$role, "system")
    expect_identical(oa[[1]]$content, "SYSTEM PROMPT")
    expect_identical(oa[[2]]$role, "user")
    # assistant turn -> tool_calls
    asst <- oa[[3]]
    expect_identical(asst$role, "assistant")
    expect_equal(length(asst$tool_calls), 1L)
    expect_identical(asst$tool_calls[[1]]$id, "call_1")
    expect_identical(asst$tool_calls[[1]]$type, "function")
    expect_identical(asst$tool_calls[[1]][["function"]]$name, "ask_user")
    expect_match(as.character(asst$tool_calls[[1]][["function"]]$arguments), "Prior for beta")
    # tool_result -> role:"tool"
    toolmsg <- oa[[4]]
    expect_identical(toolmsg$role, "tool")
    expect_identical(toolmsg$tool_call_id, "call_1")
    expect_identical(toolmsg$content, "non-informative")

    # ---- from_openai: a tool_calls message -> Anthropic tool_use block ----
    resp_tu <- list(choices = list(list(message = list(
        content = "Let me ask.",
        tool_calls = list(list(id = "call_9", type = "function",
            `function` = list(name = "ask_user",
                arguments = "{\"question\":\"Prior for sigma?\"}")))))))
    a <- AI4BayesCode:::.ai4b_from_openai_response(resp_tu)
    expect_identical(a$stop_reason, "tool_use")
    types <- vapply(a$content, function(b) b$type, "")
    expect_true("text" %in% types)
    tub <- a$content[[which(types == "tool_use")]]
    expect_identical(tub$type, "tool_use")
    expect_identical(tub$id, "call_9")
    expect_identical(tub$name, "ask_user")
    expect_identical(tub$input$question, "Prior for sigma?")

    # ---- from_openai: a plain text message -> text block + end_turn ----
    resp_txt <- list(choices = list(list(message = list(
        content = "```cpp\n// path: Z.cpp\nclass Z{};\n```", tool_calls = NULL))))
    b <- AI4BayesCode:::.ai4b_from_openai_response(resp_txt)
    expect_identical(b$stop_reason, "end_turn")
    expect_equal(length(b$content), 1L)
    expect_identical(b$content[[1]]$type, "text")
    expect_match(b$content[[1]]$text, "path: Z.cpp")
})

test_that("non-interactive missing model_description errors", {
    expect_error(
        ai4bayescode_generate(interactive = FALSE, API_key = ""),
        regexp = "model_description.*required")
})

test_that("default LLM is a valid bare model id", {
    expect_identical(formals(ai4bayescode_generate)$LLM, "claude-opus-4-8")
})

test_that("effort levels are per-model", {
    expect_true("effort_levels" %in% names(ai4bayescode_models()))
    expect_true("max" %in% AI4BayesCode:::.ai4b_model_effort_levels("claude-opus-4-8"))
    expect_false("xhigh" %in% AI4BayesCode:::.ai4b_model_effort_levels("claude-sonnet-4-6"))
    expect_identical(AI4BayesCode:::.ai4b_model_effort_levels("claude-haiku-4-5"), character(0))
    expect_null(AI4BayesCode:::.ai4b_model_effort_levels("totally-unknown-model"))
})

test_that("interactive: model + effort menus shown; effort offered from the model's levels", {
    asked <- character(0); eff_opts <- NULL
    ask <- function(prompt, options = NULL, default = NULL) {
        asked <<- c(asked, prompt)
        if (grepl("effort", prompt, ignore.case = TRUE)) { eff_opts <<- options; return("high") }
        if (grepl("LLM model", prompt)) return(default)           # accept default model (sonnet)
        if (!is.null(default)) return(default)
        "high"
    }
    responder <- function(messages) list(stop_reason = "end_turn", content = list(
        list(type = "text", text = "```cpp\n// path: E.cpp\n#include <x>\nclass E{};\n```")))
    td <- file.path(tempdir(), "ai4b_eff"); unlink(td, recursive = TRUE)
    res <- ai4bayescode_generate("Model with sigma.", classname = "E", LLM = "claude-sonnet-4-6",
                             effort = "xhigh", output_path = td, backend = "R",
                             interactive = TRUE, verbose = FALSE,
                             .responder = responder, .ask = ask,
                             .validate = function(...) list(ok = TRUE))
    expect_true(any(grepl("LLM model", asked)))                   # model menu shown
    expect_true(any(grepl("effort", asked, ignore.case = TRUE)))  # effort menu shown
    expect_false("xhigh" %in% eff_opts)                           # sonnet's levels exclude xhigh
    expect_true("high" %in% eff_opts)
    expect_true(res$called_api)
})

test_that("auth: OAuth subscription token -> Bearer; API key -> x-api-key", {
    h_oauth <- AI4BayesCode:::.ai4b_anthropic_headers("sk-ant-oat01-abc")
    expect_true(grepl("^Bearer ", h_oauth[["Authorization"]]))
    expect_identical(unname(h_oauth[["anthropic-beta"]]), "oauth-2025-04-20")
    h_api <- AI4BayesCode:::.ai4b_anthropic_headers("sk-ant-api03-xyz")
    expect_identical(unname(h_api[["x-api-key"]]), "sk-ant-api03-xyz")
    expect_true(is.na(h_api["Authorization"]))
})

test_that("provider key falls back to ANTHROPIC_AUTH_TOKEN", {
    old_api <- Sys.getenv("ANTHROPIC_API_KEY", unset = NA)
    old_tok <- Sys.getenv("ANTHROPIC_AUTH_TOKEN", unset = NA)
    on.exit({
        if (is.na(old_api)) Sys.unsetenv("ANTHROPIC_API_KEY") else Sys.setenv(ANTHROPIC_API_KEY = old_api)
        if (is.na(old_tok)) Sys.unsetenv("ANTHROPIC_AUTH_TOKEN") else Sys.setenv(ANTHROPIC_AUTH_TOKEN = old_tok)
    }, add = TRUE)
    Sys.setenv(ANTHROPIC_API_KEY = "", ANTHROPIC_AUTH_TOKEN = "sk-ant-oat01-zzz")
    expect_identical(AI4BayesCode:::.ai4b_provider_key("anthropic"), "sk-ant-oat01-zzz")
})

test_that("effort invalid for the model errors non-interactively", {
    expect_error(
        ai4bayescode_generate("X", classname = "X", LLM = "claude-sonnet-4-6", effort = "xhigh",
                          interactive = FALSE, .responder = function(m) NULL),
        regexp = "not valid for")
})

# ---------------------------------------------------------------------------
# ai4bayescode_set_key / key_status + blank key -> offline (NOT the claude CLI)
# ---------------------------------------------------------------------------
test_that("ai4bayescode_set_key sets the session env var; never prints the full key", {
    old <- Sys.getenv("OPENAI_API_KEY", unset = NA)
    on.exit(if (is.na(old)) Sys.unsetenv("OPENAI_API_KEY") else Sys.setenv(OPENAI_API_KEY = old))
    msg <- capture.output(ai4bayescode_set_key("sk-proj-abcdef123456789", "openai"), type = "message")
    expect_identical(Sys.getenv("OPENAI_API_KEY"), "sk-proj-abcdef123456789")
    expect_identical(AI4BayesCode:::.ai4b_provider_key("openai"), "sk-proj-abcdef123456789")
    expect_false(any(grepl("abcdef123456789", msg)))   # full key NOT echoed in the message
    expect_true(any(grepl("sk-pro", msg)))             # masked prefix shown
})

test_that("ai4bayescode_set_key validates the provider and the key", {
    expect_error(ai4bayescode_set_key("", "openai"), "non-empty")
    expect_error(ai4bayescode_set_key("sk-x", "nope"), "should be one of|arg")
})

test_that("ai4bayescode_key_status masks keys and returns a presence vector", {
    old <- Sys.getenv("ANTHROPIC_API_KEY", unset = NA)
    on.exit(if (is.na(old)) Sys.unsetenv("ANTHROPIC_API_KEY") else Sys.setenv(ANTHROPIC_API_KEY = old))
    Sys.setenv(ANTHROPIC_API_KEY = "sk-ant-api-XYZ987654321")
    st  <- suppressMessages(ai4bayescode_key_status())
    expect_true(st[["anthropic"]])
    msg <- capture.output(suppressWarnings(ai4bayescode_key_status()), type = "message")
    expect_false(any(grepl("XYZ987654321", msg)))      # full key NOT echoed
})

test_that("blank key with use_cli = FALSE -> offline PROMPT.txt, not the claude CLI", {
    a <- Sys.getenv("ANTHROPIC_API_KEY", unset = NA); t <- Sys.getenv("ANTHROPIC_AUTH_TOKEN", unset = NA)
    Sys.unsetenv("ANTHROPIC_API_KEY"); Sys.unsetenv("ANTHROPIC_AUTH_TOKEN")
    on.exit({
        if (!is.na(a)) Sys.setenv(ANTHROPIC_API_KEY = a)
        if (!is.na(t)) Sys.setenv(ANTHROPIC_AUTH_TOKEN = t)
    })
    td <- file.path(tempdir(), "ai4b_offline"); unlink(td, recursive = TRUE)
    res <- ai4bayescode_generate("Linear regression.", classname = "LR", API_key = "",
                             output_path = td, backend = "R", interactive = FALSE, verbose = FALSE)
    expect_false(res$called_api)                            # no API/CLI call was made
    expect_true(file.exists(file.path(td, "PROMPT.txt")))   # predictable offline emit
})

# ---------------------------------------------------------------------------
# "stuck after prior" bug: model emits NO code block -> must detect + recover
# ---------------------------------------------------------------------------
test_that("repair message for a no-code attempt forcefully demands the code", {
    m <- AI4BayesCode:::.ai4b_repair_msg(list(stage = "no_code", detail = "no block."))
    expect_match(m, "No sampler code was emitted")
    expect_match(m, "ONLY the fenced code")
    expect_match(m, "// path:")
})

test_that("0-file attempt is detected as no_code, repaired, and the loop recovers", {
    turn_n <- 0L
    responder <- function(messages) {
        turn_n <<- turn_n + 1L
        if (turn_n == 1L)                       # model asks for a prior
            return(list(stop_reason = "tool_use", content = list(
                list(type = "text", text = "Eliciting."),
                list(type = "tool_use", id = "t1", name = "ask_user",
                     input = list(question = "Prior for sigma?",
                                  options = c("Jeffreys", "Half-Normal"))))))
        if (turn_n == 2L)                       # BUG: defers, emits NO code block
            return(list(stop_reason = "end_turn", content = list(
                list(type = "text", text = "Great, I'll generate the sampler now."))))
        list(stop_reason = "end_turn", content = list(list(type = "text",   # then emits code
            text = "```cpp\n// path: LR.cpp\n#include <x>\nclass LR{};\n```")))
    }
    val_cpp <- list()
    validate <- function(cpp_path, runner_path, classname, verbose) {
        val_cpp[[length(val_cpp) + 1L]] <<- cpp_path
        if (is.na(cpp_path)) list(ok = FALSE, stage = "compile", detail = "x") else list(ok = TRUE)
    }
    td <- file.path(tempdir(), "ai4b_nocode"); unlink(td, recursive = TRUE)
    res <- ai4bayescode_generate("Linear regression.", classname = "LR", API_key = "sk-x",
                             output_path = td, backend = "R", interactive = FALSE, verbose = FALSE,
                             max_attempts = 1L, .responder = responder,   # only ONE validation attempt
                             .ask = function(q, options = NULL, default = NULL) "Jeffreys",
                             .validate = validate)
    expect_true(res$validated)                  # recovered even with max_attempts = 1 ...
    expect_identical(res$attempts, 1L)          # ... because the 0-file reply was a FREE retry
    expect_length(val_cpp, 1L)                  # validate ran only for the real code attempt
    expect_match(val_cpp[[1]], "LR\\.cpp$")
})

test_that("truncated reply (stop_reason=max_tokens) is flagged as truncated no_code", {
    responder <- function(messages)             # opens a fence, never closes it
        list(stop_reason = "max_tokens", content = list(list(type = "text",
            text = "```cpp\n// path: LR.cpp\n#include <x>\nclass LR{ // ...cut off")))
    validate <- function(...) stop("validate must not run when 0 files were written")
    td <- file.path(tempdir(), "ai4b_trunc"); unlink(td, recursive = TRUE)
    res <- ai4bayescode_generate("Linear regression.", classname = "LR", API_key = "sk-x",
                             output_path = td, backend = "R", interactive = FALSE, verbose = FALSE,
                             max_attempts = 1L, .responder = responder, .validate = validate)
    expect_false(res$validated)
    expect_identical(res$validation$stage, "no_code")
    expect_match(res$validation$detail, "TRUNCATED")
})

# ---------------------------------------------------------------------------
# confirm_model: state the model + get user approval BEFORE writing code
# ---------------------------------------------------------------------------
test_that("confirm_model toggles a pre-code confirmation step in the prompt", {
    p_yes <- ai4bayescode_prompt("Linear regression.", confirm_model = TRUE)
    expect_match(p_yes$user, "PRE-GENERATION MODEL CONFIRMATION")
    expect_match(p_yes$user, "ask_user")
    expect_match(p_yes$user, "display-math")            # REQUIRED: full LaTeX formulas
    expect_match(p_yes$user, "parameter summary table") # REQUIRED: parameter table
    p_no <- ai4bayescode_prompt("Linear regression.", confirm_model = FALSE)
    expect_match(p_no$user, "SKIP the model-confirmation")
})

test_that("generate(): confirm_model follows `interactive`, and can be forced on", {
    resp <- function(m) list(stop_reason = "end_turn", content = list(list(type = "text",
        text = "```cpp\n// path: LR.cpp\n#include <x>\nclass LR{};\n```")))
    ok <- function(...) list(ok = TRUE)
    td <- file.path(tempdir(), "ai4b_confirm"); unlink(td, recursive = TRUE)
    r1 <- ai4bayescode_generate("LR.", classname = "LR", API_key = "sk-x", output_path = td,
                            backend = "R", interactive = FALSE, verbose = FALSE,
                            .responder = resp, .validate = ok)
    expect_match(r1$prompt$user, "SKIP the model-confirmation")   # non-interactive -> no confirm
    r2 <- ai4bayescode_generate("LR.", classname = "LR", API_key = "sk-x", output_path = td,
                            backend = "R", interactive = FALSE, confirm_model = TRUE, verbose = FALSE,
                            .responder = resp, .validate = ok)
    expect_match(r2$prompt$user, "PRE-GENERATION MODEL CONFIRMATION")   # forced on
})

test_that("confirm_model: model confirms via ask_user, then emits code", {
    tn <- 0L
    resp <- function(m) { tn <<- tn + 1L
        if (tn == 1L) return(list(stop_reason = "tool_use", content = list(
            list(type = "text", text = "Here is my understanding."),
            list(type = "tool_use", id = "c1", name = "ask_user",
                 input = list(question = "Model: y~N(Xb,s2), beta flat, sigma Jeffreys. Correct?",
                              options = c("Correct -- generate", "Not quite"))))))
        list(stop_reason = "end_turn", content = list(list(type = "text",
            text = "```cpp\n// path: LR.cpp\n#include <x>\nclass LR{};\n```")))
    }
    asked <- character(0)
    ask <- function(q, options = NULL, default = NULL) { asked <<- c(asked, q); "Correct -- generate" }
    td <- file.path(tempdir(), "ai4b_confirm2"); unlink(td, recursive = TRUE)
    res <- ai4bayescode_generate("LR.", classname = "LR", API_key = "sk-x", output_path = td,
                             backend = "R", interactive = FALSE, confirm_model = TRUE, verbose = FALSE,
                             max_attempts = 1L, .responder = resp, .ask = ask,
                             .validate = function(...) list(ok = TRUE))
    expect_true(res$validated)
    expect_true(any(grepl("Correct\\?", asked)))   # the confirmation reached the user before code
})

test_that("max_tokens defaults to NULL (no chosen cap; model maximum)", {
    expect_null(formals(ai4bayescode_generate)$max_tokens)
})
