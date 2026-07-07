# AI4BayesCode -- START HERE (read this FIRST)

You are an LLM (Claude, GPT, Gemini, Llama -- any) using the AI4BayesCode
header-only library -- either GENERATING A SAMPLER for a model (the standard
flow) or DESIGNING A NEW BLOCK for the library (the block_design flow). This
file is the canonical entry point. Read it FIRST, then follow the phase-by-phase
load schedule below. Do NOT pre-load all of `skills/*.md` -- that
wastes context. Load each instruction file ONLY when entering its
phase.

The user invokes the workflow by saying something like:
> "Read AI4BayesCode/start.md first, then [model description]."

**Trigger semantics.** The **mode-routing question** (generate vs design --
see "Mode routing" below) is the FIRST gate and DOES fire as soon as you have
read this file when the mode is not already clear (e.g. the user only said
"read start.md"). It asks ONLY the flow, never any model detail, so it is always
safe to pop -- so on a docs-only read your correct response is: briefly confirm
what you read, then **pop the mode-routing picker** (do NOT just stop with a
prose "what would you like to do?"). What still WAITS for a model description is
the **upfront-question BATCH** (backend / class name / output folder / library
path) and every later phase: do NOT pop those speculatively, do NOT prompt for
the model, do NOT proceed into a phase until the model arrives (sampler branch)
or the design flow is entered. In short: read this file -> pop the routing picker
-> then wait per the chosen flow.

---

## Mode routing -- the FIRST gate (right after reading this file)

AI4BayesCode has two flows. Decide which BEFORE anything else.

- If the user's request ALREADY makes the mode clear, enter that flow directly
  (**no routing question**): a model to fit ("build me a logistic regression") ->
  **generate sampler**; an intent to build a new primitive ("I want to design a new
  block for X") -> **design new block**.
- If the mode is NOT clear (e.g. only "read start.md", or ambiguous), pop ONE routing
  question -- this is safe (it asks only the mode, not any model detail). Use the
  environment-appropriate mechanism (AskUserQuestion in Claude Code; the Sec.0 markdown
  fallback otherwise):

  ```
  What do you want to do?
  (a) Generate a sampler (default) -- compose existing blocks into a sampler for your model
  (b) Design a new block -- create a new primitive for the library
  ```

- **(a) Generate a sampler** -> the standard flow: continue with the Sec.1 phase schedule
  below (codegen.md -> codegen_priors -> codegen_cpp -> runner -> validator).
- **(b) Design a new block** -> load `skills/block_design_skills/00_flow.md` and follow the
  block_design interactive flow (it carries its own phase schedule and sign-off gates).

The routing question is the ONLY thing that may fire before a model description; the
upfront-question batch (backend / class / folder / path) still waits per the trigger
semantics above (sampler branch).

---

## 0. ABSOLUTE RULE -- how to ask the user

**The asking is mandatory; the mechanism is environment-dependent.**

NEVER skip asking the user just because your environment lacks a
structured-question tool. If no such tool is available in the
current session (e.g., Codex CLI in Default mode, plain chat,
Gemini CLI, etc.), you MUST still ask -- use the markdown fallback
in chat (see "Concrete mechanism per environment" below). NEVER
silently fall through to defaults for upfront questions (backend,
class name, output folder, library path), prior choices, model
confirmation, block selection, or any other design decision listed
below. Defaults exist to suggest a standing choice IN the question,
not to bypass the question.

**Delivery is gated on full validation.** NEVER tell the user the
code is done before L1 + L2 + L3 R1/R2/R3 all pass. See Sec.5.

Every **design question** to the user MUST be a structured prompt
with explicit, labeled options. Use whatever your environment's
user-prompt mechanism is (a structured-question tool, a chat
prompt, etc.) -- the key invariants are independent of the tool:

- **One question or a small batch (<=4)** per prompt. No paragraph of
  prose with embedded questions.
- **Explicit options**: label them (a) / (b) / (c) / ..., with trade-off
  content (paper citations, equations, cost notes) inline in each
  option's text -- NOT in a separate prose paragraph the user has to
  parse.
- **Mark the standing / suggested choice as `(default)`** in its
  label -- e.g. `"Weakly informative N(0, 5^2) (default)"`. NEVER label
  it `(Recommended)` -- too absolute; users misread it as the only
  correct answer.
- **Always include an "Other / custom" fallback** so the user can
  type a free-text answer.
- **2-3 sensible defaults** for any free-text-natural question.
- **Substantial review-content goes in the chat BODY, not the option text.** Short
  trade-off notes belong inline in an option (above). But when the user must REVIEW a full
  artifact before confirming -- a drafted model spec, a `SelectWhen` line, a generated config,
  a derived formula -- PRINT it into the chat first (a `>` blockquote / fenced code block /
  **bold**, easy to read), then keep the options SHORT (`Use it (default) / Refine -- tell me
  the change / Other`). Cramming a paragraph-long draft into an option's small grey
  description is unreadable.

### Concrete mechanism per environment

The invariants above are environment-agnostic; the mechanism depends
on your runtime. Pick whichever path matches your environment -- do
NOT invoke a tool by name that your environment does not actually
expose.

**Claude Code** -- call the `AskUserQuestion` tool (available in Claude Code
sessions; batches up to 4 questions per call):

```json
{
  "questions": [{
    "question": "Which prior for sigma?",
    "header":   "sigma prior",
    "multiSelect": false,
    "options": [
      {"label": "HalfNormal(0, 1) (default)",
       "description": "Weakly informative; OK for unit-scale data."},
      {"label": "Jeffreys 1/sigma",
       "description": "Improper but conditionally proper posterior; Gelman 2006."},
      {"label": "InverseGamma(1, 1)",
       "description": "Discouraged -- see Gelman 2006 critique."}
    ]
  }]
}
```

**Codex CLI (or any agent with a structured-input tool)** -- IF a structured
user-input tool is present (historically Codex exposes `request_user_input`, and
only in Plan mode), you MAY use it (1-3 short questions per call). Do NOT call a
tool by a name you have not confirmed exists in the current session -- if it is
absent (e.g. Codex Default mode), use the markdown fallback below.

**Any environment without a structured-question tool** (Codex CLI
Default mode, Gemini CLI, plain ChatGPT, generic chat UIs, etc.) --
do NOT try to call a structured tool. Emit markdown labeled options
directly in chat; the user replies with the option letter or
`(d) Other: <free text>`:

```
Which prior for sigma?
(a) HalfNormal(0, 1) (default) -- weakly informative; OK for unit-scale data
(b) Jeffreys 1/sigma -- improper but conditionally proper posterior; Gelman 2006
(c) InverseGamma(1, 1) -- discouraged; see Gelman 2006 critique
(d) Other / custom
```

The invariants (one question/small batch, explicit labeled options,
`(default)` on the standing choice, `Other` fallback, no prose
questions) apply identically across all three mechanisms.

Covers (non-exhaustive):

- Upfront questions (backend / class name / output folder / path)
- Prior specification (every parameter, every choice)
- Model confirmation, DAG sign-off
- Block-selection follow-up
- Methodology forks -- paper variant A vs B, sampler choice, etc.
- Pre-flight / "before I touch files" -- no "discussion-only" or
  "methodology-lockdown" phase bypasses this rule. The workflow is
  active from the moment the user describes a model.

### Which moments require a structured-question prompt -- and which don't

Asking is for **design decisions only**. Procedural workflow steps
run automatically without asking.

| | Ask the user? |
|---|---|
| **Design questions** | YES |
| Runtime backend / class name / output folder / AI4BayesCode path | YES |
| Missing or ambiguous prior on a parameter | YES |
| Methodology forks (paper variant A vs B, sampler choice, augmentation strategy) | YES |
| Block-selection disambiguation (NUTS vs joint NUTS vs specialised) | YES |
| Model confirmation / DAG sign-off after presenting the inline DAG + table | YES |
| Ship-ready confirmation ("save final files and finish?") | YES |
| **Procedural workflow steps** | NO -- run automatically |
| Compile the generated `.cpp` (`ai4bayescode_sourceCpp(...)` in R, equivalent in Python) | NO |
| L3 R1 smoke test (10 steps, finiteness, `predict_at` non-mutation, throwaway `verify_*.cpp` Check #12) | NO |
| L3 R2 two-chain diagnostic (4k+4k, R-hat, ESS) | NO |
| L3 R3 posterior-predictive Bayesian p-values + PSIS-LOO | NO |
| Render the prediction DAG inline (Mermaid / edge list) for the Sec.3 step | NO |
| Apply the full L2 semantic-check suite (`validator.md`) against the `.cpp` | NO |
| Delete the throwaway gradient-verify `.cpp` after Check #12 passes | NO |
| Write the `.cpp` / runner files to the user's chosen output folder (default `./generated/<ClassName>/` only if not specified upfront) | NO (just write) |
| Load the next instruction file per the phase schedule in Sec.1 | NO |

Never pause and ask "Should I proceed with verification?", "Run the
smoke test now?", "Compile first?" -- those are workflow steps you JUST
DO.

### Failure-recovery policy: auto-retry up to N attempts (default N = 5), then report

On a procedural-step failure (compile error, R-hat >= 1.05 at the
budget, BPV out of range, LOO Pareto-k too high, etc.), **do NOT
pause to ask the user**. Auto-retry up to N attempts total
(including attempt #1), escalating the fix between attempts --
e.g., compile error -> fix the syntax / typo / missing include and
recompile; R-hat fail at 4k+4k -> re-run at 20k+20k; still fails ->
re-tune metric / try non-centered reparam / try a joint block; etc.

**N defaults to 5.** The user may override by specifying a different
budget -- either upfront ("retry up to 3 times" / "retry up to 10
times") or mid-session at any point before validation starts. If
the user has not specified anything, use N = 5 silently.

If the N-th attempt still fails, **report the failure and stop**.
Emit a final message describing what failed, which N attempts you
tried, and what the user can do next -- but do NOT pop a "retry /
tune / abort?" picker. The user reads the report and responds if
they want.

### R-hat gray zone (1.01 < R-hat <= 1.05) -- opt-in stricter convergence

After validation completes (either on the first attempt or after the
auto-retry brought R-hat back under 1.05), check whether R-hat lies
in the gray zone (1.01, 1.05]. If yes, do NOT auto-retry, but DO
surface a structured question asking whether to keep tightening
toward the strict Vehtari (2021) rank-normalized threshold of 1.01.
Use the mechanism appropriate to your environment (AskUserQuestion /
request_user_input / markdown fallback per Sec.0 above).

Options must include:

- `(a) Ship as-is at R-hat = ___ (default)` -- 1.05 is the standard
  "converged" boundary; sufficient for exploratory work.
- `(b) Extend chain budget to 20k+20k warmup+sampling per chain
  (escalate to 40k+40k or 80k+80k if already at 20k)` -- push
  toward strict R-hat < 1.01.
- `(c) AI-proposed structural fix` -- the agent diagnoses the
  bottleneck from the current model + trace + R-hat profile and
  proposes a context-appropriate change (e.g., reparameterization,
  alternative block, prior tuning, identifiability constraint,
  marginalize a nuisance param, ...). The exact fix depends on the
  model.
- `(d) Other / custom`

The strict 1.01 threshold is the Vehtari (2021) rank-normalized
recommendation for publication-grade inference; 1.05 is sufficient
for exploratory work. The gray-zone question lets the user opt in
to the stricter standard when relevant.

The only time a procedural step warrants a question is the R-hat
gray-zone exception above, or when the fix is a genuine design
choice with multiple valid paths the user should weigh (e.g.,
"model is unidentified at any chain length -- add an ordering
constraint, restrict to K=2 only, or pick a different family?").
Other verification convergence-tuning is NOT that case.

---

## 1. Token discipline -- phase-by-phase load schedule

**Do NOT read all of `skills/*.md` upfront.** Reading them all
consumes the majority of the context window before any work begins.
Load each instruction file ONLY when entering its phase, and AT MOST
once per session.

| Phase | When | Instruction file to load |
|---|---|---|
| 0. Entry | session start | `start.md` (this file) -- already in progress |
| 1. Upfront questions | after reading this file | `skills/codegen.md` Sec.1 |
| 2. Prior elicitation | when a user prior is missing / ambiguous | `skills/codegen_priors.md` |
| 3. Model confirmation | before writing any code | `skills/codegen.md` Sec.2 (inline DAG + summary table) |
| 4. C++ emission | when starting to write `.cpp` | `skills/codegen_cpp.md` |
| 5. R / Python runner emission | after `.cpp` is written | `skills/codegen_r_runner.md` (R backend) / `skills/codegen_python_runner.md` (Python backend; load whichever the chosen backend needs) |
| 6. Verification (L1+L2+L3) | after `.cpp` + runner compile | `skills/validator.md` |

**On-demand instruction files** (load only when the topic comes up;
many generations need 0 of these):

- `skills/block_catalogue/index.md` -- when picking a block type
- `skills/system_design.md` -- for measure-theory questions, BLAS
  patterns, DAG semantics, the metric (diagonal/dense) + warmup
  (single-pilot/3-phase) decision (Sec.13)
- `skills/joint_nuts_failure.md` -- when a `joint_nuts_block` will not
  converge: the documented failure modes (funnel / banana / centered-
  hierarchical) and their fix (NCR / untwist REPARAMETERIZATION, not a
  metric change). Load this before retrying a stuck joint-NUTS model.
- `skills/hierarchical_re.md` -- hierarchical random-effects models
- `skills/label_switching.md` -- mixture / HMM identifiability
- `skills/constraints.md` -- constraint transforms + Jacobians
- `skills/rcpp_api.md` -- Rcpp-specific corner cases

If unsure whether an instruction file is needed now, default to NOT
loading it. You can always load it later if a specific question
demands it.

---

## 1b. Detect the AI4BayesCode install FIRST (installed package vs raw checkout)

Before the upfront questions, detect how AI4BayesCode is available in the user's
runtime -- it decides the compile call AND whether to even ask for a library path:

- R:      `requireNamespace("AI4BayesCode", quietly = TRUE)`
- Python: `importlib.util.find_spec("AI4BayesCode") is not None`

**If the package IS installed (the recommended setup):**

- Use the installed-package API everywhere, with a **RELATIVE** path:
  - R:      `ai4bayescode_sourceCpp("<ClassName>.cpp")`
  - Python: `Mod = AI4BayesCode.source("<ClassName>.cpp")`
- **SKIP the "Path to AI4BayesCode folder?" upfront question** -- it is not needed.
- **NEVER** emit `AI4BayesCode_path=`, a `source(".../AI4BayesCode_helpers.R")`
  runner line, or an absolute `/Users/...` / `C:\...` path anywhere (compile call,
  runner, or `@example`) -- the packaged API needs none of them, and absolute /
  checkout paths break on another machine or if the folder moves. (Working from a
  raw git checkout instead of an installed package? Install it first --
  `remotes::install_github(...)` / `pip install ...` -- then use the packaged form.)

The generated `@example` header block (what `doc()` shows) ALWAYS uses the
installed-package form when the package is present -- the clean relative
`ai4bayescode_sourceCpp("<ClassName>.cpp")` ONLY: no path, no runner
line, no absolute path. See `codegen_cpp.md` Sec.5.

---

## 2. Output folder convention

The output folder comes from the upfront questions (Sec.1 phase 1).
**Respect whatever the user picks** -- that path is the truth, even
if it's not the default.

- **Default** (used ONLY if the user did not specify a folder):
  `./generated/<ClassName>/`, at the same level as `./AI4BayesCode/`
  -- OUTSIDE the AI4BayesCode tree.
- **Never default to `AI4BayesCode/examples/`.** That folder is
  reserved for the shipped reference examples that come with the
  library; user-generated code lives in a separate folder so it
  doesn't pollute future `git pull` / re-install operations. Only
  place output under `AI4BayesCode/examples/` if the user EXPLICITLY
  asks "add this as a library example".
- **Always write to the path the user actually provided** -- do not
  silently rewrite it to `./generated/` or any other folder. If the
  user typed `./my_models/SpikeSlab/`, write there.

---

## 3. DAG: inline only, never PNG-render

When the model-confirmation phase needs a DAG, render it **inline in
chat** as Mermaid / Graphviz `dot` source / edge list. Do NOT spawn
R / igraph / external rendering -- that wastes tokens and time. See
`codegen.md` Sec.2(b) for the exact templates.

The DAG you show MUST be a **prediction DAG** (generative / causal
direction), not a Gibbs / dependency DAG. Edges mean "A produces B
in the data-generating story", not "block B reads A during MCMC
sampling". See `codegen.md` Sec.2(b) MANDATORY note.

## 3b. Math: DISPLAY-block LaTeX for equations; ASCII for inline symbols (NEVER raw Unicode)

**Hard rule: NO raw Unicode characters anywhere** -- not in chat prose, not in
table cells, not in generated code (identifiers, comments, strings). Write math
in plain ASCII (`sigma^2`, `beta`, `x_i^T beta`, `sum_i`, `tau`, `mu`, `>=`,
`<=`, `in`, `R^p`, `n x p`, `theta_j := mu + tau * eta_j`). The only place math
"renders" is a `$$ ... $$` display block (whose LaTeX source is itself ASCII).

This chat client renders **display / block math** but does NOT render **inline
`$...$`** (it shows the raw source). So match how the codegen flow already does it
(`codegen.md` "(a) LaTeX formulas"), which renders correctly:

- **Displayed equations** (the model / likelihood / priors / the model-confirmation
  block) -> put them in a `$$ ... $$` **display block with the `$$` on their OWN lines**
  and `\begin{aligned}` for multi-line. This renders. Template:

  ```
  $$
  \begin{aligned}
  y_i \mid \beta, \sigma &\sim \text{Normal}(x_i^\top \beta,\, \sigma^2), \quad i = 1, \ldots, N \\
  \beta_0 &\sim \text{Normal}(120,\, 50^2) \\
  \sigma &\sim \text{Jeffreys}\ (p(\sigma) \propto 1/\sigma)
  \end{aligned}
  $$
  ```

- **PREFER display blocks for ALL real math** -- a `$$ ... $$` block renders and reads
  best, so when in doubt lift the formula into one. A multi-line `\begin{aligned}` block
  is the default for any model / likelihood / prior set.
- **A lone symbol mid-sentence** -> plain ASCII (`beta`, `sigma`, `sigma^2`, `r_i`,
  `x_i^T beta`), NEVER raw Unicode and NEVER inline `$...$`. Lift any real expression into
  a `$$ ... $$` block instead of writing it out.
- **NEVER** inline `$...$` for anything you want rendered -- it shows the raw source here.
- **Tables are the trap** (the most common slip). A markdown cell CANNOT hold a `$$...$$`
  block and inline `$...$` shows raw -- so **inside any table cell use plain ASCII ONLY**:
  `X`, `y`, `beta`, `sigma`, `sigma^2`, `mu`, `Sigma`, `grad`, `R^p`, `n x p`, `X^T`,
  `r_i`, `>0`, `>=`, `<=`. If a cell's math is too complex for clean ASCII, lift it into a
  `$$...$$` block ABOVE the table and keep the cell to a short ASCII label.
- **Inline expressions in prose**: a whole expression (e.g. `grad_beta log p = X^T W r`)
  goes in a `$$...$$` block; a lone symbol stays inline as plain ASCII. Never Unicode,
  never inline `$...$`.

(DAGs stay inline per Sec.3 above.)

---

## 4. Validator must run for every generation

Before declaring a generated `.cpp` complete, walk through the L2
checks in `skills/validator.md`. The most commonly missed checks
this skill set has caught:

- **Check #6 (DAG consistency).** Every key read inside a
  `register_stochastic_refresher` lambda must either be a
  `declare_data_input` OR have a `declare_predict_edges` path to the
  refreshed output. Missing this is the silent-broken-
  `predict_at(list(X = X_new))` bug.
- **Check #6 sibling (predict_at wrapper forwarding).** If the class
  declares any `declare_data_input`, the Rcpp wrapper
  `predict_at(Rcpp::List)` MUST forward to
  `impl_->predict_at(replaced, predict_rng_)` -- NOT hard-reject
  non-empty `new_data` with `Rcpp::stop`. The hard-reject pattern
  compiles + R-hat-passes but silently breaks predictions.

---

## 5. Hard rules (from `codegen.md` Sec.12)

These apply at every phase; you may skim `codegen.md` Sec.12 for the
full list but the critical ones:

- **No hand-written Gibbs**: prefer existing blocks (`nuts_block`,
  `joint_nuts_block` -- handles real + per-slice POSITIVE/INTERVAL/
  ORDERED/SUM_TO_ZERO constraints; the old `joint_nuts_block_mixed`
  was folded into it 2026-06-18 -- `pg_logistic_block`, conjugate-Gibbs
  blocks, `bart_block`, etc.) over emitting a custom block from
  scratch. See `block_catalogue/index.md` for the full menu.
- **Block-selection priority** (TWO-PHASE, not a linear "first item wins"):
  **specialized / structural blocks claim what they match FIRST** (by
  applicability -- conjugate-Gibbs, `pg_logistic_block`, `bart_block`, `rjmcmc`,
  HMM, etc.; a genuine structural match ALWAYS outranks generic NUTS) ->
  **`joint_nuts_block` is the DEFAULT** for every continuous parameter left ->
  standalone `nuts_block` is LOW-priority (single scalar only) -> slice is the last
  resort. See `codegen_priors.md` Sec.2b (authoritative).
- **No Mermaid PNG render**: see Sec.3 above.
- **`predict_at` is `const`**: never mutate MCMC state in
  `predict_at`. Use the mutable `predict_rng_` for any RNG advance.
- **Every generated `.cpp` carries a runnable `@example:<backend>` block
  in its header** for the chosen backend(s) -- a small single-chain DGP +
  `new()` + `step()` that `doc()` surfaces. It is part of the deliverable
  (like the license header), authored C++-demo-first and kept in sync with
  the standalone `int main()` (see `codegen_cpp.md` Sec.5 "Header `@example`
  block" + "Authoring order").
- **Output goes to `./generated/`** (or the user's specified
  folder): see Sec.2 above.
- **NEVER deliver code before ALL validation phases complete.**
  Writing the `.cpp` and runner to the output folder during the
  pipeline (so compile / L3 R1 / R2 / R3 can actually run) is NOT
  delivery. Delivery happens only after L1 + L2 + L3 R1/R2/R3 all
  pass. While any phase is still running, do not phrase output as
  "done" / "ready" / "complete" / "the sampler compiles and the
  smoke test passes" in any way that suggests the work is finished.
  If the user asks "is it done?" the answer is "No -- [phase X] is
  still running."

---

## Quick orientation

You should now:

1. **Pop the mode-routing picker FIRST** (the first gate; see "Mode routing"
   above) when the user's intent is not already clear -- generate a sampler vs
   design a new block. Do this right after reading this file; do NOT pre-load
   any phase file (`codegen.md`, `block_design_skills/*`) before the mode is
   chosen, and do NOT just stop with a prose "what would you like to do?".
2. **Then follow the chosen flow.** **(b) Design a new block** -> load
   `skills/block_design_skills/00_flow.md`. **(a) Generate a sampler** -> its
   upfront-question BATCH (backend / class / folder / path) still WAITS for the
   model: if no model description has arrived, do NOT emit the batch and do NOT
   prompt for the model -- the routing picker is all that fires until the model
   spec is sent.
3. **Once the user sends the model description** (sampler branch), load
   `skills/codegen.md` and emit your first structured-question prompt -- batch the
   upfront questions. Each
   question MUST be its own labeled-option block. Required format
   when using the markdown fallback (no structured-question tool):

   ```
   Runtime backend?
   (a) R (default)
   (b) Python
   (c) C++ standalone
   (d) Both R + Python (dual-module)
   (e) Other / custom

   Output folder?
   (a) ./generated/<ClassName>/ (default)
   (b) Other / custom

   Class name (CamelCase)?
   (a) <a sensible model-derived suggestion> (default)
   (b) Other / custom

   Sampler / block preference? (codegen.md Sec.1.6)
   (a) Skip -- let the skill auto-pick blocks (default)
   (b) I have a suggestion -- describe here: ___

   Max generate->validate attempts? (codegen.md Sec.1.7)
   (a) 5 (default)
   (b) 10
   (c) 20
   (d) Unlimited (opt-in)
   ```

   For the `AskUserQuestion` tool (Claude Code), the same questions
   go into one or two structured calls (the tool caps at 4 per
   call, so split across 2 calls).

   **Forbidden formats -- every one is a bug** (the user can't
   parse and can't pick by letter):

   - Prose paragraph listing the questions: [NO] "Please provide:
     runtime backend, output folder, class name, AI4BayesCode
     path, sampler preference, max attempts."
   - Comma-separated defaults: [NO] "Standing defaults are: R,
     ./generated/<X>/, <ClassName>, ./AI4BayesCode, let the
     workflow choose, 5."
   - Any phrasing where the user can't reply with a single letter
     per question.
   - Skipping any question and silently using the default -- every
     question must appear as a block; the user must explicitly
     pick.

   The user replies with the option letter for each question (or
   "Other: ___" for free text).
4. After the user answers, proceed phase-by-phase per Sec.1 above.

