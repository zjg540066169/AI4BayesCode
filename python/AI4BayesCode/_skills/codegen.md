---
name: AI4BayesCode-codegen
description: |
  Entry point for generating an AI4BayesCode sampler (.cpp + .R
  runner) from a user's Bayesian model description. Trigger when the
  user describes a model and asks for an MCMC sampler, or says
  "generate AI4BayesCode code", "use AI4BayesCode", etc.

  Covers the high-level workflow: upfront questions, model
  confirmation with inline DAG + summary table, prior elicitation,
  C++ emission, R runner emission, verification + smoke-test-delete
  rule, top-level hard rules.

  Sub-skills (codegen_priors.md, codegen_cpp.md, codegen_r_runner.md,
  block_catalogue/index.md, validator.md, system_design.md, etc.) are
  loaded ON DEMAND per phase -- see Sec.Token discipline at the top of
  this file. DO NOT preload them at session start.
---

# AI4BayesCode code generator -- entry point

Turn a user's model into a single `.cpp` file + `.R` runner using the
AI4BayesCode header-only library.

## WARNING Token discipline -- DO NOT preload sub-skills

The codegen workflow references many sub-skills. **Reading them all
at session start consumes the majority of the context window before
any work begins.** Do not do this. Load each sub-skill **on demand**
when its content is needed for the current phase, and at most ONCE
per session.

**Load schedule:**

- `codegen.md` (this file) -- load at session start.
- `codegen_priors.md` -- load when entering **Sec.2 (Prior specification)**,
  or earlier only if the user's prior is ambiguous and you need
  block-selection / Gibbs discipline / discrete-variable decision
  tree content right now.
- `codegen_cpp.md` -- load only when entering **Sec.5+ (C++ emission)**.
  Skip until then.
- `codegen_r_runner.md` -- load only when emitting the R runner file
  (AFTER the `.cpp` is written).
- `block_catalogue/index.md`, `validator.md`, `system_design.md`,
  `hierarchical_re.md`, `label_switching.md`, `constraints.md`,
  `rcpp_api.md` -- load **on demand** when their specific topic comes
  up in the current question/decision. Most generations need only
  0-2 of these, not all.

If you find yourself reading multiple sub-skills before the user has
confirmed the model, STOP -- that is the token-waste pattern this
rule prevents. If unsure whether a sub-skill is needed now, default
to NOT loading it.

---

## 0. ABSOLUTE RULE -- asking is mandatory; the mechanism is environment-dependent

**ZERO EXCEPTIONS on ASKING** (the *mechanism* varies, the *asking* does not).
Every design question to the user -- at every stage, in every context -- MUST be a
structured prompt with explicit labeled options. WHICH mechanism depends on the
runtime (see `start.md` Sec.0 "Concrete mechanism per environment"):

- **In Claude Code** -- `AskUserQuestion` is the ONLY way: no markdown option
  bullets, no numbered lists, no "(a) / (b) / (c)" prose, no yes/no in chat.
- **In any environment WITHOUT a structured-question tool** (Codex Default mode,
  Gemini CLI, plain ChatGPT, generic chat UIs) -- use the markdown labeled-option
  fallback from `start.md` Sec.0; the user replies with the option letter. NEVER
  silently fall through to a default because the tool is missing.

(This SUPERSEDES any earlier "AskUserQuestion is the only way, no exceptions"
phrasing -- that wording is Claude-Code-specific; the asking requirement is
universal, the tool is not.)

Covers (non-exhaustive):

- **Upfront** -- backend / class name / output folder / path.
- **Prior specification** -- every parameter, every choice.
- **Model confirmation, DAG sign-off**.
- **Block-selection follow-up**.
- **Methodology forks** -- paper variant A vs B, sampler choice, etc.
  Trade-off content (paper citations, equations, costs) goes in each
  option's `description` field, NOT in chat prose.
- **Pre-flight / "before I touch files"** -- no "pre-codegen",
  "methodology-lockdown", or "discussion-only" phase bypasses this
  rule. The workflow is active from the moment the user describes a
  model.
- **Free-text-natural questions** -- give 2-3 sensible defaults; the
  auto "Other" lets the user type custom text.
- **Single questions** -- even one yes/no must be a tool call.
- **Default / suggested option** -- put the suggested choice as the
  first option and label it `"X (default)"`, NOT `"X (Recommended)"`
  and not as chat prose. Never use "(recommended)" on an option --
  "recommended" implies the choice is correct when it may not be;
  "(default)" states the standing value without that connotation.

In Claude Code: if `AskUserQuestion` is deferred (not yet loaded), FIRST run
`ToolSearch select:AskUserQuestion`, THEN emit the call. Max 4 questions per call;
batch if more; 2-4 fixed options + auto "Other". In a markdown-fallback
environment: same invariants, <=4 questions per batch, with an explicit
"(d) Other / custom" line for free text.

---

## 0b. Math rendering -- display blocks + plain ASCII, NEVER Unicode or inline `$...$`

**Hard rule: NO raw Unicode anywhere** -- chat prose, table cells, OR generated code
(identifiers, comments, strings). This chat renders `$$ ... $$` **display blocks** but NOT
inline `$...$` (inline prints raw source). So EVERYWHERE in the codegen conversation -- the
"read of your model", the Sec.1 confirm, the prior phase, the Sec.3 confirmation -- apply:

- **Any equation / expression** (a likelihood, a prior, a gradient, a multi-line model) -> a
  `$$ ... $$` **display block** with `$$` on its OWN lines and `\begin{aligned}` for multi-line
  (the Sec.3(a) form; its LaTeX source is itself ASCII). This renders. Do NOT inline it.
- **A lone symbol mid-sentence** -> plain ASCII (`beta`, `sigma`, `sigma^2`, `R^p`,
  `x_i^T beta`, `r_i`), NEVER raw Unicode, never `$...$`.
- **Table cells** -> plain ASCII ONLY (a `$$` block cannot live in a cell and inline `$...$`
  shows raw): `X y beta sigma sigma^2 R^p n x p >0 x_i^T beta`.
- **NEVER** write inline `$...$` for anything you want rendered. "Likelihood: `$y_i \sim
  N(x_i^\top\beta,\sigma^2)$`" is WRONG -- lift it into a `$$` block; a lone `beta` mid-sentence
  is just `beta`; `p(sigma^2) prop.to 1/sigma^2` is ASCII (or a `$$` block).

---

## 1. Upfront questions

**IMPORTANT: Follow the FULL workflow for EVERY model generation
request, even if you already generated a model earlier in this
conversation. Each model is independent -- never skip prior resolution,
model confirmation, DAG declaration, or compile verification just
because a previous model succeeded.**

Per **Sec.0 ABSOLUTE RULE**, ask these via `AskUserQuestion` before
writing any code. For
questions already answered earlier in the same session (e.g.
AI4BayesCode path), reuse the previous answer without re-asking.

### Step 0 -- render + CONFIRM the model FIRST (before any upfront question)

The MOMENT a model arrives, BEFORE asking anything else: **render the captured model spec** --
likelihood + parameters + supports + any priors the user already gave, each still-missing prior
marked `(prior: to be chosen)` -- as ONE `$$` **display block** (Sec.3(a) form; never inline `$...$`,
per Sec.0b), then **fire the structured confirm gate** (Sec.0 mechanism -- `AskUserQuestion` in Claude
Code, markdown labeled-option fallback elsewhere). It is NOT just the likelihood -- show any
priors already given too. Template:

```
$$
\begin{aligned}
y_i \mid \beta, \sigma^2 &\sim \mathcal{N}(x_i^\top \beta,\ \sigma^2), \quad i = 1, \ldots, n \\
p(\sigma^2) &\propto 1/\sigma^2 \quad \text{(Jeffreys -- fully specified)} \\
\beta &\sim \text{(prior: to be chosen)}
\end{aligned}
$$
```

> confirm gate -- *"Does this capture your model?"* -- (a) Looks right (default) / (b) Fix the
> model / (c) Other.

**HARD STOP -- confirm the model FIRST.** Do NOT ask ANY upfront question below (#1-#8) -- not even
the model-independent backend / folder / class / path -- until the user confirms the model.
Printing the block and continuing is NOT the gate; you MUST fire the structured ask and WAIT.
(Why first: don't make the user answer config questions before you've shown you even read their
model right.) Prior *completion* (filling `(prior: to be chosen)`) and the full-spec sign-off come
later, at Sec.2 and Sec.3. **Once the model is confirmed**, ask the upfront questions:

1. **Which language's usage example?** Write this question for a BEGINNER --
   do NOT use the words "host language", "runner", "tri-module", "backend", or
   "#ifdef" in what the user sees. The key thing to REASSURE them: the generated
   sampler ALWAYS works from R, Python, AND C++ -- they lose nothing by their
   answer. This choice ONLY decides which language a ready-to-run **usage
   example** is written in (a short script that simulates data, runs the sampler,
   and shows the diagnostics). Phrase it plainly, e.g.:

   > Your sampler will run from R, Python, and C++ either way. Which language
   > would you like the ready-to-run example written in?
   > - R (default) -- a short .R script that runs the sampler and shows diagnostics.
   > - Python -- the same as a .py script (needs Python >= 3.11 and pybind11 >= 2.11).
   > - Both R and Python -- an example in each.
   > - C++ only -- just the .cpp file and its built-in demo; no R or Python script.

   (Emit it as `AskUserQuestion` per Sec.0; keep the option descriptions in this
   plain, jargon-free style -- not `ai4bayescode_run_chains` / `sourceCpp`.)

   TECHNICAL (agent-only -- do NOT surface any of this to the user): the answer
   picks which runner file(s) get written -- R (`ai4bayescode_run_chains` +
   `ai4bayescode_diagnose`), Python (`AI4BayesCode.run_chains` +
   `AI4BayesCode.diagnose`), or both. The `.cpp` is ALWAYS the same file
   regardless: it carries BOTH `#ifdef AI4BAYESCODE_RCPP_MODULE` and
   `#ifdef AI4BAYESCODE_PYBIND_MODULE` blocks AND a fenced standalone `int main()`,
   so it source-compiles in R (`ai4bayescode_source`), Python
   (`AI4BayesCode.source`), or as a plain binary -- a user who generates in R can
   hand the identical `.cpp` to Python and run it there. **Never gate the `.cpp`
   binding blocks on this choice -- always emit both**, per `codegen_cpp.md`
   Sec."C++ standalone demo first". See the shipped examples for the pattern.
2. **Class name.** CamelCase, e.g. `HierarchicalNormal`. **Ask this BEFORE the output folder** --
   the folder default derives from it (`./generated/<ClassName>/`), so asking the folder first
   would have to default to a name you have not elicited yet (the "folder-before-name" absurdity).
3. **Output folder.** Default `./generated/<ClassName>/` (per-model subfolder -- avoids collisions
   across models; `<ClassName>` is the name from #2 above) at the same level as
   `./AI4BayesCode/`, **OUTSIDE the AI4BayesCode tree**. Auto-create if missing.
   **Do NOT default to `AI4BayesCode/examples/`.** That folder is
   reserved for the shipped reference examples that come with the
   library; user-generated code lives in a separate `./generated/`
   (or user-named) folder so it stays isolated from the library
   distribution and doesn't pollute future `git pull` / re-install
   operations. Only place output under `AI4BayesCode/examples/` if
   the user explicitly asks "add this as a library example".
4. **Path to AI4BayesCode folder.** **If you are running as the installed `/AI4BayesCode` skill,
   do NOT ask** -- the library is bundled at the skill root (the directory `start.md` / `SKILL.md`
   was loaded from), so RESOLVE the path to that root automatically and use `<skill-root>/include`
   for the compile `-I` (substitute the actual absolute path for the bare `AI4BayesCode/` prefix in
   every compile recipe). Only ASK when NOT running from the installed skill (a local checkout):
   then default `./AI4BayesCode`. Reuse from earlier in the session if already known. (FUTURE: an
   R / Python package install resolves this even more cleanly -- `system.file("include", package =
   "AI4BayesCode")` / the Python package path -- no question at all.)
5. **Model description** (if not already given; normally the model is already captured +
   confirmed at Step 0 above).

6. **Sampler engine.** Engine-level choice between MCMC and Variational
   Inference (VI). VI is an opt-in alternative -- MCMC is the safer
   default; VI shines when MCMC fails (high-dim with sign/permutation
   symmetry, BNN, large embedding spaces) AND the user accepts
   point-estimate-of-q semantics (biased CIs in exchange for tractable
   speed). Ask this via `AskUserQuestion` BEFORE the sampler/block
   preference question below, so picking pure VI can skip the MCMC
   block elicitation entirely:

   > "Sampler engine for this model:"
   >
   > Options:
   > - **(a) Let codegen recommend after seeing the model spec (default)**
   >   -- codegen inspects parameter dims / symmetries after the model is
   >   spec'd and recommends MCMC vs VI vs hybrid. Most users should pick
   >   this; codegen surfaces a recommendation with explanation at Sec.X
   >   (VI sub-flow) and the user can accept or override.
   > - **(b) MCMC for all parameters** -- proceed straight to the
   >   sampler/block preference question and standard MCMC code gen.
   > - **(c) VI** -- codegen will follow up at Sec.X (VI sub-flow) on
   >   pure vs hybrid, which parameters get VI, and the per-block VI
   >   family (mean-field / full-rank / categorical / structured).

   **Conditional follow-ups based on this answer:**
   - **(a)** -> question #7 below (sampler/block preference) fires
     normally. Sec.X VI sub-flow runs after model confirmation and may
     still recommend VI based on the spec.
   - **(b)** -> question #7 fires normally. Sec.X VI sub-flow is skipped
     entirely.
   - **(c)** -> question #7 is SKIPPED (MCMC block prefs are
     irrelevant for pure VI; hybrid follow-up at Sec.X handles which
     params get VI and which stay MCMC).

7. **Suggested sampling algorithm or preferred blocks** (optional).
   **CONDITIONAL: only fire this question if the user picked (a) or (b)
   in #6 above.** Some users have a strong preference for which block
   family handles a particular parameter (e.g. "use
   `joint_nuts_block` for the hierarchical scale + raw-effect
   pair", "use `pg_logistic_block` instead of NUTS-on-beta for the
   logistic regression coefficients", "wrap the discrete state sequence
   with `hmm_block`"). Offer two choices via `AskUserQuestion`:
   - **(a) Skip** -- let the skill pick blocks via its standard
     decision rules (`codegen_priors.md Sec.2b` block-selection priority +
     Sec.3a discrete-variable decision tree). This is the default for
     most users; the recommended block menu is documented and almost
     always optimal.
   - **(b) I have a suggestion** -- accept a free-form text answer
     describing the preferred algorithm / block(s) per parameter. The
     skill MUST then honor that preference in code generation, EXCEPT
     when it would violate a hard validator rule (Check #17
     "no hand-written Gibbs", `system_design.md` Sec.11.2 "what NUTS
     cannot sample", etc.). If the user's preference conflicts with
     a hard rule, surface the conflict in an `AskUserQuestion`
     follow-up before proceeding.
8. **Maximum generation attempts.** Cap on the
   generate->validate->fix loop. Without a cap, hard models can burn
   unbounded user time / tokens / compute. Ask this with the
   upfront setup batch (alongside backend / folder / class name / path),
   BEFORE code generation -- the model was already confirmed at Step 0, so
   all of #1-#8 are free to ask now (batch by the tool's 4-per-call cap):

   > "Maximum number of full generate->validate attempts before I stop
   > and report? **The first generation counts as attempt 1** (so 5 =
   > the initial generation + at most 4 fixes). The validator's internal
   > R-hat budget escalation does NOT count as an attempt. Default 5."
   >
   > Options: 5 (default) / 10 / 20 / Unlimited (opt-in)

   Counting semantics -- state these to the user if asked, and follow
   them exactly:

   - **One "attempt" = one full `generate -> validate` cycle** (code
     generation + compile + Layer-3 R1/R2/R3 + Checks #1-20).
   - **The first generation IS attempt 1** -- it is NOT a free or
     uncounted "attempt 0". The counter starts at 1 on the initial
     generation. `max_attempts = N` therefore means at most **N total**
     cycles INCLUDING the first generation (N = 5 => initial generation
     + at most 4 regenerations).
   - **The validator's internal MCMC-budget escalation** (R2 R-hat
     4k+4k -> 20k+20k re-run) is part of the CURRENT attempt and does
     NOT increment the attempt counter. Only a full regenerate
     increments it.
   - **On reaching the cap** (attempt N has been generated and still
     fails validation): STOP. Do not silently continue and do not
     hard-error. Report: the best attempt so far, the exact validator
     failures (which checks / which layer), and a one-paragraph
     diagnosis of what is failing and the likely cause, so the user can
     decide whether to raise the cap or revise the model.
   - "Unlimited" is opt-in only; default is the finite value so
     unbounded grinding never happens by accident. Treat silence /
     no answer as the default (5).

   Do not label any option "(recommended)"; the standing value is the
   "(default)".

The backend choice affects ONLY the `.cpp` module-declaration section
and the runner file -- the log-density, block assembly, and DAG
declaration are identical across all backends. The AI4BayesCode library
headers return backend-neutral types (std::unordered_map, struct
`dag_t`) which both Rcpp and pybind11 auto-convert to their native
types.

---

## 2. Prior specification -- overview

For each parameter in the model, the user may or may not specify a prior.
When a prior is missing, use AskUserQuestion to offer choices.

**MANDATORY -- every prior-related AskUserQuestion call MUST include a
"Literature-informed" option** (search the domain literature, or use
LLM training knowledge when web search is unavailable, to propose
informative hyperparameters). This applies to Case 2 (specified
family, missing hyperparameters), Case 3 (no prior at all), and any
sub-decision in `codegen_priors.md` that asks the user to pick between
default / alternative / informative priors. Users should never have to
ask "can you also try a literature-based prior?" -- that option is
always on offer. See `codegen_priors.md` "Literature-informed prior
elicitation" for the elicitation workflow.

**This section is an overview; load `codegen_priors.md` for the full
details** (scale / variance discipline per Gelman 2006, block-selection
priority, Gibbs-block Checks #15-17, discrete-variable decision tree,
literature-informed elicitation, BART sigma override).

### Step 0 -- prior completeness routing (model structure already confirmed at the Sec.1 Step-0 gate)

The model structure + any priors the user already gave were rendered and CONFIRMED at the
**Step 0 model-structure confirm** at the top of Sec.1 (before any upfront question). So Sec.2 does
NOT re-render or re-confirm the structure -- its only job is to COMPLETE the missing priors.
Route on completeness:

- **Spec already COMPLETE** (every prior specified -- nothing left to elicit): skip the entire
  Decision flow below and jump straight to the Sec.3 model-confirmation gate (full LaTeX + DAG +
  summary table, signed off in one place). Do NOT run the Sec.2 elicitation and do NOT make the
  user confirm twice.
- **Anything missing:** run the Decision flow below to elicit each missing prior (the ones
  marked `(prior: to be chosen)` at the Sec.1 gate). No need to re-confirm the structure -- that
  is already done. The full spec WITH all priors is then signed off at the Sec.3 gate.

(Two-gate pattern: gate 1 = model structure, at Sec.1 *before* the engine question; gate 2 =
full spec with priors, at Sec.3.)

### Decision flow per parameter

1. **User specified a prior with hyperparameters** (e.g. `sigma ~ HalfNormal(0, 10)`)
   -> Use it directly. Hardcode the hyperparameter values in the log-density.

2. **User specified a prior family but no hyperparameters** (e.g. `sigma ~ HalfNormal`)
   -> Offer three choices via AskUserQuestion:
   - **(a) Default hyperparameters** -- use the defaults from the table below,
     hardcoded in the `.cpp`.
   - **(b) Expose as constructor arguments** -- pass the hyperparameters
     into the class constructor so the user can tune them from R without
     recompiling.
   - **(c) Literature-informed hyperparameters** -- search the literature
     for domain-relevant effect sizes (or use LLM training knowledge if
     web search is unavailable) and translate them into informative
     hyperparameter values. See "Literature-informed prior elicitation"
     in `codegen_priors.md`.

3. **User did not specify a prior at all** -> Offer four choices:
   - **(a) Fixed value** -- treat the parameter as a known constant (not
     sampled). Pass it as a constructor argument. Do NOT create a block
     for it.
   - **(b) Default weakly informative prior** -- pick from the table
     below, hardcoded.
   - **(c) Literature-informed prior** -- search the literature for
     domain-relevant effect sizes and translate them into informative
     priors (see Literature-informed prior elicitation in
     `codegen_priors.md`).
   - **(d) Ask user to provide** -- request the prior family and
     hyperparameters before proceeding.

### Recommended default priors

| Parameter type | Default prior | Hyperparameters |
|---------------|--------------|-----------------|
| Location (mu, beta) | `Normal(0, 100)` | mean=0, sd=100 |
| **Scale (sigma, tau) -- DEFAULT** | **Jeffreys: `p(sigma) prop.to 1/sigma`** | **none (scale-invariant)** |
| Scale -- k=0 transient guard | `half-Normal(0, 1)` pin (inline fallback) | (dimensionless reference) |
| Scale -- genuinely sparse (k stays < 5) | `half-Normal(0, A)` | A ~= 2.5 x data-scale |
| Precision (1/sigma^2) | DO NOT default to `Gamma(epsilon, epsilon)` -- use Jeffreys on sigma | -- |
| Probability (p) | `Beta(1, 1)` | a=1, b=1 (uniform) |
| Simplex (theta) | `Dirichlet(1, ..., 1)` | alpha=1 (uniform) |
| Concentration (kappa, theta) | `Gamma(1, 1)` | shape=1, rate=1 |
| Correlation matrix | `LKJ(2)` | eta=2 |
| Count rate (lambda) | `Gamma(1, 1)` | shape=1, rate=1 |
| Regression coeff | `Normal(0, 10)` | mean=0, sd=10 |

Tell the user which defaults you chose and suggest they adjust if they
have domain knowledge. See `codegen_priors.md Sec.2a` for the variance-
prior rationale.

**Before writing any code, load `codegen_priors.md`** if:
- The model has a scale/variance parameter (Sec.2a).
- You need to pick a block type (Sec.2b).
- Gibbs discipline matters (Sec.2c-e).
- There is any discrete latent variable (Sec.3a).

---

## 3. Model confirmation

After parsing the model and resolving all priors, ask the user via
AskUserQuestion **before generating any code**:

> "I've parsed your model and resolved all priors. Before I generate
> code, would you like to review a summary (formulas, DAG, and parameter
> table) to make sure everything is correct?"
>
> Options:
> - **Yes, show me the model summary first** (default)
> - **No, go ahead and generate directly**

This prompt is important: without it, users may think code generation
has already started and miss the confirmation step entirely.

### Delivered-code validation harness (ask in this same pre-generation round)

Alongside the summary-review question above, also ask whether the
runtime-validation harness should be SHIPPED in the delivered
code. The generated R runner currently carries a large Layer-3
harness (R-hat/ESS, posterior-predictive p-values, PSIS-LOO, smoke
checks); for a first-time user that scaffolding obscures the simple
"how do I use this" path.

Ask:

> "Include the runtime-validation harness (R-hat/ESS, posterior-
> predictive p-values, PSIS-LOO, smoke checks) in the delivered code?
> Validation runs during generation EITHER WAY -- this only controls
> whether the harness ships in the final code."
>
> Options:
> - **No -- deliver a usage example without the validation harness (default)**
> - **Yes -- ship the full validation runner**

**CORRECTNESS INVARIANT -- read this exactly; it is not optional.**
The validator (Layer-3 R1/R2/R3 + Checks #1-20) **ALWAYS runs and
must PASS during generation, regardless of this answer.** This
question controls ONLY what is *shipped* in the final deliverable,
NEVER whether the validator runs. "No (default)" does **NOT** mean
"skip validation" -- it means: run it, gate on it, and on PASS
deliver a clean minimal example *instead of* the validation runner
(the runner becomes a throwaway, exactly like the Step-3 smoke test
and the Sec.6b autodiff-verify file). A future codegen agent that reads
"default: no validation" as "do not validate" has broken the
system's core guarantee.

Deliverable on PASS:
- **No (default):** the production `<ClassName>.cpp` (which already
  contains zero validation code) **plus a usage example**
  `example_<ClassName>.R`. This is NOT a stripped snippet -- it is
  the R runner content **up to (excluding) the first two-chain
  diagnostic chain** (`c1 <- run_chain_<ClassName>(... seed=101 ...)`),
  WITH comments retained: header comment, `source(AI4BayesCode
  helpers)` + `ai4bayescode_sourceCpp(...)`, the constructor
  reference block, the **full `run_chain_<ClassName>()` definition**
  (WITH its `diagnosis = FALSE` parameter, whose diagnosis branch CALLS
  the shipped `ai4bayescode_diagnose()` -- the diagnostics table AND
  the trace+ACF+density plot are a library function, NOT a hand-emitted
  helper),
  simulation/toy-data code, a monolithic non-stateful
  `run_chain_<ClassName>()` call PLUS a `diagnosis = TRUE` call that
  exposes `mono_chain$diagnosis` and `mono_chain$diagnosis_plot`, and
  commented stateful-API usage
  (`new(..., keep_history=TRUE)`, `step`, `get_history`,
  `get_current`, `set_current(list(...))`, `predict_at(list(...))`).
  Everything from the first diagnostic chain onward (R1/R2/R3,
  R-hat/ESS, BPV, PSIS-LOO) is the throwaway Layer-3 harness and is
  DELETED on PASS (same rule as the smoke test: "Delete on PASS. No
  exceptions."). Exact template: `codegen_r_runner.md` "Usage-example
  template".
- **Yes:** ship the full `run_<ClassName>.R` Layer-3 runner
  (the `codegen_r_runner.md` template) as today.

**RUNNER DIAGNOSIS GATE (hard -- applies to BOTH deliverables).**
Codegen LLMs reliably write their OWN `run_chain` (a `clear_history()` /
`get_current()` loop, a custom return shape) instead of copying the
template, and in doing so they drop the diagnostics. The diagnostics +
plot are now a SHIPPED library function -- `ai4bayescode_diagnose()`
(R) / `AI4BayesCode.diagnose()` (Python) -- so the runner just CALLS
it and must NOT reimplement it. The recurring bug is a bespoke runner
with an inline `posterior::summarise_draws(...)` / `az.summary(...)`
assigned to `$summary` (or `hist$diagnosis`) with **NO plot**. This is
NON-COMPLIANT. Before delivering the example file you MUST verify the
contract -- run:

```bash
# R:
grep -q 'ai4bayescode_diagnose' example_<ClassName>.R && grep -q 'diagnosis_plot' example_<ClassName>.R
# Python:
grep -q 'ai4bayescode_diagnose' example_<ClassName>.py && grep -q 'diagnosis_plot' example_<ClassName>.py
```

If either grep fails, the runner is non-compliant -- REWRITE it so it:
(1) CALLS `ai4bayescode_diagnose()` (R) / `AI4BayesCode.diagnose()`
(Python) -- never an inline reimplementation; (2) takes `diagnosis = FALSE`;
(3) attaches BOTH `$diagnosis` (the returned summary) AND `$diagnosis_plot`
(the returned plot). This is INDEPENDENT of how the runner collects draws
-- whatever named draw list you built (`get_history` slice,
`clear_history()`+loop, or `get_current()` loop), pass THAT to the library
function, e.g. `ai4bayescode_diagnose(list(beta = beta, sigma = sigma))`.
Do NOT deliver until both greps pass.

On FAIL within `max_attempts`: stop-and-report (per "Generation
attempt budget" above) -- do NOT silently ship the usage example.
The minimal-example deliverable is produced ONLY on a validator
PASS. Treat silence / no answer as the default (No / minimal
example). Orthogonal to `max_attempts` (that caps attempts; this
governs only what a passing run delivers).

If the user picks "Yes" (or does not respond -- treat silence as "Yes"),
present the confirmation below. If "No", skip to code generation but
still print a one-line summary of what you understood (e.g. "Generating
a 3-block sampler: mu (real), sigma (positive), theta (simplex)...").

The confirmation has three parts:

### (a) LaTeX formulas

Write the full model as math in a fenced block. Use standard notation:

```
$$
\begin{aligned}
y_i \mid \mu, \sigma &\sim \text{Normal}(\mu,\, \sigma^2), \quad i = 1, \ldots, N \\
\mu &\sim \text{Normal}(0,\, 100^2) \\
\sigma &\sim \text{Half-Normal}(0,\, 10)
\end{aligned}
$$
```

### (b) DAG

**MANDATORY -- the DAG shown to the user MUST be a PREDICTION DAG
(generative / causal direction), NEVER a Gibbs / dependency DAG.**

An edge `A -> B` means "A produces B in the data-generating story",
NOT "block B reads A during MCMC sampling". The two DAGs are
distinct and confusing them is a common failure mode (see
`system_design.md` Sec.4). Practical rules:

- The edges drawn here MUST match the `declare_predict_edges` calls
  emitted in the generated C++. They do NOT correspond to
  `declare_dependencies` (that's the Gibbs DAG, internal to the
  sampler).
- Sibling priors that are independent in the generative story have
  **NO edge between them**, even if their Gibbs full conditionals
  read each other. Counter-example to internalize: in spike-and-slab
  Laplace with `beta_j | gamma_j=1, tau^2_j, sigma ~ N(0, sigma^2tau^2_j)` and
  `tau^2_j | b ~ Exp(1/(2b^2))`, the predict DAG has
  `pi -> gamma`, `b -> tau^2`, and `gamma, tau^2, sigma -> beta`, plus `beta, sigma, X -> y`.
  sigma and tau^2 are **siblings** (both parents of beta), with **no sigma<->tau^2
  edge**, even though the sigma-block's Gibbs conditional reads tau^2 and
  vice versa via the slab term sigma^2tau^2.
- Deterministic derived quantities (e.g. `linear_predictor = X beta`,
  `y_rep`, `f_bart`) are nodes with incoming arrows from each input.
- Root nodes are unconditioned RVs (top-level priors with no
  hyperparents) and observed data inputs (X).

If you find yourself about to draw an arrow because "block A's
log-density reads B" or "B's full conditional depends on A" -- STOP.
That is a Gibbs-DAG edge and does NOT belong in this picture.

Render the DAG **inline in chat** using a text-based format. DO NOT
spawn R / igraph / Graphviz / PNG rendering -- that wastes tokens
and time. One of the following is sufficient for user verification:

1. **Mermaid code block** (preferred if the client renders it):

   ````
   ```mermaid
   flowchart TD
       pi --> gamma
       b --> tau2
       gamma --> beta
       tau2 --> beta
       sigma --> beta
       X --> y
       beta --> y
   ```
   ````

2. **Graphviz `dot` source** in a code block (clients without
   Mermaid often render `dot`):

   ````
   ```dot
   digraph G {
       pi -> gamma;
       b -> tau2;
       {gamma, tau2, sigma} -> beta;
       {X, beta} -> y;
   }
   ```
   ````

3. **Edge list** (plain text, always works):

   `pi -> gamma;  b -> tau^2;  gamma, tau^2, sigma -> beta;  X, beta -> y`

Conventions:

- **Latent params vs observed data:** annotate observed nodes
  (e.g., `y[shape=circle, style=filled]` in `dot`, or `y["y (observed)"]`
  in Mermaid). Edge-list: append `(observed)` after the data node.
- **Plates:** annotate "i = 1..N" near repeated nodes (text label
  is fine; no need for a plate-box render).
- **Priors:** NOT in the DAG; they live in the summary table (c) below.
- **Edges:** generative / causal direction ONLY (per the MANDATORY
  rule above -- prediction DAG, not Gibbs DAG).

### (c) Summary table

List all parameters, data, and fixed constants:

| Name | Type | Prior / Value | Block type | Hyperparams |
|------|------|--------------|-----------|-------------|
| y | data | observed | -- | -- |
| X | data | observed | -- | -- |
| mu | param | Normal(0, 100) | nuts_block (real) | hardcoded |
| sigma | param | HalfNormal(0, sd) | nuts_block (positive) | `sd` exposed |
| nu | fixed | 3 | -- | -- |

The Hyperparams column clarifies whether prior hyperparameters are
`hardcoded` to specific values or `exposed` as constructor arguments
the user can tune from R.

### Confirmation prompt

This is the final sign-off gate before code generation. The model STRUCTURE was already
confirmed at the **Step 0 model-structure confirm** (top of Sec.1, before any upfront question), so
don't re-derive it here -- this gate confirms the COMPLETE spec now that priors are filled in
(with the DAG + summary table). If priors were already complete (no Sec.2 elicitation ran), this is
simply the next gate after Step 0; if priors were just elicited in Sec.2, focus the user on the
priors + wiring
added since the Sec.1 confirm. After presenting all three, ask the user:

> "Does this match your intended model? If anything is wrong -- priors,
> likelihood, parameter constraints, or dependencies -- tell me and I'll
> fix it before generating code."

**Do NOT proceed to code generation until the user confirms.**

### VI sub-flow (fires only if Sec.1 #6 picked (a) or (c))

This sub-flow runs **right after model confirmation** and BEFORE
block-specific follow-ups. Fires when EITHER:

- Sec.1 #6 picked **(c) VI** -- user committed upfront; codegen confirms
  pure-vs-hybrid + per-block VI family + clique partition (if any
  `structured_categorical`), and SKIPS the block-specific follow-ups
  for VI parameters.
- Sec.1 #6 picked **(a) Let codegen recommend** AND codegen sees a
  model structure where VI is appropriate (e.g. a parameter with
  dim > 100 + known symmetry: BNN weights, embedding, deep model;
  or a tightly-coupled continuous parameter group that would need
  full-rank metric estimation under joint NUTS). If codegen decides
  MCMC is better, this sub-flow is SKIPPED.

If Sec.1 #6 picked **(b) MCMC**, this sub-flow does NOT run.

**Always print the disclosure caveats first** (Bishop Sec.10.1.2 /
Blei 2017, never skipped):

- Mean-field VI **underestimates posterior variance** -- every
  credible interval reported is biased small. If you need honest
  CIs on a specific low-dim parameter, keep that parameter on
  MCMC (hybrid mode) or revert to all-MCMC.
- Exclusive KL is **mode-seeking** -- multimodal posteriors yield
  ONE mode. If the model has known multimodality (mixtures,
  permutation symmetry, BNN sign/permutation symmetry), VI
  returns ONE mode and reports it cleanly. Feature for
  prediction, NOT the full posterior.
- Label-switching behavior differs from MCMC -- VI on a mixture
  converges to ONE component assignment per seed; MCMC bounces
  between them. Both are diseases of underlying non-identification.
- Hybrid convergence: no joint convergence criterion in v1 -- the
  runner just runs the user's outer iteration budget and reports
  per-child diagnostics. If hybrid mixing is suspect, fall back
  to all-MCMC or pure VI.

**Layer-3 diagnostic implications**: VI blocks use joint PSIS-k-hat
(< 0.7) instead of R-hat (Yao 2018, Dhaka 2021). Hybrid samplers
report BOTH -- R-hat on MCMC children, k-hat on VI children, plus an
ELBO trajectory plot. See `validator.md` for the VI-specific
portion of Layer-3 R2.

---

#### Step 1: Pure vs hybrid + parameter assignment

**`AskUserQuestion` (REQUIRED):**

> "VI scope for this model:"
>
> Options:
> - **(a) Pure VI** -- every parameter goes through a VI block. Fastest,
>   uniform diagnostic (joint PSIS-k-hat). Use if you want a clean
>   point-estimate-of-q for the whole model.
> - **(b) Hybrid: VI for some parameters, MCMC for the rest** -- use VI
>   on tractable high-dim continuous blocks (often the bulk of
>   parameters) and keep MCMC for low-dim parameters where honest
>   CIs matter. After picking this, a multi-select question follows.

**If Hybrid**: immediately fire a `AskUserQuestion` with
`multiSelect: true` listing every parameter declared in the model.
Suggest as default-checked the parameters with dim > 100 OR known
to suffer symmetry (BNN hidden-layer weights, embedding parameters,
GP latent functions). Show each parameter's declared dim and a
one-line hint of why VI may help (e.g. "alpha (M=784 x J=10 =
7840-dim, BNN hidden-layer symmetric)").

**If Pure VI**: every parameter becomes a VI block. Skip the
multi-select.

---

#### Step 2: VI block assignment (codegen automatic)

**Step 2a -- coupling merge (FIRST; mirrors the MCMC `joint_nuts_block`
rule).** BEFORE assigning per-parameter blocks, GROUP tightly-coupled
continuous parameters into ONE VI block over their concatenated
unconstrained vector. Do NOT emit a separate VI block per parameter and
couple them through the composite. Two continuous parameters are
"tightly coupled" when they jointly shape the likelihood/posterior:
- regression coefficients `beta` and the noise scale `sigma^2` (beta's marginal
  variance inflates with sigma^2 uncertainty);
- collinear / correlated coefficient groups;
- additive linear-mean components, fixed + random effects;
- a hierarchical scale and the parameters it scales.

Emit them as a SINGLE `full_rank_gaussian_vi_block` over
`(beta, log sigma^2, ...)` when the merged dim <= 50. This is the VI analog of
"tightly-coupled real params -> `joint_nuts_block`, do NOT split".

> **WHY this is not optional.** Splitting coupled continuous params into
> separate VI blocks (`q(beta)*q(sigma^2)` coupled by composite q-sample passing)
> is a TRAP. The cross-block mean-field factorization cannot represent
> the joint posterior covariance, so it UNDERESTIMATES marginal variance
> -- often badly (measured: ~66% too small on a collinear coefficient
> pair; conjugate-NIG linear-regression demo). Worse, the per-block
> PSIS-k-hat is a FALSE reassurance: it scores fit to each block's
> CONDITIONAL (given a sibling q-sample), NOT the joint marginal, so a
> too-narrow q can show a low k-hat while its variance is wrong. A single
> joint block removes the coupling noise AND gives ONE well-defined
> joint k-hat (the only honest VI diagnostic for coupled params).

**Merge-dim fallback**: merged coupled dim > 50 -> single
`mean_field_gaussian_vi_block` over the concatenated vector (still ONE
block, ONE joint k-hat -- strictly better than splitting) and WARN that
mean-field underestimates the off-diagonal coupling; dim > 500 forces
mean-field regardless.

**Step 2b -- per-parameter assignment** for the remaining INDEPENDENT
parameter groups (and to pick the family for a merged group), codegen
picks a VI block by rule (do NOT ask the user 4-way "which VI family?" --
surface a single confirmation in Step 3):

| Parameter kind | Dimension / coupling | VI block |
|---|---|---|
| Continuous, low-dim with known correlation (regression with collinear predictors, ARMA `(phi, theta)`, GP hyperpriors, hierarchical funnel) | dim <= 50 | `full_rank_gaussian_vi_block` |
| Continuous, low-dim without expected correlation | dim <= 50 | `mean_field_gaussian_vi_block` |
| Continuous, medium-dim | 50 < dim <= 200 | `mean_field_gaussian_vi_block` (full-rank too expensive; warn user) |
| Continuous, high-dim | dim > 200 | `mean_field_gaussian_vi_block` (forced -- full-rank infeasible) |
| Discrete latent (`z_i  in  {0..K-1}`), no known coupling structure | any | `mean_field_categorical_vi_block` |
| Discrete latent, HMM / Markov-chain topology detected from prompt | any | `structured_categorical_vi_block` + sliding-pair clique default |
| Discrete latent, user has indicated explicit coupling clusters (CRF / Ising / spatial graph) in the prompt | any | `structured_categorical_vi_block` + user partition |
| Discrete latent, other graphical structure but no partition info | any | `mean_field_categorical_vi_block` (safe fallback) |

Reject `full_rank_gaussian_vi_block` for dim > 500 outright -- switch
to mean-field and warn.

---

#### Step 3: Assignment confirmation

**`AskUserQuestion` (REQUIRED):**

> "VI block assignment (codegen recommendation):
>
> > `<param_1>` (`<dim>`-dim, `<hint>`)
>     -> `<block_choice>`
>     [<extra info, e.g. clique partition for structured>]
> > `<param_2>` ...
>     ...
>
> Options:
> - **(a) Accept (default)**
> - **(b) Customize** -- type changes per parameter. Examples:
>   - `'phi full_rank'` to switch block family
>   - `'z structured cliques=[{0,1},{2,3},...]'` to override partition
>   - `'z mean_field_categorical'` to drop clique structure entirely"

If the user picks (b) and the override produces an invalid
configuration (e.g. full_rank on dim > 500, or clique partition
that doesn't cover all latents), surface the issue in a follow-up
question rather than silently fall back.

---

#### Step 4: Clique partition (REQUIRED if any `structured_categorical_vi_block`)

For **every** parameter that was assigned `structured_categorical_vi_block`
(in Step 2 default OR Step 3 override), fire a dedicated
`AskUserQuestion`. **Never auto-derive without asking**, even when
the model is "obviously" HMM.

Use the variable's own name (e.g. `z`) and subscript notation
(`z_t`) -- NOT raw 0-based indices like `{0,1}` which are unreadable.
Per-parameter template:

> "Clique partition for `structured_categorical_vi_block` on `<param>`
> (`<N>` values, each in `{0..K-1}`).
>
> VI approximates the joint posterior -- the trade-off is how MUCH it
> tries to capture correlation between neighboring `<param>` values.
>
> Options:
> - **(a) Adjacent pairs [default -- for HMM / sequential models]** --
>   keep correlation between `<param>_t` and `<param>_{t+1}` (first-
>   order Markov). VI captures two-step joint distributions; overlapping
>   pairs are factorised.
> - **(b) Adjacent triples** -- sharper, captures 3-step joint. ~2x
>   slower per step than (a).
> - **(c) Disjoint pairs of 2** -- cheaper than (a) but misses
>   correlation across boundaries (`<param>_2` and `<param>_3`
>   independent under `((<param>_1, <param>_2), (<param>_3, <param>_4))`).
> - **(d) Disjoint blocks of 4** -- larger blocks -> more accurate
>   per-block but cost grows like K^4.
> - **(e) Treat `<param>`s as independent** -- switches back to
>   `mean_field_categorical_vi_block`. Fastest; use if you believe the
>   `<param>` values truly don't influence each other.
> - **(f) Custom partition (advanced)** -- paste as
>   `<param>_1+<param>_2+<param>_5, <param>_3+<param>_4, ...`.
>   Each comma-separated group is one clique. Indices are positions in
>   the `<param>` array, starting from 1."

The default (a) sliding pairs covers HMM / state-space / sequential
models. For CRF / Ising / spatial discrete fields where adjacency is
non-trivial, the user must use (f) custom -- codegen cannot guess the
graph.

If the user's choice produces an invalid partition (e.g. indices
out of range, latents not covered), reject with a follow-up question
explaining the issue. NEVER silently fall back to a different
partition.

### Block-specific follow-up questions

**CONDITIONAL: skip entirely for parameters going through a VI block
in pure-VI or hybrid mode** -- BART DART, spike-slab spike-family, and
similar block-specific knobs apply only to MCMC blocks. In hybrid
mode, fire these questions only for parameters that remained on MCMC.

Some block types expose optional modes that the user may not have
mentioned explicitly but would likely want. After the model is
confirmed, walk through the list below and **issue an
`AskUserQuestion` tool call** for each that applies to the model
structure. Do not list the choices inline; the tool call provides
the structured picker UI. Three firm rules:

1. **Do NOT silently enable optional modes** -- always ask.
2. **Do NOT expose an optional flag as a constructor argument unless the
   user opts in.** If the user says "no, I don't need DART", the
   generated C++ constructor should simply hardcode `dart = false,
   aug = false` inside the class body and not surface those parameters
   to R at all. This keeps the generated interface minimal. If the user
   later changes their mind, they regenerate.
3. **Spike-and-slab models trigger a required clarifying question
   about the spike family** -- see the dedicated bullet below.

- **BART / genBART -- sparse variable selection (DART, Linero 2018).**
  If the model has a `bart_block` or `genbart_block` AND the user
  mentioned anything like "variable selection", "which features
  matter", "feature importance", "sparse", or the predictor matrix X
  has moderately large p (say p >= 10), AskUserQuestion whether to
  enable DART sparsity. Two outcomes:
  - **Yes, enable DART** -> for `bart_block`, expose `dart` and `aug`
    as constructor args; for `genbart_block`, set
    `cfg.hypers.dart_active = true` (and optionally
    `cfg.hypers.dart_const_theta = false` to let theta adapt). The DART
    option is off by default in both; tell the user in one line what
    it does (Linero 2018 Dirichlet prior on split-variable
    probabilities, shrinks unimportant features toward zero split
    probability).
  - **No, don't enable** -> leave DART flags at their defaults (off).
  See `skills/block_catalogue/index.md` (`bart_block` / `genbart_block`
  sections) for the recipe.
- **Spike-and-slab -- Dirac vs normal-approximate spike (REQUIRED).**
  If the user describes any "spike-and-slab" / "variable selection
  with inclusion indicator" / "sparsity prior with inclusion
  probability" model and has not specified the spike family,
  **`AskUserQuestion` MUST fire** with these three choices before
  any code is written:
  - **(a) Dirac spike** -- `beta_j` is exactly 0 when `gamma_j = 0` (point
    mass at zero). State space is dimension-changing. Implementation
    routes through `rjmcmc_block` with a `continuous_update` hook;
    reference template `examples/SpikeSlabRJMCMC.cpp`. Use when the
    user wants exact zero coefficients and posterior inclusion
    probabilities. See `codegen_priors.md` Sec.3a Class 2b.
  - **(b) Normal-approximate (Gaussian-mixture) spike** -- `beta_j |
    gamma_j = 0 ~ N(0, tau_0^2)` with a very tight `tau_0` (e.g. 1e-3),
    `beta_j | gamma_j = 1 ~ N(0, tau_1^2)`. Fixed-dim continuous relaxation;
    gamma updated by `binary_gibbs_block`, beta by `nuts_block` with the
    Gaussian-mixture prior. Use when the user is OK approximating
    the point mass with a tight Gaussian (mixing is dramatically
    easier; coefficients never reach exactly zero). See
    `codegen_priors.md` Sec.3a Class 2a.
  - **(c) Continuous shrinkage (horseshoe / DL / R2-D2)** -- no
    inclusion indicator; pure continuous prior with heavy tails plus
    a spike near zero. Use when the user actually wants shrinkage
    without exact zeros or per-coefficient inclusion probabilities.
    See `examples/ARDLasso.cpp` and the horseshoe variant.
  Never silently pick one -- every spike-and-slab generation must
  resolve this question via `AskUserQuestion` first.
- (Add future optional-mode triggers here as the block catalogue
   grows -- driven by the block's documented flags, not a hardcoded
   list in this skill. Same "only expose when opted in" principle
   applies to every new flag, and the same "use `AskUserQuestion`,
   never an inline bullet list" rule applies to every new
   clarifying question.)

---

## 4. Workflow

1. **Parse the model into parameter blocks.** Classify each parameter's
   support and pick a block type. See `codegen_priors.md Sec.2b` for the
   block selection priority (specialized / conjugate-Gibbs blocks by
   applicability -> `joint_nuts_block` = DEFAULT for continuous ->
   `nuts_block` = LOW-priority single fallback -> slice -> `Sec.2c`
   Exception 4 custom code only for genuine structural gaps). If the
   model has any discrete latent, run through `codegen_priors.md Sec.3a`
   (Class 1-5 decision tree) FIRST. Reference block catalogue:
   `skills/block_catalogue/index.md`.
2. **Coupling analysis.** See `codegen_cpp.md Sec.4a`. **Default is JOINT:**
   collect the continuous parameters not claimed by a specialized block
   into ONE `joint_nuts_block` -- you write the joint natural-scale
   log-density (likelihood + all priors + per-slice constraints) and the
   block provides the NUTS machinery. A standalone `nuts_block` is the
   low-priority fallback (genuinely scalar params / post-NCR branch /
   deliberate isolation). Do NOT contort the model to fit a built-in
   block; if none fits AND NUTS is structurally inapplicable, `Sec.2c`
   Exception 4 permits a justified custom block.

   **Pattern triggers for MANDATORY joint sampling -- read the linked
   skill in full BEFORE generating code:**

   - **Hierarchical random effects** (`u_i ~ N(mu_u, tau)` with
     group-level `mu_u`, `tau`, regardless of likelihood family):
     read `skills/hierarchical_re.md`. NEVER use separate
     `nuts_block`s for `(mu_u, tau, u_i)` -- funnel geometry causes
     silently wrong posterior (cov_AI ~= 0.07-0.75 vs cov_REF ~= 0.94
     in the radon_*, surgical_model failures).
   - **LDA / topic-model** (`z_n` discrete, `theta`, `phi` Dirichlet):
     read `skills/block_catalogue/index.md` `lda_collapsed_gibbs_block` Sec..
3. **Resolve priors** for each parameter using the decision flow in
   Sec.2 above + `codegen_priors.md` for details. Ask the user when
   needed. Do NOT silently pick priors.
4. **Ask if the user wants to review** (Sec.3 above). If yes, show the
   three-part confirmation and wait for approval.
5. **For each NUTS block, pick the constraint transform** -- see
   `skills/constraints.md`.
6. **Write one natural-scale log-density function per block.** This is
   the ONLY mathematical content. Use `constraints::<KIND>::wrap` --
   never hand-roll Jacobians. See `codegen_cpp.md Sec.6` for the template.
7. **Register the y_rep posterior-predictive stochastic refresher** for
   every block that has an observation layer. See `codegen_cpp.md Sec.6a`
   per-observation-family templates (Gaussian / Bernoulli / Poisson /
   Multinomial / BART-noise).
8. **Gradient verification (Check #12).** For every NUTS block with
   a hand-written gradient, write + run + DELETE a throwaway
   `tests_autodiff/verify_<ClassName>.cpp`. See `codegen_cpp.md Sec.6b`.
9. **Assemble the composite in the constructor.** See
   `codegen_cpp.md Sec.7`.
10. **Output the `.cpp` + `.R` runner.** See `codegen_r_runner.md`
    for the runner template + Layer 3 validator wiring.
11. **Verify (Sec.11 below) -- STRICT ORDER L1 -> L2 -> L3, no reversal.** Compile (L1), then the
    semantic validator checklist + AD-twin gradient check (L2), and ONLY AFTER L2 passes the
    runtime checks (L3: smoke test, 2-chain diagnostic). Do NOT write/run the 2-chain runner
    before L2 passes, and do NOT report "validation passed" off the runtime run alone -- see
    the **HARD ORDERING GATE** at the top of Sec.11. Delete smoke-test file on PASS.

---

## 11. Compile and run verification

**After generating the .cpp and .R files, you MUST verify them before
reporting success to the user.** Order is **L1 -> L2 -> L3** -- compile, then the
**semantic validator checklist (L2)**, then the **runtime checks (L3: smoke -> two-chain
-> posterior)** -- cheapest-to-most-expensive per `validator.md`. Run the STATIC L2 checks
BEFORE the expensive runtime: L3 passing does NOT prove correctness (a wrong Jacobian /
parallel-update / parameterization runs clean -- R-hat ~1.00, BPV in range, LOO k<0.5 -- yet
samples the WRONG posterior), so clear L2 first. Fix-and-re-run a layer before the next:

> **HARD ORDERING GATE -- L2 must PASS before you write or run the L3 runtime harness.**
> The steps below are an ORDER, not a menu. You may NOT write or run the two-chain /
> runtime R harness until BOTH L2 checks have PASSED: the **semantic validator checklist**
> AND the **AD-twin gradient-vs-autodiff comparison** (Check #12 -- a SEPARATE throwaway C++
> file, `codegen_cpp.md Sec.6b`). The L3 runtime harness is the **LAST** thing you build.
> The `codegen_r_runner.md` runner is a single file that does compile + smoke + two-chain in
> one shot -- that is exactly why it tempts you to run it first. Do not.
>
> **FORBIDDEN (the recurring reversal):** writing the validation runner, running the 4k+4k
> two-chain diagnostic, reporting *"runtime validation passed cleanly,"* and only THEN doing
> the L2 gradient check + static semantic audit. That is the order REVERSED. A wrong
> Jacobian/gradient passes L3 clean (R-hat ~= 1.00, BPV in range, LOO k < 0.5) while sampling
> the WRONG posterior -- so a *"runtime passed"* reported before L2 is a FALSE all-clear.
>
> **SELF-CHECK before you say "validation passed":** you must have run L1 -> L2 -> L3 in that
> order. If the status you are about to send would read *"runtime passed, still need the L2
> gradient check / semantic audit"* -- you have ALREADY violated this gate. STOP, run L2
> (checklist + AD-twin) NOW, fix anything it catches (re-run the affected layer), and report
> success ONLY once L2 AND L3 have both passed in order.

### Step 1: Compile check

Run `ai4bayescode_sourceCpp("<file>.cpp", ...)` and verify it compiles
without errors. If it fails, read the error, fix the code, and retry.

### Step 2: Validator checklist (L2 semantic -- run BEFORE the runtime checks)

After compile passes, run the full validator checklist in `skills/validator.md` (the
Semantic-layer checks: distribution parameterization, parallel-update bugs, dead
parameters, missing intercepts, Jacobian handling, DAG consistency, RNG separation, the
gradient-verification-via-autodiff Check #12, plus the conditional #19-#25). Any
failure must be fixed -- and the layer re-run -- before proceeding. **This is the layer that
catches the silent-correctness bugs the runtime checks CANNOT**, so it must be cleared
before the expensive L3 chains below. In particular the **AD-twin (Check #12)** -- the
autodiff-vs-hand-written gradient comparison -- is the cheapest correctness check and runs
HERE (gen-time / L2), BEFORE the Step 3-4 runtime: a wrong gradient caught by the twin saves
running the smoke + two-chain diagnostic on broken code (a chain on a wrong gradient can even
*look* like it converges).

### Step 3: Smoke test (THROWAWAY -- delete file on PASS)

Write a minimal smoke-test runner file at
`tests_autodiff/smoke_<ClassName>.R` (or inline in the `generated/`
folder when the repo layout doesn't have tests_autodiff). The smoke
test does:

- `model <- new(<ClassName>, ..., seed = 42L, keep_history = FALSE)`
- `model$step(10L)`
- `d <- model$get_current()` -- verify it returns a list with the
  expected keys, all values finite (no NaN, no Inf)
- `pp <- model$predict_at(list())` -- verify predict_at runs without
  error and `get_current()` is unchanged afterwards (state preservation
  per validator Check #10)

**On PASS, DELETE the smoke-test file.** The smoke test is a
throwaway -- it verifies the generation once, then it is gone. Ongoing
compile-smoke coverage is provided by the repo's F4 audit
(`tests_autodiff/audit_compile_smoke.R`), not by per-model smoke
scripts. Same pattern as Check #12 verify files in `codegen_cpp.md Sec.6b`.

Do NOT leave behind a `smoke_<ClassName>.R` or `smoke_test.R` with a
"kept for reuse" note. Delete on PASS. No exceptions.

### Step 4: Two-chain diagnostic (REQUIRED to run+pass at gen-time; shipping is governed by the validation-harness opt-in)

Run 2 chains from different initial values to check for
non-identifiability and basic convergence -- the Layer 3 R2 section of
the runner template in `codegen_r_runner.md`. Use a small synthetic
dataset (e.g. N=50-100) but long chains (4000 burnin + 4000 keep) to
avoid false positives from short runs.

**This is REQUIRED to RUN and PASS during generation regardless of
any user option** -- it is part of the correctness gate, not optional.
What is *conditional* is whether it is **shipped**: per the
"Delivered-code validation harness" question above,
- **default (No):** the Layer-3 runner (R1/R2/R3 incl. this Step-4
  diagnostic) is a **throwaway** -- it gates generation, then on PASS
  it is DELETED and only `example_<ClassName>.R` (the usage example
  defined in `codegen_r_runner.md`) is delivered. Same throwaway rule
  as the Step-3 smoke test.
- **opt-in (Yes):** the full `run_<ClassName>.R` is delivered with
  the Step-4 diagnostic retained.
(This supersedes the earlier "part of the runner, NOT throwaway"
wording -- R2/R3 are never skipped, but by default they are not
*shipped*.)

```r
chain1 <- run_chain_<ClassName>(data, init1, seed = 1L, n_burnin = 4000, n_keep = 4000)
chain2 <- run_chain_<ClassName>(data, init2, seed = 2L, n_burnin = 4000, n_keep = 4000)

# Check: do the chains agree on posterior means?
# Check: are all values finite?
# Check: is the posterior variance non-zero (not stuck)?
# If posterior::rhat is available, check R-hat < 1.05
```

**What to look for:**
- **R-hat > 1.5**: likely non-identifiability (NI) or a bug. Report to
  user with the specific parameter and suggest prior tightening.
- **Posterior sd = 0 or NaN**: chain stuck. Check initial step size,
  initial values, or gradient correctness.
- **Posterior mean = initial value**: NUTS not accepting. Step size too
  large or gradient wrong.
- **One chain diverged (Inf/NaN)**: bad prior (too flat) or numerical
  overflow. Suggest tighter prior or check parameterization.

If any check fails, fix the code or report the issue to the user with
a specific diagnosis. Do NOT silently pass a model that shows NI.

### Common fixes for failing checks

- Wrong Rcpp type conversions (see `skills/rcpp_api.md`).
- Missing `#include` or wrong header order.
- Mismatched constructor argument types in RCPP_MODULE.
- Wrong key names in `ctx.at("...")` vs `declare_dependencies`.
- Step size issues: add `cfg.initial_step_size = 0.05` for constrained
  blocks.
- Non-identifiability: suggest tighter priors or reparameterization.

**Do NOT skip this step.** The user expects generated code to compile,
run, and produce reasonable samples out of the box.

---

## 12. Hard rules

- Never write save_state / load_state.
- Never store raw pointers to shared_data inside a leaf.
- Never call `shared_data_t` from a log-density lambda (use `block_context&`).
- Never mix constraint transforms and hand-written Jacobians.
- **Never leave `tests_autodiff/verify_<ClassName>.cpp` behind after
  Check #12 PASS** (the production `.cpp` must be scaffolding-free) and
  never depend on autodiff or Eigen at runtime. **Also `rmdir
  tests_autodiff/` if that directory becomes empty after the verify
  file is deleted** -- the codegen agent's final deliverable to the
  user workspace is *only* the production `<ClassName>.cpp` and the
  R / Python runner (1 .cpp + 1 .R / .py). A leftover empty
  `tests_autodiff/` confuses users into thinking it's part of their
  output.
- **Never leave a `smoke_<ClassName>.R` / `smoke_test.R` file behind
  after Step 3 PASS.** Smoke tests are throwaway; same rule as
  Check #12 verify files. Delete on PASS. No "kept for reuse" option.
  Ongoing compile-smoke coverage is provided by
  `tests_autodiff/audit_compile_smoke.R`, not by per-model smoke
  scripts.
- Never rename the six public methods (step, get_current, set_current,
  predict_at, get_dag, get_history).
- **Never generate a sampler whose target has a dimension-changing
  state space or strong discrete-discrete dependence.** Specifically
  do NOT emit code for: Dirac spike-and-slab (beta_j  in  {0}  U  R),
  model-size priors with unknown K, HMM / change-point with
  Markov-correlated latents, Ising / MRF. Both naive Gibbs (not
  irreducible) AND naive marginalize-gamma + NUTS (silently samples
  slab only, inclusion probability hallucinated with clean R-hat /
  ESS / LOO) are broken on these targets. Follow `codegen_priors.md
  Sec.3a` decision tree: propose Class 2a continuous spike / horseshoe;
  for unknown-K BNP mixtures (DP / PY / HDP) use the **truncated
  SBP** path shipped in `examples/DPGaussianMixture.cpp` /
  `examples/PYGaussianMixture.cpp` /
  `examples/DPGaussianMixture_DerivedAlpha.cpp` (composes
  `categorical_gibbs_block` + `stick_breaking_block` +
  `normal_gamma_cluster_gibbs_block` + `nuts_block` on log alpha);
  for non-BNP unknown-K use `rjmcmc_block` with
  `examples/SpikeSlabRJMCMC.cpp` as template; or decline. See
  `skills/system_design.md` Sec.11 for the measure-theory reason.
- **The `set_current(Rcpp::List)` dispatcher on every BART-based
  wrapper (BartNoise, GBart*, and anything you derive from them)
  already accepts the following keys -- USE them in generated R
  runners when the model nests BART inside an outer Gibbs:**
  - BartNoise: `X` (imputed design), `y` (working residual), `sigma`,
    or any combination. Rejects only `f_bart`.
  - GBartPoisson / GBartLogistic / GBartHeteroscedastic:
    `X` and `y`, or any combination. Rejects only `r` (tree forest
    has no unique inverse).
  - GBartMultinomial: `X` and `y`, or any combination. Rejects
    `r_j` for each non-reference category j.
  - For unshipped GBart variants (NB / AFT / beta / gamma_shape /
    beta_binomial) composed from `genbart_block + genbart::lik::*`,
    follow the same `X` / `y` rule. Likelihood-cached fields like
    `delta` (censoring indicator) are fixed training data per the
    likelihood subclass contract.
  Unified pattern (applies to every stateful module, BART or not):
  ```r
  for (iter in seq_len(n_iter)) {
      # ... outer sampler produces X_new / y_new / ... ...
      m$set_current(list(X = X_new, y = y_new))   # any subset OK
      m$step(1L)
  }
  ```
  Do NOT call any `$set_X`, `$set_Y`, `$set_offset` R method -- those
  do not exist on the R interface. The R interface is ALWAYS exactly
  six methods: `step`, `get_current`, `set_current`, `predict_at`,
  `get_dag`, `get_history`. If you need to ADD a new BART-family
  block (not compose an existing one), that is a system-design task
  and you should not be the agent doing it -- the design contract
  lives in `skills/system_design.md` (a separate skill scoped
  for system-design agents).
- Never add non-English comments to generated code.
- **Always emit the GPL-3.0-or-later license header at the top of
  every generated `.cpp`** (see `codegen_cpp.md Sec.5` for the exact
  three-line form). Do NOT use `Apache-2.0` or any other license in
  generated headers -- the entire project is GPL-3.0-or-later
  (`LICENSE` / `THIRD_PARTY_LICENSES.md` at the repo root).
  The license MUST match across:
  - include/AI4BayesCode/ headers (all GPL-3.0-or-later),
  - examples/*.cpp (all GPL-3.0-or-later),
  - any `.cpp` the generator emits.
  Vendored third-party code (mcmclib, Eigen, autodiff, bart_pure_cpp/, libgp_kernels/,
  celerite/) retains its own upstream licenses; do not touch those
  headers.
- Always consult `skills/rcpp_api.md` for Rcpp type and API conventions.
- **Never use runtime path detection in a generated R runner.**
  That includes `sys.frame(1)$ofile`, `commandArgs(trailingOnly = FALSE)`
  script-directory parsing, `rstudioapi::getSourceEditorContext()`,
  and any "walk up the filesystem looking for a marker file" trick.
  Each of these breaks under at least one of `Rscript` / `source()` /
  RStudio click-run / knitr, and the resulting code is hard for
  non-CS users to read and debug.
  **Instead: emit HARDCODED RELATIVE PATHS** that the codegen agent
  already knows at generation time (e.g. `"AI4BayesCode"` and
  `"<folder>/<ClassName>.cpp"`). The runner assumes the R session's
  cwd is the project root; document that assumption in a top-of-file
  comment. If the user invokes from a different cwd, they get a clear
  filesystem error and an obvious fix (`setwd()` or edit two paths).
  See `codegen_r_runner.md Sec.9` for the exact template.
- **Never expose per-parameter initial values as constructor arguments**
  (e.g. `mu_init`, `sigma_init`, `theta_init`). Initial values must be
  computed inside the wrapper class's constructor from the data and
  prior hyperparameters -- see `codegen_cpp.md Sec.8` "Initial values" for
  the recipe. The ONLY exception is an explicit user request during
  the prior-elicitation flow, in which case mark the arg as
  `[init, exposed by user request]` in the summary table. A user who
  wants overdispersed starts for R-hat diagnostics should use
  `model$set_current(...)` after construction -- not a constructor arg.
- **Always vectorize the natural-scale log-density via armadillo /
  BLAS.** Inside the `wrap(...)` lambda, write the likelihood and
  gradient using armadillo's high-level operators (`X * theta`,
  `X.t() * (y - p)`, `arma::dot(...)`, `arma::accu(...)`,
  `arma::exp(...)`, `solve(A, b)`) rather than nested scalar `for`
  loops over `(i, j)` index pairs. armadillo's `*` dispatches to BLAS
  (Apple Accelerate / OpenBLAS / MKL) and is 10-50x faster than a
  hand-rolled loop on the same math. Pre-store design matrices as
  `arma::mat` (not flat `arma::vec`) in `shared_data`. EVERY BLAS
  expression must carry an inline dimension comment (e.g.,
  `// (N x p)(p x 1) = (N x 1)`). See `codegen_cpp.md Sec.6.1` for the
  per-family templates (Bernoulli logistic / Gaussian / Poisson) and
  the storage / dimension-comment rules. Concrete cost of violating:
  on a small (N~=80, p~=7) Bernoulli logistic GLM, the AI scalar-loop
  pattern ran ~11x slower than Stan's `bernoulli_logit_glm` reference.
- **The chain helper MUST be named `run_chain_<ClassName>()`
  (model-specific, NOT bare `run_chain`) and MUST take all data
  inputs as function arguments -- never close over `y_obs` / `X_obs`
  / etc. from the script's outer scope.** Signature:
  `run_chain_<ClassName> <- function(<data_args>, seed, n_burnin, n_keep) { ... }`
  with `<data_args>` filled in to match the constructor's data inputs
  in the same order (e.g., `y, X` or `y, X, group, J`). Inside the
  body, pass the SAME argument names to `new(<ClassName>, ...)`. Why
  the name: a runner/example may be sourced alongside others in one
  session; a model-specific name prevents clobbering. Why args-not-
  closure: sim1 cross-impl tests call the helper with a fresh dataset
  per replicate -- one that grabs `y_obs` from the enclosing scope
  picks up the wrong data on every replicate after the first.
  Also: held-out prediction, unit testing, and Python-style
  reusability all break if data is closed over. See
  `skills/codegen_r_runner.md Sec.9 "R runner template"` for the
  WRONG/RIGHT contrast and the call-site pattern.
- **Never override `cfg.n_warmup_per_step` -- leave it at the default 0.**
  Do not write `cfg.n_warmup_per_step = 5` or any other non-zero value
  in any `nuts_block_config`, regardless of any "5-8% variance bias
  acceptable" language an older comment in `nuts_block.hpp` may suggest
  (that comment predated the 2026-04-12 mcmclib bugfix and is no longer
  a valid mental model). Non-zero values re-enable a chain-state
  corruption mechanism that produces the "stuck-fast" runtime failure:
  L3 single-dataset PASS but sim1 cross-dataset `rhat_max ~= 2.2`,
  `ess_bulk_AI = NA`, AI walltime SHORTER than Stan's because the locked
  chain does almost no tree exploration. If a sigma block keeps rejecting
  with `n_warmup_per_step = 0`, the actual problem is a non-centered
  hierarchical funnel -- fix it methodologically:
  (a) `joint_nuts_block` over `(sigma_*, z_*)` per `codegen_cpp.md
  Sec.4a` row "scale + raw effect"; (b) bump `n_warmup_first_call` to
  1500-3000; (c) better init via OLS / method of moments. Do NOT escape
  to `n_warmup_per_step > 0`. See `skills/block_catalogue/index.md` "nuts_block
  -> Configuration discipline" for the full table of which fields
  code-gen may set; `validator.md` Check #20 catches violations.
- **Never write a custom conjugate-Gibbs block.** Continuous parameters
  MUST be sampled by `nuts_block` regardless of whether a conjugate
  closed-form update is available. The ONLY Gibbs blocks you are allowed
  to instantiate are the `*_gibbs_block` types already shipped as header
  files in `include/AI4BayesCode/` -- see `skills/block_catalogue/index.md` for
  the authoritative list. Use whichever of those headers exist in the
  repo at generation time, verbatim; never fork, reimplement, or write a
  new one inline in the generated .cpp. When the core library grows a
  new conjugate block, the catalogue is updated and it automatically
  becomes available here -- do NOT hardcode any subset of block names
  into this rule.
  Rationale: letting the generator choose between NUTS and custom Gibbs
  doubles the bug surface (wrong full-conditional derivation, shape-vs-
  rate parameterization, missing Jacobians, etc.); NUTS with the right
  log-density is uniform and inspectable.
- **Never do parallel updates when components are conditionally dependent.**
  If updating parameter_j changes the conditional for parameter_k, you
  MUST update sequentially: sample j, write back to shared_data / context,
  THEN sample k from the updated context. This applies to:
  - binary_gibbs_block: flip gamma_j, update context, then compute
    gamma_{j+1}'s log-odds from updated gamma.
  - categorical_gibbs_block: sample z_i, update context, then compute
    z_{i+1}'s log-probs from updated z.
  - Any custom Gibbs block with vector parameters.
  Parallel update converges to a DIFFERENT stationary distribution.
- **Always register a y_rep stochastic refresher.** `codegen_cpp.md Sec.6a`
  gives the per-observation-family templates. Validator Layer 3 R3
  consumes `model$predict_at(list())$y_rep`; without the refresher,
  R3 cannot run. All 4 `GBart*` examples register y_rep refreshers
  that call the appropriate draw (`rpois`, `rbinom`, NB gamma-Poisson
  mixture, heteroscedastic Normal, AFT Logistic). Every other example,
  including the joint-NUTS examples (`IRT1PL_joint`,
  `HierarchicalLM_joint`, `LinearRegJointMixed`), registers y_rep.
- **Always emit a `mutable std::mt19937_64 predict_rng_` in the
  wrapper**, seeded once in the constructor with
  `rng_seed ^ 0x9E3779B97F4A7C15ULL`, and pass it to every
  `impl_->predict_at(replaced, predict_rng_)` call. See validator
  Check #13 (RNG separation). NEVER call `std::random_device{}()`
  inside `predict_at`.
- **`joint_nuts_block` supports per-slice constraints.** If a joint
  block needs to include a positive scalar (e.g. sigma), declare that
  slice as `joint_constraint::POSITIVE` (likewise SIMPLEX /
  CHOLESKY_CORR / COV_MATRIX / ... per slice). Do NOT concatenate real
  + log(pos) manually -- that bypasses the block's transform bookkeeping
  and fails validator Check #11.3.
- **When the sampler uses `joint_nuts_block`**, emit a header comment at the top of the
  generated `.cpp` stating which parameters are joint and why, and
  emit in the R runner's comments the "requires validator Check #11"
  note shown in `codegen_cpp.md Sec.4a`. Set `USES_JOINT_NUTS <- TRUE`
  in the runner so R3's Bayesian-p-value threshold tightens from
  (0.05, 0.95) to (0.02, 0.98).
- **Always emit `ai4bayescode_perf_hint(...)` at the end of the R
  runner** (see `codegen_r_runner.md Sec.9` template). The helper emits
  a friendly escape-hatch hint if per-sweep time is slow. Set
  `uses_joint_nuts = TRUE` when the runner already uses joint NUTS.
- **Always run Check #12 gradient verification at generation time.**
  For every NUTS block with a hand-written gradient, the AI must:
    1. write a companion file `tests_autodiff/verify_<ClassName>.cpp`
       that imports the hand-written log-densities verbatim and adds
       templated AD versions + a verify function (see `codegen_cpp.md
       Sec.6b` for the full template);
    2. compile + run the verify function;
    3. assert every `max_diff < 1e-8` for AD-backed blocks and
       `< 1e-5` for FD-backed blocks;
    4. **DELETE `tests_autodiff/verify_<ClassName>.cpp` after PASS,
       THEN `rmdir tests_autodiff/` if empty.** Leaving either the
       verify file or an empty `tests_autodiff/` directory in the
       user's workspace is a Hard Rule violation (Sec.12).
  The production `<ClassName>.cpp` stays 100% clean -- no `#ifdef`, no
  autodiff references, no Eigen, no verification artifacts. Users never see
  the verify code, and compile the production `.cpp` directly with
  `Rcpp::sourceCpp(...)`.
- **Never call `std::random_device{}()` inside `predict_at`.** Always
  use the `mutable std::mt19937_64 predict_rng_` member seeded once
  at construction from `rng_seed ^ 0x9E3779B97F4A7C15ULL`. Validator
  Check #13 (RNG separation).
