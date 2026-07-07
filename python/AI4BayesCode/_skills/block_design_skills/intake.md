# block_design MODULE -- intake.md -- INTAKE phase (Stage 0): math spec + I/O, naming gate, SelectWhen, advisory novelty note, borrow/vendor license-gate. Loaded lazily when block_design enters intake; carries forward to design.md. Cites system_design modules + constraints.md/validator.md; does not restate them.

# INTAKE -- what block is this, and is it allowed to exist?

> **GPL-3 LICENSE GATE must already have PASSED.** The mandatory GPL-3.0-or-later
> license gate (`00_flow.md Sec.0.5`) fires BEFORE this phase. If you reached `intake.md`
> WITHOUT the user explicitly agreeing to license their block + example + skill under
> GPL-3.0-or-later, **STOP -- fire that gate NOW** before INTAKE Step 1, and end the flow if
> the user does not agree. It is a HARD gate with no loophole; do not start INTAKE without it.

This is the FIRST phase of the `block_design` flow. It is ENABLING, not gatekeeping: its job
is to capture *what* the user wants to build precisely enough that DESIGN
can build it correctly -- NOT to talk them out of it. The only hard gates
here are mechanical (a name collision, a license incompatibility); the
novelty discussion is **advisory information**, never a block.

The recurring villain across this whole flow is the **silent correctness
bug**: a block that compiles, runs, and returns a wrong posterior. Intake
defends against the version of that villain that strikes before any code
exists -- a fuzzy math spec, an un-stated parameter support, an ambiguous
input/output contract. Pin those down here and DESIGN has something real
to verify against.

**Token discipline.** This module is loaded only when intake is entered.
Do NOT preload `design.md` / later phases. Pull a `system_design` module
only when a step below points at one.

**Ask-the-user discipline.** Every decision below is a structured
labeled-option prompt. In Claude Code use `AskUserQuestion`; in any env
without a structured tool use the markdown labeled-option fallback from
`start.md Sec.0` ("Concrete mechanism per environment"). Mark the standing
choice `(default)` -- NEVER `(Recommended)`. Always include an
`Other / custom` fallback. These invariants are LLM-independent.

**Staging discipline (TOP RULE).** Intake writes NOTHING to the real tree.
It produces a captured spec held for DESIGN; the bundle directory
`./blocks_local/<Block>/` is created only later, on explicit
"go". Nothing is compiled here.

---

## Step 1 -- Elicit the MATH SPEC (and/or the code to BORROW)

> **A block is a reusable PRIMITIVE, not a whole model's sampler.** You are
> designing an EXTENSION SURFACE, not one fixed model. Gold standard:
> `stick_breaking_block` samples a generic length-`K_trunc` simplex and takes the
> process-specific math as two callbacks (`a_fn`/`b_fn`) -- so the ONE block covers
> Dirichlet Process, Pitman-Yor, and custom BNP; it knows NOTHING about the
> likelihood, the cluster parameters, or whether the concentration alpha is fixed or
> inferred (those are sibling blocks + the EXAMPLE). So state the block's target
> GENERICALLY -- what it samples, with the model-specific pieces (likelihood,
> energy, couplings, data) injected as CALLBACKS / refreshable context (Step 1b).
> **Discriminator:** if you catch yourself pinning a specific observation model or
> prior that COULD be a user plug-in, that belongs in the EXAMPLE (or it is a
> codegen composition) -- NOT baked into the block.

A new block must be backed by an explicit model, not a vibe. Capture the
grounding source(s) -- they are NOT mutually exclusive, so **ask this as a
MULTI-SELECT** (in Claude Code, `AskUserQuestion` with `multiSelect: true`; in the markdown
fallback, a "select all that apply -- reply with the letters" list). Offer the ATOMIC
sources, NEVER a pre-combined option (multi-select already covers combinations
such as paper + reference impl, so do not add a "paper + reference code" choice):

- **A specific paper** -- citation + the relevant equations / section refs
 (a DOI / URL, or a local PDF path, all work).
- **My own formulas** -- likelihood / priors / supports handed to you directly.
- **A reference implementation to port / vendor** -- an R / C++ / Julia package
 or a local code path; routes through the Step-5 license-gate; you still owe the
 math spec it implements.
- **Let me (the AI) search for it** -- I run a web search to locate the paper /
 method / reference, read it, then surface what I found for your confirmation
 BEFORE it grounds the spec (SECOND RULE -- never proceed on "what the method
 probably does"; a raw search hit is not a locked spec until you confirm it).
- **Any additional information** -- free-text: extra context, constraints, partial
 details, links, or notes you want folded into the spec.
- **Other / custom.**

> **Local files welcome.** Any source above may be given as a LOCAL FILE PATH -- a
> paper PDF, a reference-code file, or a notes file -- and I will read it from disk
> (a DOI / URL also works for a paper I can fetch).

Whatever the mix, you MUST still finish Step 1 with a COMPLETE, SOURCED math
spec -- see (a). A reference impl (or a raw search hit) alone does not exempt you --
see (b).

**(a) A math spec -- a paper OR formulas.** Capture it as an ORDERED sequence, not one
half-filled dump -- confirm the model STRUCTURE before resolving the priors (display-block
format per `start.md Sec.3b`; never inline `$...$` -- and any roles / I-O TABLE you show has
plain-Unicode cells like `X`, `beta`, `R^p`, `nxp`, `>0`, NEVER `$...$`, since a `$$` display
block cannot live inside a table cell):

1. **Likelihood / observation model FIRST -- present it, then CONFIRM it.** Show
   `p(y | theta, ...)` (the full observation model) as a `$$ ... $$` **display block**
   (`\begin{aligned}`, codegen "(a)" style; it renders), **without prior placeholders**.
   Then fire a structured confirm gate -- via the env ask mechanism (top-of-file note /
   `start.md Sec.0`: `AskUserQuestion` in Claude Code, the markdown labeled-option fallback
   elsewhere), options `(a) Looks right (default) / (b) Fix the model / (c) Other` -- BEFORE
   asking any prior.
   Do NOT show a spec littered with "prior -- fork Q2/Q3" placeholders and dive straight into
   the fork questions -- confirm the structural math first.
2. **THEN elicit the priors + supports** (once the likelihood is confirmed):
   - **Every prior** `p(theta)`, `p(hyper)`,... -- each with its parameterization stated (e.g.
     Gamma **shape-rate vs shape-scale** -- the Check #1 bug class, `validator.md Sec.1`); no
     "standard prior" hand-waving.
   - **Each parameter's SUPPORT** -- real / positive / simplex / interval / ordered /
     Cholesky-corr / unit-vector / binary / categorical (`system_design Sec.11` geometry rule 1;
     the raw material DESIGN feeds into constraint mapping, `constraints.md`). A missing
     support is a latent silent bug: an unconstrained sampler on a positive parameter looks
     fine and is wrong.
3. **THEN re-display the COMPLETE spec and CONFIRM it.** Re-show the same `$$ ... $$` block,
   now with the RESOLVED priors filled in (no placeholders), then fire a **second** structured
   confirm gate -- same env mechanism (`AskUserQuestion` in Claude Code, markdown labeled-option
   fallback elsewhere) -- options `(a) Spec confirmed -- proceed (default) / (b) Change
   <which item> / (c) Other`. This is the **math-spec sign-off**: gate 1 confirmed the model
   STRUCTURE, gate 2 confirms the full spec WITH priors. (The later Step 6 gate is the OVERALL
   intake sign-off -- name + Description + SelectWhen + I/O -- and does NOT re-litigate this already-confirmed
   math.) This confirmed complete spec is what DESIGN builds + verifies against.

If the user hands you a **paper**, do NOT proceed on "what the method
probably does." Read the relevant algorithm/equations; if you do not have
them, ASK for the section/equation refs (SECOND RULE -- methodology from a
paper). Surface every implementation choice the paper pins (sampler,
update order, marginalization, noise propagation) so DESIGN can lock them
against the source -- but the deep methodology lock-in happens in DESIGN;
intake's job is to make sure the spec is COMPLETE and SOURCED.

**(b) Existing package/code to BORROW or VENDOR.** If the user is porting
or wrapping a published implementation (an R/C++/Julia package, a
reference kernel), capture *what* code and *from where*. This routes
through the license-gate in Step 5 before anything is reused. Borrowing
code does NOT exempt you from (a): you still need the math spec the code
implements, so DESIGN can verify the port reproduces the right posterior.

It is common to have BOTH: a paper for the math + a reference
implementation to vendor the kernel from.

### Step 1b -- Define the EXTENSION SURFACE (data + callback contract)

This is the core block-design artifact: the seams that make the block REUSABLE.
Capture FOUR buckets (`stick_breaking_block`'s config is the template):

- **Config (fixed at construction)** -- dimensions, hyper-constants, the block's
 `name`, and the shared_data KEY NAMES it reads/writes (e.g. `K_trunc`,
 `counts_key`, `initial_*`). Name them conceptually here; the `<Block>_config`
 struct is built in DESIGN.
- **Callbacks (the model-specific math the USER plugs in)** -- the `std::function`
 seams that let ONE block serve many models. This is the heart of extensibility:
 `stick_breaking_block`'s `a_fn`/`b_fn` (Beta-stick params -> DP / PY / custom),
 `nuts_block`'s log-density+gradient oracle, `gmrf_whitened_ess_block`'s
 `log_lik(x, ctx)`, a `*_gibbs_block`'s `params_fn`. **Do NOT hardcode a specific
 likelihood / energy / prior the user should supply -- expose it as a callback.**
- **Refreshable context (read each sweep from `shared_data` via `ctx.at(key)`)** --
 quantities OTHER blocks update between sweeps: working residuals, cluster
 counts, an imputed design matrix (`system_design Sec.3`). The block READS, never
 owns, these.
- **Outputs (written to `shared_data`)** -- the sampled object(s) + key name(s) +
 dimension. Single-parameter block => its `name` IS the key. **JOINT block => EACH
 sub-param's name is its own `shared_data` + `get_history()` key**; name them
 `<block>_<slice>` (e.g. `myblk_phi`, `myblk_tau`) to avoid collisions and so
 child-sync + history-mode `predict_at` read the right slice.

**Hyperparameters are TUNABLE NAMED SLOTS, never hardcoded literals.** Any prior scale, `nu`,
concentration, etc. the block's log-density / prior uses MUST be a named slot, not a number buried
in the math (`system_design Sec.3` lists hyperparameters as shared_data entries; Sec.4's generative DAG
only sees NAMED `hyperparam -> param` edges). Tunability spectrum -- pick the lightest that fits, but NEVER a frozen literal:
- usually fixed for a run -> a **config field with a sensible default** that is ALSO **adjustable
 via `set_current`** (route the hyperparameter key through the Tier-A dispatcher -> update the
 stored value + any shared_data mirror; `dataflow Sec.7`), so it is tunable post-construction WITHOUT
 rebuilding the block;
- could be **inferred** by a sibling block / changed online every sweep -> a **refreshable
 `shared_data` key** read via `ctx` each sweep (the `lambda_key` pattern; this is what "accept
 external input" / full-Bayes means here, and it auto-gives the Sec.4 context edge);
- the prior/likelihood **functional FORM** itself -> a **callback** (so the user plugs N(0,tau^2) /
 horseshoe / Laplace -- the `a_fn`/`b_fn`, `log_lik` pattern).
Bake ONLY genuine structural constants (dims, key names). When you elicit a prior/hyperparameter,
elicit its DEFAULT but design the SLOT to be tunable -- **at minimum `set_current`-adjustable**,
never a hardcoded literal in the math.

> **Read the DP block as your worked reference before filling these in:**
> `include/AI4BayesCode/stick_breaking_block.hpp` -- see how its `_config` maps to
> the four buckets (config `name` / `K_trunc` / `counts_key` / `initial_pi`;
> callbacks `a_fn` / `b_fn`; refreshable `counts_key`, read each sweep; output: the
> simplex under `name`). Then `examples/DPGaussianMixture.cpp` +
> `examples/PYGaussianMixture.cpp` -- the SAME primitive composed into TWO different
> full models, with the likelihood, cluster parameters, and alpha living in sibling
> blocks + the example, NOT in the block. That split is exactly what you are aiming
> for.

Without this surface the block cannot be wired into a composite, cannot be
reused, and cannot be selected. The Gibbs-DAG / predict-DAG wiring of these keys
is DECLARED later (DESIGN); here we just name the seams.

-> **No sign-off gate yet** (the consolidated intake sign-off is Step 6).

---

## Step 2 -- NAMING GATE (hard, mechanical)

Elicit the block name and validate it. This is a HARD gate: a bad name is rejected here, not later.

The `Block` name is the ONE source of identity -- ClassName, header
filename, and bundle dir are all derived from it by convention
(`class <Block>`, `<Block>.hpp`, `blocks_local/<Block>/`; the manifest `Block:` field, schema in `skill.md Sec.3`).
So it must satisfy ALL of:

1. **CRAN-flat + globally unique** -- a single flat name, NO `@author/`
 scope (registry-first-come is FUTURE). Fits the R-centric ecosystem.
2. **`snake_case` STEM; the system appends `_block`** -- the user supplies only the
 semantic stem (e.g. `t_regression`, `poisson_icar_ess`); YOU auto-complete the
 canonical `Block` = `<stem>_block` (e.g. `t_regression_block`). **Idempotent**: if the
 user's input already ends in `_block`, use it AS-IS -- never double-append
 (`..._block_block`). The final `Block` is `snake_case` + a single `_block` suffix, so
 contrib blocks look native alongside core (e.g. `poisson_icar_ess_block`).
3. **NOT a core-reserved name** -- a contrib block may NOT take a core
 block's name. Core names are reserved.
4. **Unique vs core + existing local** -- check at creation against the
 core `block_catalogue/index.md` entries and every existing
 `blocks_local/*/manifest.dcf` `Block:` field. (Registry availability
 is a FUTURE check; local + core is the MVP check.)

Mechanically (exact, resolvable from the repo root):
- The **authoritative reserved-name list = the shipped block headers**:
 `ls AI4BayesCode/include/AI4BayesCode/*_block*.hpp` (each `<name>_block.hpp` is a
 reserved core name). The routing table in `AI4BayesCode/skills/block_catalogue/index.md`
 lists the same names with prose, but the headers ARE the list -- do not depend on the
 catalogue's path being known.
- Existing local names: the `Block:` line of each `./blocks_local/*/manifest.dcf`.

Reject on any clash (reserved or duplicate) and ask for another name. Offer 2-3
spec-derived candidate **stems** as labeled options -- show them WITHOUT the `_block`
suffix (e.g. `student_t_regression_nuts`, `robust_t_regression_nuts`, `t_regression`),
and note that the system appends `_block`; plus `Other / custom`. Validate the
auto-completed `<stem>_block` (snake_case, not core-reserved, unique vs core + local).

If the proposed name fails any rule, state WHICH rule and re-ask. Do not
proceed past a colliding or malformed name -- every downstream derivation
(class symbol, header, dir, manifest key, index line) depends on it.

-> This gate must PASS before Step 6 sign-off.

---

## Step 3 -- Description (human) FIRST, then SelectWhen (match line) -- in plain language

**Audience: a working statistician / applied researcher creating a block, NOT a library
developer.** Everything you SHOW or SAY in this step is in plain statistical language. Do NOT
put internal-system jargon in user-facing text -- never say "auto-selector", "Stage B",
"candidate block", "token cost", "loaded on every run", "the selector", "routing key". (Keep
such terms in your own reasoning if useful; just never show them to the user.)

Two DISTINCT fields, two readers (`skill.md Sec.3`), elicited in THIS ORDER -- do not conflate
them. The common mistake: drafting a paragraph-long "match line" that is really a Description.

**Before the drafts, explain WHY there are two -- in plain terms** (one or two sentences):
- a **Description** -- a plain-English summary of the model and when you'd use it, for someone
  later **browsing the list of available blocks**; it can be detailed;
- a **one-line "use this when..." summary** -- what lets the tool **automatically suggest your
  block to a future user whose model matches**; keep it to one short line.
They are separate because the description can be long (people read it once they are already
looking at your block), while the one-liner must stay short so the tool can quickly match it
against a new model. (Agent-only -- do NOT recite to the user: the one-liner is the
auto-selector's Stage-B trigger, loaded per-block at selection time -> a real token cost; the
Description is never loaded by the selector.)

**3a -- `Description` -- draft + confirm FIRST. Write it in CRAN-package `Description:` style.**
Blocks are meant to be published / uploadable like CRAN packages, so the Description follows
CRAN's `Description:` conventions: a concise paragraph of complete sentences, present tense, no
marketing words ("powerful", "state-of-the-art"); do NOT open with "This block" / the block
name / "A block that..." -- open with the substance; optionally cite the method's source in CRAN
form (`Author (Year) <doi:...>` / `<arXiv:...>`).

**"CRAN-style" is YOUR internal writing standard -- do NOT say the word "CRAN" to the user
abruptly.** The user is a statistician with no idea what their block has to do with CRAN; dropping
"CRAN" on them is confusing ("what does CRAN have to do with my block?"). When you present or
explain the Description, describe the requirement PLAINLY -- "a concise, model-first plain-English
summary of the block" -- and either explain the link in one clause if you mention CRAN at all
("written like a package abstract, since blocks are meant to be publishable") OR don't mention CRAN.
Never just announce "I'll write it in CRAN style" with no context.

**Content order -- WHAT before HOW.** (1) State WHAT it IS -- the model -- FIRST, naming the
observation model / likelihood with the formula in plain Unicode (e.g.
`y_i ~ Student-t(nu, x_i^Tbeta, sigma^2)`). (2) The prior on each parameter. (3) One plain sentence on what
it is good for. (4) The **sampling strategy matters too** -- add ONE plain, high-level sentence
on how it is sampled ("beta and sigma are sampled jointly with the No-U-Turn sampler"), but do NOT
lead with it and do NOT dwell on deep implementation ("single NUTS kernel over (beta, log sigma)",
"hand-written log-density", "analytic gradient" -> those belong to DESIGN). Example shape:
*"Implements Bayesian linear regression with heavy-tailed Student-t errors,
y_i ~ Student-t(nu, x_i^Tbeta, sigma^2), for robust inference when residuals contain gross outliers. The
coefficients beta take a weakly-informative Gaussian prior and the scale sigma a half-Normal prior;
the degrees of freedom nu is a fixed, tunable setting. beta and sigma are sampled jointly with the
No-U-Turn sampler."* The Description is never loaded by the auto-selector, so this paragraph
can carry model + priors + purpose + a brief sampling note -- just keep the MODEL first, in a
statistician's language. PRINT the draft into the chat BODY (a `>` blockquote / **bold**, per
`start.md Sec.0`), then a SHORT confirm (`Use it (default) / Refine -- tell me the change /
Other`). -> manifest `Description:`.

**3b -- `SelectWhen` -- the one-line "use this when..." summary, ONLY after the Description is
confirmed.** Tell the user plainly (no jargon): *this is the single short line that lets the
tool find and suggest your block when a future user describes a matching model -- so keep it to
**about 10 words** (~6-12, one clause), not a sentence with multiple clauses.* (Agent-only
reason it must stay tiny: it is the auto-selector's Stage-B trigger, loaded per-block at
selection time.) Draft a COMPACT one -- NOT a paragraph; the detail already lives in
`Description`. It must:
- name the **model structure** that triggers the block, not the algorithm -- "linear regression
  with heavy-tailed / Student-t errors" is matchable; "robust models" is not;
- **minimize abbreviations / method acronyms** (NNGP, PGAS, HMC, ICAR...): an acronym DOES help the
  selector index a specialist who types it, so AT MOST ONE may ride along as a brief trailing TAG --
  but the LOAD-BEARING match words must be the plain structure a non-specialist would write (their
  free-form model description rarely contains the acronym). Lean on structure; keep acronyms few;
- naming the **baseline it beats** is OPTIONAL and only if it still fits ~10 words; drop it if
  it would blow the budget (the Description already covers it).
Target ~10 words. Good: **"Linear regression with heavy-tailed Student-t errors, robust to
outliers"** (~9 words). Acronym as a trailing TAG, structure load-bearing (good): **"Large-n
spatial regression, GP random effects, dense GP too slow (NNGP)"** (~10 words) -- NOT "NNGP spatial
GP block" (acronym carries the match -> a non-specialist never matches). Too long: "Linear
regression with heavy-tailed or outlier-contaminated continuous errors (Student-t); use instead of
Gaussian/OLS regression when residuals have gross outliers" (~22 words). PRINT the compact draft into the chat BODY (blockquote / **bold**, per
`start.md Sec.0`), then a SHORT confirm (`Use it (default) / Refine / Other`). **A vague OR bloated
one-liner = a block the tool never suggests.** This becomes the manifest `SelectWhen:` field;
`design.md` sanity-checks it against the impl.

-> Both carried forward to DESIGN + manifest (`Description` + `SelectWhen`).

---

## Step 4 -- ADVISORY novelty / benchmark note (information, NOT a gate)

**Why this note exists -- say it to the user up front.** We do NOT want a brand-new block built
when existing blocks already **compose** to the same model. A redundant primitive is pure cost
-- more code to maintain, more clutter in the block registry, and a duplicate / near-duplicate
block has NEGATIVE value (it splits attention and invites the wrong block being picked later).
So before building, honestly assess one thing: **can existing blocks already do this by
composition?** The baseline to weigh against is **`joint_nuts_block` + non-centered
reparameterization (NC)**, which already reaches the target posterior for most fixed-dim
continuous models with tightly-coupled parameters.

This is INFORMATION the user decides on; it does **NOT** block local creation (the real
novelty/benchmark gate is at registry submission -- FUTURE). Two cases:

- **NOT composable** (the model genuinely needs something existing blocks can't express): say
 so briefly and proceed to build -- no justification needed.
- **Composable** (existing blocks + NC would reach the same posterior): say so plainly, and the
 DEFAULT is to **compose instead** (route to codegen), NOT build a redundant block. **Do NOT
 write a justification for the new block yourself** -- do not manufacture "convenience" or
 "it'll be faster" arguments on the user's behalf. That (the AI rationalizing a redundant
 block) is the exact failure to avoid. If the user wants to build it anyway, the reason is
 THEIRS to give: ask them to state it in their own words and record it verbatim; you supply
 NONE. You may verify a *speed* claim the user makes with a **benchmark plan** (same model +
 budget, `joint_nuts_block`+NC vs the proposed block, ESS/sec + R-hat side by side) -- offered to
 TEST their claim, not to argue it for them.

Acknowledgement prompt **when composable**: "(a) Compose from existing blocks instead -- route
to codegen `(default)` / (b) Build it anyway -- I'll ask you for your reason / (c) Other". If
the user picks (b), ask for their justification and record it; do NOT supply one. **Whatever
the user chooses, local creation proceeds if they want it.** Do NOT gate.

> Honest caveat: composability is not an AUTOMATIC disqualifier -- a composable-but-much-slower
> block can still be worth shipping, so "cannot be composed" is too strict as a HARD veto. But
> the justification for building a composable block is the USER's to give, NEVER the AI's to
> manufacture; absent a user-stated reason, the default is compose-instead.

---

## Step 5 -- BORROW / VENDOR: license-gate + mandatory attribution

Triggered only if Step 1(b) applies (reusing third-party code). If the
block is written from scratch against only the FIXED system-vendored set
(`eigen` / `mcmclib` / `autodiff` / `BaseMatrixOps` / `bart` / `celerite`
/ `libgp_kernels`, already on `-I` -- `system_design Sec.2` Tier-C), this step
is a no-op: you just `#include` from it, nothing new is vendored.

For NEW third-party code being borrowed into the bundle:

**(1) License-gate (hard).** The combined AI4BayesCode distributed work is
**GPL-3.0-or-later** (it ALREADY vendors Apache-2.0 mcmclib / BaseMatrixOps for the NUTS
backend -- see the README "Licensing" section), and **any block you generate inherits
GPL-3.0-or-later**. So accept any **GPL-3-compatible** upstream and reject GPL-3-incompatible
(authoritative statement: `system_design Sec.14` Step 1 = `lifecycle.md`; do not restate, apply it):

- **GPL-3 / GPL-2.0+ / MIT / BSD / MPL-2.0 / Apache-2.0 / CeCILL-B** upstream -> GPL-3-compatible
 -> **accept**. **Apache-2.0 needs NO special handling** -- the combined work is ALREADY GPL-3,
 so there is no new "branch" to cross; just record the vendored license in
 `THIRD_PARTY_LICENSES.md`.
- **Non-free / proprietary / unclear** -> **REJECT**. Do not vendor. If the
 license cannot be determined, treat as unclear -> reject and ask the user
 to clarify provenance.

Present the determined license + its consequence as a structured prompt
before reusing anything; on reject, the user must supply
differently-licensed code or drop the borrow.

**(2) MANDATORY attribution.** If accepted, the original copyright +
license is preserved **verbatim** inside the bundle's `vendor/` directory.
For a **single-header** vendored routine (e.g. a special function), the
license/copyright header carried IN the `.h` itself satisfies "verbatim" -- a
separate `LICENSE` file is NOT additionally required; for a multi-file
upstream, also copy its `LICENSE` into `vendor/`. Do NOT alter the upstream's
own license headers -- they stay as-is (the project-wide GPL-3.0-or-later applies to code WE
author + the combined distributed work, not to the vendored upstream). This is non-negotiable
attribution, recorded as an action for DESIGN to stage into
`blocks_local/<Block>/vendor/`.

**(2b) VERIFY the vendored routine before building on it.** Borrowed numerics
are an untrusted input: check the routine standalone against KNOWN values
(and/or an `lgamma`-style finite-difference) BEFORE wiring it into the block,
and carry that accuracy number into VALIDATE (where T0 / T1b double as its
regression -- see `validate.md`). A silently-wrong vendored `digamma` would
otherwise surface only as a mysterious downstream gradient mismatch.

**(2c) STATEFUL adaptation -- `vendor.md`.** A vendored KERNEL is external code not written for the
stateful-block contract; it usually needs MODIFICATION (isolate hidden global / `static` state into the
instance, thread the block's `rng` instead of its own RNG, rebuild data-derived caches on `set_context`)
to be safe as a reused primitive. When you reach DESIGN, load `block_design_skills/vendor.md` for those
stateful-modification patterns; VALIDATE then runs the vendored-kernel stateful-compatibility check
(`validate.md` Sec.1 / Sec.2).

**(3) Record in the manifest.** The borrowed dependency is recorded in the
manifest `Vendored:` field (the upstream name + license + the bundle
`vendor/` path), so the bundle self-documents what third-party code it
carries. (This is distinct from the dropped per-block `VendoredDeps`
declaration of the FIXED system libs -- which a block just
`#include`s and never re-declares; `Vendored:` is ONLY for NEW borrowed
code added to this bundle.)

-> License-gate must PASS (or "no vendoring") before Step 6 sign-off.

---

## Step 6 -- INTAKE sign-off gate + CARRY-FORWARD

Before leaving intake you MUST (1) present a compact **carry-forward** summary --
the already-confirmed math spec in BRIEF (it was signed off at Step 1 gate 2 -- give a
one-line recap or re-show the `$$...$$` block, do NOT re-derive or re-litigate it) PLUS
the items this gate actually locks: **name / Description / SelectWhen / I/O (+ vendor)** -- THEN
(2) fire a **structured confirm gate** -- `AskUserQuestion` in Claude Code (the Sec.0 markdown fallback
otherwise), NOT a prose "does this look right?". This is a **HARD STOP**: do NOT
ask any DESIGN-phase question and do NOT load `design.md` until the user confirms.
(TOP RULE -- this fixes the problem statement DESIGN builds against; SECOND RULE --
methodology sourcing is locked here for paper-backed blocks.) The gate:
`(a) Confirmed -- proceed to DESIGN (default) / (b) Change <which item> / (c) Other`.
On (b)/(c): apply the change, re-show the summary, and re-confirm before advancing.

The **carry-forward bundle** handed to `design.md` is:

| Carried item | What it pins for DESIGN |
|---|---|
| **Math spec** (likelihood + every prior + each parameter's support; paper refs if any) | the model DESIGN implements + verifies against (geometry, constraint mapping, parity tests) |
| **Block name** (validated: flat, `snake_case`+`_block`, not reserved, unique) | derives ClassName / `<Block>.hpp` / `blocks_local/<Block>/` / manifest `Block:` / index line |
| **Description** (human-facing, detailed -- what the block IS) | manifest `Description:`; for catalogue readers, NOT loaded by the auto-selector |
| **SelectWhen** (crisp, COMPACT "use WHEN `<structure>`") | manifest `SelectWhen:` + the auto-selector's Stage-B trigger (the ONLY block-specific line it loads -- keep it short); DESIGN sanity-checks it vs the impl |
| **INPUT / OUTPUT** (consumed data + refreshable-vs-config; produced keys + dims) | `<Block>_config` fields, `set_context` refresh, `shared_data` write-back, Gibbs/predict-DAG wiring |
| **Vendored** (any borrowed code: upstream + license + `vendor/` path), or "none" | `vendor/` staging + license header preservation + manifest `Vendored:` + GPL-3-compatibility check (combined work already GPL-3.0-or-later) |

Only after this sign-off does the flow advance to DESIGN. Intake has
written nothing to the real tree; it has produced a complete, sourced,
name-validated, license-cleared spec -- the foundation DESIGN needs so the
block it builds targets the *right* posterior.

---

### Pointers (cite, don't restate)

- Geometry / parameter support classes -> `system_design Sec.11` (`geometry.md`).
- Three-tier architecture + FIXED vendored set + `shared_data`/`block_context`
 -> `system_design Sec.2`/`Sec.3` (`interface.md` / `dataflow.md`).
- License-compatibility rule (authoritative) -> `system_design Sec.14` Step 1
 (`lifecycle.md`).
- Constraint/transform kinds (used in DESIGN's constraint mapping) ->
 `constraints.md`.
- Parameterization (shape-rate vs scale) bug class -> `validator.md Sec.1`.
- Manifest field definitions (the full 13-field schema) -> `skill.md Sec.3`
 (the SKILL phase authors `manifest.dcf`); INTAKE only CAPTURES the values
 of `Block` / `SelectWhen` / `Vendored` here as carry-forward.
- The two-stage auto-selection + compact index that `SelectWhen` feeds ->
 `contrib.md Sec.4` (Discovery + auto-selection).
