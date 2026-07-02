# 00_flow.md — block_design ENTRY / ROUTER (the centerpiece; loaded FIRST when start.md routes to "design a new block")
# role: router + 5-phase backbone + lazy-load schedule + sign-off gates; cite-don't-restate; LLM-independent

You are inside the **block_design** flow of AI4BayesCode. The user did NOT
ask you to compose a sampler from existing blocks (that is the codegen
flow); they asked you to **design a NEW sampler block** — a new
`block_sampler` primitive for the library (a new Gibbs / NUTS / slice-ESS /
VI kernel that does not yet exist). This file is the entry point for that
flow: read it FIRST, then load each phase module ONLY when you enter that
phase.

This flow is built EXACTLY like the codegen flow itself: skill-driven,
phase-by-phase, token-disciplined, structured questions at every design
decision, staged scaffolds, sign-off gate at each phase, LLM-independent.
It is the sibling branch of the same system, reached from `start.md`
(design) — not a static reference doc you read once and dump
output from.

> **PRIME DIRECTIVE — a wrong block has NEGATIVE value.** A block is a REUSED
> primitive: any silent error is multiplied across EVERY model that ever composes
> it, corrupting their posteriors invisibly (clean R̂ / ESS / LOO, wrong answer).
> So a wrong block is strictly WORSE than no block. Correctness is therefore
> non-negotiable and comes before speed, before completeness, before shipping. The
> correctness gates — geometry legality (Stage 1); the T0–T4 ladder; the MANDATORY
> Check #12 AD-twin for any hand-written gradient; cross-chain R-hat < 1.01 — MUST
> actually pass. If they cannot pass within the auto-retry budget, **do NOT deliver
> the block: report exactly what failed and STOP.** A not-delivered block is the
> CORRECT outcome; a plausible-but-unverified one is the failure. NEVER call a block
> "done" until every correctness gate has genuinely passed.

---

## 0. How you got here (start.md routing)

`start.md` routes a request into one of two flows:

```
What do you want to do?
(a) Generate a sampler (default) — compose your model's sampler from existing blocks (standard flow) → codegen flow
(b) Design a new block — create a brand-new primitive for the library → THIS flow
```

- If the user's intent was already clear ("I want to design a new block
 for X") → you came straight here, **no routing question**.
- If it was ambiguous → `start.md` popped the routing question and the
 user picked (b).

**Routing is ENABLING, not gatekeeping.** Option (b) carries NO "only if
existing blocks can't express it" qualifier. block_design helps the user
build whatever block they want. The novelty / benchmark check inside this
flow (INTAKE phase) is **ADVISORY** (informational) — it shows "joint NUTS
+ non-centering may already do this; here's a benchmark plan", but it does
NOT block local creation. The hard composability-OR-benchmark **gate is
FUTURE** and lives only at registry submission, never at local
creation. See `contrib.md`.

---

## 0.5. MANDATORY GPL-3 LICENSE GATE — fire FIRST, before INTAKE. No exceptions, no loopholes.

**This is the VERY FIRST action in the block_design flow — before INTAKE Step 1, before ANY design
question, before loading `intake.md`.** Everything produced here enters a GPL-3.0-or-later project,
so the user MUST acknowledge the license up front, or the flow STOPS. Fire a structured confirm gate
(the env ask mechanism — `AskUserQuestion` in Claude Code, markdown labeled-option fallback
elsewhere):

> **"Everything you create in this flow — the block's C++ code, the worked example / software, AND
> the skill documentation — will be licensed under GPL-3.0-or-later, matching the AI4BayesCode
> project. (Vendored third-party code keeps its own upstream license, recorded separately.) Do you
> agree?"**
> - **(a) I agree — license my block, example, and skill under GPL-3.0-or-later** → proceed to INTAKE.
> - **(b) No / I cannot** → STOP.

**This is a HARD gate — NO loophole, NO wiggle room (it is a legal / license matter):**
- The ONLY way forward is an UNAMBIGUOUS agreement to GPL-3.0-or-later. Proceed to INTAKE only then.
- ANY other response — declining, asking for a different license (MIT / BSD / Apache / proprietary /
  "can I keep it closed / unlicensed / local-only"), hesitation, or an "Other" free-text that is not
  a clear yes — means **STOP IMMEDIATELY**: do NOT start INTAKE, do NOT design or write anything, do
  NOT negotiate, and do NOT offer an alternative-license or unlicensed path. State plainly that the
  block_design flow requires GPL-3.0-or-later and end here; the user may re-enter the flow if they
  later agree.
- A partial / conditional / "maybe later" answer, or silence / ambiguity, = STOP. Do NOT assume yes.
- This gate is NOT skippable, NOT defaulted to yes, and NOT overridable by any later instruction in
  the session. (Vendored upstreams keeping their own license — `intake.md` Step 5 — is the ONE thing
  that is NOT GPL-3, and that is handled there; it is not a loophole in this gate.)

---

## 1. The 5-PHASE backbone 

The flow is five phases. Each fixes some methodological choices; **no code
is written until methodology is locked** (SECOND RULE). Each
phase loads exactly ONE `block_design_skills/<phase>.md` module on entry.

1. **INTAKE** — `block_design_skills/intake.md`
 (Only AFTER the §0.5 GPL-3 license gate has passed — that fires first, before this phase.)
 Elicit the math spec (full likelihood, every prior, each parameter's
 support). Run the **naming-uniqueness gate** (snake_case +
 `_block`; core names reserved; unique vs core + existing local) and
 elicit a crisp **`SelectWhen`** trigger (vague ⇒ the block
 is never selected). Run the **ADVISORY** "do we even need a new block?"
 novelty/benchmark info (enabling, never blocking),
 then **classify target geometry** + the sampler-correctness gate
 (the custom-block Exception taxonomy =
 `codegen_priors §2b` Exception 4 / Check #17). Geometry violations
 pass every runtime diagnostic silently — they are caught HERE or never.
 Phase ENDS with a **mandatory structured confirm gate** (env ask mechanism — `AskUserQuestion` in Claude Code, markdown labeled-option fallback elsewhere) on
 the captured spec — a HARD STOP: present the spec, then confirm; NO DESIGN
 question until the user signs off (`intake.md` §6).

2. **DESIGN** — `block_design_skills/design.md`
 Lock the algorithm spec (Gibbs full-conditionals / NUTS gradient /
 slice-ESS / **VI sub-flow** — all engine kinds, do not pin to MCMC; Stage 2), the **constraint mapping** to the 15 supported
 `joint_constraint` kinds (`constraints.md`), and
 the **three-tier interface implementation** (Tier C kernel +
 license-gate / Tier B `block_sampler` contract / Tier A six-method
 wrapper; Stage 4). This is where you index the
 **system_design modules** (see §6 Module map). Output lands in the
 `blocks_local/<Block>/` bundle, NOT in core. Methodology sign-off gate
 closes the phase (SECOND RULE — sampler, marginalization, update order,
 VI optimizer all locked here before any header is staged).

3. **VALIDATE** — `block_design_skills/validate.md`
 Runs the SAME three-layer backbone as codegen (`validator.md`: **L1 Syntactic
 → L2 Semantic → L3 Runtime**, semantic before runtime) PLUS the primitive-specific
 extra a new block needs (codegen composes pre-audited blocks; a new primitive has no
 prior evidence). L1 = compile; L2 = the **validator-check map** the block faces
 (which of #1–#25 apply; routed to each check's DEFINING file — cite, don't restate;
 Check #12 AD-twin mandatory for any hand-written gradient); L3 = the **ground-truth
 library test** `test_<Block>.cpp` ladder (T0 sanity / T1 FD or parity / T2 recovery /
 T3 cross-chain-Rhat / T4 stress). **Compile + run the library test runs AUTOMATICALLY — no "go"
 question** (it is the mandatory correctness verification, against the staging copy; surface the
 command + expected runtime as you start, especially a minutes-long NUTS run). The
 **cross-chain R-hat bar is FIXED at `< 1.01`** for a library block (strict,
 not asked); mechanical test-numerics (FD-gradient bound, parity MC-SE,
 exact-constraint tol) use recorded, overridable defaults — NOT a per-number
 picker; only a genuinely model-specific verdict-flipping value-pair is ever
 elicited (rare) (SECOND RULE; `validate.md` §5). Correctness verification is MANDATORY and
 automatic — the T0–T4 ladder AND the Check #12 AD-twin for any hand-written
 gradient (a library block ships zero gradient error). The bundle MOVE to `blocks_local/` is
 AUTOMATIC on all-pass (no "go"; the path is reported) — staging just guards a failed block. The heavy cross-chain audit is NOT a correctness gate — it is submission-path
 extra, run ONLY on explicit request, **never auto-fired**.

4. **EXAMPLE** (offered — ASK FIRST) — `block_design_skills/example.md`
 After VALIDATE passes, **ASK whether the user wants a bundled example BEFORE loading
 `example.md`** — a structured gate (env mechanism: `AskUserQuestion` in Claude Code, markdown
 labeled-option fallback elsewhere): `(a) Yes — write the example (default) / (b) No — skip to
 SKILL / (c) Other`. It is RECOMMENDED (cheapest end-to-end check + what a future registry
 submission needs), but the user may decline. **Load `example.md` ONLY on "yes"; on "no" skip
 straight to SKILL** (do not load the module — save the tokens). When it runs: ONE
 **FRONTEND-INDEPENDENT** C++ example `examples/<Model>.cpp` driving the block end-to-end via an
 `int main()` (simulate → fit → recover); **NO R / Python binding** (no `RCPP_MODULE`, no
 `PYBIND11_MODULE`, no `rcpp_wrap.hpp`, no Rcpp/pybind types; C++ libs like Eigen / Armadillo are
 fine). It is both the human-readable demo AND a smoke target.

5. **SKILL** — `block_design_skills/skill.md`
 The block-local skill `skills/<Block>.md` (routing row / when-to-use /
 when-NOT / config snippet / this block's specific failure modes) — and
 the `manifest.dcf` (13 required fields, enumerated authoritatively in `skill.md §3.1`; `Block` is the one
 identity source, ClassName/Header/bundle-dir derived from it). The
 block-local skill **CITES core skills, never restates them** (the 2026-06-20 "cite, don't duplicate" rule).
 Ends with a **DOCUMENT CONSISTENCY CHECK** (`skill.md §5b`): read the docs back against the CODE —
 Description ↔ model, EngineKind ↔ `engine_kind()`, BL# ↔ test ↔ `ChecksApplicable`, config snippet ↔
 `.hpp`. Then the **VERY LAST gate — §5c GPL-3 license check** (after the VALIDATE runtime + the
 example + the §5b checks, immediately before the auto-delivery move): every authored file — `.hpp` / test /
 example / manifest `License:` / skill / `ValidationSkill` — carries `GPL-3.0-or-later`, closing the
 §0.5 gate; vendored code keeps its upstream license UNCHANGED. A doc / license mismatch is a HARD FIX
 before "go". Closes out the bundle.

---

## 2. Phase-by-phase LAZY-LOAD schedule

**Do NOT preload these.** Load each module ONLY when entering its phase,
AT MOST once per session (token discipline, inherited from `start.md` §1).
If unsure whether you need a module yet, do NOT load it.

| Phase | Enter when | Load |
|---|---|---|
| 0. Entry | routed here from `start.md` | `block_design_skills/00_flow.md` (this file) |
| 1. INTAKE | user has described the block they want | `block_design_skills/intake.md` |
| 2. DESIGN | INTAKE signed off | `block_design_skills/design.md` (+ system_design modules per §6 Module map) |
| 3. VALIDATE | DESIGN methodology signed off | `block_design_skills/validate.md` |
| 4. EXAMPLE | VALIDATE passed **and** the user accepted the example gate (asked first; default yes, skippable) | `block_design_skills/example.md` |
| 5. SKILL | block + test exist | `block_design_skills/skill.md` |
| — | the block VENDORS an external kernel (decided at INTAKE Step 5) | `block_design_skills/vendor.md` — the STATEFUL adaptation of the borrowed kernel (pulled in during DESIGN, checked in VALIDATE) |
| — | building toward / asking about registry submission | `block_design_skills/contrib.md` (tiers, naming, manifest, FUTURE submission gate) |

`contrib.md` is **reference**, not a phase — load it on demand when a
question about tiers, CRAN-flat naming, the `manifest.dcf` schema, or the
FUTURE submission/registry gate comes up.

---

## 3. SIGN-OFF GATES (methodology locked before ANY code)

Two stacked disciplines govern every transition (they are the TOP RULE +
SECOND RULE made operational):

- **Methodology before code (SECOND RULE).** INTAKE and DESIGN each end at
 a sign-off gate that FIXES a methodological choice. **No `.hpp`, no test,
 no example is staged until INTAKE + DESIGN are signed off.** Surface
 every implementation choice (sampler kind, marginalization, update order,
 constraint transform, VI optimizer, decision-relevant numerical thresholds) as a
 structured labeled-option question BEFORE coding — do not bury a choice
 inside staged code.

- **Stage, then AUTO-DELIVER (TOP RULE = staging is the delivery guard).** Every artifact is written
 to a **staging dir** first. Compile + library-test run + the example all run AUTOMATICALLY (no "go")
 against staging — the MANDATORY correctness verification (T0–T4 + Check #12 AD-twin for any
 hand-written gradient; not optional, not asked). Once ALL checks pass, the whole bundle MOVES to
 `blocks_local/<Block>/` **automatically — no "go"** — and you REPORT the final path. Staging guards
 delivery: if any check FAILS, the bundle STAYS in staging, you report what failed and STOP (a
 not-delivered block is the correct outcome for a failed one). The heavy cross-chain audit is NOT a
 correctness gate — submission-path extra, run ONLY on explicit request, NEVER auto-run.

The gates that DO require an explicit answer — the §0.5 GPL-3 gate, the per-phase DESIGN sign-offs,
the EXAMPLE offer — must not be passed by assuming permission from context; wait for the answer. (The
bundle MOVE is NOT one of these — it auto-delivers on all-pass.)

---

## 4. Inherited disciplines (from `start.md` §0)

These are LLM-independent and apply at every phase; do NOT restate them in
each module — they live in `start.md` §0 and are inherited here:

- **ASK-THE-USER DISCIPLINE.** Every design decision is a **structured,
 labeled-option** prompt. In Claude Code use `AskUserQuestion` (≤4 per
 call). In any environment without a structured-question tool, use the
 **markdown labeled-option fallback** from `start.md` §0 (`(a)/(b)/(c)…`
 in chat). Mark the standing choice **`(default)`**, NEVER
 `(Recommended)`. Always include an **"Other / custom"** fallback. The
 asking is mandatory; only the mechanism is environment-dependent — never
 silently fall through to a default.

- **PLAIN LANGUAGE — the user is a statistician / applied researcher, not a library
 developer.** EVERY piece of user-facing text — option labels AND descriptions AND
 explanatory prose AND any draft you show — is plain model / statistics language. Keep
 internal jargon OUT of what the user reads; it comes in two flavours:
   - **implementation / mechanism terms** — `shared_data`, `ctx`, `block_context`,
     "refreshable context", "sibling-sampled", "single NUTS kernel over (β, log σ)",
     "hand-written log-density", "analytic gradient";
   - **selector / catalogue plumbing** — "auto-selector", "Stage A/B", "candidate block",
     "routing key", "token cost", "loaded on every run".
 These are your design notes, not what the user reads. Example — "how is the hyperparameter
 λ handled?": **"Another block samples it"** / **"Fixed constant you set"** / **"This block
 samples it too"** — NOT "Refreshable context (sibling-sampled): λ read each sweep from
 shared_data via ctx." And when you describe the MODEL to the user, describe the MODEL (its
 formula, its priors, what it is good for), NOT how it is sampled. If a mechanism detail
 genuinely matters, add at most a short plain gloss in parentheses.

- **REVIEW-CONTENT GOES IN THE CHAT BODY, not in the option text.** When a gate asks
 the user to confirm / refine a substantial generated artifact (a `SelectWhen` draft, the
 captured math spec, a config snippet, a derived formula), FIRST print that artifact into the
 chat — clearly set off in a `>` blockquote / fenced code block / **bold** so it is easy to
 read — THEN fire the structured ask (env mechanism: `AskUserQuestion` in Claude Code, markdown fallback elsewhere) with SHORT options (`Use it (default) / Refine — tell me
 the change / Other`). Do NOT cram the full artifact into an option's description (it renders
 as small grey text that is hard to read). The option names the CHOICE; the chat body carries
 the CONTENT to review.

- **INTERACTION FEEDBACK — keep the user oriented, never a black box.** At
 every PHASE boundary, give a SHORT recap: what this phase DECIDED and what
 happens NEXT (one or two lines, not a wall of text). Ask ONE question (or a
 ≤4 batch) at a time — never a paragraph with questions buried in it — and
 after each answer, briefly REFLECT it back so the user sees it was captured.
 **Two summaries are MANDATORY**, each a compact table the user signs off on:
   1. the **DESIGN-LOCK summary** BEFORE any code is staged — math spec ·
      engine · geometry class · constraint mapping · the three-tier plan ·
      every elicited threshold. This IS the methodology sign-off.
   2. the **BUNDLE summary** BEFORE the final "go" — the files produced, the
      test PASS/FAIL numbers (verbatim), the `SelectWhen`, and where the
      bundle will land (`blocks_local/<Block>/`).
 Mirror the codegen flow's model-confirmation gate: the user should always
 know what was decided, what is being asked, and what comes next.
 **Batch adjacent confirmations** that lock RELATED choices into ONE
 structured prompt (e.g. the two DESIGN wiring sign-offs, or a phase's
 small clustered choices) rather than firing ~12 separate gates for one
 small block — the gates are mandatory, but adjacent ones can share a
 prompt. NEVER collapse a methodology gate (INTAKE spec, DESIGN algorithm)
 INTO a code-writing step, and never batch a "go" that authorizes a
 file-move or a compile with an unrelated design question.

- **TOKEN DISCIPLINE.** Load modules lazily (§2). Never read the whole
 `block_design_skills/` or `system_design_skills/` folder upfront.

- **CITE, DON'T RESTATE.** For shared design knowledge, point to the
 system_design modules (`system_design §N` → resolves via
 `skills/system_design.md`'s §N map to one small
 `system_design_skills/<module>.md`) and to `validator.md` /
 `constraints.md` / `codegen_priors.md`. Do NOT copy their content into
 block_design modules or into the block-local skill.

- **CONSISTENCY with the 2026-06-20 audited skills**: there is
 **no "Check #11.7"**; the dense metric is **diagnostic-driven
 escalation** (start diagonal, measure, escalate — NOT a dimension
 threshold); a **CENTERED scale-governed effect is a hard Check #24(a)
 FAIL** (non-centered reparam mandatory, no centered+dense escape hatch);
 **`use_three_phase_warmup` defaults false**. Do not re-introduce a
 cleared contradiction.

---

## 5. Output of this flow

A single **self-contained bundle** — one folder `<Block>/` holding EVERYTHING the block needs, NOT a
set of edits scattered into core `block_catalogue/index.md` / `Makefile` / `validator.md`. **Self-contained =
the whole `<Block>/` folder is ONE uploadable / downloadable unit** (the FUTURE registry ships exactly
this folder): any vendored third-party code lives INSIDE it (`vendor/`), so whoever downloads the folder
gets the block AND its borrowed kernel together — nothing to fetch separately (except the core library
it `Depends:` on). That is WHY vendored code goes in the block's own folder, not a shared tree.

```
./blocks_local/<Block>/
 <Block>.hpp # Tier-B block (implements the block_sampler contract)
 test_<Block>.cpp # library test (FD / parity / recovery / cross-chain-Rhat / stress)
 manifest.dcf # routing + metadata (DCF / DESCRIPTION-style)
 skills/<Block>.md # block-local skill (cites core, never restates)
 skills/<Block>_validation.md # OPTIONAL — the BL# checks (only if any; manifest ValidationSkill:)
 examples/<Model>.cpp # OPTIONAL frontend-independent C++ demo (ask first; int main; no Rcpp/pybind)
 vendor/<lib>/ # OPTIONAL pinned third-party source + its LICENSE — travels WITH the bundle
```

It is **inside** the AI4BayesCode tree (developing a block IS system
development) but in its own subdir, separate from vetted core `include/`.
The core→package sync does NOT package `blocks_local/` into `inst/` (MVP):
local is a development area; "shipping a block" is the FUTURE registry
path (`contrib.md`).

---

## 6. Module map (block_design_skills) + system_design modules used in DESIGN

**block_design_skills modules** (this flow):

| Module | Role |
|---|---|
| `00_flow.md` | THIS router: 5-phase backbone, lazy-load schedule, sign-off gates |
| `intake.md` | INTAKE: math spec · naming-uniqueness gate · `SelectWhen` · advisory novelty/benchmark · geometry classification + correctness gate |
| `design.md` | DESIGN: algorithm spec (Gibbs/NUTS/slice-ESS/VI) · constraint mapping · three-tier interface impl |
| `validate.md` | VALIDATE: library test scaffold · validator-check map (#1–#25) · compile+test on "go" (mandatory T0–T4 + Check #12 AD-twin) · cross-chain R-hat < 1.01 fixed · heavy audit = submission-path extra |
| `example.md` | EXAMPLE: **offered (ask first; default yes, skippable)** — ONE **frontend-independent** C++ demo `examples/<Model>.cpp` (`int main`, no Rcpp/pybind; cheapest end-to-end check + needed for registry submission) |
| `skill.md` | SKILL: block-local `skills/<Block>.md` (cites core) + `manifest.dcf` (13 fields) · §5b doc-consistency · §5c FINAL GPL-3 license check |
| `vendor.md` | VENDOR sub-skill (loaded ONLY if the block vendors a kernel): STATEFUL adaptation of the borrowed code — isolate global/`static` state, thread the block's `rng`, rebuild caches on `set_context` · minimal-diff staging + `PATCHES.md` · upstream license kept verbatim |
| `contrib.md` | REFERENCE: tiers (core/local/[FUTURE] downloaded) · CRAN-flat naming · manifest schema · FUTURE submission/registry gate |

**system_design modules** indexed during DESIGN (load only the one a step
needs — cite via "system_design §N", resolves through
`skills/system_design.md`):

| Need | system_design module | §N |
|---|---|---|
| six-method R contract + C++ `block_sampler` contract + three tiers | `interface.md` | §1, §2 |
| shared_data / Gibbs-DAG vs predict-DAG / refreshers / set_current / RNG / history | `dataflow.md` | §3–§9 |
| Jacobian discipline (`constraints::<kind>::wrap`; never hand-write) | `jacobian.md` | §10 |
| target-geometry legality gate (the correctness gate; validator can't catch it) | `geometry.md` | §11 |
| per-family conventions + the metric (diagonal/dense) + warmup decision | `families.md` | §13 |
| new-block 7-step lifecycle + pre-merge checklist + anti-patterns | `lifecycle.md` | §14–§17 |
| VI architectural extension | `vi.md` | §18 |

---

## 7. Quick orientation

1. The user is here to design a NEW block. If they have NOT yet described
 the block / model, acknowledge and STOP — do not pop questions
 speculatively (same trigger semantics as `start.md`).
2. Once they describe it, enter **INTAKE** and load
 `block_design_skills/intake.md`. Begin with the structured math-spec +
 naming + `SelectWhen` questions.
3. Proceed phase-by-phase per §2, honoring every sign-off gate (§3) and
 the inherited disciplines (§4).

The recurring villain at every phase is the **SILENT correctness bug** — a
block that compiles, runs, and returns a WRONG posterior. Geometry
violations (INTAKE) and methodology errors (DESIGN) are exactly that class:
no runtime diagnostic catches them. Lock them at the gate, or never.
