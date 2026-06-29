# module: skill.md — block_design SKILL phase — author the block-local skill `skills/<Block>.md` + the `manifest.dcf`

<!--
 block_design MODULE. Role: the SKILL phase (follows constraint mapping;
 this phase runs AFTER the interface impl is staged, BEFORE final move/compile).
 This module is self-contained (the historical design record is provenance only).
 Key sections: §1 (skill delimitation), §3 (manifest.dcf), §4 (SelectWhen / two-stage selection).
 LAZY-LOAD: this module is read ONLY when the SKILL phase is entered. Do not preload
 sibling block_design modules. Cite core (system_design modules / constraints.md /
 validator.md / codegen_priors.md) — do NOT restate them.
-->

## 0. What this phase produces (2 artifacts always + a SEPARATE validation skill if `BL#` exist)

This phase fills the final documentation artifacts of the bundle (the `.hpp` + `test_*.cpp`
were staged in earlier phases):

1. `skills/<Block>.md` — the **block-local skill**: its catalogue entry (routing / when-to-use /
 when-NOT / config snippet). It CITES core for everything shared; it restates nothing. **The
 block-SPECIFIC `BL#` checks do NOT go here** (§2.2) — they go in artifact 3.
2. `manifest.dcf` — the **DCF manifest** (13 required fields) that drives discovery, compile, and
 (future) registry submission.
3. `skills/<Block>_validation.md` — a **SEPARATE block-local validation skill** holding the
 block-SPECIFIC `BL#` checks, **authored ONLY if the block has any** (a block whose only checks are
 core `#1`–`#25` skips it). It is pointed to by the manifest **`ValidationSkill:`** field. It is kept
 SEPARATE from the main skill ON PURPOSE: the main skill is loaded on every block selection / use,
 but the `BL#` checks are needed only at VALIDATE / re-validation / audit — so splitting them out
 (lazy-loaded via `ValidationSkill:`) saves tokens in the common case. A block with NO block-specific
 checks omits BOTH the file and the field.

All are STAGED to `.block_design_staging/<Block>/`, never written in place. The
library test + example already compiled + ran AUTOMATICALLY in VALIDATE / EXAMPLE; once ALL checks
pass the whole bundle MOVES to `AI4BayesCode/blocks_local/<Block>/` **automatically** (no "go") and
the final path is REPORTED. Staging is the delivery guard — a FAILED block is never moved (TOP RULE).

Every decision in this phase is a structured labeled-option prompt (AskUserQuestion, or
the markdown labeled-option fallback from `start.md §0`); the standing choice is marked
**"(default)"**, never "(Recommended)"; always offer "Other".

---

## 1. The DELIMITATION RULE — decide it BEFORE writing a line of skill prose

The block-local skill carries knowledge **specific to THIS block**. Everything general
stays in core skills and the block-local skill **CITES** them. The one-line rule:

> **specific-to-THIS-block → block-local** (travels with the bundle);
> **general-to-a-class / all-blocks → core** (cited, not restated).

| Knowledge | Belongs to | If core, cite |
|---|---|---|
| this block's routing row · WHEN-to-use · WHEN-NOT · geometry class · config-struct snippet · example path · **its specific** failure modes | **block-local** (`skills/<Block>.md`) | — |
| constraint-transform table, the 15 `joint_constraint` kinds | core | `constraints.md` |
| block-selection framework / Exception taxonomy | core | `codegen_priors.md §2b` |
| three-tier interface contract (six R methods + readapt_NUTS) | core | `system_design §1` (`interface.md`) |
| metric (diagonal→dense) + warmup policy | core | `system_design §13` (`families.md`) |
| general validator check DEFINITIONS (#1–#25) | core | `validator.md` — EXCEPT #15–#17, defined in `codegen_priors.md §2c–§2e` |
| target-geometry legality (§11.1's three shapes) | core | `system_design §11` (`geometry.md`) |
| a silent-failure check **SPECIFIC to this block** | **block-local** (scoped) | — |
| a **GENERAL** new check applying to a CLASS of blocks | core (maintainer) | `validator.md` |

**Test for each candidate paragraph:** "Would this sentence be true for ANY block of this
family, or only for THIS one?" Family-true ⇒ delete it and cite core. Only-this-block-true
⇒ keep it block-local. This is the 2026-06-20 audit's "cite, don't duplicate" applied to
the bundle.

**The recurring villain:** the SILENT correctness bug (compiles + runs + wrong posterior).
The block-local skill exists mainly to name THIS block's silent-failure modes — the things
that pass every MCMC diagnostic yet sample the wrong distribution. If you restate generic
material instead, you bury the one thing only this skill can warn about.

---

## 2. Author `skills/<Block>.md` (the block-local skill)

YAML-ish frontmatter (`name:` + one-line `description:`), then the sections below. Fill
every `{{SLOT}}` from the answers gathered in Stages 0–4 (math spec, geometry class,
algorithm, constraints, config struct). Keep it tight — this file is loaded **on-demand**
when the block is auto-selected, so it pays its own token cost every time.

### 2.1 Catalogue entry — the parts the selector and the codegen agent read

**(a) Routing row** — ONE row in the core `block_catalogue.md` 3-column shape
(`| parameter kind | block type | constraint wrap |`). Without this row the codegen agent
cannot route a parameter to the block. Mirror the existing rows' density (parenthetical
target-distribution sketch in the first cell). Example shape to match:

```
| **<one-line parameter kind / target sketch>** | **`<Block>`** (<algo + citation>) | **(<constraint wrap or "none — …">)** |
```

**(b) WHEN-to-use** — the crisp model-structure trigger. This is the SAME text that becomes
the manifest `SelectWhen` (§4 below); keep them identical so Stage-B selection and the
human-readable skill never drift.

**(c) WHEN-NOT-to-use** — EXPLICIT. Name the neighbor block(s) that look similar and the
boundary that separates them (e.g. "Gaussian observation ⇒ use `gmrf_precision_block`, not
this"). A missing WHEN-NOT is the #1 cause of mis-selection. Required, not optional.

**(d) Geometry class** — which of `system_design §11.1`'s three shapes the target is:
fixed-dim ℝ^d (after constraint transforms) · simplex · trans-dimensional/§11.2. State it in
one line and CITE `geometry.md` — do not re-explain the legality gate. (If trans-dim, the
block faces Check #25; if a `joint_nuts_block` is composed alongside, Check #24.)

**(e) Config-struct snippet** — a `<Block>_config cfg;` block showing every `_key`/callback
field with a one-line `// …` purpose comment, exactly mirroring the staged `.hpp`. This is
the field map the codegen agent fills. Follow the `gmrf_whitened_ess_block` catalogue
snippet as the template (config name = `<Block>_config`, `cfg.name`, the user callbacks,
the invariant flags). Show a one-line "typical composite" comment (what hyperparameter
blocks it pairs with) — do NOT write a full worked example here; that lives in `examples/`.

**(f) Example path** — `examples/<Model>.cpp`, the REQUIRED Tier-A reference example (every block
ships exactly ONE — the EXAMPLE phase is mandatory, `00_flow.md` §1). The file is discovered as the
bundle's `examples/<Model>.cpp`; optionally record its path in the manifest's OPTIONAL `Example:`
field (§3.2) for non-standard names.

### 2.2 Block-SPECIFIC failure modes (`BL#`) — in a SEPARATE validation skill, NOT this file

Each silent-failure mode unique to THIS block's mechanism becomes a FIRST-CLASS block-local
semantic check `BL1`, `BL2`, … (block-scoped IDs — never collide with core `#1`–`#25`), written
with the SAME anatomy as a core check: **Trigger · Why** (the silent bug; why it passes R̂ / ESS /
LOO) **· What to look for** (`// WRONG` vs `// RIGHT`, a `grep -nE` pattern, or a runtime
assertion) **· Fix**. Mark each *runnable* (an assertion / regime in `test_<Block>.cpp`) or
*static* (a code-review grep).

**These checks DO NOT live in this main skill file.** They are needed only during VALIDATE /
re-validation / audit — rarely, NOT on every block selection or use — so to keep the always-loaded
main skill lean, **write them in a SEPARATE `skills/<Block>_validation.md`** and point the manifest
`ValidationSkill:` field at it. Author that file ONLY if the block actually has block-specific
checks; a block with NONE simply omits `ValidationSkill:` (and VALIDATE then runs only the core
`#1`–`#25`). This main file keeps just a ONE-LINE pointer ("block-specific checks: see
`ValidationSkill`") plus, in §2.1, any brief *usage* pitfall a composer genuinely needs. List the
`BL#` IDs in the manifest `ChecksApplicable`. They travel with the bundle; a `BL#` that turns out
GENERAL is the maintainer's to promote into core `validator.md`, not an auto-append.

Draw them from the Stage 1–3 answers. Typical sources (keep only the ones that apply):

- **Invariant violations** the algorithm must preserve (e.g. sum-to-zero under a linear
 proposal; ordering; normalization) and exactly where they are enforced.
- **Callback-contract** silent breaks (e.g. a user `Q_fn` that returns a non-PSD / asymmetric
 matrix, a `log_lik` that returns non-finite) — these compile and run.
- **Scope assumptions baked into the kernel** (e.g. "symbolic factorisation cached ⇒ sparsity
 pattern assumed fixed across steps") that produce wrong answers if violated.
- **Convergence-budget reality** — a one-line caveat that the user must pilot the budget for
 their topology/likelihood (cite the block's own test results, never invent numbers).

For any **generic** failure mode (Jacobian discipline, RNG separation, dense-metric misuse,
non-centering of a centered scale-governed effect = hard Check #24(a) FAIL), DO NOT restate
it — cite the defining file (`validator.md` check #N, or the relevant `system_design` module)
and move on. Per the 2026-06-20 audit: no phantom "Check #11.7"; dense metric is a
diagnostic-driven escalation (start diagonal, measure, escalate), NOT a dimension threshold;
`use_three_phase_warmup` defaults false.

### 2.3 The closing CITE block (mandatory)

End the file with an explicit "shared conventions live in core" pointer list, so the file
visibly obeys the delimitation rule:

```
## Shared conventions (cited, not restated)
- interface contract (6 R methods + readapt_NUTS): system_design §1 (interface.md)
- constraint transforms / the 15 joint_constraint kinds: constraints.md
- block-selection / Exception taxonomy: codegen_priors.md §2b
- metric + warmup policy: system_design §13 (families.md)
- validator checks this block faces: validator.md (#1–#14, #18–#25 as applicable); codegen_priors.md §2c/§2d/§2e (#15/#16/#17 as applicable)
- geometry legality gate: system_design §11 (geometry.md)
```

---

## 3. Author `manifest.dcf` (reproduce this exactly)

DCF format (`Key: Value`, continuation lines indented one space — the format R's
`DESCRIPTION` uses). **Convention over configuration + single name source:** `Block` is the
ONE identity. Three derivations are ENFORCED CONVENTIONS, NOT fields:

- **ClassName** `== Block` (the C++ `block_sampler` subclass symbol).
- **Header** `== <Block>.hpp`.
- **Bundle dir** `== blocks_local/<Block>/`.

### 3.1 Required fields (13)

| Field | Purpose | Notes |
|---|---|---|
| `Block` | unique flat name (registry key) | snake_case + `_block`; not core-reserved; unique vs core+local; **derives** ClassName/Header/bundle-dir |
| `Version` | semver | default `0.1.0` |
| `Title` | one-line human title | short |
| `Description` | **detailed, human-facing**, in **CRAN-package `Description:` style** (concise complete sentences, present tense, no marketing) — states WHAT the block IS: the model FIRST (likelihood / formula), then priors, then a brief high-level sampling note. **NOT read by the auto-selector.** See `intake.md §3a`. | the human "what is this" |
| `Author` | provenance | name(s) (flat naming → author recorded to deter squatting) |
| `License` | the block's own license = **`GPL-3.0-or-later`** — every generated block inherits the project license (README "Licensing"); vendored upstreams are recorded in `Vendored:` + gated at INTAKE Step 5 | inherited, not chosen |
| `EngineKind` *(facet)* | `mcmc` / `vi` / future | matches `engine_kind_t` |
| `ConstraintKinds` *(facet)* | which `joint_constraint` kinds | from the 15 (`constraints.md`); or `none` |
| `RoutingKey` *(index label)* | short noun-phrase for the compact routing index | the Stage-A line |
| `SelectWhen` *(Stage-B core; AI-read)* | crisp, **COMPACT** "use this WHEN \<model structure\>" trigger, **~10 words** — the ONLY block-specific text the auto-selector loads at scale; keep it tight (token cost scales with block count). NOT the human description (that is `Description`). | selection-quality field; **vague ⇒ never selected** |
| `Skill` | block-local skill path | KEPT EXPLICIT (forcing function + allows non-standard/multiple); convention `skills/<Block>.md` |
| `Tests` | library test path(s) | KEPT EXPLICIT (forcing function + allows multiple); convention `test_<Block>.cpp` |
| `Depends` | core version floor | `core (>= x.y)`; v1 depends on core ONLY — **no contrib→contrib** |

### 3.2 Optional fields

| Field | Purpose | When |
|---|---|---|
| `Example` | Tier-A reference path | if `examples/<Model>.cpp` authored |
| `LabelSwitching` | `true` if the target is permutation-invariant over K exchangeable components (finite mixture / HMM / LDA / BNP clusters) → its draws need POST-MCMC relabeling; `false` / absent otherwise | **conditionally REQUIRED: set `true` whenever the block is exchangeable** (design.md Stage 2 rule). Signals the analysis / runner layer to relabel per `label_switching.md`; the block itself NEVER resolves labels |
| `Maintainer` | contact | **REQUIRED at submission** [FUTURE] |
| `ChecksApplicable` | core `#N` faced + block-local `BL#` defined | informational locally; submission uses it |
| `ValidationSkill` | path to the block's **separate** block-specific-check skill `skills/<Block>_validation.md` (holds the `BL#` checks) | **OPTIONAL — present iff the block has any `BL#`**; absent ⇒ VALIDATE runs only core `#1`–`#25`. Kept OUT of the main `Skill` so block selection / use never load it; VALIDATE lazy-loads it only when this field is set (token saving). |
| `Benchmark` | ESS/s vs baseline | **REQUIRED at submission** [FUTURE] |
| `Vendored` | NEW third-party code the block borrows into its OWN `vendor/<lib>/` dir (upstream name + license + path) | OPTIONAL — present iff the block vendors code (`intake.md` Step 5 / `vendor.md`). NOT the FIXED system libs (mcmclib / Eigen / celerite / libgp), which are on `-I`, just `#include`d, and NEVER declared per-block. |
| `KernelTier` | whether the block carries its OWN **Tier-C kernel** code + provenance: `none` (pure Tier-B block — only sampler logic / callbacks / `#include`s of FIXED system libs), `own` (a hand-written low-level numerical kernel), or `vendored` (a borrowed kernel staged in `vendor/`) | OPTIONAL; default `none`. `vendored` cross-refs `Vendored:` and triggers `vendor.md` (stateful adaptation) + the VALIDATE vendored-kernel stateful check; `own` / `vendored` flag a bigger correctness surface (deeper review). |

### 3.3 DROPPED fields (do NOT add them)

- `TargetGeometry` — coarse; the flow's geometry gate already pre-routes (geometry is named
 in the block-local skill §2.1(d) for humans, not duplicated as a manifest field).
- `ClassName`, `Header` — derived 1:1 from `Block` (enforced convention, §3 above).
- `VendoredDeps` — the system's vendored set is FIXED and already on `-I`; per-block
 declaration is redundant for the build. (The optional `Vendored` line above is purely
 informational/provenance, not a build input.)

### 3.4 Filled example (locked example — reproduce this shape)

```
Block: poisson_icar_ess_block # ⇒ class poisson_icar_ess_block; header <Block>.hpp;
 # bundle blocks_local/<Block>/
Version: 0.1.0
Title: Elliptical-slice sampler for sparse-GMRF latent fields
Description: ESS (Murray 2010) on an implicit GMRF prior via sparse-Q backsolve,
 plus a user log-likelihood for non-Gaussian observations.
Author: Jungang Zou
License: GPL-3.0-or-later
EngineKind: mcmc
ConstraintKinds: none
RoutingKey: sparse-precision Gaussian MRF
SelectWhen: latent field has sparse GMRF precision (ICAR/BYM2) + non-Gaussian
 (e.g. Poisson) likelihood; joint NUTS is orders of magnitude slower here.
Skill: skills/poisson_icar_ess_block.md
Tests: test_poisson_icar_ess_block.cpp
Depends: core (>= 1.0)
```

### 3.5 Pre-stage validation (do these BEFORE staging the manifest)

- **Name uniqueness**: `Block` is not a core-reserved name and collides with
 no existing `blocks_local/*/manifest.dcf`. This was first run at Stage 0; re-confirm here.
- **License gate**: `License` is **`GPL-3.0-or-later`** (inherited). Any VENDORED upstream must be
 GPL-3-compatible — the combined work is ALREADY GPL-3.0-or-later, so Apache-2.0 is fine with no
 special branch. If the block `#include`s GPL kernels (BART/genBART), flag the transitive GPL like
 the core catalogue tags.
- **SelectWhen quality bar**: read it back as "use this block WHEN …" — if it could match
 many unrelated models, it is too vague and the block will never be selected. Tighten it
 (elicit a sharper trigger) before staging.

---

## 4. Finalize SelectWhen (the two-stage selection contract)

Selection of a block at codegen time is **two-stage, fully auto** (no mid-flow picker); the
manifest supplies BOTH stages' inputs:

- **Stage-A — facet pre-filter (the compact-index line).** Derived deterministically from
 `Block | EngineKind | ConstraintKinds | RoutingKey`. This one short line per block goes
 into the auto-rebuilt compact index (aggregated from core `block_catalogue.md` +
 `blocks_local/*/manifest.dcf`). Stage A is CONSERVATIVE (recall > precision): it drops
 only blocks whose engine/constraint/geometry clearly can't fit; when in doubt, keep.
 → So `RoutingKey` must be a faithful, terse facet of WHAT the block targets.

- **Stage-B — `SelectWhen` trigger.** Loaded only for Stage-A survivors. The AI reads the
 surviving `SelectWhen` lines and AUTO-selects ONE (no match → fallback to the default
 `joint_nuts_block` / standard flow). → So `SelectWhen` must be the crisp, decision-relevant
 "use this WHEN \<model structure\>" sentence. A vague `SelectWhen` = a block that never
 gets selected.

**Two fields, two audiences — do NOT conflate.** `Description` is the **human-facing** detail
(what the block is, read by people in the catalogue; never loaded by the selector, so it may be a
full paragraph). `SelectWhen` is the **AI-selector-facing** trigger (the ONLY block-specific line
Stage-B loads — with hundreds of blocks it MUST stay COMPACT to keep selection token-cheap). Put
the detail in `Description`; keep `SelectWhen` to the one decision-relevant sentence. Do NOT offer
a "full vs concise SelectWhen" choice — the concise trigger IS `SelectWhen`; the fuller prose IS
`Description`.

**Identity rule:** the skill's WHEN-to-use (§2.1(b)) and the manifest `SelectWhen` are the
SAME sentence. Author once; reuse. If you sharpen one, sharpen the other.

**Confirmation — no new interruption.** The auto-selected block surfaces in the
EXISTING model-confirmation gate (`codegen.md §2`), with the alternative
candidate blocks listed below the summary table for one-click switching. This phase does NOT
add a picker prompt; it only ensures the manifest's `RoutingKey`/`SelectWhen` are good enough
to make that auto-selection correct. (When the codegen skills are next edited, `codegen_priors
§2b`'s wording is synced to "auto-select + confirm at the gate" — out of scope for this phase,
flagged for the maintainer.)

---

## 5. End-of-phase checklist (before offering the "go" to move + compile)

- [ ] `skills/<Block>.md` has: routing row · WHEN-to-use · explicit WHEN-NOT · geometry class
 (cited) · config snippet · example path · closing CITE block. (The `BL#` checks are NOT here —
 they live in the separate `ValidationSkill`; see the next item.)
- [ ] **Validation skill:** if the block has any `BL#`, `skills/<Block>_validation.md` is authored +
 staged with the `BL#` (Trigger / Why / What-to-look-for / Fix), and the manifest `ValidationSkill:`
 points at it. If the block has NO `BL#`, NEITHER the file nor the `ValidationSkill:` field exists.
- [ ] No paragraph in the block-local skill restates a core convention (delimitation rule §1).
- [ ] `manifest.dcf` has all 13 required fields; ClassName/Header/bundle-dir match the `Block`
 convention; no DROPPED field present.
- [ ] `Skill` and `Tests` paths point at the staged files; `SelectWhen` == skill WHEN-to-use.
- [ ] Name-uniqueness + license-gate + SelectWhen-quality re-confirmed (§3.5).
- [ ] Both artifacts are in `.block_design_staging/<Block>/`, NOT in `blocks_local/`.

## 5b. DOCUMENT CONSISTENCY CHECK — read the docs back against the CODE (not just structure)

The checklist above is mostly STRUCTURAL (sections present, paths resolve, fields complete). This
step is SEMANTIC: **read each generated document back against the actual `.hpp` / `.cpp` and the
INTAKE-signed spec, and confirm they MATCH.** Docs drift from code silently, and a correct block
with a wrong manifest / skill misroutes and misleads every future user — so a doc mismatch is a FIX
before "go", the SAME bar as a failed code check.

- [ ] **Description ↔ model.** The `Description`'s model (likelihood / formula / priors) is the model
  the `.hpp` actually implements AND the spec signed off at INTAKE — not a paraphrase that drifted.
  Same for the header doc-comment's MODEL block.
- [ ] **SelectWhen ↔ block.** The trigger is ACCURATE (not merely non-vague): a model matching it
  really should pick this block; one that does not really should not.
- [ ] **Facets ↔ code.** `EngineKind` == `engine_kind()`; `ConstraintKinds` == the constraint
  transforms the block actually applies (or `none`); `LabelSwitching` == the block's exchangeability;
  `KernelTier` matches reality (`vendored` ⟺ a `vendor/<lib>/` dir + a `Vendored:` entry; `own` ⟺ a
  hand-written low-level kernel; otherwise `none`).
- [ ] **BL# 3-way consistency.** The `BL#` defined in the `ValidationSkill` (or §2.2) == the `BL#`
  listed in `ChecksApplicable` == the `BL#` actually exercised by a regime / assertion in
  `test_<Block>.cpp`. No `BL#` declared-but-untested or tested-but-undeclared.
- [ ] **Config snippet ↔ `.hpp config`.** The config example matches the real `config` struct's
  fields + defaults (no stale field name, no wrong default).
- [ ] **Paths + cross-refs resolve.** `Skill` / `Tests` / `Example` / `ValidationSkill` files exist
  on disk; every `system_design §`, `validator.md #`, and file path the docs cite actually exists.
Report the read-back result (what you checked, what matched). Any mismatch → fix, then re-check.

## 5c. FINAL STEP — GPL-3 license check (the VERY LAST gate, after everything)

**This is the last-last check of the entire block_design flow — run it AFTER the VALIDATE runtime
(T0–T4), AFTER the EXAMPLE phase, and AFTER the §5b doc-consistency checks, immediately before the
auto-delivery move into `blocks_local/`.** (It is the final check; passing it is what lets the
automatic move proceed.) It closes the loop with the `00_flow.md §0.5` GPL-3 gate the user agreed to at the
start: they promised GPL-3.0-or-later up front; here you VERIFY every artifact actually carries it.

Read EVERY file the user authored and confirm each carries the **GPL-3.0-or-later** header / field:
- the block `.hpp` header comment,
- `test_<Block>.cpp` header,
- `examples/<Model>.cpp` header (if the example was authored — it is OPTIONAL),
- the manifest `License:` field == `GPL-3.0-or-later`,
- `skills/<Block>.md` and `skills/<Block>_validation.md` (if present).

The ONLY exception is **vendored** code under `vendor/`: it KEEPS its own upstream license verbatim
(recorded in `Vendored:`) and is NEVER relicensed to GPL-3 — verify those headers are UNCHANGED.

**Any authored file missing / mis-stating the GPL-3.0-or-later header, OR any vendored file whose
upstream license was altered, is a HARD FIX before "go" — a license error is not shippable.** This
gate has NO loophole, same as §0.5.

Then **deliver the bundle automatically — no "go" gate.** Once EVERY check has passed (VALIDATE
T0–T4 + Check #12 + any vendored-correctness / stateful checks; the example if authored; the §5b
doc-consistency; the §5c GPL-3 license check), MOVE the whole bundle from staging into
`AI4BayesCode/blocks_local/<Block>/` and **REPORT the final path + the files written** (e.g.
"Block created at `…/blocks_local/<Block>/` — `<Block>.hpp`, `test_…`, `manifest.dcf`, `skills/…`,
`vendor/…`"). Do NOT ask for a "go".

Staging still matters as the DELIVERY GUARD, not a gate: build + run in staging, and move ONLY on
all-pass. **If ANY check FAILED, do NOT move** — leave the bundle in staging, report exactly what
failed, and STOP (the prime directive: a not-delivered block is the correct outcome for a failed
one). The heavy cross-chain / multi-seed audit is NOT a correctness gate — submission-path extra,
run ONLY on explicit request, NEVER auto-run.
