<!-- block_design MODULE -- contrib.md -- CONTRIB ARCHITECTURE reference (load on demand).
 ROLE: where a local block bundle lives, how the system DISCOVERS + auto-selects it, and
 the FUTURE submission/registry gate. Loaded only when a flow needs the contrib-architecture
 answer (naming gate in Stage 0, bundle location in Stage 4, discovery/auto-selection at the
 model-confirmation gate). This module is self-contained (the historical design record is provenance only).
 CITE, don't restate: per-block skill split -> skill.md Sec.1; manifest fields -> skill.md Sec.3;
 check numbers + metric/warmup policy -> the system_design modules + validator.md / constraints.md. -->

# block_design -- contrib architecture

This is the "where does my block live, and how does the system find it?" reference. It does
NOT teach you to build the block (that's the Stage flow) -- it explains the contrib model the
bundle slots into. The recurring villain everywhere below is the SILENT correctness bug:
a block that compiles, runs, and returns the wrong posterior. The architecture is shaped to
keep such a block (a) isolated from vetted core and (b) impossible to ship without evidence.

---

## 1. Three tiers -- MVP is core + local; downloaded is [FUTURE]

| Tier | What it is | Who edits it | Trust |
|---|---|---|---|
| **core / default** | team-vetted blocks shipped with AI4BayesCode (`block_catalogue/index.md`) | maintainer-only | high |
| **local** | a block YOU build via this flow (work-in-progress) | you, freely | self |
| **downloaded** | installed from the shared registry into the **user-global** store `~/.AI4BayesCode/blocks_download/<Block>/` (one shared store across R / Python / C++ and all projects; override root with `$AI4BAYESCODE_DATA_HOME`) via `ai4bayescode_install_block()`. The install client exists today; the registry SERVER + AI-review acceptance gate remain **[FUTURE]** | immutable + versioned | provenance / badge |

**MVP scope = core + local + the install client.** `ai4bayescode_install_block()` already
downloads a reviewed bundle into the user-global `~/.AI4BayesCode/blocks_download/`, and codegen
discovers it (see `block_catalogue/index.md`). What stays deliberately deferred -- the "running a
stranger's code" surface -- is the registry SERVER, the AI-review acceptance gate, use-time
sandboxing, and cross-version `block_sampler` interface stability. None of those arise for a
block you wrote and run yourself.

Lifecycle (one line through the tiers):
```
block_design -> local block -> (submit, justify -> registry) -> downloaded by others
 ^MVP ends here ^[FUTURE]
```

Why the tiers are SEPARATE (not one pile): trust boundary, license isolation, namespace
hygiene, distribution control, maintenance boundary -- the CRAN-flavored vision.

---

## 2. Location -- inside the tree, beside core, NOT shipped

Building a block IS system development (unlike a user's `./generated/` model output), so a
local block lives INSIDE the AI4BayesCode tree -- but in its OWN subdirectory, walled off from
the vetted core `include/`:

```
./blocks_local/<Block>/  # PROJECT-RELATIVE (your project -- or the AI4BayesCode tree
                         # if you are contributing to the library); separate from core include/
```

- A local block is a **self-contained bundle** (`<Block>.hpp` + `test_<Block>.cpp` +
 `examples/<Model>.cpp` + `manifest.dcf` + `skills/<Block>.md`, plus optional `vendor/` +
 `benchmark/`). It does NOT
 edit core files -- it does not scatter rows into `block_catalogue/index.md` / the Makefile /
 `validator.md`. Core stays untouched.
- **The core->package sync does NOT auto-package `blocks_local/` into the R-package `inst/`.**
 Local is a dev area; "shipping a block" is the [FUTURE] registry/submission path. (Same
 rule as the `.bak` / `_archive` dev-only strip: dev content stays in dev, never ships.)
- Compile include-path wiring for `ai4bayescode_sourceCpp` (putting `blocks_local/*/` on
 `-I`) is an [OPEN] integration detail -- staging/compile is a Stage-4/5 "go" action, not
 something this reference resolves.

---

## 3. Naming -- CRAN-flat, unique, core-reserved, provenance recorded

Names are **globally unique flat names** (no `@author/` scope), registry-first-come
[FUTURE] -- fitting the R-centric ecosystem.

- **Core names are RESERVED.** A contrib block may not take a core block's name.
- Convention follows existing blocks: `snake_case` + `_block` suffix, so contrib blocks look
 native (e.g. `poisson_icar_ess_block`).
- The name is the unique key and the single source of identity -- `ClassName`, header
 `<Block>.hpp`, and bundle dir `blocks_local/<Block>/` are all DERIVED from it (see
 the manifest `Block:` field conventions, `skill.md Sec.3`).
- Flat naming carries typosquatting/grab risk, mitigated by the manifest's `Author` /
 provenance record (+ trust badges [FUTURE]).
- **block_design checks uniqueness AT CREATION** against `core  U  existing local` (this is the
 Stage-0 naming-uniqueness gate). The registry-availability check is [FUTURE].

---

## 4. Discovery + auto-selection -- the context-scaling solution

The bottleneck is **LLM context**, not disk: you cannot pour hundreds of block descriptions
into the agent to pick one. Selection is therefore **two-stage and fully automatic -- there is
NO mid-flow picker**.

```
Stage A deterministic facet pre-filter hundreds -> dozens
 (drop blocks whose engine / constraint / geometry clearly can't fit)
 CONSERVATIVE: recall > precision -- when in doubt, KEEP it
Stage B AI reads the surviving SelectWhen triggers, AUTO-selects ONE
 no match -> auto-fallback to the default joint_nuts_block / standard path
```

- **What makes Stage B work is `SelectWhen`.** Facets (`EngineKind` / `ConstraintKinds` /
 `RoutingKey`) are too terse for the AI to match a free-form model to the right block. The
 crisp "use this WHEN `<model structure>`" trigger is the decision-relevant text. block_design
 FORCES a clear `SelectWhen` per block -- a vague one means the block is never selected (it's a
 bundle quality bar, not boilerplate).
- **A compact auto-generated index** is what the agent actually reads: ONE short line per
 block (`Block | EngineKind | ConstraintKinds | RoutingKey`), aggregated from core
 `block_catalogue/index.md` + every `blocks_local/*/manifest.dcf`, auto-rebuilt on add/remove (no
 manual drift). `SelectWhen` is loaded only for the Stage-A survivors. Context stays bounded
 regardless of block count N -- ONE small file, not N files, not N prose entries.
- When a local block is auto-selected, its block-local skill (`skills/<Block>.md`) loads
 **on demand alongside the block code** -- never preloaded (token discipline; see skill.md Sec.4,
 the two-stage selection/loading contract).

**Scale-up path [FUTURE]** (thousands of blocks, or SelectWhen matching no longer enough):
family-level categorization for two-level narrowing, or embedding/RAG retrieval over
SelectWhen + descriptions.

### 4a. Confirmation folds into the EXISTING model-confirmation gate

No new interruption is added. The auto-selected block surfaces in the EXISTING
model-confirmation summary (`start.md` Stage 3 / `codegen.md` Sec.2 -- inline DAG + summary table +
sign-off). Below the summary table, **LIST the alternative candidate blocks** (a plain list for
switching -- no per-block rejection prose):

```
[Model confirmation -- before writing code]
 model / DAG / priors...
 selected block:
 * poisson_icar_ess_block (AI auto-selected)
 alternative blocks (to switch): gmrf_precision_block * joint_nuts_block (default)
 -> confirm / what to change?
```

This **refines** `codegen_priors.md Sec.2b`'s older "block-selection disambiguation -> ask the user
(mid-flow)" into "**auto-select + confirm at the one model-confirmation gate**" (one
consolidated gate, not scattered prompts). When this flow is wired in, KEEP Sec.2b's wording in
sync so the two do not re-create an audit-style contradiction.

---

## 5. Submission gate -- justify on upload [FUTURE; criterion LOCKED]

**Local creation has NO gate -- it is ENABLING.** The novelty/benchmark question is surfaced as
ADVISORY info in Stage 0, never as a blocker. The gate lives at **submission to the shared
registry**, which is exactly where proliferation + the index-scaling problem originate; gating
at the source keeps the shared pool curated. (Gate and retrieval Sec.4 are complementary: the gate
keeps the pool small, retrieval handles whatever accumulates.)

**Acceptance criterion (LOCKED).** Novelty is necessary but NOT sufficient -- the deciding
evidence is a benchmark. The bar is NOT "can't be composed from existing blocks" (too strict:
that would reject `gmrf_whitened_ess_block`, which IS composable via `joint_nuts_block` but is
~10000x slower on BYM2-shape targets):

> A new block is justified iff it **cannot be composed** from existing blocks, **OR** the
> existing composition is **benchmark-dominated** (orders-of-magnitude worse wall-time / ESS).
> Composable AND not-dominated -> reject, route to codegen composition.

**Acceptance mechanism [FUTURE, deferred].** CRAN has a human team; we do not, so the gate is a
LAYERED automated check designed NOT to rely on "an AI says yes":
- **Deterministic backbone (primary):** compile; bundled tests pass; validator static checks
 (#1-#26 as applicable -- see `validator.md`); **sandboxed re-run** of the submitted
 benchmark/tests (reproduce, don't trust submitted numbers).
- **AI adversarial concurrence:** multiple independent reviewers (the audit-workflow pattern)
 for semantic correctness, novelty/quality, and security source review.
- **Sandbox is the real security net** (running third-party C++ is a supply-chain surface):
 isolated compile+run, no network, no FS beyond scratch, resource/time limits. AI source
 review only FLAGS; the sandbox CONTAINS.
- **Optional human spot-check** for high-download or AI-uncertain submissions -> trust badge.
- **Trust tiers/badges** surfaced at download (AI-auto-accepted vs human-verified).

None of this executes in the MVP -- it is the shape the registry tier will take.

---

## 6. Where the per-block skill fits (pointer, not a restatement)

Every bundle carries its OWN block-local skill `skills/<Block>.md` holding knowledge SPECIFIC
to that block (its routing row / when-to-use / when-NOT / config snippet / example / its
specific silent-failure modes); GENERAL conventions stay in core and the block-local skill
CITES them, does not restate. For the full local-vs-core delimitation table see **skill.md Sec.1**,
and for the discovery/loading rule see **skill.md Sec.4** -- do not duplicate them here.
