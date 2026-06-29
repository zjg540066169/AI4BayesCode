---
name: AI4BayesCode-system-design
audience: |
  SYSTEM DESIGN agents (designing / auditing blocks) AND any agent that needs a
  specific architectural rule. This file is now a THIN INDEX: the content lives in
  `system_design_skills/` modules — load ONLY the module you need (lazy), never the
  whole guide. The codegen skills AND the block_design flow both index these modules
  via the §N map below. (The code-generating agent's primary flow is
  `skills/codegen.md`; it pulls the specific system_design module a topic needs.)
description: |
  Index + router for the AI4BayesCode system-design guide, split 2026-06-21 into
  lazy-loadable modules under `system_design_skills/`. Maps every §N to its module so
  existing "system_design §N" citations resolve to one small file. Covers the
  non-negotiable interface invariant (six-method R contract), three-tier separation,
  shared_data / Gibbs-DAG / predict-DAG semantics, Jacobian/constraint discipline, the
  target-geometry boundary, block-family conventions (incl. the metric/warmup decision),
  the new-block design lifecycle + pre-merge checklist, and the VI extension.
---

# AI4BayesCode — system design guide (INDEX / ROUTER)

**Split 2026-06-21 into lazy-loadable modules.** The full content now lives in
`skills/system_design_skills/`. THIS file is a thin index. To answer a system-design
question, load ONLY the module(s) you need below — do NOT load the whole guide.
**Edit the MODULES (single live source), not this index.**

**Project license.** AI4BayesCode is distributed under GPL-3.0-or-later (see `LICENSE` /
`COPYING` at the repo root). All code under `include/AI4BayesCode/`, `examples/`, `R/`,
`skills/`, `tests_autodiff/`, and all project-level documentation is GPL-3.0-or-later.
Every file you add or edit MUST carry a matching license header (see `skills/codegen_cpp.md`
§5 for the canonical three-line form). Vendored third-party code under
`include/{mcmclib,eigen,autodiff}/` and `bart_pure_cpp/` retains its own upstream license — do NOT
modify those headers. Compatibility notes are in `THIRD_PARTY_LICENSES.md`.

---

## §N → module map (load the module, NOT the monolith)

| Sections | Module (`system_design_skills/…`) | Topic |
|---|---|---|
| **§0** Audience · **§1** non-negotiable interface invariant (the six-method R contract: step / get_current / set_current / predict_at / get_dag / get_history, +7th readapt_NUTS) · **§2** three-tier architecture (wrapper / block / kernel) | `interface.md` | the non-negotiable interface contract |
| **§3** shared_data + block_context · **§4** Gibbs DAG vs Predict DAG · **§5** refreshers (deterministic / stochastic) · **§6** working-response / data-injection · **§7** set_current dispatcher · **§8** RNG discipline · **§9** history / keep_history | `dataflow.md` | data flow + wiring contracts |
| **§10** Jacobian discipline (`constraints::<kind>::wrap`; users NEVER hand-write Jacobians) | `jacobian.md` | constraint / Jacobian discipline |
| **§11** target-distribution geometry boundary (the correctness gate; validator cannot catch geometry violations) · **§12** memory safety | `geometry.md` | geometry legality gate |
| **§13** block-family design conventions (NUTS / GMRF / Gibbs / BART / Joint-NUTS / VI families; the **metric (diagonal/dense) + warmup (single-pilot/3-phase) decision**) | `families.md` | per-family conventions |
| **§14** how to design a new block (7-step lifecycle) · **§15** BART case study · **§16** universal pre-merge checklist · **§17** anti-patterns | `lifecycle.md` | new-block lifecycle + checklist |
| **§18** Variational Inference architectural extension · **§19** design rationale / history | `vi.md` | VI blocks + rationale |

**Citation scheme (stable):** keep writing **"system_design §N"** — it resolves via this
table to one small module. Existing citations across the skill corpus (codegen, validator,
block_catalogue, …) do NOT need rewriting; they look up §N here and load that module.

**For block_design:** the `block_design_skills/` flow indexes these modules directly (load
only the module a Stage needs — e.g. geometry classification → `geometry.md` §11; family
conventions → `families.md` §13; lifecycle → `lifecycle.md` §14).
