---
name: AI4BayesCode
description: Build Bayesian MCMC samplers with AI4BayesCode — a header-only C++ library of composable, stateful block-wise MCMC kernels (Gibbs / NUTS / slice / ESS / VI / BART / GP / HMM / particle-Gibbs / RJMCMC / ...). Use when the user wants to (a) GENERATE a sampler for a Bayesian model by composing existing blocks, or (b) DESIGN a brand-new sampler block primitive for the library. Triggers: "I want a sampler for <model>", "compose / generate an MCMC sampler", "design a new block", a model written as y ~ ... with priors, or any AI4BayesCode request.
---

# AI4BayesCode

You are launching **AI4BayesCode** — a stateful, block-wise MCMC sampler system. Your job is to
either compose an existing-block sampler for the user's model (the **codegen** flow) or design a
brand-new block primitive for the library (the **block_design** flow).

## Bootstrap — do this FIRST

Read **`start.md`** at this skill's root. It is the canonical entry point / router; follow it
exactly:

- it routes the request into the **codegen** flow (compose existing blocks for a model the user
  describes) or the **block_design** flow (design a new primitive);
- it carries the phase schedule, the lazy-load rule, the structured-question protocol, and the
  per-phase sign-off gates.

Do NOT restate or duplicate `start.md` here — READ it, then follow it.

## Token discipline (lazy-load)

Read `start.md` first. Then load each sub-skill ONLY when you enter that phase —
`skills/codegen.md`, `skills/block_design_skills/00_flow.md`, the `skills/system_design_skills/`
modules, etc. NEVER preload the whole `skills/` tree. If unsure whether you need a module yet, do
not load it.

## The library lives HERE

This skill bundles the AI4BayesCode C++ library, not just the guidance:

- the **built-in block headers** + system dependencies are under **`include/`** at this skill's
  root; the routing index of built-in blocks is **`skills/block_catalogue.md`**; reference
  examples are in **`examples/`**.
- **The AI4BayesCode library root IS this skill's directory** (where this `SKILL.md` / `start.md`
  live). Treat it as the RESOLVED AI4BayesCode path for the whole session: the codegen flow does
  **NOT** ask "Path to the AI4BayesCode folder" — use this skill root automatically, and substitute
  its absolute path for the bare `AI4BayesCode/` prefix in every compile recipe (so the `-I` points
  at `<this-skill>/include`). Only ask for a path if the user explicitly wants a DIFFERENT
  AI4BayesCode checkout.
- **User-GENERATED output** (the new `.cpp` + runner, or a new `blocks_local/<Block>/` bundle)
  goes in the USER's own project (`./generated/...`, the user's chosen folder), **NEVER inside
  this installed skill directory.**

## Install / update

See `AI_AGENT_INSTALL.md` for the agent-driven install runbook (Claude Code / Codex / Cursor).
After install, reload the agent; then typing `/AI4BayesCode` loads this skill (which bootstraps
`start.md`).
