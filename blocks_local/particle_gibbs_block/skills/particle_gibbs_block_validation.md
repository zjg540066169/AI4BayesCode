---
name: particle_gibbs_block_validation
description: Block-specific silent-failure checks (BL1-BL6) for particle_gibbs_block. Loaded only at VALIDATE / re-validation / audit, not on block selection or use.
---

# particle_gibbs_block — block-local validation checks (BL1–BL6)

Silent-failure modes specific to a conditional-SMC / particle-Gibbs path
sampler — each compiles, runs, and produces plausible paths while sampling the
WRONG distribution (or freezing). Generic checks (RNG threading, geometry
legality, Jacobian discipline) are NOT here — they are cited in
`skills/particle_gibbs_block.md`'s closing block. The core check the block
faces is `#17` (the Exception-4 justification comment must be present).

---

## BL1 — reference-trajectory retention  (static; exercised by T1a)

- **Trigger.** Every conditional sweep (`step()` after the reference exists).
- **Why.** The conditional-SMC kernel is invariant ONLY because the reference
  path x' is RETAINED as one particle (slot 0) at every t. If slot 0 is
  propagated/overwritten instead of pinned to x'_t, the kernel silently becomes
  an ordinary (unconditional) particle filter, which does NOT leave
  p(x_{1:T}|y,θ) invariant. The chain still runs and the paths look fine, so
  R̂/ESS/LOO pass — the stationary law is just wrong.
- **What to look for.** In `run_csmc_`, slot 0 is set to the reference at EVERY
  time and EXCLUDED from the free-particle propagation (`first_free = 1` when
  conditional):
  ```cpp
  // RIGHT: if (conditional) X[t].col(0) = state_at_(ref_, t);
  //        for (i = first_free; i < N; ++i) X[t].col(i) = transition_sample(...);
  // WRONG: propagating slot 0 through transition_sample (reference not retained)
  ```
- **Fix.** Pin slot 0 to the reference state at every t; free-particle loop
  starts at `first_free = 1`.
- **Exercised by.** T1a (exact RTS-smoother parity) — a broken reference fails
  the parity z-test (the empirical marginals drift from the exact smoother).

## BL2 — PGAS ancestor-weight correctness  (static; exercised by T1a + T4)

- **Trigger.** `ancestor_sampling = true`.
- **Why.** The reference's ancestor must be drawn with probability
  ∝ W_{t-1}^j · f_θ(x'_t | x_{t-1}^j) — the FORWARD transition density from
  particle j's time-(t-1) state to the reference's time-t state, times the
  normalized weight. A swapped argument order, a missing W_{t-1}^j factor, or an
  unnormalized weight subtly distorts the ancestor distribution; PGAS still runs
  and may mix acceptably on easy/short series, so the bug is silent.
- **What to look for.**
  ```cpp
  // RIGHT: log_aw[j] = log(W[j]) + transition_logpdf(t, ref_t, X[t-1].col(j));
  //        (ref_t = x_t is the FIRST density arg; particle j = x_{t-1} is SECOND)
  // WRONG: transition_logpdf(t, X[t-1].col(j), ref_t)  // swapped direction
  // WRONG: omitting the + log(W[j]) term
  ```
- **Fix.** ancestor log-weight = log W_{t-1}[j] + log f_θ(ref_t | particle_j);
  normalize in log-space; sample.
- **Exercised by.** T1a (parity with PGAS on) and T4 (PGAS mixing vs vanilla).

## BL3 — log-space weight normalization  (static)

- **Trigger.** Every weight normalization and the PGAS ancestor weights.
- **Why.** Observation log-weights span a wide range (a badly-placed particle
  can have log g ≈ −1e3). Exponentiating raw log-weights without subtracting the
  max underflows to 0 → 0/0 → NaN weights → NaN paths. The NaN can propagate or
  be masked; a naive normalization is a latent numerical bug.
- **What to look for.** `normalize_log_weights` uses log-sum-exp (subtracts the
  max before exponentiating), coerces non-finite entries to −inf, and falls back
  to uniform when ALL entries are −inf.
  ```cpp
  // RIGHT: lse = log_sum_exp(logw); W[i] = exp(logw[i] - lse);  (uniform if all -inf)
  // WRONG: W[i] = exp(logw[i]) / sum_j exp(logw[j]);            (underflow -> NaN)
  ```
- **Fix.** Use log-sum-exp normalization with max subtraction; NaN→−inf;
  uniform fallback. (Any regime would surface NaN paths if violated.)

## BL4 — N ≥ 2 guard  (runnable; test_BL)

- **Trigger.** Construction.
- **Why.** Conditional SMC needs the reference PLUS at least one free particle.
  With N=1 the only particle IS the reference, the trace-back always returns it,
  and the chain is FROZEN (never moves) — a silent non-mixing failure.
- **What to look for.** Constructor throws `std::invalid_argument` when
  `n_particles < 2`.
- **Fix.** Validate `n_particles >= 2` at construction.
- **Exercised by.** `test_BL` BL4 (N=1 throws).

## BL5 — PGAS requires transition_logpdf  (runnable; test_BL)

- **Trigger.** `ancestor_sampling = true` with `transition_logpdf` unset.
- **Why.** The PGAS ancestor weight needs f_θ(x'_t | x_{t-1}^j). If
  `transition_logpdf` is null while ancestor sampling is on, the kernel either
  crashes mid-sweep (`std::bad_function_call`) or, if mis-guarded, silently
  degrades to vanilla PG without the user knowing — a hidden loss of the mixing
  fix.
- **What to look for.** Constructor throws when
  `ancestor_sampling && !transition_logpdf`.
- **Fix.** Validate at construction.
- **Exercised by.** `test_BL` BL5 (missing logpdf throws).

## BL6 — first-sweep unconditional seeding  (runnable; test_BL)

- **Trigger.** First `step()` with no reference (no `initial_path`, no prior
  `set_current`).
- **Why.** The first sweep has no reference trajectory. Running conditional SMC
  against the default (zero) reference would condition on a meaningless path,
  biasing the early chain (a stuck/slow start). The first sweep must run an
  UNCONDITIONAL particle filter to seed the reference.
- **What to look for.** `step()` calls `run_csmc_(rng, conditional = has_reference_)`;
  `has_reference_` is false until the first step (or until `set_current` /
  `initial_path`).
  ```cpp
  // RIGHT: run_csmc_(rng, /*conditional=*/has_reference_);  // false on first sweep
  // WRONG: run_csmc_(rng, /*conditional=*/true);            // conditions on zeros
  ```
- **Fix.** Track `has_reference_`; first sweep runs unconditional.
- **Exercised by.** `test_BL` BL6 (first-sweep path is nonzero, not the zero init).
