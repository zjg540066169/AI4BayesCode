## nuts_block

```cpp
nuts_block_config cfg;
cfg.name        = "<shared_data key>";
cfg.initial_unc = arma::vec{...};        // unconstrained starting point
cfg.constrain   = constraints::positive::constrain;   // omit for real
cfg.unconstrain = constraints::positive::unconstrain;  // omit for real
cfg.log_density_grad = /* lambda */;
// For simplex/ordered/cholesky_corr:
cfg.initial_step_size   = 0.05;
cfg.n_warmup_first_call = 400;
```

The mcmclib backend fork (2026-07-25) auto-detects the metric kind
(IDENTITY / DIAGONAL / DENSE) from the installed `precond_mat`. The
default `nuts_block` never installs a `precond_mat`, so it always
runs the O(n) IDENTITY dispatch (no dense matvec per leap). Users
who install a matrix via `set_precond_matrix()` get DIAGONAL (if the
matrix is diagonal-only) or DENSE dispatch automatically. The same
fork [Fix #3] also removes the per-leaf ColVec_t allocation load in
`nuts_build_tree` via a per-depth scratch pool owned by `nuts_impl`
(bit-identical draws; see nuts.ipp comment). See
`joint_nuts_block.md` for the joint-block details and
`tests/test_mcmclib_metric_dispatch.cpp` for the correctness suite.

### Configuration discipline

The `nuts_block_config` exposes several knobs that affect MCMC correctness;
not all are free for code-generation to set. The table below is the
authoritative codegen-time guide.

| Field | Code-gen may set? | Allowed values | Notes |
|---|---|---|---|
| `name` | YES | any unique string within the composite | shared_data key the block writes to |
| `initial_unc` | YES | `arma::vec` of correct dim | unconstrained starting point; from data-driven init (OLS, MOM) when possible |
| `constrain` / `unconstrain` | YES | one of the `constraints::<kind>::*` pairs | must match the parameter's natural-space constraint |
| `log_density_grad` | YES | `[](theta_unc, ctx, grad){ return constraints::<kind>::wrap(...); }` | natural-scale lp + grad inside the wrap; BLAS per `codegen_cpp.md` Sec.6.1 |
| `n_warmup_first_call` | YES | 200 (default) - 3000 | longer first-call warmup is the right escape valve when initial adaptation needs more runway; bumping this is **always** preferred over per-step adaptation |
| `initial_step_size` | YES | 0.01 - 0.1 | only set explicitly for simplex / ordered / cholesky_corr where the default 1.0 is too aggressive |
| `n_draws_per_step` | NO | leave at default 1 | larger values thin-but-recompute; not needed for outer-Gibbs composition |
| **`n_warmup_per_step`** | **NEVER** | **leave at default 0** | see "n_warmup_per_step is mandatory 0" below |
| `target_accept_rate` | YES (rare) | 0.5 - 0.99 | AI4BayesCode default 0.8 (Stan/PyMC/NumPyro mainstream; the vendored mcmclib default of 0.55 is silently overridden in the nuts_block ctor). Raise to 0.9 - 0.99 for stiff funnel / hierarchical models where divergences persist. Leave at 0.8 unless a specific model demands otherwise. Also settable per readapt() call via the `readapt_NUTS(..., target_accept)` 4th arg. |

### `n_warmup_per_step` is mandatory 0

**Code-gen agents must NOT set `cfg.n_warmup_per_step` to any value other
than the default 0.** Do not write `cfg.n_warmup_per_step = 5` or any
other non-zero value, regardless of any "5-8% variance bias acceptable"
language an older comment in `nuts_block.hpp` might suggest.

**Why.** Setting `n_warmup_per_step > 0` re-enables a chain-state
corruption mechanism that the 2026-04-12 mcmclib bugfix removed.
Per-step adaptation re-runs dual averaging against the same draws being
kept, violating detailed balance and progressively shrinking the step
size into a neighborhood the chain cannot leave. **The runtime symptom
is the "stuck-fast" pattern:** sim1 reports `rhat_max ~= 2.2`,
`ess_bulk_AI = NA / single digits`, and AI walltime SHORTER than Stan's
(stuck chains do almost no tree exploration). L3 single-dataset checks
PASS because the agent's calibration init lands in a friendly region;
sim1 cross-dataset checks FAIL catastrophically.

**Common temptation: "the sigma block keeps rejecting at default 0".**
This is a real symptom but the actual problem is a non-centered
hierarchical funnel between `(sigma_*, raw_effect_*)` -- the scale and
its raw-effect partner are in separate blocks but their conditionals are
multiplicatively coupled. The correct fix is methodological:
1. **`joint_nuts_block`** over `(sigma_*, z_*)` per
   `codegen_cpp.md Sec.4a` row "scale + raw effect, non-centered". The
   joint block samples the multiplicative coupling exactly.
2. **Bump `n_warmup_first_call`** to 1500-3000 so the first-call
   adaptation lands in the typical set.
3. **Better init values** via OLS / method of moments in the wrapper
   constructor.

Use one or more of (1)-(3). Do NOT escape to `n_warmup_per_step > 0` --
that produces silently wrong posteriors that pass L3 while failing sim1.

The same reasoning applies to other blocks where the AI might be tempted
to "patch up rejection by re-adapting": same answer applies (joint
block, longer first-call warmup, better init).
