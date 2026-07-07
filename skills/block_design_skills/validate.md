# module: validate.md -- role: block_design VALIDATE phase (library test + by-mechanism validator checks + staged compile/run; thresholds elicited)
# loaded lazily when the VALIDATE phase is entered (after Stages 0-4 are signed off); do NOT preload from other phases.

> **What this phase is for.** The block code now compiles in your head and the bundle
> is staged. The recurring villain is the **silent correctness bug**: a block that
> compiles, runs, and returns the WRONG posterior. This phase builds the evidence that
> it does not -- a library test with ground-truth regimes -- and runs the by-mechanism
> validator checks that catch the silent failures static review can see. The library-test
> compile+run runs **automatically** (it is the mandatory correctness verification, against the
> staging copy -- it moves nothing); the bundle MOVE to `blocks_local/` happens **automatically once
> all checks pass** (then report the final path -- no "go" gate), and the heavy Sec.4 cross-chain audit
> stays opt-in (TOP RULE: staging is the delivery guard -- a FAILED block is never moved). Decision-relevant thresholds are **elicited**; mechanical test-numerics use
> **recorded, overridable defaults** -- not a per-number picker (SECOND RULE; see Sec.5).

---

## 0. The validation backbone -- the shared three layers, plus the primitive-specific extra

VALIDATE runs the SAME three-layer framework as codegen (`skills/validator.md`:
**L1 Syntactic -> L2 Semantic -> L3 Runtime**), in the same execution order -- cheapest first,
**semantic before runtime** (a silent bug like a wrong Jacobian must be caught by static review
BEFORE you spend minutes on ground-truth runs a wrong block can still "pass" on the easy
regimes). On that shared backbone a NEW primitive needs **extra** validation codegen does not:
codegen COMPOSES already-audited blocks, so its L3 is a light smoke + R-hat + posterior check;
a brand-new block has **no prior correctness evidence**, so block_design adds a **ground-truth
library test** at L3 and runs the **FULL** by-mechanism sweep at L2.

| Layer | Shared backbone (`validator.md`) | What block_design runs (this file) |
|---|---|---|
| **L1 Syntactic** | does it compile | Sec.3 -- compile the staged block + test (exact recipe) |
| **L2 Semantic** | code-level checks for silent bugs | Sec.2 -- the by-mechanism `validator.md` checks #1-#25 the block triggers, **including the mandatory Check #12 AD-twin** for any hand-written gradient (the FULL sweep, not codegen's subset) |
| **L3 Runtime** | codegen: R1 smoke * R2 2-chain R-hat * R3 BPV/LOO | Sec.1 -- the **ground-truth library-test ladder T0-T4** (run via Sec.3 on "go"); SUBSUMES codegen's L3 and ADDS the primitive extra: **T0** sanity * **T1a** parity / **T1b** FD-gradient * **T2** recovery-from-known-truth * **T3** cross-chain R-hat **< 1.01** (stricter than codegen's 1.05) * **T4** stress/funnel |

The block-design EXTRA = everything codegen skips because its inputs are pre-audited: the
ground-truth regimes (T0/T1a/T1b/T2/T4) and the full #1-#25 sweep. That extra is the point -- a
primitive must PROVE correctness against ground truth, not just run clean.

**Execution order: L1 compile -> L2 semantic -> L3 runtime ladder**, run AUTOMATICALLY (no "go"
gate; the bundle MOVE is also automatic on all-pass -- staging just guards against delivering a failed block). The sections below are authored
test-design-first (Sec.1 ladder, Sec.2 checks, Sec.3 compile+run); RUN them L1 -> L2 -> L3.

**Cheapest correctness check FIRST -- the AD-twin before the runtime ladder.** Within L2, the
**AD-twin (Check #12)** -- the autodiff-vs-hand-written gradient comparison (compile+run the tiny
`verify_<Block>.cpp`, 5-20 random-theta compare, delete on PASS) -- is the cheapest meaningful
correctness check. **Run it right after compile and BEFORE the L3 T0-T4 ladder**, NOT bundled
with or after it. A wrong gradient caught by the twin (a handful of gradient evals) saves running
the expensive T2/T3 chains on broken code -- and a chain on a wrong gradient can even *look* like
it converges. (T1b -- the in-test FD gradient check -- still runs inside the L3 ladder; the
gradient ends up verified two independent ways, AD-twin first then FD.)

---

## 1. (L3 Runtime) The library test `test_<Block>.cpp` -- regimes

Stage a self-contained, zero-external-dependency test. For the **R-hat helper**: the repo ships
`split_rhat` (rank-normalized **split**-R-hat) in `tests/test_simplex_dirichlet.cpp` -- for T3 keep
the rank-normalization but use the **cross-chain (NON-split) form**: rank-normalize the pooled
draws, then take the classical BETWEEN-chain R-hat over the two FULL different-seed chains (drop the
per-chain halving). compute sample mean/cov directly with `arma::mean` / `arma::cov`
(there is no `sample_cov` / `sample_mean` helper to copy -- do not cite one). The test links
ONLY `block_sampler.hpp` + `<Block>.hpp`, prints per-regime PASS/FAIL with the tolerance
inline, and `main ` returns non-zero if any regime fails. Build the regime ladder T0->T4;
which of T1's two forms applies is **decided by the block's mechanism** (Stage 2):

| Regime | Proves | Applies to |
|---|---|---|
| **T0 -- sanity** | a trivial closed-form case (e.g. diagonal `Q` => `Var(x_i)=1/kappa`; a flat/known-conjugate target) recovers the analytic mean+variance within MC SE | every block |
| **T1a -- closed-form / dense-inverse parity** | empirical sample cov/mean from many `step ` draws matches the **dense analytic** posterior (e.g. `inv(Q)`, `Q^-^1b`, projected `P(Q+epsilonI)^-^1P^T`) within MC SE | **Gibbs / ESS / direct-draw** blocks (a conjugate or Gaussian-conditional draw HAS a closed form to match) |
| **T1b -- FD-gradient check** | for **any hand-written gradient**, a finite-difference `(logp(theta+h*e_j)-logp(theta-h*e_j))/(2h)` matches the analytic `grad[j]` at several random theta, to the Sec.5 default tolerance | **any block with a hand-written gradient** (NUTS-gradient blocks). This is the single highest-signal correctness test for a hand-derived gradient -- see Sec.2 Check #12 |
| **T2 -- recovery-from-synthetic-truth** | simulate data from KNOWN parameters, sample, and the posterior mean/CI covers truth (parameter recovery, not just internal consistency) | every block (catches a self-consistent-but-wrong target T1 can miss) |
| **T3 -- cross-chain R-hat** | run **TWO chains with DIFFERENT SEEDS** from **deliberately over-dispersed inits** (e.g. +/-5) on a non-trivial dim; compute the **cross-chain (between-chain) rank-normalized R-hat** -- Vehtari 2021 rank-normalization + classical between-chain R-hat, **NOT split-R-hat** -- across the marginals; it stays **< 1.01** (MANDATORY for a library block; NOT the looser exploratory 1.05; NOT user-selectable). Two over-dispersed INDEPENDENT seeds agreeing is the direct test of the reducibility / multimodality failure mode (a within-chain split cannot detect a mode the chain never visited) | every block (mixing failure / reducibility) |
| **T4 -- stress / funnel-resolution** | the block's known hard regime: large dim, near-singular / stiff `Q`, an exact constraint held to machine precision (`max|sum(x)|<1e-10`), or -- for a funnel-resolving block -- that the non-centered form mixes where centered freezes. **For the funnel case, TUNE the contrast to be DECISIVE**: pick a deep-enough funnel (many groups J, small tau, few obs/group) so the CENTERED R-hat is unambiguously broken (e.g. > 1.3, not a borderline 1.05) while NCR stays < 1.01 -- a too-easy funnel passes the assertion while demonstrating nothing (the same "make the baseline visibly fail" principle as the EXAMPLE phase). | every block; the regime is **block-specific** (elicit it from the Stage-0 spec) |

Notes that keep T1 honest:
- **T1a needs a real closed form -- and each engine has its own honest surrogate.** Do NOT
 fake a dense inverse; use the engine-correct parity and SAY which in the test header:
   - *Gibbs / direct-draw* (Gaussian conditional): empirical cov/mean vs `inv(Q)` / `Q^-^1b`.
   - *NUTS-gradient* on a Gaussian target: empirical cov/mean vs the closed-form `Lambda^-^1`.
   - *ESS / any intractable-posterior block* (e.g. GMRF prior + non-Gaussian likelihood, or a
     z-marginalized mixture): there is no closed-form posterior, so **T1a = PRIOR recovery
     under a FLATTENED likelihood** -- make the likelihood contribute NOTHING, then the
     stationary law IS the prior; check sample cov/mean vs the prior. **The cleanest flattening
     is ZERO observations (`N=0` -> loglik = 0 identically), NOT a huge variance** -- a huge
     variance can retain a parameter-dependent tilt (a mixture's log-sum-exp keeps a
     weight-tilt even as sigma^2->inf, which will masquerade as a likelihood artifact and hide a real
     bias). Then T2 = recovery WITH the real likelihood. (Core `gmrf_whitened_ess_block` does
     prior recovery -- "pure-prior recovery 0.998".)
- **T1b is mechanism-gated, not optional.** If Stage 2 produced ANY hand-written gradient,
 T1b is REQUIRED even if T1a also applies (a block can have both an analytic conditional
 and a hand gradient).
- **Vendored numerical routine -> add an independent correctness regime, and T1b/T0 doubles as
 its regression.** If the block uses a borrowed routine (INTAKE Step 5 -- e.g. a vendored
 `digamma`), first verify it standalone against KNOWN values (and/or an `lgamma`-FD), THEN note
 that the T0 log-density-vs-reference and T1b FD-gradient checks ALSO regression-test the vendor
 (a wrong `digamma` shows up as a T1b mismatch). Cite the standalone accuracy number in the test
 header so a future vendor swap can't silently drift.
- **Vendored KERNEL -> ALSO a STATEFUL-compatibility check (additional to the correctness regime
 above).** A borrowed kernel is external code NOT written for AI4BayesCode's stateful-block contract,
 so it often carries hidden state that silently breaks it -- a `static`/global mutable, a mutable
 singleton, its OWN RNG, or a cache built from the data. Correctness alone does NOT catch a hidden-state
 bug, so verify ALL of:
   - **No hidden global / static mutable state.** `grep` the vendored headers you actually `#include`
     for non-const `static` locals, global / namespace-scope mutables, singletons, thread-locals. Any
     such state is SHARED across block instances -- block A's `step` corrupts block B. It MUST be
     instance-owned (a member), never global.
   - **Determinism under the block's RNG.** The kernel must consume the block's passed-in `rng` (or be
     purely deterministic). If it uses its own generator / `rand()` / a global RNG it breaks
     reproducibility -- seed it from the block's `rng`. **TEST:** two runs, SAME seed -> identical (or
     stat-identical) output; a mismatch = hidden nondeterministic state.
   - **Stale cache on `set_context`.** If the kernel caches anything derived from the context / data (a
     k-d tree on locations, a factorization, a precomputed table) and `set_context` swaps the data, the
     cache MUST rebuild. **TEST:** `set_context(A)` -> step -> `set_context(B!=A)` -> step; the output
     reflects B, not a stale A-cache.
   - **Instance isolation / fresh-state.** Two independent block instances stepped interleaved do NOT
     interfere; a freshly constructed block carries no leftover kernel state.
 Add these as a runnable regime (two-instance independence + same-seed determinism + set_context
 refresh) in `test_<Block>.cpp`, plus the static grep. (The stateful MODIFICATIONS that make this pass --
 isolating global / `static` state into the instance, threading the block's `rng`, cache rebuild on
 `set_context` -- are authored per `vendor.md` Sec.3, during DESIGN.)
- **T3 for a deterministic VI block = fixed-point + init-independence, NOT a chain R-hat.** A
 closed-form CAVI / deterministic `vi_block` has no Markov chains, so a cross-chain R-hat is
 structurally meaningless. The mechanism-correct T3 substitute is: (a) the fixed-point holds (one more
 sweep moves the parameters < tol -- a no-op), AND (b) init-independence (two over-dispersed
 starts converge to the SAME optimum within tol). State the substitution in the test header
 so it does not read as a skipped T3. (Stochastic-gradient VI instead reports the ELBO-trace
 convergence + PSIS-k-hat, per `vi.md` Sec.18.8.)
- **Tolerances are MC-SE-budgeted, not vibes.** Comment each tolerance with its SE basis
 (`SE(Var)~=sqrt(2*Var^2/M)`, `SE(cov-entry)~=sqrt2*Var/sqrtM`), so a future regression can't hide under
 a lax bound. The *numeric* bounds are set per Sec.5 (mechanical ones defaulted, the R-hat bar elicited).
- **Those SE formulas are IID -- for an MCMC/NUTS chain they are WRONG (too small).** A z-test
 on a posterior mean/variance from autocorrelated draws using `sqrt(.../M)` silently INFLATES the
 z-score (a CORRECT block then "fails" T1a at z~=6 purely from autocorrelation). For any
 chain-based regime, budget the SE with the **effective sample size** (`SE ~= sqrt(Var/ESS)`) or a
 **batch-means** estimator -- NOT the iid `M`. (Closed-form-draw / Gibbs / ESS regimes that emit
 independent draws may use the iid formula.) State which SE you used in the comment.
- **An entrywise-MAX statistic needs a MULTIPLICITY-aware bar (a real integrity trap).**
 `max_ij |Cov_emp - Cov_true|` / the max z-score over the ~n^2/2 cov entries is a MAX over K
 entries; under CORRECT sampling its tail routinely exceeds the per-entry ~5sigma -- so a
 single-entry `z<5` bar gives INTERMITTENT T1a FAILs on a CORRECT block. Honest fixes:
 (a) a FAMILY-WISE bar (e.g. `z < 7` for K~n^2/2; false-reject ~=1e-10), stating the
 per-entry-vs-max distinction in the comment; OR (b) a POOLED statistic (mean-z, or the
 Frobenius `||Cov_emp - Cov_true||_F / SE`) instead of the raw max. **Never "fix" a borderline
 max-z by quietly bumping the tolerance** -- that is the exact integrity trap the ladder warns
 about. And note: raising the draw count M shrinks each entry's SE but does NOT shrink the
 max-over-K multiplicity -- for a max-z bar, **M is the WRONG lever** (it can make max-z worse).

---

## 2. (L2 Semantic) Applicable validator checks -- BY MECHANISM

Run ONLY the checks the block's mechanism actually triggers. Each points to `validator.md`
(do not restate the check body here -- load it from there). A check that does not apply is
**silent** for this family; call that out so the user does not read silence as a pass.

| Block mechanism (from Stages 1-2) | Triggered check -> `validator.md` |
|---|---|
| **hand-written gradient** (NUTS-gradient block) | **Check #12** (gradient verification via autodiff -- a semantic check that runs a throwaway compile at generation) -- recipe in `jacobian.md` Sec.10.1 (paste the natural-scale functions, add `autodiff::var` twins, compare 5-20 random-theta < 1e-8, delete on PASS). For a LIBRARY block, for ANY hand-written gradient, **BOTH are MANDATORY and NOT asked**: **T1b (FD)** as the runnable in-test check AND the **AD-twin (Check #12)** as the exact verify. The AD-twin is mandatory **regardless of how strong the FD looks** -- correctness is the highest goal, the twin is a cheap throwaway compile, and a reused primitive's gradient is verified two INDEPENDENT ways (analytic-vs-FD AND analytic-vs-autodiff). A wrong gradient is the silent-bug class a library primitive must NEVER ship -- so just run the throwaway twin (deleted on PASS), never offer it as optional. |
| **natural-scale-only log-density, no hand Jacobian** | **Check #5** (Jacobian handling): the user log-density stays natural-scale; the block adds `log\|Jacobian\|` via `constraints::<kind>::wrap` -- a hand-written `+ log(scale)` is a FAIL even when "it looks right". |
| **BLAS-vectorized gradient** | **Check #19** (vectorized gradient / BLAS compliance, Sec.6.1): a hand-unrolled loop where a matrix op belongs is a silent perf+correctness risk. |
| **dense mass metric** used by the block | **Check #18** (dense metric justification + pilot scaling): dense is **diagnostic-driven escalation** -- start diagonal, MEASURE, escalate on evidence; it is NOT a dimension threshold. Requires an inline `// JUSTIFICATION (Check #18): ...` with specific evidence. |
| **wraps `nuts_block` / exposes `n_warmup_per_step`** | **Check #20** (`n_warmup_per_step` must stay 0): any nonzero re-enables the 2026-04-12 chain-corruption mechanism (R-hat~=2.2, ESS~=NA, walltime deceptively short). |
| **hierarchical funnel** -- a positive scale (tau,sigma) governs raw-effect spread | **Check #24(a)** (joint-NUTS funnel NCR pre-flight): a **centered** scale-governed effect is a **hard FAIL** -- NCR (non-centered: standardized `eta_j` + deterministic `theta_j := loc + scale*eta_j`) is **mandatory**. There is **no centered+dense escape hatch.** Also #24(b) constraint-kind consistency, #24(c) lambda completeness. |
| **constructs / behaves like `joint_nuts_block`** | **Check #24** full (a/b/c) as above. |
| **trans-dimensional / RJ block** (composes `rjmcmc_block`) | **Check #14** (trans-dim proposal Jacobian is library-owned: identity-coordinate `|det J|=1`, library 1-D transforms, or runtime-AD on a templated forward + analytic inverse -- `jacobian.md` Sec.10.2; `rjmcmc_block_config` has NO `jacobian_fn`). Verify the proposal is Jacobian-free by construction; for a conjugate-marginalized model with a constant `propose_logq`, check `birth_logR == -death_logR` on a fixed pair. |
| **vendors an external KERNEL / library** (INTAKE Step 5; manifest `KernelTier: vendored`) | **Vendored-kernel STATEFUL-compatibility check** (Sec.1, additional to the vendored-correctness regime): external code is NOT written for the stateful-block contract -- verify NO hidden global / `static` mutable state (cross-instance leakage), determinism under the block's `rng` (same seed -> same output), correct cache rebuild on `set_context`, and instance-isolated state. Runnable two-instance + same-seed + refresh regime in `test_<Block>.cpp` + a static grep of the used vendored headers. |

**Family-silence notes (state explicitly, do not let silence read as a pass):**
- **#18 is silent for an ESS / Gibbs / direct-draw block** -- there is no mass metric to
 justify, so #18 simply does not fire. That is correct, not a clean bill.
- **#12 / T1b is silent for a block with NO hand-written gradient** (pure Gibbs/ESS) -- there
 is no gradient to verify; #12 does not fire.
- **#24 is silent unless the block is funnel-shaped or `joint_nuts`-like.**
- The **geometry-legality gate** (`geometry.md` Sec.11) is NOT in this table because it is a
 Stage-1 correctness gate the validator CANNOT catch at this phase -- if Stage 1 flagged it,
 it was resolved before any code was staged, not deferred to here.

**Block-local extra semantic checks (`BL#`) -- first-class, not just prose.** A silent-failure
mode unique to THIS block becomes a numbered block-local check with the SAME anatomy as a core
check: **Trigger** (what makes it apply) * **Why** (the silent bug it catches, and why it passes
R-hat / ESS / LOO) * **What to look for** (a `// WRONG` vs `// RIGHT` pair, a `grep -nE` static
pattern, or a runtime assertion) * **Fix**. ID them `BL1`, `BL2`, ... -- block-scoped, so they
never collide with core `#1`-`#25`. ELICIT them from the Stage 1-3 design (the callback
contracts, the invariants the algorithm must hold, the kernel's scope assumptions) and APPLY
each one:
- **runnable** items -> an assertion / regime in `test_<Block>.cpp` (run on "go" with the ladder);
- **static** items -> a code-review grep / `WRONG`-vs-`RIGHT`, recorded for review.

They live in a **SEPARATE block-local validation skill** `skills/<Block>_validation.md`, pointed
to by the manifest **`ValidationSkill:`** field (kept OUT of the main `skills/<Block>.md` so block
selection / use don't pay for them). During VALIDATE, **load that file ONLY if the manifest has a
`ValidationSkill:`** -- if the field is absent the block has no `BL#` and only the core `#1`-`#25`
apply (lazy-load = token saving). The `BL#` IDs are listed in the manifest `ChecksApplicable`
alongside the core `#N` the block faces. They travel with the bundle and are NEVER edited into core
`validator.md`. If a `BL#` turns out to be GENERAL (applies to a CLASS of
blocks), promoting it to a core `#N` is the maintainer's call (a `validator.md` sign-off) -- NOT
an auto-append by this flow.

---

## 3. (L1 Syntactic compile + L3 run) COMPILE + library-test RUN -- automatic (no "go" gate)

This is **cheap, high-signal, and the MANDATORY correctness verification**, so **run it
automatically -- do NOT ask a "go" question first.** It runs against the STAGING copy (nothing
moves to the bundle), so there is nothing to gate. (By the time you run this T0-T4 ladder, the L2
**AD-twin (Check #12)** has ALREADY verified any hand-written gradient -- Sec.0; do NOT defer the
twin to after the ladder.) Still surface the exact command + expected
runtime AS YOU START (a heads-up, NOT a gate -- especially for a minutes-long NUTS run, so the
user is not surprised and can interrupt if they want).

What to surface as you start (a heads-up, NOT a gate):
- **the exact compile command.** Verified-working recipe (run from the repo root):
 ```
 c++ -std=c++17 -O2 -Wno-unused-parameter \
 -I AI4BayesCode/include -I AI4BayesCode/include/mcmclib \
 -I AI4BayesCode/include/mcmclib/BaseMatrixOps/include -I AI4BayesCode/include/eigen \
 -I /Library/Frameworks/R.framework/Versions/Current/Resources/library/RcppArmadillo/include \
 -I ./blocks_local/<Block> \
 -DMCMC_ENABLE_ARMA_WRAPPERS -DARMA_DONT_USE_WRAPPER \
 -o /tmp/test_<Block> ./blocks_local/<Block>/test_<Block>.cpp -framework Accelerate
 ```
 Two non-obvious points an author WILL hit: (1) `-I ./blocks_local/<Block>` is
 REQUIRED so `#include "<Block>.hpp"` resolves; (2) the arma backend needs
 `-DMCMC_ENABLE_ARMA_WRAPPERS -DARMA_DONT_USE_WRAPPER` + RcppArmadillo's include dir on `-I`
 + `-framework Accelerate` (macOS; on Linux use `-lblas -llapack` instead). The block header
 `#include "AI4BayesCode/block_sampler.hpp"`; the test `#include "<Block>.hpp"`.
 Two portability cautions: (a) the RcppArmadillo include path above is the macOS
 `/Library/Frameworks/R.framework/...` default -- on Linux/other, resolve it with
 `Rscript -e 'cat(system.file("include", package="RcppArmadillo"))'`; (b) **pass the flags as
 LITERAL tokens, not bundled into one shell `$VAR`** -- zsh/bash do NOT word-split an unquoted
 variable the way you expect, so `-framework Accelerate` inside a `$FLAGS` string collapses into
 one invalid argument. Type the flags inline (or use a bash array).
- which regimes (T0-T4) will execute and the tolerances they will use (R-hat < 1.01; mechanical defaults),
- expected runtime -- **engine-dependent, surface it honestly.** A Gibbs / direct-draw / ESS
 ladder is seconds (small MC loops). A **NUTS-gradient** parity regime is MINUTES, not seconds
 (NUTS runs a full tree per kept draw -- a T1a at M~100k over n~12 can take ~8 min); say so as
 you start the run, and do not promise "seconds" for a NUTS-engine block.

Compile, run, and report **every** regime's PASS/FAIL verbatim with its tolerance. Do
NOT cherry-pick converging regimes; a partial run is not done. If a regime FAILS, report it
as a named failure with the reason -- never hide it to inflate a pass count, never tighten or
loosen a tolerance post-hoc to flip the verdict.

**Staging discipline (the delivery GUARD, not a gate):** compile+run happens AUTOMATICALLY against
the STAGING copy -- it moves nothing. The bundle MOVE to `./blocks_local/<Block>/` is also
AUTOMATIC, once ALL checks pass (no "go" gate) -- then REPORT the final path. Staging exists so a
FAILED block is never delivered: if any check fails, the bundle STAYS in staging, you report what
failed and STOP.

---

## 4. Long cross-chain audit -- NOT a correctness gate; not asked during local creation

A full cross-chain audit (multi-agent, sim1-style cross-dataset / multi-seed sweep at the
standing **20K+20K** budget, R-hat/ESS across many synthetic datasets) is **expensive** (tens of
minutes-hours). It is **NOT a correctness gate** -- correctness is already fully established by
the MANDATORY ladder (T0-T4, cross-chain R-hat < 1.01) plus the by-mechanism checks (Sec.2,
including the mandatory Check #12 AD-twin for any hand-written gradient). The sweep is
extra-robustness BENCHMARK evidence for the FUTURE submission path (its sandboxed re-run),
not for local creation.

So do **NOT** fire a gating question for it during local creation -- that is pure friction, and
auto-running an hours-long sweep without consent is the exact "wasted hours" failure the TOP
RULE forbids. Mention it in ONE line as available on explicit request, with runtime stated;
run it ONLY if the user asks. Skipping it does NOT weaken the block's correctness -- the
mandatory gates already guarantee that.

---

## 5. Thresholds -- ask the decision-relevant ones, default the mechanical ones

Two kinds of numeric, handled differently. **Do NOT fire a separate picker for every
number** -- that is friction with no decision value.

**FIXED for a library block (do NOT ask):**

- **Cross-chain R-hat bar (T3) is `< 1.01`** -- TWO chains with DIFFERENT seeds from
 over-dispersed inits; the **cross-chain (between-chain) rank-normalized R-hat** (Vehtari 2021
 rank-normalization + classical between-chain R-hat, **NOT split-R-hat**) across the marginals. Strict
 publication-grade: NOT the looser exploratory `1.05`, NOT user-selectable. (A rank-R-hat failure
 is a real mixing failure -- "multimodal" is NEVER a valid excuse.) If it can't reach `< 1.01`
 at budget, that is a FAIL to FIX (reparam / metric / longer chains, per the auto-retry policy),
 never a bar to loosen.

**ASK only** when a genuinely model-specific value-pair is BOTH scientifically defensible AND
flips the PASS/FAIL verdict (rare) -- otherwise nothing in the standard ladder is asked. (The
real methodology choices -- sampler / marginalization / prior -- are surfaced earlier, in
DESIGN, not here.)

**DEFAULT silently (mechanical test-numerics -- no methodology content, a near-universal
right value):** do NOT pop a question. Apply the vetted default, RECORD it in the staged
`test_<Block>.cpp` header (with its SE basis), and LIST them in the DESIGN-LOCK summary so
the user can override any -- but never interrogate them one at a time:

- **FD-gradient tolerance (T1b)** -- `1e-6` (central difference, `h~=1e-5`); bump to `1e-5`
 only if it proves flaky near a boundary.
- **MC-SE budget for parity (T1a/T2)** -- ~5sigma of slack at the chosen draw count M.
- **exact-constraint tolerance (T4)** -- `max|sum(x)| < 1e-10` (`1e-12` if the constraint is
 algebraically exact).

**Why the split:** the SECOND RULE forbids silently baking a METHODOLOGY constant (sampler /
marginalization / prior) -- those carry scientific judgment and are surfaced in DESIGN; the
convergence standard is fixed by LIBRARY POLICY at rank-R-hat < 1.01 (stated, not silently baked,
not a per-block choice). An FD-gradient tolerance carries NO methodology judgment (`1e-6` just means
"the gradient is correct to numerical precision"), so interrogating it is pure friction.
RECORDING + SHOWING the default (overridable) honors the rule's intent without a per-number
picker.

Echo the final thresholds in the staged test header (with each tolerance's SE basis) so the
bundle records exactly what bar it was validated against.

---

## 6. Phase exit

VALIDATE is done when: the staged `test_<Block>.cpp` exists with the mechanism-correct T0-T4
ladder and its tolerances (cross-chain R-hat < 1.01; mechanical defaults); the by-mechanism checks (Sec.2) have been run and any FAIL
resolved or honestly reported; the library test compiles and ALL regimes pass (run
automatically) -- INCLUDING the mandatory Check #12 AD-twin for any hand-written gradient. The
separate `ValidationSkill` (`skills/<Block>_validation.md`, manifest field) records any
block-specific `BL#` checks -- present only if the block has any. (The heavy cross-chain audit
is NOT required -- it is submission-path extra, run only on explicit request.)

**Then -- before EXAMPLE -- fire the example gate. Do NOT auto-enter EXAMPLE.** An example is
OPTIONAL: some shipped core blocks have no example, and a block + passing library test is already
a complete bundle. So ASK the user (structured gate, env mechanism -- `AskUserQuestion` in Claude
Code, markdown labeled-option fallback elsewhere): `(a) Yes -- write a bundled example (default) /
(b) No -- skip to SKILL / (c) Other`.
- **(a) yes** -> load `example.md` and run EXAMPLE.
- **(b) no** -> **do NOT load `example.md`** -- skip straight to the SKILL phase (`skill.md`).
Do NOT announce "Entering the EXAMPLE phase / loading its module" until this gate is ANSWERED yes.

The bundle MOVES to `blocks_local/<Block>/` AUTOMATICALLY once all checks pass (no "go" gate); then report the final path. A failed block is never moved.
