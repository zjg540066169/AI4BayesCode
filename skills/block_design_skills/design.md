<!-- block_design MODULE -- design.md -- the DESIGN phase.
 ROLE: drive the actual block design BY READING the system_design modules (this is where
 block_design "reads system_design"). Loaded ONLY when the design phase is entered (lazy);
 do NOT preload sibling block_design modules. Cite system_design modules / validator.md /
 constraints.md / codegen_priors.md -- do NOT restate their content. Output: the staged
 `<Block>.hpp`. Sign-off gate: methodology locked BEFORE any code is written. -->

# block_design -- DESIGN phase (Stages 1-5)

You arrive here with Stage 0 done: a stated math spec, a unique `Block` name, a crisp
`SelectWhen`, and the advisory "do you even need a new block?" note shown (enabling, never
gatekeeping). This phase produces the **methodology lock** and then the staged `<Block>.hpp`.

**The recurring villain.** Every stage below exists to stop the SILENT correctness bug -- code
that compiles, runs, and gives clean R-hat / ESS / LOO, but samples a DIFFERENT posterior than
the user wrote down. The validator (Layer 2/3) CANNOT catch most of these; they are prevented at
DESIGN TIME. That is the whole reason this phase reads the system_design modules.

**Discipline (every stage).** Each numbered decision is a structured labeled-option prompt (in
Claude Code: AskUserQuestion; elsewhere: the markdown labeled-option fallback from `start.md` Sec.0).
Standing choice labeled **"(default)"**, never "(Recommended)". Always an **"Other"** escape.
Decision-relevant numerical thresholds are **ELICITED** ("(default)"); mechanical test-numerics use recorded, overridable defaults (`validate.md` Sec.5). NOTHING is written
to disk until the Stage-1-3 methodology is signed off (SECOND RULE). Then the
`<Block>.hpp` is STAGED to a staging dir and MOVED to `blocks_local/<Block>/` only on explicit
"go" (TOP RULE).

**Token discipline / how to read system_design.** Each sub-step below loads ONE module from
`skills/system_design_skills/<module>.md` (resolved via "system_design Sec.N" -> the
`system_design.md` index). Load only the module the current sub-step names; never the monolith,
never all of them.

---

## Stage 1 -- GEOMETRY classification (the correctness GATE)

> LOAD: `skills/system_design_skills/geometry.md` (system_design Sec.11). This is the ONE gate the
> validator structurally cannot enforce (Sec.11.3): a geometry violation converges cleanly to the
> WRONG distribution. Get it right here or nowhere.

**Step 1.1 -- State the target distribution explicitly** (Sec.11.4 rule 1): prior, likelihood, and
the SUPPORT of every variable. Write it into the block-header comment draft now -- every block
header must declare which of Sec.11.1's three supported shapes it samples.

**Step 1.2 -- Classify against the boundary** (Sec.11.1 supported / Sec.11.2 NOT supported):

| Class | Geometry | Path |
|---|---|---|
| Fixed-dim absolutely continuous | smooth log-density on R^d after transforms | Stage 2 -> NUTS / VI / slice |
| Fixed-dim discrete, finite support | {0,1}^p, {1..K}^N, simplex/interval | Stage 2 -> Gibbs / specialized |
| Product / mixture of the above | assembled via `composite_block` | Stage 2 per sub-block |
| **Dimension-changing / trans-dim** (Sec.11.2a) | support depends on a discrete indicator (spike-and-slab Dirac, model-size, change-point, basis-birth, BNP-K) | **NOT a new ad-hoc block** -- STOP |
| **Discrete w/ strong local dependence** (Sec.11.2b) | HMM z_{1:T}, Ising/CRF/MRF, BN structure | **NOT per-site Gibbs** -- STOP |

**Step 1.3 -- The STOP cases (ask before proceeding).** If Sec.11.2(a) or Sec.11.2(b): a naive
Gibbs-only or NUTS-only block is SILENTLY WRONG (Sec.11.2 spells out both failure modes -- Gibbs is
not irreducible across the Dirac; gamma-marginalized NUTS hallucinates inclusion probabilities;
per-site Gibbs on coupled discrete sites converges in the limit but is useless). Surface the
shipped remedy as the leading option BEFORE letting the user author a bespoke block:

- Sec.11.2(a) -> `rjmcmc_block` (identity-coordinate, Jacobian-free; ref `SpikeSlabRJMCMC.cpp`); or
 the truncated-SBP stack for BNP-K (`stick_breaking_block` + `normal_gamma_cluster_gibbs_block`
 + `categorical_gibbs_block`); or a continuous relaxation (tight Gaussian spike).
- Sec.11.2(b) -> the matching shipped specialized block (`hmm_block`, `ising_cluster_block`,
 `gmrf_precision_block` / `gmrf_whitened_ess_block`, `order_mcmc_block`).

A genuinely-new block here must legitimately be a STRUCTURED algorithm that exploits the
dependence/dimension structure (forward-backward, cluster move, RJ proposal), not per-site Gibbs
or marginalized NUTS. The variance/scale-prior discipline (Sec.11.6) -- Jeffreys default with k=0
half-Normal(0,1) fallback; IG(epsilon,epsilon) PROHIBITED as a "noninformative" default -- is in force the
moment your block introduces a scale parameter.

> **SIGN-OFF 1 (geometry).** Lock: the stated target, its Sec.11.1/Sec.11.2 class, and (if a STOP
> class) the explicit reason a structured custom block is correct where Gibbs/NUTS is not. No
> further stage proceeds until this is signed off.

---

## Stage 2 -- ALGORITHM spec (engine-kind branch)

> LOAD: `skills/system_design_skills/families.md` (system_design Sec.13) for the family conventions
> and the metric/warmup decision. For a VI block ALSO load `skills/system_design_skills/vi.md`
> (system_design Sec.18). Do not pin to MCMC -- `engine_kind_t` admits `MCMC` and `VI`.

**Step 2.1 -- Pick the engine kind / family** (structured prompt; default depends on the Stage-1
class). Each family has its own setter naming and Tier-B contract (Sec.13):

| Branch | When | Tier-B contract highlights (Sec.13) |
|---|---|---|
| **NUTS-gradient** (default for fixed-dim continuous) | smooth log-density; hand-written natural-scale logp + grad | `set_current(arma::vec)` unconstrained; gradient Check #12 (semantic AD-verify via a throwaway compile at generation, see `jacobian.md` Sec.10.1) |
| **Gibbs** (closed-form conjugate / discrete / **Gaussian-conditional**) | Exception 1/3 applies (`codegen_priors Sec.2b`); discrete, NUTS-wasteful textbook conjugate, OR a Gaussian / GMRF field where each site has an exact 1-D Gaussian full conditional read off the precision row (m_i, v_i closed-form) | `params_fn`/`log_odds_fn`/`alpha_post_fn`; Check #15 parity + Check #16/#17 (highest silent-bug risk) |
| **Slice / ESS** | 1-D non-diff/black-box (`univariate_slice_sampling_block`, `codegen_priors Sec.2b.1`), or a GMRF **prior** under a **NON-Gaussian** likelihood (Murray 2010 ESS). NOTE: if the likelihood is Gaussian-conjugate the site conditionals ARE closed-form -> use the **Gibbs** row above, not ESS | metric-free; no dense-metric machinery |
| **VI** | OPT-IN (`vi.md` Sec.18) -- high-dim with symmetry, or a fast factorized-q summary suffices | derives `vi_block`; `engine_kind ==VI`; writes q-SAMPLE not q-mean to shared_data (Sec.18.4); optimizer = RAABBVI (stochastic ADVI, Sec.18.7) **OR closed-form CAVI** when the mean-field updates are analytic (conjugate-exponential / Gaussian MRF -- mu_i, s_i^2 in closed form, deterministic, no reparam-gradient); PSIS-k-hat not R-hat (Sec.18.8) |

**Building a NUTS-gradient block -- compose a child `nuts_block`, do NOT hand-roll NUTS.** The
clean Tier-B pattern for the NUTS-gradient row is to internally OWN a child `nuts_block` (or
`joint_nuts_block`) and feed it your natural-scale log-density + analytic gradient -- not to
re-implement the sampler. Read `AI4BayesCode/include/AI4BayesCode/nuts_block.hpp` for the config
contract; the load-bearing pieces:
- the oracle `log_density_grad(const arma::vec& theta, const block_context& ctx, arma::vec* grad)`
  returns `logp` and writes the **natural-scale** gradient into `*grad` (Check #5: NO hand
  Jacobian -- any constrained slice uses `constraints::<kind>::wrap` from `constraints.md`);
- `initial_step_size`, `initial_unc` (the unconstrained init);
- **`n_warmup_per_step` MUST stay 0** (Check #20 -- any nonzero re-enables the 2026-04-12
  chain-corruption mechanism).
Your block's `step()` delegates to the child's `step()`; `current()` returns its draw. A block
that internally owns a NUTS kernel **IS NUTS-family** -- it returns `supports_readapt() == true`,
so the Tier-A wrapper must expose the **kernel-control method `readapt_NUTS(n, reset=false,
max_tree_depth=-1)`** (forwarding to the child), even though the new block itself is not literally
a `nuts_block` composite child. The Tier-A wrapper ALSO exposes `freeze / unfreeze / get_frozen`
unconditionally (kernel-control category, interface.md Sec.1); those bind via the
`AI4BAYESCODE_BIND_KERNEL_CONTROL(ClassName)` macro from
`include/AI4BayesCode/kernel_control_mixin.hpp`.

**Step 2.2 -- Metric + warmup (NUTS / joint-NUTS only).** Follow `families.md` "Metric + warmup
decision" -- the **ESCALATION-DRIVEN-BY-DIAGNOSTICS** ladder, NOT a dimension threshold:

1. START diagonal + single-pilot (the common case).
2. Escalate to DENSE + single-pilot ONLY if runtime diagnostics (R-hat > 1.01, low ESS,
 tree-depth saturation, low E-BFMI) show diagonal is inadequate. Dense is **opt-in**, justified
 per Check #18 -- never speculative (`families.md`; the 2026-06-20 policy).
3. DENSE + 3-phase (`use_three_phase_warmup = true`) ONLY for a documented extreme-cond bootstrap
 failure. **`use_three_phase_warmup` DEFAULTS to `false`.** The one FIRM, code-enforced rule:
 **diagonal + 3-phase is BANNED.**
4. NON-Gaussian geometry (funnel / centered-hierarchical / curvature) is NOT a metric problem --
 no metric helps; the fix is REPARAMETERIZATION.

> **The centered hard FAIL (do not design around it).** A CENTERED scale-governed effect (the
> funnel) is a hard **Check #24(a) FAIL**. Non-centered reparameterization (NCR) is MANDATORY;
> dense metric is NOT an accepted substitute (the funnel freezes under any static metric -- see
> `validator.md Sec.24` / `joint_nuts_failure.md` and `families.md`). There is NO centered+dense escape hatch. If
> your block parameterizes a hierarchical scale x effect, design it non-centered from the start.

There is NO "Check #11.7" -- do not cite it (cleared 2026-06-20).

**Non-Gaussian / discrete variational families (e.g. Bernoulli mean-field).** `vi.md` Sec.18 and
the `vi_block` base contract are written around LOCATION-SCALE (Gaussian) families -- the state
is `(mu, log_sd)` and `set_variational_state(mu, log_sd)` / `get_log_sd()` are Gaussian-shaped.
A discrete or otherwise non-Gaussian variational family IS still a legitimate `vi_block`
(`engine_kind == VI`): MAP your family's parameters into the `mu` / `log_sd` slots and DOCUMENT
the mapping in the block header (e.g. a Bernoulli mean-field uses `mu := q_i`,
`log_sd := log sd(Bernoulli(q_i))`). Closed-form **CAVI** is the natural optimizer (the
mean-field fixed point is analytic -- no reparam-gradient). Precedent: the local
`gmrf_meanfield_cavi_block` (Gaussian) and `ising_meanfield_vi_block` (Bernoulli). `vi.md`
Sec.18.10 defers a first-class non-Gaussian VI contract to a future version; until then this
slot-mapping is the **documented** path -- state it explicitly in the header, it is not a hack
to hide. (PSIS-k-hat can be ill-defined / NaN at a near-point-mass q -- report that transparently,
do not gate on it; it is a PSIS limitation, not a block bug.)

> **SIGN-OFF 2 (algorithm).** Lock: engine kind / family; the sampling step's math; metric +
> warmup choice with its diagnostic/Check-#18 justification (or "N/A -- not NUTS"); for VI, the
> family (mean-field vs full-rank, Sec.18.5) + the q-sample-to-shared_data commitment.

**Label switching / non-identification -- NOT the block's job; resolve it POST-MCMC.** If the
target is invariant under permuting K exchangeable components (finite mixture, HMM states, LDA
topics, DP/BNP clusters -- see `label_switching.md` Sec.1), the block MUST sample that
(non-identified, multimodal) target AS-IS. Do **NOT** "fix labels" inside the block:

- no in-sweep relabeling / reordering of the state -- it is not a valid MCMC move and corrupts
  the chain;
- do not silently auto-impose an ordering constraint (`mu_1 < mu_2 < ...`) to force
  identification. Ordering IS available as a **user-declared modelling choice** -- an `ORDERED`
  constraint slice in Stage 3, the convention `label_switching.md Sec.2` documents -- but the USER
  states it in the spec; the block never adds it on its own.

Pure label-switching relabeling (simple-sort / Stephens 2000 / Hungarian-to-truth) is a
**post-MCMC** step on the saved draws, owned by the analysis / runner layer, NOT the sampler --
see `label_switching.md` (Sec.3-Sec.6). A correct sampler that explores all K! modes is fine; collapsing
them to one labeling is post-processing. If the block faces this: set the manifest flag
**`LabelSwitching: true`** (`skill.md` Sec.3.2) so the analysis / runner layer knows to relabel, AND
record it as a block-local failure-mode note (`skill.md` Sec.2.2) citing `label_switching.md` -- never
as in-block logic.

---

## Stage 3 -- CONSTRAINT mapping (no hand-written Jacobians, ever)

> LOAD: `constraints.md` (the 15 `joint_constraint` kinds + the standalone `constraints::*::wrap`
> table) and `skills/system_design_skills/jacobian.md` (system_design Sec.10). The universal rule
> (Sec.10): **users NEVER hand-write a Jacobian formula -- in NO family.** It is the single most
> violated invariant in AI-generated code.

**Step 3.1 -- Map each constrained parameter to a supported transform.** For a standalone
`nuts_block`-style block use the `constraints::<kind>::wrap` table in `constraints.md` (real /
positive / simplex / lower_/upper_bounded / interval / ordered / cholesky_corr / unit_vector,
with step-size-seed flags). For a joint block, declare per-slice `joint_constraint` from the
**15 supported kinds** (`constraints.md`: REAL, POSITIVE, LOWER_BOUNDED, UPPER_BOUNDED, INTERVAL,
ORDERED, POSITIVE_ORDERED, UNIT_VECTOR, OFFSET_MULTIPLIER, SIMPLEX, SUM_TO_ZERO, CHOLESKY_CORR,
CHOLESKY_FACTOR_COV, CORR_MATRIX, COV_MATRIX). Do NOT split a constrained parameter into its own
block -- keep it a slice (`constraints.md`).

**`joint_nuts_block` per-slice contract -- three foot-guns (lift these from the header; do NOT
rediscover them by a constructor throw):**
1. **`sub_param.dim` is the NATURAL width**, not the unconstrained width. For a SIMPLEX over K
   categories set `dim = K` (the block derives the K-1 unconstrained internally); setting K-1
   is wrong. Same for the matrix kinds (natural element count).
2. **`initial_cat` is on the NATURAL scale and is support-validated in the constructor.** An
   ORDERED slice must be strictly increasing; a SIMPLEX slice must sum to 1; a POSITIVE slice
   must be > 0 -- a bad init THROWS at construction with no upstream warning. Seed a valid
   natural-scale point.
3. **Your natural-scale gradient is the FULL natural-width vector** (e.g. the full K-vector for
   a SIMPLEX slice), NOT the K-1 unconstrained gradient. The library chain-rules your natural
   gradient through the transform -- you never write the K->K-1 (or log|J|) bookkeeping (Step 3.2).

**Step 3.2 -- The Jacobian lives in the library, not your code** (`jacobian.md` Sec.10.1): the
user-supplied log-density lambda writes ONLY the natural-scale `log p` + natural-scale gradient.
NO `+ log(sigma)`, NO `* sigma`, NO hand-rolled log-det. `constraints::<kind>::wrap` adds
`log|J|` AND chain-rules the gradient. This is **Check #5** -- and for a block with a hand-written
gradient pair, the **Check #12** semantic AD-verify via a throwaway compile at generation (`jacobian.md` Sec.10.1: paste functions, add
`autodiff::var` twins, compare 5-20 random-theta points < 1e-8, delete on PASS) is mandatory.
Matrix-valued / dimension-changing joint slices use the library's dual-offset scheme and
auto-enable the diagonal metric -- that is library machinery, not yours.

**The CONSTRAINT Jacobian (banned) vs the MODELLING change-of-variable (yours) -- do NOT
conflate them.** "Never hand-write a Jacobian" targets the CONSTRAINT-transform `log|J|` that
maps a sampled unconstrained coordinate onto a parameter's natural SUPPORT (POSITIVE's `log`,
SIMPLEX's stick-breaking, ...) -- that one is ALWAYS library-owned (`constraints::<kind>::wrap`).
It is a DIFFERENT object from a **modelling change-of-variable**: a prior the user CHOSE to write
on one parameterization while you sample another. Example: `tau ~ HalfNormal(0,1)` but you sample
`log_tau`, so the model density needs `p(log_tau) = p(tau)*tau` -> a `+ log_tau` term. That CoV is
part of the MODEL -- it belongs in your natural-scale `log p`, and you DO write it. **The clean
pattern: declare the sampled coordinate (`log_tau`) as a `REAL` slice** (no constraint wrap, so
there is no library `log|J|` to double-count) and put the user-chosen prior's CoV term in your
own log-density. Rule of thumb: a `log|J|` that exists ONLY to honor a parameter's SUPPORT =
library's job; a CoV that exists because the USER expressed the prior on a different scale =
model = yours. (Funnel/NCR blocks hit this -- the half-Normal-on-`tau`-but-sample-`log_tau` prior
is the canonical case.) **Watch the "invisible CoV" trap:** when the user's prior is
`sigma ~ LogNormal(mu,s)` and you sample `l_sigma = log sigma`, the CoV is ALREADY baked in -- `LogNormal` just
means `log sigma ~ Normal(mu,s)`, so the transported prior is `N(l_sigma; mu, s^2)` with NO extra `+ log sigma`.
Adding an explicit `+ log sigma` on top of the Normal-on-log density DOUBLE-COUNTS (the posterior mean
of `l_sigma` then lands at `mu + s^2` instead of `mu` -- a clean T4 oracle). Write an explicit CoV term
ONLY when the prior density is given on the natural parameter (HalfNormal / HalfCauchy / Gamma on
`sigma` itself), NOT when it is already a log-scale (LogNormal) prior.

**Step 3.3 -- Trans-dimensional case** (only if Stage 1 routed you to an RJ-style structured
block): compose the shipped core **`rjmcmc_block`** (`include/AI4BayesCode/rjmcmc_block.hpp` -- read
its contract). The proposal Jacobian is ALSO library-owned (`jacobian.md` Sec.10.2) --
identity-coordinate (|det J| = 1), library 1D transforms, or runtime-AD on a single templated
forward + analytic inverse (Check #14). `rjmcmc_block_config` has NO `jacobian_fn` slot. The user
still writes no Jacobian.

**Know the core `rjmcmc_block`'s SCOPE before routing here.** It is a per-coefficient
**spike-and-slab over a FIXED candidate set of size p**: the "dimension change" is encoded by `gamma`
indicators in fixed-`2p` storage (a birth = flip `gamma_j` on + draw its slab), NOT genuine
variable-length storage. A variable COUNT over a fixed grid fits cleanly -- a **changepoint** model
fits by (a) discretizing candidate locations onto a fixed grid AND (b) marginalizing the segment
parameters (so there is no continuous slab to birth). A genuinely **free continuous location**
(not grid-snapped) or an UNmarginalized segment parameter is NOT the identity path -- it needs the
transform/bijection route (`rjmcmc_custom_bijection.hpp`) + a non-trivial `propose_sample`. Flag
that boundary in the block's WHEN-NOT.

**Conjugate-marginalized -> the "constant-proposal" trick.** `rjmcmc_block` requires a
`propose_sample`/`propose_logq`, but if you marginalized the segment/coefficient parameters there
is nothing to propose. Use a **constant `propose_logq`** so the forward-birth and reverse-death
proposal terms cancel exactly -- the accept ratio reduces to the pure marginal-likelihood (xprior)
ratio. Sanity-check `birth_logR == -death_logR` on a fixed pair.

> **SIGN-OFF 3 (constraints).** Lock: every parameter's constraint kind and its library
> transform; confirm the log-density is natural-scale only (Check #5 clean); confirm whether
> Check #12 applies (any hand-written gradient). After this sign-off the METHODOLOGY is locked --
> code may be written.

---

## Stage 4 -- THREE-TIER interface (the central confusion: C++ contract != R wrapper)

> LOAD: `skills/system_design_skills/interface.md` (system_design Sec.1-Sec.2) and
> `skills/system_design_skills/lifecycle.md` (system_design Sec.14). **Keep the two contracts
> SEPARATE** -- conflating them is the recurring error.

**The new block class implements the C++ `block_sampler` contract (Tier B).** It is a subclass of
`block_sampler` (`include/AI4BayesCode/block_sampler.hpp`) overriding the pure-virtual contract:
`set_context(const block_context&)`, `step(std::mt19937_64&)`, `current `, `set_current(arma::vec)`,
`name `, `dim `, and `engine_kind ` (default `MCMC`; override to `VI` for a VI block). This is
C++-only machinery -- NONE of these get a `.method ` and R never sees them (interface.md Sec.2,
Tier B). The lifecycle to follow is `lifecycle.md` Sec.14's 7-step (Tier C kernel + license-gate ->
Tier B block -> Tier A wrapper -> skills/catalogue -> tests -> validator -> cross-chain audit); for
block_design, Steps 4-7 land in the `blocks_local/<Block>/` bundle, NOT in core.

**The R wrapper contract is a SEPARATE thing (Tier A).** A Tier-A example wrapper (part of the
REQUIRED `examples/<Model>.cpp` -- every block ships exactly ONE example; see the EXAMPLE phase)
exposes to R the **core-six state methods**
(`step / get_current / set_current / predict_at / get_dag / get_history`) plus the
**kernel-control category**: `freeze / unfreeze / get_frozen` UNCONDITIONALLY (via the
`AI4BAYESCODE_BIND_KERNEL_CONTROL(ClassName)` macro + `kernel_control_mixin<ClassName>` CRTP
inheritance), and `readapt_NUTS` ONLY if the composite contains a NUTS-family child
(interface.md Sec.1). Tier A's `set_current(Rcpp::List)` is a pure DISPATCHER routing keys into
Tier B's fine-grained C++ setters (interface.md Sec.2; dispatcher contract in `dataflow.md` Sec.7).
**There is NEVER a `$set_X`, `$set_Y`, `$set_sigma`, or any other state-method on the wrapper**
(interface.md Sec.1; the Sec.17 anti-patterns live in `lifecycle.md`) -- kernel-control methods
(`freeze / unfreeze / get_frozen / readapt_NUTS`) are the only allowed extensions. The C++
contract and the R contract are DIFFERENT vocabularies at DIFFERENT tiers; the block class
implements the former, never the latter.

**Tier C / license gate.** If your block vendors a new kernel, it goes under
`blocks_local/<Block>/vendor/` with the upstream LICENSE preserved verbatim, and must pass the
GPL-3-compatibility gate (`lifecycle.md` Sec.14 Step 1: GPL-3 / GPL-2+ / MIT / BSD / MPL-2.0 /
Apache-2.0 / CeCILL-B OK -- the combined work is already GPL-3.0-or-later, so Apache needs no
special branch; non-free / GPL-3-incompatible -> reject). The system's already-vendored libs are
on `-I` and need no per-block declaration.

> **SIGN-OFF 4 (interface).** Confirm the new class's Tier-B `block_sampler` override list and
> any fine-grained C++ setters; confirm (if a wrapper ships) the R contract matches interface.md
> Sec.1 formula = core-6 state methods + kernel-control category
> (`freeze / unfreeze / get_frozen` always via `AI4BAYESCODE_BIND_KERNEL_CONTROL` macro +
> `kernel_control_mixin` CRTP; `readapt_NUTS` iff NUTS-family child; BART tree-serialization
> carve-out iff BART-family child), with set_current-as-dispatcher and zero state-method leaks;
> confirm the license gate for any vendored kernel.

---

## Stage 5 -- WIRING (shared_data, refreshers, predict edges, RNG)

> LOAD: `skills/system_design_skills/dataflow.md` (system_design Sec.3-Sec.9).

- **shared_data sync** (Sec.3): a single-parameter block's `name ` IS its data key. After ANY
 `set_current`, every shared_data entry the block advertises (data inputs, derived quantities)
 MUST reflect the new state BEFORE the next `step ` -- a desync silently feeds DAG traversal and
 predict_at the OLD value (Sec.3 invariant).
- **Refreshers** (Sec.5): deterministic (`register_refresher`, pure fn of shared_data, e.g.
 working residual) vs stochastic (`register_stochastic_refresher`, takes an rng, called ONLY by
 predict_at, e.g. `y_rep`). A wrapper expecting Layer-3 R3 MUST register a `y_rep` stochastic
 refresher. R3.b PSIS-LOO is DIAGNOSTIC ONLY -- `warning `, never `stopifnot ` (Sec.5).
- **Two DAGs, kept disjoint** (Sec.4): the Gibbs DAG (`declare_dependencies` / `declare_invalidates`
 -- who READS whom, for sampling) is SEPARATE from the predict DAG (`declare_predict_edges` -- who
 PRODUCES whom, generatively, for predict_at BFS). A new derived quantity needs BOTH its Gibbs
 invalidation AND its predict edge. The viz-only context-edge set (`declare_context_edges`) is
 NEVER traversed by predict_at and must never overlap `predict_edges_` (Sec.4).
- **RNG separation** (Sec.8): the wrapper carries `rng_` (MCMC, advanced by `step `),
 `mutable predict_rng_` (predict_at only), and -- iff NUTS-family -- `mutable readapt_rng_`. Seeds
 are the documented SplitMix64-mixed constants (Sec.8). Conflating streams is a silent reproducibility
 bug (Check #13 / Check #23). Note: BART-family uses R's RNG; the mt19937 arg is ignored.
- **History / memory** (Sec.9, Sec.12): `keep_history` set once at construction; leaves never store a
 raw `shared_data_t*` (copy or use the per-sweep `ctx` reference).

> **SIGN-OFF 5 (wiring).** Confirm shared_data keys + sync points, refresher kinds, both DAG
> edge-sets (disjoint), and RNG streams.

---

## EXCEPTION taxonomy -- legitimizing the custom block (Check #17, Exception 4)

A custom new block authored by block_design is the **structural-gap LAST resort**, legitimized by
`codegen_priors Sec.2b` **Exception 4** (AI-authored custom block/sampler) -- NOT the Gibbs
exceptions (#16 / Exception 1/2/3, which justify a `*_gibbs_block`). Exception 4 applies when,
judged AT DESIGN TIME from model structure (not from a bad R-hat at runtime), (a) no specialized
or conjugate-Gibbs block fits AND (b) neither `joint_nuts_block` nor `nuts_block` is structurally
appropriate. Exhaust the specialized blocks and the joint-NUTS default FIRST; within Exception 4
still PREFER a NUTS-based custom log-density over hand-written Gibbs (the Check #17 ban on inline
distribution samplers still holds -- Exception 4 is not a backdoor for it).

The block MUST carry the inline justification comment (verbatim form from `codegen_priors Sec.2b`):

```cpp
// JUSTIFICATION (Check #17): Exception 4 -- no blessed block fits because <...>;
// joint/single NUTS structurally inapplicable because <...>; custom scheme = <...>;
// targets the correct posterior because <...>.
```

The block choice is fixed at design time and NEVER swapped at runtime.

---

## Phase output + the methodology-lock gate

**Output of this phase: the STAGED `<Block>.hpp`** -- the Tier-B class implementing the
`block_sampler` contract (Stage 4), with the constraint/Jacobian discipline (Stage 3) and the
wiring hooks (Stage 5) wired, the geometry/target declared in its header comment (Stage 1), and
the Exception-4 justification comment present. It is written to a STAGING dir, NOT to
`blocks_local/<Block>/` -- the MOVE, the compile, and the library test happen only on explicit
"go" (TOP RULE; automation philosophy).

**The hard gate:** the `.hpp` is written ONLY AFTER Sign-offs 1-3 (geometry -> algorithm ->
constraints) are locked -- methodology before code (SECOND RULE). Sign-offs 4-5 (interface,
wiring) are locked as the `.hpp` is drafted. The subsequent phase (TEST / manifest / block-local
skill / staging-to-bundle) is handled by the next block_design module -- do not preload it here.
