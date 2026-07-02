<!-- system_design MODULE — Jacobian discipline (extracted 2026-06-21 from system_design.md §10).
     SINGLE LIVE SOURCE: edit HERE, not the monolith. system_design.md is now a thin index
     (§N -> module).
     Cross-refs keep the "§N" scheme, resolved via the system_design.md index. -->

## 10. Jacobian discipline across block families

The single most violated architectural invariant in AI-generated
code. Read it twice.

### Universal rule: users NEVER hand-write Jacobian formulas

Two kinds of Jacobian appear in AI4BayesCode code, owned by different
layers:

- **Constraint Jacobians** (`nuts_block`, `joint_nuts_block*`) — from
  the unconstrained → natural transform (`log` for positive,
  stick-breaking for simplex, etc.). OWNED by library: computed
  inside `constraints::<kind>::wrap(...)`.

- **Proposal Jacobians** (`rjmcmc_block`) — from the trans-
  dimensional bijection between old and new parameter spaces in
  Green 1995 MH acceptance ratio. OWNED by library or auto-computed:
  using identity-coordinate proposals (Jacobian = 1 by construction);
  library-provided transforms (`include/AI4BayesCode/rjmcmc_transforms.hpp`
  — ported from librjmcmc); and runtime autodiff on a user-supplied
  templated forward + analytic inverse
  (`include/AI4BayesCode/rjmcmc_custom_bijection.hpp`).

**In no block family does a user ever write a Jacobian formula.**
The rule is universal: Jacobians are framework-level machinery, not
per-model content. The 2026-04-19 survey of 7 RJMCMC packages
(NIMBLE, rjmcmc CRAN, librjmcmc, BayesMix, CU-MSDSp, BoomSpikeSlab,
spikeSlabGAM) confirmed zero of them require user-written Jacobians
either.

### §10.1 NUTS-family — constraints::<kind>::wrap

Every unconstrained NUTS sample must flow through
`constraints::<kind>::wrap(...)` from
`include/AI4BayesCode/constraints.hpp`. The wrap adds `log|Jacobian|`
AND chain-rules the gradient automatically. The user-supplied
log-density lambda ONLY writes the natural-scale log p and
natural-scale gradient — no `+ log(sigma)`, no `* sigma`, no
hand-rolled log-det.

Violating this produces a compile-and-run-but-wrong-posterior bug
that Layer 3 R3 might catch (if the observation model is
sensitive) but Layer 2 Check #5 MUST catch at code review.

#### Rule: Check #12 (AD verification) runs at gen-time

For every block with a hand-written log-density + gradient pair,
the code-gen agent writes a throwaway
`tests_autodiff/verify_<ClassName>.cpp` that pastes the hand-
written functions verbatim, adds templated `autodiff::var` versions
of the same math, and compares 5–20 random-θ points. Max diff must
be < 1e-8 (AD) or < 1e-5 (finite-diff fallback for blocks using
`lgamma` / `digamma`). On PASS, the verify file is DELETED. On
FAIL, the hand-written gradient has a bug; fix it before shipping.

The production `.cpp` is scaffolding-free: no `#ifdef
AI4BAYESCODE_VERIFY_GRADIENTS`, no autodiff includes, no Eigen
includes.

See `skills/validator.md` §12 for the full template.

#### joint_nuts_block per-slice constraints (current scope)

Supports 15 per-slice constraint kinds. DIMENSION-PRESERVING (unconstrained
slice width == natural slice width): REAL, POSITIVE, LOWER_BOUNDED,
UPPER_BOUNDED, INTERVAL, ORDERED, POSITIVE_ORDERED, UNIT_VECTOR,
OFFSET_MULTIPLIER. DIMENSION-CHANGING (unconstrained width != natural width):
SIMPLEX (K-1→K), SUM_TO_ZERO (K-1→K), CHOLESKY_CORR, CHOLESKY_FACTOR_COV,
CORR_MATRIX, COV_MATRIX. The block applies each per-slice transform and adds
`log|J|` internally (via `constraints::<kind>::wrap`); the user's log-density
stays on the natural scale. Dimension-changing kinds use a DUAL offset scheme
(`unc_offsets_/unc_dims_/total_unc_dim_` for the unconstrained sampler+metric
space; `offsets_/total_dim_` for the natural oracle space), gated by
`has_dim_changing_` so dim-preserving blocks stay bit-identical. Matrix-valued
kinds (CHOLESKY_CORR/_FACTOR_COV, CORR_/COV_MATRIX) AUTO-ENABLE the diagonal
metric (identity freezes their ill-conditioned posteriors). Validated 2026-06-17:
FD K=2..6 ≤3e-9, multi-chain R-hat ~1.001 + correct recovery, incl. heterogeneous
mixed blocks. `stochastic_column`
remains single-`nuts_block`-only (wrap needs a columns count).

### §10.2 RJMCMC-family — three-tier Jacobian story

`rjmcmc_block` is AI4BayesCode's trans-dimensional kernel. The identity-coordinate path shipped
2026-04-19 (reference template `examples/SpikeSlabRJMCMC.cpp`); the library 1D transforms
shipped 2026-04-20 (type-erased 1D transforms with library-computed
Jacobians); custom bijections with runtime AD are also supported (`rjmcmc_custom_bijection.hpp`).

#### Identity-coordinate proposals (SHIPPED)

Birth proposes a new coordinate `β_j_new ~ q(β | context)`
directly; the proposed β_j IS the auxiliary variable u. The
bijection `(x, u) → (x', u')` is the identity, `|det J| = 1` by
construction, and the accept ratio contains no Jacobian term.

**The user supplies:**

- `log_joint(gamma, beta, ctx)` — joint log density on (γ, β) given
  everything else (NATURAL scale). Standard Check #5 applies
  (no `+ log(…)` Jacobian correction).
- `propose_sample(rng, j, ctx)` — draws a new β_j.
- `propose_logq(beta_new, j, ctx)` — log q evaluated at β_new.
- `continuous_update(rng, j, ctx)` (optional) — direct update of
  β_j when γ_j = 1; typically a closed-form Gibbs draw from the
  conditional posterior. Critical for good mixing; without it β_j
  stays at its birth-time value for as long as γ_j = 1.

**The user does NOT supply a Jacobian formula anywhere.** The
`rjmcmc_block_config` has no `jacobian_fn` slot.

Validator implications: the existing Check #5 (no `+ log(...)` in
natural-scale log-density lambda) applies unchanged to `log_joint`.
No new Check is needed for the identity-coordinate path.

#### Library-provided 1D transforms (SHIPPED 2026-04-20, T5)

For birth proposals that are linear / affine rescalings of an
auxiliary 1D variable (e.g., "propose the new coefficient as
`β = σ × u` for u ~ N(0,1)"), `rjmcmc_block_config` accepts an
optional `transform` field holding a type-erased
`rjmcmc_transforms::transform_1d_base`. The library ships three
concrete wrappers at N=1 (where linear = diagonal_linear and affine
= diagonal_affine):
- `identity_transform_1d`       (β = u, |J| = 1)
- `diagonal_linear_transform_1d(scale)` (β = scale × u, |J| = |scale|)
- `diagonal_affine_transform_1d(scale, offset)` (β = scale × u + offset)

Each wrapper computes `|det J|` internally. The user writes NO
Jacobian formula — the transform class owns that.

**Semantic change when `transform` is set**: the user's
`propose_sample` and `propose_logq` now operate on the **auxiliary
u**, not on β_new. The library transforms u → β and computes the
Jacobian for the MH accept ratio (Green 1995).

When NOT set (default nullptr), the identity path is used
unchanged. All existing identity-only examples work as-is; the library transforms are purely
additive.

`include/AI4BayesCode/rjmcmc_transforms.hpp` hosts both the N-dim
templated transforms (from the T4 librjmcmc port, CeCILL-B
attribution) and the new 1D type-erased wrappers (library transforms).
Multi-dim birth (birthing a block of coefs at once via a single
transform) is a future multi-dim extension not required for per-
coefficient spike-and-slab style problems.

Tests: `tests_autodiff/test_rjmcmc_block_transform.cpp` covers
(i) wrapper arithmetic correctness, (ii) byte-identical parity
between identity-via-transform and no-transform path (identity-vs-no-transform regression
guard), (iii) analytical-posterior agreement at scale=2 (Jacobian
correctness check).

#### Custom bijection with runtime AD Jacobian (SHIPPED)

For the exotic cases where neither identity nor any linear transform
fits (custom non-linear monotone maps, asinh / sinh, user-tuned
proposals), the user supplies a **single templated forward map**
plus a non-templated analytic inverse:

```cpp
struct UserForward {
    template <typename T>
    T operator()(T u) const { return /* map at scalar T */; }
};
struct UserInverse {
    double operator()(double beta) const { return /* T^{-1}(beta) */; }
};

cfg.transform = AI4BayesCode::rjmcmc_transforms::
    make_templated_bijection_1d(UserForward{}, UserInverse{});
```

The framework instantiates `UserForward` at:
- `double` for the sampling step (`β = T(u)`),
- `autodiff::var` for runtime AD computation of `|dβ/du|`.

The wrapper class `templated_bijection_1d` (in
`include/AI4BayesCode/rjmcmc_custom_bijection.hpp`) satisfies the existing
`transform_1d_base` interface that already plumbs through
`rjmcmc_block`. No changes to `rjmcmc_block.hpp` were needed —
The custom-bijection class piggybacks on the transform contract.

**Why single-template instead of "two functions" (`bijection_double`
+ `bijection_var`):** the original sketch asked the user for two
functions and Check #14 verified they encoded the same map. A
single template instantiated twice eliminates the typo class by
construction (one source of truth for the math). Check #14 retains
a useful role — see below — but no longer needs to compare twin
implementations.

**Updated Check #14 (SHIPPED, see validator.md §14):** three
numerical sanity probes on a small grid of u points, all derived
from the SINGLE templated forward + the user's analytic inverse:

1. **Round-trip**: `|inv(fwd(u)) - u| < 1e-10` — catches typos in
   the analytic inverse.
2. **Jacobian non-singularity**: `|dβ/du| > 1e-12` — catches
   degenerate forwards (poles, constant sections, sign flips).
3. **Forward / reverse Jacobian inverse-pair**:
   `|fwd_J(u) * rev_J(T(u)) - 1| < 1e-10` — the bijection invariant.

On any probe failure the agent fixes the bijection before shipping;
companion file `tests_autodiff/verify_<ClassName>_v1.cpp` is
generated, run, and deleted on PASS (mirrors the Check #12
companion-file pattern).

**Users still never write a Jacobian.** They write the forward map
once (templated), an analytic inverse once (non-templated), and the
framework computes `|det J|` automatically via autodiff.

**Out of scope at this version:** multi-dim bijections `R^n → R^n` with
`n > 1`. The `rjmcmc_block` API is per-coefficient (1D-to-1D);
multi-dim bijections (e.g., Stephens 2000 split-merge for finite
mixtures with unknown K) belong to a future block class and are
not part of the current scope.

---

