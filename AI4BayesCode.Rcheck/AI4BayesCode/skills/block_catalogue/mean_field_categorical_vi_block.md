## mean_field_categorical_vi_block (v1.2 SHIPPED 2026-05-31)

```cpp
mean_field_categorical_vi_block_config cfg;
cfg.name          = "z";                          // shared_data key for q-sample
cfg.cardinalities = arma::uvec{2, 3, 2, 4};       // K_i per latent variable
cfg.log_density   = [](const arma::uvec& z,
                       const block_context& ctx) -> double {
    // user-supplied log p~(z, ctx), evaluated on integer-encoded z
};
cfg.dependencies     = {"y", "X"};
cfg.exact_enumeration = false;                    // MC by default
cfg.n_mc_samples      = 16;                       // gradient MC samples
cfg.optimizer         = vi_optimizer::raabbvi_config{};
auto blk = std::make_shared<mean_field_categorical_vi_block>(cfg);
impl_->add_child(std::move(blk));
```

**What it does**: maintains `q(z) = ∏_i Categorical(z_i; φ_i)` over
`n` discrete latents (cardinalities `K_i` per variable), with internal
anchored-softmax parameterisation `η_i ∈ R^{K_i−1}` and `φ_{i,0}`
anchored to `1 − Σ_{l>0} φ_{i,l}`. Optimises `−ELBO` via RAABBVI on
the analytically-correct gradient `∂ELBO/∂φ_{i,k} = G_{i,k} − log
φ_{i,k} − 1` where `G_{i,k} = E_{q\i}[log p̃(z_{-i}, z_i = k)]`.

**Two gradient modes** (selected by `cfg.exact_enumeration`):
- **Exact**: enumerate full joint state space `∏_i K_i` (capped at
  `cfg.exact_state_cap`, default 4096). Used for unit tests and small
  models. Zero variance.
- **Monte Carlo** (default): sample `S = cfg.n_mc_samples` joint
  draws `z_s ~ q`; for each `(i, k)`, substitute `z_s[i] := k` and
  evaluate `log_density`. Cost = `S × Σ_i K_i` evaluations per step.

**JUSTIFICATION (Check #16)**: Discrete latents with strong local
dependence (`system_design.md` §11.2(b)). Per-site Gibbs irreducible
but mixes catastrophically on coupled chains; categorical mean-field
VI is the deterministic deterministic alternative — converges cleanly
to a (biased) joint with correct marginals in many regimes, with
known underestimate-of-joint-variance caveat (Bishop §10.1.2)
correctly diagnosed by PSIS-k̂.

**Empirical validation** (`tests/test_mean_field_categorical_vi_chain.cpp`
on 4-node K=3 Potts chain at β=0.8 vs 81-state exact enumeration):

| Test | Result |
|---|---|
| Symmetric chain, per-var KL(q‖p) | < 1e-5 (all 4 vars) |
| Asymmetric chain (h ≠ 0), per-var KL | max 0.00144 |
| Asymmetric chain, Pearson cor(q,p) on 12-vec | **0.999** |
| ELBO ≤ log Z (variational bound) | PASS |
| Joint PSIS-k̂ on coupled chain | 5.78 (FAIL — correctly diagnoses MF bias) |

**Public interface** (`vi_block` contract):
- `current()` returns concatenated φ (length `Σ_i K_i`) — the q-mean
  point estimate
- `current_sample(rng)` returns one z draw (length `n` integer
  indices in `{0..K_i-1}`)
- `get_log_sd()` returns the unconstrained η vector (length `Σ_i (K_i−1)`)
- `set_current(phi_concat)` accepts a probability vector and recovers
  η via inverse softmax
- composite_block writes integer-indexed q-samples to shared_data
  under `cfg.name` (per §18.4 hybrid-correctness invariant)

**Reference template:**
- `examples/CategoricalIsingChainVI.cpp` — n-node K-state Potts chain
  with optional per-node external field `h`, RAABBVI-converged VI
  marginals + predict_at sampling. R-level audit
  (`tests/audit_CategoricalIsingChainVI.R`) cross-checks per-var KL
  < 0.05 vs 81-state exact enumeration on n=4 K=3 chain.

**Caveats (Bishop §10.1.2, validated by k̂ diagnostic)**:
- Marginals recovered well; joint covariance NOT captured by MF
- For tightly coupled targets (large β, dense graph), use Block 5
  (`structured_categorical_vi_block`, v1.2 future) with user-specified
  clique factorisation, OR drop to MCMC

**Scope (v1.2 ship):**
- Cardinality `K_i ≥ 2` per variable, arbitrary mix across nodes
- Generic `log_density` callback — no exponential-family assumption
- Both exact and Monte Carlo gradient modes
- RAABBVI optimizer + PSIS-k̂ Layer-3 diagnostic

**Deferred to v1.2.1:**
- Structured factorisation via user-specified cliques (this is Block 5)
- CAVI closed-form updates when `log_density` is exp-family (faster
  alternative for that special case)
- Continuous-relaxation alternatives (Concrete / Gumbel-softmax) —
  explicitly rejected by plan §4 (analytical-KL is more accurate)

**Constraints + validator obligations**: same as
`mean_field_gaussian_vi_block` (Checks #21–#22 plus the Layer-3 R2-VI
PSIS-k̂ diagnostic).
