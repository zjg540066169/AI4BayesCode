## mean_field_gaussian_vi_block (primary v1)

```cpp
mean_field_gaussian_vi_block_config cfg;
cfg.name           = "<shared_data key>";
cfg.initial_unc    = arma::vec{...};                // unconstrained mean init
cfg.initial_log_sd = arma::vec(K, arma::fill::value(-2.0));  // log σ init
cfg.constrain      = constraints::positive::constrain;       // omit for real
cfg.unconstrain    = constraints::positive::unconstrain;     // omit for real
cfg.log_density_grad = /* same lambda you'd write for nuts_block */;
cfg.dependencies     = {"y", "X", ...};   // shared_data keys this block reads
cfg.optimizer        = vi_optimizer::raabbvi_config{};  // default: RAABBVI
auto blk = std::make_shared<mean_field_gaussian_vi_block>(cfg);
impl_->add_block(blk);
```

**What it does**: maintains q(η) = ∏_i N(η_i; μ_i, σ_i²) on the
unconstrained scale η = constraints::unconstrain(θ). Each step()
call runs ONE RAABBVI optimizer step (see `vi_optimizer.hpp` below
for the algorithm; `system_design.md §18.7` for the why). After
step(), `composite_block` writes a q-SAMPLE θ = constrain(η_draw)
to shared_data under `cfg.name`; siblings reading that key see a
fresh draw each outer iteration. The R-level `get_current()`
returns the q-mean (point estimate), not a sample.

**Public interface (Tier B)**:

| Method | Returns | Notes |
|---|---|---|
| `step(rng)` | void | one RAABBVI optimizer step |
| `current()` | `const arma::vec&` | q-mean = constrain(μ); deterministic |
| `current_sample(rng)` | `arma::vec` | q-sample = constrain(μ + σ⊙ε), ε~N(0,I) |
| `set_current(μ)` | void | overwrites variational mean ONLY |
| `set_variational_state(μ, log_sd)` | void | overwrites both |
| `get_log_sd()` | `arma::vec` | current log σ |
| `current_elbo()` | double | last-step ELBO |
| `history()` | `const vi_history_t&` | per-step (elbo, μ, log_sd, γ, epoch) + final_khat |

**Constraints**: uses `constraints::*::wrap` identical to
`nuts_block`. The user's log-density lambda lives on the natural
scale; no hand-written Jacobian. Validator Check #5 unchanged.

**Validator obligations**:
- Layer 2 Check #21 (VI block contract conformance) — mandatory.
- Layer 2 Check #22 (VI optimizer = RAABBVI) — mandatory.
- The Layer-3 R2-VI PSIS-k̂ diagnostic (joint PSIS-k̂ < 0.7) — mandatory
  pre-merge.

**Reference tests** (`tests/`, non-shipped but illustrative — show the
hybrid composition + non-centered reparameterization pattern):

- `test_vi_glmm_logistic.cpp` — Hierarchical Bayesian logistic GLMM with
  random intercepts. VI on the J random effects (non-centered z_j);
  NUTS on β, α_0, log σ_α. The canonical applied-stats VI use case
  (multi-site / multi-school / multi-group binary outcomes).
- `test_vi_bnn_regression.cpp` — 1-hidden-layer Bayesian neural network
  for continuous regression. VI on the non-centered weights
  (α_0, Ã, β_0, β̃); NUTS on log σ_α, log σ_β, log σ_y. The
  canonical ML VI use case (high-dim weights with hierarchical scale
  funnel — Bishop 2006 §10.1, Welandawe 2022, see system_design §18).
- `test_vi_hybrid_composition.cpp` — Bayesian ridge regression with
  sampled coefficient-prior scale. The minimal hybrid demo.
- `test_vi_mean_field_gaussian.cpp`, `test_vi_correlated_gaussian.cpp`
  — pure-VI baselines on Gaussian targets (k̂ PASS / k̂ FAIL respectively,
  documenting Bishop §10.1.2 variance-underestimation caveat).
