---
name: particle_gibbs_block
description: Particle Gibbs (conditional SMC, optional ancestor sampling) for the latent path of a nonlinear/non-Gaussian state-space model; gradient-free; samples x_{1:T} | y, theta with theta read from context.
---

# particle_gibbs_block

A gradient-free sampler for the **latent state path** x_{1:T} of a general
state-space model, via **particle Gibbs / conditional SMC** (Andrieu, Doucet &
Holenstein 2010) with optional **ancestor sampling** (PGAS; Lindsten, Jordan &
Schön 2014). The model is supplied as four plug-ins (initial / transition /
observation densities), so one block serves stochastic volatility, nonlinear
dynamic, and state-space count models. It samples the path GIVEN the parameters
θ; θ is read from context each sweep (a sibling block samples it, or it is
fixed) — the path-update half of a full particle Gibbs scheme.

## (a) Routing row

```
| **latent path x_{1:T} of a nonlinear/non-Gaussian state-space model (continuous states, strong sequential dependence)** | **`particle_gibbs_block`** (conditional SMC + PGAS; Andrieu et al. 2010, Lindsten et al. 2014) | **(none — gradient-free; samples in the natural state space; no constraint transform)** |
```

## (b) WHEN to use

Nonlinear state-space time series, continuous latent path (e.g. stochastic
volatility). Use it when the observation or transition density makes exact
filtering impossible (no Kalman / forward–backward) **or** the densities are
non-differentiable / intractable so a gradient sampler does not apply — the
kernel needs only to SAMPLE the transition and EVALUATE the observation
density.

## (c) WHEN NOT to use

- **Discrete latent state sequence** (finite-state HMM z_{1:T}) → use
  `hmm_block` (exact forward-filter backward-sample), not this. This block is
  for CONTINUOUS states.
- **Linear-Gaussian SSM** where you want the path → the exact RTS/Kalman
  smoother (or a Gaussian-conjugate draw) is exact and cheaper; this block
  still works but is overkill.
- **Gaussian-Markov random FIELD** (spatial precision, not a sequence) →
  `gmrf_precision_block` (Gaussian conditional) or `gmrf_whitened_ess_block`
  (non-Gaussian likelihood). Those exploit a sparse precision, not a forward
  filter.
- **Smooth, differentiable, fixed-dim continuous target** with no
  sequential/intractable structure → `joint_nuts_block` (gradient-based, the
  default) is usually better.
- **Sampling the SSM parameters θ = (μ, φ, σ, …)** → NOT this block. It samples
  the PATH given θ; θ is sampled by sibling blocks (conjugate-Gibbs / NUTS /
  MH) reading the path.

## (d) Geometry class

Target is **§11.1 shape 1 — fixed-dim absolutely continuous** on
ℝ^(T·state_dim) (T fixed; each x_t ∈ ℝ^state_dim). Not trans-dimensional, not
discrete-with-local-dependence. The custom kernel is legitimate as a
**structured algorithm exploiting the sequential Markov dependence** (Exception
4) — the continuous, non-Gaussian generalization of HMM forward–backward. See
`system_design §11` (`geometry.md`) for the legality gate; the cSMC kernel is
invariant for any N ≥ 2 (ADH 2010, Thm 5).

## (e) Config-struct snippet

```cpp
particle_gibbs_block_config cfg;
cfg.name          = "x";        // output key: the sampled path lives here
cfg.T             = T;          // number of time steps
cfg.state_dim     = 1;          // dim of each x_t (1 = scalar; >1 = vector state)
cfg.n_particles   = 64;         // N >= 2; mixing/cost knob (NOT correctness)
cfg.ancestor_sampling = true;   // PGAS on (default); breaks path degeneracy
cfg.resampling    = pg_resampling_scheme::SYSTEMATIC;   // or STRATIFIED / MULTINOMIAL
cfg.initial_path  = {};         // optional warm start (len T*state_dim); else
                                //   first sweep seeds via an unconditional PF
// model-specific plug-ins (read theta & y_t from ctx themselves):
cfg.init_sample       = [](const block_context& c, std::mt19937_64& g){ ... };          // x_1 ~ mu_theta
cfg.transition_sample = [](int t, const arma::vec& xp, const block_context& c, std::mt19937_64& g){ ... }; // x_t ~ f_theta
cfg.obs_loglik        = [](int t, const arma::vec& x, const block_context& c){ ... };   // log g_theta(y_t|x_t)
cfg.transition_logpdf = [](int t, const arma::vec& x, const arma::vec& xp, const block_context& c){ ... }; // log f_theta (PGAS only)
// Typical composite: this block ("x") + sibling blocks (or inline updates) that
// sample theta = (mu, phi, sigma, ...) from the current path. Declare the
// block's context deps: data().declare_dependencies("x", {theta keys..., "y"}).
```

Block-specific silent-failure checks (`BL1`–`BL6`): see the `ValidationSkill`
(`skills/particle_gibbs_block_validation.md`) — needed only at validation/audit,
not on every use.

## (f) Example

`examples/PoissonStateSpace.cpp` — a frontend-independent C++ demo: a Poisson
count SSM (y_t ~ Poisson(exp x_t), x_t an AR(1) log-rate) fit by the full
particle-Gibbs scheme (this block for the path + inline conjugate/MH updates for
θ), recovering the latent path (beating a naive per-time baseline) and the
parameters, with a posterior-predictive check.

## Shared conventions (cited, not restated)

- interface contract (6 R methods + readapt_NUTS): system_design §1 (interface.md)
- block_sampler / composite / shared_data semantics: system_design §1–§9 (interface.md / dataflow.md)
- constraint transforms / the 15 joint_constraint kinds: constraints.md (this block uses NONE)
- block-selection / Exception taxonomy (Exception 4 = AI-authored custom block): codegen_priors.md §2b
- metric + warmup policy: system_design §13 (families.md) — N/A here (gradient-free, no metric/warmup)
- validator checks this block faces: validator.md #17 (Exception-4 justification); BL1–BL6 in the ValidationSkill
- geometry legality gate: system_design §11 (geometry.md)
