<!-- system_design MODULE -- target geometry + memory (extracted 2026-06-21 from system_design.md Sec.11-Sec.12).
     SINGLE LIVE SOURCE: edit HERE, not the monolith. system_design.md is now a thin index
     (Sec.N -> module).
     Cross-refs keep the "Sec.N" scheme, resolved via the system_design.md index. -->

## 11. Target distribution geometry -- current architectural boundary

This is a **correctness invariant** about which kinds of posterior
distributions AI4BayesCode's current step mechanisms can sample. Every
system-design proposal MUST land on the "supported" side of this
boundary. The boundary is fundamental (rooted in MCMC theory); doing
the wrong thing produces a sampler that converges cleanly but to a
different distribution than the user wrote down -- a silent
correctness bug Layer 3 R3 usually cannot catch.

### 11.1 What the current step mechanisms can sample

AI4BayesCode supports these three target shapes:

1. **Fixed-dimension absolutely continuous** targets. Sampled by
   `nuts_block`, `joint_nuts_block`. The
   target has a smooth log-density w.r.t. Lebesgue measure on a
   fixed-dim $\mathbb{R}^d$ (after the constraint transforms).
2. **Fixed-dimension discrete with known finite support**. Sampled
   by `binary_gibbs_block`, `categorical_gibbs_block`,
   `dirichlet_gibbs_block`, `beta_gibbs_block`. Target is supported
   on $\{0,1\}^p$, $\{1,\dots,K\}^N$, or the continuous simplex /
   interval (still fixed-dim).
3. **Products / mixtures of the above** assembled via
   `composite_block`.

### 11.2 What the current step mechanisms CANNOT sample

Two classes of target fall outside the boundary; neither Gibbs nor
NUTS produces a correct sampler when confronted with them.

#### (a) Dimension-changing / trans-dimensional targets

Canonical example: **Dirac spike-and-slab** for variable selection:

$$\gamma_j \in \{0, 1\}, \quad \beta_j \mid \gamma_j = 0 \;=\; 0 \text{ (point mass)}, \quad \beta_j \mid \gamma_j = 1 \sim \mathcal{N}(0, \tau^2)$$

The joint state space for one coordinate is $\{(0,0)\} \cup \{1\}
\times \mathbb{R}$ -- a single point glued to a line. This is
**not a fixed-dim manifold**. Two naive approaches both fail:

- **Gibbs on hybrid space (`binary_gibbs_block` + "fix beta=0 when gamma=0")**
  is **not irreducible**. From $(\gamma,\beta)=(0,0)$, the Gibbs step
  for $\gamma$ computes
  $p(\gamma=1\mid\beta=0,\dots) \propto \pi \cdot N(0;0,\tau^2) \cdot p(y|0,\dots)$
  against
  $p(\gamma=0\mid\beta=0,\dots) \propto (1-\pi) \cdot \delta_0 \cdot p(y|0,\dots)$.
  The Dirac $\delta_0$ dominates the Gaussian-density-at-zero, so
  $p(\gamma=1)\to 0$. Symmetrically, from any $(\gamma=1,\beta\ne 0)$
  the move $\gamma=1\to 0$ has probability 0. Chain frozen in its
  starting component.

- **NUTS on the gamma-marginalized density** is **silently wrong**.
  Marginalizing gamma leaves
  $p(\beta_j) = (1-\pi)\,\delta_0(\beta_j) + \pi\,N(\beta_j;0,\tau^2)$,
  a **mixed Lebesgue + atomic measure**. HMC / NUTS requires a
  target that is absolutely continuous w.r.t. Lebesgue; it never
  evaluates the density at $\beta_j=0$ exactly (measure zero).
  NUTS therefore samples from the slab-only conditional $N(\beta_j;
  0,\tau^2)$, with $\pi$ absorbed into normalization. Chain
  converges, R-hat / ESS / LOO all look clean, but the posterior
  probability of inclusion is broken -- variable-selection inference
  is **silently hallucinated**.

Both failure modes extend to any target where a variable's support
depends on a discrete indicator (model-size priors over # mixture
components, change-point counts, basis-size selection, ...).

**Remedy:** Reversible-Jump MCMC with trans-dimensional proposals
(birth / death moves + accept ratio that accounts for the bijection's
Jacobian). **AI4BayesCode ships `rjmcmc_block` as of 2026-04-19
(`include/AI4BayesCode/rjmcmc_block.hpp`, reference template
`examples/SpikeSlabRJMCMC.cpp`).** The default identity-coordinate path supports
identity-coordinate proposals (the most common case, including
Dirac spike-and-slab variable selection -- Jacobian = 1 by
construction, so Sec.10.2's universal rule "users never write a
Jacobian" holds). Library 1D transforms (linear / affine ported
from librjmcmc) plus runtime autodiff on user-supplied
bijections) will extend coverage; see Sec.10.2 for the Jacobian
ownership story at each tier.

#### BNP mixture models: subclass with a Jacobian-free specialized path

An important subcase of dimension-changing targets is
**Bayesian-nonparametric mixture models** with unknown number of
components K (Dirichlet Process, Pitman-Yor, HDP, etc.). These
target shapes ARE dimension-changing, but the canonical algorithms
-- Neal's Algorithm 2 / 3 (conjugate, marginal), Algorithm 8
(non-conjugate, auxiliary variables), and Jain & Neal 2004
split-merge -- all sidestep the Jacobian problem because the
dimension change happens in **discrete allocation space** (which
clusters exist), while the continuous cluster-specific parameters
$\tau_h$ are either marginalized or sampled from conjugate
conditionals. The MH accept ratio is
$(\text{likelihood ratio}) \times (\text{EPPF ratio})$ -- no
continuous-parameter Jacobian.

**Shipped 2026-05-02 (truncated SBP path)**:
- `stick_breaking_block` -- generic Ishwaran-James 2001 truncated
  stick-breaking; covers DP / PY / custom Beta-stick processes via
  user-supplied `a_fn` / `b_fn`.
- `normal_gamma_cluster_gibbs_block` -- vectorized diagonal Normal-
  Gamma cluster sampler over K_trunc clusters (empty clusters draw
  from prior).
- `bnp_utils.hpp` -- header-only CRP / PY weight + log-prior helpers.
- `examples/DPGaussianMixture.cpp` / `examples/PYGaussianMixture.cpp` /
  `examples/DPGaussianMixture_DerivedAlpha.cpp` -- Tier-A composition
  templates.

**Future**: CRP-marginal Neal Alg 2/8, Jain-Neal 2004 split-
merge, full-covariance NIW cluster sampler, and Pitman-Yor with
sampled discount. The truncated SBP architecture mixes well on
moderate-N problems; these are optimisations rather than
correctness gates. Algorithm references: Neal 2000, Jain & Neal
2004, Pitman & Yor 1997, Ishwaran & James 2001; BayesMix (Beraha
et al. 2022, arXiv 2205.08144) has full C++ implementations.

#### (b) Discrete targets with strong local dependence

Canonical examples:
- **HMM** with latent state sequence $z_{1:T}$ Markov in $t$.
- **Ising / CRF / discrete MRF** with spatial dependence.
- **Bayesian network structure learning** with combinatorial
  adjacency matrix.

Per-site Gibbs on these targets IS irreducible, but mixing is
catastrophic -- the chain needs thousands of sweeps to propagate
information across strongly-coupled sites. A `categorical_gibbs_block`
over a $T$-length HMM converges in the limit but is useless in
practice.

**Remedy:** structured algorithms that exploit the dependence
structure (forward-backward / Kalman smoothing for HMM; Swendsen-
Wang clusters for Ising; order-MCMC / PC / GES for DAG structure).
Each needs a specialized block.
- `hmm_block` (T10, SHIPPED 2026-04-20) covers finite-state HMMs
  via exact forward-filter backward-sample; see
  `examples/HMMGaussian2State.cpp`.
- `ising_cluster_block` (SHIPPED 2026-05-31) covers Ising / Potts /
  general discrete MRFs on user-supplied undirected graphs via
  Swendsen-Wang 1987 cluster moves; see `examples/IsingPrior.cpp`.
  As of v1.2.1 the block also supports an external field $h \ne 0$
  (per-site $\alpha_i(k)$), per-edge $\beta_{ij}$, and partial decoupling
  $\delta_{ij}$ (Higdon 1998 Sec.2.3) -- all OPTIONAL (defaults reproduce the
  pure-prior Swendsen-Wang sampler). $\delta < 1$ is the mixing remedy for
  a strong external field.

  **Note: hidden Ising / hidden MRF ($h \ne 0$ from data likelihood).**
  The "per-site Gibbs is catastrophic" claim above applies to **pure**
  MRF / Ising at $\beta > \beta_c$ without observation likelihood.
  When a non-trivial observation likelihood $y_i \mid x_i$ provides an
  external field $h_i \ne 0$ (hidden image segmentation / GLM with
  spatial random effect / similar), the likelihood breaks the modes'
  symmetry and per-site Gibbs becomes sufficient -- provided the
  emission hyperparameters are sampled via conjugate Gibbs (NOT NUTS).
  The conjugate path is robust to slow $x$ mixing because each
  $(\mu, \sigma)$ draw is exact given current $x$; NUTS with an
  identifying ordering constraint (e.g. $\delta > 0$ to force $\mu_0 <
  \mu_1$) interacts badly with slow $x$ mixing and silently biases the
  posterior. For the library this means: route hidden-MRF-with-Normal-
  emission models the same way as finite-K Gaussian mixtures -- see
  `codegen_cpp.md` Sec.4a "Hidden discrete latent + Normal emission" row.

- `gmrf_precision_block` (SHIPPED 2026-05-31) covers Gaussian MRFs
  with sparse precision Q (Rue 2001 Sec.2 + Sec.3.1.2 + Sec.3.1.3 simplified
  single-constraint case) via Eigen SimplicialLLT + AMD reordering.
  Two Tier A demos:
  * `examples/GMRFPrior.cpp` -- pure-prior 2D ICAR demo
  * `examples/ICARSpatialGMRF.cpp` -- full Bayesian ICAR with
    Gaussian observations (hybrid: gmrf_precision_block for phi +
    3 separate nuts_block for Intercept / tau / sigma).
- `order_mcmc_block` (SHIPPED 2026-05-31) covers Bayesian-network
  structure learning over discrete data with BDeu scoring (n <= 64
  variables, FK Sec.4.2 three-tier cache) via Friedman-Koller 2003
  order MCMC + Heckerman-Geiger-Chickering 1995 BDe / BDeu score;
  see `examples/OrderMCMCBN.cpp` and:
  * `tests/audit_OrderMCMCBN.R` (30/30 PASS -- Layer-3 plateau,
    STRICT R-hat < 1.01 (Vehtari 2021), Sec.16 R-level pre-merge
    checklist including derived-key rejection / predict_at state
    preservation / round-trip identity).
  * `tests/test_order_mcmc_block_diagnostics.cpp` -- **exact
    posterior gold standard**: D1 n = 3 enumerates 25 DAGs and
    compares MCMC empirical edge marginals to the closed-form
    posterior (HARD max |Delta| < 0.05, achieves ~= 0.001). D2 n = 4
    enumerates 543 DAGs (achieves ~= 0.009). D3 conditional
    P(Pa_i | order, D). D4 ESS > 200 on 10000 samples.
  * `tests/test_order_mcmc_block_stress.cpp` -- 9 stress tests
    including 4-chain R-hat < 1.01 strict on a unimodal target.
  * `tests/audit_OrderMCMCBN_bnlearn_cross.R` (ASIA cross-check
    vs `bnlearn::hc`: 7/7 perfect skeleton agreement, 7/8 true
    Markov-equivalent edges in top-8 inclusion).
  * `tests/audit_OrderMCMCBN_vs_BiDAG.R` -- **reference-
    implementation comparison** against `BiDAG::orderMCMC`
    (Kuipers-Moffa), 4 matched chains: both implementations
    achieve R-hat < 1.01 (ours 1.00073, BiDAG 1.00000) -> our
    code converges to the same target as the reference. v1.2 scope: discrete
  data + BDeu only; documented FK Sec.4.1 induced-structure-prior
  bias within Markov equivalence classes (the algorithm finds the
  correct **skeleton** but may flip directions within an
  equivalence class). Kuipers-Moffa 2017 partition MCMC and BGe
  Gaussian score SHIPPED 2026-07-06 (`cfg.method = partition`,
  `cfg.continuous_data`; default byte-identical to order + BDe;
  exact-enum n<=8 + BiDAG cross-check verified). Mixed
  conditional-Gaussian BN remains deferred. Tempered chains are
  out of scope by design principle (target-changing kernels break
  the compositional invariant -- every kernel must target the
  same joint).

### 11.3 The validator cannot catch geometry violations

Layer 2 checks (constraint discipline, parameterization, etc.) look
at the code structure. Layer 3 runtime checks (R-hat, ESS, Bayesian
p-value, LOO) all pass for these silent-wrong-answer cases because
the sampler IS converging to some stationary distribution -- just
not the one the user wrote down. **Geometry violations must be
prevented at design time.** That is why this is a system-design-
skill rule, not a validator check.

### 11.4 Rule for system designers

When designing a new block or a new example:

1. **State the target distribution explicitly** in the design doc
   (prior, likelihood, parameter supports).
2. **Check the support of every variable.** If any variable's
   support depends on the value of a discrete indicator (i.e. the
   state space is not a fixed-dim manifold), you are in Sec.11.2(a).
   Do NOT ship a Gibbs-only or NUTS-only sampler. Choose one of:
   - redefine the prior to avoid the dimension change (continuous
     relaxation with tight Gaussian spike);
   - use `rjmcmc_block` (identity-coordinate proposals, covers
     variable selection + most change-point / basis-birth cases);
     see reference template `SpikeSlabRJMCMC.cpp`;
   - for BNP mixture-K subcase, use the truncated SBP infrastructure
     (`stick_breaking_block` + `normal_gamma_cluster_gibbs_block` +
     `categorical_gibbs_block`; reference templates
     `examples/DPGaussianMixture.cpp` / `examples/PYGaussianMixture.cpp`).
     Future Neal Alg 2/8 marginal CRP and Jain-Neal split-merge are
     optimisations, not correctness gates.
3. **Check for strong local dependence** among discrete variables.
   If yes, you are in Sec.11.2(b). Do NOT assume per-site Gibbs is
   enough. Either:
   - reformulate the model to remove the dependence, or
   - implement a specialized block (HMM forward-backward, etc.).
4. **Document the target class in the new block's header comment.**
   Every block header should state which of Sec.11.1's three shapes
   it samples. If the shape is exotic, explain why the block is
   nevertheless correct.

### 11.5 Code-gen agent's operational rule

Code-generating agents must NOT need to understand measure theory.
`skills/codegen_priors.md Sec.3a` carries the decision tree (Class 1-5)
and `skills/codegen.md Sec.12` carries a Hard Rule pointing here. The
code-gen agent's only obligation is: if a user
model description implies Sec.11.2(a) or Sec.11.2(b), the agent falls
back to a supported alternative (Class 2a continuous spike, or
horseshoe, or explain "not supported yet") and points the user
here for the theory.

### 11.6 Variance / scale prior discipline (Gelman 2006)

**The default prior on every scale parameter sigma > 0 is the
scale-invariant Jeffreys prior p(sigma) prop.to 1/sigma,** implemented via
`nuts_block` + `constraints::positive::wrap` on log(sigma). On the
unconstrained eta=log(sigma) scale, Jeffreys + Jacobian exactly
cancel, leaving the user's log-density to contain only the
likelihood (with the likelihood's natural -N log(sigma) normalization
factor intact -- that is NOT the prior and does NOT cancel). See
`examples/SpikeSlabRJMCMC.cpp` sigma block for the reference.

**PROHIBITED as a default "noninformative":** InverseGamma(epsilon, epsilon)
with small epsilon. Gelman (2006) Bayesian Analysis 1(3):515-533 Sec.3
shows this prior has highly unstable behavior as epsilon -> 0 and posterior
inference is sensitive to epsilon. No new example or block may use
IG(epsilon, epsilon) as its default variance prior. `inv_gamma_gibbs_block`
is therefore DISCOURAGED as a default and carries a large warning
in `skills/block_catalogue/index.md`.

**Escape when Jeffreys gives improper posterior (k=0 transient):**

The preferred pattern is **Jeffreys with inline k=0 fallback**:
keep `p(param) prop.to 1/param` as the main prior; inside the natural-
scale log-density, detect when the effective count k=0 and fall
back to a half-Normal(0, 1) density just for that sweep. The
fallback softly pins the parameter near the reference scale 1 until
k >= 1 is restored. Pair with an init-value guard (e.g., marginal-
OLS screening for spike-slab) so the chain starts with k >= 1.

This is implemented in `examples/SpikeSlabRJMCMC.cpp` tau block --
see `tau_natural_log_density` for the pattern. The design keeps
Jeffreys as the principled default for well-fed posteriors AND
contains the degenerate k=0 state with minimal intervention.

For genuinely-sparse-info models (group-level variance with few
groups, k remaining small throughout), drop to Gelman 2006's
half-Normal(0, A) with A ~= 2.5 x data-scale as a persistent proper
prior. Do NOT use half-Cauchy -- its polynomial tail is slower for
NUTS without providing additional robustness in practice.

**INFORMATIVE IG prior** is permitted when the hyperparameters are
derived from external knowledge (e.g., BART's calibrated IG using
`sigest`, where the prior is deliberately informative to resolve an
identifiability issue in `y = BART(X) + N(0, sigma^2)`). Document
the source.

System designers adding new example types: default to Jeffreys on
sigma with inline k=0 half-Normal(0, 1) fallback pattern; for
genuinely-sparse-info group-level variances, half-Normal(0, A) with
A ~= 2.5 x data-scale. Do NOT reach for IG unless you can justify
it in writing. See `skills/codegen_priors.md Sec.2a` for the precise recipe.

### 11.7 Gibbs-block discipline (Checks #15, #16, #17)

Every example that uses a `*_gibbs_block` or a hand-written Gibbs
sampler inside an `rjmcmc_block` hook incurs three validator
obligations:

- **Check #15** -- library-level OR per-usage parity test present
  (see `skills/codegen_priors.md` Sec.2c). Library-level test
  `tests_autodiff/block_tests/test_<blockname>_gibbs_block.cpp`
  covers the sampling mechanism for textbook `params_fn`; per-usage
  test `tests_autodiff/test_<model>_<...>_gibbs_parity.cpp` covers
  non-textbook `params_fn` (e.g., active-subset indexing) or hand-
  written Gibbs in rjmcmc hooks. Tolerance 5% mean / 10% variance
  on 10k draws.

- **Check #16** -- inline "JUSTIFICATION (Check #16): Exception N
  from codegen_priors.md Sec.2b -- ..." comment above every gibbs usage stating
  which Exception 1/2/3 applies. Validator static grep flags missing
  comments.

- **Check #17** -- no hand-written distribution samplers
  (`std::gamma_distribution`, `Rcpp::rbeta`, etc.) in example code
  outside a narrow whitelist (rjmcmc hooks, stochastic refreshers,
  library internals). Validator static grep flags non-whitelisted
  inline sampler usage.

Why these checks exist: conjugate-Gibbs derivation errors are the
highest-risk silent-correctness bug class in AI4BayesCode. An AI agent
writing a wrong `params_fn` formula produces a sampler that
converges (R-hat, ESS, LOO all pass) but to the wrong posterior.
Checks #15-17 form the retrospective safety net.

---

## 12. Memory safety

Leaves (child blocks) NEVER store raw pointers to `shared_data_t`.
The composite passes `block_context& ctx` by reference for each
sweep's duration. A leaf that caches `&data_` will dangle if the
composite is ever moved / copied.

Vectors returned from shared_data via `ctx.at(key)` are
`const arma::vec&` references into the composite's storage. Do NOT
keep references longer than the current sweep. Copy if you need to
survive.

### RCPP_MODULE ownership

`new(ClassName, ...)` in R gives the user a reference-counted
handle. The wrapper's `impl_` (composite_block owning children)
lives for the handle's lifetime. Do NOT introduce shared_ptr
cycles between the wrapper and its children.

---

