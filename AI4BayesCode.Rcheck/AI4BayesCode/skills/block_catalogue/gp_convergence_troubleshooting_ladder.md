## GP convergence troubleshooting ladder

This ladder applies to GP samplers that retain an explicit latent
(non-Gaussian likelihood — `GPClassification.cpp` and its
adaptations). For Gaussian likelihoods, use the marginal architecture
in `GPRegression.cpp` directly; it does not have these mixing pitfalls
because f is integrated out.

The shipped `GPClassification.cpp` already applies the two most
important fixes in this ladder — **whitening** (step 4 below) **and
joint NUTS over `(amp, ell)`** (step 3) — by default. The remaining
steps are escalations on top of those when posterior recovery / ESS
is still poor.

When hyperparameter R-hat fails or ESS is too low on a whitened-ESS
GP sampler, escalate:

1. **Tighten the lengthscale prior.** Half-Normal(0, scale_x/4) or an
   InverseGamma fit to the data's distance-quantile range localises
   the posterior away from extreme regions. (Stan's GP case study
   uses InverseGamma(5, 5) for unit-scale `x`.)

2. **Reparameterise to log-scale.** Sample `log_amp`, `log_ell` as
   REAL via the joint block (or fall back to `nuts_block` per slice),
   with the +log(amp)+log(ell) Jacobian added to the log-density.
   Removes boundary-related step-size collapse.

3. **Joint NUTS over `(amp, ell)` — DEFAULT in `GPClassification.cpp`.**
   The `(amp, ell)` banana ridge requires JOINT trajectory steps —
   modular NUTS on each parameter slow-mixes along the ridge,
   typically dropping `amp` ESS into the dozens at long chains. The
   shipped example uses one `joint_nuts_block({amp, ell})`
   (POSITIVE × 2) so the two hyperparameters are sampled together
   under a shared mass matrix. If you split them back into separate
   `nuts_block`s for any reason, expect 5–10× ESS loss on `amp`.

4. **Whitening (Murray & Adams 2010).** Already applied in
   `GPClassification.cpp`. Replaces explicit `f` with `z ~ N(0, I)`
   via `f = L · z`, so z's prior does NOT depend on `(amp, ell)`.
   The hyperparameter conditional sees the data only via the
   likelihood (Bernoulli-logit, Poisson, …), avoiding the
   `(amp ≈ 0, f ≈ 0)` collapse mode of the centered parameterisation.

5. **Switch the joint log-density to an analytic gradient.** The
   shipped example uses finite differences (4 K-builds + chols per
   gradient call). Reverse-mode Cholesky AD (Murray 2016) gives a
   noise-free gradient at O(N³) one-shot + O(N²) per θ-component;
   typically lifts `ell` ESS from the dozens to the hundreds. Worth
   it when the whitened-ESS bulk autocorrelation on `ell` is the
   bottleneck.

6. **Marginalise the latent if you can.** If the likelihood happens
   to be Gaussian (or any likelihood that admits a closed-form latent
   integral), drop the ESS block entirely and sample only the
   hyperparameters from the marginal posterior. This is the route
   taken by `GPRegression.cpp` for Gaussian observations.

**References**: Rasmussen & Williams 2006 §5 (marginal-likelihood
gradient formula `0.5 tr((α α' − K⁻¹) ∂K/∂θ)`, used in
`GPRegression.cpp`); Murray & Adams 2010 *Slice Sampling Covariance
Hyperparameters of Latent Gaussian Models* (whitening, used in
`GPClassification.cpp`); Betancourt 2017 *Robust Gaussian Processes
in Stan* (priors + reparameterisations).
