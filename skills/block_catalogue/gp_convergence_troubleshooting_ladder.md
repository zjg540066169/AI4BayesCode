## GP convergence troubleshooting ladder

This ladder applies to GP samplers that retain an explicit latent
(non-Gaussian likelihood -- `GPClassification.cpp` and its
adaptations). For Gaussian likelihoods, use the marginal architecture
in `GPRegression.cpp` directly; it does not have these mixing pitfalls
because f is integrated out.

The shipped `GPClassification.cpp` already applies steps 3-5 below --
**whitening**, **one joint block over `(amp, ell, z)`**, and an
**analytic gradient** -- by default. The remaining steps are
escalations on top of those when posterior recovery / ESS is still
poor.

When hyperparameter R-hat fails or ESS is too low on a whitened GP
sampler, escalate:

1. **Tighten the lengthscale prior.** Half-Normal(0, scale_x/4) or an
   InverseGamma fit to the data's distance-quantile range localises
   the posterior away from extreme regions. (Stan's GP case study
   uses InverseGamma(5, 5) for unit-scale `x`.)

2. **Reparameterise to log-scale.** A `POSITIVE` sub-param in
   `joint_nuts_block` already does this -- it samples `log(theta)` and
   adds the Jacobian internally. Only reach for a hand-rolled REAL
   slice plus a manual `+log(theta)` if you need an unusual transform.
   Removes boundary-related step-size collapse.

3. **Whitening (Murray & Adams 2010) -- DEFAULT in
   `GPClassification.cpp`.** Replace explicit `f` with `z ~ N(0, I)`
   via `f = L * z`, so z's prior does NOT depend on `(amp, ell)`. The
   hyperparameters then see the data only via the likelihood,
   avoiding the `(amp ~= 0, f ~= 0)` collapse mode of the centered
   parameterisation.

4. **Put `(amp, ell, z)` in ONE `joint_nuts_block` -- DEFAULT in
   `GPClassification.cpp`.** This is the step that actually makes it
   converge, and it is easy to get wrong: whitening plus a Gibbs
   alternation (hyperparameter block, then
   `elliptical_slice_sampling_block` on z) is still slow, because with
   z held fixed `f = L(theta) z` is a deterministic function of theta,
   so `p(theta | z, y)` is far sharper than `p(theta | y)`. Measured on
   `GPClassification.cpp`'s own dataset, 4 chains x (1000 burn + 2000
   keep), worst cross-chain rank R-hat / bulk ESS:

   | sampler | amp | ell | f |
   |---|---|---|---|
   | ESS-Gibbs, diagonal metric | 1.623 / 7 | 1.179 / 41 | 1.279 / 11 |
   | ESS-Gibbs, dense metric | 1.102 / 28 | 1.159 / 17 | 1.067 / 49 |
   | ONE joint block, diagonal metric | 1.001 / 1937 | 1.002 / 1539 | 1.003 / 3074 |

   Use a diagonal metric: the block is `N + 2` dimensional, so a dense
   metric would mean estimating and inverting an `(N+2) x (N+2)`
   covariance. The `(amp, ell)` banana ridge is handled by the two
   being in the same trajectory, not by the metric.

5. **Analytic gradient, and a MULTIPLICATIVE jitter -- DEFAULT in
   `GPClassification.cpp`.** Write `K = amp^2 (R + eps I)`, not
   `amp^2 R + eps I`. Then `L = amp * L_R` holds exactly, so
   `df/d amp = f / amp` and

       d/d amp = (r . f) / amp + d log p(amp)/d amp,   r = d loglik / d f

   is O(N) with no matrix derivative at all. Only the lengthscale
   needs the Cholesky derivative (Murray 2016)

       dL = L Phi(L^-1 (dK/d ell) L^-T),   Phi(A) = tril(A), diagonal halved
       d/d ell = r' L Phi(...) z + d log p(ell)/d ell

   with `dK/d ell = amp^2 R .* r_ij^2 / ell^3` on the UN-jittered R.
   Finite differences here cost 4 K-builds + Cholesky factorisations
   per gradient call AND inject noise into the bulk autocorrelation;
   the analytic form is both cheaper and quieter.

6. **Guard the Cholesky.** Reject (`return -inf`) when
   `chol` fails or `min(diag(L_R)) < 1e-8`. A near-singular K at a
   long proposed lengthscale otherwise produces a finite-but-garbage
   log-density that the sampler will happily wander into. `-inf` is a
   valid NUTS reject.

7. **Start z from its prior, not from zero.** At `z = 0` the latent is
   `f = L z = 0` identically, so the likelihood -- and with it the
   hyperparameters' entire share of the gradient -- vanishes at the
   initial point, and the block's first-call warmup tunes its step
   size and metric on the prior geometry alone. Observed effect: at
   N = 120 the hyperparameters froze completely (posterior SD exactly
   0.0) for the whole run. Draw `z_init ~ N(0, I)` from a seeded RNG.

8. **Marginalise the latent if you can.** If the likelihood happens
   to be Gaussian (or any likelihood that admits a closed-form latent
   integral), drop the latent entirely and sample only the
   hyperparameters from the marginal posterior. This is the route
   taken by `GPRegression.cpp` for Gaussian observations.

**References**: Rasmussen & Williams 2006 Sec.5 (marginal-likelihood
gradient formula `0.5 tr((alpha alpha' - K^-1) dK/dtheta)`, used in
`GPRegression.cpp`); Murray & Adams 2010 *Slice Sampling Covariance
Hyperparameters of Latent Gaussian Models* (whitening, used in
`GPClassification.cpp`); Murray 2016 *Differentiation of the Cholesky
decomposition*; Betancourt 2017 *Robust Gaussian Processes in Stan*
(priors + reparameterisations).
