## probit_aug_block

**Reference example:** `examples/ProbitRegression.cpp` (Bayesian probit
linear regression via Albert-Chib + NUTS-on-beta). The same
Albert-Chib augmentation pattern composes naturally with `bart_block`
when `cfg.binary = true` (probit BART), but no shipped example
combines them — see `bart_block`'s "Binary mode" docstring for the
recipe.

Closed-form Gibbs leaf for the **Albert-Chib (1993)** data-augmentation
latent z in any probit-link binary likelihood:

```
y_i        ~ Bernoulli(p_i),  p_i = Phi(mu_i + offset_i)
z_i | rest ~ N(mu_i + offset_i, 1) truncated to:
                 (0, +inf)   if y_i = 1
                 (-inf, 0)   if y_i = 0
```

Compose with any Gaussian-likelihood downstream block (the downstream
block sees `z` — or `z - offset` if you bake the offset out — as its
working response). Standard pattern:

```
composite "ProbitWhatever":
  child(0) z      probit_aug_block (this block)
  child(1) <mean> nuts_block / bart_block (binary=true) /
                   linear_block / GP / hierarchical / ...
```

**Why it exists.** Albert-Chib augmentation is *the* textbook closed-
form Gibbs step for probit. Before this block shipped, probit examples
had to inline the truncated-normal step in the wrapper's `step()`
method, breaking uniformity. `probit_aug_block` makes the Gibbs leaf
library-blessed (Exception 3 of `codegen_priors.md` §2b — closed-form
vector conjugate sample, NUTS-wasteful for N independent truncated
normals).

**Sigma is FIXED at 1.0** by probit identifiability. There is no config
knob for it. Anyone wanting `sigma != 1` is targeting a different
likelihood (Tobit / censored Gaussian) and should use a different
block.

```cpp
probit_aug_block_config cfg;
cfg.name        = "z";          // shared_data key for output z (length N)
cfg.n_obs       = N;
cfg.y_key       = "y";          // length-N {0, 1}
cfg.mu_key      = "mu_lin";     // length-N linear predictor (Gaussian
                                //   block writes this — bart_block's
                                //   "f_bart", or a refresher computing
                                //   X*beta, etc.)
cfg.offset_key  = "";           // optional; "" = no offset; else scalar
                                //   (length 1) or per-obs (length N)
cfg.initial_z   = arma::vec();  // optional warm start; default snaps to
                                //   sign(2 y - 1)
```

**Conditional independence (no sequential update needed).** Each `z_i`
depends only on `(y_i, mu_i, offset_i)`. The block draws all `z_i`'s
in a single vectorised pass per `step()` — no inner loop dependency,
no need to refresh the context mid-sweep. (Contrast with
`binary_gibbs_block` where sequential update IS required.)

**Composition with bart_block for probit BART.** Set `cfg.binary = true`
on the sibling `bart_block` so its leaf-prior tau matches BART::pbart's
`3 / (k * sqrt(ntrees))` formula. Without that, `bart_block` falls
back to the Gaussian formula based on the working-response range and
over-shrinks the leaves by a factor of 3. See `bart_block_config::binary`
in this catalogue.

**Check #15 parity test:**
`tests_autodiff/block_tests/test_probit_aug_block.cpp` — verifies the
empirical mean of drawn z's against the closed-form `TN(mu, 1, [a, b])`
expectation across 5 regimes (centred / shifted positive / shifted
negative / mixed-y vector with scalar offset / per-obs offset
broadcast) at 5% relative tolerance, 10000 draws each.
