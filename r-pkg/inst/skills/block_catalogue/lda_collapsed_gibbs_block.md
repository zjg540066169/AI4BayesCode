## lda_collapsed_gibbs_block

**Reference example:** `examples/LdaCollapsedGibbs.cpp` (shipped 2026-05-08).
Use this block for any LDA-style model: token-level latent topic
assignment `z_n  in  {1..K}` paired with per-document Dirichlet topic
proportions `theta_d` and per-topic Dirichlet word distributions `phi_k`,
with FIXED-K and FIXED Dirichlet hyperparameters `alpha, beta`. Replaces
the naive composition `categorical_gibbs_block(z) + dirichlet_gibbs_block(theta_d) x M + dirichlet_gibbs_block(phi_k) x K`,
which is correct in the limit but mixes catastrophically because of
the strong cross-block coupling between (z, theta, phi) -- this is the
canonical Sec.11.2(b) "discrete target with strong local dependence"
case where a specialized algorithm is mandatory.

The block samples z via the Griffiths & Steyvers (2004) collapsed
Gibbs update with (theta, phi) integrated out:

  P(z_n = k | z_{-n}, w, alpha, beta)
       prop.to (n_{d,k}^{-n} + alpha_k)
         x (n_{k,w_n}^{-n} + beta_{w_n})
         / (n_{k,*}^{-n} + sum(beta))

After the z sweep, theta and phi are sampled directly from their Dirichlet
conjugate posteriors via gamma-normalization (same mechanism as
`dirichlet_gibbs_block`). The block exposes all three under
`current_named_outputs()` so the composite writes z, theta, phi to
shared_data after each sweep.

**Output layout** (column-major flat, recover via `matrix(., M, K)`
or `matrix(., K, V)` in R):
- `z`     : length-N integer vector of topic assignments (1-indexed)
- `theta` : length-(M*K), entry [d + k*M] = theta_{d,k}
- `phi`   : length-(K*V), entry [k + v*K] = phi_{k,v}

**Label switching**: this block does NOT canonicalize topics
internally (system_design.md guarantees deterministic block output
for given seed + init). Apply Stephens 2000 in R-level alignment
code per `skills/label_switching.md` if the reference implementation
also has unconstrained topics.

**JUSTIFICATION (Check #16):** Exception 1 from `codegen_priors.md` Sec.2b
-- z is discrete; NUTS cannot target it. Library Check #15 parity
test is at
`tests_autodiff/block_tests/test_lda_collapsed_gibbs_block.cpp`
(5 regimes: count integrity, theta conjugate parity, phi conjugate
parity, cross-impl parity vs in-test reference via label-invariant
statistics, recovery from synthetic truth at K=2 with Hungarian
match).

```cpp
lda_collapsed_gibbs_block_config cfg;
cfg.name           = "lda";
cfg.M              = M;          // number of documents
cfg.V              = V;          // vocabulary size
cfg.K              = K;          // number of topics (FIXED)
cfg.alpha          = arma::vec(K, arma::fill::ones);  // default Dir(1,..,1)
cfg.beta           = arma::vec(V, arma::fill::ones);
cfg.w_key          = "w";        // ctx key for length-N word ids (1..V)
cfg.doc_key        = "doc";      // ctx key for length-N doc ids (1..M)
cfg.z_out_key      = "z";        // output keys for shared_data
cfg.theta_out_key  = "theta";
cfg.phi_out_key    = "phi";
// cfg.z_init left empty -> deterministic (i mod K) + 1 init.
impl_->add_child(
    std::make_unique<lda_collapsed_gibbs_block>(std::move(cfg)));
```

The Tier-A wrapper exposes only the canonical six R-methods. The
block's name (`"lda"`) is a label for Gibbs DAG bookkeeping; it is
NOT a shared_data key (the joint-block pattern from
system_design.md Sec.3 / Sec.13). Downstream consumers read z, theta,
phi via the OUTPUT keys above.
