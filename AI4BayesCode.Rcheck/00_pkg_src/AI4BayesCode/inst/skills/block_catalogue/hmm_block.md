## hmm_block (T10, forward-filter backward-sample)

Exact FFBS sampler for finite-state Hidden Markov Models. Samples
the latent state sequence z_1:T jointly from its full conditional
p(z_1:T | y, A, pi, theta) via log-space forward filter + backward
sampling. K and T fixed per construction; A (K*K row-major), pi
(length K), and emission log-density are read from context each
sweep (typically sampled by sibling `dirichlet_gibbs_block` for A
and pi, and `nuts_block` / `beta_gibbs_block` for emission theta).

**JUSTIFICATION (Check #16): Exception 1** (discrete state
sequence; NUTS cannot target). Algorithm is Baum-Welch / Fruhwirth-
Schnatter 2006 Ch. 11 standard FFBS. Check #15 parity test at
`tests_autodiff/test_hmm_block.cpp` verifies empirical marginals
P(z_t = k | y) against analytical Baum-Welch smoothing to
max_abs_err < 0.2% (10k draws on K=2, T=5 fixture).

Reference template: `examples/HMMGaussian2State.cpp` (minimal 2-
state Gaussian-emission demo with fixed A, pi, emission params).

```cpp
hmm_block_config cfg;
cfg.name = "z";
cfg.T    = T; cfg.K = K;
cfg.A_key = "A"; cfg.pi_key = "pi";
cfg.emission_logp =
    [](std::size_t t, std::size_t k, const block_context& ctx) {
        const arma::vec& y = ctx.at("y");
        // log p(y_t | z_t = k, theta)
        return ...;
    };
cfg.initial_z = arma::vec(T, arma::fill::zeros);
```
