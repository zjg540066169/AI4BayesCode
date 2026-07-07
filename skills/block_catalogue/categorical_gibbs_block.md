## categorical_gibbs_block

**Reference example:** NONE shipped -- library-only. A finite-mixture or
LDA example using `categorical_gibbs_block` + `dirichlet_gibbs_block`
is on the future roadmap (todo T7-T9: DP mixture via Neal Alg 2/3/8).
For now, codegen agents targeting Class 1 mixture / LCA / LDA should
compose this block inline following the header-comment pattern in
`categorical_gibbs_block.hpp`.

Per-observation closed-form Gibbs for z in {1..K}. Provide
`log_probs_fn(ctx) -> arma::mat(N, K)` of unnormalized log-probs.
Use for mixture labels, LDA assignments.

**Target-geometry caveat:** valid only when the {z_i} are
**conditionally independent** across i given the continuous
parameters (the standard Class 1 assumption -- mixture models, LCA,
ZIP). It is **NOT** valid for Markov-structured latents (HMM
z_{1:T}, change-point) -- per-site Gibbs mixes catastrophically
slowly on those. See `skills/system_design.md` Sec.11 and
`skills/codegen_priors.md` Sec.3a. HMM / state-space support is provided by
`hmm_block` (T10, SHIPPED 2026-04-20) via forward-filter
backward-sample; see the hmm_block entry below.

```cpp
categorical_gibbs_block_config cfg;
cfg.name         = "z";
cfg.n_obs        = N;
cfg.n_categories = K;
cfg.initial_labels = arma::vec(N, arma::fill::ones);
cfg.log_probs_fn = [](const block_context& ctx) -> arma::mat { ... };
```
