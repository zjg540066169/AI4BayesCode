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

**Oracle calls and the block's own labels.** If `log_probs_fn` uses the
block's own labels (count-dependent or CRP conditionals, Potts couplings),
list the block's own name in its dependency list, e.g.
`impl_->data().declare_dependencies("z", {"z", ...})`; `step()` then calls
the oracle once per observation, each time after writing the previous label
into that context entry. Otherwise the context has no entry under the block's
name and the oracle is called once per `step()` for all rows, so it must not
keep per-call state (a call counter, a test for its own key). When the own
key is in the context, each step starts from its shared value: labels written
there by another block (`split_merge_block`, `stick_label_swap_block`) are
the starting state of the sweep, so a restore of this block must also write
the shared key (`data().set`) or go through `composite_block::set_current`,
and an entry whose length is not `n_obs` makes `step()` throw.
