## vi_optimizer.hpp (helper utilities, NOT a block)

Header-only port of Welandawe et al. 2022 RAABBVI. Exposes:

- `vi_optimizer::raabbvi_config` — POD struct with `gamma_0` (initial
  learning rate, default 0.1), `rho` (geometric decay between epochs,
  default 0.5), `tau` (SKL termination threshold, default 0.1),
  `W_min` (minimum R̂-convergence window, default 100), `max_epochs`
  (cap on outer loops, default 50), `S_khat` (samples for terminal
  k̂ computation, default 1000).
- `vi_optimizer::raabbvi<GradFn>` — templated struct taking a
  gradient functor and orchestrating avgAdam + Polyak-Ruppert
  iterate averaging + R̂ convergence at fixed γ + SKL termination.
- `vi_optimizer::psis_khat(log_weights) → double` — Pareto-smoothed
  importance-sampling k̂ computation; used by VI blocks at SKL
  termination and by `psis_diagnostic.hpp` for Layer-3 R2-VI.

Users do NOT construct `vi_optimizer::raabbvi` directly; configure
via the VI block's `cfg.optimizer` field.

See `system_design.md §18.7` for full algorithm details.
