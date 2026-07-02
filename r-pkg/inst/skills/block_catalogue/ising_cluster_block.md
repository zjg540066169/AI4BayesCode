## ising_cluster_block (Swendsen-Wang cluster sampler)

Specialised non-conjugate Gibbs sweep for the Ising / Potts target on
a user-supplied undirected graph:

  pi(x) ∝ exp{β · Σ_{i~j} I[x_i = x_j]},  x ∈ {0..Q-1}^n

via Swendsen-Wang (1987): (i) bond augmentation per like-coloured edge
with probability `1 − exp(−β)`; (ii) union-find cluster identification;
(iii) per-cluster uniform recolour over ALL Q states (stay-prob `1/Q`,
detailed-balance-preserving). For strongly-coupled discrete MRFs this
is the standard remedy for per-site Gibbs's catastrophic mixing
(`system_design.md` §11.2(b)).

**JUSTIFICATION (Check #16):** discrete-MRF target with strong local
dependence (Exception §11.2(b)). Algorithm is Swendsen-Wang 1987
(physics) / Higdon 1998 (statistician framing). Check #15 parity panel
under `tests/`:
- `test_ising_cluster_block.cpp` — 4×4 enumeration vs MC at β=0.5 and
  β=1.0; β=0 iid-Uniform boundary; two-init mixing; Q=3 Potts
  symmetry. 5 sub-tests.
- `test_ising_cluster_block_diagnostics.cpp` — split-R-hat across
  4 chains (incl. ordered-phase mode-mixing test on signed m);
  batch-means ESS; 17-bucket Pearson χ² vs enumeration; energy
  moments. 8 sub-tests.
- `test_ising_sw_vs_single_site.cpp` — quantified ≥ 5× per-sweep
  efficiency advantage over single-site Metropolis at β=1.0 ordered
  phase (empirically 6× on 10×10).

Ground truth is closed-form enumeration / textbook only — **zero
external package dependency** in the shipped tree.

Reference template: `examples/IsingPrior.cpp` (pure-prior 2D Ising /
Potts demo wiring `ising_cluster_block` through `composite_block` +
`RCPP_MODULE`).

**Scope (v1.2 ship):** h = 0 (no external field), scalar β (no per-
edge β_ij), no partial decoupling (Higdon §2.3). Each deferred to
v1.2.1 (deferred; see project roadmap).

```cpp
ising_cluster_block_config cfg;
cfg.name         = "x";
cfg.n_vertices   = L_x * L_y;
cfg.n_states     = 2;                  // 2 = Ising; ≥ 3 = Potts
cfg.edges        = AI4BayesCode::make_2d_lattice_edges(
                       L_x, L_y, /*periodic=*/false, /*eight_nn=*/false);
cfg.beta_key     = "beta";              // optional ctx slot for β
cfg.beta_default = 0.44;                // used if beta_key missing
cfg.initial_state = arma::vec(N, arma::fill::zeros);  // optional
```

**Helper:** `make_2d_lattice_edges(L_x, L_y, periodic, eight_nn)`
builds the 4-NN / 8-NN edge list for rectangular lattices. Supply
arbitrary `arma::umat (2 × n_edges)` for non-rectangular topologies.
