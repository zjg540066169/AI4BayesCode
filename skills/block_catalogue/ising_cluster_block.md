## ising_cluster_block (Swendsen-Wang cluster sampler)

Specialised non-conjugate Gibbs sweep for the Ising / Potts target on
a user-supplied undirected graph, with an OPTIONAL external field +
partial decoupling:

  pi(x) ∝ exp{ Σ_i α_i(x_i) + Σ_{i~j} β_ij · I[x_i = x_j] },  x ∈ {0..Q-1}^n

via Swendsen-Wang (1987) / Higdon (1998): (i) bond augmentation per
like-coloured edge with probability `1 − exp(−δ_ij·β_ij)`; (ii)
union-find cluster identification; (iii) cluster recolour by one of
three EXACT paths — uniform (no field, δ=1), field-weighted
`∝ exp{Σ_{i∈C} α_i(k)}` (δ=1), or a cluster-conditional Gibbs sweep
carrying the residual `(1−δ)β` coupling (δ<1, Higdon §2.3). For
strongly-coupled discrete MRFs this is the standard remedy for
per-site Gibbs's catastrophic mixing (`system_design.md` §11.2(b));
partial decoupling (δ<1) additionally rescues mixing under a strong
external field.

**JUSTIFICATION (Check #16):** discrete-MRF target with strong local
dependence (Exception §11.2(b)). Algorithm is Swendsen-Wang 1987
(physics) / Higdon 1998 (statistician framing). Check #15 parity panel
under `tests/`:
- `test_ising_cluster_block.cpp` — 4×4 enumeration vs MC at β=0.5 and
  β=1.0; β=0 iid-Uniform boundary; two-init mixing; Q=3 Potts
  symmetry; **T6** external-field per-site marginals vs 2^16 enumeration
  (δ=1, Case B); **T7** per-edge β_ij agreement vs enumeration; **T8**
  partial-decoupling δ=0.5 (Case C) recovering the SAME enumeration
  (target-invariance). 8 sub-tests; each v1.2.1 test also gates on
  split-R-hat < 1.01.
- `test_ising_cluster_block_diagnostics.cpp` — split-R-hat across
  4 chains (incl. ordered-phase mode-mixing test on signed m);
  batch-means ESS; 17-bucket Pearson χ² vs enumeration; energy
  moments. 8 sub-tests.
- `test_ising_sw_vs_single_site.cpp` — quantified ≥ 5× per-sweep
  efficiency advantage over single-site Metropolis at β=1.0 ordered
  phase (empirically 6× on 10×10).

Ground truth is closed-form enumeration / textbook only — **zero
external package dependency** in the shipped tree.

Reference templates (both frontend-independent standalone `int main()`
demos — no Rcpp/pybind, so R and Python both build them):
- `examples/IsingPrior.cpp` — pure-prior 2D Ising / Potts demo wiring
  `ising_cluster_block` through `composite_block`.
- `examples/IsingHiddenPotts.cpp` — hidden-Potts image segmentation
  exercising the v1.2.1 external-field + partial-decoupling path (emission
  field published under `field_key`, δ=0.6); recovers the segmentation at
  ~95% accuracy with two-chain R-hat < 1.01.

**Scope (v1.2.1 shipped):** external field `h ≠ 0` (per-site α_i(k)),
per-edge β_ij, and partial decoupling δ_ij (Higdon §2.3) are all
supported. Every extension is OPTIONAL — omit them (empty `field`,
scalar β, δ=1) and the block is the exact v1.2 Swendsen-Wang sampler
(verified: T1–T5 unchanged). δ affects mixing only; the stationary
target is unchanged (T8 enumeration-invariance).

```cpp
ising_cluster_block_config cfg;
cfg.name         = "x";
cfg.n_vertices   = L_x * L_y;
cfg.n_states     = 2;                  // 2 = Ising; ≥ 3 = Potts
cfg.edges        = AI4BayesCode::make_2d_lattice_edges(
                       L_x, L_y, /*periodic=*/false, /*eight_nn=*/false);
cfg.beta_key     = "beta";              // optional ctx slot for scalar β
cfg.beta_default = 0.44;                // used if beta_key missing
// ---- v1.2.1 extensions (ALL optional; empty ⇒ standard SW) ----
cfg.field         = alpha;             // n × Q log-potentials α_i(k), OR
cfg.field_key     = "log_lik";         //   ctx slot for vectorise(alpha)
cfg.beta_edge     = beta_ij;           // per-edge β (length n_edges), OR beta_edge_key
cfg.delta_default = 0.5;               // partial decoupling δ ∈ [0,1]
                                       //   (δ<1 rescues mixing under a strong field)
cfg.delta_edge    = delta_ij;          // optional per-edge δ (length n_edges)
cfg.initial_state = arma::vec(N, arma::fill::zeros);  // optional
```

**Hidden-Potts wiring:** a sibling emission block publishes the per-site
log-likelihood `α_i(k) = log p(y_i | θ_k)` under `field_key` (as
`vectorise` of an n×Q matrix); the ising block reads it each sweep and
recolours ∝ exp{Σ_{i∈C} α_i(k)}. Use `delta_default < 1` when the field
is strong (Higdon §2.3.2; Moores et al.).

**Helper:** `make_2d_lattice_edges(L_x, L_y, periodic, eight_nn)`
builds the 4-NN / 8-NN edge list for rectangular lattices. Supply
arbitrary `arma::umat (2 × n_edges)` for non-rectangular topologies.
