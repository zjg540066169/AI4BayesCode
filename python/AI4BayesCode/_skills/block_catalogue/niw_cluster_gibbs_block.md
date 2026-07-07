## niw_cluster_gibbs_block (full-covariance Gaussian clusters)

Closed-form vectorized Gibbs leaf that samples per-cluster
(mu_k, Sigma_k) from a conjugate Normal-Inverse-Wishart prior across
K_trunc clusters. Companion to `normal_gamma_cluster_gibbs_block`
(diagonal Sigma); use this when within-cluster correlation matters.

```cpp
#include "AI4BayesCode/niw_cluster_gibbs_block.hpp"

niw_cluster_gibbs_block_config cfg;
cfg.name        = "cluster_params";   // block label
cfg.K_trunc     = 20;
cfg.d           = 3;
cfg.N           = 200;
cfg.z_key       = "z";
cfg.y_key       = "y";                // length N * d, row-major
cfg.mu_name     = "mu";               // output: K * d, cluster-major
cfg.sigma_name  = "sigma";            // output: K * d * d, cluster-major,
                                      //         row-major within dxd block
cfg.mu_0        = arma::vec(3, arma::fill::zeros);
cfg.kappa_0     = 0.1;
cfg.Psi_0_flat  = arma::vectorise(arma::eye<arma::mat>(3, 3)); // I_3 row-major
cfg.nu_0        = 5.0;                 // > d - 1 (here > 2). nu - d - 1 > 0
                                       // for E[Sigma|prior] to exist.
cfg.initial_mu      = ...;             // K * d
cfg.initial_sigma   = ...;             // K * d * d (PSD per cluster)
```

**Posterior** (per cluster, n_k > 0): the standard NIW conjugate update
(Murphy 2007 Sec.4 eqs 4-11). **IW sampling**: Bartlett decomposition
(Bartlett 1933, Anderson 2003) -- numerically stable, no matrix inverse
on the hot path beyond one inv_sympd of `Psi_n` per cluster per step.

**Empty cluster** (n_k == 0): draws Sigma_k ~ IW(Psi_0, nu_0); mu_k | Sigma_k ~
N(mu_0, Sigma_k / kappa_0). Matches `normal_gamma_cluster_gibbs_block` and
Ishwaran & James 2001 Sec.3.2.

**Output convention**: `sigma_name` exposes COVARIANCE (NOT precision).
This differs from `normal_gamma_cluster_gibbs_block` whose `lambda_name`
is precision. The names disambiguate: `lambda` for precision, `sigma`
for covariance. User downstream blocks (`categorical_gibbs_block`'s
`log_probs_fn`) need to know which they're consuming.

**JUSTIFICATION (Check #16): Exception 1 from codegen_priors.md Sec.2b**
(NEW Tier-B block; conjugate cluster update is the textbook Murphy
2007 Sec.4 formula; Bartlett decomposition is the standard IW sampling
method).

**Check #15** parity:
`tests_autodiff/block_tests/test_niw_cluster_gibbs_block.cpp` covers
empty-cluster prior recovery, populated single-cluster posterior
recovery (mu within 0.4% rel err, Sigma diagonal within 1% rel err
on 4000 draws), and multi-cluster PSD-shape sanity.

**When to pick NIW over Normal-Gamma**: NIW for d >= 2 with off-diagonal
structure; Normal-Gamma for d == 1 or for diagonal covariance assumption.
At d = 1, NIW reduces to NormalGamma but normal_gamma_cluster is
cheaper there.
