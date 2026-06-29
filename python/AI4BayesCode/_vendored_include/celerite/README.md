# celerite (vendored)

Vendored subset of celerite v0.4 from
https://github.com/dfm/celerite (MIT License).

Imports only the C++ core (solver + utils + exceptions). Skips the
Python bindings, Stan wrapper, autodiff tests, and benchmarks — we
use celerite purely as a fast O(N) 1-D semi-separable Gaussian-process
kernel solver inside AI4BayesCode.

See AI4BayesCode/include/AI4BayesCode/celerite_gp_block.hpp for the
AI4BayesCode integration.

License: MIT (see LICENSE in this directory).

Citation: Foreman-Mackey, D., Agol, E., Angus, R., and Ambikasaran, S.
(2017). "Fast and scalable Gaussian process modeling with applications
to astronomical time series." AJ 154:220. arXiv:1703.09710.
