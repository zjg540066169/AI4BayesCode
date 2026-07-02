/*
 *  Copyright (C) 2017-2018 Robert McCulloch, Rodney Sparapani
 *                          and Charles Spanbauer
 *  Copyright (C) 2024-2026 Jungang Zou
 *
 *  GPL-2-or-later.  https://www.R-project.org/Licenses/GPL-2
 */

/*
 * GENBART/BART/bart_model_matrix.h — pure C++ (Armadillo) per-column
 * cutpoint (xinfo) builder for generalized BART.  R-FREE.
 * Ported 2026-06-21; cutpoint logic byte-identical to the Rcpp version
 * (see BART/bart_model_matrix.h).  `genbart_pp` namespace.
 */

#ifndef GENBART_BART_MODEL_MATRIX_H_
#define GENBART_BART_MODEL_MATRIX_H_

#include <armadillo>
#include <algorithm>
#include <cmath>
#include <vector>

namespace genbart_pp {

struct BmmResult {
  arma::mat                         X;        // passthrough N x p design matrix
  std::vector<int>                  numcut;   // length p — cuts per variable
  std::vector<std::vector<double>>  xinfo;    // p rows; row j has numcut[j] cuts
};

inline BmmResult compute_bart_model_matrix(
    const arma::mat& X, int numcut = 100, bool usequants = false,
    int type = 7, bool cont = false)
{
  (void)type;
  const int N = (int)X.n_rows;
  const int p = (int)X.n_cols;
  BmmResult out;
  out.X = X;
  out.numcut.assign(p, 0);
  out.xinfo.assign(p, std::vector<double>());
  if (N == 0 || p == 0 || numcut <= 0) return out;

  std::vector<double> col_sorted(N);
  for (int j = 0; j < p; ++j) {
    for (int i = 0; i < N; ++i) col_sorted[i] = X(i, j);
    std::sort(col_sorted.begin(), col_sorted.end());
    std::vector<double> xs; xs.reserve(N);
    for (int i = 0; i < N; ++i)
      if (i == 0 || col_sorted[i] != col_sorted[i - 1]) xs.push_back(col_sorted[i]);
    const int k = (int)xs.size();
    if (k <= 1) {
      out.numcut[j] = 1; out.xinfo[j].assign(1, (k == 1) ? xs[0] : 0.0);
    } else if (cont) {
      out.numcut[j] = numcut; out.xinfo[j].resize(numcut);
      const double lo = xs.front(), hi = xs.back();
      const double step = (hi - lo) / (numcut + 1.0);
      for (int c = 0; c < numcut; ++c) out.xinfo[j][c] = lo + (c + 1) * step;
    } else if (k < numcut) {
      out.numcut[j] = k - 1; out.xinfo[j].resize(k - 1);
      for (int c = 0; c < k - 1; ++c) out.xinfo[j][c] = 0.5 * (xs[c] + xs[c + 1]);
    } else if (usequants) {
      out.numcut[j] = numcut; out.xinfo[j].resize(numcut);
      for (int c = 0; c < numcut; ++c) {
        const double prob = (double)(c + 1) / (numcut + 1.0);
        const double h = prob * (N - 1);
        int lo_idx = (int)std::floor(h), hi_idx = lo_idx + 1;
        if (hi_idx >= N) hi_idx = N - 1;
        const double gamma = h - lo_idx;
        out.xinfo[j][c] = col_sorted[lo_idx]*(1.0-gamma) + col_sorted[hi_idx]*gamma;
      }
    } else {
      out.numcut[j] = numcut; out.xinfo[j].resize(numcut);
      const double lo = xs.front(), hi = xs.back();
      const double step = (hi - lo) / (numcut + 1.0);
      for (int c = 0; c < numcut; ++c) out.xinfo[j][c] = lo + (c + 1) * step;
    }
  }
  return out;
}

}  // namespace genbart_pp
#endif  // GENBART_BART_MODEL_MATRIX_H_
