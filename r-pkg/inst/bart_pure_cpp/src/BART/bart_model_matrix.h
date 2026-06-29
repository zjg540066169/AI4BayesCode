/*
 *  Copyright (C) 2017-2018 Robert McCulloch, Rodney Sparapani
 *                          and Charles Spanbauer
 *  Copyright (C) 2024-2026 Jungang Zou
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, a copy is available at
 *  https://www.R-project.org/Licenses/GPL-2
 */

/*
 * BART/bart_model_matrix.h — pure C++ (Armadillo) per-column cutpoint
 * (xinfo) builder from a numeric design matrix.  R-FREE: no Rcpp types.
 *
 * Ported 2026-06-21 from the Rcpp version to plain Armadillo so the BART
 * model wrapper compiles as a standalone C++ / Python-bindable core.
 * Cutpoint logic is byte-for-byte identical to the prior Rcpp version
 * (bartModelMatrix.R lines 65-110 for the numeric-matrix case):
 *   k = #unique(X[,j])
 *   if k <= 1           : constant column, nc[j] = 1, single cutpoint
 *   else if cont==TRUE  : nc[j] = numcut, evenly spaced (endpoints dropped)
 *   else if k <  numcut : nc[j] = k - 1, midpoints of unique values
 *   else if usequants   : nc[j] = numcut, R type-7 quantiles
 *   else                : nc[j] = numcut, evenly spaced
 *
 * Output is a *ragged* xinfo (std::vector<std::vector<double>>, p rows,
 * row j has numcut[j] cutpoints) — exactly the `xinfo` type the BART
 * kernel's setxinfo() consumes, with no NA padding.
 */

#ifndef BART_BART_MODEL_MATRIX_H_
#define BART_BART_MODEL_MATRIX_H_

#include <armadillo>
#include <algorithm>
#include <cmath>
#include <vector>

namespace bart_pp {

struct BmmResult {
  std::vector<int>                  numcut;   // length p — cuts per variable
  std::vector<std::vector<double>>  xinfo;    // p rows; row j has numcut[j] cuts
};

// Pure C++.  X is N x p (observations in rows), like the user-facing matrix.
inline BmmResult compute_bart_model_matrix(
    const arma::mat& X,
    int   numcut    = 100,
    bool  usequants = false,
    int   type      = 7,    // quantile type (R's default); only type 7 supported
    bool  cont      = false)
{
  (void)type;
  const int N = (int)X.n_rows;
  const int p = (int)X.n_cols;

  BmmResult out;
  out.numcut.assign(p, 0);
  out.xinfo.assign(p, std::vector<double>());

  if (N == 0 || p == 0 || numcut <= 0) return out;

  std::vector<double> col_sorted(N);

  for (int j = 0; j < p; ++j) {
    for (int i = 0; i < N; ++i) col_sorted[i] = X(i, j);
    std::sort(col_sorted.begin(), col_sorted.end());

    std::vector<double> xs;
    xs.reserve(N);
    for (int i = 0; i < N; ++i)
      if (i == 0 || col_sorted[i] != col_sorted[i - 1]) xs.push_back(col_sorted[i]);
    const int k = (int)xs.size();

    if (k <= 1) {
      out.numcut[j] = 1;
      out.xinfo[j].assign(1, (k == 1) ? xs[0] : 0.0);
    }
    else if (cont) {
      out.numcut[j] = numcut;
      out.xinfo[j].resize(numcut);
      const double lo = xs.front(), hi = xs.back();
      const double step = (hi - lo) / (numcut + 1.0);
      for (int c = 0; c < numcut; ++c) out.xinfo[j][c] = lo + (c + 1) * step;
    }
    else if (k < numcut) {
      out.numcut[j] = k - 1;
      out.xinfo[j].resize(k - 1);
      for (int c = 0; c < k - 1; ++c) out.xinfo[j][c] = 0.5 * (xs[c] + xs[c + 1]);
    }
    else if (usequants) {
      out.numcut[j] = numcut;
      out.xinfo[j].resize(numcut);
      for (int c = 0; c < numcut; ++c) {
        const double prob = (double)(c + 1) / (numcut + 1.0);
        const double h    = prob * (N - 1);
        int lo_idx = (int)std::floor(h);
        int hi_idx = lo_idx + 1;
        if (hi_idx >= N) hi_idx = N - 1;
        const double gamma = h - lo_idx;
        out.xinfo[j][c] = col_sorted[lo_idx] * (1.0 - gamma)
                        + col_sorted[hi_idx] * gamma;
      }
    }
    else {
      out.numcut[j] = numcut;
      out.xinfo[j].resize(numcut);
      const double lo = xs.front(), hi = xs.back();
      const double step = (hi - lo) / (numcut + 1.0);
      for (int c = 0; c < numcut; ++c) out.xinfo[j][c] = lo + (c + 1) * step;
    }
  }
  return out;
}

}  // namespace bart_pp

#endif  // BART_BART_MODEL_MATRIX_H_
