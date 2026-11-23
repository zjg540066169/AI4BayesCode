// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_bart_sigest.cpp -- sigest of bart_model's sigma prior on fixed designs, against values computed once in R.
//
//  With p < n the constructor sets sigest = sqrt(RSS / (n - rank)), with RSS and rank of the least-squares fit of
//  y - mean(y) on an intercept and the columns of x_train, a column being counted in the rank as lm() counts it (not
//  counted when it is zero or its norm after the earlier columns is below 1e-7 times its own norm); the initial sigma
//  is sigest and lambda = sigest^2 qchisq(0.1, nu) / nu. The expected sigest of each design is the sigest of
//  BART::wbart with rm.const as in the design,
//      yc <- y - mean(y); summary(lm(yc ~ ., data.frame(x, yc)))$sigma,
//  and the rank in the messages is lm(...)$rank; a rank other than lm's changes sigest by far more than the tolerance.
//  With p >= n, or with rank = n (RSS 0), sigest is sd(y) (expected value: sd(y) in R) and the initial sigma is 1.
//  The designs have exactly and nearly dependent columns on both sides of the threshold (relative residuals 7e-9,
//  7.6e-8, 1.6e-7 and 1e-6, each reached in one step, where lm() recomputes the column norm), a zero column, columns
//  with a large offset, y with a large offset and n - rank = 1. They are made from 20-bit uniforms by sums and products
//  whose results are exact in double precision (y + 2^26 is one correctly rounded addition), so every compiler and
//  platform builds the same data. "--dump <dir>" writes each design to <dir>/sigest_case_<c>.txt (first line n p
//  rm_const, then one line per observation: x_1 .. x_p y in %a format) for recomputing the expected values.
//  get_sigest() and the initial sigma are checked against the expected sigest, and bart_block's prior_sigest() and
//  prior_lambda() against bart_model's get_sigest() and get_lambda().
// ============================================================================
#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif
#include <armadillo>
#include "AI4BayesCode/bart_block.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>

static int g_fail = 0;
static void check(bool ok, const char* what) { std::printf("  %-104s %s\n", what, ok ? "ok" : "FAILED"); if (!ok) g_fail++; }

// x = k 2^-20 with k uniform on 0 .. 2^20 - 1; z = the sum of four such values minus 2 (mean 0, sd 0.58)
struct Gen {
  std::mt19937_64 g;
  explicit Gen(std::uint64_t s) : g(s) {}
  double u() { return (double)(g() >> 44) * 0x1p-20; }
  double z() { return u() + u() + u() + u() - 2.0; }
};

struct Case {
  const char* name; long n, p; bool rm_const;
  unsigned seed;     // seed of the data
  int rank;          // lm(...)$rank; 0 for the designs where sigest is sd(y)
  double sigest;     // expected sigest (R): BART::wbart's sigest, or sd(y) when rank is 0
  double tol;        // relative tolerance of sigest
};

// The design of case c: x uniform as above, y = 1.5 x0 - x1 + 0.5 x_{p-1} + 0.75 x0 x1 + 0.5 z, then the change of c.
static void make(int c, const Case& k, arma::mat& X, arma::vec& y)
{
  Gen r(k.seed);
  X.set_size(k.n, k.p); y.set_size(k.n);
  for (long j = 0; j < k.p; j++) for (long i = 0; i < k.n; i++) X(i, j) = r.u();
  for (long i = 0; i < k.n; i++) y(i) = 1.5 * X(i, 0) - X(i, 1) + 0.5 * X(i, k.p - 1) + 0.75 * X(i, 0) * X(i, 1) + 0.5 * r.z();
  switch (c) {
    case 2: case 3: X.col(2).fill(1.0); break;                        // a column of ones
    case 4: X.col(3).fill(5.0); break;                                // a constant column
    case 5: X.col(3) = X.col(1); break;                               // a duplicated column
    case 6: for (long i = 0; i < k.n; i++) { X(i, 2) = X(i, 3) = X(i, 4) = 0.0; X(i, 2 + i % 3) = 1.0; } break;
    case 7: for (long i = 0; i < k.n; i++) X(i, 3) = X(i, 1) + 0x1p-27 * r.z(); break;   // relative residual 7e-9
    case 8: for (long i = 0; i < k.n; i++) X(i, 3) = X(i, 1) + 0x1p-20 * r.z(); break;   // relative residual 1e-6
    case 9: X += 0x1p23; break;                                       // relative residual after the intercept 3e-8 to 4e-8
    case 10: y += 0x1p26; break;
    case 12: X.col(10) = X.col(3); break;
    case 15: X.col(5).zeros(); break;                                 // a zero column
    case 16: for (long i = 0; i < k.n; i++) X(i, 3) = X(i, 1) + 5 * 0x1p-26 * r.z(); break;   // relative residual 7.6e-8
    case 17: for (long i = 0; i < k.n; i++) X(i, 3) = X(i, 1) + 3 * 0x1p-24 * r.z(); break;   // relative residual 1.6e-7
    default: break;
  }
}

static const Case kCases[] = {
  {"full rank (n 60, p 4)",                                         60,  4, false, 1000,  5, 0x1.056b8a13255acp-2, 1e-12},
  {"full rank (n 200, p 40)",                                      200, 40, false, 1001, 41, 0x1.3ffc48bed3f94p-2, 1e-12},
  {"a column of ones, rm_const = false (n 80, p 5)",                80,  5, false, 1002,  5, 0x1.3bff4956b077ep-2, 1e-12},
  {"the same design, rm_const = true",                              80,  5, true,  1002,  5, 0x1.3bff4956b077ep-2, 1e-12},
  {"a constant column 5 (n 80, p 5)",                               80,  5, false, 1004,  5, 0x1.3e02a1e133e49p-2, 1e-12},
  {"a duplicated column (n 80, p 5)",                               80,  5, false, 1005,  5, 0x1.39458595d2caap-2, 1e-12},
  {"three indicator columns summing to 1 (n 90, p 5)",              90,  5, false, 1006,  5, 0x1.49bb00f72021p-2, 1e-12},
  {"x3 = x1 + 2^-27 z (relative residual 7e-9, not counted)",       120,  5, false, 1007,  5, 0x1.2c9ce60309ac9p-2, 1e-12},
  {"x3 = x1 + 2^-20 z (relative residual 1e-6, counted)",           120,  5, false, 1008,  6, 0x1.15f582422a84dp-2, 1e-10},
  {"all columns + 2^23: none counted (n 100, p 3)",                100,  3, false, 1009,  1, 0x1.62cacd7040e48p-1, 1e-12},
  {"y + 2^26 (n 100, p 3)",                                        100,  3, false, 1010,  4, 0x1.1c29ab43ed12cp-2, 1e-12},
  {"p = n - 2 (n 30, p 28): n - rank = 1",                          30, 28, false, 1011, 29, 0x1.7b2aa7a65ad97p-4, 1e-12},
  {"p = n - 1, a duplicated column (n 30, p 29): n - rank = 1", 30, 29, false, 1012, 29, 0x1.21ac4c638fa7dp-5, 1e-12},
  {"p = n - 1 (n 30, p 29): rank n, sigest = sd(y)",                30, 29, false, 1013,  0, 0x1.2841e7b029ceap-1, 1e-14},
  {"p = n (n 30, p 30): sigest = sd(y)",                            30, 30, false, 1014,  0, 0x1.3b03df9fafd6cp-1, 1e-14},
  {"a zero column (n 80, p 6)",                                       80,  6, false, 1015,  6, 0x1.5266943fe2a9bp-2, 1e-12},
  {"x3 = x1 + 5 2^-26 z (relative residual 7.6e-8, not counted)",   120,  5, false, 1016,  5, 0x1.2ed1261f5508dp-2, 1e-12},
  {"x3 = x1 + 3 2^-24 z (relative residual 1.6e-7, counted)",       120,  5, false, 1017,  6, 0x1.07e7dcd65f85ap-2, 1e-10},
};
static const int kN = (int)(sizeof kCases / sizeof kCases[0]);

int main(int argc, char** argv)
{
  if (argc > 2 && std::strcmp(argv[1], "--dump") == 0) {
    for (int c = 0; c < kN; c++) {
      arma::mat X; arma::vec y; make(c, kCases[c], X, y);
      const std::string fn = std::string(argv[2]) + "/sigest_case_" + std::to_string(c) + ".txt";
      FILE* f = std::fopen(fn.c_str(), "w");
      if (!f) { std::printf("cannot write %s\n", fn.c_str()); return 1; }
      std::fprintf(f, "%ld %ld %d\n", kCases[c].n, kCases[c].p, kCases[c].rm_const ? 1 : 0);
      for (long i = 0; i < kCases[c].n; i++) {
        for (long j = 0; j < kCases[c].p; j++) std::fprintf(f, "%a ", X(i, j));
        std::fprintf(f, "%a\n", y(i));
      }
      std::fclose(f);
    }
    return 0;
  }
  const double nu = 3.0, q = bart_sf::qchisq(0.1, nu);
  std::printf("relative difference of sigest (get_sigest(); the initial sigma equals it, or is 1 when sigest is sd(y)) and "
              "lambda from the values computed in R\n");
  for (int c = 0; c < kN; c++) {
    const Case& k = kCases[c];
    arma::mat X; arma::vec y; make(c, k, X, y);
    stdbart::bart_model m(X, y, 100L, false, false, k.rm_const, 20);
    char msg[256];
    const double rs = std::fabs(m.get_sigest() - k.sigest) / k.sigest;
    const double lam = k.sigest * k.sigest * q / nu, rl = std::fabs(m.get_lambda() - lam) / lam;
    if (k.rank > 0) {
      std::snprintf(msg, sizeof msg, "%s; rank %d: sigest %.1e, lambda %.1e", k.name, k.rank, rs, rl);
      check(rs < k.tol && m.get_sigma() == m.get_sigest() && rl < 2.0 * k.tol + 1e-15, msg);
    } else {
      std::snprintf(msg, sizeof msg, "%s: sigest (sd(y)) %.1e, lambda %.1e, initial sigma %g", k.name, rs, rl,
                    m.get_sigma());
      check(rs < k.tol && rl < 2.0 * k.tol + 1e-15 && m.get_sigma() == 1.0, msg);
    }
  }
  std::printf("bart_block: prior_sigest() and prior_lambda() equal bart_model's get_sigest() and get_lambda()\n");
  for (int c : {0, 14}) {
    arma::mat X; arma::vec y; make(c, kCases[c], X, y);
    AI4BayesCode::bart_block_config cfg; cfg.x_train = X; cfg.y_init = y; cfg.ntrees = 20;
    AI4BayesCode::bart_block b(cfg);
    const stdbart::bart_model& m = *b.underlying();
    char msg[256];
    std::snprintf(msg, sizeof msg, "%s: prior_sigest %.6g, prior_lambda %.6g, current_sigma %g", kCases[c].name,
                  b.prior_sigest(), b.prior_lambda(), b.current_sigma());
    check(b.prior_sigest() == m.get_sigest() && b.prior_lambda() == m.get_lambda() &&
          b.current_sigma() == (kCases[c].rank > 0 ? b.prior_sigest() : 1.0), msg);
  }
  std::printf("%s\n", g_fail ? "SOME FAILED" : "ALL PASS");
  return g_fail ? 1 : 0;
}
