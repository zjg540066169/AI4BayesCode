// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_bart_lambda.cpp -- the sigma prior that bart_model's constructor calibrates from the residual variance
//  of the least-squares fit of y on X: sigest = sqrt(RSS / (n - p)), lambda = sigest^2 qchisq(0.1, nu) / nu.
//
//   1. RSS is the residual sum of squares of the least-squares fit, also for designs whose columns are linearly
//      dependent: lambda equals the value computed from a reference RSS (closed form, or the fit on the design with
//      the dependent columns removed). For a saturated design (p = n - 1) RSS is 0, so lambda comes from sd(y) and
//      the initial sigma is 1.
//   2. lambda is the same, bit for bit, in the first construction in each of several freshly started processes (this
//      program run again with the argument --first-lambda; n 1000, p 500) and in a second construction in each of
//      them. When the least-squares fit used LAPACK, the first construction in a process could give another value than
//      the later ones.
//   3. With p < n, a NaN or infinite value in x_train makes the constructor throw std::invalid_argument.
// ============================================================================
#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif
#include <armadillo>
#include "AI4BayesCode/bart_block.hpp"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static int g_fail = 0;
static void check(bool ok, const char* what) { std::printf("  %-104s %s\n", what, ok ? "ok" : "FAILED"); if (!ok) g_fail++; }
static std::uint64_t bits(double d) { std::uint64_t b; std::memcpy(&b, &d, sizeof b); return b; }

static double lambda_of(double rss, long n, long p, double nu) {
  const double s = std::sqrt(rss / (double)(n - p));
  return s * s * bart_sf::qchisq(0.1, nu) / nu;
}
// RSS of a full-rank least-squares fit with an intercept, as a reference: y minus its projection on the columns of the
// thin Q of a QR factorisation
static double rss_ref(const arma::mat& X, const arma::vec& y) {
  arma::mat A = X; A.insert_cols(0, arma::ones<arma::vec>(X.n_rows));
  arma::mat Q, R; arma::qr_econ(Q, R, A);
  const arma::vec r = y - Q * (Q.t() * y);
  return arma::dot(r, r);
}
static double rel(double a, double b) { return std::fabs(a - b) / std::fabs(b); }

// ---------------------------------------------------------------------------------------------------------------
// 1. lambda from the residual sum of squares of the least-squares fit
// ---------------------------------------------------------------------------------------------------------------
static void part1()
{
  std::mt19937_64 g(41); std::uniform_real_distribution<double> U(0.0, 1.0); std::normal_distribution<double> Z(0.0, 1.0);
  const double nu = 3.0;
  char msg[200];
  std::printf("1. lambda from the residual sum of squares of the least-squares fit\n");
  {   // full rank
    const long n = 400, p = 6; arma::mat X(n, p); arma::vec y(n);
    for (arma::uword i = 0; i < X.n_elem; i++) X(i) = U(g);
    for (long i = 0; i < n; i++) y[i] = 2.0 * X(i, 0) - X(i, 3) + 0.5 * Z(g);
    stdbart::bart_model m(X, y, 100L, false, false, false, 20);
    const double r = rel(m.get_lambda(), lambda_of(rss_ref(X, y), n, p, nu));
    std::snprintf(msg, sizeof msg, "full-rank design (n 400, p 6): relative difference %.1e", r); check(r < 1e-10, msg);
  }
  {   // three indicator columns that sum to the intercept: RSS = within-group sum of squares
    const long n = 300, p = 3; arma::mat X(n, p, arma::fill::zeros); arma::vec y(n);
    for (long i = 0; i < n; i++) { X(i, i % 3) = 1.0; y[i] = 0.5 * (i % 3) + 0.3 * Z(g); }
    double rss = 0.0;
    for (long k = 0; k < 3; k++) {
      double mu = 0.0; long c = 0; for (long i = k; i < n; i += 3) { mu += y[i]; c++; } mu /= c;
      for (long i = k; i < n; i += 3) rss += (y[i] - mu) * (y[i] - mu);
    }
    stdbart::bart_model m(X, y, 100L, false, false, false, 20);
    const double r = rel(m.get_lambda(), lambda_of(rss, n, p, nu));
    std::snprintf(msg, sizeof msg, "indicators summing to the intercept, RSS = within-group sum of squares: rel. diff. %.1e", r);
    check(r < 1e-12, msg);
  }
  {   // a duplicated column, and a constant column next to the intercept (rm_const = true): the fit of the design
      // without the dependent column
    const long n = 300; arma::mat X(n, 4); arma::vec y(n);
    for (long i = 0; i < n; i++) { X(i, 0) = U(g); X(i, 1) = X(i, 0); X(i, 2) = U(g); X(i, 3) = U(g); y[i] = X(i, 0) + X(i, 2) + 0.3 * Z(g); }
    const arma::mat Xr = arma::join_rows(X.col(0), X.cols(2, 3));
    stdbart::bart_model m(X, y, 100L, false, false, false, 20);
    const double r1 = rel(m.get_lambda(), lambda_of(rss_ref(Xr, y), n, 4, nu));
    std::snprintf(msg, sizeof msg, "duplicated column: relative difference %.1e", r1); check(r1 < 1e-10, msg);
    arma::mat Xc = X; Xc.col(1).fill(1.0);
    const arma::mat Xcr = arma::join_rows(Xc.col(0), Xc.cols(2, 3));
    stdbart::bart_model mc(Xc, y, 100L, false, false, /*rm_const=*/true, 20);
    const double r2 = rel(mc.get_lambda(), lambda_of(rss_ref(Xcr, y), n, 4, nu));
    std::snprintf(msg, sizeof msg, "constant column with rm_const = true: relative difference %.1e", r2); check(r2 < 1e-10, msg);
  }
  {   // saturated design: the fit interpolates y, RSS = 0, so lambda comes from sd(y) and the initial sigma is 1
    const long n = 50, p = 49; arma::mat X(n, p); arma::vec y(n);
    for (arma::uword i = 0; i < X.n_elem; i++) X(i) = U(g);
    for (long i = 0; i < n; i++) y[i] = Z(g);
    stdbart::bart_model m(X, y, 100L, false, false, false, 20);
    const double s = arma::stddev(y), lam_sd = (s * s * bart_sf::qchisq(0.1, nu)) / nu;
    const double r = rel(m.get_lambda(), lam_sd);
    std::snprintf(msg, sizeof msg, "saturated design (n 50, p 49): lambda from sd(y) (rel. diff. %.1e), initial sigma %.17g", r, m.get_sigma());
    check(r < 1e-14 && m.get_sigma() == 1.0, msg);
  }
}

// ---------------------------------------------------------------------------------------------------------------
// 2. the same lambda in the first construction of every freshly started process
// ---------------------------------------------------------------------------------------------------------------
static void first_lambda_data(arma::mat& X, arma::vec& y)
{
  const long n = 1000, p = 500; const double pi = 3.14159265358979323846;
  std::mt19937_64 g(777); std::uniform_real_distribution<double> U(0.0, 1.0); std::normal_distribution<double> Z(0.0, 1.0);
  X.set_size(n, p); for (long j = 0; j < p; j++) for (long i = 0; i < n; i++) X(i, j) = U(g);
  y.set_size(n);
  for (long i = 0; i < n; i++) y(i) = 10 * std::sin(pi * X(i, 0) * X(i, 1)) + 20 * std::pow(X(i, 2) - 0.5, 2) + 10 * X(i, 3) + 5 * X(i, 4) + Z(g);
}
// run as a child process: the bits of lambda from the first and the second construction in this process
static int child_first_lambda()
{
  arma::mat X; arma::vec y; first_lambda_data(X, y);
  double lam[2];
  for (int r = 0; r < 2; r++) { stdbart::bart_model m(X, y, 100L, false, false, false, 200); lam[r] = m.get_lambda(); }
  std::printf("%016llx %016llx\n", (unsigned long long)bits(lam[0]), (unsigned long long)bits(lam[1]));
  return 0;
}

static void part2(const char* self)
{
  std::printf("2. the same lambda in the first construction of every freshly started process (n 1000, p 500)\n");
  const int K = 16;
  std::string prog = self ? self : "";
#ifdef _WIN32
  for (char& ch : prog) if (ch == '/') ch = '\\';   // cmd.exe runs a path written with backslashes
#endif
  const std::string tag = std::to_string((unsigned long long)std::chrono::steady_clock::now().time_since_epoch().count());
  std::vector<std::string> first, second; int not_run = 0;
  for (int c = 0; c < K; c++) {   // each child's output goes to a file in the working directory, removed after reading
    const std::string out = "test_fix_bart_lambda_child_" + tag + "_" + std::to_string(c) + ".txt";
    std::string cmd = "\"" + prog + "\" --first-lambda > \"" + out + "\"";
#ifdef _WIN32
    cmd = "\"" + cmd + "\"";   // cmd.exe /c removes the outermost pair of quotes
#endif
    const int rc = std::system(cmd.c_str());
    std::string a, b;
    { std::ifstream in(out); if (rc != 0 || !(in >> a >> b)) not_run++; else { first.push_back(a); second.push_back(b); } }
    std::remove(out.c_str());
  }
  char msg[200];
  std::snprintf(msg, sizeof msg, "%d of %d child processes (this program with --first-lambda) printed two values", K - not_run, K);
  check(not_run == 0, msg);
  std::map<std::string, int> cnt; for (const std::string& s : first) cnt[s]++;
  std::snprintf(msg, sizeof msg, "first construction in %zu fresh processes: %zu distinct value(s) of lambda", first.size(), cnt.size());
  check(!first.empty() && cnt.size() == 1, msg);
  if (cnt.size() > 1)
    for (const auto& kv : cnt) {
      const std::uint64_t u = std::strtoull(kv.first.c_str(), nullptr, 16); double d; std::memcpy(&d, &u, sizeof d);
      std::printf("      lambda %a in %d processes\n", d, kv.second);
    }
  int differ = 0; for (size_t c = 0; c < first.size(); c++) if (first[c] != second[c]) differ++;
  std::snprintf(msg, sizeof msg, "second construction in the same process: differs from the first in %d of %zu processes", differ, first.size());
  check(!first.empty() && differ == 0, msg);
}

// ---------------------------------------------------------------------------------------------------------------
// 3. a non-finite x_train
// ---------------------------------------------------------------------------------------------------------------
static void part3()
{
  std::printf("3. with p < n, a NaN or infinite value in x_train makes the constructor throw std::invalid_argument\n");
  const double bad[3] = {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()};
  const char* names[3] = {"NaN", "+Inf", "-Inf"};
  for (int c = 0; c < 3; c++) {
    std::mt19937_64 g(5u + (unsigned)c); std::uniform_real_distribution<double> U(0.0, 1.0);
    arma::mat X(50, 3); arma::vec y(50);
    for (arma::uword i = 0; i < X.n_elem; i++) X(i) = U(g);
    for (arma::uword i = 0; i < y.n_elem; i++) y(i) = U(g);
    X(7, 1) = bad[c];
    std::string got = "no exception"; bool ok = false;
    try { stdbart::bart_model m(X, y, 100L, false, false, false, 20); }
    catch (const std::invalid_argument&) { ok = true; got = "std::invalid_argument"; }
    catch (const std::exception& e) { got = std::string("another exception: ") + e.what(); }
    char msg[200]; std::snprintf(msg, sizeof msg, "x_train(7, 1) = %s (n 50, p 3): %.60s", names[c], got.c_str());
    check(ok, msg);
  }
}

int main(int argc, char** argv)
{
  if (argc > 1 && std::strcmp(argv[1], "--first-lambda") == 0) return child_first_lambda();
  part1();
  part2(argc > 0 ? argv[0] : nullptr);
  part3();
  std::printf("%s\n", g_fail ? "SOME FAILED" : "ALL PASS");
  return g_fail ? 1 : 0;
}
