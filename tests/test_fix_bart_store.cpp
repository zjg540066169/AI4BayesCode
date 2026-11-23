// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_bart_store.cpp -- the limit up to which the BART kernel keeps the fits of its m trees at the n
//  observations between draws, set at compile time by AI4BAYESCODE_BART_FIT_STORE_MAX_DOUBLES (here 2000 doubles).
//
//   1. the kernel's limit is the value of the macro;
//   2. with m n at most the limit the fits are kept, above it none is kept, also when setm and setdata move m n across
//      the limit;
//   3. through these changes, with new x values before every draw, the draws, fits and trees equal, bit for bit, those
//      of a kernel that keeps no fit (limit 0).
// ============================================================================
#define AI4BAYESCODE_BART_FIT_STORE_MAX_DOUBLES 2000
#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif
#include <armadillo>
#include "AI4BayesCode/bart_block.hpp"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <vector>

static int g_fail = 0;
static void check(bool ok, const char* what) { std::printf("  %-100s %s\n", what, ok ? "ok" : "FAILED"); if (!ok) g_fail++; }

// the library kernel with its limit set to c doubles, and the number of fits it keeps
struct kernel : public heterbart {
  explicit kernel(size_t m_) : heterbart(m_) {}
  size_t cap() const { return tfit_cap_; }
  void set_cap(size_t c) { tfit_cap_ = c; }
  size_t kept() const { return tfit_.size(); }
};

struct Hash {
  std::uint64_t h = 1469598103934665603ULL;
  void add(const void* q, size_t nb) { const unsigned char* c = (const unsigned char*)q; for (size_t i = 0; i < nb; i++) { h ^= c[i]; h *= 1099511628211ULL; } }
};

// One scripted run of the kernel with limit `cap`; returns the hash of the fits after every draw, the trees and the
// engine state, and the number of fits kept after each step in `kept`.
static std::uint64_t script(size_t cap, std::vector<size_t>& kept)
{
  const size_t p = 3, nmax = 100;
  std::mt19937_64 g(11); std::uniform_real_distribution<double> U(0.0, 1.0); std::normal_distribution<double> Z(0.0, 1.0);
  std::vector<double> x(nmax * p), y(nmax), sv(nmax);
  for (double& v : x) v = U(g);
  for (size_t i = 0; i < nmax; i++) { y[i] = 1.5 * (x[i * p] > 0.5) + x[i * p + 1] - 1.0 + 0.3 * Z(g); sv[i] = 0.5 + U(g); }
  bart_rng::set_seed(21); arn gen; Hash h;
  kernel b(20); b.set_cap(cap); b.setprior(0.95, 2.0, 0.4);
  size_t n = nmax;
  auto step = [&](int draws) {
    for (int s = 0; s < draws; s++) {
      for (int k = 0; k < 6; k++) x[(size_t)(U(g) * n * p)] = U(g);                     // new x values, then setdata
      b.setdata(p, n, x.data(), y.data(), (size_t)25);
      if (s % 2) b.draw(0.8, gen); else b.draw(sv.data(), gen);
      for (size_t k = 0; k < n; k++) { const double f = b.f(k); h.add(&f, 8); }
    }
    kept.push_back(b.kept());
  };
  b.setdata(p, n, x.data(), y.data(), (size_t)25);
  step(6);                          // m 20, n 100: m n = 2000
  b.setm(21); step(4);              // m n = 2100
  n = 90; step(4);                  // m n = 1890
  n = 100; step(3);                 // m n = 2100
  b.setm(15); step(4);              // m n = 1500
  { tree::npv bt; b.gettree(2).getbots(bt); bt[0]->settheta(0.3); b.refresh_allfit(); } step(3);
  for (size_t j = 0; j < 15; j++) { std::ostringstream os; os.precision(17); os << b.gettree(j); const std::string t = os.str(); h.add(t.data(), t.size()); }
  const std::uint64_t e = bart_rng::engine()(); h.add(&e, 8);
  return h.h;
}

int main()
{
  std::printf("the kept tree fits with AI4BAYESCODE_BART_FIT_STORE_MAX_DOUBLES = 2000\n");
  { kernel b(1); char msg[120]; std::snprintf(msg, sizeof msg, "limit of the kernel: %zu doubles", b.cap()); check(b.cap() == 2000, msg); }
  std::vector<size_t> k_macro, k_none;
  const std::uint64_t h_macro = script(kernel(1).cap(), k_macro), h_none = script(0, k_none);
  const std::vector<size_t> want = {2000, 0, 1890, 0, 1500, 1500};
  char msg[200];
  std::snprintf(msg, sizeof msg, "fits kept at m n = 2000, 2100, 1890, 2100, 1500, 1500 (setm, setdata): %zu %zu %zu %zu %zu %zu",
                k_macro[0], k_macro[1], k_macro[2], k_macro[3], k_macro[4], k_macro[5]);
  check(k_macro == want, msg);
  check(k_none == std::vector<size_t>(6, 0), "limit 0: no fit kept");
  check(h_macro == h_none, "draws, fits and trees equal those with no fit kept, bit for bit");
  std::printf("%s\n", g_fail ? "SOME FAILED" : "ALL PASS");
  return g_fail ? 1 : 0;
}
