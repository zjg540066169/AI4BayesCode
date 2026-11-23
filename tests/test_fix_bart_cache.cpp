// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_bart_cache.cpp -- the values the BART kernel keeps between calls give the same draws as recomputing them.
//
//  The reference implementations below recompute everything on every call, as the kernel did before it kept these
//  values; each comparison is exact (bit for bit) over the selected indices, the trees, the fits and the state of the
//  random number engine.
//   1. arn::set_wts + arn::discrete (the birth proposal's variable draw) against the normalisation and the
//      R::rmultinom draw done on every call: repeated, changed (also in place) and alternating weights, zero, signed
//      and degenerate weights, copies of the generator, other draws in between.
//   2. bart_cchange::cchange_all with one Work kept across calls (cumulative sums of pv kept while pv is unchanged,
//      the two candidate variables passed as a list) against the scan with the p-length mask and the sums recomputed
//      on every call; rule_weights with a candidate list against the mask.
//   3. bart::draw and heterbart::draw (the tree fits kept between draws, and written by the rebuild of allfit in
//      setdata, setm and refresh_allfit) against a draw loop that recomputes t_j(x) at the start of every tree update,
//      recomputes allfit after setdata, setm and refresh_allfit as predict() does, and uses the references of 1 and 2,
//      with every change of the kernel's inputs in between: setm, tonull, birth, death, edits through gettree with and
//      without refresh_allfit, sety, setdata (new buffer, same buffer with new values, another n, and new x values
//      before every draw), setxinfo, operator=, the copy constructor, DART (pv redrawn every sweep). The same script
//      runs with the fits kept, with no fit kept (limit 0) and with a limit that m n crosses in both directions; m n
//      equal to the default limit of 2^22 doubles keeps the fits and m n above it keeps none.
// ============================================================================
#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif
#include <armadillo>
#include "AI4BayesCode/bart_block.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

static int g_fail = 0;
static void check(bool ok, const char* what) { std::printf("  %-100s %s\n", what, ok ? "ok" : "FAILED"); if (!ok) g_fail++; }
static std::uint64_t engine_next() { return bart_rng::engine()(); }

// ---------------------------------------------------------------------------------------------------------------
// references: the generator's variable draw and the collapsed change, recomputed on every call
// ---------------------------------------------------------------------------------------------------------------
struct ref_arn : public arn {
  std::vector<double> w;
  void set_wts(std::vector<double>& in) override {
    double s = 0.; w.clear();
    for (size_t j = 0; j < in.size(); j++) s += in[j];
    for (size_t j = 0; j < in.size(); j++) w.push_back(in[j] / s);
  }
  size_t discrete() override {
    size_t p = w.size(), x = 0;
    std::vector<int> o(p, 0);
    R::rmultinom(1, &w[0], p, &o[0]);
    if (o[0] == 0) for (size_t j = 1; j < p; j++) x += j * o[j];
    return x;
  }
};

static int cchange_all_ref(tree& x, xinfo& xi, dinfo& di, pinfo& pi, std::vector<size_t>& nv, std::vector<double>& pv,
                           const int* kidx, const double* w, rn& gen)
{
  if (x.treesize() == 1) return 0;
  bart_cchange::Work W;
  const size_t p = pv.size();
  bart_cchange::Route& Rt = W.rt; Rt.build(x, xi, di);
  tree::npv all; x.getnodes(all);
  std::vector<size_t> reg, cv, cc; std::vector<double> lw; std::vector<int> la, ra; tree::npv leaves;
  std::vector<char> only(p, 0);
  std::vector<double> cum(p); double ssum = 0.0; for (size_t u = 0; u < p; u++) { ssum += pv[u]; cum[u] = ssum; }
  int changed = 0;
  for (size_t a = 0; a < all.size(); a++) {
    tree::tree_p eta = all[a];
    if (eta->getl() == 0) continue;
    const size_t v = eta->getv();
    size_t vp = p - 1;
    { const double u = gen.uniform() * ssum;
      const size_t q = (size_t)(std::upper_bound(cum.begin(), cum.end(), u) - cum.begin()); if (q < p) vp = q; }
    Rt.region_of(eta, reg);
    only[v] = 1; only[vp] = 1;
    bart_cchange::rule_weights(eta, reg, xi, di, pi, pv, kidx, w, lw, cv, cc, la, ra, leaves, &only, 0, &W);
    only[v] = 0; only[vp] = 0;
    size_t vn = v;
    if (vp != v) {
      const double lr = bart_cchange::log_ratio(lw, cv, v, vp, pv, 0);
      if (lr == std::numeric_limits<double>::infinity()) vn = vp;
      else if (lr > -std::numeric_limits<double>::infinity() && std::log(gen.uniform()) < lr) vn = vp;
    }
    std::vector<double> lwc; std::vector<size_t> ccc;
    for (size_t k = 0; k < lw.size(); k++) if (cv[k] == vn) { lwc.push_back(lw[k]); ccc.push_back(cc[k]); }
    if (lwc.empty()) continue;
    const size_t cnew = ccc[bart_cchange::draw_log(lwc, gen)];
    if (vn == eta->getv() && cnew == eta->getc()) continue;
    changed++;
    nv[eta->getv()]--; nv[vn]++; eta->setv(vn); eta->setc(cnew);
    std::vector<int> lid(leaves.size());
    for (size_t b = 0; b < leaves.size(); b++) { lid[b] = Rt.id_of(leaves[b]); Rt.mem[lid[b]].clear(); }
    const int* kv = kidx + vn * di.n;
    for (size_t k = 0; k < reg.size(); k++) { const size_t i = reg[k]; Rt.mem[lid[(kv[i] <= (int)cnew) ? la[k] : ra[k]]].push_back(i); }
  }
  return changed;
}

static std::string tree_str(const tree& t) { std::ostringstream os; os.precision(17); os << t; return os.str(); }

// ---------------------------------------------------------------------------------------------------------------
// 1. the variable draw of the birth proposal
// ---------------------------------------------------------------------------------------------------------------
// runs a scripted sequence of set_wts / discrete calls (plus other draws and a copy of the generator) and returns the
// indices drawn followed by the next engine value
template <class G>
static std::vector<std::uint64_t> arn_script(int script, unsigned seed)
{
  bart_rng::set_seed(seed);
  std::mt19937_64 g(seed + 99u); std::uniform_real_distribution<double> U(0.0, 1.0);
  G gen; std::vector<std::uint64_t> out;
  auto uni = [](size_t p) { return std::vector<double>(p, 1.0 / (double)p); };
  auto rnd = [&](size_t p) { std::vector<double> w(p); for (double& v : w) v = 0.01 + U(g); return w; };
  switch (script) {
  case 0: {   // uniform weights of several sizes, each repeated
    const size_t ps[] = {1, 2, 3, 10, 257, 2000};
    for (size_t p : ps) { std::vector<double> w = uni(p); for (int r = 0; r < 300; r++) { gen.set_wts(w); out.push_back(gen.discrete()); } }
    break; }
  case 1: {   // random weights, alternating between two vectors, between two sizes, and changed in place
    std::vector<double> a = rnd(50), b = rnd(50), c = rnd(7);
    for (int r = 0; r < 400; r++) {
      std::vector<double>& w = (r % 3 == 0) ? a : ((r % 3 == 1) ? b : c);
      gen.set_wts(w); out.push_back(gen.discrete()); out.push_back(gen.discrete());
      if (r % 10 == 4) a[(size_t)(U(g) * a.size())] = 0.01 + U(g);   // same object, new value
    }
    break; }
  case 2: {   // zero weights (leading, interior, trailing), one nonzero weight, signed weights, -0.0 against +0.0
    std::vector<std::vector<double> > ws = {
      {0.0, 0.0, 1.0, 2.0, 0.0}, {0.0, 0.0, 0.0, 0.0, 3.0}, {3.0, 0.0, 0.0, 0.0, 0.0}, {0.2, 0.8, 0.0, 0.0},
      {0.5, -0.25, 0.75}, {1.0, -3.0}, {0.0, 1.0}, {-0.0, 1.0}, {1e-300, 1.0, 1e-300}, {0.1, 0.2, 0.3, 0.4},
      {1.0 / 3, 1.0 / 3, 1.0 / 3, 0.0}};
    for (int r = 0; r < 600; r++) { std::vector<double>& w = ws[(size_t)(U(g) * ws.size())]; gen.set_wts(w); out.push_back(gen.discrete()); }
    break; }
  case 3: {   // other draws of the same engine in between, and copies of the generator taking over the stream
    std::vector<double> w = rnd(300);
    for (int r = 0; r < 200; r++) {
      gen.set_wts(w); out.push_back(gen.discrete());
      double u = gen.uniform(), z = gen.normal(); std::uint64_t b; std::memcpy(&b, &u, 8); out.push_back(b); std::memcpy(&b, &z, 8); out.push_back(b);
      if (r == 50) { G g2(gen); for (int k = 0; k < 20; k++) { g2.set_wts(w); out.push_back(g2.discrete()); } }
      if (r == 120) { G g3; g3 = gen; for (int k = 0; k < 20; k++) { out.push_back(g3.discrete()); g3.set_wts(w); } }
    }
    break; }
  }
  out.push_back(engine_next());
  return out;
}

static void part1()
{
  std::printf("1. variable draw of the birth proposal (arn::set_wts + arn::discrete) against the recomputed draw\n");
  const char* names[] = {"uniform weights, sizes 1 to 2000, repeated", "random weights: repeated, alternating, changed in place",
                         "zero, single, signed and degenerate weights", "other draws in between, copies of the generator"};
  for (int s = 0; s < 4; s++) {
    bool ok = true;
    for (unsigned seed = 1; seed <= 3; seed++) ok = ok && (arn_script<arn>(s, seed) == arn_script<ref_arn>(s, seed));
    check(ok, names[s]);
  }
}

// ---------------------------------------------------------------------------------------------------------------
// 2. the collapsed change
// ---------------------------------------------------------------------------------------------------------------
struct Data {
  size_t n, p; std::vector<double> x, y, w; xinfo xi; dinfo di; std::vector<int> kidx; pinfo pi;
  Data(size_t n_, size_t p_, unsigned seed) : n(n_), p(p_) {
    std::mt19937_64 g(seed); std::uniform_real_distribution<double> U(0.0, 1.0); std::normal_distribution<double> Z(0.0, 1.0);
    x.resize(n * p); y.resize(n); w.resize(n);
    for (double& v : x) v = U(g);
    for (size_t i = 0; i < n; i++) { y[i] = 2.0 * (x[i * p] > 0.5) + x[i * p + 1] + 0.5 * Z(g); w[i] = 0.5 + U(g); }
    xi.resize(p); for (size_t v = 0; v < p; v++) { const int nc = 4 + (int)(v % 5) * 3; xi[v].resize(nc); for (int j = 0; j < nc; j++) xi[v][j] = (j + 1.0) / (nc + 1.0); }
    di.n = n; di.p = p; di.x = x.data(); di.y = y.data();
    bart_cchange::make_kidx(xi, di, kidx);
    pi.alpha = 0.95; pi.mybeta = 1.0; pi.tau = 0.5;
  }
};

// a tree grown by births and changed by the reference scan, the same for every caller with the same seed
static void grow(tree& t, Data& D, std::vector<size_t>& nv, std::vector<double>& pv, unsigned seed)
{
  bart_rng::set_seed(seed); ref_arn gen;
  for (int k = 0; k < 25; k++) bd(t, D.xi, D.di, D.pi, 1.0, nv, pv, false, gen);
}

static void part2()
{
  std::printf("2. collapsed change with a Work kept across calls against the recomputed scan\n");
  // 2a. rule_weights: candidate list against the mask, every internal node and every pair of variables
  {
    Data D(160, 6, 3u); bool ok = true; long cmp = 0;
    for (unsigned rep = 0; rep < 6; rep++) {
      tree t; std::vector<size_t> nv(D.p, 0); std::vector<double> pv(D.p, 1.0 / D.p); grow(t, D, nv, pv, 100u + rep);
      bart_cchange::Route rt; rt.build(t, D.xi, D.di);
      tree::npv nodes; t.getnodes(nodes);
      bart_cchange::Work W1, W2;
      for (tree::tree_p eta : nodes) {
        if (!eta->getl()) continue;
        std::vector<size_t> reg; rt.region_of(eta, reg);
        for (size_t a = 0; a < D.p; a++) for (size_t b = a; b < D.p; b++) {
          std::vector<char> only(D.p, 0); only[a] = 1; only[b] = 1;
          std::vector<double> lw1, lw2; std::vector<size_t> cv1, cv2, cc1, cc2; std::vector<int> la1, la2, ra1, ra2; tree::npv lv1, lv2;
          bart_cchange::rule_weights(eta, reg, D.xi, D.di, D.pi, pv, D.kidx.data(), D.w.data(), lw1, cv1, cc1, la1, ra1, lv1, &only, 0, &W1);
          const size_t vars[2] = {a, b};
          bart_cchange::rule_weights(eta, reg, D.xi, D.di, D.pi, pv, D.kidx.data(), D.w.data(), lw2, cv2, cc2, la2, ra2, lv2, nullptr, 0, &W2, vars, a == b ? 1 : 2);
          ok = ok && lw1.size() == lw2.size() && std::memcmp(lw1.data(), lw2.data(), lw1.size() * sizeof(double)) == 0 &&
               cv1 == cv2 && cc1 == cc2 && la1 == la2 && ra1 == ra2 && lv1 == lv2;
          cmp++;
        }
      }
    }
    char msg[160]; std::snprintf(msg, sizeof msg, "rule_weights, candidate list against mask (%ld node and variable-pair cases)", cmp);
    check(ok, msg);
  }
  // 2b. cchange_all with one Work for every call (pv changed between calls, two data sets of different p) against the
  //     reference; same trees, split counts, number of changes and engine state
  {
    Data D1(200, 7, 5u), D2(120, 3, 6u);
    bart_cchange::Work W; bool ok = true; long calls = 0;
    std::mt19937_64 g(17); std::uniform_real_distribution<double> U(0.0, 1.0);
    std::vector<double> pv1(D1.p, 1.0 / D1.p), pv2(D2.p, 1.0 / D2.p);
    for (unsigned rep = 0; rep < 40; rep++) {
      Data& D = (rep % 4 == 3) ? D2 : D1; std::vector<double>& pv = (rep % 4 == 3) ? pv2 : pv1;
      if (rep % 5 == 2) { double s = 0; for (double& v : pv) { v = (U(g) < 0.15) ? 0.0 : 0.05 + U(g); s += v; } if (s == 0) pv[0] = 1; }
      if (rep % 7 == 6) pv[0] = std::nextafter(pv[0], 1.0);   // a change in the last bit only
      tree t1, t2; std::vector<size_t> nv1(D.p, 0), nv2(D.p, 0); std::vector<double> pvg(D.p, 1.0 / D.p);
      grow(t1, D, nv1, pvg, 200u + rep); grow(t2, D, nv2, pvg, 200u + rep);
      std::uint64_t e1 = 0, e2 = 0; int c1 = 0, c2 = 0;
      { bart_rng::set_seed(300u + rep); arn gen; for (int k = 0; k < 4; k++) c1 += bart_cchange::cchange_all(t1, D.xi, D.di, D.pi, nv1, pv, D.kidx.data(), D.w.data(), gen, 0, &W); e1 = engine_next(); }
      { bart_rng::set_seed(300u + rep); ref_arn gen; for (int k = 0; k < 4; k++) c2 += cchange_all_ref(t2, D.xi, D.di, D.pi, nv2, pv, D.kidx.data(), D.w.data(), gen); e2 = engine_next(); }
      ok = ok && tree_str(t1) == tree_str(t2) && nv1 == nv2 && c1 == c2 && e1 == e2;
      calls += 4;
    }
    char msg[160]; std::snprintf(msg, sizeof msg, "cchange_all with a kept Work, pv changed between calls, two data sets (%ld scans)", calls);
    check(ok, msg);
  }
}

// ---------------------------------------------------------------------------------------------------------------
// 3. the class kernels: library draw against the recomputing draw loop
// ---------------------------------------------------------------------------------------------------------------
struct ref_kernel : public heterbart {
  ref_kernel(size_t m_) : heterbart(m_) {}
  void dart_step(rn& gen) {
    if (dartOn) {
      if (grp) draw_s_grp(nv, lpv, theta, gen, grp, rho); else draw_s(nv, lpv, theta, gen);
      draw_theta0(const_theta, theta, lpv, a, b, rho, gen);
      for (size_t j = 0; j < p; j++) pv[j] = ::exp(lpv[j]);
    }
  }
  void draw_ref(double sigma, rn& gen) {   // bart::draw recomputing t_j(x) at the start of every tree update
    size_t i = 0;
    cchange_prepare_(nullptr, sigma, 1);
    for (size_t j = 0; j < m; j++) {
      fit3(t[j], xi, p, n, x, ftemp);
      for (size_t k = 0; k < n; k++) { allfit[k] = allfit[k] - ftemp[k]; r[k] = y[k] - allfit[k]; }
      if (bd(t[j], xi, di, pi, sigma, nv, pv, aug, gen)) i++;
      cchange_all_ref(t[j], xi, di, pi, nv, pv, cc_kidx_.data(), cc_w_.data(), gen);
      drmu(t[j], xi, di, pi, sigma, gen);
      fit3(t[j], xi, p, n, x, ftemp);
      for (size_t k = 0; k < n; k++) allfit[k] += ftemp[k];
    }
    accept = i / (double)m;
    dart_step(gen);
  }
  void allfit_ref() { if (allfit && x && xi.size() == p) predict(p, n, x, allfit); }   // allfit as predict() computes it
  void draw_ref(double* sigma, rn& gen, int shards = 1) {   // heterbart::draw, likewise
    size_t i = 0;
    cchange_prepare_(sigma, 0.0, shards);
    for (size_t j = 0; j < m; j++) {
      fit(t[j], xi, p, n, x, ftemp);
      for (size_t k = 0; k < n; k++) { allfit[k] = allfit[k] - ftemp[k]; r[k] = y[k] - allfit[k]; }
      if (heterbd(t[j], xi, di, pi, sigma, nv, pv, aug, gen, shards)) i++;
      cchange_all_ref(t[j], xi, di, pi, nv, pv, cc_kidx_.data(), cc_w_.data(), gen);
      heterdrmu(t[j], xi, di, pi, sigma, gen);
      fit(t[j], xi, p, n, x, ftemp);
      for (size_t k = 0; k < n; k++) allfit[k] += ftemp[k];
    }
    accept = i / (double)m;
    dart_step(gen);
  }
};

// the library kernel with the limit of kept fits set to C doubles (C = 0: no fit is kept; the copy keeps the limit)
template <size_t C> struct cap_kernel : public heterbart {
  cap_kernel(size_t m_) : heterbart(m_) { tfit_cap_ = C; }
  cap_kernel(const cap_kernel& o) : heterbart(o) { tfit_cap_ = C; }
  cap_kernel& operator=(const cap_kernel& o) { heterbart::operator=(o); return *this; }
};
// the library kernel, with the number of fits it keeps, the number of trees whose kept fit is valid, and its limit
struct probe_kernel : public heterbart {
  probe_kernel(size_t m_) : heterbart(m_) {}
  size_t kept() const { return tfit_.size(); }
  size_t valid() const { size_t c = 0; for (const auto& sg : tsig_) c += sg.empty() ? 0 : 1; return c; }
  size_t cap() const { return tfit_cap_; }
  void set_cap(size_t c) { tfit_cap_ = c; }
};

static void draw_one(heterbart& b, int s, double* sv, rn& gen) { if (s % 2) b.draw(0.8, gen); else b.draw(sv, gen); }
static void draw_one(ref_kernel& b, int s, double* sv, rn& gen) { if (s % 2) b.draw_ref(0.8, gen); else b.draw_ref(sv, gen); }
// after setdata, setm and refresh_allfit: the reference sets allfit as predict() computes it
static void after_rebuild(heterbart&) {}
static void after_rebuild(ref_kernel& b) { b.allfit_ref(); }

struct Hash {
  std::uint64_t h = 1469598103934665603ULL;
  void add(const void* q, size_t nb) { const unsigned char* c = (const unsigned char*)q; for (size_t i = 0; i < nb; i++) { h ^= c[i]; h *= 1099511628211ULL; } }
  void fits(const bart& b, size_t n) { for (size_t k = 0; k < n; k++) { const double f = b.f(k); add(&f, 8); } }
  void forest(heterbart& b, size_t m) { for (size_t j = 0; j < m; j++) { const std::string s = tree_str(b.gettree(j)); add(s.data(), s.size()); } }
};

// One scripted run: the kernel K (heterbart with the library draw, or ref_kernel with the reference draw) through every
// change of its inputs; returns the hash of the fits after every draw, the forests, split counts and probabilities and
// the engine state.
template <class K, class G>
static std::uint64_t kernel_script(unsigned seed)
{
  const size_t n = 180, p = 6;
  std::mt19937_64 g(seed); std::uniform_real_distribution<double> U(0.0, 1.0); std::normal_distribution<double> Z(0.0, 1.0);
  std::vector<double> x(n * p), y(n), x2(n * p), y2(n), sv(n);
  for (double& v : x) v = U(g);
  for (size_t i = 0; i < n; i++) { y[i] = 2.0 * (x[i * p] > 0.4) + x[i * p + 1] - 1.0 + 0.4 * Z(g); sv[i] = 0.4 + U(g); }
  bart_rng::set_seed(seed + 1000u); G gen; Hash h;
  size_t m = 12;
  auto setdata = [&](K& b, size_t nn, double* xx, double* yy) { b.setdata(p, nn, xx, yy, (size_t)40); after_rebuild(b); };
  K kb(m); kb.setprior(0.95, 2.0, 0.4); setdata(kb, n, x.data(), y.data());
  auto draw = [&](K& b, int k, size_t nn) { for (int s = 0; s < k; s++) { draw_one(b, s, sv.data(), gen); h.fits(b, nn); } };
  draw(kb, 20, n);
  kb.setm(20); after_rebuild(kb); m = 20; draw(kb, 8, n);
  kb.setm(9); after_rebuild(kb); m = 9; draw(kb, 8, n);
  { tree::npv b; kb.gettree(2).getbots(b); b[0]->settheta(b[0]->gettheta() + 0.3); }            // leaf value, no refresh
  draw(kb, 3, n);
  { tree::npv nd; kb.gettree(3).getnodes(nd); for (tree::tree_p q : nd) if (q->getl()) { q->setc(q->getc() / 2); break; } }   // rule, no refresh
  draw(kb, 3, n);
  { tree::npv b; kb.gettree(4).getbots(b); b.back()->settheta(-0.4); kb.refresh_allfit(); after_rebuild(kb); }   // with refresh
  draw(kb, 3, n);
  { tree::npv b; kb.gettree(5).getbots(b); b[0]->settheta(0.2); kb.refresh_allfit(); after_rebuild(kb); }      // refresh, then
  { tree::npv b; kb.gettree(5).getbots(b); b[0]->settheta(0.25); }                                            // an edit
  draw(kb, 3, n);
  for (size_t j = 0; j < m; j++) {                                                                // public birth / death
    tree& tj = kb.gettree(j);
    if (tj.treesize() == 1) kb.birth(j, 1, 1, 20, 0.1, -0.1); else { tree::npv ng; tj.getnogs(ng); kb.death(j, ng[0]->nid(), 0.05); }
  }
  draw(kb, 3, n);
  kb.tonull(); draw(kb, 4, n);
  for (size_t i = 0; i < n; i++) y2[i] = y[i] + 0.3 * Z(g);
  kb.sety(y2.data()); draw(kb, 3, n);
  x2 = x; for (int k = 0; k < 40; k++) x2[(size_t)(U(g) * n * p)] = U(g);
  setdata(kb, n, x2.data(), y2.data()); draw(kb, 3, n);                                           // new buffer
  for (int k = 0; k < 40; k++) x2[(size_t)(U(g) * n * p)] = U(g);
  setdata(kb, n, x2.data(), y2.data()); draw(kb, 3, n);                                           // same buffer, new values
  setdata(kb, 140, x2.data(), y2.data()); draw(kb, 3, 140);                                       // another n
  setdata(kb, n, x2.data(), y2.data()); draw(kb, 3, n);
  std::vector<double> x3(x2);
  for (int s = 0; s < 8; s++) {                                                                   // new x values before
    std::vector<double>& xs = (s % 3 == 2) ? x3 : x2;                                             // every draw (same and
    for (int k = 0; k < 15; k++) xs[(size_t)(U(g) * n * p)] = U(g);                               // another buffer)
    setdata(kb, n, xs.data(), y2.data()); draw(kb, 1, n);
    if (s == 4) { kb.setm(11); after_rebuild(kb); m = 11; draw(kb, 1, n); }
  }
  setdata(kb, n, x2.data(), y2.data()); draw(kb, 2, n);
  xinfo xi2 = kb.getxinfo(); for (auto& cv : xi2) for (double& c : cv) c += 0.006;
  kb.setxinfo(xi2); draw(kb, 3, n);                                                                // new cutpoints
  K kc(m); kc = kb; kc.setxinfo(xi2); kc.setprior(0.95, 2.0, 0.4); setdata(kc, n, x2.data(), y2.data());
  draw(kc, 4, n);                                                                                  // operator=
  K kd(kc); kd.setxinfo(xi2); kd.setprior(0.95, 2.0, 0.4); setdata(kd, n, x2.data(), y2.data());
  draw(kd, 4, n);                                                                                  // copy constructor
  kd.setdart(0.5, 1.0, (double)p, false, true); kd.startdart(); draw(kd, 8, n);                    // DART: pv redrawn
  kd.setdart(0.5, 1.0, (double)p, false, true, 1.0); draw(kd, 8, n);                             // DART, fixed theta
  h.forest(kb, m); h.forest(kd, m);
  for (double v : kd.getpv()) h.add(&v, 8);
  for (size_t c : kd.getnv()) h.add(&c, sizeof c);
  const std::uint64_t e = engine_next(); h.add(&e, 8);
  return h.h;
}

// m = 16 trees and n observations around the default limit of kept fits (m n = 2^22 doubles); kept = the number of fits
// the library kernel holds after the draws (0 for the reference)
static size_t kept_of(probe_kernel& b) { return b.kept(); }
static size_t kept_of(ref_kernel&) { return 0; }
template <class K, class G>
static std::uint64_t large_script(size_t n, size_t& kept)
{
  const size_t p = 2, m = 16;
  std::mt19937_64 g(23); std::uniform_real_distribution<double> U(0.0, 1.0);
  std::vector<double> x(n * p), y(n);
  for (double& v : x) v = U(g);
  for (size_t i = 0; i < n; i++) y[i] = (x[i * p] > 0.5 ? 1.0 : -1.0) + 0.5 * (U(g) - 0.5);
  bart_rng::set_seed(31); G gen; Hash h;
  K kb(m); kb.setprior(0.95, 2.0, 0.3); kb.setdata(p, n, x.data(), y.data(), (size_t)20); after_rebuild(kb);
  std::vector<double> sv(n, 0.5);
  for (int s = 0; s < 2; s++) { draw_one(kb, s, sv.data(), gen); h.fits(kb, n); }
  for (int k = 0; k < 64; k++) x[(size_t)(U(g) * n * p)] = U(g);
  kb.setdata(p, n, x.data(), y.data(), (size_t)20); after_rebuild(kb);                            // new x values
  draw_one(kb, 0, sv.data(), gen); h.fits(kb, n);
  kept = kept_of(kb);
  h.forest(kb, m); const std::uint64_t e = engine_next(); h.add(&e, 8);
  return h.h;
}

static void part3()
{
  std::printf("3. class kernels: draws with the kept tree fits against the recomputing draw loop\n");
  bool ok = true, ok0 = true, okm = true, okd = true;
  for (unsigned seed = 1; seed <= 3; seed++) {
    const std::uint64_t ref = kernel_script<ref_kernel, ref_arn>(seed);
    ok = ok && (kernel_script<heterbart, arn>(seed) == ref);
    ok0 = ok0 && (kernel_script<cap_kernel<0>, arn>(seed) == ref);
    okm = okm && (kernel_script<cap_kernel<3000>, arn>(seed) == ref);
    okd = okd && (kernel_script<cap_kernel<1500>, arn>(seed) == ref);
  }
  check(ok, "fits kept: setm, tonull, birth/death, gettree edits, sety, setdata (also before every draw), setxinfo, operator=, copy, DART (3 seeds)");
  check(ok0, "no fit kept (limit 0): the same script (3 seeds)");
  check(okm, "limit 3000 doubles, crossed by setm (12 -> 20 -> 9 trees at n = 180): the same script (3 seeds)");
  check(okd, "limit 1500 doubles, crossed by setdata (9 trees, n = 180 -> 140 -> 180): the same script (3 seeds)");
  { probe_kernel b(1); char msg[160]; std::snprintf(msg, sizeof msg, "default limit of kept fits: %zu doubles (2^22, 32 MiB)", b.cap()); check(b.cap() == (size_t(1) << 22), msg); }
  {   // which fits are valid after each call: the rebuild of allfit writes the fit of every tree
    const size_t n = 150, p = 4; std::mt19937_64 g(7); std::uniform_real_distribution<double> U(0.0, 1.0);
    std::vector<double> x(n * p), y(n); for (double& v : x) v = U(g); for (double& v : y) v = U(g) - 0.5;
    bart_rng::set_seed(5); arn gen;
    probe_kernel b(10); b.setprior(0.95, 2.0, 0.4); b.setdata(p, n, x.data(), y.data(), (size_t)30);
    std::vector<size_t> v; v.push_back(b.valid());                                                 // setdata
    b.draw(0.7, gen); b.draw(0.7, gen); v.push_back(b.valid());                                    // draws
    x[3] = 0.5; b.setdata(p, n, x.data(), y.data(), (size_t)30); v.push_back(b.valid());           // setdata, new x
    b.setm(14); v.push_back(b.valid());                                                            // setm
    { tree::npv bt; b.gettree(1).getbots(bt); bt[0]->settheta(0.1); } b.refresh_allfit(); v.push_back(b.valid());
    xinfo xi = b.getxinfo(); b.setxinfo(xi); v.push_back(b.valid());                               // setxinfo
    b.draw(0.7, gen); v.push_back(b.valid());
    probe_kernel c(14); c = b; v.push_back(c.valid());                                             // operator=
    b.set_cap(0); b.setdata(p, n, x.data(), y.data(), (size_t)30); v.push_back(b.valid()); v.push_back(b.kept());   // limit 0
    const std::vector<size_t> want = {10, 10, 10, 14, 14, 0, 14, 0, 0, 0};
    char msg[200]; std::snprintf(msg, sizeof msg, "valid kept fits after setdata, draws, setdata, setm, refresh_allfit, setxinfo, draw, operator=, limit 0: %zu %zu %zu %zu %zu %zu %zu %zu %zu",
                                 v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]);
    check(v == want, msg);
  }
  const size_t n_on = (size_t(1) << 22) / 16, n_off = n_on + 1;
  size_t k1 = 0, k2 = 0, kr = 0;
  const bool on = large_script<probe_kernel, arn>(n_on, k1) == large_script<ref_kernel, ref_arn>(n_on, kr);
  const bool off = large_script<probe_kernel, arn>(n_off, k2) == large_script<ref_kernel, ref_arn>(n_off, kr);
  char msg[160];
  std::snprintf(msg, sizeof msg, "m n = 2^22 (m 16, n %zu): fits kept (%zu doubles), draws equal", n_on, k1);
  check(on && k1 == 16 * n_on, msg);
  std::snprintf(msg, sizeof msg, "m n = 2^22 + 16 (m 16, n %zu): no fit kept (%zu doubles), draws equal", n_off, k2);
  check(off && k2 == 0, msg);
}

int main()
{
  part1();
  part2();
  part3();
  std::printf("%s\n", g_fail ? "SOME FAILED" : "ALL PASS");
  return g_fail ? 1 : 0;
}
