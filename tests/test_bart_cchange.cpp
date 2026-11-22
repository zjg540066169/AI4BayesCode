// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_bart_cchange.cpp -- the collapsed CHANGE move of BART (bart_pure_cpp/src/BART/cchange.h).
//
//  A. Exactness at a fixed topology. With the topology fixed and only the collapsed change running, the chain over
//     the rules of the internal nodes has the target restricted to that topology as its stationary law. The target is
//     computed by enumerating every rule assignment and routing the observations through tree::bn / tree::rg -- not
//     through the prefix sums of cchange.h. Two configurations, built so that every factor of the target varies:
//       A1: p = 3 with unequal cut grids (3, 4, 6 cutpoints) and unequal s (0.5, 0.3, 0.2);
//       A2: p = 1 with 6 cutpoints, so whether a leaf can split depends on the rules.
//     The correct move must agree with the target (total variation and a thinned chi-square test); each negative
//     control (one factor dropped, see cchange.h) must disagree.
//  B. Invariance of the full kernel (BIRTH/DEATH + collapsed change + leaf draws). One tree, sigma fixed: trees drawn
//     exactly from the posterior stay exactly distributed after 10 kernel iterations (20000 replicates), without
//     relying on mixing. BIRTH/DEATH alone does not mix across root rules on this data, so a long-run comparison
//     cannot serve as the reference.
//  C. The cutpoint-index cache is rebuilt after setdata, also when the same buffer holds new values.
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
#include <cstdio>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>
#include "portable_rng.hpp"

// ---------------------------------------------------------------------------------------------------------------
// shared helpers
// ---------------------------------------------------------------------------------------------------------------
static double chisq_upper_p(double chi2, double df)   // Wilson-Hilferty approximation of P(chi2_df > chi2)
{
    if (df <= 0) return 1.0;
    const double z = (std::cbrt(chi2 / df) - (1.0 - 2.0 / (9.0 * df))) / std::sqrt(2.0 / (9.0 * df));
    return 0.5 * std::erfc(z / std::sqrt(2.0));
}

struct Fit { double tv; double p; };

// chi-square p-value from observed and expected COUNTS. Bins with an expected count below 5 are pooled, and a pooled
// bin that still falls short of 5 is merged into the regular bin with the smallest expected count.
static double chisq_p_pooled(const std::vector<double>& e, const std::vector<double>& o)
{
    std::vector<double> E, O; double pe = 0.0, po = 0.0;
    for (size_t k = 0; k < e.size(); k++) {
        if (e[k] < 5.0) { pe += e[k]; po += o[k]; } else { E.push_back(e[k]); O.push_back(o[k]); }
    }
    if (pe > 0.0 || po > 0.0) {
        if (pe >= 5.0 || E.empty()) { E.push_back(pe); O.push_back(po); }
        else { const size_t j = (size_t)(std::min_element(E.begin(), E.end()) - E.begin()); E[j] += pe; O[j] += po; }
    }
    double chi2 = 0.0;
    for (size_t k = 0; k < E.size(); k++) chi2 += (O[k] - E[k]) * (O[k] - E[k]) / E[k];
    return chisq_upper_p(chi2, (double)E.size() - 1.0);
}


// TV distance of all post-burn-in states, and a chi-square test on every `thin`-th state
static Fit compare(const std::vector<double>& target, const std::vector<long>& counts_all,
                   const std::vector<long>& counts_thin, long n_all, long n_thin)
{
    double tv = 0.0;
    for (size_t k = 0; k < target.size(); k++) tv += std::fabs((double)counts_all[k] / n_all - target[k]);
    tv *= 0.5;
    std::vector<double> e(target.size()), o(target.size());
    for (size_t k = 0; k < target.size(); k++) { e[k] = target[k] * n_thin; o[k] = (double)counts_thin[k]; }
    return {tv, chisq_p_pooled(e, o)};
}

// ---------------------------------------------------------------------------------------------------------------
// A. exactness at a fixed topology: root (node 1) and its left child (node 2) are internal; leaves 4, 5, 3
// ---------------------------------------------------------------------------------------------------------------
struct Setup {
    size_t n, p;
    std::vector<double> x, y, w;      // x row-major (n x p)
    xinfo xi;
    pinfo pi;
    std::vector<double> pv;
};

static Setup make_setup(size_t p, const std::vector<int>& ncuts, const std::vector<double>& pv, unsigned seed)
{
    Setup S; S.n = 120; S.p = p; S.pv = pv;
    std::mt19937_64 g(seed);
    S.x.resize(S.n * p); S.y.resize(S.n); S.w.assign(S.n, 1.0);   // sigma = 1
    for (size_t i = 0; i < S.n; i++) {
        for (size_t v = 0; v < p; v++) S.x[i * p + v] = prng::u01(g);
        const double x0 = S.x[i * p], x1 = (p > 1) ? S.x[i * p + 1] : 0.0;
        S.y[i] = 0.9 * (x0 > 0.45 ? 1.0 : 0.0) + 0.6 * x1 + prng::n01(g);
    }
    S.xi.resize(p);
    for (size_t v = 0; v < p; v++) {
        S.xi[v].resize(ncuts[v]);
        for (int j = 0; j < ncuts[v]; j++) S.xi[v][j] = (j + 1.0) / (ncuts[v] + 1.0);
    }
    S.pi.alpha = 0.95; S.pi.mybeta = 2.0; S.pi.tau = 0.6;
    return S;
}

static int range_of(tree::tree_p nd, size_t v, xinfo& xi, int& L, int& U)
{
    L = 0; U = (int)xi[v].size() - 1; nd->rg(v, &L, &U); return U - L + 1;
}

// the target over rule assignments (v1, c1, v2, c2), by enumeration
static void enumerate_target(Setup& S, std::vector<std::tuple<size_t, size_t, size_t, size_t>>& states,
                             std::vector<double>& prob)
{
    tree t; t.birth(1, 0, 0, 0.0, 0.0); t.birth(2, 0, 0, 0.0, 0.0);
    tree::tree_p root = t.getptr(1), n2 = t.getptr(2);
    std::vector<double> lw;
    for (size_t v1 = 0; v1 < S.p; v1++) for (size_t c1 = 0; c1 < S.xi[v1].size(); c1++)
    for (size_t v2 = 0; v2 < S.p; v2++) for (size_t c2 = 0; c2 < S.xi[v2].size(); c2++) {
        root->setv(v1); root->setc(c1); n2->setv(v2); n2->setc(c2);
        int L, U; const int N2 = range_of(n2, v2, S.xi, L, U);
        if ((int)c2 < L || (int)c2 > U) continue;                        // inadmissible rule at node 2
        tree::npv bots; t.getbots(bots);
        std::vector<double> nn(bots.size(), 0.0), bb(bots.size(), 0.0), MM(bots.size(), 0.0);
        for (size_t i = 0; i < S.n; i++) {
            tree::tree_p b = t.bn(&S.x[i * S.p], S.xi);
            size_t a = 0; while (bots[a] != b) a++;
            nn[a] += 1.0; bb[a] += S.w[i]; MM[a] += S.w[i] * S.y[i];
        }
        bool ok = true; for (double c : nn) if (c < 5.0) ok = false;
        if (!ok) continue;                                               // a leaf below 5 observations
        double l = std::log(S.pv[v1]) - std::log((double)S.xi[v1].size()) + std::log(S.pv[v2]) - std::log((double)N2);
        for (size_t a = 0; a < bots.size(); a++) {
            bool can = false;
            for (size_t u = 0; u < S.p && !can; u++) { int Lu, Uu; if (range_of(bots[a], u, S.xi, Lu, Uu) >= 1) can = true; }
            if (can) l += std::log1p(-S.pi.alpha / std::pow(1.0 + bots[a]->depth(), S.pi.mybeta));
            l += bart_cchange::leaf_lik(bb[a], MM[a], S.pi.tau);
        }
        states.emplace_back(v1, c1, v2, c2); lw.push_back(l);
    }
    double mx = lw[0]; for (double l : lw) mx = std::max(mx, l);
    double s = 0.0; prob.resize(lw.size());
    for (size_t k = 0; k < lw.size(); k++) { prob[k] = std::exp(lw[k] - mx); s += prob[k]; }
    for (double& q : prob) q /= s;
}

static Fit run_chain(Setup& S, const std::vector<std::tuple<size_t, size_t, size_t, size_t>>& states,
                     const std::vector<double>& target, int ctl, unsigned seed)
{
    std::map<std::tuple<size_t, size_t, size_t, size_t>, size_t> idx;
    for (size_t k = 0; k < states.size(); k++) idx[states[k]] = k;
    // start from the most probable state
    size_t k0 = 0; for (size_t k = 1; k < target.size(); k++) if (target[k] > target[k0]) k0 = k;
    tree t; t.birth(1, std::get<0>(states[k0]), std::get<1>(states[k0]), 0.0, 0.0);
    t.birth(2, std::get<2>(states[k0]), std::get<3>(states[k0]), 0.0, 0.0);
    dinfo di; di.n = S.n; di.p = S.p; di.x = S.x.data(); di.y = S.y.data();
    std::vector<int> kidx; bart_cchange::make_kidx(S.xi, di, kidx);
    std::vector<size_t> nv(S.p, 0); nv[std::get<0>(states[k0])]++; nv[std::get<2>(states[k0])]++;
    std::vector<double> pv = S.pv;
    bart_rng::set_seed(seed);
    arn gen;
    const long burn = 2000, N = 400000, thin = 20;
    std::vector<long> all(states.size(), 0), thinned(states.size(), 0);
    long n_thin = 0;
    for (long it = -burn; it < N; it++) {
        bart_cchange::cchange_all(t, S.xi, di, S.pi, nv, pv, kidx.data(), S.w.data(), gen, ctl);
        if (it < 0) continue;
        tree::tree_p r = t.getptr(1), c = t.getptr(2);
        auto f = idx.find(std::make_tuple(r->getv(), r->getc(), c->getv(), c->getc()));
        if (f == idx.end()) { std::printf("    visited a state outside the target's support\n"); return {1.0, 0.0}; }
        all[f->second]++;
        if (it % thin == 0) { thinned[f->second]++; n_thin++; }
    }
    return compare(target, all, thinned, N, n_thin);
}

static bool part_A(const char* label, Setup S, const std::vector<int>& must_fail, unsigned seed)
{
    std::vector<std::tuple<size_t, size_t, size_t, size_t>> states; std::vector<double> target;
    enumerate_target(S, states, target);
    std::printf("A %s: %zu admissible rule states\n", label, states.size());
    bool ok = true;
    const Fit f0 = run_chain(S, states, target, 0, seed);
    const bool pass0 = f0.tv < 0.02 && f0.p > 1e-3;
    std::printf("    correct move      TV %.4f  chi-square p %.3g  -> %s\n", f0.tv, f0.p, pass0 ? "PASS" : "FAIL");
    ok = ok && pass0;
    for (int ctl : must_fail) {
        const Fit f = run_chain(S, states, target, ctl, seed);
        const bool detected = f.tv > 0.04 || f.p < 1e-6;
        std::printf("    control %d         TV %.4f  chi-square p %.3g  -> %s\n", ctl, f.tv, f.p,
                    detected ? "detected (PASS)" : "NOT detected (FAIL)");
        ok = ok && detected;
    }
    return ok;
}

// ---------------------------------------------------------------------------------------------------------------
// B. Invariance of the full kernel, from exact draws. One tree, sigma = 1 fixed. The exact posterior over trees
//    (every leaf >= 5 observations) comes from a sum-product recursion over node regions (the cutpoint-index bounds
//    [lo_v, hi_v] of every variable). Each replicate draws a tree EXACTLY from that posterior (top-down from the
//    recursion), applies 10 iterations of the kernel -- bd() and drmu() from the BART kernel, with or without the
//    collapsed change between them -- and records the final number of leaves and root rule. If the kernel leaves the
//    posterior invariant, the final states are exact draws as well, whatever the mixing; 20000 independent replicates
//    give a chi-square test. A negative control (the collapsed change with uniform cutpoints) must fail.
// ---------------------------------------------------------------------------------------------------------------
struct ExactB {
    size_t n, p; const std::vector<int>* kidx; const double* y; double tau, alpha, beta; std::vector<double> s; int Kmax;
    struct Res { std::vector<double> Zk; double Z = 0.0; };
    std::map<std::vector<int>, Res> memo;
    double pgrow(const std::vector<int>& lo, const std::vector<int>& hi, int depth) const {
        bool can = false; for (size_t v = 0; v < p; v++) if (hi[v] - lo[v] >= 1) can = true;
        return can ? alpha / std::pow(1.0 + depth, beta) : 0.0;
    }
    const Res& solve(const std::vector<int>& lo, const std::vector<int>& hi, int depth)
    {
        std::vector<int> key = lo; key.insert(key.end(), hi.begin(), hi.end()); key.push_back(depth);
        auto it = memo.find(key); if (it != memo.end()) return it->second;
        Res r; r.Zk.assign(Kmax + 1, 0.0);
        double nn = 0.0, M = 0.0;
        for (size_t i = 0; i < n; i++) {
            bool in = true;
            for (size_t v = 0; v < p && in; v++) { const int k = (*kidx)[v * n + i]; if (k < lo[v] || k > hi[v]) in = false; }
            if (in) { nn += 1.0; M += y[i]; }                              // w_i = 1 (sigma = 1)
        }
        const double pg = pgrow(lo, hi, depth);
        if (nn >= 5) r.Zk[1] = (1.0 - pg) * std::exp(bart_cchange::leaf_lik(nn, M, tau));
        if (pg > 0.0 && nn >= 10) {
            for (size_t v = 0; v < p; v++) {
                const int N = hi[v] - lo[v]; if (N < 1) continue;         // admissible cutpoints c in [lo_v, hi_v - 1]
                for (int c = lo[v]; c <= hi[v] - 1; c++) {
                    std::vector<int> hl = hi, ll = lo; hl[v] = c; ll[v] = c + 1;
                    const Res L = solve(lo, hl, depth + 1);
                    const Res R = solve(ll, hi, depth + 1);
                    if (L.Z == 0.0 || R.Z == 0.0) continue;
                    const double f = pg * s[v] / N;
                    for (int k1 = 1; k1 <= Kmax; k1++) if (L.Zk[k1] > 0.0)
                        for (int k2 = 1; k1 + k2 <= Kmax; k2++) r.Zk[k1 + k2] += f * L.Zk[k1] * R.Zk[k2];
                }
            }
        }
        for (int k = 1; k <= Kmax; k++) r.Z += r.Zk[k];
        return memo.emplace(key, r).first->second;
    }
    // one exact draw of the subtree at `node`, top-down
    void sample(tree::tree_p node, const std::vector<int>& lo, const std::vector<int>& hi, int depth, std::mt19937_64& g)
    {
        const Res me = solve(lo, hi, depth);
        double u = prng::u01(g) * me.Z;
        if ((u -= me.Zk[1]) < 0.0) return;                                 // leaf
        const double pg = pgrow(lo, hi, depth);
        size_t lv = 0; int lc = -1; std::vector<int> lhl, lll;
        for (size_t v = 0; v < p; v++) {
            const int N = hi[v] - lo[v]; if (N < 1) continue;
            for (int c = lo[v]; c <= hi[v] - 1; c++) {
                std::vector<int> hl = hi, ll = lo; hl[v] = c; ll[v] = c + 1;
                const double w = pg * s[v] / N * solve(lo, hl, depth + 1).Z * solve(ll, hi, depth + 1).Z;
                if (w <= 0.0) continue;
                lv = v; lc = c; lhl = hl; lll = ll;                        // last admissible split (rounding fallback)
                if ((u -= w) < 0.0) { node->birthp(node, v, (size_t)c, 0.0, 0.0);
                    sample(node->getl(), lo, hl, depth + 1, g); sample(node->getr(), ll, hi, depth + 1, g); return; }
            }
        }
        if (lc >= 0) { node->birthp(node, lv, (size_t)lc, 0.0, 0.0);
            sample(node->getl(), lo, lhl, depth + 1, g); sample(node->getr(), lll, hi, depth + 1, g); }
    }
};

static bool part_B()
{
    const size_t n = 40, p = 2; const int nc = 6; const double tau = 0.8; std::mt19937_64 g(7u);
    std::vector<double> x(n * p), y(n);
    for (size_t i = 0; i < n; i++) {
        x[i * p] = prng::u01(g); x[i * p + 1] = prng::u01(g);
        y[i] = 1.5 * (x[i * p] > 0.5 ? 1.0 : 0.0) + 1.0 * x[i * p + 1] + prng::n01(g);
    }
    xinfo xi(p); for (size_t v = 0; v < p; v++) { xi[v].resize(nc); for (int j = 0; j < nc; j++) xi[v][j] = (j + 1.0) / (nc + 1.0); }
    dinfo di; di.n = n; di.p = p; di.x = x.data(); di.y = y.data();
    std::vector<int> kidx; bart_cchange::make_kidx(xi, di, kidx);
    std::vector<double> w(n, 1.0);
    pinfo pi; pi.alpha = 0.95; pi.mybeta = 2.0; pi.tau = tau;
    ExactB E; E.n = n; E.p = p; E.kidx = &kidx; E.y = y.data(); E.tau = tau; E.alpha = pi.alpha; E.beta = pi.mybeta;
    E.s.assign(p, 1.0 / p); E.Kmax = (int)(n / 5);
    const std::vector<int> lo0(p, 0), hi0(p, nc);
    const ExactB::Res root = E.solve(lo0, hi0, 0);
    // exact distribution of (number of leaves) and of the root rule (index 0 = no split)
    std::vector<double> eK(E.Kmax + 1, 0.0), eR(1 + p * nc, 0.0);
    for (int k = 1; k <= E.Kmax; k++) eK[k] = root.Zk[k] / root.Z;
    eR[0] = root.Zk[1] / root.Z;
    for (size_t v = 0; v < p; v++) for (int c = 0; c < nc; c++) {
        std::vector<int> hl = hi0, ll = lo0; hl[v] = c; ll[v] = c + 1;
        eR[1 + v * nc + c] = E.pgrow(lo0, hi0, 0) * E.s[v] / nc * E.solve(lo0, hl, 1).Z * E.solve(ll, hi0, 1).Z / root.Z;
    }
    auto chisq_p = [](const std::vector<double>& e, const std::vector<double>& o, double R) {
        std::vector<double> ec(e.size()); for (size_t k = 0; k < e.size(); k++) ec[k] = e[k] * R;
        return chisq_p_pooled(ec, o);
    };
    std::printf("B invariance from exact draws (one tree, n = %zu, p = %zu; exact posterior over %zu node regions):\n",
                n, p, E.memo.size());
    std::printf("    exact P(leaves = 1..5): "); for (int k = 1; k <= 5; k++) std::printf(" %.4f", eK[k]); std::printf("\n");
    const long R = 20000; const int K = 10;
    struct Run { const char* name; int ncc; int ctl; bool must_pass; };
    const Run runs[] = { {"BD only", 0, 0, true}, {"BD + change", 1, 0, true}, {"control 5 (uniform cutpoint)", 1, 5, false} };
    bool ok = true;
    for (const Run& run : runs) {
        std::mt19937_64 gs(20261122u); bart_rng::set_seed(77u); arn gen;
        std::vector<double> oK(E.Kmax + 1, 0.0), oR(1 + p * nc, 0.0);
        for (long r = 0; r < R; r++) {
            tree t; E.sample(&t, lo0, hi0, 0, gs);
            std::vector<size_t> nv(p, 0); { tree::npv nds; t.getnodes(nds); for (auto nd : nds) if (nd->getl()) nv[nd->getv()]++; }
            std::vector<double> pv(p, 1.0 / p);
            for (int k = 0; k < K; k++) {
                bd(t, xi, di, pi, 1.0, nv, pv, false, gen);
                for (int q = 0; q < run.ncc; q++) bart_cchange::cchange_all(t, xi, di, pi, nv, pv, kidx.data(), w.data(), gen, run.ctl);
                drmu(t, xi, di, pi, 1.0, gen);
            }
            oK[std::min<size_t>(t.nbots(), E.Kmax)] += 1.0;
            oR[t.treesize() == 1 ? 0 : 1 + t.getv() * nc + t.getc()] += 1.0;
        }
        const double pK = chisq_p(eK, oK, R), pR = chisq_p(eR, oR, R);
        const bool pass = pK > 1e-3 && pR > 1e-3;
        std::printf("    %-30s leaves p = %.3g, root rule p = %.3g -> %s\n", run.name, pK, pR,
                    run.must_pass ? (pass ? "invariant (PASS)" : "NOT invariant (FAIL)")
                                  : (pass ? "NOT detected (FAIL)" : "detected (PASS)"));
        ok = ok && (run.must_pass ? pass : !pass);
    }
    return ok;
}

// ---------------------------------------------------------------------------------------------------------------
// C. cache rebuild after setdata
// ---------------------------------------------------------------------------------------------------------------
struct probe : public heterbart {
    probe(size_t m) : heterbart(m) {}
    const std::vector<int>& kidx() const { return cc_kidx_; }
    dinfo& dinf() { return di; }
};

static bool part_C()
{
    const size_t n = 50, p = 3; std::mt19937_64 g(11u);
    std::vector<double> x(n * p), y(n), sig(n, 1.0);
    for (double& v : x) v = prng::u01(g);
    for (double& v : y) v = prng::n01(g);
    xinfo xi(p); for (size_t v = 0; v < p; v++) { xi[v].resize(9); for (int j = 0; j < 9; j++) xi[v][j] = (j + 1) / 10.0; }
    std::vector<int> nc(p, 9);
    bart_rng::set_seed(5u); arn gen;
    probe hb(5); hb.setxinfo(xi); hb.setprior(0.95, 2.0, 0.5); hb.setdata(p, n, x.data(), y.data(), nc.data());
    bool ok = true;
    auto fresh = [&](void) { std::vector<int> k; bart_cchange::make_kidx(xi, hb.dinf(), k); return k; };
    hb.draw(sig.data(), gen);
    const bool c1 = (hb.kidx() == fresh());
    std::vector<double> x2(n * p); for (double& v : x2) v = prng::u01(g);        // a new buffer
    hb.setdata(p, n, x2.data(), y.data(), nc.data()); hb.draw(sig.data(), gen);
    const bool c2 = (hb.kidx() == fresh());
    for (double& v : x2) v = prng::u01(g);                                        // same buffer, new values
    hb.setdata(p, n, x2.data(), y.data(), nc.data()); hb.draw(sig.data(), gen);
    const bool c3 = (hb.kidx() == fresh());
    std::printf("C cache rebuilt: first draw %s, new buffer %s, same buffer with new values %s\n",
                c1 ? "yes" : "NO", c2 ? "yes" : "NO", c3 ? "yes" : "NO");
    ok = c1 && c2 && c3;
    std::printf("    -> %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

int main()
{
    bool ok = true;
    ok = part_A("A1 (p = 3, cut grids 3/4/6, s = .5/.3/.2)", make_setup(3, {3, 4, 6}, {0.5, 0.3, 0.2}, 31u),
                {1, 2, 4, 5}, 1001u) && ok;
    ok = part_A("A2 (p = 1, 6 cutpoints)", make_setup(1, {6}, {1.0}, 32u), {1, 3, 5}, 1002u) && ok;
    ok = part_B() && ok;
    ok = part_C() && ok;
    std::printf("%s\n", ok ? "ALL PASS" : "SOME FAILED");
    return ok ? 0 : 1;
}
