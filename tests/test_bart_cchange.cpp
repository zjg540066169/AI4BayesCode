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
//  C. State kept by the kernel. After setdata or setxinfo the cutpoint index equals a full rebuild (new buffer, same
//     buffer with new values, a few changed entries, new cutpoints), and sety keeps it; bart_model::set_Y (y only)
//     and set_data give the same chain when X is unchanged; the split counts nv match a forest loaded by set_tree
//     and stay current over later draws; an X with a different number of columns is an error in
//     bart_model::set_data, bart_block::set_data and the kernel.
//  D. Exact-start invariance of one scan at a topology with an internal grandchild, with heterogeneous observation
//     weights and with a flat likelihood (where the cutpoint counts of internal descendants dominate the target).
//  E. The class kernels bart::draw and heterbart::draw pass the right observation weights to the move and draw the
//     leaf values after it, for the final rules of the draw.
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
// C. state kept by the kernel: the cutpoint-index cache, the split counts nv, and the dimension checks
// ---------------------------------------------------------------------------------------------------------------
struct probe : public heterbart {
    probe(size_t m) : heterbart(m) {}
    const std::vector<int>& kidx() const { return cc_kidx_; }
    bool kidx_ok() const { return cc_kidx_ok_; }
    dinfo& dinf() { return di; }
};

// split counts of a serialized forest ("T\n<tree>---\n..."), counted from its internal nodes
static arma::uvec forest_counts(const std::string& s, size_t p)
{
    std::istringstream is(s); long T = 0; is >> T;
    arma::uvec cnt(p, arma::fill::zeros);
    for (long j = 0; j < T; j++) {
        tree t; std::string sep; is >> t >> sep;
        tree::npv nds; t.getnodes(nds);
        for (tree::tree_p nd : nds) if (nd->getl()) cnt(nd->getv())++;
    }
    return cnt;
}

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
    auto fresh = [&](void) { std::vector<int> k; bart_cchange::make_kidx(hb.getxinfo(), hb.dinf(), k); return k; };
    hb.draw(sig.data(), gen);
    const bool c1 = (hb.kidx() == fresh());
    std::vector<double> x2(n * p); for (double& v : x2) v = prng::u01(g);        // a new buffer
    hb.setdata(p, n, x2.data(), y.data(), nc.data()); hb.draw(sig.data(), gen);
    const bool c2 = (hb.kidx() == fresh());
    for (double& v : x2) v = prng::u01(g);                                        // same buffer, new values
    hb.setdata(p, n, x2.data(), y.data(), nc.data()); hb.draw(sig.data(), gen);
    const bool c3 = (hb.kidx() == fresh());
    for (size_t i = 0; i < n; i += 5) x2[i * p] = prng::u01(g);                // same buffer, a few entries change
    hb.setdata(p, n, x2.data(), y.data(), nc.data()); hb.draw(sig.data(), gen);
    const bool c3b = (hb.kidx() == fresh());
    xinfo xi2 = xi; for (auto& cv : xi2) for (double& u : cv) u += 0.035;         // new cutpoints, same x
    hb.setxinfo(xi2); hb.draw(sig.data(), gen);
    const bool c3c = (hb.kidx() == fresh());
    std::vector<double> y2(n); for (double& v : y2) v = prng::n01(g);
    hb.sety(y2.data());                                                           // y only: the cache stays valid
    const bool c4 = hb.kidx_ok();
    hb.setdata(p, n, x2.data(), y2.data(), nc.data());                            // setdata: the cache is dropped
    const bool c4b = !hb.kidx_ok();
    std::printf("C cache equals a full rebuild: first draw %s, new buffer %s, same buffer with new values %s, a few entries "
                "changed %s, new cutpoints %s; kept by sety %s, dropped by setdata %s\n",
                c1 ? "yes" : "NO", c2 ? "yes" : "NO", c3 ? "yes" : "NO", c3b ? "yes" : "NO", c3c ? "yes" : "NO",
                c4 ? "yes" : "NO", c4b ? "yes" : "NO");

    // bart_model: with X unchanged, set_Y (y only) and set_data (X and y) give the same chain
    const long N = 200, P = 5, K = 40;
    std::mt19937_64 gx(3u);
    arma::mat X(N, P); for (arma::uword i = 0; i < X.n_elem; i++) X(i) = prng::u01(gx);
    std::vector<arma::vec> ys(K, arma::vec(N));
    for (long k = 0; k < K; k++) for (long i = 0; i < N; i++) ys[k](i) = 2.0 * X(i, 0) - X(i, 1) + 0.5 * prng::n01(gx);
    auto chain = [&](bool via_set_data) {
        bart_rng::set_seed(17u);
        stdbart::bart_model mdl(X, ys[0], 100L, false, false, false, /*ntrees=*/20);
        mdl.set_history(false);
        for (long k = 0; k < K; k++) { if (via_set_data) mdl.set_data(X, ys[k]); else mdl.set_Y(ys[k]); mdl.update_step(1.0); }
        return arma::vec(mdl.current_fit_train());
    };
    const double dfit = arma::abs(chain(false) - chain(true)).max();
    const bool c5 = dfit < 1e-8;
    std::printf("  bart_model set_Y vs set_data, X unchanged, %ld draws: max |fit difference| %.1e -> %s\n",
                K, dfit, c5 ? "same chain" : "DIFFERENT");

    // bart_model::set_tree: nv equals the split counts of the loaded forest and stays equal over later draws
    bart_rng::set_seed(29u);
    stdbart::bart_model src(X, ys[0], 100L, false, false, false, 20); src.set_history(false);
    for (long k = 0; k < K; k++) { src.set_Y(ys[k]); src.update_step(1.0); }
    const std::string forest = src.get_tree();
    stdbart::bart_model dst(X, ys[0], 100L, false, false, false, 20); dst.set_history(false);
    dst.set_tree(forest);
    const arma::uword n_split = arma::accu(forest_counts(forest, P));
    bool c6 = n_split > 0 && arma::all(dst.current_var_counts() == forest_counts(forest, P));
    for (long k = 0; k < K && c6; k++) {
        dst.set_Y(ys[k]); dst.update_step(1.0);
        c6 = arma::all(dst.current_var_counts() == forest_counts(dst.get_tree(), P));
    }
    std::printf("  split counts after set_tree (%llu splits loaded) and %ld draws match the forest: %s\n",
                (unsigned long long)n_split, K, c6 ? "yes" : "NO");

    // an X with a different number of columns is an error, not an out-of-bounds access
    arma::mat X6(N, P + 1); for (arma::uword i = 0; i < X6.n_elem; i++) X6(i) = prng::u01(gx);
    bool c7m = false, c7b = false, c8 = false;
    try { dst.set_data(X6, ys[0]); } catch (const std::invalid_argument&) { c7m = true; }
    {
        AI4BayesCode::bart_block_config cfg;
        cfg.name = "f"; cfg.x_train = X; cfg.y_init = ys[0]; cfg.ntrees = 10; cfg.sigma_init = 1.0;
        AI4BayesCode::bart_block blk(cfg);
        try { blk.set_data(X6, ys[0]); } catch (const std::exception&) { c7b = true; }
    }
    {
        probe hb3(5); hb3.setxinfo(xi); hb3.setprior(0.95, 2.0, 0.5);       // cutpoints for p = 3, data with p + 1
        std::vector<double> x4(n * (p + 1)); for (double& v : x4) v = prng::u01(g);
        std::vector<int> nc4(p + 1, 9);
        hb3.setdata(p + 1, n, x4.data(), y.data(), nc4.data());
        try { hb3.draw(sig.data(), gen); } catch (const std::invalid_argument&) { c8 = true; }
    }
    std::printf("  X with another column count is an error: bart_model::set_data %s, bart_block::set_data %s, kernel %s\n",
                c7m ? "yes" : "NO", c7b ? "yes" : "NO", c8 ? "yes" : "NO");

    const bool ok = c1 && c2 && c3 && c3b && c3c && c4 && c4b && c5 && c6 && c7m && c7b && c8;
    std::printf("    -> %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------------------------------------------------------
// D. Exact-start invariance at a deeper fixed topology: internal root 1, left child 2 and left grandchild 4 (leaves
//    8, 9, 5, 3), so that the cutpoint counts of internal descendants two levels below a node enter its factors.
//    Rule assignments are drawn exactly from the enumerated target (s_v / N_eta(v) of every internal node, the
//    log(1 - pgrow) of every leaf that can split, >= 5 observations per leaf, and the leaf marginals, routed through
//    tree::bn / tree::rg), one scan of the collapsed change is applied, and the final assignments are compared with
//    the target. D1 uses heterogeneous observation weights (1/sigma_i^2 in [0.25, 3.25]); D2 a flat likelihood
//    (w = 1e-9), where the target is the tree prior on this topology and the cutpoint-count factors dominate it.
//    Negative control 1 (descendant cutpoint counts dropped) must fail under the flat likelihood.
// ---------------------------------------------------------------------------------------------------------------
static double leaf_marginal(double b, double M, double tau)   // log of the leaf marginal, written out independently
{
    const double k = 1.0 + b * tau * tau;
    return -0.5 * std::log(k) + 0.5 * M * M * tau * tau / k;
}

static Fit part_D_run(bool flat, int ctl, long R, unsigned seed, size_t& n_states)
{
    const size_t n = 300, p = 2; const std::vector<int> ncuts = {4, 6}; std::vector<double> pv = {0.6, 0.4};
    std::mt19937_64 g(41u);
    std::vector<double> x(n * p), y(n), w(n);
    for (size_t i = 0; i < n; i++) {
        for (size_t v = 0; v < p; v++) x[i * p + v] = prng::u01(g);
        y[i] = 1.2 * (x[i * p] > 0.45 ? 1.0 : 0.0) + 0.8 * x[i * p + 1] + prng::n01(g);
        w[i] = flat ? 1e-9 : 0.25 + 3.0 * prng::u01(g);
    }
    xinfo xi(p);
    for (size_t v = 0; v < p; v++) { xi[v].resize(ncuts[v]); for (int j = 0; j < ncuts[v]; j++) xi[v][j] = (j + 1.0) / (ncuts[v] + 1.0); }
    pinfo pi; pi.alpha = 0.95; pi.mybeta = 2.0; pi.tau = 0.6;
    tree t; t.birth(1, 0, 0, 0.0, 0.0); t.birth(2, 0, 0, 0.0, 0.0); t.birth(4, 0, 0, 0.0, 0.0);
    tree::tree_p nd[3] = { t.getptr(1), t.getptr(2), t.getptr(4) };
    typedef std::vector<size_t> St;                               // (v1, c1, v2, c2, v4, c4)
    std::vector<St> states; std::vector<double> lw; St s(6);
    for (s[0] = 0; s[0] < p; s[0]++) for (s[1] = 0; s[1] < xi[s[0]].size(); s[1]++)
    for (s[2] = 0; s[2] < p; s[2]++) for (s[3] = 0; s[3] < xi[s[2]].size(); s[3]++)
    for (s[4] = 0; s[4] < p; s[4]++) for (s[5] = 0; s[5] < xi[s[4]].size(); s[5]++) {
        for (int k = 0; k < 3; k++) { nd[k]->setv(s[2 * k]); nd[k]->setc(s[2 * k + 1]); }
        double l = 0.0; bool ok = true;
        for (int k = 0; k < 3 && ok; k++) {
            int L, U; const int N = range_of(nd[k], s[2 * k], xi, L, U);
            if ((int)s[2 * k + 1] < L || (int)s[2 * k + 1] > U) ok = false;
            else l += std::log(pv[s[2 * k]]) - std::log((double)N);
        }
        if (!ok) continue;
        tree::npv bots; t.getbots(bots);
        std::vector<double> nn(bots.size(), 0.0), bb(bots.size(), 0.0), MM(bots.size(), 0.0);
        for (size_t i = 0; i < n; i++) {
            tree::tree_p b = t.bn(&x[i * p], xi); size_t a = 0; while (bots[a] != b) a++;
            nn[a] += 1.0; bb[a] += w[i]; MM[a] += w[i] * y[i];
        }
        for (double c : nn) if (c < 5.0) ok = false;
        if (!ok) continue;
        for (size_t a = 0; a < bots.size(); a++) {
            bool can = false;
            for (size_t u = 0; u < p && !can; u++) { int Lu, Uu; if (range_of(bots[a], u, xi, Lu, Uu) >= 1) can = true; }
            if (can) l += std::log1p(-pi.alpha / std::pow(1.0 + bots[a]->depth(), pi.mybeta));
            l += leaf_marginal(bb[a], MM[a], pi.tau);
        }
        states.push_back(s); lw.push_back(l);
    }
    n_states = states.size();
    const double mx = *std::max_element(lw.begin(), lw.end());
    std::vector<double> prob(lw.size()), cdf(lw.size()); double z = 0.0;
    for (size_t k = 0; k < lw.size(); k++) { prob[k] = std::exp(lw[k] - mx); z += prob[k]; }
    double cs = 0.0; for (size_t k = 0; k < prob.size(); k++) { prob[k] /= z; cs += prob[k]; cdf[k] = cs; }
    std::map<St, size_t> idx; for (size_t k = 0; k < states.size(); k++) idx[states[k]] = k;
    dinfo di; di.n = n; di.p = p; di.x = x.data(); di.y = y.data();
    std::vector<int> kidx; bart_cchange::make_kidx(xi, di, kidx);
    std::vector<size_t> nv(p, 0);
    bart_rng::set_seed(seed); arn gen; std::mt19937_64 gs(seed + 17u);
    std::vector<double> fin(prob.size(), 0.0);
    for (long r = 0; r < R; r++) {
        size_t k = (size_t)(std::lower_bound(cdf.begin(), cdf.end(), prng::u01(gs) * cs) - cdf.begin());
        if (k >= prob.size()) k = prob.size() - 1;
        std::fill(nv.begin(), nv.end(), 0);
        for (int q = 0; q < 3; q++) { nd[q]->setv(states[k][2 * q]); nd[q]->setc(states[k][2 * q + 1]); nv[states[k][2 * q]]++; }
        bart_cchange::cchange_all(t, xi, di, pi, nv, pv, kidx.data(), w.data(), gen, ctl);
        St cur(6); for (int q = 0; q < 3; q++) { cur[2 * q] = nd[q]->getv(); cur[2 * q + 1] = nd[q]->getc(); }
        auto f = idx.find(cur);
        if (f == idx.end()) { std::printf("    a scan left the target's support\n"); return {1.0, 0.0}; }
        fin[f->second] += 1.0;
    }
    double tv = 0.0; for (size_t k = 0; k < prob.size(); k++) tv += std::fabs(fin[k] / R - prob[k]); tv *= 0.5;
    std::vector<double> e(prob.size()); for (size_t k = 0; k < prob.size(); k++) e[k] = prob[k] * R;
    return {tv, chisq_p_pooled(e, fin)};
}

static bool part_D()
{
    std::printf("D exact-start invariance at a deeper topology (internal root, left child, left grandchild; p = 2; 100000 replicates):\n");
    struct Run { const char* label; bool flat; int ctl; bool must_pass; };
    const Run runs[] = { {"heterogeneous weights", false, 0, true}, {"flat likelihood", true, 0, true},
                         {"control 1, flat likelihood", true, 1, false} };
    bool ok = true;
    for (const Run& run : runs) {
        size_t ns = 0; const Fit f = part_D_run(run.flat, run.ctl, 100000, 2024u, ns);
        const bool pass = f.p > 1e-3, detected = f.p < 1e-6;
        std::printf("    %-28s %4zu states  TV %.4f  chi-square p %.3g  -> %s\n", run.label, ns, f.tv, f.p,
                    run.must_pass ? (pass ? "PASS" : "FAIL") : (detected ? "detected (PASS)" : "NOT DETECTED (FAIL)"));
        ok = ok && (run.must_pass ? pass : detected);
    }
    return ok;
}

// ---------------------------------------------------------------------------------------------------------------
// E. The class kernels. bart::draw and heterbart::draw must pass the observation weights 1/sigma^2 and
//    sigma_i^-2 / shards to the move, and must draw the leaf values after the move, for the final rules of the
//    draw. With one tree the partial residual is y, and drmu / heterdrmu draw each leaf value from
//    N(M / (1/tau^2 + b), 1 / (1/tau^2 + b)), with b and M the weighted count and sum of the leaf's observations
//    (weights 1/sigma^2 and sigma_i^-2). After every draw each leaf value is standardized with the moments of the
//    FINAL partition; the mean of z^2 over all leaves and draws must equal 1 within 5 standard errors. A leaf value
//    drawn before the move changed the leaf's observations inflates it.
// ---------------------------------------------------------------------------------------------------------------
struct probe_e : public heterbart {
    probe_e(size_t m) : heterbart(m) {}
    tree& tr() { return t[0]; }
    double tau() const { return pi.tau; }
    const std::vector<double>& ccw() const { return cc_w_; }
};

static bool part_E()
{
    const size_t n = 200, p = 2; std::mt19937_64 g(23u);
    std::vector<double> x(n * p), y(n), sv(n);
    for (size_t i = 0; i < n; i++) {
        x[i * p] = prng::u01(g); x[i * p + 1] = prng::u01(g);
        y[i] = 3.0 * x[i * p] + 0.4 * prng::n01(g);
        sv[i] = 0.3 + 0.5 * prng::u01(g);
    }
    xinfo xi(p); for (size_t v = 0; v < p; v++) { xi[v].resize(19); for (int j = 0; j < 19; j++) xi[v][j] = (j + 1) / 20.0; }
    std::vector<int> nc(p, 19);
    const double sigma = 0.5;
    bool ok = true;
    std::printf("E class kernels (one tree, n = 200):\n");
    for (int path = 0; path < 2; path++) {   // 0: bart::draw(sigma); 1: heterbart::draw(sigma_i)
        bart_rng::set_seed(31u + path); arn gen;
        probe_e hb(1); hb.setxinfo(xi); hb.setprior(0.95, 2.0, 0.6); hb.setdata(p, n, x.data(), y.data(), nc.data());
        bool wok = true;
        double s2 = 0.0; long nz = 0;
        for (int it = 0; it < 4200; it++) {
            if (path == 0) hb.draw(sigma, gen); else hb.draw(sv.data(), gen);
            for (size_t i = 0; i < n; i++) {
                const double want = path == 0 ? 1.0 / (sigma * sigma) : std::pow(sv[i], -2.0);
                if (std::fabs(hb.ccw()[i] - want) > 1e-12 * want) wok = false;
            }
            if (it < 200) continue;
            tree::npv bots; hb.tr().getbots(bots);
            std::vector<double> b(bots.size(), 0.0), M(bots.size(), 0.0);
            for (size_t i = 0; i < n; i++) {
                tree::tree_p lf = hb.tr().bn(&x[i * p], xi); size_t a = 0; while (bots[a] != lf) a++;
                const double wi = path == 0 ? 1.0 / (sigma * sigma) : std::pow(sv[i], -2.0);
                b[a] += wi; M[a] += wi * y[i];
            }
            for (size_t a = 0; a < bots.size(); a++) {
                const double prec = 1.0 / (hb.tau() * hb.tau()) + b[a];
                const double zz = (bots[a]->gettheta() - M[a] / prec) * std::sqrt(prec);
                s2 += zz * zz; nz++;
            }
        }
        if (path == 1) {   // shards: the move uses sigma_i^-2 / shards, as heterbd does
            hb.draw(sv.data(), gen, 2);
            for (size_t i = 0; i < n; i++) if (std::fabs(hb.ccw()[i] - std::pow(sv[i], -2.0) / 2.0) > 1e-12) wok = false;
        }
        const double m2 = s2 / nz, se = std::sqrt(2.0 / nz);
        const bool good = wok && std::fabs(m2 - 1.0) < 5.0 * se;
        std::printf("    %-22s weights of the move %s; leaf values vs final rules: mean z^2 %.3f (%ld values, 5 se = %.3f) -> %s\n",
                    path == 0 ? "bart::draw(sigma)" : "heterbart::draw(sigma_i)", wok ? "correct" : "WRONG", m2, nz, 5.0 * se,
                    good ? "PASS" : "FAIL");
        ok = ok && good;
    }
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
    ok = part_D() && ok;
    ok = part_E() && ok;
    std::printf("%s\n", ok ? "ALL PASS" : "SOME FAILED");
    return ok ? 0 : 1;
}
