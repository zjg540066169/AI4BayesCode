// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later
// (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
// test_fix_lda_token_reload.cpp
//
// Token data (w, doc) replaced after the first set_context of
// lda_collapsed_gibbs_block must be used by every later step.
//
// A. Exact target. M = 3, V = 3, K = 2, N = 6, so z has 2^6 = 64 states.
//    A composite runs 200 sweeps on corpus 1; its shared w / doc are then
//    replaced by corpus 2 and 200000 later sweeps are compared with the
//    enumerated collapsed posterior p(z | corpus 2): total variation below
//    0.02 and every state within 5 standard errors (batch means over 40
//    batches, floored at the binomial standard error).
//    TV(p(z | corpus 1), p(z | corpus 2)) = 0.42.
// B. Example size (M = 40, V = 6, K = 2, N = 1200). After the swap the
//    counts equal a recount from (z, w2, doc2), and 300 sweeps equal bit for
//    bit those of a block built on corpus 2 with z_init = z at the swap and
//    a copy of the RNG state.
// C. LdaCollapsedGibbs::set_current({w, doc}) after 10 steps: the wrapper's
//    draws and its predict_at y_rep equal those of the same reference. A
//    set_current with the loaded w / doc leaves the draws unchanged.
// D. A swap made while the block is frozen is loaded at the first step
//    after unfreeze.
// E. Copies and moves (construction and assignment) of a block taken
//    before a swap reload the same way.
// F. A later context with a different N or an out-of-range entry throws
//    std::runtime_error and leaves the block unchanged; after a context
//    without w / doc the loaded tokens are used; tokens that differ in
//    value but not in rounded id reload without changing the chain; a
//    context with w only replaces w; a change of doc alone, or of the
//    last token alone, is reloaded.
// G. Tokens A, then B, then A again, set by a context with both keys, with
//    w only (B differs from A in w) and with doc only (B differs from A in
//    doc). Right after the revert the counts equal a recount from (z, A),
//    and the next 100 sweeps equal bit for bit those of a block built on A
//    with z_init = z and a copy of the RNG state.
//
// Build from tests/ with the compile line `make -n` prints for any other
// test, with test_fix_lda_token_reload as the target name.
// ============================================================================

#include "AI4BayesCode/block_sampler.hpp"
#include "AI4BayesCode/backend_neutral.hpp"
#include "AI4BayesCode/shared_data.hpp"
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/rcpp_wrap.hpp"
#include "AI4BayesCode/lda_collapsed_gibbs_block.hpp"
#include "AI4BayesCode/kernel_control_mixin.hpp"

// The shipped wrapper class, with its demo main() renamed.
#define main lda_example_demo_main
#include "../examples/LdaCollapsedGibbs.cpp"
#undef main

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using AI4BayesCode::state_map;

int g_fail = 0;

void check(bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

struct Corpus { arma::vec w, doc; };

bool same(const arma::mat& a, const arma::mat& b) {
    return a.n_rows == b.n_rows && a.n_cols == b.n_cols
        && std::equal(a.begin(), a.end(), b.begin());
}

double max_abs_diff(const arma::mat& a, const arma::mat& b) {
    if (a.n_rows != b.n_rows || a.n_cols != b.n_cols) return INFINITY;
    return a.n_elem ? arma::abs(a - b).max() : 0.0;
}

// Largest difference between two blocks over z, theta, phi and the counts.
double block_diff(const lda_collapsed_gibbs_block& a,
                  const lda_collapsed_gibbs_block& b) {
    double m = 0.0;
    m = std::max(m, max_abs_diff(a.current(),       b.current()));
    m = std::max(m, max_abs_diff(a.current_theta(), b.current_theta()));
    m = std::max(m, max_abs_diff(a.current_phi(),   b.current_phi()));
    m = std::max(m, max_abs_diff(a.counts_dk(),     b.counts_dk()));
    m = std::max(m, max_abs_diff(a.counts_kv(),     b.counts_kv()));
    m = std::max(m, max_abs_diff(a.counts_k(),      b.counts_k()));
    return m;
}

// Count tables recomputed from labels z and the tokens of corpus c.
bool counts_match(const lda_collapsed_gibbs_block& b, const arma::vec& z,
                  const Corpus& c, std::size_t M, std::size_t V,
                  std::size_t K) {
    arma::mat ndk(M, K, arma::fill::zeros), nkv(K, V, arma::fill::zeros);
    arma::vec nk(K, arma::fill::zeros);
    for (std::size_t t = 0; t < z.n_elem; ++t) {
        const std::size_t k = static_cast<std::size_t>(std::lround(z[t])) - 1;
        const std::size_t d = static_cast<std::size_t>(std::lround(c.doc[t])) - 1;
        const std::size_t v = static_cast<std::size_t>(std::lround(c.w[t])) - 1;
        ndk(d, k) += 1.0;
        nkv(k, v) += 1.0;
        nk[k]     += 1.0;
    }
    return same(b.counts_dk(), ndk) && same(b.counts_kv(), nkv)
        && same(b.counts_k(), nk);
}

lda_collapsed_gibbs_block_config make_cfg(std::size_t M, std::size_t V,
                                          std::size_t K,
                                          const arma::vec& alpha,
                                          const arma::vec& beta,
                                          const arma::vec& z_init) {
    lda_collapsed_gibbs_block_config cfg;
    cfg.name  = "lda";
    cfg.M     = M;
    cfg.V     = V;
    cfg.K     = K;
    cfg.alpha = alpha;
    cfg.beta  = beta;
    cfg.z_init = z_init;
    return cfg;
}

// Composite with the same wiring as LdaCollapsedGibbs.
std::unique_ptr<composite_block> make_comp(const Corpus& c, std::size_t M,
                                           std::size_t V, std::size_t K,
                                           const arma::vec& alpha,
                                           const arma::vec& beta,
                                           const arma::vec& z_init) {
    auto comp = std::make_unique<composite_block>("lda_test");
    comp->data().set("w", c.w);
    comp->data().set("doc", c.doc);
    comp->data().declare_dependencies("lda", {"w", "doc"});
    comp->add_child(std::make_unique<lda_collapsed_gibbs_block>(
        make_cfg(M, V, K, alpha, beta, z_init)));
    return comp;
}

lda_collapsed_gibbs_block& blk(composite_block& c) {
    return dynamic_cast<lda_collapsed_gibbs_block&>(c.child(0));
}

block_context ctx_of(const Corpus& c) {
    block_context ctx;
    ctx["w"]   = c.w;
    ctx["doc"] = c.doc;
    return ctx;
}

void set_tokens(composite_block& comp, const Corpus& c) {
    comp.data().set("w", c.w);
    comp.data().set("doc", c.doc);
}

// ---- A. Exact target on a 64-state corpus ---------------------------------

const std::size_t SM = 3, SV = 3, SK = 2, SN = 6, NSTATE = 64;

Corpus small_corpus(const std::size_t* doc1, const std::size_t* w1) {
    Corpus c;
    c.w.set_size(SN);
    c.doc.set_size(SN);
    for (std::size_t t = 0; t < SN; ++t) {
        c.w[t]   = static_cast<double>(w1[t]);
        c.doc[t] = static_cast<double>(doc1[t]);
    }
    return c;
}

// p(z | w, doc) for every state s, where z_t = ((s >> t) & 1) + 1: the
// Dirichlet-multinomial marginal likelihood from lgamma of whole count
// tables.
std::vector<double> exact_posterior(const Corpus& c, const arma::vec& alpha,
                                    const arma::vec& beta) {
    std::vector<double> lp(NSTATE);
    const double sa = arma::accu(alpha), sb = arma::accu(beta);
    for (std::size_t s = 0; s < NSTATE; ++s) {
        double ndk[SM][SK] = {}, nkv[SK][SV] = {};
        for (std::size_t t = 0; t < SN; ++t) {
            const std::size_t k = (s >> t) & 1u;
            const std::size_t d = static_cast<std::size_t>(c.doc[t]) - 1;
            const std::size_t v = static_cast<std::size_t>(c.w[t]) - 1;
            ndk[d][k] += 1.0;
            nkv[k][v] += 1.0;
        }
        double l = 0.0;
        for (std::size_t d = 0; d < SM; ++d) {
            double nd = 0.0;
            for (std::size_t k = 0; k < SK; ++k) {
                l  += std::lgamma(ndk[d][k] + alpha[k]);
                nd += ndk[d][k];
            }
            l -= std::lgamma(nd + sa);
        }
        for (std::size_t k = 0; k < SK; ++k) {
            double nk = 0.0;
            for (std::size_t v = 0; v < SV; ++v) {
                l  += std::lgamma(nkv[k][v] + beta[v]);
                nk += nkv[k][v];
            }
            l -= std::lgamma(nk + sb);
        }
        lp[s] = l;
    }
    const double mx = *std::max_element(lp.begin(), lp.end());
    double tot = 0.0;
    for (double& x : lp) { x = std::exp(x - mx); tot += x; }
    for (double& x : lp) x /= tot;
    return lp;
}

double tv(const std::vector<double>& p, const std::vector<double>& q) {
    double s = 0.0;
    for (std::size_t i = 0; i < p.size(); ++i) s += std::abs(p[i] - q[i]);
    return 0.5 * s;
}

void part_a() {
    std::printf("A. stationary distribution after a swap, exact target\n");
    const std::size_t doc1[SN] = {1, 1, 2, 2, 3, 3};
    const std::size_t w1[SN]   = {1, 2, 2, 3, 1, 3};
    const std::size_t doc2[SN] = {1, 1, 1, 2, 3, 3};
    const std::size_t w2[SN]   = {1, 1, 1, 3, 3, 3};
    const Corpus c1 = small_corpus(doc1, w1), c2 = small_corpus(doc2, w2);
    const arma::vec alpha = {0.8, 1.6};
    const arma::vec beta  = {0.6, 1.0, 1.4};
    const std::vector<double> p1 = exact_posterior(c1, alpha, beta);
    const std::vector<double> p2 = exact_posterior(c2, alpha, beta);

    auto comp = make_comp(c1, SM, SV, SK, alpha, beta, arma::vec());
    std::mt19937_64 rng(20261122);
    for (int i = 0; i < 200; ++i) comp->step(rng);
    set_tokens(*comp, c2);
    for (int i = 0; i < 1000; ++i) comp->step(rng);

    const std::size_t NB = 40, BS = 5000, S = NB * BS;
    std::vector<std::vector<double>> batch(NB, std::vector<double>(NSTATE, 0.0));
    for (std::size_t b = 0; b < NB; ++b) {
        for (std::size_t i = 0; i < BS; ++i) {
            comp->step(rng);
            const arma::vec& z = blk(*comp).current();
            std::size_t s = 0;
            for (std::size_t t = 0; t < SN; ++t)
                if (std::lround(z[t]) == 2) s |= (std::size_t{1} << t);
            batch[b][s] += 1.0;
        }
    }
    std::vector<double> emp(NSTATE, 0.0);
    double max_sig = 0.0;
    for (std::size_t s = 0; s < NSTATE; ++s) {
        double mean = 0.0, ss = 0.0;
        for (std::size_t b = 0; b < NB; ++b) mean += batch[b][s] / BS;
        mean /= NB;
        for (std::size_t b = 0; b < NB; ++b) {
            const double dev = batch[b][s] / BS - mean;
            ss += dev * dev;
        }
        emp[s] = mean;
        const double mcse = std::sqrt(ss / (NB - 1) / NB);
        const double bin  = std::sqrt(p2[s] * (1.0 - p2[s]) / S);
        max_sig = std::max(max_sig, std::abs(mean - p2[s]) / std::max(mcse, bin));
    }
    const double tv2 = tv(emp, p2), tv1 = tv(emp, p1);
    std::printf("    TV(p1, p2) = %.4f; empirical after swap: TV to p2 = %.4f, "
                "TV to p1 = %.4f, max |emp - p2| / se = %.2f\n",
                tv(p1, p2), tv2, tv1, max_sig);
    check(tv2 < 0.02, "A: z after the swap follows p(z | corpus 2) (TV < 0.02)");
    check(max_sig < 5.0, "A: every one of the 64 states within 5 standard errors");
}

// ---- Example-size corpora --------------------------------------------------

const std::size_t M = 40, V = 6, K = 2, LD = 30, N = M * LD;

Corpus gen_corpus(const arma::mat& phi, bool alternate_lead,
                  std::uint64_t seed) {
    std::mt19937_64 r(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    Corpus c;
    c.w.set_size(N);
    c.doc.set_size(N);
    std::size_t t = 0;
    for (std::size_t d = 0; d < M; ++d) {
        const double lead = 0.7 + 0.25 * u(r);
        const bool first = alternate_lead ? (d % 2 == 0) : (d < M / 2);
        const double w0 = first ? lead : 1.0 - lead;
        for (std::size_t n = 0; n < LD; ++n, ++t) {
            const std::size_t z = (u(r) < w0) ? 0u : 1u;
            const double x = u(r);
            double cum = 0.0;
            std::size_t v = V - 1;
            for (std::size_t j = 0; j < V; ++j) {
                cum += phi(z, j);
                if (x < cum) { v = j; break; }
            }
            c.w[t]   = static_cast<double>(v + 1);
            c.doc[t] = static_cast<double>(d + 1);
        }
    }
    return c;
}

// Corpus 1 follows the example's data-generating process; corpus 2 has a
// different phi, a different document structure and shuffled tokens.
void make_corpora(Corpus& c1, Corpus& c2) {
    arma::mat phi1(K, V), phi2(K, V);
    phi1.row(0) = arma::rowvec{0.40, 0.35, 0.15, 0.05, 0.03, 0.02};
    phi1.row(1) = arma::rowvec{0.02, 0.03, 0.05, 0.15, 0.35, 0.40};
    phi2.row(0) = arma::rowvec{0.05, 0.05, 0.40, 0.40, 0.05, 0.05};
    phi2.row(1) = arma::rowvec{0.30, 0.30, 0.05, 0.05, 0.15, 0.15};
    c1 = gen_corpus(phi1, false, 2024);
    c2 = gen_corpus(phi2, true, 99);
    std::vector<std::size_t> perm(N);
    std::iota(perm.begin(), perm.end(), 0);
    std::mt19937_64 pr(5);
    std::shuffle(perm.begin(), perm.end(), pr);
    const Corpus tmp = c2;
    for (std::size_t t = 0; t < N; ++t) {
        c2.w[t]   = tmp.w[perm[t]];
        c2.doc[t] = tmp.doc[perm[t]];
    }
}

// ---- B. Composite swap at the example size ---------------------------------

void part_b(const Corpus& c1, const Corpus& c2) {
    std::printf("B. composite: shared w / doc replaced after 10 sweeps\n");
    const arma::vec a(K, arma::fill::ones), b(V, arma::fill::ones);
    auto X = make_comp(c1, M, V, K, a, b, arma::vec());
    std::mt19937_64 rng(7);
    for (int i = 0; i < 10; ++i) X->step(rng);
    const arma::vec z_swap = blk(*X).current();
    std::mt19937_64 rng_y = rng;
    set_tokens(*X, c2);
    auto Y = make_comp(c2, M, V, K, a, b, z_swap);

    X->step(rng);
    Y->step(rng_y);
    check(counts_match(blk(*X), blk(*X).current(), c2, M, V, K),
          "B: counts after the swap equal a recount from (z, w2, doc2)");
    double md = block_diff(blk(*X), blk(*Y));
    for (int s = 1; s < 300; ++s) {
        X->step(rng);
        Y->step(rng_y);
        md = std::max(md, block_diff(blk(*X), blk(*Y)));
    }
    std::printf("    max |X - Y| over 300 sweeps (z, theta, phi, counts) = %g\n", md);
    check(md == 0.0,
          "B: draws after the swap equal a block built on corpus 2 (bitwise)");
}

// ---- C. Wrapper set_current({w, doc}) after steps ---------------------------

// The wrapper's y_rep refresher evaluated at (doc, theta, phi) with a fresh
// copy of its predict RNG.
arma::vec manual_yrep(const arma::vec& doc, const arma::vec& theta,
                      const arma::vec& phi, int seed) {
    std::mt19937_64 rng(static_cast<std::uint64_t>(seed) ^ 0x9E3779B97F4A7C15ULL);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    arma::vec out(doc.n_elem);
    for (std::size_t t = 0; t < doc.n_elem; ++t) {
        const std::size_t d = static_cast<std::size_t>(std::lround(doc[t])) - 1;
        const double u1 = uniform(rng);
        double cum = 0.0;
        std::size_t z_rep = K - 1;
        for (std::size_t k = 0; k < K; ++k) {
            cum += theta[d + k * M];
            if (u1 < cum) { z_rep = k; break; }
        }
        const double u2 = uniform(rng);
        double cum2 = 0.0;
        std::size_t w_rep = V - 1;
        for (std::size_t v = 0; v < V; ++v) {
            cum2 += phi[z_rep + v * K];
            if (u2 < cum2) { w_rep = v; break; }
        }
        out[t] = static_cast<double>(w_rep + 1);
    }
    return out;
}

void part_c(const Corpus& c1, const Corpus& c2) {
    std::printf("C. wrapper: LdaCollapsedGibbs::set_current({w, doc}) after 10 steps\n");
    const arma::vec a(K, arma::fill::ones), b(V, arma::fill::ones);
    const int seed = 7;
    LdaCollapsedGibbs A(c1.w, c1.doc, static_cast<int>(M), static_cast<int>(V),
                        static_cast<int>(K), a, b, seed, false);
    A.step(10);
    // Reference composite: same wiring and seed, so the same chain up to here.
    auto R = make_comp(c1, M, V, K, a, b, arma::vec());
    std::mt19937_64 rng_r(static_cast<std::uint64_t>(seed));
    for (int i = 0; i < 10; ++i) R->step(rng_r);
    const arma::vec z_swap = A.get_current().at("z");
    check(same(z_swap, blk(*R).current()),
          "C: reference composite reproduces the wrapper chain before the swap");
    auto Y = make_comp(c2, M, V, K, a, b, z_swap);
    std::mt19937_64 rng_y = rng_r;

    state_map sw;
    sw["w"]   = c2.w;
    sw["doc"] = c2.doc;
    A.set_current(sw);
    double md = 0.0;
    for (int s = 0; s < 300; ++s) {
        A.step(1);
        Y->step(rng_y);
        const state_map g = A.get_current();
        md = std::max(md, max_abs_diff(g.at("z"),     blk(*Y).current()));
        md = std::max(md, max_abs_diff(g.at("theta"), blk(*Y).current_theta()));
        md = std::max(md, max_abs_diff(g.at("phi"),   blk(*Y).current_phi()));
    }
    std::printf("    max |wrapper - reference| over 300 steps (z, theta, phi) = %g\n", md);
    check(md == 0.0,
          "C: wrapper draws after the swap equal the corpus-2 reference (bitwise)");

    const arma::mat yrep = A.predict_at(state_map{}).at("y_rep");
    const arma::vec expect = manual_yrep(c2.doc, blk(*Y).current_theta(),
                                         blk(*Y).current_phi(), seed);
    check(same(arma::vectorise(yrep), expect),
          "C: predict_at y_rep uses doc2 with the corpus-2 theta and phi");

    // Writing back the loaded tokens is not a change.
    LdaCollapsedGibbs P(c1.w, c1.doc, static_cast<int>(M), static_cast<int>(V),
                        static_cast<int>(K), a, b, 5, false);
    LdaCollapsedGibbs Q(c1.w, c1.doc, static_cast<int>(M), static_cast<int>(V),
                        static_cast<int>(K), a, b, 5, false);
    P.step(10);
    Q.step(10);
    state_map same_tokens;
    same_tokens["w"]   = c1.w;
    same_tokens["doc"] = c1.doc;
    P.set_current(same_tokens);
    double mq = 0.0;
    for (int s = 0; s < 50; ++s) {
        P.step(1);
        Q.step(1);
        const state_map gp = P.get_current(), gq = Q.get_current();
        for (const char* key : {"z", "theta", "phi"})
            mq = std::max(mq, max_abs_diff(gp.at(key), gq.at(key)));
    }
    check(mq == 0.0,
          "C: set_current with the loaded w / doc leaves the draws unchanged");
}

// ---- D. Swap while frozen ---------------------------------------------------

void part_d(const Corpus& c1, const Corpus& c2) {
    std::printf("D. composite: swap while the block is frozen\n");
    const arma::vec a(K, arma::fill::ones), b(V, arma::fill::ones);
    auto X = make_comp(c1, M, V, K, a, b, arma::vec());
    std::mt19937_64 rng(11);
    for (int i = 0; i < 10; ++i) X->step(rng);
    const arma::vec z_f = blk(*X).current();
    std::mt19937_64 rng_y = rng;
    X->freeze(std::vector<std::string>{"lda"}, /*quiet=*/true);
    set_tokens(*X, c2);
    for (int i = 0; i < 3; ++i) X->step(rng);   // the frozen block is skipped
    X->unfreeze(std::vector<std::string>{"lda"});
    auto Y = make_comp(c2, M, V, K, a, b, z_f);
    double md = 0.0;
    for (int s = 0; s < 200; ++s) {
        X->step(rng);
        Y->step(rng_y);
        md = std::max(md, block_diff(blk(*X), blk(*Y)));
    }
    check(md == 0.0,
          "D: the first step after unfreeze loads the new tokens (bitwise)");
}

// ---- E. Copies and moves ----------------------------------------------------

void part_e(const Corpus& c1, const Corpus& c2) {
    std::printf("E. copies and moves taken before a swap\n");
    const arma::vec a(K, arma::fill::ones), b(V, arma::fill::ones);
    const block_context ctx1 = ctx_of(c1), ctx2 = ctx_of(c2);
    lda_collapsed_gibbs_block A(make_cfg(M, V, K, a, b, arma::vec()));
    std::mt19937_64 rng(13);
    A.set_context(ctx1);
    for (int i = 0; i < 10; ++i) { A.set_context(ctx1); A.step(rng); }

    lda_collapsed_gibbs_block copy_c(A);
    lda_collapsed_gibbs_block tmp1(A);
    lda_collapsed_gibbs_block move_c(std::move(tmp1));
    lda_collapsed_gibbs_block copy_a(make_cfg(M, V, K, a, b, arma::vec()));
    copy_a = A;
    lda_collapsed_gibbs_block tmp2(A);
    lda_collapsed_gibbs_block move_a(make_cfg(M, V, K, a, b, arma::vec()));
    move_a = std::move(tmp2);

    lda_collapsed_gibbs_block Y(make_cfg(M, V, K, a, b, A.current()));
    std::vector<lda_collapsed_gibbs_block*> xs = {&A, &copy_c, &move_c,
                                                 &copy_a, &move_a};
    std::vector<std::mt19937_64> rngs(xs.size(), rng);
    std::mt19937_64 rng_y = rng;
    double md = 0.0;
    for (int s = 0; s < 100; ++s) {
        Y.set_context(ctx2);
        Y.step(rng_y);
        for (std::size_t i = 0; i < xs.size(); ++i) {
            xs[i]->set_context(ctx2);
            xs[i]->step(rngs[i]);
            md = std::max(md, block_diff(*xs[i], Y));
        }
    }
    check(md == 0.0,
          "E: original, copy/move-constructed and copy/move-assigned blocks "
          "all follow corpus 2 (bitwise)");
}

// ---- F. Rejected and partial contexts --------------------------------------

template <typename F>
bool throws_runtime_error(F f, std::string& msg) {
    try { f(); } catch (const std::runtime_error& e) { msg = e.what(); return true; }
    return false;
}

void part_f(const Corpus& c1, const Corpus& c2) {
    std::printf("F. rejected and partial contexts after initialization\n");
    const arma::vec a(K, arma::fill::ones), b(V, arma::fill::ones);
    const block_context ctx1 = ctx_of(c1);
    lda_collapsed_gibbs_block G(make_cfg(M, V, K, a, b, arma::vec()));
    std::mt19937_64 rng(17);
    for (int i = 0; i < 5; ++i) { G.set_context(ctx1); G.step(rng); }
    const lda_collapsed_gibbs_block H0(G);   // state before the contexts below
    lda_collapsed_gibbs_block H(H0);
    std::mt19937_64 rng_h = rng;

    Corpus longer = c1;
    longer.w.resize(N + 1);
    longer.doc.resize(N + 1);
    longer.w[N]   = 1.0;
    longer.doc[N] = 1.0;
    std::string msg;
    const bool t1 = throws_runtime_error([&] { G.set_context(ctx_of(longer)); }, msg);
    std::printf("    different N: %s\n", t1 ? msg.c_str() : "no error");
    check(t1 && block_diff(G, H0) == 0.0,
          "F: a context with a different N throws and leaves the block unchanged");

    Corpus bad = c2;
    bad.w[N / 2] = static_cast<double>(V + 1);
    const bool t2 = throws_runtime_error([&] { G.set_context(ctx_of(bad)); }, msg);
    const bool t3 = throws_runtime_error([&] { G.set_context(ctx_of(bad)); }, msg);
    std::printf("    out-of-range w: %s\n", t2 ? msg.c_str() : "no error");
    check(t2 && t3 && block_diff(G, H0) == 0.0,
          "F: an out-of-range entry throws on every call and leaves the block "
          "unchanged");

    const bool t4 = throws_runtime_error([&] { G.set_context(block_context{}); }, msg);
    check(!t4 && block_diff(G, H0) == 0.0,
          "F: after a context without w / doc the loaded tokens are used");

    double md = 0.0;
    for (int s = 0; s < 50; ++s) {
        G.set_context(ctx1);
        G.step(rng);
        H.set_context(ctx1);
        H.step(rng_h);
        md = std::max(md, block_diff(G, H));
    }
    check(md == 0.0,
          "F: after the rejected contexts the chain equals a copy made before "
          "them (bitwise)");

    // Values that differ but round to the same ids: the reload rebuilds the
    // same counts, so the chain is unchanged.
    lda_collapsed_gibbs_block E(H0);
    lda_collapsed_gibbs_block E_ref(H0);
    std::mt19937_64 rng_e = rng_h, rng_e_ref = rng_h;
    Corpus nudged = c1;
    nudged.w   += 1e-9;
    nudged.doc -= 1e-9;
    const block_context ctx_n = ctx_of(nudged);
    double me = 0.0;
    for (int s = 0; s < 50; ++s) {
        E.set_context((s % 2) ? ctx1 : ctx_n);
        E.step(rng_e);
        E_ref.set_context(ctx1);
        E_ref.step(rng_e_ref);
        me = std::max(me, block_diff(E, E_ref));
    }
    check(me == 0.0,
          "F: reloading tokens with the same ids leaves the chain unchanged "
          "(bitwise)");

    // w only: the loaded doc is used.
    lda_collapsed_gibbs_block W(H0);
    block_context w_only;
    w_only["w"] = c2.w;
    W.set_context(w_only);
    Corpus mixed = c1;
    mixed.w = c2.w;
    check(counts_match(W, W.current(), mixed, M, V, K),
          "F: a context with w only reloads w; the loaded doc is used");

    // doc changed, w unchanged.
    lda_collapsed_gibbs_block D(H0);
    Corpus redoc = c1;
    for (std::size_t t = 0; t < N; ++t)
        redoc.doc[t] = static_cast<double>(M - (t / LD));
    D.set_context(ctx_of(redoc));
    check(counts_match(D, D.current(), redoc, M, V, K),
          "F: a context with only doc changed reloads doc");

    // A change confined to the last token, of w and of doc.
    Corpus last_w = c1, last_doc = c1;
    last_w.w[N - 1]     = static_cast<double>(std::lround(c1.w[N - 1]) % V + 1);
    last_doc.doc[N - 1] = static_cast<double>(std::lround(c1.doc[N - 1]) % M + 1);
    lda_collapsed_gibbs_block blk_w(H0), blk_doc(H0);
    blk_w.set_context(ctx_of(last_w));
    blk_doc.set_context(ctx_of(last_doc));
    check(counts_match(blk_w, blk_w.current(), last_w, M, V, K)
              && counts_match(blk_doc, blk_doc.current(), last_doc, M, V, K),
          "F: a change of the last token's w or doc alone is reloaded");
}

// ---- G. Revert to tokens loaded earlier ------------------------------------

// The block runs 10 sweeps on A and 10 on B (contexts with both keys); then
// every set_context passes `revert`, which resolves to A.
void revert_case(const char* label, const Corpus& A, const Corpus& B,
                 const block_context& revert, std::uint64_t seed) {
    const arma::vec a(K, arma::fill::ones), b(V, arma::fill::ones);
    const block_context ctxA = ctx_of(A), ctxB = ctx_of(B);
    lda_collapsed_gibbs_block X(make_cfg(M, V, K, a, b, arma::vec()));
    std::mt19937_64 rng(seed);
    for (int i = 0; i < 10; ++i) { X.set_context(ctxA); X.step(rng); }
    for (int i = 0; i < 10; ++i) { X.set_context(ctxB); X.step(rng); }
    const bool on_b = counts_match(X, X.current(), B, M, V, K);
    X.set_context(revert);
    const bool on_a = counts_match(X, X.current(), A, M, V, K);

    lda_collapsed_gibbs_block Y(make_cfg(M, V, K, a, b, X.current()));
    std::mt19937_64 rng_y = rng;
    double md = 0.0;
    for (int s = 0; s < 100; ++s) {
        X.set_context(revert);
        X.step(rng);
        Y.set_context(ctxA);
        Y.step(rng_y);
        md = std::max(md, block_diff(X, Y));
    }
    std::printf("    %s: counts on B %s, on A right after the revert %s, "
                "max |X - Y| over 100 sweeps = %g\n",
                label, on_b ? "match" : "differ", on_a ? "match" : "differ", md);
    check(on_b && on_a,
          std::string("G: ") + label + ": counts equal a recount from (z, A) "
          "right after the revert");
    check(md == 0.0,
          std::string("G: ") + label + ": sweeps after the revert equal a block "
          "built on A (bitwise)");
}

void part_g(const Corpus& c1, const Corpus& c2) {
    std::printf("G. revert to tokens loaded earlier (A -> B -> A)\n");
    Corpus w_changed = c1, doc_changed = c1;
    w_changed.w     = c2.w;
    doc_changed.doc = c2.doc;
    block_context w_only, doc_only;
    w_only["w"]     = c1.w;
    doc_only["doc"] = c1.doc;
    revert_case("both keys", c1, c2, ctx_of(c1), 19);
    revert_case("w only", c1, w_changed, w_only, 23);
    revert_case("doc only", c1, doc_changed, doc_only, 29);
}

}  // namespace

int main() {
    std::printf("test_fix_lda_token_reload\n");
    part_a();
    Corpus c1, c2;
    make_corpora(c1, c2);
    part_b(c1, c2);
    part_c(c1, c2);
    part_d(c1, c2);
    part_e(c1, c2);
    part_f(c1, c2);
    part_g(c1, c2);
    if (g_fail) {
        std::printf("FAILED: %d check(s)\n", g_fail);
        return 1;
    }
    std::printf("ALL PASS\n");
    return 0;
}
