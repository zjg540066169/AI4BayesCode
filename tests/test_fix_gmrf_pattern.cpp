// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_gmrf_pattern.cpp
//
//  gmrf_precision_block and gmrf_whitened_ess_block factorise Q with a cached
//  symbolic analysis. Armadillo does not store exact zeros, so the stored
//  pattern of the proper CAR precision Q(a, c) = 2 I - a W1 - c W2 on a ring
//  (W1, W2: neighbours at distance 1 and 2) depends on which of a, c is 0.
//  A factorisation that uses the analysis of another pattern
//    - shrink (a = 0.9 at step 1, then a = 0): leaves slots of L holding
//      values of the previous factor, so the draws have a wrong covariance
//      and no error is raised;
//    - growth (a = 0 at step 1, then a = 0.9): reads and writes out of bounds
//      (the elimination-tree root -1 is used as an index);
//    - swap (a = 0.6, c = 0, then a = 0, c = 0.6): another pattern with the
//      same number of entries; also produced by element edits inside Q_fn,
//      which leave Armadillo's compressed arrays to be synchronised on access.
//  For both blocks and each case the test checks that (1) every draw after
//  the switch equals bit for bit the draw of a freshly constructed block given
//  the same context, state and rng, and (2) the empirical covariance of the
//  draws matches the exact Q^{-1}.
//
//  Check (1) is also applied where the cached factorisation must follow a
//  change of Q's values (all, the first or the last stored entry) and where it
//  may be reused (unchanged Q, changed b or likelihood), after a failed
//  factorisation, and to gmrf_gaussian_joint_block, whose cached g(kappa),
//  mu(kappa) and factor of Q(kappa) must equal those a fresh block computes.
//  The growth and swap cases run last: without the fix they can abort.
// ============================================================================
#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif
#include <armadillo>
#include "AI4BayesCode/gmrf_precision_block.hpp"
#include "AI4BayesCode/gmrf_whitened_ess_block.hpp"
#include "AI4BayesCode/gmrf_gaussian_joint_block.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>

using namespace AI4BayesCode;

static int g_pass = 0, g_fail = 0;

static void report(bool ok, const std::string& label, const std::string& detail) {
    (ok ? g_pass : g_fail)++;
    std::printf("  [%s] %-52s %s\n", ok ? "PASS" : "FAIL", label.c_str(), detail.c_str());
    std::fflush(stdout);
}

static bool same_bits(const arma::vec& a, const arma::vec& b) {
    return a.n_elem == b.n_elem &&
           std::memcmp(a.memptr(), b.memptr(), a.n_elem * sizeof(double)) == 0;
}

static const arma::uword N_RING = 10;

// Adjacency of the ring's neighbours at distance d.
static arma::sp_mat ring_W(arma::uword n, arma::uword d = 1) {
    arma::sp_mat W(n, n);
    for (arma::uword i = 0; i < n; ++i) {
        W(i, (i + d) % n) = 1.0;
        W((i + d) % n, i) = 1.0;
    }
    return W;
}

// Q(a, c) = 2 I - a W1 - c W2, positive definite for |a| + |c| < 1.
static arma::sp_mat car_Q(double a, double c) {
    static const arma::sp_mat W1 = ring_W(N_RING, 1);
    static const arma::sp_mat W2 = ring_W(N_RING, 2);
    static const arma::sp_mat D  = 2.0 * arma::speye<arma::sp_mat>(N_RING, N_RING);
    return D - a * W1 - c * W2;   // a zero coefficient stores no entries
}

static block_context ctx_ac(double a, double c) {
    return {{"a", arma::vec{a}}, {"c", arma::vec{c}}};
}

static arma::sp_mat car_Q_ctx(const block_context& ctx) {
    return car_Q(ctx.at("a")[0], ctx.at("c")[0]);
}

static gmrf_precision_block_config prec_car_cfg() {
    gmrf_precision_block_config cfg;
    cfg.name = "x";
    cfg.n    = N_RING;
    cfg.Q_fn = car_Q_ctx;
    return cfg;
}

static gmrf_whitened_ess_block_config ess_car_cfg() {
    gmrf_whitened_ess_block_config cfg;
    cfg.name    = "x";
    cfg.n       = N_RING;
    cfg.Q_fn    = car_Q_ctx;
    cfg.log_lik = [](const arma::vec&, const block_context&) { return 0.0; };  // target = prior
    return cfg;
}

// Relative Frobenius error of the covariance of M draws of `A` (mean 0)
// against Sigma, after 50 steps in which an ESS chain's covariance settles.
template <class Block>
static double cov_rel_err(Block& A, std::mt19937_64& rng, const arma::mat& Sigma, long M) {
    arma::mat sum_xx(N_RING, N_RING, arma::fill::zeros);
    for (long s = 0; s < 50; ++s) A.step(rng);
    for (long s = 0; s < M; ++s) {
        A.step(rng);
        sum_xx += A.current() * A.current().t();
    }
    const arma::mat C = sum_xx / static_cast<double>(M);
    return arma::norm(C - Sigma, "fro") / arma::norm(Sigma, "fro");
}

// A step at (a1, c1) after a step at (a0, c0), compared with a fresh block;
// then the covariance of M further draws at (a1, c1) against the exact one.
template <class Block, class Cfg>
static void check_switch(const std::string& label, const Cfg& cfg, double a0, double c0,
                         double a1, double c1, bool copy_state, long M) {
    Block A(cfg);
    std::mt19937_64 rng(20261122u);
    A.set_context(ctx_ac(a0, c0));
    A.step(rng);

    bool exact = true;
    for (int r = 0; r < 3; ++r) {
        Block B(cfg);
        if (copy_state) B.set_current(A.current());
        std::mt19937_64 rngB = rng;
        A.set_context(ctx_ac(a1, c1));
        A.step(rng);
        B.set_context(ctx_ac(a1, c1));
        B.step(rngB);
        exact = exact && same_bits(A.current(), B.current()) && rng == rngB;
    }

    const double rel = cov_rel_err(A, rng, arma::inv_sympd(arma::mat(car_Q(a1, c1))), M);
    char buf[160];
    std::snprintf(buf, sizeof buf, "fresh-block draws identical=%d  cov rel err=%.4f (M=%ld)",
                  (int)exact, rel, M);
    report(exact && rel < 0.05, label, buf);
}

// Q_fn edits a synchronised Q(0.6, 0) element by element when ctx "e" != 0:
// entry (0,1) is removed and (0,2) inserted (and their transposes), so the
// number of entries is unchanged and the compressed arrays still hold Q(0.6, 0)
// until they are synchronised. Every draw must equal a fresh block's, and the
// covariance of the draws after the last edit must match the exact one.
static arma::sp_mat edited_Q(const block_context& ctx) {
    arma::sp_mat Q = car_Q(0.6, 0.0);
    if (ctx.at("e")[0] != 0.0) {
        Q(0, 1) = 0.0;  Q(1, 0) = 0.0;
        Q(0, 2) = -0.6; Q(2, 0) = -0.6;
    }
    return Q;
}

template <class Block, class Cfg>
static void check_edit_swap(const std::string& label, Cfg cfg, bool copy_state, long M) {
    cfg.Q_fn = edited_Q;
    Block A(cfg);
    std::mt19937_64 rng(11u);
    const double es[] = {0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0};
    bool exact = true;
    for (double e : es) {
        Block B(cfg);
        if (copy_state) B.set_current(A.current());
        std::mt19937_64 rngB = rng;
        const block_context ctx = {{"e", arma::vec{e}}};
        A.set_context(ctx); A.step(rng);
        B.set_context(ctx); B.step(rngB);
        exact = exact && same_bits(A.current(), B.current()) && rng == rngB;
    }
    const double rel = cov_rel_err(A, rng, arma::inv_sympd(arma::mat(edited_Q({{"e", arma::vec{1.0}}}))), M);
    char buf[160];
    std::snprintf(buf, sizeof buf, "fresh-block draws identical=%d  cov rel err=%.4f (M=%ld)",
                  (int)exact, rel, M);
    report(exact && rel < 0.05, label, buf);
}

// Precision block, b = ctx "b", and either
//   ridge: Q = kappa (2 I - W1) + diag(d0, 0, ..., 0, dl) + ridge (sum_to_zero);
//   else:  Q = kappa (2.5 I - W1) + diag(d0, 0, ..., 0, dl), diagonal inserted
//          element by element (compressed arrays synchronised only on demand).
// d0 and dl change the first and the last stored entry of Q. Each context of
// the sequence changes kappa, one entry, only b, or nothing; every draw must
// equal a fresh block's.
static void check_prec_values(bool ridge) {
    const arma::uword n = N_RING;
    const arma::sp_mat W = ring_W(n);
    const arma::sp_mat R = 2.0 * arma::speye<arma::sp_mat>(n, n) - W;
    gmrf_precision_block_config cfg;
    cfg.name = "x";
    cfg.n    = n;
    if (ridge) {
        cfg.Q_fn = [R](const block_context& ctx) {
            arma::sp_mat Q = ctx.at("kappa")[0] * R;
            Q(0, 0)                   += ctx.at("d0")[0];
            Q(N_RING - 1, N_RING - 1) += ctx.at("dl")[0];
            return Q;
        };
    } else {
        cfg.Q_fn = [W](const block_context& ctx) {
            const double kappa = ctx.at("kappa")[0];
            arma::sp_mat Q = -kappa * W;
            for (arma::uword i = 0; i < N_RING; ++i) Q(i, i) = 2.5 * kappa;
            Q(0, 0)                   += ctx.at("d0")[0];
            Q(N_RING - 1, N_RING - 1) += ctx.at("dl")[0];
            return Q;
        };
    }
    cfg.b_fn = [](const block_context& ctx) { return arma::vec(ctx.at("b")); };
    cfg.sum_to_zero = ridge;

    struct C { double kappa, d0, dl, b_scale; };
    const double e = 1e-12;
    const C seq[] = {{1.0, 0.5, 0.25, 1.0}, {1.0, 0.5, 0.25, 1.0}, {2.0, 0.5, 0.25, 1.0},
                     {2.0, 0.5, 0.25, 1.0}, {2.0, 0.5 + e, 0.25, 1.0}, {2.0, 0.5 + e, 0.25, 1.0},
                     {2.0, 0.5 + e, 0.25 + e, 1.0}, {2.0, 0.5 + e, 0.25 + e, -3.0},
                     {1.0, 0.5, 0.25, -3.0}, {1.0, 0.5, 0.25, -3.0}, {2.0, 0.5, 0.25, 2.0}};
    gmrf_precision_block A(cfg);
    std::mt19937_64 rng(7u);
    bool exact = true;
    for (int rep = 0; rep < 3; ++rep) {
        for (const C& c : seq) {
            block_context ctx;
            ctx["kappa"] = arma::vec{c.kappa};
            ctx["d0"]    = arma::vec{c.d0};
            ctx["dl"]    = arma::vec{c.dl};
            ctx["b"]     = c.b_scale * arma::linspace<arma::vec>(-1.0, 1.0, n);
            gmrf_precision_block B(cfg);
            std::mt19937_64 rngB = rng;
            A.set_context(ctx); A.step(rng);
            B.set_context(ctx); B.step(rngB);
            exact = exact && same_bits(A.current(), B.current());
        }
    }
    report(exact, ridge ? "precision block: Q values / b change (ridge)"
                        : "precision block: Q values / b change (no ridge)",
           exact ? "all draws equal a fresh block's" : "a draw differs from a fresh block's");
}

// ESS block, proper Q = kappa (2.5 I - W1) built as in check_prec_values (no
// ridge), with a Gaussian log-likelihood whose data change between rounds.
static void check_ess_values() {
    const arma::uword n = N_RING;
    const arma::sp_mat W = ring_W(n);
    gmrf_whitened_ess_block_config cfg;
    cfg.name = "x";
    cfg.n    = n;
    cfg.Q_fn = [W](const block_context& ctx) {
        const double kappa = ctx.at("kappa")[0];
        arma::sp_mat Q = -kappa * W;
        for (arma::uword i = 0; i < N_RING; ++i) Q(i, i) = 2.5 * kappa;
        return Q;
    };
    cfg.log_lik = [](const arma::vec& x, const block_context& ctx) {
        return -0.5 * arma::accu(arma::square(ctx.at("y") - x)) / 0.3;
    };
    const double kap[] = {1.0, 1.0, 4.0, 4.0, 4.0, 0.5, 1.0, 1.0};
    gmrf_whitened_ess_block A(cfg);
    std::mt19937_64 rng(9u);
    bool exact = true;
    for (int rep = 0; rep < 3; ++rep) {
        for (double k : kap) {
            block_context ctx;
            ctx["kappa"] = arma::vec{k};
            ctx["y"]     = (rep + 1.0) * arma::sin(arma::linspace<arma::vec>(0.0, 6.0, n));
            gmrf_whitened_ess_block B(cfg);
            B.set_current(A.current());
            std::mt19937_64 rngB = rng;
            A.set_context(ctx); A.step(rng);
            B.set_context(ctx); B.step(rngB);
            exact = exact && same_bits(A.current(), B.current());
        }
    }
    report(exact, "ESS block: Q values / likelihood change (no ridge)",
           exact ? "all draws equal a fresh block's" : "a draw differs from a fresh block's");
}

// A Q that is not positive definite must throw at every step, and the next
// valid Q must give the draw of a fresh block.
static void check_prec_failure() {
    gmrf_precision_block A(prec_car_cfg());
    std::mt19937_64 rng(3u);
    A.set_context(ctx_ac(0.5, 0.0)); A.step(rng);
    int n_throw = 0;
    for (int r = 0; r < 2; ++r) {
        A.set_context(ctx_ac(1.5, 0.0));          // smallest eigenvalue 2 - 3 < 0
        try { A.step(rng); } catch (const std::runtime_error&) { ++n_throw; }
    }
    gmrf_precision_block B(prec_car_cfg());
    std::mt19937_64 rngB = rng;
    A.set_context(ctx_ac(0.5, 0.0)); A.step(rng);
    B.set_context(ctx_ac(0.5, 0.0)); B.step(rngB);
    const bool same = same_bits(A.current(), B.current());
    char buf[96];
    std::snprintf(buf, sizeof buf, "throws=%d/2, next draw identical=%d", n_throw, (int)same);
    report(n_throw == 2 && same, "precision block: non-PD Q throws, then recovers", buf);
}

// Joint block: after t steps, a block constructed at the current kappa and x
// must give the same next step as the running block.
static void check_joint_fresh() {
    const arma::uword P = 5, n = P * P;
    arma::sp_mat R(n, n);
    for (arma::uword r = 0; r < P; ++r)
        for (arma::uword c = 0; c < P; ++c) {
            const arma::uword k = r * P + c;
            if (c + 1 < P) { R(k, k + 1) = -1; R(k + 1, k) = -1; R(k, k) += 1; R(k + 1, k + 1) += 1; }
            if (r + 1 < P) { R(k, k + P) = -1; R(k + P, k) = -1; R(k, k) += 1; R(k + P, k + P) += 1; }
        }
    gmrf_gaussian_joint_block_config cfg;
    cfg.name = "x"; cfg.n = n; cfg.R = R; cfg.sigma2 = 0.25;
    cfg.y = arma::sin(arma::linspace<arma::vec>(0.0, 9.0, n)) +
            0.3 * arma::cos(arma::linspace<arma::vec>(0.0, 41.0, n));
    cfg.kappa_a = 1.0; cfg.kappa_b = 0.01; cfg.kappa_init = 3.0;
    cfg.log_kappa_rw_sd = 2.0; cfg.sum_to_zero = true;

    gmrf_gaussian_joint_block A(cfg);
    std::mt19937_64 rng(5u);
    bool exact = true;
    int n_cmp = 0, n_cmp_acc = 0;
    for (int t = 0; t < 400; ++t) {
        const double k_before = A.kappa();
        if (t < 40 || t % 37 == 0) {
            gmrf_gaussian_joint_block_config cb = cfg;
            cb.kappa_init = A.kappa();
            cb.initial_x  = A.current();
            gmrf_gaussian_joint_block B(cb);
            std::mt19937_64 rngB = rng;
            A.step(rng);
            B.step(rngB);
            const double ka = A.kappa(), kb = B.kappa();
            exact = exact && same_bits(A.current(), B.current()) &&
                    std::memcmp(&ka, &kb, sizeof ka) == 0 && rng == rngB;
            ++n_cmp;
            if (A.kappa() != k_before) ++n_cmp_acc;
        } else {
            A.step(rng);
        }
    }
    char buf[128];
    std::snprintf(buf, sizeof buf, "%d compared steps identical=%d (%d of them accepted kappa)",
                  n_cmp, (int)exact, n_cmp_acc);
    report(exact && n_cmp_acc >= 5 && n_cmp - n_cmp_acc >= 5,
           "joint block: running block vs fresh block", buf);
}

int main() {
    std::printf("test_fix_gmrf_pattern\n");
    const long M = 100000;
    check_switch<gmrf_precision_block>("precision block: pattern shrinks (a 0.9 -> 0)",
                                       prec_car_cfg(), 0.9, 0.0, 0.0, 0.0, false, M);
    check_switch<gmrf_whitened_ess_block>("ESS block: pattern shrinks (a 0.9 -> 0)",
                                          ess_car_cfg(), 0.9, 0.0, 0.0, 0.0, true, M);
    check_prec_values(true);
    check_prec_values(false);
    check_ess_values();
    check_prec_failure();
    check_joint_fresh();
    check_switch<gmrf_precision_block>("precision block: pattern grows (a 0 -> 0.9)",
                                       prec_car_cfg(), 0.0, 0.0, 0.9, 0.0, false, M);
    check_switch<gmrf_whitened_ess_block>("ESS block: pattern grows (a 0 -> 0.9)",
                                          ess_car_cfg(), 0.0, 0.0, 0.9, 0.0, true, M);
    check_switch<gmrf_precision_block>("precision block: same-size pattern swap",
                                       prec_car_cfg(), 0.6, 0.0, 0.0, 0.6, false, M);
    check_switch<gmrf_whitened_ess_block>("ESS block: same-size pattern swap",
                                          ess_car_cfg(), 0.6, 0.0, 0.0, 0.6, true, M);
    check_edit_swap<gmrf_precision_block>("precision block: pattern swap by element edits",
                                          prec_car_cfg(), false, M);
    check_edit_swap<gmrf_whitened_ess_block>("ESS block: pattern swap by element edits",
                                             ess_car_cfg(), true, M);
    std::printf("\n=== SUMMARY: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
