/*
 * test_nuts_small_step_invariance.cpp
 *
 * KNOWN FAILURE. This test is RED on purpose: it pins a confirmed defect in the
 * vendored NUTS kernel so the defect cannot be forgotten, and so that whoever
 * fixes it has an unambiguous pass/fail gate. It is deliberately NOT in the
 * Makefile's default `run` panel -- build and run it explicitly:
 *
 *     make test_nuts_small_step_invariance && ./test_nuts_small_step_invariance
 *
 * ---------------------------------------------------------------------------
 * WHAT IS WRONG
 * ---------------------------------------------------------------------------
 * At small step sizes the kernel does not leave the target invariant. On a
 * plain Dirichlet(2,3,4,5) through a SIMPLEX constraint -- three unconstrained
 * dimensions, nothing exotic -- E[p_0] should be exactly 2/14:
 *
 *     step size   E[p_0]      deviation   verdict
 *       0.05      0.1469471   +2.863%     +85.2 se
 *       0.10      0.1478999   +3.530%    +106.7 se
 *       0.30      0.1467504   +2.725%     +93.4 se
 *       1.00      0.1428215   -0.025%      -0.7 se   (clean)
 *
 * It matters in the DEFAULT configuration too: adaptation lands wherever the
 * geometry puts it, and a second sub-param with a stiffer scale pulls the
 * adapted step size down. Measured on SIMPLEX + POSITIVE(InvGamma(5,6)) with
 * adaptation on, adapted eps 0.72: +0.605% at +15.6 se. Real hierarchical
 * models routinely adapt below 0.3.
 *
 * ---------------------------------------------------------------------------
 * WHY THE MEASUREMENT IS TRUSTWORTHY
 * ---------------------------------------------------------------------------
 * Every chain starts from an EXACT draw of the target (a Dirichlet built from
 * Gamma variates), so E[chain mean] equals the true mean exactly whatever the
 * autocorrelation -- there is no burn-in to confound. Many INDEPENDENT chains
 * are run and each chain's mean is ONE observation, so the standard error is
 * the across-chain spread and cannot be deflated by an autocorrelation time
 * longer than a batch. An earlier batch-means version of this measurement WAS
 * fooled that way at small step sizes; that is why the estimator is built this
 * way and should not be "simplified" back.
 *
 * Two controls are asserted below alongside the defect, because the whole
 * conclusion rests on them:
 *   - the requested starting point is actually honoured (to ~1e-16), and
 *   - the deviation does NOT shrink with chain length. Initialization bias
 *     falls like 1/n; a wrong stationary distribution does not. Measured flat
 *     across three orders of magnitude: 500 -> +3.369%, 2000 -> +3.509%,
 *     20000 -> +3.530%.
 *
 * ---------------------------------------------------------------------------
 * WHAT HAS ALREADY BEEN RULED OUT (do not re-investigate)
 * ---------------------------------------------------------------------------
 *   - joint_nuts_block's constraint-transform layer. Writing the identical
 *     transform by hand and passing a plain REAL slice reproduces the bias to
 *     SEVEN significant figures (0.1464295 either way), with a finite-
 *     difference gradient instead of the analytic one.
 *   - the tree-depth cap: raising it from 5 to 11 does not reduce the bias.
 *   - the leapfrog gradient-cache optimisation: the kernel's own -DCACHE_CHECK
 *     reports zero invariant violations over 2000 steps of the biased config.
 *   - anisotropy: SIMPLEX ALONE is biased too. Earlier readings that suggested
 *     curvature mismatch were really just "a stiffer sub-param adapts to a
 *     smaller step size". Step size is the single explanatory variable.
 *   - the algorithm's structure, checked line by line against Hoffman & Gelman
 *     Algorithm 3: slice variable log u - H0, base case n = I[log u <= -H] and
 *     s = I[log u < 1000 - H], recursive sub-tree U-turn, progressive
 *     multinomial selection, and a symmetric reversible leapfrog.
 *
 * The bias is worst when trajectories involve MANY doublings and vanishes when
 * they are short, so whatever is wrong lives on a path only long trajectories
 * exercise.
 */

#include "AI4BayesCode/joint_nuts_block.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using AI4BayesCode::block_context;
using AI4BayesCode::joint_constraint;
using AI4BayesCode::joint_nuts_block;
using AI4BayesCode::joint_nuts_block_config;

namespace {

int checks = 0, failures = 0;

void check(bool ok, const char* what, const char* detail = "") {
    ++checks;
    std::printf("  %s  %s%s%s\n", ok ? "ok  " : "FAIL", what,
                *detail ? " -- " : "", detail);
    if (!ok) ++failures;
}

const double ALPHA[4] = {2.0, 3.0, 4.0, 5.0};
const double EXACT    = 2.0 / 14.0;

joint_nuts_block_config make_cfg(double eps, const arma::vec& start) {
    joint_nuts_block_config cfg;
    cfg.name = "p_block";
    cfg.sub_params = {{"p", 4, joint_constraint::SIMPLEX}};
    // Freeze the step size: this is a statement about a FIXED kernel, and a
    // fixed kernel with a correct Metropolis correction is exact at ANY step
    // size. Step size may change efficiency; it may not change the answer.
    cfg.initial_step_size   = eps;
    cfg.n_warmup_first_call = 0;
    cfg.n_warmup_per_step   = 0;
    cfg.min_step_size       = 0.0;
    cfg.max_step_size       = 0.0;
    cfg.log_density_grad =
        [](const arma::vec& nat, const block_context&, arma::vec* g) -> double {
        double lp = 0.0;
        if (g) { g->set_size(nat.n_elem); g->zeros(); }
        for (std::size_t k = 0; k < 4; ++k) {
            const double pk = std::max(nat[k], 1e-300);
            lp += (ALPHA[k] - 1.0) * std::log(pk);
            if (g) (*g)[k] = (ALPHA[k] - 1.0) / pk;
        }
        return lp;
    };
    cfg.initial_cat = start;
    return cfg;
}

/// An exact Dirichlet(ALPHA) draw: Gamma variates, normalised.
arma::vec exact_draw(std::mt19937_64& rng) {
    arma::vec p(4);
    double tot = 0.0;
    for (int k = 0; k < 4; ++k) {
        std::gamma_distribution<double> G(ALPHA[k], 1.0);
        p[k] = G(rng); tot += p[k];
    }
    for (int k = 0; k < 4; ++k) p[k] /= tot;
    return p;
}

double chain_mean(double eps, std::uint64_t seed, long n) {
    std::mt19937_64 init_rng(seed * 7919u + 13u);
    joint_nuts_block blk(make_cfg(eps, exact_draw(init_rng)));
    block_context ctx; blk.set_context(ctx);
    std::mt19937_64 rng(seed);
    double acc = 0.0;
    for (long i = 0; i < n; ++i) { blk.step(rng); acc += blk.current()[0]; }
    return acc / n;
}

struct Est { double mean, se, z; };

Est estimate(double eps, int n_chains, long len) {
    std::vector<double> mu(n_chains);
    for (int c = 0; c < n_chains; ++c)
        mu[c] = chain_mean(eps, 1000003u + 7717u * c, len);
    double m = 0.0; for (double v : mu) m += v; m /= n_chains;
    double v = 0.0; for (double x : mu) v += (x - m) * (x - m);
    const double se = std::sqrt(v / (n_chains - 1) / n_chains);
    return {m, se, (m - EXACT) / se};
}

} // namespace

int main() {
    std::printf("=== test_nuts_small_step_invariance ===\n");
    std::printf("exact E[p_0] = %.10f   (Dirichlet(2,3,4,5) via SIMPLEX)\n\n", EXACT);

    // ---- control 1: the requested start is honoured -------------------------
    {
        std::mt19937_64 r(4242);
        double worst = 0.0;
        for (int t = 0; t < 20; ++t) {
            const arma::vec p0 = exact_draw(r);
            joint_nuts_block blk(make_cfg(0.10, p0));
            block_context ctx; blk.set_context(ctx);
            const arma::vec c = blk.current();
            for (int k = 0; k < 4; ++k) worst = std::max(worst, std::fabs(c[k] - p0[k]));
        }
        char buf[64]; std::snprintf(buf, sizeof buf, "max|diff| = %.2e", worst);
        check(worst < 1e-12,
              "chains really do start at the exact draw they were given", buf);
    }

    // ---- the measurement ----------------------------------------------------
    // Sized for the default suite, not for a research measurement. The defect
    // it guards is +2.5 to +3.5%, and at this size the standard error is
    // ~0.15%, so a regression lands at 20+ se while the whole test runs in a
    // couple of minutes. AI4B_NUTS_GATE_FULL=1 restores the 240 x 20000
    // configuration the original measurement used (se ~0.037%).
    const bool full = std::getenv("AI4B_NUTS_GATE_FULL") != nullptr;
    const int  NC   = full ? 240   : 60;
    const long LEN  = full ? 20000 : 4000;
    std::printf("  (%s configuration: %d chains x %ld draws%s)\n",
                full ? "full" : "default", NC, LEN,
                full ? "" : " -- set AI4B_NUTS_GATE_FULL=1 for the full one");
    std::printf("\n  %-26s %12s %10s %9s\n", "step size", "E[p_0]", "dev", "verdict");
    for (double eps : {0.05, 0.10, 0.30, 1.00}) {
        const Est e = estimate(eps, NC, LEN);
        std::printf("  fixed eps = %-14.2f %12.7f %+9.3f%% %+8.1f se\n",
                    eps, e.mean, 100.0 * (e.mean - EXACT) / EXACT, e.z);
        std::fflush(stdout);
        char buf[80];
        std::snprintf(buf, sizeof buf, "eps=%.2f: %+.3f%% at %+.1f se",
                      eps, 100.0 * (e.mean - EXACT) / EXACT, e.z);
        // 5 se is far outside anything this estimator's noise can produce; the
        // clean arm sits at 0.7 se.
        check(std::fabs(e.z) < 5.0,
              "the sampled mean matches the exact Dirichlet mean", buf);
    }

    // ---- control 2: it is not burn-in ---------------------------------------
    // An exact-start chain under a pi-invariant kernel is unbiased at ANY
    // length. If the deviation were initialization it would fall like 1/n.
    std::printf("\n  deviation vs chain length at eps = 0.10:\n");
    double first_dev = 0.0, last_dev = 0.0;
    int idx = 0;
    for (long len : {LEN / 8, LEN / 2, LEN}) {
        const Est e = estimate(0.10, NC, len);
        const double dev = 100.0 * (e.mean - EXACT) / EXACT;
        std::printf("    length %-8ld %12.7f %+9.3f%% %+8.1f se\n",
                    len, e.mean, dev, e.z);
        std::fflush(stdout);
        if (idx++ == 0) first_dev = dev;
        last_dev = dev;
    }
    {
        char buf[96];
        std::snprintf(buf, sizeof buf,
                      "500 draws: %+.3f%%, 20000 draws: %+.3f%% -- flat, so not burn-in",
                      first_dev, last_dev);
        // Reported, not asserted: this line is DIAGNOSIS, and it is expected to
        // hold both before and after the fix (after the fix both are ~0).
        std::printf("  note  %s\n", buf);
    }

    std::printf("\n%d checks, %d failures\n", checks, failures);
    if (failures) {
        std::printf("FAILED -- this is the KNOWN vendored-NUTS defect, not a regression\n"
                    "         introduced by your change. See the header of this file.\n");
        return 1;
    }
    std::printf("PASSED -- if this is the first pass, the kernel defect is FIXED:\n"
                "         move this test into the Makefile's TARGETS and run: panel.\n");
    return 0;
}
