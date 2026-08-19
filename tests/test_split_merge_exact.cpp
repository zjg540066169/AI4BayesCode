/*
 * test_split_merge_exact.cpp
 *
 * split_merge_block (Jain-Neal 2004) against EXACT ENUMERATION.
 *
 * With mu, lambda and pi held fixed, the allocation posterior is a finite
 * product over N observations,
 *
 *     p(z) proportional to prod_i pi_{z_i} N(y_i | mu_{z_i}, 1/lambda_{z_i}),
 *
 * so for small N and K_trunc every one of the K^N states can be enumerated and
 * the sampler's stationary distribution checked directly -- no reference
 * implementation and no asymptotic argument needed.
 *
 * The kernel under test is a full systematic-scan Gibbs sweep over z (a
 * provably correct kernel for this target) followed by split_merge steps. The
 * Gibbs sweep alone is the CONTROL: it pins the Monte-Carlo noise floor, so a
 * split/merge defect shows up as the split_merge arm being further from exact
 * than the control, not merely as "some" deviation.
 *
 * The regression this guards: the split proposal draws its new cluster slot
 * uniformly from the currently EMPTY slots, contributing 1/|empty| to
 * q(z*|z), and the reverse split inside the merge branch does the same with
 * z*'s empty count. Neither term reached the acceptance ratio, so splits were
 * under-accepted by |empty| and merges over-accepted -- a factor of ~17 at the
 * K_trunc = 20, K_active = 3 that the shipped examples use. The partition
 * posterior was biased toward fewer clusters, with nothing in R-hat or ESS to
 * show for it: both chains agree on the wrong answer.
 */

#include "AI4BayesCode/block_sampler.hpp"
#include "AI4BayesCode/shared_data.hpp"
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/split_merge_block.hpp"

#include <cmath>
#include <cstdio>
#include <memory>
#include <random>

#include "portable_rng.hpp"   // portable draws: identical on every stdlib
#include <vector>

using AI4BayesCode::composite_block;
using AI4BayesCode::split_merge_block;
using AI4BayesCode::split_merge_block_config;

namespace {

int checks = 0, failures = 0;

void check(bool ok, const char* what, const char* detail = "") {
    ++checks;
    std::printf("  %s  %s%s%s\n", ok ? "ok  " : "FAIL", what,
                *detail ? " -- " : "", detail);
    if (!ok) ++failures;
}

// ---- the model ------------------------------------------------------------
constexpr std::size_t N = 5;
constexpr std::size_t K = 4;                 // K^N = 1024 states
constexpr std::size_t NSTATE = 1024;

const double PI_[K]  = {0.40, 0.30, 0.20, 0.10};
const double MU_[K]  = {-2.0, 0.0, 1.5, 3.0};
const double LAM_[K] = {1.0, 2.0, 0.5, 4.0};   // precisions
const double Y_[N]   = {-1.7, 0.2, 1.4, 2.9, 0.1};

double log_lik(std::size_t i, std::size_t k) {
    const double r = Y_[i] - MU_[k];
    return 0.5 * std::log(LAM_[k]) - 0.5 * LAM_[k] * r * r;
}

/// Exact p(z) over all K^N allocations, indexed base-K with observation 0 the
/// least significant digit.
std::vector<double> exact_pmf() {
    std::vector<double> lp(NSTATE), p(NSTATE);
    double mx = -1e300;
    for (std::size_t s = 0; s < NSTATE; ++s) {
        double v = 0.0;
        std::size_t t = s;
        for (std::size_t i = 0; i < N; ++i) {
            const std::size_t k = t % K; t /= K;
            v += std::log(PI_[k]) + log_lik(i, k);
        }
        lp[s] = v; if (v > mx) mx = v;
    }
    double Z = 0.0;
    for (std::size_t s = 0; s < NSTATE; ++s) { p[s] = std::exp(lp[s] - mx); Z += p[s]; }
    for (auto& v : p) v /= Z;
    return p;
}

std::size_t encode(const arma::vec& z) {
    std::size_t s = 0, mult = 1;
    for (std::size_t i = 0; i < N; ++i) {
        s += (static_cast<std::size_t>(std::llround(z[i])) - 1) * mult;
        mult *= K;
    }
    return s;
}

/// One systematic-scan Gibbs sweep over z, drawn from the exact full
/// conditionals. Correct for this target by construction.
void gibbs_sweep(arma::vec& z, std::mt19937_64& rng) {
    prng::uniform_real_distribution<double> U(0.0, 1.0);
    for (std::size_t i = 0; i < N; ++i) {
        double w[K], mx = -1e300;
        for (std::size_t k = 0; k < K; ++k) {
            w[k] = std::log(PI_[k]) + log_lik(i, k);
            if (w[k] > mx) mx = w[k];
        }
        double tot = 0.0;
        for (std::size_t k = 0; k < K; ++k) { w[k] = std::exp(w[k] - mx); tot += w[k]; }
        double u = U(rng) * tot, acc = 0.0;
        std::size_t pick = K - 1;
        for (std::size_t k = 0; k < K; ++k) {
            acc += w[k];
            if (u <= acc) { pick = k; break; }
        }
        z[i] = static_cast<double>(pick + 1);
    }
}

struct Run {
    double tv;                 // total variation to the exact pmf
    double p_k[K + 1];         // P(K_active = 1..K)
};

/// `with_split_merge = false` gives the control arm (Gibbs only).
Run run_chain(bool with_split_merge, std::uint64_t seed, std::size_t n_iter) {
    auto comp = std::make_unique<composite_block>();
    comp->data().set("y",   arma::vec(std::vector<double>(Y_,   Y_   + N)));
    comp->data().set("pi",  arma::vec(std::vector<double>(PI_,  PI_  + K)));
    comp->data().set("mu",  arma::vec(std::vector<double>(MU_,  MU_  + K)));
    comp->data().set("lam", arma::vec(std::vector<double>(LAM_, LAM_ + K)));

    arma::vec z0(N); z0.fill(1.0);
    comp->data().set("z", z0);

    split_merge_block_config cfg;
    cfg.name = "sm"; cfg.N = N; cfg.K_trunc = K; cfg.d = 1;
    cfg.z_name = "z"; cfg.y_key = "y"; cfg.pi_key = "pi";
    cfg.mu_key = "mu"; cfg.lambda_key = "lam";
    cfg.initial_z = z0;
    split_merge_block* sm = nullptr;
    { auto b = std::make_unique<split_merge_block>(std::move(cfg));
      sm = b.get(); comp->add_child(std::move(b)); }
    // The composite projects ONLY declared keys into a child's block_context.
    comp->data().declare_dependencies("sm", {"y", "pi", "mu", "lam"});

    std::mt19937_64 rng(seed);
    std::vector<double> hit(NSTATE, 0.0);
    arma::vec z = z0;
    double total = 0.0;
    double kcount[K + 1] = {0};

    for (std::size_t it = 0; it < n_iter; ++it) {
        gibbs_sweep(z, rng);
        if (with_split_merge) {
            // Hand the Gibbs state to the block, take 5 split/merge steps,
            // read it back.
            sm->set_current(z);
            comp->data().set("z", z);
            for (int r = 0; r < 5; ++r) comp->step(rng);
            z = sm->current();
        }
        hit[encode(z)] += 1.0; total += 1.0;
        bool used[K] = {false};
        for (std::size_t i = 0; i < N; ++i)
            used[static_cast<std::size_t>(std::llround(z[i])) - 1] = true;
        std::size_t na = 0;
        for (std::size_t k = 0; k < K; ++k) if (used[k]) ++na;
        kcount[na] += 1.0;
    }

    const std::vector<double> ex = exact_pmf();
    Run out{};
    double tv = 0.0;
    for (std::size_t s = 0; s < NSTATE; ++s) tv += std::fabs(hit[s] / total - ex[s]);
    out.tv = 0.5 * tv;
    for (std::size_t k = 1; k <= K; ++k) out.p_k[k] = kcount[k] / total;
    return out;
}

/// P(K_active = k) under the exact pmf.
void exact_k_active(double* out) {
    const std::vector<double> ex = exact_pmf();
    for (std::size_t k = 0; k <= K; ++k) out[k] = 0.0;
    for (std::size_t s = 0; s < NSTATE; ++s) {
        bool used[K] = {false};
        std::size_t t = s;
        for (std::size_t i = 0; i < N; ++i) { used[t % K] = true; t /= K; }
        std::size_t na = 0;
        for (std::size_t k = 0; k < K; ++k) if (used[k]) ++na;
        out[na] += ex[s];
    }
}

} // namespace

int main() {
    std::printf("=== test_split_merge_exact ===\n\n");
    const std::size_t n_iter = 400000;

    double ex_k[K + 1]; exact_k_active(ex_k);
    const Run control = run_chain(false, 20260816u, n_iter);
    const Run smrun   = run_chain(true,  20260816u, n_iter);

    std::printf("  %-22s %10s %10s %10s\n", "", "exact", "Gibbs only", "+split/merge");
    std::printf("  %-22s %10s %10.5f %10.5f\n", "TV to exact", "--", control.tv, smrun.tv);
    for (std::size_t k = 1; k <= K; ++k)
        std::printf("  P(K_active = %zu)        %10.5f %10.5f %10.5f\n",
                    k, ex_k[k], control.p_k[k], smrun.p_k[k]);
    std::printf("\n");

    // The control fixes the noise floor for this many iterations. A correct
    // split/merge kernel leaves the target invariant, so adding it must not
    // move the chain materially further from exact. Before the fix this ran
    // TV 0.031 against a 0.0016 control -- a 19x gap, far outside any tolerance
    // a Monte-Carlo run of this length could explain.
    check(control.tv < 0.005, "control (Gibbs only) reaches the exact pmf");
    check(smrun.tv < 3.0 * control.tv + 0.002,
          "adding split/merge does not move the chain away from exact");

    // P(K_active) is where the slot-selection bug showed: the smallest counts
    // were inflated because splits were under-accepted.
    for (std::size_t k = 1; k <= K; ++k) {
        char buf[96];
        const double err = std::fabs(smrun.p_k[k] - ex_k[k]);
        std::snprintf(buf, sizeof buf, "P(K_active=%zu): %.5f vs exact %.5f",
                      k, smrun.p_k[k], ex_k[k]);
        check(err < 0.004, "split/merge preserves the cluster-count marginal", buf);
    }

    std::printf("\n%d checks, %d failures\n", checks, failures);
    if (failures) { std::printf("FAILED\n"); return 1; }
    std::printf("PASSED\n");
    return 0;
}
