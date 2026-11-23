// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_dp-mixture_atom_swap_exact.cpp
//
//  cluster_atom_block together with stick_label_swap_block. The swap block
//  permutes the shared atoms, so the atom block must start its next update
//  from the shared atoms, not from its own pre-swap state. Tiny truncated DP
//  mixture: K = 4 slots, n = 6, y_i | z, mu ~ N(mu_{z_i}, 1), mu_k ~ N(0, 4),
//  alpha = 0.5 fixed. The exact posterior over all 4^6 labelled allocations
//  gives P(z_i = k) and E[mu_k^2]. The composite runs categorical z,
//  cluster_atom_block for mu | z (with "mu" among its dependencies),
//  stick-breaking pi and the label swap, as in examples/DPGammaMixture.cpp.
//  The same composite without the swap calibrates the Monte Carlo error.
// ============================================================================
#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif
#include <armadillo>
#include "AI4BayesCode/bnp_utils.hpp"
#include "AI4BayesCode/categorical_gibbs_block.hpp"
#include "AI4BayesCode/cluster_atom_block.hpp"
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/stick_breaking_block.hpp"
#include "AI4BayesCode/stick_label_swap_block.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace AI4BayesCode;

namespace {

constexpr int K = 4, N = 6;
const double Y[N] = {-2.1, -1.8, -2.4, 1.9, 2.2, 0.1};
constexpr double S2 = 1.0, T2 = 4.0, ALPHA = 0.5;
int g_fail = 0;

void check(bool ok, const char* what, double value, double bound) {
    std::printf("  [%s] %-50s %.4f (bound %.4f)\n", ok ? "PASS" : "FAIL", what,
                value, bound);
    if (!ok) ++g_fail;
}

bool same_bits(const arma::vec& a, const arma::vec& b) {
    return a.n_elem == b.n_elem &&
           std::memcmp(a.memptr(), b.memptr(), a.n_elem * sizeof(double)) == 0;
}

double lbeta(double a, double b) {
    return std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b);
}

// Exact P(z_i = k) (marg, N x K) and E[mu_k^2] (m2, K).
void exact(std::vector<double>& marg, std::vector<double>& m2) {
    const int NS = 1 << (2 * N);
    std::vector<double> logw(NS), e2(NS * K);
    double mx = -1e300;
    for (int code = 0; code < NS; ++code) {
        double nk[K] = {0}, S[K] = {0}, Q[K] = {0};
        for (int i = 0; i < N; ++i) {
            const int k = (code >> (2 * i)) & 3;
            nk[k] += 1; S[k] += Y[i]; Q[k] += Y[i] * Y[i];
        }
        double lp = 0.0, tail = 0.0;
        for (int k = K - 1; k >= 0; --k) {
            if (k < K - 1) lp += lbeta(1.0 + nk[k], ALPHA + tail) - lbeta(1.0, ALPHA);
            tail += nk[k];
        }
        for (int k = 0; k < K; ++k) {
            if (nk[k] > 0) {
                const double m = nk[k];
                lp += -0.5 * m * std::log(2 * M_PI * S2) - 0.5 * std::log(1 + m * T2 / S2)
                      - 0.5 / S2 * (Q[k] - T2 * S[k] * S[k] / (S2 + m * T2));
                const double v = 1.0 / (1.0 / T2 + m / S2), mean = v * S[k] / S2;
                e2[code * K + k] = mean * mean + v;
            } else {
                e2[code * K + k] = T2;
            }
        }
        logw[code] = lp;
        mx = std::max(mx, lp);
    }
    marg.assign(N * K, 0.0);
    m2.assign(K, 0.0);
    double tot = 0.0;
    for (int code = 0; code < NS; ++code) {
        const double w = std::exp(logw[code] - mx);
        tot += w;
        for (int i = 0; i < N; ++i) marg[i * K + ((code >> (2 * i)) & 3)] += w;
        for (int k = 0; k < K; ++k) m2[k] += w * e2[code * K + k];
    }
    for (auto& v : marg) v /= tot;
    for (auto& v : m2) v /= tot;
}

double atom_lp(const arma::vec& th, std::size_t k, const block_context& ctx) {
    const double mu = th[0];
    const arma::vec& z = ctx.at("z");
    const arma::vec& y = ctx.at("y");
    double lp = -0.5 * mu * mu / T2;
    for (std::size_t i = 0; i < z.n_elem; ++i)
        if (static_cast<std::size_t>(std::llround(z[i])) == k + 1)
            lp += -0.5 * (y[i] - mu) * (y[i] - mu) / S2;
    return lp;
}

std::unique_ptr<composite_block> build(bool with_swap, bool atom_reads_mu) {
    auto comp = std::make_unique<composite_block>("dp");
    arma::vec y(N), z0(N), mu0(K, arma::fill::zeros), pi0(K), V0(K);
    for (int i = 0; i < N; ++i) { y[i] = Y[i]; z0[i] = (i % K) + 1; }
    for (int k = 0; k < K; ++k) pi0[k] = 1.0 / K;
    for (int k = 0; k + 1 < K; ++k) V0[k] = 1.0 / (K - k);
    V0[K - 1] = 1.0;
    auto& d = comp->data();
    d.set("y", y); d.set("z", z0); d.set("mu", mu0); d.set("pi", pi0);
    d.set("stick_V", V0); d.set("alpha", arma::vec{ALPHA});
    d.set("cluster_counts", arma::vec(K, arma::fill::zeros));
    d.register_refresher("cluster_counts", [](const shared_data_t& s) -> arma::vec {
        return bnp::counts_from_z(s.get("z"), K);
    });
    d.declare_dependencies("z", {"y", "pi", "mu"});
    d.declare_invalidates("z", {"cluster_counts"});
    if (atom_reads_mu) d.declare_dependencies("cluster_params", {"z", "y", "cluster_counts", "mu"});
    else               d.declare_dependencies("cluster_params", {"z", "y", "cluster_counts"});
    d.declare_dependencies("pi", {"cluster_counts", "alpha"});
    if (with_swap) {
        d.declare_dependencies("label_swap", {"z", "mu", "pi", "stick_V", "alpha"});
        d.declare_invalidates("label_swap", {"cluster_counts"});
    }
    d.refresh_all();
    {
        categorical_gibbs_block_config cfg;
        cfg.name = "z"; cfg.n_obs = N; cfg.n_categories = K; cfg.initial_labels = z0;
        cfg.log_probs_fn = [](const block_context& ctx) -> arma::mat {
            const arma::vec& yy = ctx.at("y");
            const arma::vec& pi = ctx.at("pi");
            const arma::vec& mu = ctx.at("mu");
            arma::mat lp(N, K);
            for (int i = 0; i < N; ++i)
                for (int k = 0; k < K; ++k)
                    lp(i, k) = std::log(pi[k] + 1e-300) -
                               0.5 * (yy[i] - mu[k]) * (yy[i] - mu[k]) / S2;
            return lp;
        };
        comp->add_child(std::make_unique<categorical_gibbs_block>(std::move(cfg)));
    }
    {
        cluster_atom_block_config cfg;
        cfg.name = "cluster_params"; cfg.K_trunc = K;
        cfg.sub_params.push_back(joint_nuts_sub_param{"mu", 1u, joint_constraint::REAL});
        cfg.initial_cat = mu0;
        cfg.log_density = &atom_lp;
        comp->add_child(std::make_unique<cluster_atom_block>(std::move(cfg)));
    }
    {
        stick_breaking_block_config cfg;
        cfg.name = "pi"; cfg.K_trunc = K; cfg.counts_key = "cluster_counts";
        cfg.v_name = "stick_V";
        cfg.a_fn = [](std::size_t k, const arma::vec& c, const block_context&) {
            return 1.0 + c[k];
        };
        cfg.b_fn = [](std::size_t k, const arma::vec& c, const block_context& ctx) {
            double tail = 0.0;
            for (std::size_t j = k + 1; j < c.n_elem; ++j) tail += c[j];
            return ctx.at("alpha")[0] + tail;
        };
        cfg.initial_pi = pi0;
        comp->add_child(std::make_unique<stick_breaking_block>(std::move(cfg)));
    }
    if (with_swap) {
        stick_label_swap_block_config cfg;
        cfg.name = "label_swap"; cfg.K_trunc = K; cfg.atom_keys = {{"mu", 1}};
        comp->add_child(std::make_unique<stick_label_swap_block>(std::move(cfg)));
    }
    return comp;
}

struct Est { std::vector<double> marg, m2; };

// End-of-sweep shared z and mu.
Est run(bool with_swap, unsigned seed, long burn, long sweeps) {
    auto comp = build(with_swap, /*atom_reads_mu=*/true);
    std::mt19937_64 rng(seed);
    Est e;
    e.marg.assign(N * K, 0.0);
    e.m2.assign(K, 0.0);
    for (long s = 0; s < burn + sweeps; ++s) {
        comp->step(rng);
        if (s < burn) continue;
        const arma::vec& z = comp->data().get("z");
        const arma::vec& mu = comp->data().get("mu");
        for (int i = 0; i < N; ++i) e.marg[i * K + std::lround(z[i]) - 1] += 1.0;
        for (int k = 0; k < K; ++k) e.m2[k] += mu[k] * mu[k];
    }
    for (auto& v : e.marg) v /= double(sweeps);
    for (auto& v : e.m2) v /= double(sweeps);
    return e;
}

double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
    double m = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

}  // namespace

int main() {
    std::printf("=== test_fix_dp-mixture_atom_swap_exact ===\n");
    std::vector<double> ex_marg, ex_m2;
    exact(ex_marg, ex_m2);
    std::printf("  exact E[mu_k^2]:");
    for (int k = 0; k < K; ++k) std::printf(" %.4f", ex_m2[k]);
    std::printf("\n");

    const long burn = 5000, sweeps = 400000;
    const Est ctrl = run(false, 1000u, burn, sweeps);
    const Est swap = run(true, 1000u, burn, sweeps);
    std::printf("  chain E[mu_k^2], no swap:  ");
    for (int k = 0; k < K; ++k) std::printf(" %.4f", ctrl.m2[k]);
    std::printf("\n  chain E[mu_k^2], with swap:");
    for (int k = 0; k < K; ++k) std::printf(" %.4f", swap.m2[k]);
    std::printf("\n");

    check(max_abs_diff(ctrl.marg, ex_marg) < 0.012,
          "no swap: max |P(z_i=k) - exact|", max_abs_diff(ctrl.marg, ex_marg), 0.012);
    check(max_abs_diff(ctrl.m2, ex_m2) < 0.06,
          "no swap: max |E[mu_k^2] - exact|", max_abs_diff(ctrl.m2, ex_m2), 0.06);
    check(max_abs_diff(swap.marg, ex_marg) < 0.012,
          "with swap: max |P(z_i=k) - exact|", max_abs_diff(swap.marg, ex_marg), 0.012);
    check(max_abs_diff(swap.m2, ex_m2) < 0.06,
          "with swap: max |E[mu_k^2] - exact|", max_abs_diff(swap.m2, ex_m2), 0.06);

    // Without a second writer of "mu", reading it back changes nothing: the
    // shared atoms are this block's own output.
    {
        auto a = build(false, false), b = build(false, true);
        std::mt19937_64 ra(77u), rb(77u);
        for (int s = 0; s < 20000; ++s) { a->step(ra); b->step(rb); }
        bool same = ra() == rb();
        for (const char* key : {"z", "mu", "pi", "stick_V"})
            same = same && same_bits(a->data().get(key), b->data().get(key));
        std::printf("  [%s] no second writer: 20000 sweeps bit-identical with \"mu\" declared\n",
                    same ? "PASS" : "FAIL");
        if (!same) ++g_fail;
    }

    std::printf("\n=== SUMMARY: %d failed ===\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
