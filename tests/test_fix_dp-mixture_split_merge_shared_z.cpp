// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_dp-mixture_split_merge_shared_z.cpp
//
//  (1) split_merge_block after categorical_gibbs_block, both writing "z", with
//      stick_label_swap_block in the sweep. split_merge must start each step
//      from the shared z (the categorical draw, relabelled by the swap), not
//      from its own previous z. Truncated DP mixture, K = 4 slots, n = 6,
//      d = 1, Normal-Gamma atoms, alpha = 0.5 fixed; the exact posterior over
//      all 4^6 labelled allocations gives the partition distribution, the
//      co-clustering probabilities and P(K_active). The same composite
//      without split_merge calibrates the Monte Carlo error.
//  (2) Full-covariance emissions (sigma_key), d = 2, with pi, mu and Sigma
//      fixed: Gibbs sweeps plus split/merge steps against exact enumeration
//      of the 3^5 allocations.
//  (3) A z in the context is checked (length, labels) before it is loaded.
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
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/normal_gamma_cluster_gibbs_block.hpp"
#include "AI4BayesCode/split_merge_block.hpp"
#include "AI4BayesCode/stick_breaking_block.hpp"
#include "AI4BayesCode/stick_label_swap_block.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "portable_rng.hpp"

using namespace AI4BayesCode;

namespace {

int g_fail = 0;

void check(bool ok, const char* what, double value, double bound) {
    std::printf("  [%s] %-52s %.4f (bound %.4f)\n", ok ? "PASS" : "FAIL", what,
                value, bound);
    if (!ok) ++g_fail;
}

void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++g_fail;
}

double lbeta(double a, double b) {
    return std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b);
}

// ---------------------------------------------------------------- part (1)
constexpr std::size_t N1 = 6, K1 = 4;
const double Y1[N1] = {-2.1, -1.8, -2.4, 1.9, 2.2, 0.1};
constexpr double MU0 = 0.0, KAPPA0 = 0.25, A0 = 2.0, B0 = 2.0, ALPHA = 0.5;

double log_ng_marg(const std::vector<double>& ys) {
    if (ys.empty()) return 0.0;
    const double n = static_cast<double>(ys.size());
    double s = 0.0;
    for (double v : ys) s += v;
    const double ybar = s / n;
    double ss = 0.0;
    for (double v : ys) ss += (v - ybar) * (v - ybar);
    const double kn = KAPPA0 + n, an = A0 + 0.5 * n;
    const double bn = B0 + 0.5 * ss + 0.5 * KAPPA0 * n * (ybar - MU0) * (ybar - MU0) / kn;
    return std::lgamma(an) - std::lgamma(A0) + A0 * std::log(B0) - an * std::log(bn)
         + 0.5 * std::log(KAPPA0 / kn) - 0.5 * n * std::log(2.0 * M_PI);
}

std::size_t n_states(std::size_t N, std::size_t K) {
    std::size_t s = 1;
    for (std::size_t i = 0; i < N; ++i) s *= K;
    return s;
}

std::vector<double> exact_dp_pmf() {
    const std::size_t NS = n_states(N1, K1);
    std::vector<double> lp(NS), p(NS);
    double mx = -1e300;
    for (std::size_t code = 0; code < NS; ++code) {
        std::size_t t = code;
        std::vector<double> nk(K1, 0.0);
        std::vector<std::vector<double>> ys(K1);
        for (std::size_t i = 0; i < N1; ++i) {
            const std::size_t k = t % K1;
            t /= K1;
            nk[k] += 1.0;
            ys[k].push_back(Y1[i]);
        }
        double v = 0.0, tail = 0.0;
        for (long k = static_cast<long>(K1) - 1; k >= 0; --k) {
            if (k < static_cast<long>(K1) - 1)
                v += lbeta(1.0 + nk[k], ALPHA + tail) - lbeta(1.0, ALPHA);
            tail += nk[k];
        }
        for (std::size_t k = 0; k < K1; ++k) v += log_ng_marg(ys[k]);
        lp[code] = v;
        mx = std::max(mx, v);
    }
    double Z = 0.0;
    for (std::size_t s = 0; s < NS; ++s) { p[s] = std::exp(lp[s] - mx); Z += p[s]; }
    for (auto& x : p) x /= Z;
    return p;
}

std::size_t encode(const arma::vec& z, std::size_t K) {
    std::size_t s = 0, m = 1;
    for (std::size_t i = 0; i < z.n_elem; ++i) {
        s += (static_cast<std::size_t>(std::llround(z[i])) - 1) * m;
        m *= K;
    }
    return s;
}

struct Summary {
    std::map<std::vector<int>, double> part;   // canonical partition -> prob
    std::vector<double> cocl;                  // P(z_i == z_j), N x N
    std::vector<double> pk;                    // P(K_active = k), k = 0..K
};

Summary summarise(const std::vector<double>& pmf, std::size_t N, std::size_t K) {
    Summary S;
    S.cocl.assign(N * N, 0.0);
    S.pk.assign(K + 1, 0.0);
    for (std::size_t code = 0; code < pmf.size(); ++code) {
        const double w = pmf[code];
        if (w == 0.0) continue;
        std::vector<int> z(N), canon(N), map(K, -1);
        std::size_t t = code;
        for (std::size_t i = 0; i < N; ++i) { z[i] = static_cast<int>(t % K); t /= K; }
        int next = 0;
        for (std::size_t i = 0; i < N; ++i) {
            if (map[z[i]] < 0) map[z[i]] = next++;
            canon[i] = map[z[i]];
        }
        S.part[canon] += w;
        S.pk[static_cast<std::size_t>(next)] += w;
        for (std::size_t i = 0; i < N; ++i)
            for (std::size_t j = 0; j < N; ++j)
                if (z[i] == z[j]) S.cocl[i * N + j] += w;
    }
    return S;
}

double tv_partitions(const Summary& a, const Summary& b) {
    double tv = 0.0;
    for (const auto& kv : a.part) {
        const auto it = b.part.find(kv.first);
        tv += std::fabs(kv.second - (it == b.part.end() ? 0.0 : it->second));
    }
    for (const auto& kv : b.part)
        if (!a.part.count(kv.first)) tv += kv.second;
    return 0.5 * tv;
}

double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
    double m = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

std::unique_ptr<composite_block> build_dp(bool with_sm) {
    const std::size_t N = N1, K = K1, d = 1;
    auto comp = std::make_unique<composite_block>("dpmm");
    auto& D = comp->data();
    arma::vec y(N), z0(N), pi0(K, arma::fill::value(1.0 / K)), V0(K);
    for (std::size_t i = 0; i < N; ++i) { y[i] = Y1[i]; z0[i] = static_cast<double>(i % K + 1); }
    double rem = 1.0;
    for (std::size_t k = 0; k + 1 < K; ++k) { V0[k] = pi0[k] / rem; rem *= 1.0 - V0[k]; }
    V0[K - 1] = 1.0;
    arma::vec mu0(K * d), lam0(K * d, arma::fill::value(A0 / B0));
    for (std::size_t k = 0; k < K; ++k) mu0[k] = Y1[(k * N) / K];
    D.set("y", y); D.set("z", z0); D.set("pi", pi0); D.set("stick_V", V0);
    D.set("mu", mu0); D.set("lambda", lam0); D.set("alpha", arma::vec{ALPHA});
    D.set("cluster_counts", arma::vec(K, arma::fill::zeros));
    D.register_refresher("cluster_counts", [](const shared_data_t& s) -> arma::vec {
        return bnp::counts_from_z(s.get("z"), K1);
    });
    D.declare_dependencies("z", {"y", "pi", "mu", "lambda"});
    D.declare_invalidates("z", {"cluster_counts"});
    if (with_sm) {
        D.declare_dependencies("sm", {"y", "pi", "mu", "lambda", "z"});
        D.declare_invalidates("sm", {"cluster_counts"});
    }
    D.declare_dependencies("cluster_params", {"z", "y"});
    D.declare_dependencies("pi", {"cluster_counts", "alpha"});
    D.declare_dependencies("label_swap", {"z", "mu", "lambda", "pi", "stick_V", "alpha"});
    D.declare_invalidates("label_swap", {"cluster_counts"});
    D.refresh_all();
    {
        categorical_gibbs_block_config c;
        c.name = "z"; c.n_obs = N; c.n_categories = K; c.initial_labels = z0;
        c.log_probs_fn = [](const block_context& ctx) -> arma::mat {
            const arma::vec& yy = ctx.at("y");
            const arma::vec& pi = ctx.at("pi");
            const arma::vec& mu = ctx.at("mu");
            const arma::vec& lam = ctx.at("lambda");
            arma::mat lp(N1, K1);
            for (std::size_t i = 0; i < N1; ++i)
                for (std::size_t k = 0; k < K1; ++k) {
                    const double r = yy[i] - mu[k];
                    lp(i, k) = std::log(pi[k] + 1e-300) + 0.5 * std::log(lam[k]) -
                               0.5 * lam[k] * r * r;
                }
            return lp;
        };
        comp->add_child(std::make_unique<categorical_gibbs_block>(std::move(c)));
    }
    if (with_sm) {
        split_merge_block_config c;
        c.name = "sm"; c.N = N; c.K_trunc = K; c.d = d; c.z_name = "z";
        c.y_key = "y"; c.pi_key = "pi"; c.mu_key = "mu"; c.lambda_key = "lambda";
        c.n_restricted_gibbs_iters = 5; c.initial_z = z0;
        comp->add_child(std::make_unique<split_merge_block>(std::move(c)));
    }
    {
        normal_gamma_cluster_gibbs_block_config c;
        c.name = "cluster_params"; c.K_trunc = K; c.d = d; c.N = N;
        c.z_key = "z"; c.y_key = "y"; c.mu_name = "mu"; c.lambda_name = "lambda";
        c.mu_0 = arma::vec(d, arma::fill::value(MU0)); c.kappa_0 = KAPPA0;
        c.a_lambda_0 = A0; c.b_lambda_0 = B0;
        c.initial_mu = mu0; c.initial_lambda = lam0;
        comp->add_child(std::make_unique<normal_gamma_cluster_gibbs_block>(std::move(c)));
    }
    {
        stick_breaking_block_config c;
        c.name = "pi"; c.K_trunc = K; c.counts_key = "cluster_counts"; c.v_name = "stick_V";
        c.a_fn = [](std::size_t k, const arma::vec& counts, const block_context&) {
            return 1.0 + counts[k];
        };
        c.b_fn = [](std::size_t k, const arma::vec& counts, const block_context& ctx) {
            double t = 0.0;
            for (std::size_t j = k + 1; j < counts.n_elem; ++j) t += counts[j];
            return ctx.at("alpha")[0] + t;
        };
        c.initial_pi = pi0;
        comp->add_child(std::make_unique<stick_breaking_block>(std::move(c)));
    }
    {
        stick_label_swap_block_config c;
        c.name = "label_swap"; c.K_trunc = K; c.atom_keys = {{"mu", d}, {"lambda", d}};
        comp->add_child(std::make_unique<stick_label_swap_block>(std::move(c)));
    }
    if (with_sm) comp->declare_shared_history("z");
    return comp;
}

// Distribution of the end-of-sweep shared z over all labelled states.
std::vector<double> run_dp(bool with_sm, unsigned seed, long burn, long sweeps) {
    auto comp = build_dp(with_sm);
    std::mt19937_64 rng(seed);
    std::vector<double> hist(n_states(N1, K1), 0.0);
    for (long s = 0; s < burn + sweeps; ++s) {
        comp->step(rng);
        if (s >= burn) hist[encode(comp->data().get("z"), K1)] += 1.0;
    }
    for (auto& h : hist) h /= static_cast<double>(sweeps);
    return hist;
}

// ---------------------------------------------------------------- part (2)
constexpr std::size_t N2 = 5, K2 = 3, D2 = 2;
const double Y2[N2 * D2] = {-1.2, -0.8, -0.6, -1.4, 0.3, 0.2, 1.1, 1.3, 1.6, 0.7};
const double PI2[K2] = {0.45, 0.35, 0.20};
const double MU2[K2 * D2] = {-1.0, -1.0, 1.2, 1.0, 0.0, 0.0};
// Row-major 2 x 2 covariances, with correlation.
const double SIG2[K2 * D2 * D2] = {0.5, 0.2, 0.2, 0.4,
                                   0.6, -0.25, -0.25, 0.5,
                                   2.0, 0.8, 0.8, 1.5};

double log_mvn2(std::size_t i, std::size_t k) {
    const double* S = SIG2 + k * 4;
    const double det = S[0] * S[3] - S[1] * S[2];
    const double a = Y2[i * 2] - MU2[k * 2], b = Y2[i * 2 + 1] - MU2[k * 2 + 1];
    const double q = (S[3] * a * a - 2.0 * S[1] * a * b + S[0] * b * b) / det;
    return -std::log(2.0 * M_PI) - 0.5 * std::log(det) - 0.5 * q;
}

std::vector<double> exact_fixed_pmf() {
    const std::size_t NS = n_states(N2, K2);
    std::vector<double> p(NS);
    double Z = 0.0;
    for (std::size_t code = 0; code < NS; ++code) {
        std::size_t t = code;
        double lp = 0.0;
        for (std::size_t i = 0; i < N2; ++i) {
            const std::size_t k = t % K2;
            t /= K2;
            lp += std::log(PI2[k]) + log_mvn2(i, k);
        }
        p[code] = std::exp(lp);
        Z += p[code];
    }
    for (auto& x : p) x /= Z;
    return p;
}

void gibbs_sweep(arma::vec& z, std::mt19937_64& rng) {
    for (std::size_t i = 0; i < N2; ++i) {
        double w[K2], tot = 0.0;
        for (std::size_t k = 0; k < K2; ++k) {
            w[k] = PI2[k] * std::exp(log_mvn2(i, k));
            tot += w[k];
        }
        double u = prng::u01(rng) * tot;
        std::size_t k = 0;
        while (k + 1 < K2 && u > w[k]) { u -= w[k]; ++k; }
        z[i] = static_cast<double>(k + 1);
    }
}

std::vector<double> run_fixed(bool with_sm, unsigned seed, long sweeps) {
    auto comp = std::make_unique<composite_block>();
    auto& D = comp->data();
    D.set("y", arma::vec(std::vector<double>(Y2, Y2 + N2 * D2)));
    D.set("pi", arma::vec(std::vector<double>(PI2, PI2 + K2)));
    D.set("mu", arma::vec(std::vector<double>(MU2, MU2 + K2 * D2)));
    D.set("sig", arma::vec(std::vector<double>(SIG2, SIG2 + K2 * D2 * D2)));
    arma::vec z(N2, arma::fill::ones);
    D.set("z", z);
    split_merge_block_config c;
    c.name = "sm"; c.N = N2; c.K_trunc = K2; c.d = D2; c.z_name = "z";
    c.y_key = "y"; c.pi_key = "pi"; c.mu_key = "mu"; c.sigma_key = "sig";
    c.initial_z = z;
    split_merge_block* sm = nullptr;
    {
        auto b = std::make_unique<split_merge_block>(std::move(c));
        sm = b.get();
        comp->add_child(std::move(b));
    }
    D.declare_dependencies("sm", {"y", "pi", "mu", "sig"});
    std::mt19937_64 rng(seed);
    std::vector<double> hist(n_states(N2, K2), 0.0);
    for (long s = 0; s < sweeps; ++s) {
        gibbs_sweep(z, rng);
        if (with_sm) {
            sm->set_current(z);
            for (int r = 0; r < 5; ++r) comp->step(rng);
            z = sm->current();
        }
        hist[encode(z, K2)] += 1.0;
    }
    for (auto& h : hist) h /= static_cast<double>(sweeps);
    return hist;
}

double tv(const std::vector<double>& a, const std::vector<double>& b) {
    double t = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) t += std::fabs(a[i] - b[i]);
    return 0.5 * t;
}

}  // namespace

int main() {
    std::printf("=== test_fix_dp-mixture_split_merge_shared_z ===\n");

    std::printf("(1) categorical z + split_merge + label swap vs exact (n=6, K=4)\n");
    {
        const Summary ex = summarise(exact_dp_pmf(), N1, K1);
        const long burn = 5000, sweeps = 400000;
        const Summary ctrl = summarise(run_dp(false, 20261122u, burn, sweeps), N1, K1);
        const Summary sm = summarise(run_dp(true, 20261122u, burn, sweeps), N1, K1);
        std::printf("  P(K_active=1..4): exact");
        for (std::size_t k = 1; k <= K1; ++k) std::printf(" %.4f", ex.pk[k]);
        std::printf(" | no split_merge");
        for (std::size_t k = 1; k <= K1; ++k) std::printf(" %.4f", ctrl.pk[k]);
        std::printf(" | split_merge");
        for (std::size_t k = 1; k <= K1; ++k) std::printf(" %.4f", sm.pk[k]);
        std::printf("\n");
        check(tv_partitions(ctrl, ex) < 0.02, "no split_merge: TV over partitions",
              tv_partitions(ctrl, ex), 0.02);
        check(tv_partitions(sm, ex) < 0.02, "split_merge: TV over partitions",
              tv_partitions(sm, ex), 0.02);
        check(max_abs_diff(sm.cocl, ex.cocl) < 0.01, "split_merge: max |P(z_i = z_j) - exact|",
              max_abs_diff(sm.cocl, ex.cocl), 0.01);
        check(max_abs_diff(sm.pk, ex.pk) < 0.01, "split_merge: max |P(K_active = k) - exact|",
              max_abs_diff(sm.pk, ex.pk), 0.01);
    }

    std::printf("(2) full covariance (sigma_key), fixed pi, mu, Sigma vs exact (n=5, K=3, d=2)\n");
    {
        const std::vector<double> ex = exact_fixed_pmf();
        const long sweeps = 400000;
        const double tv_ctrl = tv(run_fixed(false, 20261123u, sweeps), ex);
        const double tv_sm = tv(run_fixed(true, 20261123u, sweeps), ex);
        check(tv_ctrl < 0.01, "Gibbs only: TV to exact", tv_ctrl, 0.01);
        check(tv_sm < 3.0 * tv_ctrl + 0.002, "Gibbs + split_merge: TV to exact",
              tv_sm, 3.0 * tv_ctrl + 0.002);
    }

    std::printf("(3) a z in the context is checked before it is loaded\n");
    {
        split_merge_block_config c;
        c.name = "sm"; c.N = 4; c.K_trunc = 3; c.d = 1; c.lambda_key = "lambda";
        c.initial_z = arma::vec{1, 1, 2, 2};
        split_merge_block blk(c);
        const arma::vec before = blk.current();
        bool threw_len = false, threw_lab = false;
        try { blk.set_context({{"z", arma::vec{1, 2, 3}}}); }
        catch (const std::invalid_argument&) { threw_len = true; }
        try { blk.set_context({{"z", arma::vec{1, 2, 4, 1}}}); }
        catch (const std::invalid_argument&) { threw_lab = true; }
        const bool unchanged = arma::all(blk.current() == before);
        blk.set_context({{"z", arma::vec{3, 1, 3, 2}}});
        const bool loaded = arma::all(blk.current() == arma::vec{3, 1, 3, 2});
        check(threw_len && threw_lab && unchanged,
              "context z of wrong length or with a label outside 1..K_trunc throws, z kept");
        check(loaded, "a valid context z is the block's starting allocation");
    }

    std::printf("\n=== SUMMARY: %d failed ===\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
