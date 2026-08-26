/*
 * test_cluster_atom_block.cpp
 * Ground-truth ladder for cluster_atom_block (block_design VALIDATE, T0-T4).
 * ---------------------------------------------------------------------------
 * The block samples, per mixture component k and GIVEN the allocations z,
 *     p(theta_k | z, y) prop. pi(theta_k) * prod_{i: z_i = k+1} f(y_i | theta_k)
 * (Ishwaran & James 2001 blocked Gibbs step (a), Eq. 18). z is FIXED here --
 * it belongs to another block -- so these regimes test the ATOM step alone and
 * label switching cannot contaminate them.
 *
 * T1b (FD gradient) is N/A BY CONSTRUCTION: the block has no hand-written
 * gradient. Slice needs only the density, which is exactly why the block uses
 * it -- there is no derived artifact for Check #12 to police.
 *
 * The conditional under a CONJUGATE prior has a closed form even though the
 * block exists for the NON-conjugate case, so T0/T1a are genuine parity tests
 * against analytic moments rather than self-consistency checks.
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "AI4BayesCode/cluster_atom_block.hpp"
#include "portable_rng.hpp"

using AI4BayesCode::block_context;
using AI4BayesCode::cluster_atom_block;
using AI4BayesCode::cluster_atom_block_config;
using AI4BayesCode::joint_constraint;
using AI4BayesCode::joint_nuts_sub_param;

static int g_fail = 0;
static void check(bool ok, const std::string& what) {
    std::printf("  %s  %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

// ---------------------------------------------------------------------------
// Cross-chain rank-normalized R-hat (Vehtari 2021 rank-normalization +
// classical BETWEEN-chain R-hat). NOT split-R-hat: two over-dispersed,
// independent chains agreeing is the direct test of the reducibility failure
// mode, which a within-chain split cannot see.
// ---------------------------------------------------------------------------
static double cross_chain_rank_rhat(const std::vector<arma::vec>& chains) {
    const std::size_t M = chains.size(), N = chains[0].n_elem, T = M * N;
    std::vector<std::pair<double, std::size_t>> v;
    v.reserve(T);
    for (std::size_t m = 0; m < M; ++m)
        for (std::size_t i = 0; i < N; ++i)
            v.emplace_back(chains[m][i], m * N + i);
    std::sort(v.begin(), v.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    std::vector<double> z(T);
    for (std::size_t r = 0; r < T; ++r) {           // average ranks over ties
        std::size_t s = r;
        while (s + 1 < T && v[s + 1].first == v[r].first) ++s;
        const double rank = 0.5 * (double(r) + double(s)) + 1.0;
        const double p = (rank - 0.375) / (double(T) + 0.25);   // Blom
        const double zz = std::sqrt(2.0) *
            [](double x) {                                       // inverse erf
                double w = -std::log((1.0 - x) * (1.0 + x)), p2;
                if (w < 5.0) { w -= 2.5;
                    p2 = 2.81022636e-08; p2 = 3.43273939e-07 + p2*w;
                    p2 = -3.5233877e-06 + p2*w; p2 = -4.39150654e-06 + p2*w;
                    p2 = 0.00021858087 + p2*w;  p2 = -0.00125372503 + p2*w;
                    p2 = -0.00417768164 + p2*w; p2 = 0.246640727 + p2*w;
                    p2 = 1.50140941 + p2*w;
                } else { w = std::sqrt(w) - 3.0;
                    p2 = -0.000200214257; p2 = 0.000100950558 + p2*w;
                    p2 = 0.00134934322 + p2*w;  p2 = -0.00367342844 + p2*w;
                    p2 = 0.00573950773 + p2*w;  p2 = -0.0076224613 + p2*w;
                    p2 = 0.00943887047 + p2*w;  p2 = 1.00167406 + p2*w;
                    p2 = 2.83297682 + p2*w;
                }
                return p2 * x;
            }(2.0 * p - 1.0);
        for (std::size_t j = r; j <= s; ++j) z[v[j].second] = zz;
        r = s;
    }
    std::vector<double> mean(M, 0.0);
    double W = 0.0;
    for (std::size_t m = 0; m < M; ++m) {
        for (std::size_t i = 0; i < N; ++i) mean[m] += z[m * N + i];
        mean[m] /= double(N);
        double s2 = 0.0;
        for (std::size_t i = 0; i < N; ++i) {
            const double d = z[m * N + i] - mean[m];
            s2 += d * d;
        }
        W += s2 / double(N - 1);
    }
    W /= double(M);
    const double gm = std::accumulate(mean.begin(), mean.end(), 0.0) / double(M);
    double B = 0.0;
    for (std::size_t m = 0; m < M; ++m) B += (mean[m] - gm) * (mean[m] - gm);
    B *= double(N) / double(M - 1);
    const double var_plus = (double(N) - 1.0) / double(N) * W + B / double(N);
    return std::sqrt(var_plus / W);
}

// ---------------------------------------------------------------------------
// Shared fixture: a Gaussian mixture whose per-component conditional is the
// user-supplied density. Prior: mu ~ N(m0, 1/(kappa0*lambda)),
// lambda ~ Gamma(a0, rate b0)  -- i.e. Normal-Gamma, so the conditional has a
// closed form to compare against.  Atom = (mu REAL, lambda POSITIVE).
// ---------------------------------------------------------------------------
struct NGFixture {
    double m0 = 0.0, kappa0 = 0.05, a0 = 2.0, b0 = 1.0;
    static double lp(const arma::vec& th, std::size_t k,
                     const block_context& ctx, const NGFixture& f) {
        const double mu = th[0], lam = th[1];
        if (!(lam > 0.0) || !std::isfinite(mu)) return -INFINITY;
        const arma::vec& y = ctx.at("y");
        const arma::vec& z = ctx.at("z");
        double n = 0.0, s1 = 0.0, s2 = 0.0;
        for (std::size_t i = 0; i < z.n_elem; ++i) {
            if (std::size_t(std::llround(z[i])) != k + 1) continue;
            n += 1.0; s1 += y[i]; s2 += y[i] * y[i];
        }
        const double sse = s2 - 2.0 * mu * s1 + n * mu * mu;
        const double dm  = mu - f.m0;
        return (f.a0 - 1.0) * std::log(lam) - f.b0 * lam          // Gamma prior
             + 0.5 * std::log(lam) - 0.5 * f.kappa0 * lam * dm * dm  // mu | lam
             + 0.5 * n * std::log(lam) - 0.5 * lam * sse;            // likelihood
    }
    /// Exact Normal-Gamma posterior for component k (Murphy 2007 Sec.4).
    void posterior(double n, double s1, double s2, double& mu_n, double& kap_n,
                   double& a_n, double& b_n) const {
        const double ybar = (n > 0.0) ? s1 / n : 0.0;
        kap_n = kappa0 + n;
        mu_n  = (kappa0 * m0 + s1) / kap_n;
        a_n   = a0 + 0.5 * n;
        const double sse = (n > 0.0) ? (s2 - n * ybar * ybar) : 0.0;
        b_n   = b0 + 0.5 * sse
              + 0.5 * kappa0 * n * (ybar - m0) * (ybar - m0) / kap_n;
    }
};

static cluster_atom_block make_block(std::size_t K, const NGFixture& f,
                                     const arma::vec& init) {
    cluster_atom_block_config cfg;
    cfg.name    = "atoms";
    cfg.K_trunc = K;
    cfg.sub_params.push_back(joint_nuts_sub_param{"mu",     1u, joint_constraint::REAL});
    cfg.sub_params.push_back(joint_nuts_sub_param{"lambda", 1u, joint_constraint::POSITIVE});
    cfg.initial_cat = init;
    cfg.log_density = [f](const arma::vec& th, std::size_t k,
                          const block_context& ctx) {
        return NGFixture::lp(th, k, ctx, f);
    };
    return cluster_atom_block(std::move(cfg));
}

static block_context make_ctx(const arma::vec& y, const arma::vec& z,
                              std::size_t K) {
    block_context ctx;
    ctx["y"] = y; ctx["z"] = z;
    arma::vec cnt(K, arma::fill::zeros);
    for (std::size_t i = 0; i < z.n_elem; ++i)
        cnt[std::size_t(std::llround(z[i])) - 1] += 1.0;
    ctx["cluster_counts"] = cnt;
    return ctx;
}

int main() {
    std::printf("====== cluster_atom_block ground-truth ladder ======\n");

    // ---------------- T0 -- sanity: ONE component, analytic moments --------
    {
        std::printf("--- T0: single component vs analytic Normal-Gamma ---\n");
        NGFixture f;
        std::mt19937_64 rng(20260826u);
        const std::size_t N = 40;
        arma::vec y(N), z(N, arma::fill::ones);
        for (std::size_t i = 0; i < N; ++i) y[i] = prng::normal(rng, 1.5, 0.8);
        auto ctx = make_ctx(y, z, 1);
        auto blk = make_block(1, f, arma::vec{0.0, 1.0});
        blk.set_context(ctx);
        const std::size_t B = 2000, S = 40000;
        for (std::size_t t = 0; t < B; ++t) blk.step(rng);
        double sm = 0, sm2 = 0, sl = 0, sl2 = 0;
        for (std::size_t t = 0; t < S; ++t) {
            blk.step(rng);
            const double mu = blk.current()[0], lam = blk.current()[1];
            sm += mu; sm2 += mu * mu; sl += lam; sl2 += lam * lam;
        }
        const double em = sm / S, vm = sm2 / S - em * em;
        const double el = sl / S, vl = sl2 / S - el * el;
        double n = double(N), s1 = arma::accu(y), s2 = arma::accu(y % y);
        double mu_n, kap_n, a_n, b_n;
        f.posterior(n, s1, s2, mu_n, kap_n, a_n, b_n);
        const double E_lam = a_n / b_n, V_lam = a_n / (b_n * b_n);
        const double V_mu  = b_n / ((a_n - 1.0) * kap_n);
        std::printf("    mu:     E %.4f vs %.4f   Var %.5f vs %.5f\n",
                    em, mu_n, vm, V_mu);
        std::printf("    lambda: E %.4f vs %.4f   Var %.5f vs %.5f\n",
                    el, E_lam, vl, V_lam);
        check(std::fabs(em - mu_n) < 0.05 * std::sqrt(V_mu) + 0.01,
              "T0 E[mu] matches the analytic Normal-Gamma posterior");
        check(std::fabs(vm - V_mu) / V_mu < 0.10,
              "T0 Var[mu] matches within 10%");
        check(std::fabs(el - E_lam) / E_lam < 0.05,
              "T0 E[lambda] matches within 5%");
        check(std::fabs(vl - V_lam) / V_lam < 0.10,
              "T0 Var[lambda] matches within 10%");
    }

    // ---------------- T1a -- parity across SEVERAL components --------------
    {
        std::printf("--- T1a: 3 components, each vs its own analytic posterior ---\n");
        NGFixture f;
        std::mt19937_64 rng(777u);
        const std::size_t K = 3, N = 60;
        arma::vec y(N), z(N);
        for (std::size_t i = 0; i < N; ++i) {
            z[i] = double(i % K + 1);
            y[i] = prng::normal(rng, -2.0 + 2.0 * double(i % K), 0.7);
        }
        auto ctx = make_ctx(y, z, K);
        auto blk = make_block(K, f, arma::vec{0, 1, 0, 1, 0, 1});
        blk.set_context(ctx);
        for (std::size_t t = 0; t < 2000; ++t) blk.step(rng);
        const std::size_t S = 40000;
        arma::vec sm(K, arma::fill::zeros), sl(K, arma::fill::zeros);
        for (std::size_t t = 0; t < S; ++t) {
            blk.step(rng);
            for (std::size_t k = 0; k < K; ++k) {
                sm[k] += blk.current()[2 * k];
                sl[k] += blk.current()[2 * k + 1];
            }
        }
        bool ok_mu = true, ok_lam = true;
        for (std::size_t k = 0; k < K; ++k) {
            double n = 0, s1 = 0, s2 = 0;
            for (std::size_t i = 0; i < N; ++i)
                if (std::size_t(std::llround(z[i])) == k + 1) {
                    n += 1; s1 += y[i]; s2 += y[i] * y[i];
                }
            double mu_n, kap_n, a_n, b_n;
            f.posterior(n, s1, s2, mu_n, kap_n, a_n, b_n);
            const double em = sm[k] / S, el = sl[k] / S;
            const double V_mu = b_n / ((a_n - 1.0) * kap_n);
            std::printf("    k=%zu  mu %.4f vs %.4f   lambda %.4f vs %.4f\n",
                        k, em, mu_n, el, a_n / b_n);
            if (std::fabs(em - mu_n) > 0.06 * std::sqrt(V_mu) + 0.02) ok_mu = false;
            if (std::fabs(el - a_n / b_n) / (a_n / b_n) > 0.06)       ok_lam = false;
        }
        check(ok_mu,  "T1a every component's E[mu] matches its analytic posterior");
        check(ok_lam, "T1a every component's E[lambda] matches within 6%");
        std::printf("    T1b (FD gradient): N/A -- this block has no hand-written gradient\n");
    }

    // ---------------- T2 -- recovery from synthetic truth ------------------
    {
        std::printf("--- T2: recovery of KNOWN (mu, sigma) per component ---\n");
        NGFixture f;
        std::mt19937_64 rng(31337u);
        const std::size_t K = 2, N = 400;
        const double mu_true[2] = {-3.0, 2.5}, sd_true[2] = {0.6, 0.9};
        arma::vec y(N), z(N);
        for (std::size_t i = 0; i < N; ++i) {
            const std::size_t k = i % K;
            z[i] = double(k + 1);
            y[i] = prng::normal(rng, mu_true[k], sd_true[k]);
        }
        auto ctx = make_ctx(y, z, K);
        auto blk = make_block(K, f, arma::vec{0, 1, 0, 1});
        blk.set_context(ctx);
        for (std::size_t t = 0; t < 2000; ++t) blk.step(rng);
        const std::size_t S = 20000;
        arma::vec sm(K, arma::fill::zeros), ss(K, arma::fill::zeros);
        for (std::size_t t = 0; t < S; ++t) {
            blk.step(rng);
            for (std::size_t k = 0; k < K; ++k) {
                sm[k] += blk.current()[2 * k];
                ss[k] += 1.0 / std::sqrt(blk.current()[2 * k + 1]);
            }
        }
        bool ok = true;
        for (std::size_t k = 0; k < K; ++k) {
            const double em = sm[k] / S, es = ss[k] / S;
            std::printf("    k=%zu  mu %.3f (truth %.3f)   sd %.3f (truth %.3f)\n",
                        k, em, mu_true[k], es, sd_true[k]);
            if (std::fabs(em - mu_true[k]) > 0.15)  ok = false;
            if (std::fabs(es - sd_true[k]) > 0.12)  ok = false;
        }
        check(ok, "T2 posterior means recover the simulating truth");
    }

    // ---------------- T3 -- cross-chain rank R-hat < 1.01 ------------------
    {
        std::printf("--- T3: two OVER-DISPERSED chains, cross-chain rank R-hat ---\n");
        NGFixture f;
        const std::size_t K = 4, N = 80, S = 8000;
        std::mt19937_64 gen(4242u);
        arma::vec y(N), z(N);
        for (std::size_t i = 0; i < N; ++i) {
            z[i] = double(i % 2 + 1);              // components 3,4 stay EMPTY
            y[i] = prng::normal(gen, 0.0, 1.0);
        }
        auto ctx = make_ctx(y, z, K);
        std::vector<std::vector<arma::vec>> draws(2 * K);
        for (int chain = 0; chain < 2; ++chain) {
            const double off = (chain == 0) ? -5.0 : 5.0;
            arma::vec init(2 * K);
            for (std::size_t k = 0; k < K; ++k) {
                init[2 * k]     = off;
                init[2 * k + 1] = (chain == 0) ? 0.05 : 20.0;
            }
            auto blk = make_block(K, f, init);
            blk.set_context(ctx);
            std::mt19937_64 rng(chain == 0 ? 11u : 99u);
            for (std::size_t t = 0; t < 2000; ++t) blk.step(rng);
            for (std::size_t k = 0; k < 2 * K; ++k)
                draws[k].push_back(arma::vec(S));
            for (std::size_t t = 0; t < S; ++t) {
                blk.step(rng);
                for (std::size_t k = 0; k < 2 * K; ++k)
                    draws[k][chain][t] = blk.current()[k];
            }
        }
        double worst = 0.0;
        for (std::size_t k = 0; k < 2 * K; ++k)
            worst = std::max(worst, cross_chain_rank_rhat(draws[k]));
        std::printf("    worst cross-chain rank R-hat over %zu coords = %.5f\n",
                    2 * K, worst);
        check(worst < 1.01, "T3 cross-chain rank R-hat < 1.01 (library bar)");
    }

    // ---------------- T4 -- stress: the EMPTY component must be the PRIOR ---
    // The block's signature property, and the one the 20-dim joint-NUTS
    // baseline loses: with n_k = 0 the likelihood terms vanish, so component
    // k's conditional IS the Normal-Gamma prior. Made decisive -- K = 20 with
    // only 2 occupied, data far from the prior mean, so 18 components are
    // driven purely by the prior and are compared to its ANALYTIC moments.
    {
        std::printf("--- T4: K=20 with 18 EMPTY components vs the analytic prior ---\n");
        NGFixture f;
        std::mt19937_64 rng(2718u);
        const std::size_t K = 20, N = 60, S = 20000;
        arma::vec y(N), z(N);
        for (std::size_t i = 0; i < N; ++i) {
            z[i] = double(i % 2 + 1);
            y[i] = prng::normal(rng, 4.0, 0.5);
        }
        auto ctx = make_ctx(y, z, K);
        arma::vec init(2 * K);
        for (std::size_t k = 0; k < K; ++k) { init[2*k] = 0.0; init[2*k+1] = 1.0; }
        auto blk = make_block(K, f, init);
        blk.set_context(ctx);
        for (std::size_t t = 0; t < 3000; ++t) blk.step(rng);
        double sm = 0, sm2 = 0, sl = 0, n_obs = 0;
        for (std::size_t t = 0; t < S; ++t) {
            blk.step(rng);
            for (std::size_t k = 2; k < K; ++k) {
                const double mu = blk.current()[2 * k];
                sm += mu; sm2 += mu * mu;
                sl += blk.current()[2 * k + 1];
                n_obs += 1.0;
            }
        }
        const double em = sm / n_obs, vm = sm2 / n_obs - em * em, el = sl / n_obs;
        const double E_lam_pr = f.a0 / f.b0;
        const double V_mu_pr  = f.b0 / (f.kappa0 * (f.a0 - 1.0));
        std::printf("    empty mu:     E %.4f vs %.4f   Var %.3f vs %.3f\n",
                    em, f.m0, vm, V_mu_pr);
        std::printf("    empty lambda: E %.4f vs %.4f\n", el, E_lam_pr);
        check(std::fabs(em - f.m0) < 0.6,
              "T4 empty components sit at the PRIOR mean, not pulled to the data");
        check(std::fabs(vm - V_mu_pr) / V_mu_pr < 0.25,
              "T4 empty components' Var[mu] matches the analytic prior within 25%");
        check(std::fabs(el - E_lam_pr) / E_lam_pr < 0.10,
              "T4 empty components' E[lambda] matches the prior within 10%");
    }

    // ---------------- guards: constraint coverage + v1 scope gate ----------
    {
        std::printf("--- guards: constraint kinds and the v1 scope gate ---\n");
        bool threw = false;
        try {
            cluster_atom_block_config bad;
            bad.name = "bad"; bad.K_trunc = 2;
            bad.sub_params.push_back(joint_nuts_sub_param{"p", 3u, joint_constraint::SIMPLEX});
            bad.initial_cat = arma::vec(6, arma::fill::value(0.33));
            bad.log_density = [](const arma::vec&, std::size_t,
                                 const block_context&) { return 0.0; };
            cluster_atom_block probe(std::move(bad));
            (void) probe;
        } catch (const std::exception&) { threw = true; }
        check(threw, "guard: a COUPLED constraint kind throws from the constructor");

        std::mt19937_64 rng(555u);
        cluster_atom_block_config cfg;
        cfg.name = "iv"; cfg.K_trunc = 2;
        joint_nuts_sub_param sp{"rho", 1u, joint_constraint::INTERVAL};
        sp.lower = -1.0; sp.upper = 1.0;
        cfg.sub_params.push_back(sp);
        cfg.initial_cat = arma::vec{0.0, 0.0};
        cfg.log_density = [](const arma::vec& th, std::size_t,
                             const block_context&) {
            return -0.5 * th[0] * th[0] / 0.25;
        };
        cluster_atom_block blk(std::move(cfg));
        auto ctx = make_ctx(arma::vec{0.0}, arma::vec{1.0}, 2);
        blk.set_context(ctx);
        bool in_support = true;
        for (std::size_t t = 0; t < 4000; ++t) {
            blk.step(rng);
            for (std::size_t k = 0; k < 2; ++k) {
                const double v = blk.current()[k];
                if (!(v > -1.0 && v < 1.0) || !std::isfinite(v)) in_support = false;
            }
        }
        check(in_support, "guard: INTERVAL draws stay strictly inside (lower, upper)");

        const auto out = blk.current_named_outputs();
        check(out.size() == 1 && out.count("rho") == 1 &&
              out.at("rho").n_elem == 2,
              "guard: current_named_outputs writes one K_trunc*dim key per slice");
    }

    std::printf("\n====== %d FAIL ======\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
