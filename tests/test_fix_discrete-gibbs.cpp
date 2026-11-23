/*================================================================================
 *  AI4BayesCode  --  regression test for categorical_gibbs_block and
 *                    binary_gibbs_block: oracle evaluations per step, and
 *                    set_current when the context holds the block's own key.
 *================================================================================
 *
 *  Part 1. When the context does not hold the block's own key, step()
 *          evaluates the oracle once. With an oracle that does not depend
 *          on the own key, a block driven without the key (one evaluation per
 *          step) and a block driven with it (one evaluation per site) must
 *          give identical draws.
 *  Part 2. With the own key in the context, the context passed to the
 *          first oracle call of the step after set_current must hold the
 *          loaded state, and the one passed for site i must hold the labels
 *          drawn at sites < i and the loaded labels at sites > i. Exact
 *          check.
 *  Part 3. A direct driver that alternates step() with an external
 *          pi-invariant move, loaded by set_current, must leave pi
 *          invariant. The empirical law of the block output is compared
 *          with pi enumerated over all states (a 3-site Potts chain with 3
 *          labels, and a 4-spin pairwise binary model). Gate: every state
 *          within 5 sigma, sigma = max(batch-means MCSE, iid binomial SE).
 *================================================================================*/

#include "AI4BayesCode/block_sampler.hpp"
#include "AI4BayesCode/binary_gibbs_block.hpp"
#include "AI4BayesCode/categorical_gibbs_block.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

using AI4BayesCode::block_context;
using AI4BayesCode::binary_gibbs_block;
using AI4BayesCode::binary_gibbs_block_config;
using AI4BayesCode::categorical_gibbs_block;
using AI4BayesCode::categorical_gibbs_block_config;

namespace {

using input_log = std::shared_ptr<std::vector<arma::vec> >;

bool same(const arma::vec& a, const arma::vec& b) {
    if (a.n_elem != b.n_elem) return false;
    for (std::size_t i = 0; i < a.n_elem; ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

// Largest |freq - p| / sigma over states, sigma = max(batch-means MCSE,
// iid binomial SE), and the total variation distance.
struct law_check {
    double worst_sigma;
    double tv;
};

law_check compare_law(const std::vector<std::vector<double> >& counts,
                      std::size_t batch_len, const std::vector<double>& p) {
    const std::size_t nb = counts.size();
    const double n = static_cast<double>(nb * batch_len);
    law_check out{0.0, 0.0};
    for (std::size_t s = 0; s < p.size(); ++s) {
        double tot = 0.0, mean = 0.0;
        for (std::size_t b = 0; b < nb; ++b) {
            tot  += counts[b][s];
            mean += counts[b][s] / static_cast<double>(batch_len);
        }
        mean /= static_cast<double>(nb);
        double var = 0.0;
        for (std::size_t b = 0; b < nb; ++b) {
            const double d = counts[b][s] / static_cast<double>(batch_len) - mean;
            var += d * d;
        }
        var /= static_cast<double>(nb - 1);
        const double mcse   = std::sqrt(var / static_cast<double>(nb));
        const double se_iid = std::sqrt(p[s] * (1.0 - p[s]) / n);
        const double sigma  = std::max(mcse, se_iid);
        const double err    = std::abs(tot / n - p[s]);
        out.tv += 0.5 * err;
        out.worst_sigma = std::max(out.worst_sigma, err / sigma);
    }
    return out;
}

// Draws index j with probability w[j] / sum(w).
std::size_t draw_index(const std::vector<double>& w, std::mt19937_64& rng) {
    double tot = 0.0;
    for (double x : w) tot += x;
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    const double u = u01(rng) * tot;
    double cum = 0.0;
    for (std::size_t j = 0; j < w.size(); ++j) {
        cum += w[j];
        if (u < cum) return j;
    }
    return w.size() - 1;
}

// ---------------------------------------------------------------------------
// Potts chain: 3 sites, labels {1, 2, 3},
//   pi(z) ~ exp( sum_i h(i, z_i) + J (1{z_1 = z_2} + 1{z_2 = z_3}) ).
// ---------------------------------------------------------------------------

const std::size_t kSites  = 3;
const std::size_t kLabels = 3;
const double      kJ      = 0.9;
const double      kField[3][3] = {{ 0.0,  0.4, -0.3},
                                  {-0.2,  0.0,  0.6},
                                  { 0.5, -0.1,  0.0}};

std::size_t potts_index(const arma::vec& z) {
    return (static_cast<std::size_t>(z[0] + 0.5) - 1) * 9
         + (static_cast<std::size_t>(z[1] + 0.5) - 1) * 3
         + (static_cast<std::size_t>(z[2] + 0.5) - 1);
}

std::vector<double> potts_pi() {
    std::vector<double> p(27);
    double tot = 0.0;
    for (std::size_t s = 0; s < 27; ++s) {
        const std::size_t a = s / 9, b = (s / 3) % 3, c = s % 3;
        double lw = kField[0][a] + kField[1][b] + kField[2][c];
        if (a == b) lw += kJ;
        if (b == c) lw += kJ;
        p[s] = std::exp(lw);
        tot += p[s];
    }
    for (double& x : p) x /= tot;
    return p;
}

// Full conditionals of the Potts chain, computed from the block's own key
// "zP"; the own-key vector of every oracle call is appended to `inputs`.
categorical_gibbs_block_config potts_config(input_log inputs) {
    categorical_gibbs_block_config cfg;
    cfg.name           = "zP";
    cfg.n_obs          = kSites;
    cfg.n_categories   = kLabels;
    cfg.initial_labels = arma::vec(kSites, arma::fill::ones);
    cfg.log_probs_fn   = [inputs](const block_context& ctx) {
        const arma::vec& z = ctx.at("zP");
        if (inputs) inputs->push_back(z);
        arma::mat out(kSites, kLabels);
        for (std::size_t i = 0; i < kSites; ++i) {
            for (std::size_t k = 0; k < kLabels; ++k) {
                const double lab = static_cast<double>(k + 1);
                double v = kField[i][k];
                if (i > 0 && std::abs(z[i - 1] - lab) < 0.5) v += kJ;
                if (i + 1 < kSites && std::abs(z[i + 1] - lab) < 0.5) v += kJ;
                out(i, k) = v;
            }
        }
        return out;
    };
    return cfg;
}

// pi-invariant move: exact draw of (z_2, z_3) given z_1.
arma::vec potts_move(const arma::vec& z, const std::vector<double>& pi,
                     std::mt19937_64& rng) {
    const std::size_t a = static_cast<std::size_t>(z[0] + 0.5) - 1;
    std::vector<double> w(9);
    for (std::size_t j = 0; j < 9; ++j) w[j] = pi[a * 9 + j];
    const std::size_t pick = draw_index(w, rng);
    arma::vec out = z;
    out[1] = static_cast<double>(pick / 3 + 1);
    out[2] = static_cast<double>(pick % 3 + 1);
    return out;
}

// ---------------------------------------------------------------------------
// Pairwise binary model: 4 spins,
//   p(z) ~ exp( sum_i a_i z_i + sum_{i<j} W_ij z_i z_j ).
// ---------------------------------------------------------------------------

const std::size_t kSpins = 4;
const double      kA[4]    = {-1.2, 0.4, -0.6, 0.9};
const double      kW[4][4] = {{ 0.0,  1.6,  0.5, -0.8},
                              { 1.6,  0.0, -1.3,  0.0},
                              { 0.5, -1.3,  0.0,  1.1},
                              {-0.8,  0.0,  1.1,  0.0}};

std::size_t spin_index(const arma::vec& z) {
    std::size_t k = 0;
    for (std::size_t i = 0; i < kSpins; ++i) {
        if (z[i] >= 0.5) k |= (std::size_t(1) << i);
    }
    return k;
}

std::vector<double> spin_pi() {
    std::vector<double> p(16);
    double tot = 0.0;
    for (std::size_t k = 0; k < 16; ++k) {
        double e = 0.0;
        for (std::size_t i = 0; i < kSpins; ++i) {
            const double zi = static_cast<double>((k >> i) & 1u);
            e += kA[i] * zi;
            for (std::size_t j = i + 1; j < kSpins; ++j) {
                e += kW[i][j] * zi * static_cast<double>((k >> j) & 1u);
            }
        }
        p[k] = std::exp(e);
        tot += p[k];
    }
    for (double& x : p) x /= tot;
    return p;
}

// Full conditionals lo_i = a_i + sum_j W_ij z_j, computed from the own key
// "zS"; the own-key vector of every oracle call is appended to `inputs`.
binary_gibbs_block_config spin_config(input_log inputs) {
    binary_gibbs_block_config cfg;
    cfg.name           = "zS";
    cfg.n_binary       = kSpins;
    cfg.initial_values = arma::vec(kSpins, arma::fill::zeros);
    cfg.log_odds_fn    = [inputs](const block_context& ctx) {
        const arma::vec& z = ctx.at("zS");
        if (inputs) inputs->push_back(z);
        arma::vec lo(kSpins);
        for (std::size_t i = 0; i < kSpins; ++i) {
            double s = kA[i];
            for (std::size_t j = 0; j < kSpins; ++j) s += kW[i][j] * z[j];
            lo[i] = s;
        }
        return lo;
    };
    return cfg;
}

// pi-invariant move: exact draw of (z_2, z_3, z_4) given z_1.
arma::vec spin_move(const arma::vec& z, const std::vector<double>& p,
                    std::mt19937_64& rng) {
    const std::size_t z0 = (z[0] >= 0.5) ? 1 : 0;
    std::vector<double> w(8);
    for (std::size_t j = 0; j < 8; ++j) w[j] = p[z0 | (j << 1)];
    const std::size_t pick = draw_index(w, rng);
    arma::vec out = z;
    for (std::size_t i = 1; i < kSpins; ++i) {
        out[i] = static_cast<double>((pick >> (i - 1)) & 1u);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Part 1: one oracle evaluation per step without the own key
// ---------------------------------------------------------------------------

// Sizes of the part-1 blocks.
const std::size_t kN = 40;
const std::size_t kK = 5;

bool test_categorical_one_evaluation() {
    std::printf("[categorical] part 1: evaluations per step, path equivalence\n");
    const std::size_t N = kN, K = kK;
    auto calls = std::make_shared<long>(0);
    categorical_gibbs_block_config cfg;
    cfg.name           = "z";
    cfg.n_obs          = N;
    cfg.n_categories   = K;
    cfg.initial_labels = arma::vec(N, arma::fill::ones);
    cfg.log_probs_fn   = [calls](const block_context& ctx) {
        ++*calls;
        const double th = ctx.at("theta")[0];
        arma::mat lp(kN, kK);
        for (std::size_t i = 0; i < kN; ++i) {
            for (std::size_t k = 0; k < kK; ++k) {
                lp(i, k) = 2.0 * std::sin(0.37 * static_cast<double>((i + 1) * (k + 1)) + th);
            }
        }
        return lp;
    };
    categorical_gibbs_block a(cfg);   // driven without "z" in its context
    categorical_gibbs_block b(cfg);   // driven with "z" in its context

    std::mt19937_64 rng_a(11u), rng_b(11u), rng_load(5u);
    std::uniform_int_distribution<int> lab(1, static_cast<int>(K));
    bool counts_ok = true, draws_ok = true;
    for (int s = 0; s < 300; ++s) {
        block_context ctx;
        ctx["theta"] = arma::vec{0.05 * s};
        a.set_context(ctx);
        ctx["z"] = b.current();
        b.set_context(ctx);
        // Every 10th sweep steps twice without a new set_context.
        const int reps = (s % 10 == 9) ? 2 : 1;
        for (int r = 0; r < reps; ++r) {
            long c0 = *calls;
            a.step(rng_a);
            const long ca = *calls - c0;
            c0 = *calls;
            b.step(rng_b);
            const long cb = *calls - c0;
            if (ca != 1 || cb != static_cast<long>(N)) counts_ok = false;
            if (!same(a.current(), b.current())) draws_ok = false;
        }
        if (s % 50 == 49) {
            arma::vec load(N);
            for (std::size_t i = 0; i < N; ++i) load[i] = lab(rng_load);
            a.set_current(load);
            b.set_current(load);
        }
    }
    const bool rng_ok = (rng_a() == rng_b());
    std::printf("  1 call per step without the key, N with it?   %s\n",
                counts_ok ? "YES" : "NO");
    std::printf("  identical draws on both paths?                %s\n",
                draws_ok ? "YES" : "NO");
    std::printf("  same number of uniforms consumed?             %s\n",
                rng_ok ? "YES" : "NO");
    return counts_ok && draws_ok && rng_ok;
}

bool test_binary_one_evaluation() {
    std::printf("[binary] part 1: evaluations per step, path equivalence\n");
    const std::size_t N = kN;
    auto calls = std::make_shared<long>(0);
    binary_gibbs_block_config cfg;
    cfg.name           = "g";
    cfg.n_binary       = N;
    cfg.initial_values = arma::vec(N, arma::fill::zeros);
    cfg.log_odds_fn    = [calls](const block_context& ctx) {
        ++*calls;
        const double th = ctx.at("theta")[0];
        arma::vec lo(kN);
        for (std::size_t i = 0; i < kN; ++i) {
            lo[i] = 3.0 * std::sin(0.53 * static_cast<double>(i + 1) + th);
        }
        return lo;
    };
    binary_gibbs_block a(cfg);   // driven without "g" in its context
    binary_gibbs_block b(cfg);   // driven with "g" in its context

    std::mt19937_64 rng_a(13u), rng_b(13u), rng_load(7u);
    std::uniform_int_distribution<int> bit(0, 1);
    bool counts_ok = true, draws_ok = true;
    for (int s = 0; s < 300; ++s) {
        block_context ctx;
        ctx["theta"] = arma::vec{0.05 * s};
        a.set_context(ctx);
        ctx["g"] = b.current();
        b.set_context(ctx);
        const int reps = (s % 10 == 9) ? 2 : 1;
        for (int r = 0; r < reps; ++r) {
            long c0 = *calls;
            a.step(rng_a);
            const long ca = *calls - c0;
            c0 = *calls;
            b.step(rng_b);
            const long cb = *calls - c0;
            if (ca != 1 || cb != static_cast<long>(N)) counts_ok = false;
            if (!same(a.current(), b.current())) draws_ok = false;
        }
        if (s % 50 == 49) {
            arma::vec load(N);
            for (std::size_t i = 0; i < N; ++i) load[i] = bit(rng_load);
            a.set_current(load);
            b.set_current(load);
        }
    }
    const bool rng_ok = (rng_a() == rng_b());
    std::printf("  1 call per step without the key, N with it?   %s\n",
                counts_ok ? "YES" : "NO");
    std::printf("  identical draws on both paths?                %s\n",
                draws_ok ? "YES" : "NO");
    std::printf("  same number of uniforms consumed?             %s\n",
                rng_ok ? "YES" : "NO");
    return counts_ok && draws_ok && rng_ok;
}

// ---------------------------------------------------------------------------
// Part 2: the step after set_current conditions on the loaded state
// ---------------------------------------------------------------------------

bool test_categorical_set_current_exact() {
    std::printf("[categorical] part 2: oracle input after set_current\n");
    input_log inputs = std::make_shared<std::vector<arma::vec> >();
    categorical_gibbs_block blk(potts_config(inputs));
    block_context ctx;
    ctx["zP"] = arma::vec(kSites, arma::fill::ones);
    blk.set_context(ctx);

    std::mt19937_64 rng(778899123u);
    std::size_t n_bad = 0;
    const std::size_t n_load = 200;
    for (std::size_t rep = 0; rep < n_load; ++rep) {
        for (int t = 0; t < 5; ++t) blk.step(rng);
        arma::vec load = blk.current();
        for (std::size_t i = 0; i < kSites; ++i) {
            load[i] = std::fmod(load[i], 3.0) + 1.0;   // differs at every site
        }
        blk.set_current(load);
        inputs->clear();
        blk.step(rng);
        const arma::vec& z = blk.current();
        bool ok = (inputs->size() == kSites);
        arma::vec expect = load;
        for (std::size_t i = 0; ok && i < kSites; ++i) {
            if (!same((*inputs)[i], expect)) ok = false;
            expect[i] = z[i];
        }
        if (!ok) ++n_bad;
    }
    std::printf("  loads whose next step passed the loaded labels: %zu / %zu\n",
                n_load - n_bad, n_load);
    return n_bad == 0;
}

bool test_binary_set_current_exact() {
    std::printf("[binary] part 2: oracle input after set_current\n");
    input_log inputs = std::make_shared<std::vector<arma::vec> >();
    binary_gibbs_block blk(spin_config(inputs));
    block_context ctx;
    ctx["zS"] = arma::vec(kSpins, arma::fill::zeros);
    blk.set_context(ctx);

    std::mt19937_64 rng(20260818u);
    std::size_t n_bad = 0;
    const std::size_t n_load = 200;
    for (std::size_t rep = 0; rep < n_load; ++rep) {
        for (int t = 0; t < 5; ++t) blk.step(rng);
        arma::vec load = 1.0 - blk.current();   // differs at every spin
        blk.set_current(load);
        inputs->clear();
        blk.step(rng);
        const arma::vec& z = blk.current();
        bool ok = (inputs->size() == kSpins);
        arma::vec expect = load;
        for (std::size_t i = 0; ok && i < kSpins; ++i) {
            if (!same((*inputs)[i], expect)) ok = false;
            expect[i] = z[i];
        }
        if (!ok) ++n_bad;
    }
    std::printf("  loads whose next step passed the loaded spins:  %zu / %zu\n",
                n_load - n_bad, n_load);
    return n_bad == 0;
}

// ---------------------------------------------------------------------------
// Part 3: step() + external pi-invariant move loaded by set_current
// ---------------------------------------------------------------------------

bool test_categorical_set_current_stationary() {
    std::printf("[categorical] part 3: law with a move loaded every sweep\n");
    const std::vector<double> pi = potts_pi();
    categorical_gibbs_block blk(potts_config(nullptr));
    block_context ctx;
    ctx["zP"] = arma::vec(kSites, arma::fill::ones);
    blk.set_context(ctx);

    std::mt19937_64 rng(5511u), rng_move(7733u);
    for (int t = 0; t < 2000; ++t) {
        blk.step(rng);
        blk.set_current(potts_move(blk.current(), pi, rng_move));
    }
    const std::size_t NB = 40, BL = 25000;
    std::vector<std::vector<double> > counts(NB, std::vector<double>(27, 0.0));
    for (std::size_t b = 0; b < NB; ++b) {
        for (std::size_t t = 0; t < BL; ++t) {
            blk.step(rng);
            counts[b][potts_index(blk.current())] += 1.0;
            blk.set_current(potts_move(blk.current(), pi, rng_move));
        }
    }
    const law_check r = compare_law(counts, BL, pi);
    const bool ok = r.worst_sigma < 5.0;
    std::printf("  TV = %.4f, worst state %.2f sigma (gate 5)    %s\n",
                r.tv, r.worst_sigma, ok ? "YES" : "NO");
    return ok;
}

bool test_binary_set_current_stationary() {
    std::printf("[binary] part 3: law with a move loaded every sweep\n");
    const std::vector<double> p = spin_pi();
    binary_gibbs_block blk(spin_config(nullptr));
    block_context ctx;
    ctx["zS"] = arma::vec(kSpins, arma::fill::zeros);
    blk.set_context(ctx);

    std::mt19937_64 rng(4242u), rng_move(9191u);
    for (int t = 0; t < 2000; ++t) {
        blk.step(rng);
        blk.set_current(spin_move(blk.current(), p, rng_move));
    }
    const std::size_t NB = 40, BL = 25000;
    std::vector<std::vector<double> > counts(NB, std::vector<double>(16, 0.0));
    for (std::size_t b = 0; b < NB; ++b) {
        for (std::size_t t = 0; t < BL; ++t) {
            blk.step(rng);
            counts[b][spin_index(blk.current())] += 1.0;
            blk.set_current(spin_move(blk.current(), p, rng_move));
        }
    }
    const law_check r = compare_law(counts, BL, p);
    const bool ok = r.worst_sigma < 5.0;
    std::printf("  TV = %.4f, worst state %.2f sigma (gate 5)    %s\n",
                r.tv, r.worst_sigma, ok ? "YES" : "NO");
    return ok;
}

} // namespace

int main() {
    const bool ok1 = test_categorical_one_evaluation();
    const bool ok2 = test_binary_one_evaluation();
    const bool ok3 = test_categorical_set_current_exact();
    const bool ok4 = test_binary_set_current_exact();
    const bool ok5 = test_categorical_set_current_stationary();
    const bool ok6 = test_binary_set_current_stationary();
    const bool all_ok = ok1 && ok2 && ok3 && ok4 && ok5 && ok6;
    std::printf("\n%s\n", all_ok ? "PASS" : "FAIL");
    return all_ok ? 0 : 1;
}
