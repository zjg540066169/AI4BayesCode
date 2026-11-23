/*================================================================================
 *  AI4BayesCode  --  regression test for categorical_gibbs_block and
 *                    binary_gibbs_block when the block's own key is in its
 *                    context and a second block writes the same key.
 *================================================================================
 *
 *  Setup: composite_block with two children, the Gibbs block (its own key in
 *  its declare_dependencies, an oracle that uses the own key) and a "mover"
 *  that also writes that key with an exact pi-invariant draw. step() must
 *  start from the shared value of the own key, i.e. the mover's output.
 *
 *  Part 1. Exact: in every sweep, the own-key vector passed to the oracle for
 *          site i holds the labels drawn at sites < i and the mover's labels
 *          at sites >= i.
 *  Part 2. Law: the Gibbs block's output and the end-of-sweep shared value
 *          are compared with pi enumerated over all states (a 3-site Potts
 *          chain with 3 labels; a 4-spin pairwise binary model). Gate: every
 *          state within 5 sigma, sigma = max(batch-means MCSE, iid binomial
 *          SE).
 *  Part 3. An own-key entry of the wrong length makes step() throw a
 *          std::runtime_error that names the key, with the block's values,
 *          the next oracle input and the RNG unchanged; labels in the entry
 *          are snapped as set_current snaps them.
 *================================================================================*/

#include "AI4BayesCode/block_sampler.hpp"
#include "AI4BayesCode/binary_gibbs_block.hpp"
#include "AI4BayesCode/categorical_gibbs_block.hpp"
#include "AI4BayesCode/composite_block.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using AI4BayesCode::block_context;
using AI4BayesCode::block_sampler;
using AI4BayesCode::binary_gibbs_block;
using AI4BayesCode::binary_gibbs_block_config;
using AI4BayesCode::categorical_gibbs_block;
using AI4BayesCode::categorical_gibbs_block_config;
using AI4BayesCode::composite_block;
using AI4BayesCode::state_map;

namespace {

using input_log = std::shared_ptr<std::vector<arma::vec> >;

bool same(const arma::vec& a, const arma::vec& b) {
    if (a.n_elem != b.n_elem) return false;
    for (std::size_t i = 0; i < a.n_elem; ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

struct law_check {
    double worst_sigma;
    double tv;
};

// Largest |freq - p| / sigma over states, sigma = max(batch-means MCSE,
// iid binomial SE), and the total variation distance.
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

// A second writer of `key`: an exact pi-invariant draw given by `move`.
class mover_block : public block_sampler {
public:
    using move_fn = arma::vec (*)(const arma::vec&, std::mt19937_64&);
    mover_block(std::string key, arma::vec init, move_fn move)
        : key_(std::move(key)), z_(std::move(init)), move_(move) {}
    void set_context(const block_context& ctx) override { z_ = ctx.at(key_); }
    void step(std::mt19937_64& rng) override { z_ = move_(z_, rng); }
    const arma::vec& current() const override { return z_; }
    void set_current(const arma::vec& t) override { z_ = t; }
    const std::string& name() const noexcept override { return name_; }
    std::size_t dim() const noexcept override { return z_.n_elem; }
    state_map current_named_outputs() const override { return {{key_, z_}}; }
private:
    std::string key_;
    arma::vec   z_;
    move_fn     move_;
    std::string name_ = "mover";
};

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

// Exact draw of (z_2, z_3) given z_1.
arma::vec potts_move(const arma::vec& z, std::mt19937_64& rng) {
    static const std::vector<double> pi = potts_pi();
    const std::size_t a = static_cast<std::size_t>(z[0] + 0.5) - 1;
    std::vector<double> w(9);
    for (std::size_t j = 0; j < 9; ++j) w[j] = pi[a * 9 + j];
    const std::size_t pick = draw_index(w, rng);
    arma::vec out = z;
    out[1] = static_cast<double>(pick / 3 + 1);
    out[2] = static_cast<double>(pick % 3 + 1);
    return out;
}

// Full conditionals of the Potts chain from the own key "zP"; the own-key
// vector of every oracle call is appended to `inputs`.
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

std::unique_ptr<composite_block> potts_composite(input_log inputs) {
    auto comp = std::make_unique<composite_block>("potts_comover");
    comp->add_child(std::make_unique<categorical_gibbs_block>(potts_config(inputs)));
    comp->add_child(std::make_unique<mover_block>(
        "zP", arma::vec(kSites, arma::fill::ones), &potts_move));
    comp->data().declare_dependencies("zP", {"zP"});
    comp->data().declare_dependencies("mover", {"zP"});
    return comp;
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

// Exact draw of (z_2, z_3, z_4) given z_1.
arma::vec spin_move(const arma::vec& z, std::mt19937_64& rng) {
    static const std::vector<double> p = spin_pi();
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

// Full conditionals lo_i = a_i + sum_j W_ij z_j from the own key "zS"; the
// own-key vector of every oracle call is appended to `inputs`.
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

std::unique_ptr<composite_block> spin_composite(input_log inputs) {
    auto comp = std::make_unique<composite_block>("spin_comover");
    comp->add_child(std::make_unique<binary_gibbs_block>(spin_config(inputs)));
    comp->add_child(std::make_unique<mover_block>(
        "zS", arma::vec(kSpins, arma::fill::zeros), &spin_move));
    comp->data().declare_dependencies("zS", {"zS"});
    comp->data().declare_dependencies("mover", {"zS"});
    return comp;
}

// ---------------------------------------------------------------------------
// Part 1: oracle inputs of every sweep
// ---------------------------------------------------------------------------

bool test_oracle_inputs(const char* tag, std::unique_ptr<composite_block> comp,
                        input_log inputs, const std::string& key,
                        std::size_t n_sites, unsigned seed) {
    std::printf("[%s] part 1: oracle inputs start from the mover's output\n", tag);
    std::mt19937_64 rng(seed);
    for (int t = 0; t < 50; ++t) comp->step(rng);
    const std::size_t n_sweeps = 400;
    std::size_t n_ok = 0, n_moved = 0;
    for (std::size_t s = 0; s < n_sweeps; ++s) {
        const arma::vec shared_before = comp->data().get(key);
        const arma::vec own_before    = comp->child(0).current();
        if (!same(shared_before, own_before)) ++n_moved;
        inputs->clear();
        comp->step(rng);
        const arma::vec& drawn = comp->child(0).current();
        bool ok = (inputs->size() == n_sites);
        arma::vec expect = shared_before;
        for (std::size_t i = 0; ok && i < n_sites; ++i) {
            if (!same((*inputs)[i], expect)) ok = false;
            expect[i] = drawn[i];
        }
        if (ok) ++n_ok;
    }
    std::printf("  sweeps where the mover changed the shared value: %zu / %zu\n",
                n_moved, n_sweeps);
    std::printf("  sweeps whose oracle inputs started from it:      %zu / %zu\n",
                n_ok, n_sweeps);
    return n_ok == n_sweeps && n_moved > 0;
}

// ---------------------------------------------------------------------------
// Part 2: law of the Gibbs block's output and of the shared value
// ---------------------------------------------------------------------------

template <class IndexFn>
bool test_law(const char* tag, std::unique_ptr<composite_block> comp,
              const std::string& key, const std::vector<double>& pi,
              IndexFn index, unsigned seed) {
    std::printf("[%s] part 2: law with a second writer of the own key\n", tag);
    std::mt19937_64 rng(seed);
    for (int t = 0; t < 2000; ++t) comp->step(rng);
    const std::size_t NB = 40, BL = 25000;
    std::vector<std::vector<double> > own(NB, std::vector<double>(pi.size(), 0.0));
    std::vector<std::vector<double> > shared(NB, std::vector<double>(pi.size(), 0.0));
    for (std::size_t b = 0; b < NB; ++b) {
        for (std::size_t t = 0; t < BL; ++t) {
            comp->step(rng);
            own[b][index(comp->child(0).current())] += 1.0;
            shared[b][index(comp->data().get(key))] += 1.0;
        }
    }
    const law_check r_own    = compare_law(own, BL, pi);
    const law_check r_shared = compare_law(shared, BL, pi);
    const bool ok = r_own.worst_sigma < 5.0 && r_shared.worst_sigma < 5.0;
    std::printf("  Gibbs block output: TV = %.4f, worst state %.2f sigma\n",
                r_own.tv, r_own.worst_sigma);
    std::printf("  shared value after the sweep: TV = %.4f, worst state %.2f sigma\n",
                r_shared.tv, r_shared.worst_sigma);
    std::printf("  both within 5 sigma?                            %s\n",
                ok ? "YES" : "NO");
    return ok;
}

// ---------------------------------------------------------------------------
// Part 3: checks on the own-key entry
// ---------------------------------------------------------------------------

template <class Block, class Cfg>
bool wrong_length_throws(const char* tag, Cfg cfg, input_log inputs,
                         const arma::vec& valid) {
    Block blk(cfg);
    block_context ctx;
    ctx[cfg.name] = valid;
    blk.set_context(ctx);
    std::mt19937_64 rng(31u);
    for (int t = 0; t < 7; ++t) blk.step(rng);
    const arma::vec before = blk.current();

    block_context bad;
    bad[cfg.name] = arma::vec(valid.n_elem + 1, arma::fill::ones);
    blk.set_context(bad);
    const std::mt19937_64 rng_before = rng;
    inputs->clear();
    bool threw = false, names_key = false;
    try {
        blk.step(rng);
    } catch (const std::runtime_error& e) {
        threw = true;
        names_key = std::string(e.what()).find("'" + cfg.name + "'") != std::string::npos;
        std::printf("  message: %s\n", e.what());
    }
    const bool unchanged = same(blk.current(), before) && rng == rng_before
                        && inputs->empty();
    // The context still holds the wrong-length entry: a second step throws too.
    bool threw_again = false;
    try { blk.step(rng); } catch (const std::runtime_error&) { threw_again = true; }
    const bool ok = threw && names_key && unchanged && threw_again && rng == rng_before;
    std::printf("[%s] part 3: wrong-length own key throws, state kept     %s\n",
                tag, ok ? "YES" : "NO");
    return ok;
}

bool test_categorical_entry_checks() {
    input_log inputs = std::make_shared<std::vector<arma::vec> >();
    bool ok = wrong_length_throws<categorical_gibbs_block>(
        "categorical", potts_config(inputs), inputs, arma::vec(kSites, arma::fill::ones));

    // Labels in the entry are snapped as in set_current: rounded, then
    // clamped into {1, ..., K}.
    categorical_gibbs_block blk(potts_config(inputs));
    block_context ctx;
    ctx["zP"] = arma::vec{2.2, 0.0, 7.0};
    blk.set_context(ctx);
    inputs->clear();
    std::mt19937_64 rng(5u);
    blk.step(rng);
    const bool snapped = !inputs->empty() && same((*inputs)[0], arma::vec{2.0, 1.0, 3.0});
    std::printf("[categorical] part 3: entry {2.2, 0, 7} starts the sweep as {2, 1, 3}  %s\n",
                snapped ? "YES" : "NO");
    return ok && snapped;
}

bool test_binary_entry_checks() {
    input_log inputs = std::make_shared<std::vector<arma::vec> >();
    bool ok = wrong_length_throws<binary_gibbs_block>(
        "binary", spin_config(inputs), inputs, arma::vec(kSpins, arma::fill::zeros));

    binary_gibbs_block blk(spin_config(inputs));
    block_context ctx;
    ctx["zS"] = arma::vec{0.7, 0.2, 1.0, 0.0};
    blk.set_context(ctx);
    inputs->clear();
    std::mt19937_64 rng(6u);
    blk.step(rng);
    const bool snapped = !inputs->empty() && same((*inputs)[0], arma::vec{1.0, 0.0, 1.0, 0.0});
    std::printf("[binary] part 3: entry {0.7, 0.2, 1, 0} starts the sweep as {1, 0, 1, 0}  %s\n",
                snapped ? "YES" : "NO");
    return ok && snapped;
}

} // namespace

int main() {
    const std::vector<double> pp = potts_pi();
    const std::vector<double> sp = spin_pi();

    input_log in_c = std::make_shared<std::vector<arma::vec> >();
    const bool ok1 = test_oracle_inputs("categorical", potts_composite(in_c), in_c,
                                        "zP", kSites, 2468u);
    input_log in_b = std::make_shared<std::vector<arma::vec> >();
    const bool ok2 = test_oracle_inputs("binary", spin_composite(in_b), in_b,
                                        "zS", kSpins, 1357u);
    const bool ok3 = test_law("categorical", potts_composite(nullptr), "zP", pp,
                              potts_index, 2468u);
    const bool ok4 = test_law("binary", spin_composite(nullptr), "zS", sp,
                              spin_index, 1357u);
    const bool ok5 = test_categorical_entry_checks();
    const bool ok6 = test_binary_entry_checks();
    const bool all_ok = ok1 && ok2 && ok3 && ok4 && ok5 && ok6;
    std::printf("\n%s\n", all_ok ? "PASS" : "FAIL");
    return all_ok ? 0 : 1;
}
