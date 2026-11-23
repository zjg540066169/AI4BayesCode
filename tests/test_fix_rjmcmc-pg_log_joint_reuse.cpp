/*================================================================================
 *  rjmcmc_block: log_joint at the current state is not evaluated again with
 *  unchanged arguments, and the draws are unchanged.
 *
 *  Within one step() only gamma_, beta_ and the gamma_key / beta_key entries
 *  of the context change, so after a coordinate's accept / reject the
 *  log_joint of the resulting state is already known. step() reuses it at the
 *  next coordinate instead of calling log_joint again with the same
 *  (gamma, beta, ctx).
 *
 *  T1  In every sweep with no set_current since the last set_context or
 *      step(), no log_joint call repeats the arguments (gamma, beta and every
 *      context entry, compared exactly) of an earlier call in the same step().
 *  T2  The block's draws and step diagnostics are bit-identical to a
 *      reference implementation that evaluates log_joint at the current state
 *      at every coordinate, through set_current before and after set_context,
 *      steps without set_context, a context with a key removed and one added,
 *      non-finite continuous_update / propose_logq / log_joint values and a
 *      transform that is degenerate for some inputs. log_joint also uses the
 *      context copies of gamma and beta, so a value computed with an
 *      out-of-date context entry would change the draws.
 *
 *  Scenarios: continuous_update, RW and neither, without a transform;
 *  continuous_update and RW with diagonal_linear_transform_1d, and with the
 *  degenerate-edge hooks; a continuous_update that returns the current value
 *  of beta_j unchanged with probability 0.4 (as an update with its own
 *  accept / reject step does on rejection) and with probability 0.15 a zero
 *  (-beta_j when beta_j is a zero, +0.0 otherwise), with a log_joint that
 *  depends on the sign bit of each active beta_j, so -0.0 in place of 0.0 has
 *  to count as a change.
 *================================================================================*/

#include "AI4BayesCode/block_sampler.hpp"
#include "AI4BayesCode/rjmcmc_block.hpp"
#include "AI4BayesCode/rjmcmc_transforms.hpp"

#include <armadillo>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace AI4BayesCode;

int n_pass = 0, n_fail = 0;
void check(bool ok, const std::string& tag, const std::string& detail = "") {
    if (ok) { ++n_pass; std::printf("  PASS  %s %s\n", tag.c_str(), detail.c_str()); }
    else    { ++n_fail; std::printf("  FAIL  %s %s\n", tag.c_str(), detail.c_str()); }
}

bool same_bytes(const arma::vec& a, const arma::vec& b) {
    return a.n_elem == b.n_elem &&
           (a.n_elem == 0 ||
            std::memcmp(a.memptr(), b.memptr(), a.n_elem * sizeof(double)) == 0);
}

// FNV-1a over every context entry, keys in sorted order.
std::uint64_t ctx_hash(const block_context& c) {
    std::vector<std::string> keys;
    for (const auto& kv : c) keys.push_back(kv.first);
    std::sort(keys.begin(), keys.end());
    std::uint64_t h = 1469598103934665603ULL;
    auto mix = [&h](const void* p, std::size_t n) {
        const unsigned char* b = static_cast<const unsigned char*>(p);
        for (std::size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ULL; }
    };
    for (const auto& k : keys) {
        mix(k.data(), k.size());
        const arma::vec& v = c.at(k);
        const std::uint64_t n = v.n_elem;
        mix(&n, sizeof n);
        mix(v.memptr(), v.n_elem * sizeof(double));
    }
    return h;
}

// Degenerate for |u| > 1.5 (forward) and |beta| > 2 (reverse).
class edge_transform : public rjmcmc_transforms::transform_1d_base {
public:
    double apply_forward(double u, double& b) const override {
        b = 1.1 * u;
        return std::abs(u) > 1.5 ? std::numeric_limits<double>::quiet_NaN() : 1.1;
    }
    double apply_reverse(double b, double& u) const override {
        u = b / 1.1;
        return std::abs(b) > 2.0 ? -1.0 : 1.0 / 1.1;
    }
};

// Reference: rjmcmc_block::step with log_joint evaluated at the current state
// at every coordinate (same RNG calls in the same order).
struct ref_rj {
    rjmcmc_block_config cfg;
    arma::vec gamma, beta;
    block_context ctx;
    std::size_t nb_try = 0, nb_ok = 0, nd_try = 0, nd_ok = 0, nrw_try = 0, nrw_ok = 0;

    explicit ref_rj(const rjmcmc_block_config& c) : cfg(c), gamma(c.p), beta(c.p) {
        for (std::size_t j = 0; j < c.p; ++j) {
            gamma[j] = c.gamma_init[j] >= 0.5 ? 1.0 : 0.0;
            beta[j] = c.beta_init[j];
        }
    }
    void set_context(const block_context& c) {
        ctx = c;
        ctx[cfg.gamma_key] = gamma;
        ctx[cfg.beta_key] = beta;
    }
    void set_current(const arma::vec& th) {
        for (std::size_t j = 0; j < cfg.p; ++j) {
            gamma[j] = th[j] >= 0.5 ? 1.0 : 0.0;
            beta[j] = th[cfg.p + j];
        }
    }
    void sync() { ctx[cfg.gamma_key] = gamma; ctx[cfg.beta_key] = beta; }
    void step(std::mt19937_64& rng) {
        const std::size_t p = cfg.p;
        std::vector<std::size_t> order(p);
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), rng);
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        std::normal_distribution<double> rw(0.0, 1.0);
        nb_try = nb_ok = nd_try = nd_ok = nrw_try = nrw_ok = 0;
        for (std::size_t j : order) {
            if (gamma[j] == 1.0 && cfg.continuous_update) {
                const double bn = cfg.continuous_update(rng, j, ctx);
                ++nrw_try;
                if (std::isfinite(bn)) { beta[j] = bn; ctx[cfg.beta_key] = beta; ++nrw_ok; }
            } else if (gamma[j] == 1.0 && cfg.rw_scale > 0.0) {
                ++nrw_try;
                const double bo = beta[j], bn = bo + cfg.rw_scale * rw(rng);
                beta[j] = bo; ctx[cfg.beta_key] = beta;
                const double lo = cfg.log_joint(gamma, beta, ctx);
                beta[j] = bn; ctx[cfg.beta_key] = beta;
                const double ln = cfg.log_joint(gamma, beta, ctx);
                if (std::log(uniform(rng)) < ln - lo) ++nrw_ok;
                else { beta[j] = bo; ctx[cfg.beta_key] = beta; }
            }
            const double gc = gamma[j], bc = beta[j];
            sync();
            const double lc = cfg.log_joint(gamma, beta, ctx);
            if (gc == 0.0) {
                ++nb_try;
                const double u = cfg.propose_sample(rng, j, ctx);
                const double lq = cfg.propose_logq(u, j, ctx);
                if (!std::isfinite(lq)) continue;
                double bn, lJ;
                if (cfg.transform) {
                    const double d = cfg.transform->apply_forward(u, bn);
                    if (!std::isfinite(d) || !(d > 0.0)) continue;
                    lJ = std::log(d);
                } else { bn = u; lJ = 0.0; }
                gamma[j] = 1.0; beta[j] = bn; sync();
                const double ln = cfg.log_joint(gamma, beta, ctx);
                const double lr = ln - lc - lq + lJ;
                if (std::isfinite(lr) && std::log(uniform(rng)) < lr) ++nb_ok;
                else { gamma[j] = gc; beta[j] = bc; sync(); }
            } else {
                ++nd_try;
                double ui, lJ;
                if (cfg.transform) {
                    const double d = cfg.transform->apply_reverse(bc, ui);
                    if (!std::isfinite(d) || !(d > 0.0)) continue;
                    lJ = std::log(d);
                } else { ui = bc; lJ = 0.0; }
                const double lq = cfg.propose_logq(ui, j, ctx);
                if (!std::isfinite(lq)) continue;
                gamma[j] = 0.0; beta[j] = 0.0; sync();
                const double ln = cfg.log_joint(gamma, beta, ctx);
                const double lr = ln - lc + lq + lJ;
                if (std::isfinite(lr) && std::log(uniform(rng)) < lr) ++nd_ok;
                else { gamma[j] = gc; beta[j] = bc; sync(); }
            }
        }
    }
};

struct call_rec { arma::vec g, b; std::uint64_t h; };

struct scenario {
    const char* name;
    bool cu, rw, linear, edge;
    bool keep;   // continuous_update returns the current beta_j with probability 0.4, and with
                 // probability 0.15 a zero: -beta_j when beta_j is a zero, +0.0 otherwise
};

void run(const scenario& sc) {
    const std::size_t N = 40, p = 12;
    const int S = 150;
    std::mt19937_64 gen(2026u + p);
    std::normal_distribution<double> nz(0.0, 1.0);
    arma::vec Xf(N * p), y(N);
    for (auto& v : Xf) v = nz(gen);
    for (std::size_t i = 0; i < N; ++i) {
        y[i] = 1.8 * Xf[i] - 1.3 * Xf[i + 5 * N] + nz(gen);
    }
    const bool edge = sc.edge;
    auto getd = [](const block_context& c, const char* k, double dflt) {
        const auto it = c.find(k);
        return it == c.end() ? dflt : it->second[0];
    };

    // Repeat detection for the block's calls (not the reference's).
    bool counting = false;
    std::vector<call_rec> calls;
    std::size_t n_block_calls = 0, n_repeats = 0;
    // Separate counters, so the block and the reference return NaN at the
    // same calls.
    std::size_t n_cu_blk = 0, n_cu_ref = 0;

    rjmcmc_block_config cfg;
    cfg.name = "gb"; cfg.gamma_key = "gamma"; cfg.beta_key = "beta"; cfg.p = p;
    cfg.log_joint = [&](const arma::vec& gm, const arma::vec& b, const block_context& c) -> double {
        if (counting) {
            ++n_block_calls;
            const std::uint64_t h = ctx_hash(c);
            for (const call_rec& r : calls)
                if (r.h == h && same_bytes(r.g, gm) && same_bytes(r.b, b)) { ++n_repeats; break; }
            calls.push_back(call_rec{gm, b, h});
        }
        const arma::vec& yy = c.at("y");
        const arma::vec& X = c.at("X");
        const double sg = getd(c, "sigma", 1.0), ta = getd(c, "tau", 1.0), pv = getd(c, "pi", 0.3);
        const std::size_t n = yy.n_elem;
        double sse = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            double r = yy[i];
            for (std::size_t j = 0; j < p; ++j) if (gm[j] > 0.5) r -= X[i + j * n] * b[j];
            sse += r * r;
        }
        double lp = -0.5 * sse / (sg * sg) - double(n) * std::log(sg);
        std::size_t k = 0;
        double sb = 0.0;
        for (std::size_t j = 0; j < p; ++j) if (gm[j] > 0.5) {
            ++k; sb += std::abs(b[j]);
            lp += -0.5 * b[j] * b[j] / (sg * sg * ta * ta) - std::log(sg * ta);
        }
        lp += double(k) * std::log(pv) + double(p - k) * std::log(1.0 - pv);
        // Zero when the context copies equal the arguments, large otherwise.
        lp += 2.0 * (arma::accu(c.at("gamma")) - arma::accu(gm)) +
              2.0 * (arma::accu(c.at("beta")) - arma::accu(b));
        if (sc.keep)
            for (std::size_t j = 0; j < p; ++j)
                if (gm[j] > 0.5 && std::signbit(b[j])) lp += 0.3;
        if (edge && sb > 6.0) return -std::numeric_limits<double>::infinity();
        if (edge && k == p) return std::numeric_limits<double>::quiet_NaN();
        return lp;
    };
    cfg.propose_sample = [&](std::mt19937_64& r, std::size_t, const block_context& c) {
        std::normal_distribution<double> d(0.0, getd(c, "sigma", 1.0) * getd(c, "tau", 1.0));
        return d(r);
    };
    cfg.propose_logq = [&](double u, std::size_t, const block_context& c) {
        if (edge && std::abs(u) > 2.5) return -std::numeric_limits<double>::infinity();
        const double s = getd(c, "sigma", 1.0) * getd(c, "tau", 1.0);
        return -0.5 * std::log(2.0 * M_PI * s * s) - u * u / (2.0 * s * s);
    };
    if (sc.cu) {
        cfg.continuous_update = [&](std::mt19937_64& r, std::size_t j, const block_context& c) {
            const std::size_t n_cu = counting ? ++n_cu_blk : ++n_cu_ref;
            if (edge && n_cu % 7 == 0) return std::numeric_limits<double>::quiet_NaN();
            if (sc.keep) {
                const double cur = c.at("beta")[j];
                const double u = std::uniform_real_distribution<double>(0.0, 1.0)(r);
                if (u < 0.40) return cur;                          // same bytes
                if (u < 0.55) return cur == 0.0 ? -cur : 0.0;      // a zero; the other sign if cur is one
            }
            const arma::vec& yy = c.at("y");
            const arma::vec& X = c.at("X");
            const arma::vec& gm = c.at("gamma");
            const arma::vec& b = c.at("beta");
            const double sg = getd(c, "sigma", 1.0), ta = getd(c, "tau", 1.0);
            const std::size_t n = yy.n_elem;
            double xtr = 0.0, xtx = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                double r0 = yy[i];
                for (std::size_t k = 0; k < p; ++k)
                    if (k != j && gm[k] > 0.5) r0 -= X[i + k * n] * b[k];
                xtr += X[i + j * n] * r0;
                xtx += X[i + j * n] * X[i + j * n];
            }
            const double den = xtx + 1.0 / (ta * ta);
            std::normal_distribution<double> d(xtr / den, sg / std::sqrt(den));
            return d(r);
        };
    }
    cfg.rw_scale = sc.rw ? 0.08 : 0.0;
    if (sc.linear) cfg.transform = std::make_shared<rjmcmc_transforms::diagonal_linear_transform_1d>(1.3);
    if (edge) cfg.transform = std::make_shared<edge_transform>();
    cfg.gamma_init = arma::vec(p, arma::fill::zeros);
    cfg.beta_init = arma::vec(p, arma::fill::zeros);
    for (std::size_t r = 0; r < 3; ++r) { cfg.gamma_init[r] = 1.0; cfg.beta_init[r] = 0.4; }

    rjmcmc_block blk(cfg);
    ref_rj ref(cfg);
    std::mt19937_64 rng_blk(17), rng_ref(17), aux(23);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    std::size_t n_mismatch = 0, n_repeats_checked = 0;
    block_context ctx;
    ctx["y"] = y;
    ctx["X"] = Xf;
    // True while set_current has run after the last set_context: the
    // context then holds the previous gamma / beta, and T1 skips the sweep.
    bool stale = false;
    for (int s = 0; s < S; ++s) {
        arma::vec th(2 * p, arma::fill::zeros);
        for (std::size_t j = 0; j < p; ++j)
            if (u01(aux) < 0.35) {
                th[j] = 1.0; th[p + j] = nz(aux);
                if (sc.keep && u01(aux) < 0.3) th[p + j] = (u01(aux) < 0.5) ? 0.0 : -0.0;
            }
        if (s % 13 == 5) { blk.set_current(th); ref.set_current(th); stale = true; }
        ctx["sigma"] = arma::vec{1.0 + 0.1 * std::sin(0.37 * s)};
        ctx["tau"] = arma::vec{1.5 + 0.2 * std::cos(0.23 * s)};
        ctx["pi"] = arma::vec{0.3 + 0.1 * std::sin(0.11 * s)};
        if (s % 19 != 7) { blk.set_context(ctx); ref.set_context(ctx); stale = false; }
        if (s % 23 == 11) {
            block_context c2 = ctx;
            c2.erase("pi");
            c2["extra"] = arma::vec(4, arma::fill::ones);
            blk.set_context(c2); ref.set_context(c2); stale = false;
        }
        if (s % 17 == 3) { blk.set_current(th); ref.set_current(th); stale = true; }

        calls.clear();
        const std::size_t rep0 = n_repeats;
        counting = true;
        blk.step(rng_blk);
        counting = false;
        if (!stale) n_repeats_checked += n_repeats - rep0;
        ref.step(rng_ref);
        stale = false;   // step() leaves the context holding gamma / beta

        arma::vec want(2 * p);
        want.head(p) = ref.gamma;
        want.tail(p) = ref.beta;
        const auto d = blk.last_step_diagnostics();
        if (!same_bytes(blk.current(), want) ||
            d.n_birth_try != ref.nb_try || d.n_birth_ok != ref.nb_ok ||
            d.n_death_try != ref.nd_try || d.n_death_ok != ref.nd_ok ||
            d.n_rw_try != ref.nrw_try || d.n_rw_ok != ref.nrw_ok) {
            ++n_mismatch;
        }
    }
    char buf[200];
    std::snprintf(buf, sizeof buf, "(%zu log_joint calls in %d sweeps)", n_block_calls, S);
    check(n_repeats_checked == 0,
          std::string("T1 ") + sc.name + ": repeated log_joint evaluations = " +
              std::to_string(n_repeats_checked), buf);
    check(n_mismatch == 0,
          std::string("T2 ") + sc.name + ": sweeps differing from reference = " +
              std::to_string(n_mismatch) + " of " + std::to_string(S));
}

}  // namespace

int main() {
    std::printf("=== test_fix_rjmcmc-pg_log_joint_reuse ===\n");
    const scenario scs[] = {
        {"cu",       true,  false, false, false, false},
        {"rw",       false, true,  false, false, false},
        {"none",     false, false, false, false, false},
        {"cu+lin",   true,  false, true,  false, false},
        {"rw+lin",   false, true,  true,  false, false},
        {"cu+edge",  true,  false, false, true,  false},
        {"rw+edge",  false, true,  false, true,  false},
        {"cu+keep",  true,  false, false, false, true},
    };
    for (const auto& sc : scs) run(sc);
    std::printf("\n=== SUMMARY: %d passed, %d failed ===\n", n_pass, n_fail);
    return n_fail == 0 ? 0 : 1;
}
