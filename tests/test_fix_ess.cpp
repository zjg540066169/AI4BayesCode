// Regression test for elliptical_slice_sampling_block::set_context / step.
//
// With an unchanged key set, set_context assigns the entries in place and
// does not copy an identity L_chol over the identity already held; step
// uses z in place of L z when the held L has the bits of the identity and
// no z_i is zero. Both must leave every draw unchanged.
//
// Each scenario runs the block next to a reference block that copies the
// whole context in set_context and forms nu = L z in every step, on equal
// RNG streams, and requires bitwise-equal draws after every sweep. The
// scenarios change the context through every path: a fixed L, an L
// refreshed every sweep, data().set of L (a change in the last element
// only, identity <-> non-identity), an identity with -0.0 off-diagonals,
// changed key sets, the same keys in another iteration order,
// missing / wrong-length L, set_current, freeze, copy and move, and an
// engine state for which z contains exact zeros.
#include <armadillo>
#include "AI4BayesCode/block_sampler.hpp"
#include "AI4BayesCode/shared_data.hpp"
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/elliptical_slice_sampling_block.hpp"
#include "AI4BayesCode/univariate_slice_sampling_block.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using AI4BayesCode::block_context;
using AI4BayesCode::block_sampler;
using AI4BayesCode::composite_block;
using AI4BayesCode::elliptical_slice_sampling_block;
using AI4BayesCode::elliptical_slice_sampling_block_config;
using AI4BayesCode::shared_data_t;
using AI4BayesCode::univariate_slice_sampling_block;
using AI4BayesCode::univariate_slice_sampling_block_config;

namespace {

int fails = 0, checks = 0;
void ok(bool cond, const std::string& what) {
    ++checks;
    if (!cond) ++fails;
    std::printf("  [%s] %s\n", cond ? "PASS" : "FAIL", what.c_str());
}

bool same_bits(const arma::vec& a, const arma::vec& b) {
    return a.n_elem == b.n_elem &&
           (a.n_elem == 0 ||
            std::memcmp(a.memptr(), b.memptr(), a.n_elem * sizeof(double)) == 0);
}

using loglik_fn = std::function<double(const arma::vec&, const block_context&)>;

// Reference ESS update: whole context copied in set_context, nu = L z
// formed in every step.
class ref_ess_block : public block_sampler {
public:
    explicit ref_ess_block(elliptical_slice_sampling_block_config cfg)
        : cfg_(std::move(cfg)) {
        f_ = cfg_.initial_f.n_elem == cfg_.N ? cfg_.initial_f
                                             : arma::vec(cfg_.N, arma::fill::zeros);
    }
    void set_context(const block_context& ctx) override { context_ = ctx; }
    void step(std::mt19937_64& rng) override {
        auto lit = context_.find(cfg_.L_chol_key);
        if (lit == context_.end()) throw std::runtime_error("missing L_chol");
        const arma::vec& L_flat = lit->second;
        if (L_flat.n_elem != cfg_.N * cfg_.N)
            throw std::runtime_error("L_chol length");
        arma::mat L(const_cast<double*>(L_flat.memptr()), cfg_.N, cfg_.N,
                    false, true);
        std::normal_distribution<double>       std_norm(0.0, 1.0);
        std::uniform_real_distribution<double> unif(0.0, 1.0);
        arma::vec z(cfg_.N);
        for (std::size_t i = 0; i < cfg_.N; ++i) z[i] = std_norm(rng);
        arma::vec nu = L * z;
        const double log_lik_cur = cfg_.log_lik(f_, context_);
        if (!std::isfinite(log_lik_cur)) return;
        const double log_y = std::log(unif(rng)) + log_lik_cur;
        const double TWO_PI = 2.0 * M_PI;
        double theta     = unif(rng) * TWO_PI;
        double theta_min = theta - TWO_PI;
        double theta_max = theta;
        arma::vec f_prime(cfg_.N);
        for (std::size_t iter = 0; iter < cfg_.max_shrink_iter; ++iter) {
            const double ct = std::cos(theta);
            const double st = std::sin(theta);
            f_prime = f_ * ct + nu * st;
            const double lp = cfg_.log_lik(f_prime, context_);
            if (std::isfinite(lp) && lp > log_y) {
                f_ = f_prime;
                return;
            }
            if (theta < 0.0) theta_min = theta;
            else             theta_max = theta;
            std::uniform_real_distribution<double> unif_iv(theta_min, theta_max);
            theta = unif_iv(rng);
            if (theta_max - theta_min < 1e-12) return;
        }
    }
    const arma::vec& current() const override { return f_; }
    void set_current(const arma::vec& t) override { f_ = t; }
    const std::string& name() const noexcept override { return cfg_.name; }
    std::size_t dim() const noexcept override { return cfg_.N; }

private:
    elliptical_slice_sampling_block_config cfg_;
    arma::vec                              f_;
    block_context                          context_;
};

elliptical_slice_sampling_block_config ess_cfg(const std::string& name,
                                               std::size_t N, loglik_fn ll) {
    elliptical_slice_sampling_block_config c;
    c.name = name;
    c.N = N;
    c.L_chol_key = "L_chol";
    c.log_lik = std::move(ll);
    return c;
}

std::unique_ptr<block_sampler> make_ess(const elliptical_slice_sampling_block_config& c,
                                        bool use_ref) {
    if (use_ref) return std::make_unique<ref_ess_block>(c);
    return std::make_unique<elliptical_slice_sampling_block>(c);
}

arma::mat corr_matrix(std::size_t N, double ell) {
    arma::vec x = arma::linspace<arma::vec>(0.0, 1.0, N);
    arma::mat R(N, N);
    for (std::size_t j = 0; j < N; ++j)
        for (std::size_t i = 0; i < N; ++i) {
            const double d = x[i] - x[j];
            R(i, j) = std::exp(-0.5 * d * d / (ell * ell));
        }
    R.diag() += 1e-6;
    return R;
}

arma::mat chol_lower(const arma::mat& K) {
    arma::mat L;
    if (!arma::chol(L, K, "lower")) throw std::runtime_error("chol failed");
    return L;
}

arma::vec flat(const arma::mat& M) { return arma::vectorise(M); }

arma::vec identity_flat(std::size_t N, bool neg_zero) {
    arma::vec v(N * N);
    for (std::size_t j = 0; j < N; ++j)
        for (std::size_t i = 0; i < N; ++i)
            v[j * N + i] = (i == j) ? 1.0 : (neg_zero ? -0.0 : 0.0);
    return v;
}

double bern(const arma::vec& f, double b, const arma::vec& y) {
    double s = 0.0;
    for (std::size_t i = 0; i < f.n_elem; ++i) {
        const double fi = f[i] + b;
        const double l1pe = fi > 0 ? fi + std::log1p(std::exp(-fi))
                                   : std::log1p(std::exp(fi));
        s += y[i] * fi - l1pe;
    }
    return s;
}

arma::vec sim_y(std::size_t N, std::uint64_t seed) {
    std::mt19937_64 g(seed);
    arma::vec y(N);
    for (std::size_t i = 0; i < N; ++i) y[i] = double(g() >> 63);
    return y;
}

// The same model built twice, with the library block and with the
// reference block, stepped on equal RNG streams.
class twin {
public:
    using setup_fn = std::function<void(composite_block&, bool)>;
    twin(const setup_fn& setup, std::uint64_t seed, std::vector<std::string> keys)
        : lib_("m"), ref_("m"), rl_(seed), rr_(seed), keys_(std::move(keys)) {
        setup(lib_, false);
        setup(ref_, true);
    }
    template <class Fn> void both(Fn fn) { fn(lib_); fn(ref_); }
    // True when every compared key is bitwise equal after every sweep.
    bool sweeps(int S) {
        bool eq = true;
        for (int s = 0; s < S; ++s) {
            lib_.step(rl_);
            ref_.step(rr_);
            for (const auto& k : keys_)
                eq = eq && same_bits(lib_.data().get(k), ref_.data().get(k));
        }
        return eq;
    }

private:
    composite_block lib_, ref_;
    std::mt19937_64 rl_, rr_;
    std::vector<std::string> keys_;
};

// log_amp slice block; a refresher sets `Lkey` = exp(log_amp) * chol(R)
// after every log_amp update.
void add_log_amp(composite_block& c, const arma::mat& R, const std::string& Lkey,
                 bool centred) {
    const arma::mat LR = chol_lower(R);
    const arma::mat Rinv = arma::inv_sympd(R);
    const double N = double(R.n_rows);
    c.data().set("log_amp", arma::vec{0.0});
    c.data().register_refresher(Lkey, [LR](const shared_data_t& d) {
        return flat(std::exp(d.get("log_amp")[0]) * LR);
    });
    c.data().declare_invalidates("log_amp", {Lkey});
    univariate_slice_sampling_block_config ac;
    ac.name = "log_amp";
    ac.initial_unc = arma::vec{0.0};
    ac.w = 0.3;
    if (centred) {
        ac.log_density = [Rinv, N](const arma::vec& la, const block_context& ctx) {
            const arma::vec& f = ctx.at("f");
            return -0.5 * la[0] * la[0] - N * la[0]
                   - 0.5 * arma::as_scalar(f.t() * Rinv * f) * std::exp(-2.0 * la[0]);
        };
        c.data().declare_dependencies("log_amp", {"f"});
    } else {
        ac.log_density = [LR](const arma::vec& la, const block_context& ctx) {
            const arma::vec f = std::exp(la[0]) * (LR * ctx.at("z"));
            return -0.5 * la[0] * la[0] + bern(f, 0.0, ctx.at("y"));
        };
        c.data().declare_dependencies("log_amp", {"z", "y"});
    }
    c.add_child(std::make_unique<univariate_slice_sampling_block>(ac));
}

// Inverse of the mt19937_64 output tempering.
std::uint64_t untemper(std::uint64_t y) {
    y ^= y >> 43;
    y ^= (y << 37) & 0xFFF7EEE000000000ULL;
    std::uint64_t x = y;
    for (int k = 0; k < 4; ++k) x = y ^ ((x << 17) & 0x71D67FFFEDA60000ULL);
    y = x;
    for (int k = 0; k < 3; ++k) x = y ^ ((x >> 29) & 0x5555555555555555ULL);
    return x;
}

// Engine whose first outputs are `prefix` (at most 150), then those of a
// seeded engine. With state words 0..155 zero, output k is the tempered
// word 156 + k (k < 155).
std::mt19937_64 engine_with_prefix(const std::vector<std::uint64_t>& prefix,
                                   std::uint64_t seed) {
    std::mt19937_64 fill(seed);
    std::ostringstream os;
    for (std::size_t k = 0; k < 312; ++k) {
        std::uint64_t w = 0;
        if (k >= 156) {
            const std::size_t j = k - 156;
            w = untemper(j < prefix.size() ? prefix[j] : fill());
        }
        os << w << (k + 1 < 312 ? " " : "");
    }
    std::istringstream is(os.str());
    std::mt19937_64 g;
    is >> g;
    return g;
}

// ---------------------------------------------------------------------------

void fixed_L_two_blocks() {
    std::printf("fixed dense L, ESS on f + slice on intercept b (N=30)\n");
    const std::size_t N = 30;
    const arma::mat L = chol_lower(corr_matrix(N, 0.15));
    const arma::vec y = sim_y(N, 99);
    twin t([&](composite_block& c, bool use_ref) {
        c.data().set("y", y);
        c.data().set("L_chol", flat(L));
        c.data().declare_dependencies("f", {"L_chol", "y", "b"});
        c.add_child(make_ess(ess_cfg("f", N, [](const arma::vec& f, const block_context& ctx) {
            return bern(f, ctx.at("b")[0], ctx.at("y"));
        }), use_ref));
        univariate_slice_sampling_block_config bc;
        bc.name = "b";
        bc.initial_unc = arma::vec{0.0};
        bc.w = 0.5;
        bc.log_density = [](const arma::vec& b, const block_context& ctx) {
            return -0.005 * b[0] * b[0] + bern(ctx.at("f"), b[0], ctx.at("y"));
        };
        c.data().declare_dependencies("b", {"f", "y"});
        c.add_child(std::make_unique<univariate_slice_sampling_block>(bc));
    }, 4242, {"f", "b"});
    ok(t.sweeps(200), "200 sweeps: f and b bitwise equal to the reference");
}

void refreshed_L() {
    std::printf("centred: L = chol(amp^2 R) refreshed after every log_amp update (N=20)\n");
    const std::size_t N = 20;
    const arma::mat R = corr_matrix(N, 0.2);
    const arma::vec y = sim_y(N, 7);
    twin t([&](composite_block& c, bool use_ref) {
        c.data().set("y", y);
        c.data().set("L_chol", flat(chol_lower(R)));
        c.data().declare_dependencies("f", {"L_chol", "y"});
        c.add_child(make_ess(ess_cfg("f", N, [](const arma::vec& f, const block_context& ctx) {
            return bern(f, 0.0, ctx.at("y"));
        }), use_ref));
        add_log_amp(c, R, "L_chol", true);
    }, 31337, {"f", "log_amp"});
    ok(t.sweeps(100), "100 sweeps: f and log_amp bitwise equal to the reference");
}

void whitened_identity() {
    std::printf("whitened: L_chol = identity, L_K rescaled every sweep (N=25)\n");
    const std::size_t N = 25;
    const arma::mat R = corr_matrix(N, 0.2);
    const arma::vec y = sim_y(N, 5);
    for (int neg = 0; neg < 2; ++neg) {
        twin t([&](composite_block& c, bool use_ref) {
            c.data().set("y", y);
            c.data().set("L_K", flat(chol_lower(R)));
            c.data().set("L_chol", identity_flat(N, neg == 1));
            c.data().declare_dependencies("z", {"L_chol", "L_K", "y"});
            c.add_child(make_ess(ess_cfg("z", N, [](const arma::vec& z, const block_context& ctx) {
                const arma::vec& Lf = ctx.at("L_K");
                const arma::uword n = z.n_elem;
                arma::mat LK(const_cast<double*>(Lf.memptr()), n, n, false, true);
                const arma::vec f = LK * z;
                return bern(f, 0.0, ctx.at("y"));
            }), use_ref));
            add_log_amp(c, R, "L_K", false);
        }, 2718, {"z", "log_amp"});
        const std::string tag = neg ? "identity with -0.0 off-diagonals" : "identity";
        ok(t.sweeps(150), "150 sweeps, " + tag + ": z and log_amp bitwise equal");
        t.both([&](composite_block& c) {
            arma::vec I = identity_flat(N, neg == 1);
            I[N * N - 1] = 0.5;                       // no longer an identity
            c.data().set("L_chol", I);
        });
        ok(t.sweeps(40), "40 sweeps after the last diagonal entry is set to 0.5");
        t.both([&](composite_block& c) {
            c.data().set("L_chol", identity_flat(N, neg == 1));
        });
        ok(t.sweeps(40), "40 sweeps after the identity is restored");
    }
}

void setters_and_loaders() {
    std::printf("setters and loaders on a composite; log_lik uses the last element of L\n");
    for (std::size_t N : {12u, 3u}) {
        const std::size_t NN = N * N;
        const arma::mat L1 = chol_lower(corr_matrix(N, 0.2));
        const arma::mat L2 = chol_lower(corr_matrix(N, 0.35));
        const arma::vec y = sim_y(N, 3);
        twin t([&](composite_block& c, bool use_ref) {
            c.data().set("y", y);
            c.data().set("L_chol", flat(L1));
            c.data().declare_dependencies("f", {"L_chol", "y"});
            c.add_child(make_ess(ess_cfg("f", N, [NN](const arma::vec& f, const block_context& ctx) {
                return bern(f, 0.25 * ctx.at("L_chol")[NN - 1], ctx.at("y"));
            }), use_ref));
        }, 555, {"f"});
        const std::string n = " (N=" + std::to_string(N) + ")";
        ok(t.sweeps(40), "fixed L1" + n);
        arma::vec L1_last = flat(L1);
        L1_last[NN - 1] = std::nextafter(L1_last[NN - 1], 10.0);
        t.both([&](composite_block& c) { c.data().set("L_chol", L1_last); });
        ok(t.sweeps(30), "data().set of L changing only its last element" + n);
        t.both([&](composite_block& c) { c.data().set("L_chol", flat(L1)); });
        ok(t.sweeps(30), "data().set of L back to L1" + n);
        t.both([&](composite_block& c) { c.data().set("L_chol", identity_flat(N, false)); });
        ok(t.sweeps(30), "data().set of L to the identity" + n);
        arma::vec I_last = identity_flat(N, false);
        I_last[NN - 1] = 0.5;
        t.both([&](composite_block& c) { c.data().set("L_chol", I_last); });
        ok(t.sweeps(30), "data().set of L to the identity with its last entry 0.5" + n);
        t.both([&](composite_block& c) { c.data().set("L_chol", flat(L2)); });
        ok(t.sweeps(30), "data().set of L to L2" + n);
        t.both([&](composite_block& c) { c.data().set("y", 1.0 - y); });
        ok(t.sweeps(30), "data().set of y with L unchanged" + n);
        arma::vec theta(N);
        for (std::size_t i = 0; i < N; ++i) theta[i] = 0.1 * double(i % 7) - 0.3;
        t.both([&](composite_block& c) { c.set_current(theta); });
        ok(t.sweeps(30), "composite set_current" + n);
        t.both([&](composite_block& c) { c.child(0).set_current(-theta); });
        ok(t.sweeps(30), "child set_current" + n);
        t.both([&](composite_block& c) {
            c.freeze({"f"}, true);
            c.data().set("L_chol", flat(L1));
        });
        ok(t.sweeps(10), "frozen while L changes" + n);
        t.both([&](composite_block& c) { c.unfreeze({"f"}); });
        ok(t.sweeps(30), "unfrozen with the changed L" + n);
    }
}

// The library and reference blocks driven directly with identical calls.
struct direct_pair {
    elliptical_slice_sampling_block lib;
    ref_ess_block ref;
    std::mt19937_64 rl, rr;
    direct_pair(const elliptical_slice_sampling_block_config& c, std::uint64_t seed)
        : lib(c), ref(c), rl(seed), rr(seed) {}
    void set(const block_context& ctx) { lib.set_context(ctx); ref.set_context(ctx); }
    bool steps(int S) {
        bool eq = true;
        for (int s = 0; s < S; ++s) {
            lib.step(rl);
            ref.step(rr);
            eq = eq && same_bits(lib.current(), ref.current());
        }
        return eq;
    }
    // True when both throw and the state is unchanged.
    bool both_throw() {
        bool tl = false, tr = false;
        try { lib.step(rl); } catch (const std::exception&) { tl = true; }
        try { ref.step(rr); } catch (const std::exception&) { tr = true; }
        return tl && tr && same_bits(lib.current(), ref.current());
    }
};

void direct_paths() {
    std::printf("direct use: key-set changes, bad L, copy and move\n");
    const std::size_t N = 10, NN = N * N;
    const arma::mat L1 = chol_lower(corr_matrix(N, 0.25));
    const arma::mat L2 = chol_lower(corr_matrix(N, 0.4));
    const arma::vec y = sim_y(N, 21);
    // The offset depends on the last entry of L and on the sign bit of
    // entry 1, so a held L whose bits differ from the latest context changes
    // the draws.
    const auto cfg = ess_cfg("f", N, [](const arma::vec& f, const block_context& ctx) {
        const arma::vec& Lf = ctx.at("L_chol");
        const arma::uword nn = f.n_elem * f.n_elem;
        if (Lf.n_elem != nn) return bern(f, 0.0, ctx.at("y"));
        const double off = 0.25 * Lf[nn - 1] + (std::signbit(Lf[1]) ? 0.3 : 0.0);
        return bern(f, off, ctx.at("y"));
    });
    direct_pair p(cfg, 808);
    block_context c1;
    c1["L_chol"] = flat(L1);
    c1["y"] = y;
    p.set(c1);
    ok(p.steps(20), "one set_context, 20 steps");
    bool eq = true;
    for (int s = 0; s < 20; ++s) { p.set(c1); eq = p.steps(1) && eq; }
    ok(eq, "the same context before each of 20 steps");
    block_context c_extra = c1;
    c_extra["extra"] = arma::vec{1.0, 2.0};
    p.set(c_extra);
    ok(p.steps(10), "an extra key with L unchanged");
    p.set(c1);
    ok(p.steps(10), "the extra key removed");
    block_context c_other;
    c_other["L_chol"] = flat(L1);
    c_other["w"] = arma::vec{3.0};
    c_other["y"] = y;
    p.set(c_other);
    ok(p.steps(10), "another key set of the same size");
    block_context c_y = c1;
    c_y["y"] = 1.0 - y;
    p.set(c_y);
    ok(p.steps(10), "y changed, L unchanged");
    block_context c2 = c1;
    c2["L_chol"] = flat(L2);
    p.set(c2);
    ok(p.steps(10), "L changed");
    block_context c_missing;
    c_missing["y"] = y;
    p.set(c_missing);
    ok(p.both_throw(), "missing L: both throw");
    block_context c_short = c1;
    c_short["L_chol"] = arma::vec(NN - 1, arma::fill::ones);
    p.set(c_short);
    ok(p.both_throw(), "wrong-length L: both throw");
    p.set(c_short);
    ok(p.both_throw(), "the same wrong-length L again: both throw");
    p.set(c1);
    ok(p.steps(10), "valid L after the errors");
    block_context c_id = c1;
    c_id["L_chol"] = identity_flat(N, false);
    p.set(c_id);
    ok(p.steps(10), "identity L");
    block_context c_unit = c1;                         // ones on the diagonal
    c_unit["L_chol"] = identity_flat(N, false);
    for (std::size_t i = 1; i < N; ++i) c_unit["L_chol"][(i - 1) * N + i] = 0.5;
    p.set(c_unit);
    ok(p.steps(10), "unit lower-triangular L (not the identity)");
    p.set(c_id);
    ok(p.steps(10), "identity L again");
    block_context c_corner = c1;                       // entry (N-2, N-1) set
    c_corner["L_chol"] = identity_flat(N, false);
    c_corner["L_chol"][NN - 2] = 0.5;
    p.set(c_corner);
    ok(p.steps(10), "identity except one entry above the last diagonal entry");
    p.set(c_id);
    ok(p.steps(10), "identity L once more");
    block_context c_id_neg = c1;                       // -0.0 off-diagonals
    c_id_neg["L_chol"] = identity_flat(N, true);
    p.set(c_id_neg);
    ok(p.steps(10), "identity with -0.0 off-diagonals after the identity");
    p.set(c_id);
    ok(p.steps(10), "identity after the one with -0.0 off-diagonals");
    block_context c2_extra = c2;                       // new key set, dense L
    c2_extra["extra"] = arma::vec{4.0};
    p.set(c2_extra);
    ok(p.steps(10), "new key set with a dense L while an identity is held");
    block_context c_id_extra = c_id;                   // new key set, identity
    c_id_extra["extra2"] = arma::vec{5.0};
    p.set(c_id_extra);
    ok(p.steps(10), "new key set with an identity L");
    p.set(c_id);
    ok(p.steps(10), "back to the identity context");

    elliptical_slice_sampling_block lcp(p.lib);       // copy construction
    ref_ess_block rcp(p.ref);
    std::mt19937_64 gl = p.rl, gr = p.rr;
    bool eqc = true;
    for (int s = 0; s < 10; ++s) {
        lcp.step(gl);
        rcp.step(gr);
        eqc = eqc && same_bits(lcp.current(), rcp.current());
    }
    bool src_eq = p.steps(10);
    ok(eqc && src_eq, "copy-constructed block and its source continue equal");
    lcp.set_context(c1);
    rcp.set_context(c1);
    p.set(c_id);
    eqc = true;
    for (int s = 0; s < 10; ++s) {
        lcp.step(gl);
        rcp.step(gr);
        eqc = eqc && same_bits(lcp.current(), rcp.current());
    }
    src_eq = p.steps(10);
    ok(eqc && src_eq, "copy and source with different L afterwards");
    lcp = p.lib;                                       // copy assignment
    rcp = p.ref;
    gl = p.rl;
    gr = p.rr;
    eqc = true;
    for (int s = 0; s < 10; ++s) {
        lcp.step(gl);
        rcp.step(gr);
        eqc = eqc && same_bits(lcp.current(), rcp.current());
    }
    ok(eqc, "copy-assigned block (identity L held)");
    elliptical_slice_sampling_block lmv(std::move(lcp));   // move construction
    ref_ess_block rmv(std::move(rcp));
    eqc = true;
    for (int s = 0; s < 10; ++s) {
        lmv.step(gl);
        rmv.step(gr);
        eqc = eqc && same_bits(lmv.current(), rmv.current());
    }
    lmv.set_context(c2);
    rmv.set_context(c2);
    for (int s = 0; s < 10; ++s) {
        lmv.step(gl);
        rmv.step(gr);
        eqc = eqc && same_bits(lmv.current(), rmv.current());
    }
    ok(eqc, "move-constructed block, before and after a new L");
    elliptical_slice_sampling_block lma(cfg);           // move assignment
    ref_ess_block rma(cfg);
    lma = std::move(lmv);
    rma = std::move(rmv);
    lma.set_context(c_id);
    rma.set_context(c_id);
    eqc = true;
    for (int s = 0; s < 10; ++s) {
        lma.step(gl);
        rma.step(gr);
        eqc = eqc && same_bits(lma.current(), rma.current());
    }
    ok(eqc, "move-assigned block with identity L");
}

// Contexts with the same keys in different iteration orders (other
// insertion orders and bucket counts): set_context assigns each value to the
// held entry of its own key. log_lik uses a weighted sum of the scalar keys
// a..f, so a value held under another key changes the draws.
void key_order() {
    std::printf("direct use: the same keys in different iteration orders\n");
    const std::size_t N = 6;
    const arma::mat L1 = chol_lower(corr_matrix(N, 0.3));
    const arma::vec y = sim_y(N, 31);
    const std::vector<std::string> keys = {"a", "b", "c", "d", "e", "f", "L_chol", "y"};
    const auto cfg = ess_cfg("f", N, [](const arma::vec& f, const block_context& ctx) {
        const char* names[] = {"a", "b", "c", "d", "e", "f"};
        double off = 0.0;
        for (int j = 0; j < 6; ++j) off += 0.05 * (j + 1) * ctx.at(names[j])[0];
        return bern(f, off, ctx.at("y"));
    });
    auto make = [&](int variant, double base) {
        block_context c;
        if (variant % 3 == 1) c.reserve(64);
        std::vector<std::string> ks = keys;
        if (variant % 2 == 1) std::reverse(ks.begin(), ks.end());
        for (const auto& k : ks) {
            if (k == "L_chol")  c[k] = flat(L1);
            else if (k == "y")  c[k] = y;
            else                c[k] = arma::vec{base + static_cast<double>(k[0] - 'a')};
        }
        if (variant % 3 == 2) c.rehash(97);
        return c;
    };
    auto order = [](const block_context& c) {
        std::string o;
        for (const auto& kv : c) o += kv.first + ",";
        return o;
    };
    direct_pair p(cfg, 909);
    auto steps = [&p](int S) {   // a step that throws counts as a difference
        try { return p.steps(S); } catch (const std::exception&) { return false; }
    };
    const block_context c0 = make(0, 0.1);
    p.set(c0);
    bool eq = steps(5);
    bool other_order = false;
    for (int v = 1; v < 12; ++v) {
        const block_context c = make(v, 0.1 + 0.37 * v);
        other_order = other_order || order(c) != order(c0);
        p.set(c);
        eq = steps(5) && eq;
    }
    ok(other_order, "some contexts iterate their keys in another order");
    ok(eq, "12 contexts with the same keys, 5 steps each: bitwise equal");
}

// Outputs 0 and 2^63 give the polar method u = -1, v = 0, s = 1 exactly,
// so the first two normals are exactly zero and step forms nu = L z for an
// identity L; the block must still match the reference.
void zero_in_z() {
    std::printf("engine state for which z contains exact zeros\n");
    const std::uint64_t half = 1ULL << 63;
    const std::uint64_t x_theta = 7378697629483820032ULL;   // 0.4 * 2^64
    const std::vector<std::uint64_t> prefix = {0, half, half, x_theta};
    {
        std::mt19937_64 g = engine_with_prefix(prefix, 1);
        bool match = true;
        for (std::uint64_t v : prefix) match = match && (g() == v);
        ok(match, "crafted engine yields the requested outputs");
        std::mt19937_64 g2 = engine_with_prefix(prefix, 1);
        std::normal_distribution<double> nd(0.0, 1.0);
        const double a = nd(g2), b = nd(g2);
        if (a != 0.0 || b != 0.0)
            std::printf("  NOTE: this standard library's normal_distribution "
                        "gives %g, %g here; the zero case is not reached\n", a, b);
    }
    for (std::size_t N : {2u, 7u}) {
        for (int neg = 0; neg < 2; ++neg) {
            const auto cfg = ess_cfg("f", N, [](const arma::vec& f, const block_context&) {
                double s = 0.0;
                for (std::size_t i = 0; i < f.n_elem; ++i) s -= 0.5 * f[i] * f[i];
                return s;
            });
            elliptical_slice_sampling_block lib(cfg);
            ref_ess_block ref(cfg);
            block_context ctx;
            ctx["L_chol"] = identity_flat(N, neg == 1);
            lib.set_context(ctx);
            ref.set_context(ctx);
            std::mt19937_64 gl = engine_with_prefix(prefix, 2 + N);
            std::mt19937_64 gr = gl;
            bool eq = true;
            for (int s = 0; s < 30; ++s) {
                lib.step(gl);
                ref.step(gr);
                eq = eq && same_bits(lib.current(), ref.current());
            }
            ok(eq, "N=" + std::to_string(N) + (neg ? ", identity with -0.0" : ", identity") +
                   ": 30 steps bitwise equal, first step with zeros in z");
        }
    }
}

} // namespace

int main() {
    std::printf("test_fix_ess: elliptical_slice_sampling_block context and identity shortcuts\n");
    fixed_L_two_blocks();
    refreshed_L();
    whitened_identity();
    setters_and_loaders();
    direct_paths();
    key_order();
    zero_in_z();
    std::printf("\n%d / %d checks passed\n", checks - fails, checks);
    if (fails) {
        std::printf("FAIL\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
