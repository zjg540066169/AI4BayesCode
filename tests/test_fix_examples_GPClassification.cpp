// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_examples_GPClassification.cpp
//
//  examples/GPClassification.cpp: the K_matrix refresher (K_matrix ->
//  L_chol -> f = L z, reported by get_current() and used by predict_at)
//  after the wrapper is moved, and set_current at values for which
//  K + amplitude^2 jitter I has no Cholesky factor.
//    C1  a model moved into another object (move construction; the
//        moved-from source is not destroyed) gives the same f, amplitude and
//        lengthscale after 2 sweeps as a twin with the same seed that is not
//        moved;
//    C2  the same after move assignment into an existing model;
//    C3  after one sweep, set_current(amplitude = 1e200) and
//        set_current(amplitude = +Inf): either the call throws and leaves
//        every shared value and the joint block's current value unchanged
//        (bit for bit), or f equals L z for the Cholesky factor L of
//        K_matrix + amplitude^2 1e-8 I at the loaded values (relative
//        tolerance 1e-10);
//    C4  set_current with an empty amplitude, or an empty lengthscale, throws
//        and leaves get_current() unchanged.
//  C3 and C4 use the first 20 of the 40 inputs.
// ============================================================================
#include <armadillo>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <string>

#include "AI4BayesCode/composite_block.hpp"

#define main gp_classification_demo_main
#include "../examples/GPClassification.cpp"
#undef main

// An explicit instantiation may name a private member: this gives the test
// read access to the wrapper's composite_block.
template <typename Tag, typename Tag::type Member>
struct expose_member {
    friend typename Tag::type member_of(Tag) { return Member; }
};
struct impl_tag {
    using type = std::unique_ptr<AI4BayesCode::composite_block> GPClassification::*;
    friend type member_of(impl_tag);
};
template struct expose_member<impl_tag, &GPClassification::impl_>;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

bool same_bits(const arma::vec& a, const arma::vec& b) {
    return a.n_elem == b.n_elem &&
           std::memcmp(a.memptr(), b.memptr(), a.n_elem * sizeof(double)) == 0;
}

bool same_state(const GPClassification& a, const GPClassification& b) {
    const auto ca = a.get_current(), cb = b.get_current();
    return same_bits(ca.at("f"), cb.at("f")) && same_bits(ca.at("amplitude"), cb.at("amplitude")) &&
           same_bits(ca.at("lengthscale"), cb.at("lengthscale"));
}

}  // namespace

int main() {
    // Unbuffered, so the lines printed before a crash of a moved model reach
    // the log.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("=== test_fix_examples_GPClassification ===\n");
    const std::size_t N = 40;
    arma::mat X(N, 1);
    arma::vec y(N);
    for (std::size_t i = 0; i < N; ++i) {
        X(i, 0) = -2.0 + 4.0 * static_cast<double>(i) / (N - 1.0);
        y[i] = (X(i, 0) + 0.3 * std::sin(7.0 * X(i, 0)) > 0.2) ? 1.0 : 0.0;
    }

    GPClassification twin(X, y, 7, false);
    GPClassification src(X, y, 7, false);
    GPClassification moved(std::move(src));
    twin.step(2);
    moved.step(2);
    check(same_state(twin, moved), "C1 a move-constructed model gives the draws of its twin");

    GPClassification twin2(X, y, 11, false);
    GPClassification src2(X, y, 11, false);
    GPClassification dst(X, y, 3, false);
    dst = std::move(src2);
    twin2.step(2);
    dst.step(2);
    check(same_state(twin2, dst), "C2 a move-assigned model gives the draws of its twin");

    // C3 and C4 use the first 20 inputs.
    const arma::mat X20 = X.rows(0, 19);
    const arma::vec y20 = y.subvec(0, 19);
    for (const double amp : {1e200, std::numeric_limits<double>::infinity()}) {
        GPClassification m(X20, y20, 5, false);
        m.step(1);
        const AI4BayesCode::composite_block& c = *(m.*member_of(impl_tag{}));
        const auto values0 = c.data().values();
        const arma::vec block0 = c.child(0).current();
        AI4BayesCode::state_map s;
        s["amplitude"] = arma::vec{amp};
        bool threw = false;
        try { m.set_current(s); } catch (const std::exception&) { threw = true; }
        bool ok = false;
        if (threw) {
            ok = c.data().values().size() == values0.size() &&
                 same_bits(block0, c.child(0).current());
            for (const auto& kv : values0)
                ok = ok && c.data().has(kv.first) &&
                     same_bits(kv.second, c.data().get(kv.first));
        } else {
            const std::size_t n = X20.n_rows;
            const arma::vec& K_flat = c.data().get("K_matrix");
            arma::mat M = arma::reshape(K_flat, n, n);
            M.diag() += amp * amp * 1e-8;
            arma::mat L;
            if (M.is_finite() && arma::chol(L, M, "lower")) {
                const arma::vec f_ref = L * c.data().get("z");
                const arma::vec& f = c.data().get("f");
                ok = f.n_elem == n;
                for (std::size_t i = 0; ok && i < n; ++i)
                    ok = std::abs(f[i] - f_ref[i]) <= 1e-10 * std::max(1.0, std::abs(f_ref[i]));
            }
        }
        std::printf("C3 set_current(amplitude = %g): threw = %d, %s\n", amp, threw ? 1 : 0,
                    threw ? (ok ? "state unchanged" : "state changed")
                          : (ok ? "f = L z at the loaded values" : "f not at the loaded values"));
        check(ok, std::string("C3 set_current(amplitude = ") + (std::isinf(amp) ? "Inf" : "1e200") +
                  "): unchanged after the error, or f at the loaded values");
    }

    {
        GPClassification me(X20, y20, 17, false);
        me.step(1);
        bool ok = true;
        for (const char* key : {"amplitude", "lengthscale"}) {
            const auto before = me.get_current();
            AI4BayesCode::state_map se;
            se[key] = arma::vec();
            bool threw = false;
            try { me.set_current(se); } catch (const std::exception&) { threw = true; }
            const auto after = me.get_current();
            ok = ok && threw && after.size() == before.size();
            for (const auto& kv : before)
                ok = ok && after.count(kv.first) && same_bits(kv.second, after.at(kv.first));
        }
        check(ok, "C4 set_current with an empty amplitude or lengthscale throws and leaves get_current() unchanged");
    }

    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "ALL PASSED", g_fail,
                g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
