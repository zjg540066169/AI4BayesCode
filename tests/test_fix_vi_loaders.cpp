/*
 * test_fix_vi_loaders.cpp
 *
 * A load (set_current / set_variational_state, directly or through
 * composite_block::set_current) in the middle of an optimizer epoch must not
 * be undone by that epoch's Polyak-Ruppert average, and the SKL reference
 * must be measured from the loaded point. After a load that changes the
 * variational parameters, every VI block has to continue exactly as a block
 * constructed at the loaded parameters (same step size and epoch count).
 *
 *  A. mean-field Gaussian, exact optimum of a Gaussian target loaded at
 *     step 150 of epoch 0 (set_variational_state, set_current, composite):
 *     the run ends at the target and matches a block built at the load.
 *  B. full-rank Gaussian, same with a correlated Gaussian target.
 *  C. mean-field categorical: after 150 sweeps toward one local optimum, a
 *     composite set_current of phi near the best optimum is kept, and the run
 *     matches a block that received the same load before its first sweep.
 *  D. structured categorical, the same with set_variational_state(eta) and
 *     set_current(phi).
 *  E. re-supplying the current value (composite set_current(current())) is
 *     not a restart: the epoch schedule of an uninterrupted run is kept.
 */

#include "AI4BayesCode/block_sampler.hpp"
#include "AI4BayesCode/shared_data.hpp"
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/mean_field_gaussian_vi_block.hpp"
#include "AI4BayesCode/full_rank_gaussian_vi_block.hpp"
#include "AI4BayesCode/mean_field_categorical_vi_block.hpp"
#include "AI4BayesCode/structured_categorical_vi_block.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

using namespace AI4BayesCode;

namespace {

int checks = 0, failures = 0;

void check(bool ok, const char* what) {
    ++checks;
    if (!ok) { ++failures; std::printf("  FAIL  %s\n", what); }
    else       std::printf("  ok    %s\n", what);
}

bool same_vec(const arma::vec& a, const arma::vec& b) {
    return a.n_elem == b.n_elem &&
           (a.n_elem == 0 ||
            std::memcmp(a.memptr(), b.memptr(), a.n_elem * sizeof(double)) == 0);
}

bool same_dbl(double a, double b) { return std::memcmp(&a, &b, sizeof(double)) == 0; }

// Rows [from, from + n) of a equal rows [0, n) of b bit for bit, b has n rows,
// and both carry the same final_khat.
bool same_tail(const vi_history_t& a, std::size_t from, const vi_history_t& b) {
    const std::size_t n = b.elbo.size();
    if (n == 0 || a.elbo.size() != from + n) return false;
    for (std::size_t t = 0; t < n; ++t) {
        if (!same_dbl(a.elbo[from + t], b.elbo[t]) ||
            !same_vec(a.mu[from + t], b.mu[t]) ||
            !same_vec(a.log_sd[from + t], b.log_sd[t]) ||
            !same_dbl(a.gamma[from + t], b.gamma[t]) ||
            a.epoch[from + t] != b.epoch[t]) return false;
    }
    return same_dbl(a.final_khat, b.final_khat);
}

// ---------------------------------------------------------------------------
// Gaussian targets with a known optimum q (the target itself).
// ---------------------------------------------------------------------------
const arma::vec m5 = {20.0, -15.0, 10.0, 25.0, -5.0};
const arma::vec s5 = {0.5, 1.0, 2.0, 0.3, 0.7};

double lp_mf(const arma::vec& e, const block_context&, arma::vec* g) {
    const arma::vec z = (e - m5) / s5;
    if (g) *g = -z / s5;
    return -0.5 * arma::dot(z, z);
}

mean_field_gaussian_vi_block_config mf_cfg(const arma::vec& mu0,
                                           const arma::vec& lsd0) {
    mean_field_gaussian_vi_block_config c;
    c.name = "theta";
    c.initial_unc = mu0;
    c.initial_log_sd = lsd0;
    c.log_density_grad = lp_mf;
    return c;
}

double mf_err_sd(const arma::vec& mu) { return arma::max(arma::abs(mu - m5) / s5); }

const arma::vec m2 = {20.0, -15.0};
const arma::mat S2 = {{0.25, 0.135}, {0.135, 0.09}};   // sd 0.5 and 0.3, corr 0.9

double lp_fr(const arma::vec& e, const block_context&, arma::vec* g) {
    const arma::vec d = arma::solve(S2, e - m2);
    if (g) *g = -d;
    return -0.5 * arma::dot(e - m2, d);
}

full_rank_gaussian_vi_block_config fr_cfg(const arma::vec& mu0, const arma::mat& L0) {
    full_rank_gaussian_vi_block_config c;
    c.name = "theta";
    c.initial_unc = mu0;
    c.initial_L = L0;
    c.log_density_grad = lp_fr;
    return c;
}

// ---------------------------------------------------------------------------
// A. mean-field Gaussian
// ---------------------------------------------------------------------------
void test_mean_field_gaussian() {
    std::printf("A. mean_field_gaussian_vi_block: load at step 150 of epoch 0\n");
    const std::size_t pre = 150, post = 3000;
    const arma::vec lsd_opt = arma::log(s5);

    // A1. set_variational_state with the exact optimum.
    {
        mean_field_gaussian_vi_block blk(mf_cfg(arma::zeros(5), arma::vec()));
        blk.set_keep_history(true);
        std::mt19937_64 r1(101);
        for (std::size_t t = 0; t < pre; ++t) blk.step(r1);
        blk.set_variational_state(m5, lsd_opt);
        std::mt19937_64 r2(202);
        for (std::size_t t = 0; t < post; ++t) blk.step(r2);

        mean_field_gaussian_vi_block ref(mf_cfg(m5, lsd_opt));
        ref.set_keep_history(true);
        std::mt19937_64 r3(202);
        for (std::size_t t = 0; t < post; ++t) ref.step(r3);

        std::printf("    final error %.3f posterior sd, converged=%d\n",
                    mf_err_sd(blk.current()), (int)blk.converged());
        check(blk.converged() && mf_err_sd(blk.current()) < 0.5 &&
              arma::max(arma::abs(blk.get_log_sd() - lsd_opt)) < 0.5,
              "set_variational_state(optimum): the run ends at the target q");
        check(same_tail(blk.vi_history(), pre, ref.vi_history()),
              "set_variational_state: continues bit for bit as a block built at the load");
    }
    // A2. set_current (mean only) with the target mean.
    {
        mean_field_gaussian_vi_block blk(mf_cfg(arma::zeros(5), arma::vec()));
        blk.set_keep_history(true);
        std::mt19937_64 r1(103);
        for (std::size_t t = 0; t < pre; ++t) blk.step(r1);
        const arma::vec lsd_at_load = blk.get_log_sd();
        blk.set_current(m5);
        std::mt19937_64 r2(204);
        for (std::size_t t = 0; t < post; ++t) blk.step(r2);

        mean_field_gaussian_vi_block ref(mf_cfg(m5, lsd_at_load));
        ref.set_keep_history(true);
        std::mt19937_64 r3(204);
        for (std::size_t t = 0; t < post; ++t) ref.step(r3);

        std::printf("    final error %.3f posterior sd, converged=%d\n",
                    mf_err_sd(blk.current()), (int)blk.converged());
        check(blk.converged() && mf_err_sd(blk.current()) < 0.5,
              "set_current(target mean): the run ends at the target mean");
        check(same_tail(blk.vi_history(), pre, ref.vi_history()),
              "set_current: continues bit for bit as a block built at the load");
    }
    // A3. composite_block::set_current, the restore path.
    {
        auto run = [&](const arma::vec& mu0, const arma::vec& lsd0, std::size_t n_pre,
                       const arma::vec* load, mean_field_gaussian_vi_block** out,
                       std::unique_ptr<composite_block>& comp) {
            comp = std::make_unique<composite_block>("m");
            comp->data().set("theta", arma::zeros(5));
            auto b = std::make_unique<mean_field_gaussian_vi_block>(mf_cfg(mu0, lsd0));
            *out = b.get();
            comp->add_child(std::move(b));
            comp->set_keep_history(true);
            std::mt19937_64 r1(105);
            for (std::size_t t = 0; t < n_pre; ++t) comp->step(r1);
            if (load) comp->set_current(*load);
            std::mt19937_64 r2(206);
            for (std::size_t t = 0; t < post; ++t) comp->step(r2);
        };
        std::unique_ptr<composite_block> c1, c2;
        mean_field_gaussian_vi_block *b1 = nullptr, *b2 = nullptr;
        run(arma::zeros(5), arma::vec(), pre, &m5, &b1, c1);
        // The reference starts at the loaded mean and the log sd held at the load.
        const arma::vec lsd_at_load = b1->vi_history().log_sd[pre - 1];
        run(m5, lsd_at_load, 0, nullptr, &b2, c2);
        std::printf("    final error %.3f posterior sd, converged=%d\n",
                    mf_err_sd(b1->current()), (int)b1->converged());
        check(b1->converged() && mf_err_sd(b1->current()) < 0.5,
              "composite set_current(target mean): the run ends at the target mean");
        check(same_tail(b1->vi_history(), pre, b2->vi_history()),
              "composite set_current: continues bit for bit as a block built at the load");
    }
}

// ---------------------------------------------------------------------------
// B. full-rank Gaussian
// ---------------------------------------------------------------------------
void test_full_rank_gaussian() {
    std::printf("\nB. full_rank_gaussian_vi_block: load at step 150 of epoch 0\n");
    const std::size_t pre = 150, post = 3000;
    const arma::mat Lopt = arma::chol(S2, "lower");
    const arma::vec packed = {std::log(Lopt(0, 0)), std::log(Lopt(1, 1)), Lopt(1, 0)};

    full_rank_gaussian_vi_block blk(fr_cfg(arma::zeros(2), arma::mat()));
    blk.set_keep_history(true);
    std::mt19937_64 r1(301);
    for (std::size_t t = 0; t < pre; ++t) blk.step(r1);
    blk.set_variational_state(m2, packed);
    const arma::mat L_at_load = blk.L();
    std::mt19937_64 r2(302);
    for (std::size_t t = 0; t < post; ++t) blk.step(r2);

    full_rank_gaussian_vi_block ref(fr_cfg(m2, L_at_load));
    ref.set_keep_history(true);
    std::mt19937_64 r3(302);
    for (std::size_t t = 0; t < post; ++t) ref.step(r3);

    const arma::vec sd2 = arma::sqrt(S2.diag());
    const double err = arma::max(arma::abs(blk.current() - m2) / sd2);
    const arma::mat Sq = blk.L() * blk.L().t();
    const double cov_err = arma::norm(Sq - S2, "fro") / arma::norm(S2, "fro");
    std::printf("    final mean error %.3f sd, relative covariance error %.3f, converged=%d\n",
                err, cov_err, (int)blk.converged());
    check(blk.converged() && err < 0.5 && cov_err < 0.3,
          "set_variational_state(optimum): the run ends at the target q");
    check(same_tail(blk.vi_history(), pre, ref.vi_history()),
          "set_variational_state: continues bit for bit as a block built at the load");
}

// ---------------------------------------------------------------------------
// Categorical targets: Potts chain, n = 4, K = 3, coupling 2, field 0.2 on
// state 1. The mean-field ELBO has one local optimum per state; the state-1
// optimum is the best one.
// ---------------------------------------------------------------------------
double potts_lp(const arma::uvec& z, const block_context&) {
    double v = 0.0;
    for (std::size_t i = 0; i + 1 < z.n_elem; ++i) if (z[i] == z[i + 1]) v += 2.0;
    for (std::size_t i = 0; i < z.n_elem; ++i) if (z[i] == 1u) v += 0.2;
    return v;
}

// Per-node phi with mass 0.9 on state k.
arma::vec phi_mode(std::size_t n, std::size_t k) {
    arma::vec p(3 * n, arma::fill::value(0.05));
    for (std::size_t i = 0; i < n; ++i) p[3 * i + k] = 0.9;
    return p;
}

// Per-clique phi (cliques {0,1}, {2,3}) with mass 0.6 on the joint state
// where both nodes are in state k (joint index k + 3k).
arma::vec clique_phi_mode(std::size_t k) {
    arma::vec p(18, arma::fill::value(0.05));
    p[4 * k] = 0.6; p[9 + 4 * k] = 0.6;
    return p;
}

// The anchored-softmax eta of clique_phi_mode(k).
arma::vec clique_eta_mode(std::size_t k) {
    arma::vec e(16, arma::fill::zeros);
    if (k > 0) { e[4 * k - 1] = std::log(12.0); e[8 + 4 * k - 1] = std::log(12.0); }
    return e;
}

mean_field_categorical_vi_block_config mfc_cfg(double eps, std::uint64_t seed) {
    mean_field_categorical_vi_block_config c;
    c.name = "z";
    c.cardinalities = arma::uvec(4); c.cardinalities.fill(3);
    c.log_density = potts_lp;
    c.exact_enumeration = true;
    c.optimizer.tau = 0.01;
    c.optimizer.max_epochs = 25;
    c.init_random_eps = eps;
    c.init_rng_seed = seed;
    return c;
}

std::unique_ptr<composite_block> mfc_model(double eps, std::uint64_t seed,
                                           mean_field_categorical_vi_block** out) {
    auto comp = std::make_unique<composite_block>("m");
    comp->data().set("z", arma::zeros(4));
    auto b = std::make_unique<mean_field_categorical_vi_block>(mfc_cfg(eps, seed));
    *out = b.get();
    comp->add_child(std::move(b));
    comp->set_keep_history(true);
    return comp;
}

// ---------------------------------------------------------------------------
// C. mean-field categorical through composite_block::set_current
// ---------------------------------------------------------------------------
void test_mean_field_categorical() {
    std::printf("\nC. mean_field_categorical_vi_block: composite set_current at sweep 150\n");
    const std::size_t pre = 150, post = 3000;

    // Run toward the state-2 optimum, then load the state-1 optimum region.
    mean_field_categorical_vi_block *b1 = nullptr, *b2 = nullptr;
    auto c1 = mfc_model(0.0, 0, &b1);
    c1->set_current(phi_mode(4, 2));
    std::mt19937_64 r1(401);
    for (std::size_t t = 0; t < pre; ++t) c1->step(r1);
    c1->set_current(phi_mode(4, 1));
    std::mt19937_64 r2(402);
    for (std::size_t t = 0; t < post; ++t) c1->step(r2);

    auto c2 = mfc_model(0.0, 0, &b2);    // fresh block, same load before sweep 1
    c2->set_current(phi_mode(4, 1));
    std::mt19937_64 r3(402);
    for (std::size_t t = 0; t < post; ++t) c2->step(r3);

    const arma::vec& p = b1->current();
    bool state1 = true;
    for (std::size_t i = 0; i < 4; ++i)
        if (!(p[3 * i + 1] > p[3 * i] && p[3 * i + 1] > p[3 * i + 2])) state1 = false;
    std::printf("    final ELBO: loaded at 150 %.6f, loaded before sweep 1 %.6f; node 0 phi (%.3f %.3f %.3f)\n",
                b1->current_elbo(), b2->current_elbo(), p[0], p[1], p[2]);
    check(b1->converged() && state1,
          "the loaded state-1 optimum is kept (final q favours state 1 at every node)");
    check(same_tail(b1->vi_history(), pre, b2->vi_history()),
          "composite set_current(phi): continues bit for bit as a block loaded before sweep 1");
}

// ---------------------------------------------------------------------------
// D. structured categorical, direct loaders
// ---------------------------------------------------------------------------
structured_categorical_vi_block_config sc_cfg(double eps, std::uint64_t seed) {
    structured_categorical_vi_block_config c;
    c.name = "z";
    c.cardinalities = arma::uvec(4); c.cardinalities.fill(3);
    c.clique_partition = {{0, 1}, {2, 3}};
    c.log_density = potts_lp;
    c.exact_enumeration = true;
    c.optimizer.tau = 0.01;
    c.optimizer.max_epochs = 25;
    c.init_random_eps = eps;
    c.init_rng_seed = seed;
    return c;
}

void test_structured_categorical() {
    std::printf("\nD. structured_categorical_vi_block: load at step 150\n");
    const std::size_t pre = 150, post = 3000;

    for (int route = 0; route < 2; ++route) {
        // Run toward the state-2 optimum, then load the state-1 optimum region.
        auto load = [route](structured_categorical_vi_block& b, std::size_t k) {
            if (route == 0) b.set_variational_state(arma::vec(), clique_eta_mode(k));
            else            b.set_current(clique_phi_mode(k));
        };
        structured_categorical_vi_block blk(sc_cfg(0.0, 0));
        blk.set_keep_history(true);
        load(blk, 2);
        std::mt19937_64 r1(501);
        for (std::size_t t = 0; t < pre; ++t) blk.step(r1);
        load(blk, 1);
        std::mt19937_64 r2(502);
        for (std::size_t t = 0; t < post; ++t) blk.step(r2);

        structured_categorical_vi_block ref(sc_cfg(0.0, 0));
        ref.set_keep_history(true);
        load(ref, 1);
        std::mt19937_64 r3(502);
        for (std::size_t t = 0; t < post; ++t) ref.step(r3);

        const arma::mat marg = blk.per_node_marginals();
        bool state1 = true;
        for (std::size_t i = 0; i < 4; ++i)
            if (marg.row(i).index_max() != 1u) state1 = false;
        std::printf("    route %s: final ELBO %.6f, node 0 marginal (%.3f %.3f %.3f)\n",
                    route == 0 ? "eta" : "phi", blk.current_elbo(), marg(0, 0), marg(0, 1), marg(0, 2));
        check(blk.converged() && state1,
              route == 0 ? "set_variational_state(eta): the loaded state-1 optimum is kept"
                         : "set_current(phi): the loaded state-1 optimum is kept");
        check(same_tail(blk.vi_history(), pre, ref.vi_history()),
              route == 0 ? "set_variational_state(eta): continues bit for bit as a block loaded before step 1"
                         : "set_current(phi): continues bit for bit as a block loaded before step 1");
    }
}

// ---------------------------------------------------------------------------
// E. re-supplying the current value keeps the epoch schedule
// ---------------------------------------------------------------------------
void test_resupply_is_not_a_restart() {
    std::printf("\nE. composite set_current(current()) is not a restart\n");
    {   // mean-field Gaussian: identical trajectory to the uninterrupted run
        auto make = [](mean_field_gaussian_vi_block** out) {
            auto comp = std::make_unique<composite_block>("m");
            comp->data().set("theta", arma::zeros(5));
            auto b = std::make_unique<mean_field_gaussian_vi_block>(mf_cfg(arma::zeros(5), arma::vec()));
            *out = b.get();
            comp->add_child(std::move(b));
            comp->set_keep_history(true);
            return comp;
        };
        mean_field_gaussian_vi_block *b1 = nullptr, *b2 = nullptr;
        auto c1 = make(&b1), c2 = make(&b2);
        std::mt19937_64 r1(601), r2(601);
        for (int t = 0; t < 150; ++t) { c1->step(r1); c2->step(r2); }
        const arma::vec cur = c1->current();
        c1->set_current(cur);
        for (int t = 0; t < 1000; ++t) { c1->step(r1); c2->step(r2); }
        check(same_tail(b1->vi_history(), 0, b2->vi_history()),
              "mean-field Gaussian: trajectory identical to the run without the restore");
    }
    {   // mean-field categorical: the epoch still closes at sweep 200
        mean_field_categorical_vi_block* b = nullptr;
        auto c = mfc_model(0.1, 7, &b);
        std::mt19937_64 r(602);
        for (int t = 0; t < 150; ++t) c->step(r);
        const arma::vec cur = c->current();
        c->set_current(cur);
        for (int t = 0; t < 100; ++t) c->step(r);
        check(b->epoch() == 1, "mean-field categorical: the epoch closes at sweep 200 as without the restore");
    }
}

} // namespace

int main() {
    std::printf("=== test_fix_vi_loaders ===\n\n");
    test_mean_field_gaussian();
    test_full_rank_gaussian();
    test_mean_field_categorical();
    test_structured_categorical();
    test_resupply_is_not_a_restart();
    std::printf("\n%d checks, %d failures\n", checks, failures);
    if (failures) { std::printf("FAILED\n"); return 1; }
    std::printf("PASSED\n");
    return 0;
}
