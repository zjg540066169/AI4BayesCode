/*
 * test_fix_nuts_metric_cache.cpp
 *
 * mcmc::nuts keeps the metric setup (1/diag(M) and sqrt(diag(M)) for a
 * diagonal metric, inv(M) and chol(M) for a dense one) in nuts_settings_t and
 * reuses it while precond_cache_valid is true. This test checks that the
 * reuse leaves every draw unchanged and that every path that replaces the
 * metric (set_precond_matrix, set_adaptation, the metric pilot and the
 * three-phase warmup windows, copy and move of a block) marks the cache
 * invalid.
 *
 * Each sweep of a block A (whose cache carries over from earlier sweeps and
 * earlier metrics) is compared with one sweep of a fresh reference block R of
 * the same shape. R gets A's current value, adaptation state and metric
 * through set_current and set_adaptation, and a copy of A's RNG. R's config
 * makes its first step() a plain transition (identity-metric config,
 * n_warmup_first_call = 0), and R builds the metric setup in that call. A
 * stale cache in A makes the two draws differ.
 *
 * Also: direct mcmc::nuts calls with one settings object reused across calls
 * against calls on a copy whose cache is marked invalid before every call.
 */

#include "AI4BayesCode/block_sampler.hpp"
#include "AI4BayesCode/nuts_block.hpp"
#include "AI4BayesCode/joint_nuts_block.hpp"

#include <cmath>
#include <cstdio>
#include <functional>
#include <memory>
#include <random>
#include <string>

using namespace AI4BayesCode;

namespace {

const std::size_t kDim = 6;

// Gaussian with scales 0.2 ... 1.7 and a nearest-neighbour correlation, so
// that identity, diagonal and dense metrics give different trajectories.
double target(const arma::vec& th, const block_context&, arma::vec* grad) {
    const std::size_t d = th.n_elem;
    const double rho = 0.4;
    arma::vec z(d);
    for (std::size_t i = 0; i < d; ++i) z[i] = (th[i] - 0.1 * double(i)) / (0.2 + 0.3 * double(i));
    double lp = 0.0;
    arma::vec gz(d, arma::fill::zeros);
    for (std::size_t i = 0; i < d; ++i) {
        lp -= 0.5 * z[i] * z[i];
        gz[i] -= z[i];
        if (i + 1 < d) { lp += rho * z[i] * z[i + 1]; gz[i] += rho * z[i + 1]; gz[i + 1] += rho * z[i]; }
    }
    if (grad) {
        grad->set_size(d);
        for (std::size_t i = 0; i < d; ++i) (*grad)[i] = gz[i] / (0.2 + 0.3 * double(i));
    }
    return lp;
}

bool same_vec(const arma::vec& a, const arma::vec& b) {
    if (a.n_elem != b.n_elem) return false;
    for (arma::uword i = 0; i < a.n_elem; ++i) if (a[i] != b[i]) return false;
    return true;
}

arma::mat diag_metric(double lo, double hi) {
    return arma::diagmat(arma::linspace<arma::vec>(lo, hi, kDim));
}
arma::mat dense_metric(double scale, double off) {
    arma::mat M(kDim, kDim, arma::fill::eye);
    for (std::size_t i = 0; i + 1 < kDim; ++i) { M(i, i + 1) = off; M(i + 1, i) = off; }
    return scale * M;
}

struct tally { std::size_t same = 0, total = 0; };

// n sweeps of A, each compared with a fresh reference block.
template <class Blk, class MakeRef>
void sweeps_vs_reference(Blk& A, MakeRef make_ref, std::mt19937_64& rng, int n, tally& t) {
    const block_context ctx;
    for (int i = 0; i < n; ++i) {
        auto R = make_ref();
        R->set_current(A.current());
        R->set_adaptation(A.get_adaptation());
        std::mt19937_64 rr = rng;
        A.set_context(ctx);  A.step(rng);
        R->set_context(ctx); R->step(rr);
        t.same += same_vec(A.current(), R->current());
        ++t.total;
    }
}

std::unique_ptr<nuts_block> make_nuts(std::size_t n_warmup_first_call) {
    nuts_block_config cfg;
    cfg.name = "theta";
    cfg.initial_unc = arma::vec(kDim, arma::fill::zeros);
    cfg.log_density_grad = &target;
    cfg.n_warmup_first_call = n_warmup_first_call;
    return std::unique_ptr<nuts_block>(new nuts_block(std::move(cfg)));
}

std::unique_ptr<joint_nuts_block> make_joint(int metric, std::size_t n_warmup_first_call) {
    joint_nuts_block_config cfg;   // metric: 0 identity, 1 diagonal, 2 dense, 3 dense + 3-phase
    cfg.name = "J";
    cfg.sub_params.push_back({"theta", kDim});
    cfg.log_density_grad = &target;
    cfg.initial_cat = arma::vec(kDim, arma::fill::zeros);
    cfg.use_diagonal_metric = (metric == 1);
    cfg.use_dense_metric = (metric >= 2);
    cfg.use_three_phase_warmup = (metric == 3);
    cfg.tp_phase1_iters = 30;
    cfg.tp_phase2_windows = {20, 40, 60};
    cfg.tp_phase3_iters = 20;
    cfg.n_warmup_first_call = n_warmup_first_call;
    cfg.dense_metric_pilot_iters = 100;
    cfg.dense_metric_adapt_iters = 150;
    return std::unique_ptr<joint_nuts_block>(new joint_nuts_block(std::move(cfg)));
}

bool check_nuts_block() {
    auto A = make_nuts(200);
    auto ref = [] { return make_nuts(0); };
    std::mt19937_64 rng(31);
    const block_context ctx;
    for (int i = 0; i < 50; ++i) { A->set_context(ctx); A->step(rng); }   // warmup
    tally t;
    sweeps_vs_reference(*A, ref, rng, 60, t);                            // identity
    const arma::mat steps[] = {diag_metric(0.5, 4.0), diag_metric(2.0, 0.3),
                               dense_metric(1.5, 0.3), diag_metric(0.5, 4.0),
                               arma::mat(), dense_metric(0.8, -0.2)};
    for (const arma::mat& M : steps) {
        A->set_precond_matrix(M);
        sweeps_vs_reference(*A, ref, rng, 60, t);
    }
    // set_adaptation with another metric, then copy and move with a filled cache.
    adaptation_info ad = A->get_adaptation();
    ad.precond_mat = diag_metric(3.0, 0.4);
    A->set_adaptation(ad);
    sweeps_vs_reference(*A, ref, rng, 60, t);
    nuts_block C(*A);
    sweeps_vs_reference(C, ref, rng, 60, t);
    nuts_block D(std::move(C));
    sweeps_vs_reference(D, ref, rng, 60, t);
    const bool ok = (t.same == t.total);
    std::printf("  nuts_block: identical %zu/%zu  %s\n", t.same, t.total, ok ? "PASS" : "FAIL");
    return ok;
}

bool check_joint(int metric) {
    auto A = make_joint(metric, 200);
    auto ref = [] { return make_joint(0, 0); };
    std::mt19937_64 rng(47);
    const block_context ctx;
    for (int i = 0; i < 50; ++i) { A->set_context(ctx); A->step(rng); }   // pilot + warmup
    tally t;
    sweeps_vs_reference(*A, ref, rng, 60, t);                            // learned metric
    const adaptation_info learned = A->get_adaptation();
    adaptation_info other = learned;
    other.precond_mat = (metric >= 2) ? diag_metric(0.7, 2.5) : dense_metric(1.2, 0.25);
    other.metric_kind = (metric >= 2) ? "diagonal" : "dense";
    A->set_adaptation(other);
    sweeps_vs_reference(*A, ref, rng, 60, t);
    A->set_adaptation(learned);
    sweeps_vs_reference(*A, ref, rng, 60, t);
    A->set_precond_matrix(diag_metric(3.0, 0.4));
    sweeps_vs_reference(*A, ref, rng, 60, t);
    A->set_precond_matrix(dense_metric(0.9, -0.15));
    sweeps_vs_reference(*A, ref, rng, 60, t);
    joint_nuts_block C(*A);
    sweeps_vs_reference(C, ref, rng, 60, t);
    joint_nuts_block D(std::move(C));
    sweeps_vs_reference(D, ref, rng, 60, t);
    const bool ok = (t.same == t.total);
    const char* kind = metric == 1 ? "diagonal" : metric == 2 ? "dense" : "dense, 3-phase warmup";
    std::printf("  joint_nuts_block (%s metric): identical %zu/%zu  %s\n",
                kind, t.same, t.total, ok ? "PASS" : "FAIL");
    return ok;
}

bool check_direct() {
    mcmc::algo_settings_t s;
    s.nuts_settings.use_persistent_adapt = true;
    auto kernel = [](const mcmc::ColVec_t& x, mcmc::ColVec_t* g, void*) -> mcmc::fp_t {
        return target(x, block_context(), g);
    };
    mcmc::algo_settings_t r = s;                  // reference: rebuilt every call
    arma::vec xa(kDim, arma::fill::zeros), xr = xa;
    std::mt19937_64 seeds(5);
    tally t;
    auto call = [&](std::size_t burn) {
        const std::size_t seed = static_cast<std::size_t>(seeds());
        for (mcmc::algo_settings_t* p : {&s, &r}) {
            p->rng_seed_value = seed;
            p->nuts_settings.n_burnin_draws = burn;
            p->nuts_settings.n_keep_draws = 1;
            p->nuts_settings.n_adapt_draws = burn;
        }
        r.nuts_settings.precond_cache_valid = false;
        mcmc::Mat_t da, dr;
        mcmc::nuts(xa, kernel, da, nullptr, s);
        mcmc::nuts(xr, kernel, dr, nullptr, r);
        xa = da.row(0).t();
        xr = dr.row(0).t();
        t.same += same_vec(xa, xr); ++t.total;
    };
    auto install = [&](const arma::mat& M) {
        s.nuts_settings.precond_mat = M; s.nuts_settings.precond_cache_valid = false;
        r.nuts_settings.precond_mat = M;
    };
    call(100);
    for (int i = 0; i < 40; ++i) call(0);
    const arma::mat steps[] = {diag_metric(0.5, 4.0), diag_metric(2.0, 0.3), dense_metric(1.5, 0.3),
                               dense_metric(0.9, -0.1), arma::mat(), diag_metric(0.5, 4.0)};
    for (const arma::mat& M : steps) {
        install(M);
        for (int i = 0; i < 40; ++i) call(0);
    }
    const bool ok = (t.same == t.total);
    std::printf("  mcmc::nuts direct: identical %zu/%zu  %s\n", t.same, t.total, ok ? "PASS" : "FAIL");
    return ok;
}

} // namespace

int main() {
    std::printf("=== test_fix_nuts_metric_cache ===\n");
    bool ok = true;
    ok &= check_nuts_block();
    ok &= check_joint(1);
    ok &= check_joint(2);
    ok &= check_joint(3);
    ok &= check_direct();
    std::printf("=== %s ===\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
