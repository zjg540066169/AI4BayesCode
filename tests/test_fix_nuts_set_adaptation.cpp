/*
 * test_fix_nuts_set_adaptation.cpp
 *
 * Checkpoint and restore through get_adaptation() / set_adaptation(), and
 * the warmup_done field of adaptation_info (true in a snapshot of a block
 * whose first step() has completed; set_adaptation() clears first_call_ only
 * when it is true and never sets first_call_ back to true).
 *
 * 1. Restore after n_pre sweeps. Chain A runs n_pre sweeps (set_context,
 *    step), then the test records get_adaptation(), current() and the RNG
 *    state, and A runs n_post more sweeps. Chain B is a fresh block with the
 *    same config: set_current(theta), set_adaptation(ad), then n_post sweeps
 *    with a copy of the recorded RNG. B's draws must equal A's bit for bit,
 *    B's first sweep must make as many target evaluations as A's, and the
 *    restored step size must stay in place. If set_adaptation leaves
 *    first_call_ = true here, B's first step() runs the first-call warmup
 *    (200, 500 or 2500 transitions), replaces the restored step size and
 *    moves theta away from the checkpoint.
 *
 * 2. A snapshot taken before any step(), or after readapt() and before the
 *    first step(), has warmup_done == false: the restored block keeps its
 *    first-call state and gives the draws of the source block, first-call
 *    warmup and (joint block with a metric) first-call metric pilot included.
 *
 * 3. A hand-built adaptation_info has warmup_done == false, so a fresh block
 *    given one keeps its first-call warmup. With only epsilon_bar set it
 *    gives the draws of a fresh block; with a complete step-size state (0.3,
 *    adapt_iter = 1) it gives the draws of a block constructed with
 *    initial_step_size = 0.3, whose constructor installs the same values.
 *
 * 4. Values set by hand in a snapshot with warmup_done == true: the next
 *    step() of a fresh block continues from them without the first-call
 *    warmup. The reference is a block that has completed one step() and is
 *    then given the same values and position by set_current() and
 *    set_adaptation() with warmup_done == false, which keeps its first_call_
 *    false.
 *
 * Blocks: nuts_block (identity metric, and a diagonal metric installed with
 * set_precond_matrix), joint_nuts_block (identity, diagonal and dense metric).
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
#include <vector>

using namespace AI4BayesCode;

namespace {

long long g_evals = 0;

// Independent Gaussian, scales from 0.1 to 10, so that a diagonal or dense
// metric differs from the identity.
double target(const arma::vec& th, const block_context&, arma::vec* grad) {
    ++g_evals;
    const std::size_t d = th.n_elem;
    double lp = 0.0;
    if (grad) grad->set_size(d);
    for (std::size_t i = 0; i < d; ++i) {
        const double s = std::pow(10.0, -1.0 + 2.0 * double(i) / double(d - 1));
        const double z = (th[i] - 0.3 * double(i)) / s;
        lp -= 0.5 * z * z;
        if (grad) (*grad)[i] = -z / s;
    }
    return lp;
}

const std::size_t kDim = 5;

// Diagonal of the inverse covariance of target(): a good diagonal metric.
arma::vec diag_metric_values() {
    arma::vec m(kDim);
    for (std::size_t i = 0; i < kDim; ++i)
        m[i] = std::pow(10.0, 2.0 - 4.0 * double(i) / double(kDim - 1));
    return m;
}

struct run_trace {
    std::vector<arma::vec> draws;
    std::vector<long long> evals;
    std::vector<double>    step;
};

template <class Blk>
void sweep(Blk& b, std::mt19937_64& rng, run_trace& tr) {
    block_context ctx;
    const long long e0 = g_evals;
    b.set_context(ctx);
    b.step(rng);
    tr.evals.push_back(g_evals - e0);
    tr.draws.push_back(b.current());
    tr.step.push_back(b.current_step_size());
}

std::size_t n_identical(const run_trace& a, const run_trace& b) {
    std::size_t n = 0;
    for (std::size_t i = 0; i < a.draws.size() && i < b.draws.size(); ++i) {
        const arma::vec& x = a.draws[i];
        const arma::vec& y = b.draws[i];
        bool same = x.n_elem == y.n_elem;
        for (arma::uword k = 0; same && k < x.n_elem; ++k) same = (x[k] == y[k]);
        if (same) ++n;
    }
    return n;
}

// Restore after n_pre sweeps: B must continue A exactly.
template <class Make>
bool check_resume(const char* label, Make make, int n_pre, int n_post) {
    auto A = make();
    std::mt19937_64 rng(20261122ULL);
    run_trace pre;
    for (int i = 0; i < n_pre; ++i) sweep(*A, rng, pre);
    const adaptation_info ad = A->get_adaptation();
    const arma::vec theta = A->current();
    const std::mt19937_64 rng_ckpt = rng;
    run_trace ta;
    for (int i = 0; i < n_post; ++i) sweep(*A, rng, ta);

    auto B = make();
    B->set_current(theta);
    B->set_adaptation(ad);
    std::mt19937_64 rng_b = rng_ckpt;
    run_trace tb;
    for (int i = 0; i < n_post; ++i) sweep(*B, rng_b, tb);

    const std::size_t same = n_identical(ta, tb);
    const bool ok = same == static_cast<std::size_t>(n_post)
                 && tb.evals[0] == ta.evals[0]
                 && tb.step[0] == ad.epsilon_bar;
    std::printf("  %-46s identical %zu/%d, first-sweep evals A=%lld B=%lld, "
                "step after restore %.6g (snapshot %.6g, metric %s)  %s\n",
                label, same, n_post, ta.evals[0], tb.evals[0], tb.step[0],
                ad.epsilon_bar, ad.metric_kind.c_str(), ok ? "PASS" : "FAIL");
    return ok;
}

// Snapshot before any step(): the restored block must still run the
// first-call warmup, exactly as the fresh source block does.
template <class Make>
bool check_fresh_snapshot(const char* label, Make make, int n_post) {
    auto S = make();
    const adaptation_info ad0 = S->get_adaptation();
    std::mt19937_64 rng_s(777);
    run_trace ts;
    for (int i = 0; i < n_post; ++i) sweep(*S, rng_s, ts);

    auto T = make();
    T->set_adaptation(ad0);
    std::mt19937_64 rng_t(777);
    run_trace tt;
    for (int i = 0; i < n_post; ++i) sweep(*T, rng_t, tt);

    const std::size_t same = n_identical(ts, tt);
    const bool ok = same == static_cast<std::size_t>(n_post)
                 && tt.evals[0] == ts.evals[0];
    std::printf("  %-46s identical %zu/%d, first-sweep evals S=%lld T=%lld  %s\n",
                label, same, n_post, ts.evals[0], tt.evals[0], ok ? "PASS" : "FAIL");
    return ok;
}

// Snapshot after readapt() and before the first step(): the restored block
// keeps its first-call state and gives the draws of the source block.
template <class Make>
bool check_readapt_snapshot(const char* label, Make make, int n_post) {
    auto S = make();
    block_context ctx;
    std::mt19937_64 rng0(5);
    S->set_context(ctx);
    S->readapt(50, false, rng0);
    const adaptation_info ad = S->get_adaptation();
    const arma::vec theta = S->current();

    auto T = make();
    T->set_current(theta);
    T->set_adaptation(ad);

    std::mt19937_64 rng_s(99), rng_t(99);
    run_trace ts, tt;
    for (int i = 0; i < n_post; ++i) sweep(*S, rng_s, ts);
    for (int i = 0; i < n_post; ++i) sweep(*T, rng_t, tt);

    const std::string ks = S->get_adaptation().metric_kind;
    const std::string kt = T->get_adaptation().metric_kind;
    const std::size_t same = n_identical(ts, tt);
    const bool ok = same == static_cast<std::size_t>(n_post)
                 && tt.evals[0] == ts.evals[0] && ks == kt;
    std::printf("  %-46s identical %zu/%d, first-sweep evals S=%lld T=%lld, "
                "metric S=%s T=%s  %s\n", label, same, n_post, ts.evals[0],
                tt.evals[0], ks.c_str(), kt.c_str(), ok ? "PASS" : "FAIL");
    return ok;
}

// Hand-built adaptation_info on a fresh block (warmup_done keeps its default
// false): the block keeps its first-call warmup and gives the draws of the
// reference block, which holds the same adaptation state without
// set_adaptation().
template <class MakeT, class MakeR>
bool check_handbuilt(const char* label, MakeT make_target, MakeR make_ref,
                     const adaptation_info& ad, int n_post) {
    auto R = make_ref();
    auto T = make_target();
    T->set_adaptation(ad);

    std::mt19937_64 rng_r(3), rng_t(3);
    run_trace tr, tt;
    for (int i = 0; i < n_post; ++i) sweep(*R, rng_r, tr);
    for (int i = 0; i < n_post; ++i) sweep(*T, rng_t, tt);

    const std::size_t same = n_identical(tr, tt);
    const bool ok = same == static_cast<std::size_t>(n_post)
                 && tt.evals[0] == tr.evals[0];
    std::printf("  %-46s identical %zu/%d, first-sweep evals ref=%lld T=%lld, "
                "step after %d sweeps %.6g  %s\n", label, same, n_post,
                tr.evals[0], tt.evals[0], n_post, tt.step.back(),
                ok ? "PASS" : "FAIL");
    return ok;
}

// Step-size state 0.3, and with diag_metric the diagonal metric of
// diag_metric_values(), written into a snapshot; warmup_done is unchanged.
void set_values(adaptation_info& ad, bool diag_metric) {
    ad.step_size   = 0.3;
    ad.epsilon_bar = 0.3;
    ad.h_val       = 0.0;
    ad.mu_val      = std::log(3.0);
    ad.adapt_iter  = 1;
    if (diag_metric) {
        const arma::vec m = diag_metric_values();
        ad.precond_mat  = arma::diagmat(m);
        ad.precond_diag = arma::conv_to<std::vector<double>>::from(m);
        ad.metric_kind  = "diagonal";
    } else {
        ad.precond_mat.reset();
        ad.precond_diag.clear();
        ad.metric_kind  = "identity";
    }
}

// Values set by hand in a snapshot with warmup_done == true (taken from a
// block after one step()): the next step() of a fresh block B continues from
// them without the first-call warmup. Reference R: a block that has completed
// one step() and is then given the same values and position by set_current()
// and set_adaptation() with a snapshot whose warmup_done == false (taken from
// a fresh block), which keeps R's first_call_ false.
template <class Make>
bool check_handset_warm(const char* label, Make make, bool diag_metric,
                        int n_post) {
    auto R = make();
    std::mt19937_64 rng0(31);
    run_trace t0;
    sweep(*R, rng0, t0);
    adaptation_info ad_warm  = R->get_adaptation();
    adaptation_info ad_fresh = make()->get_adaptation();
    set_values(ad_warm, diag_metric);
    set_values(ad_fresh, diag_metric);

    arma::vec theta(kDim);
    theta.fill(0.5);
    R->set_current(theta);
    R->set_adaptation(ad_fresh);
    auto B = make();
    B->set_current(theta);
    B->set_adaptation(ad_warm);

    std::mt19937_64 rng_r(41), rng_b(41);
    run_trace tr, tb;
    for (int i = 0; i < n_post; ++i) sweep(*R, rng_r, tr);
    for (int i = 0; i < n_post; ++i) sweep(*B, rng_b, tb);

    const std::size_t same = n_identical(tr, tb);
    const bool ok = same == static_cast<std::size_t>(n_post)
                 && tb.evals[0] == tr.evals[0]
                 && tb.step[0] == 0.3;
    std::printf("  %-46s identical %zu/%d, first-sweep evals R=%lld B=%lld, "
                "step after first sweep %.6g  %s\n", label, same, n_post,
                tr.evals[0], tb.evals[0], tb.step[0], ok ? "PASS" : "FAIL");
    return ok;
}

std::unique_ptr<nuts_block> make_nuts(bool diag_metric,
                                      double initial_step_size = 0.0) {
    nuts_block_config cfg;
    cfg.name = "theta";
    cfg.initial_unc = arma::vec(kDim, arma::fill::zeros);
    cfg.log_density_grad = &target;
    cfg.initial_step_size = initial_step_size;
    std::unique_ptr<nuts_block> b(new nuts_block(std::move(cfg)));
    if (diag_metric) b->set_precond_matrix(arma::diagmat(diag_metric_values()));
    return b;
}

std::unique_ptr<joint_nuts_block> make_joint(int metric,  // 0 id, 1 diag, 2 dense
                                             double initial_step_size = 0.0) {
    joint_nuts_block_config cfg;
    cfg.name = "J";
    cfg.sub_params.push_back({"theta", kDim});
    cfg.log_density_grad = &target;
    cfg.initial_cat = arma::vec(kDim, arma::fill::zeros);
    cfg.use_diagonal_metric = (metric == 1);
    cfg.use_dense_metric    = (metric == 2);
    cfg.initial_step_size   = initial_step_size;
    return std::unique_ptr<joint_nuts_block>(new joint_nuts_block(std::move(cfg)));
}

} // namespace

int main() {
    std::printf("=== test_fix_nuts_set_adaptation ===\n");
    const int n_pre = 300, n_post = 300, n_short = 100;
    bool ok = true;

    std::printf("Restore after %d sweeps continues the chain exactly:\n", n_pre);
    ok &= check_resume("nuts_block, identity metric",
                       [] { return make_nuts(false); }, n_pre, n_post);
    ok &= check_resume("nuts_block, set_precond_matrix(diagonal)",
                       [] { return make_nuts(true); }, n_pre, n_post);
    ok &= check_resume("joint_nuts_block, identity metric",
                       [] { return make_joint(0); }, n_pre, n_post);
    ok &= check_resume("joint_nuts_block, use_diagonal_metric",
                       [] { return make_joint(1); }, n_pre, n_post);
    ok &= check_resume("joint_nuts_block, use_dense_metric",
                       [] { return make_joint(2); }, n_pre, n_post);

    std::printf("Snapshot taken before any step() keeps the first-call warmup:\n");
    ok &= check_fresh_snapshot("nuts_block, identity metric",
                               [] { return make_nuts(false); }, n_post);
    ok &= check_fresh_snapshot("joint_nuts_block, use_diagonal_metric",
                               [] { return make_joint(1); }, n_post);

    std::printf("Snapshot taken after readapt() and before the first step() "
                "keeps the first-call warmup:\n");
    ok &= check_readapt_snapshot("nuts_block, identity metric",
                                 [] { return make_nuts(false); }, n_short);
    ok &= check_readapt_snapshot("joint_nuts_block, use_diagonal_metric",
                                 [] { return make_joint(1); }, n_short);

    std::printf("Hand-built adaptation_info on a fresh block keeps the "
                "first-call warmup:\n");
    adaptation_info partial;
    partial.epsilon_bar = 0.3;
    adaptation_info full;
    set_values(full, false);
    ok &= check_handbuilt("nuts_block, only epsilon_bar set",
                          [] { return make_nuts(false); },
                          [] { return make_nuts(false); }, partial, n_short);
    ok &= check_handbuilt("joint_nuts_block diag, only epsilon_bar set",
                          [] { return make_joint(1); },
                          [] { return make_joint(1); }, partial, n_short);
    ok &= check_handbuilt("nuts_block, full step-size state",
                          [] { return make_nuts(false); },
                          [] { return make_nuts(false, 0.3); }, full, n_short);
    ok &= check_handbuilt("joint_nuts_block diag, full step-size state",
                          [] { return make_joint(1); },
                          [] { return make_joint(1, 0.3); }, full, n_short);

    std::printf("Snapshot with warmup_done and values set by hand: the next "
                "step() runs no first-call warmup:\n");
    ok &= check_handset_warm("nuts_block, identity metric",
                             [] { return make_nuts(false); }, false, n_short);
    ok &= check_handset_warm("joint_nuts_block, use_diagonal_metric",
                             [] { return make_joint(1); }, true, n_short);

    std::printf("=== %s ===\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
