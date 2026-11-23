/*
 * test_fix_nuts_set_adaptation.cpp
 *
 * Checkpoint and restore through get_adaptation() / set_adaptation(), and
 * the warmup_done field of adaptation_info (true when the block's next step()
 * does not run the first-call warmup; set_adaptation() clears first_call_ only
 * when it is true and never sets first_call_ back to true).
 *
 * 1. Restore after n_pre sweeps. Chain A runs n_pre sweeps (set_context,
 *    step), then the test records get_adaptation(), current() and the RNG
 *    state, and A runs n_post more sweeps. Chain B is a fresh block with the
 *    same config: set_current(theta), set_adaptation(ad), then n_post sweeps
 *    with a copy of the recorded RNG. B's draws must equal A's bit for bit,
 *    B's first sweep must make as many target evaluations as A's, and the
 *    step size after B's first sweep must equal the snapshot's epsilon_bar.
 *    If set_adaptation leaves first_call_ = true here, B's first step() runs
 *    the first-call warmup (200, 500 or 2500 transitions), replaces the
 *    restored step size and moves theta away from the checkpoint.
 *
 * 2. A snapshot taken before any step(), or after readapt() and before the
 *    first step(), has warmup_done == false: the restored block runs the
 *    first-call warmup and gives the draws of the source block, first-call
 *    warmup and (joint block with a metric) first-call metric pilot included.
 *
 * 3. A hand-built adaptation_info has warmup_done == false, so a fresh block
 *    given one runs the first-call warmup. With only epsilon_bar set it
 *    gives the draws of a fresh block; with a complete step-size state (0.3,
 *    adapt_iter = 1) it gives the draws of a block constructed with
 *    initial_step_size = 0.3, whose constructor installs the same values.
 *
 * 4. Values set by hand in a snapshot with warmup_done == true: the next
 *    step() of a fresh block continues from them without the first-call
 *    warmup. The reference is a block that has completed one step() and is
 *    then given the same values and position by set_current() and
 *    set_adaptation() with warmup_done == false, so its first_call_ is
 *    unchanged (false).
 *
 * 5. A joint_nuts_block::set_adaptation() call that throws changes nothing.
 *    The snapshot is taken from a joint block that has stepped (so its
 *    warmup_done is true), its five step-size fields are set to values that
 *    differ from those of every block given it, and its precond_mat is
 *    replaced by a matrix that is not total_unc_dim_ x total_unc_dim_:
 *    (d+1) x (d+1), d x (d+1), (d+1) x d, or the d x d identity as a
 *    1 x d^2 row or a d^2 x 1 column. A fresh block given it must throw,
 *    then return the same get_adaptation() as a fresh block and give its
 *    draws and first-sweep evaluation count, so its first step() runs the
 *    first-call warmup. A block after n_pre sweeps given it must throw, then
 *    return the same get_adaptation() as a twin block (same seed, same
 *    sweeps, no call) and give the twin's draws.
 *
 * 6. A set_adaptation() call whose copy of precond_mat fails with
 *    std::bad_alloc changes nothing (nuts_block and joint_nuts_block). The
 *    test routes Armadillo's heap allocations through the two functions
 *    below and makes them fail during the call. The snapshot comes from a
 *    block that has stepped (so its warmup_done is true), with step-size
 *    values that differ from those of every block given it in all five
 *    fields and a diagonal precond_mat. A fresh block given it must throw,
 *    then return the same get_adaptation() as a fresh block and give its
 *    draws and first-sweep evaluation count, so its first step() runs the
 *    first-call warmup. A block (identity metric) after n_pre sweeps given
 *    it must throw, then return the same get_adaptation() as its twin and
 *    give the twin's draws.
 *
 * 7. nuts_block::set_adaptation(), nuts_block::set_precond_matrix() and
 *    joint_nuts_block::set_precond_matrix() with a precond_mat that is not
 *    d x d (the shapes of 5) throw std::runtime_error and change nothing,
 *    checked on a fresh block and on a block after n_pre sweeps as in 5.
 *
 * 8. The constructors of both blocks throw std::invalid_argument for a
 *    nuts_settings.nuts_settings.precond_mat that is not d x d (the shapes
 *    of 5). With a d x d diagonal precond_mat in the config a block gives the
 *    draws of a block given the same matrix by set_precond_matrix().
 *
 * Blocks: nuts_block (identity metric, and a diagonal metric installed with
 * set_precond_matrix), joint_nuts_block (identity, diagonal and dense metric).
 */

// Heap memory of every Armadillo object in this program comes from these
// two functions; while g_fail_arma_alloc is true each request returns
// nullptr, for which Armadillo throws std::bad_alloc. Defined before the
// first include of Armadillo.
#include <cstddef>
#include <cstdlib>
static bool g_fail_arma_alloc = false;
static void* test_arma_alloc(std::size_t n_bytes) {
    return g_fail_arma_alloc ? nullptr : std::malloc(n_bytes);
}
static void test_arma_free(void* p) { std::free(p); }
#define ARMA_ALIEN_MEM_ALLOC_FUNCTION test_arma_alloc
#define ARMA_ALIEN_MEM_FREE_FUNCTION  test_arma_free

#include "AI4BayesCode/block_sampler.hpp"
#include "AI4BayesCode/nuts_block.hpp"
#include "AI4BayesCode/joint_nuts_block.hpp"

#include <cmath>
#include <cstdio>
#include <exception>
#include <functional>
#include <memory>
#include <new>
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
// runs the first-call warmup and gives the draws of the source block.
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

// Hand-built adaptation_info on a fresh block (warmup_done is false, the
// default): the block runs the first-call warmup and gives the draws of the
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
// a fresh block), so R's first_call_ is unchanged (false).
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

// Field-by-field equality of two snapshots (exact for the doubles).
bool same_adaptation(const adaptation_info& a, const adaptation_info& b) {
    bool same = a.step_size == b.step_size && a.epsilon_bar == b.epsilon_bar
             && a.h_val == b.h_val && a.mu_val == b.mu_val
             && a.adapt_iter == b.adapt_iter && a.metric_kind == b.metric_kind
             && a.precond_diag == b.precond_diag
             && a.precond_mat.n_rows == b.precond_mat.n_rows
             && a.precond_mat.n_cols == b.precond_mat.n_cols;
    for (arma::uword k = 0; same && k < a.precond_mat.n_elem; ++k)
        same = (a.precond_mat[k] == b.precond_mat[k]);
    return same;
}

// Step-size values that differ from those of a fresh block and of a block
// after n_pre sweeps, so that a set_adaptation() call that installs any of
// the five fields before it throws changes the block.
void set_other_values(adaptation_info& ad) {
    ad.step_size   = 0.0625;
    ad.epsilon_bar = 0.03125;
    ad.h_val       = 0.25;
    ad.mu_val      = 1.5;
    ad.adapt_iter  = 7;
}

// true when each of the five step-size fields of a differs from that of b.
bool step_fields_differ(const adaptation_info& a, const adaptation_info& b) {
    return a.step_size != b.step_size && a.epsilon_bar != b.epsilon_bar
        && a.h_val != b.h_val && a.mu_val != b.mu_val
        && a.adapt_iter != b.adapt_iter;
}

// n sweeps; false when one of them throws.
template <class Blk>
bool sweeps(Blk& b, std::mt19937_64& rng, run_trace& tr, int n) {
    try {
        for (int i = 0; i < n; ++i) sweep(b, rng, tr);
    } catch (const std::exception&) {
        return false;
    }
    return true;
}

long long first_evals(const run_trace& tr) {
    return tr.evals.empty() ? -1 : tr.evals[0];
}

double first_step(const run_trace& tr) {
    return tr.step.empty() ? -1.0 : tr.step[0];
}

// The d x d identity as a 1 x d^2 row and as a d^2 x 1 column: d^2 elements,
// not d x d.
std::vector<arma::mat> d2_shapes() {
    const arma::mat eye_d(kDim, kDim, arma::fill::eye);
    return {arma::mat(arma::vectorise(eye_d).t()),
            arma::mat(arma::vectorise(eye_d))};
}

// (d+1) x (d+1), d x (d+1), (d+1) x d, and the two shapes of d2_shapes().
std::vector<arma::mat> all_bad_shapes() {
    std::vector<arma::mat> s = {
        arma::mat(kDim + 1, kDim + 1, arma::fill::eye),
        arma::mat(kDim, kDim + 1, arma::fill::eye),
        arma::mat(kDim + 1, kDim, arma::fill::eye)};
    for (const arma::mat& M : d2_shapes()) s.push_back(M);
    return s;
}

// apply(block, snapshot) for check_throw_unchanged.
struct call_set_adaptation {
    template <class Blk>
    void operator()(Blk& b, const adaptation_info& ad) const { b.set_adaptation(ad); }
};
struct call_set_precond_matrix {
    template <class Blk>
    void operator()(Blk& b, const adaptation_info& ad) const {
        b.set_precond_matrix(ad.precond_mat);
    }
};

// A call apply(block, bad) whose precond_mat is one of shapes must throw
// std::runtime_error and change nothing. The snapshot bad comes from a block
// S that has stepped (so its warmup_done is true), with the step-size values
// of set_other_values() and precond_mat replaced by the shape. (a) A fresh
// block T given it must return the same get_adaptation() as a fresh block R
// and give R's draws and first-sweep evaluation count. (b) A block A after
// n_pre sweeps given it must return the same get_adaptation() as its twin A2
// and give A2's draws. A block given the call without a std::runtime_error is
// not stepped, and the line reports FAIL.
template <class Make, class Apply>
bool check_throw_unchanged(const char* label, Make make, Apply apply,
                           const std::vector<arma::mat>& shapes,
                           int n_pre, int n_post) {
    auto S = make();
    std::mt19937_64 rng_s(555);
    run_trace ts;
    for (int i = 0; i < n_pre / 2; ++i) sweep(*S, rng_s, ts);
    bool ok = true;
    for (const arma::mat& M : shapes) {
        adaptation_info bad = S->get_adaptation();
        set_other_values(bad);
        bad.precond_mat = M;
        bad.precond_diag.clear();
        bad.metric_kind = "dense";
        const std::string name = std::string(label) + ", " +
            std::to_string(M.n_rows) + "x" + std::to_string(M.n_cols);

        auto T = make();
        auto R = make();
        const bool differ_t = step_fields_differ(bad, R->get_adaptation());
        bool threw_t = false;
        try { apply(*T, bad); }
        catch (const std::runtime_error&) { threw_t = true; }
        catch (const std::exception&) {}
        const bool state_t = same_adaptation(T->get_adaptation(), R->get_adaptation());
        std::mt19937_64 rng_t(808), rng_r(808);
        run_trace tt, tr;
        const bool ran_t = threw_t && sweeps(*T, rng_t, tt, n_post)
                                   && sweeps(*R, rng_r, tr, n_post);
        const std::size_t same_t = ran_t ? n_identical(tr, tt) : 0;
        const bool ok_t = threw_t && differ_t && state_t && ran_t
                       && same_t == static_cast<std::size_t>(n_post)
                       && first_evals(tt) == first_evals(tr);

        auto A = make();
        auto A2 = make();
        std::mt19937_64 rng_a(909), rng_a2(909);
        run_trace pa, pa2;
        for (int i = 0; i < n_pre; ++i) sweep(*A, rng_a, pa);
        for (int i = 0; i < n_pre; ++i) sweep(*A2, rng_a2, pa2);
        const bool differ_a = step_fields_differ(bad, A2->get_adaptation());
        bool threw_a = false;
        try { apply(*A, bad); }
        catch (const std::runtime_error&) { threw_a = true; }
        catch (const std::exception&) {}
        const bool state_a = same_adaptation(A->get_adaptation(), A2->get_adaptation());
        run_trace ta, ta2;
        const bool ran_a = threw_a && sweeps(*A, rng_a, ta, n_post)
                                   && sweeps(*A2, rng_a2, ta2, n_post);
        const std::size_t same_a = ran_a ? n_identical(ta2, ta) : 0;
        const bool ok_a = threw_a && differ_a && state_a && ran_a
                       && same_a == static_cast<std::size_t>(n_post)
                       && first_evals(ta) == first_evals(ta2);

        std::printf("  %-46s fresh: threw %d, snapshot fields differ %d, "
                    "adaptation equal to fresh block %d, identical %zu/%d, "
                    "first-sweep evals R=%lld T=%lld  %s\n", name.c_str(),
                    int(threw_t), int(differ_t), int(state_t), same_t, n_post,
                    first_evals(tr), first_evals(tt), ok_t ? "PASS" : "FAIL");
        std::printf("  %-46s stepped: threw %d, snapshot fields differ %d, "
                    "adaptation equal to twin %d, identical %zu/%d, step %.6g "
                    "(twin %.6g)  %s\n", name.c_str(), int(threw_a),
                    int(differ_a), int(state_a), same_a, n_post, first_step(ta),
                    first_step(ta2), ok_a ? "PASS" : "FAIL");
        ok = ok && ok_t && ok_a;
    }
    return ok;
}

// set_adaptation(ad) while every Armadillo heap allocation fails; true when
// the call throws std::bad_alloc.
template <class Blk>
bool set_adaptation_alloc_fails(Blk& b, const adaptation_info& ad) {
    bool threw = false;
    g_fail_arma_alloc = true;
    try { b.set_adaptation(ad); }
    catch (const std::bad_alloc&) { threw = true; }
    catch (...) {}
    g_fail_arma_alloc = false;
    return threw;
}

// set_adaptation() whose copy of precond_mat fails with std::bad_alloc must
// change nothing. The snapshot comes from a chain S that has stepped (so its
// warmup_done is true), with the step-size values of set_other_values() and
// a diagonal precond_mat. (a) A fresh block T given it must return the same
// get_adaptation() as a fresh block R and give R's draws and first-sweep
// evaluation count. (b) A and its twin A2 (identity metric, no precond_mat)
// have run n_pre sweeps; after the throw A must return the same
// get_adaptation() as A2 and give A2's draws.
template <class Make>
bool check_alloc_failure(const char* label, Make make, int n_pre, int n_post) {
    auto S = make();
    std::mt19937_64 rng_s(556);
    run_trace ts;
    for (int i = 0; i < n_pre / 2; ++i) sweep(*S, rng_s, ts);
    adaptation_info other = S->get_adaptation();
    set_other_values(other);
    const arma::vec m = diag_metric_values();
    other.precond_mat  = arma::diagmat(m);
    other.precond_diag = arma::conv_to<std::vector<double>>::from(m);
    other.metric_kind  = "diagonal";

    auto T = make();
    auto R = make();
    const bool differ_t = step_fields_differ(other, R->get_adaptation());
    const bool threw_t = set_adaptation_alloc_fails(*T, other);
    const bool state_t = same_adaptation(T->get_adaptation(), R->get_adaptation());
    std::mt19937_64 rng_t(810), rng_r(810);
    run_trace tt, tr;
    const bool ran_t = sweeps(*T, rng_t, tt, n_post) && sweeps(*R, rng_r, tr, n_post);
    const std::size_t same_t = ran_t ? n_identical(tr, tt) : 0;
    const bool ok_t = threw_t && differ_t && state_t && ran_t
                   && same_t == static_cast<std::size_t>(n_post)
                   && first_evals(tt) == first_evals(tr);

    auto A = make();
    auto A2 = make();
    std::mt19937_64 rng_a(910), rng_a2(910);
    run_trace pa, pa2;
    for (int i = 0; i < n_pre; ++i) sweep(*A, rng_a, pa);
    for (int i = 0; i < n_pre; ++i) sweep(*A2, rng_a2, pa2);
    const bool differ = step_fields_differ(other, A2->get_adaptation());
    const bool threw = set_adaptation_alloc_fails(*A, other);
    const bool state = same_adaptation(A->get_adaptation(), A2->get_adaptation());
    run_trace ta, ta2;
    const bool ran = sweeps(*A, rng_a, ta, n_post) && sweeps(*A2, rng_a2, ta2, n_post);
    const std::size_t same = ran ? n_identical(ta2, ta) : 0;
    const bool ok = threw && differ && state && ran
                 && same == static_cast<std::size_t>(n_post)
                 && first_evals(ta) == first_evals(ta2);
    std::printf("  %-46s fresh: threw bad_alloc %d, snapshot fields differ %d, "
                "adaptation equal to fresh block %d, identical %zu/%d, "
                "first-sweep evals R=%lld T=%lld  %s\n", label, int(threw_t),
                int(differ_t), int(state_t), same_t, n_post, first_evals(tr),
                first_evals(tt), ok_t ? "PASS" : "FAIL");
    std::printf("  %-46s stepped: threw bad_alloc %d, snapshot fields differ %d, "
                "adaptation equal to twin %d, identical %zu/%d, step %.6g "
                "(twin %.6g)  %s\n", label, int(threw), int(differ), int(state),
                same, n_post, first_step(ta), first_step(ta2),
                ok ? "PASS" : "FAIL");
    return ok_t && ok;
}

// A constructor given nuts_settings.nuts_settings.precond_mat that is not
// d x d (the shapes of all_bad_shapes()) must throw std::invalid_argument;
// with a d x d diagonal precond_mat the block must give the draws of the
// block from make_ref(), which installs the same matrix with
// set_precond_matrix().
template <class MakeWith, class MakeRef>
bool check_ctor(const char* label, MakeWith make_with, MakeRef make_ref,
                int n_post) {
    bool ok = true;
    for (const arma::mat& M : all_bad_shapes()) {
        bool threw = false;
        try { make_with(M); }
        catch (const std::invalid_argument&) { threw = true; }
        catch (const std::exception&) {}
        std::printf("  %-46s precond_mat %llux%llu in the config: threw "
                    "invalid_argument %d  %s\n", label,
                    static_cast<unsigned long long>(M.n_rows),
                    static_cast<unsigned long long>(M.n_cols), int(threw),
                    threw ? "PASS" : "FAIL");
        ok = ok && threw;
    }
    auto C = make_with(arma::mat(arma::diagmat(diag_metric_values())));
    auto R = make_ref();
    std::mt19937_64 rng_c(77), rng_r(77);
    run_trace tc, tr;
    for (int i = 0; i < n_post; ++i) sweep(*C, rng_c, tc);
    for (int i = 0; i < n_post; ++i) sweep(*R, rng_r, tr);
    const std::size_t same = n_identical(tr, tc);
    const bool ok_d = same == static_cast<std::size_t>(n_post)
                   && first_evals(tc) == first_evals(tr);
    std::printf("  %-46s diagonal %zux%zu precond_mat in the config: "
                "identical %zu/%d to set_precond_matrix, first-sweep evals "
                "R=%lld C=%lld  %s\n", label, kDim, kDim, same, n_post,
                first_evals(tr), first_evals(tc), ok_d ? "PASS" : "FAIL");
    return ok && ok_d;
}

// cfg_precond: nuts_settings.nuts_settings.precond_mat of the config.
std::unique_ptr<nuts_block> make_nuts(bool diag_metric,
                                      double initial_step_size = 0.0,
                                      const arma::mat& cfg_precond = arma::mat()) {
    nuts_block_config cfg;
    cfg.name = "theta";
    cfg.initial_unc = arma::vec(kDim, arma::fill::zeros);
    cfg.log_density_grad = &target;
    cfg.initial_step_size = initial_step_size;
    cfg.nuts_settings.nuts_settings.precond_mat = cfg_precond;
    std::unique_ptr<nuts_block> b(new nuts_block(std::move(cfg)));
    if (diag_metric) b->set_precond_matrix(arma::diagmat(diag_metric_values()));
    return b;
}

std::unique_ptr<joint_nuts_block> make_joint(int metric,  // 0 id, 1 diag, 2 dense
                                             double initial_step_size = 0.0,
                                             const arma::mat& cfg_precond = arma::mat()) {
    joint_nuts_block_config cfg;
    cfg.name = "J";
    cfg.sub_params.push_back({"theta", kDim});
    cfg.log_density_grad = &target;
    cfg.initial_cat = arma::vec(kDim, arma::fill::zeros);
    cfg.use_diagonal_metric = (metric == 1);
    cfg.use_dense_metric    = (metric == 2);
    cfg.initial_step_size   = initial_step_size;
    cfg.nuts_settings.nuts_settings.precond_mat = cfg_precond;
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

    std::printf("Snapshot taken before any step(): the restored block runs the "
                "first-call warmup:\n");
    ok &= check_fresh_snapshot("nuts_block, identity metric",
                               [] { return make_nuts(false); }, n_post);
    ok &= check_fresh_snapshot("joint_nuts_block, use_diagonal_metric",
                               [] { return make_joint(1); }, n_post);

    std::printf("Snapshot taken after readapt() and before the first step(): "
                "the restored block runs the first-call warmup:\n");
    ok &= check_readapt_snapshot("nuts_block, identity metric",
                                 [] { return make_nuts(false); }, n_short);
    ok &= check_readapt_snapshot("joint_nuts_block, use_diagonal_metric",
                                 [] { return make_joint(1); }, n_short);

    std::printf("Hand-built adaptation_info on a fresh block: the block runs "
                "the first-call warmup:\n");
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

    std::printf("joint_nuts_block::set_adaptation() with a precond_mat that is "
                "not d x d throws and changes nothing:\n");
    ok &= check_throw_unchanged("joint_nuts_block, identity metric",
                                [] { return make_joint(0); }, call_set_adaptation(),
                                all_bad_shapes(), n_pre, n_short);
    ok &= check_throw_unchanged("joint_nuts_block, use_diagonal_metric",
                                [] { return make_joint(1); }, call_set_adaptation(),
                                all_bad_shapes(), n_pre, n_short);
    ok &= check_throw_unchanged("joint_nuts_block, use_dense_metric",
                                [] { return make_joint(2); }, call_set_adaptation(),
                                all_bad_shapes(), n_pre, n_short);

    std::printf("set_adaptation() whose copy of precond_mat throws "
                "std::bad_alloc changes nothing:\n");
    ok &= check_alloc_failure("nuts_block, identity metric",
                              [] { return make_nuts(false); }, n_pre, n_short);
    ok &= check_alloc_failure("joint_nuts_block, identity metric",
                              [] { return make_joint(0); }, n_pre, n_short);

    std::printf("A precond_mat that is not d x d: "
                "set_adaptation() and set_precond_matrix() throw and change "
                "nothing:\n");
    ok &= check_throw_unchanged("nuts_block set_adaptation, identity",
                                [] { return make_nuts(false); }, call_set_adaptation(),
                                all_bad_shapes(), n_pre, n_short);
    ok &= check_throw_unchanged("nuts_block set_adaptation, diagonal",
                                [] { return make_nuts(true); }, call_set_adaptation(),
                                all_bad_shapes(), n_pre, n_short);
    ok &= check_throw_unchanged("nuts_block set_precond_matrix, identity",
                                [] { return make_nuts(false); }, call_set_precond_matrix(),
                                all_bad_shapes(), n_pre, n_short);
    ok &= check_throw_unchanged("nuts_block set_precond_matrix, diagonal",
                                [] { return make_nuts(true); }, call_set_precond_matrix(),
                                all_bad_shapes(), n_pre, n_short);
    ok &= check_throw_unchanged("joint set_precond_matrix, identity",
                                [] { return make_joint(0); }, call_set_precond_matrix(),
                                all_bad_shapes(), n_pre, n_short);
    ok &= check_throw_unchanged("joint set_precond_matrix, diagonal",
                                [] { return make_joint(1); }, call_set_precond_matrix(),
                                all_bad_shapes(), n_pre, n_short);
    ok &= check_throw_unchanged("joint set_precond_matrix, dense",
                                [] { return make_joint(2); }, call_set_precond_matrix(),
                                all_bad_shapes(), n_pre, n_short);

    std::printf("Constructors: a precond_mat that is not "
                "d x d throws; a d x d one gives the draws of "
                "set_precond_matrix():\n");
    ok &= check_ctor("nuts_block",
                     [](const arma::mat& M) { return make_nuts(false, 0.0, M); },
                     [] { return make_nuts(true); }, n_short);
    ok &= check_ctor("joint_nuts_block, identity metric",
                     [](const arma::mat& M) { return make_joint(0, 0.0, M); },
                     [] {
                         auto b = make_joint(0);
                         b->set_precond_matrix(arma::diagmat(diag_metric_values()));
                         return b;
                     }, n_short);

    std::printf("=== %s ===\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
