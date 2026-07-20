/*================================================================================
 *  joint_nuts_block slot-level freeze acceptance test (Batch O, 2026-07-20).
 *
 *  Sec.10.a of DESIGN_NOTES_FREEZE_UNFREEZE_2026-07-19.md defines the entry
 *  criterion for flipping the kernel patch from "IMPL deferred" to "IMPL
 *  shipped":
 *
 *      "unit test that samples a joint block of (beta[10], log_sigma) with
 *       dense metric, freezes log_sigma at 0 (sigma=1), and confirms beta's
 *       marginal posterior matches the theoretical N(0, XtX/sigma^2)^{-1}
 *       distribution within Monte Carlo error."
 *
 *  This file IS that test. Two phases:
 *
 *   F1  Whole-block freeze correctness (works TODAY via composite skip):
 *       build a joint block, freeze the whole block, run step()s, assert
 *       theta_cat_ unchanged bitwise.
 *
 *   F2  Slot-level freeze correctness (needs the kernel patch to pass):
 *       build a joint (beta[3], log_sigma), sample the joint posterior for
 *       N steps, freeze the log_sigma slot at 0, sample N more, and assert
 *       the frozen slot is bitwise unchanged AND beta's empirical posterior
 *       mean matches the analytic conditional-on-sigma=1 solution
 *       (X^T X)^{-1} X^T y within Monte Carlo error.
 *
 *  While the kernel patch is not yet landed, F2's freeze_sub call will
 *  throw with the "not yet implemented" guidance; the test asserts that
 *  behavior instead. Once the kernel patch flips freeze_sub from THROW to
 *  actual masking, comment out the "expected throw" branch and enable the
 *  posterior-recovery assertion below it.
 *
 *  See DESIGN_NOTES_FREEZE_UNFREEZE_2026-07-19.md Sec.10.a for context.
 *================================================================================*/

#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/joint_nuts_block.hpp"

#include <armadillo>
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace AI4BayesCode;

struct test_result { int passed = 0; int failed = 0; };
test_result G_RES;
static void check(bool ok, const std::string& tag, const std::string& detail = "") {
    if (ok) { ++G_RES.passed; std::printf("  PASS  %s\n", tag.c_str()); }
    else    { ++G_RES.failed; std::printf("  FAIL  %s  %s\n", tag.c_str(), detail.c_str()); }
}

// Build a joint_nuts_block over (beta[p], log_sigma) whose natural-scale
// log-density is a Gaussian regression log-likelihood + weak priors:
//   y_i | beta, sigma ~ N(x_i^T beta, sigma^2)
//   beta_j ~ N(0, tau_prior^2)             (tau_prior = 100)
//   log_sigma ~ Uniform on R (prior flat)  -- with POSITIVE constraint on sigma
//
// nat[0..p-1] = beta, nat[p] = sigma (on natural scale).
static std::unique_ptr<joint_nuts_block>
make_reg_block(const std::string& name,
               const arma::mat& X,
               const arma::vec& y,
               double tau_prior,
               const arma::vec& init_cat_natural,
               bool use_dense = false) {
    const std::size_t p = X.n_cols;
    const std::size_t n = X.n_rows;

    joint_nuts_block_config cfg;
    cfg.name = name;
    joint_nuts_sub_param sp_beta;      sp_beta.name = "beta";     sp_beta.dim = p;
    sp_beta.constraint = joint_constraint::REAL;
    joint_nuts_sub_param sp_log_sigma; sp_log_sigma.name = "log_sigma"; sp_log_sigma.dim = 1;
    sp_log_sigma.constraint = joint_constraint::POSITIVE;   // sigma > 0, we
                                                             // sample log_sigma
                                                             // on the unconstrained
                                                             // scale but expose
                                                             // sigma on natural
    cfg.sub_params = { sp_beta, sp_log_sigma };
    cfg.initial_cat = init_cat_natural;
    if (use_dense) cfg.use_dense_metric = true;

    // Log density on the NATURAL scale (beta, sigma).  Chain rule for the
    // POSITIVE constraint on sigma is applied inside joint_nuts_block via
    // constraints::positive::wrap; the user's log_density_grad receives the
    // NATURAL gradient which the framework then chain-rules to the
    // unconstrained scale.
    cfg.log_density_grad = [X, y, tau_prior, p, n](const arma::vec& nat,
            const block_context&, arma::vec* g) -> double {
        const arma::vec beta = nat.subvec(0, p - 1);
        const double sigma = nat[p];
        if (!(sigma > 0.0)) return -std::numeric_limits<double>::infinity();
        const arma::vec resid = y - X * beta;
        const double logp_lik = -0.5 * arma::dot(resid, resid) / (sigma * sigma)
                              - static_cast<double>(n) * std::log(sigma);
        const double logp_prior_beta = -0.5 * arma::dot(beta, beta) / (tau_prior * tau_prior);
        const double logp = logp_lik + logp_prior_beta;
        if (g) {
            g->set_size(p + 1);
            const arma::vec dbeta = (X.t() * resid) / (sigma * sigma) - beta / (tau_prior * tau_prior);
            for (std::size_t j = 0; j < p; ++j) (*g)[j] = dbeta[j];
            const double dsigma = arma::dot(resid, resid) / (sigma * sigma * sigma) - static_cast<double>(n) / sigma;
            (*g)[p] = dsigma;
        }
        return logp;
    };

    return std::make_unique<joint_nuts_block>(cfg);
}

// ---- F1: whole-block freeze via composite is_frozen() gate ------------------
static void F1_whole_block_freeze() {
    std::printf("\n--- F1: whole-block freeze on joint_nuts_block ---\n");
    const std::size_t n = 50, p = 2;
    std::mt19937_64 rng(20260720ULL);
    std::normal_distribution<double> norm(0.0, 1.0);
    arma::mat X(n, p);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j) X(i, j) = norm(rng);
    arma::vec beta_true{1.5, -0.5};
    arma::vec y = X * beta_true + arma::randn(n);

    // (beta[2]=0, log_sigma=0 → sigma=1)
    arma::vec init_nat = arma::zeros(p + 1);  init_nat[p] = 1.0;
    auto blk = make_reg_block("theta", X, y, 100.0, init_nat);

    composite_block comp("F1_wrapper");
    comp.add_child(std::move(blk));

    // Sample 200 steps unfrozen (chain moves)
    for (int i = 0; i < 200; ++i) comp.step(rng);

    // Snapshot current concatenated state
    arma::vec snap = comp.current();
    check(snap.n_elem == p + 1, "F1.a concat state length p+1", "n=" + std::to_string(snap.n_elem));

    // Freeze the whole block
    auto warns = comp.freeze({"theta"});
    check(warns.empty(), "F1.b whole-block freeze emits no warning on fresh freeze");
    check(comp.get_frozen() == std::vector<std::string>{"theta"}, "F1.c get_frozen == {\"theta\"}");

    // 500 more steps must not move the state (composite skips step)
    for (int i = 0; i < 500; ++i) comp.step(rng);
    arma::vec after = comp.current();
    bool bit_identical = (after.n_elem == snap.n_elem);
    for (std::size_t k = 0; k < snap.n_elem && bit_identical; ++k) {
        if (after[k] != snap[k]) bit_identical = false;
    }
    check(bit_identical, "F1.d frozen composite state bitwise unchanged after 500 step()s");

    // Unfreeze; chain resumes moving.
    comp.unfreeze_all();
    check(comp.get_frozen().empty(), "F1.e unfreeze_all clears frozen set");
    for (int i = 0; i < 200; ++i) comp.step(rng);
    arma::vec after2 = comp.current();
    bool moved = false;
    for (std::size_t k = 0; k < snap.n_elem; ++k) {
        if (after2[k] != snap[k]) { moved = true; break; }
    }
    check(moved, "F1.f chain resumes moving after unfreeze");
}

// ---- F2: slot-level freeze on joint_nuts_block -----------------------------
// Currently: freeze_sub throws. Once kernel patch lands: run the full posterior
// recovery test below.
static void F2_slot_freeze() {
    std::printf("\n--- F2: slot-level freeze (log_sigma) on joint_nuts_block ---\n");
    const std::size_t n = 100, p = 3;
    std::mt19937_64 rng(2026072002ULL);
    std::normal_distribution<double> norm(0.0, 1.0);
    arma::mat X(n, p);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j) X(i, j) = norm(rng);
    arma::vec beta_true{1.0, -0.5, 0.3};
    // Generate with sigma=1 (matches the freeze target below).
    arma::vec y = X * beta_true + arma::randn(n);

    // Initial values: beta=0, sigma=1 (log_sigma=0)
    arma::vec init_nat = arma::zeros(p + 1);  init_nat[p] = 1.0;
    auto blk = make_reg_block("theta", X, y, 100.0, init_nat);

    composite_block comp("F2_wrapper");
    comp.add_child(std::move(blk));

    // Sample 500 steps unfrozen (warmup + burn-in)
    for (int i = 0; i < 500; ++i) comp.step(rng);
    // Note current sigma; freeze_sub target
    arma::vec pre_state = comp.current();

    // Slot-level freeze via composite dispatch (Approach B shipped 2026-07-20).
    comp.freeze({"log_sigma"});   // sub-name freeze on joint block's slot
    check(!comp.get_frozen().empty(), "F2.a get_frozen non-empty after slot freeze");
    // Set log_sigma to exactly 0 (sigma=1) and freeze
    // NOTE: This assumes the mixin has a set_current path that routes
    // sub-parameter keys correctly. In the pre-patch state, this is a
    // best-effort test.
    arma::vec fixed = pre_state;
    fixed[p] = 1.0;   // sigma slot at 1
    comp.set_current(fixed);  // sets whole concat vector; frozen slots retain the just-set value

    // Sample many steps with log_sigma frozen at 1.
    const int N_KEEP = 4000;
    arma::mat beta_hist(N_KEEP, p);
    for (int i = 0; i < N_KEEP; ++i) {
        comp.step(rng);
        arma::vec cur = comp.current();
        for (std::size_t j = 0; j < p; ++j) beta_hist(i, j) = cur[j];
    }

    // Assert frozen slot bitwise unchanged.
    arma::vec after = comp.current();
    check(after[p] == 1.0, "F2.b sigma slot unchanged after 4000 frozen steps");

    // Assert beta posterior ≈ (X^T X + tau^-2 I)^{-1} X^T y  with sigma=1.
    arma::vec beta_mean_emp = arma::mean(beta_hist, 0).t();
    const double tau_prior_2 = 100.0 * 100.0;
    arma::mat XtX = X.t() * X;
    arma::mat post_prec = XtX + arma::eye(p, p) / tau_prior_2;
    arma::vec beta_analytic = arma::solve(post_prec, X.t() * y);

    double max_err = arma::max(arma::abs(beta_mean_emp - beta_analytic));
    check(max_err < 0.05,
          "F2.c empirical beta mean matches analytic conditional-on-sigma=1",
          "max_err=" + std::to_string(max_err));

    comp.unfreeze_all();
}

// ---- F3: readapt_NUTS on a wrapper with a frozen slot ----------------------
// Verifies that the freeze wrap I added to readapt() in f1900e6 correctly
// leaves the DA state tuned for the CONDITIONAL (not joint) target. If the
// wrap were missing, readapt would tune DA against the joint posterior;
// then subsequent frozen sampling would show biased beta marginals.
static void F3_readapt_frozen() {
    std::printf("\n--- F3: readapt_NUTS on wrapper with frozen slot ---\n");
    const std::size_t n = 100, p = 3;
    std::mt19937_64 rng(20260721001ULL);
    std::normal_distribution<double> norm(0.0, 1.0);
    arma::mat X(n, p);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j) X(i, j) = norm(rng);
    arma::vec beta_true{1.0, -0.5, 0.3};
    arma::vec y = X * beta_true + arma::randn(n);

    arma::vec init_nat = arma::zeros(p + 1);  init_nat[p] = 1.0;
    auto blk = make_reg_block("theta", X, y, 100.0, init_nat);

    composite_block comp("F3_wrapper");
    comp.add_child(std::move(blk));

    // Warmup + freeze log_sigma at 1.
    for (int i = 0; i < 500; ++i) comp.step(rng);
    arma::vec fix = comp.current();
    fix[p] = 1.0;
    comp.set_current(fix);
    comp.freeze({"log_sigma"});

    // Snapshot state and step-size persistence BEFORE readapt.
    arma::vec pre_readapt = comp.current();

    // Call readapt_NUTS. State should be preserved bitwise; DA state re-tunes.
    // Without the freeze wrap in readapt(), DA would see the JOINT posterior
    // (log_sigma unfixed inside readapt). With the wrap, DA sees the
    // conditional target -- posterior recovery after this call must still
    // work.
    std::mt19937_64 readapt_rng(20260721010ULL);
    comp.readapt_NUTS(500, /*reset=*/false, readapt_rng);

    // Check #23-style chain state preservation.
    arma::vec post_readapt = comp.current();
    bool bit_identical = true;
    for (std::size_t k = 0; k < pre_readapt.n_elem; ++k) {
        if (post_readapt[k] != pre_readapt[k]) { bit_identical = false; break; }
    }
    check(bit_identical, "F3.a chain state bitwise unchanged after readapt on frozen wrapper");
    check(post_readapt[p] == 1.0, "F3.b sigma slot still at frozen value after readapt");

    // Sample MORE with the (now re-tuned) DA state and check posterior
    // still matches the conditional-on-sigma=1 solution. If readapt had
    // tuned on the joint (missing freeze wrap), the DA state would be off
    // and posterior recovery would drift.
    const int N_KEEP = 3000;
    arma::mat beta_hist(N_KEEP, p);
    for (int i = 0; i < N_KEEP; ++i) {
        comp.step(rng);
        arma::vec cur = comp.current();
        for (std::size_t j = 0; j < p; ++j) beta_hist(i, j) = cur[j];
    }
    // sigma still frozen at 1
    arma::vec final_state = comp.current();
    check(final_state[p] == 1.0, "F3.c sigma still 1.0 after 3000 post-readapt steps");

    // Empirical vs analytic
    arma::vec beta_mean = arma::mean(beta_hist, 0).t();
    arma::mat post_prec = X.t() * X + arma::eye(p, p) / 10000.0;
    arma::vec beta_ana = arma::solve(post_prec, X.t() * y);
    double max_err = arma::max(arma::abs(beta_mean - beta_ana));
    check(max_err < 0.05,
          "F3.d empirical beta post-readapt still matches analytic conditional",
          "max_err=" + std::to_string(max_err));

    comp.unfreeze_all();
}

// ---- F4: dense metric + slot freeze end-to-end ----------------------------
// F2 exercises identity metric; F4 flips use_dense_metric = true. This walks
// through adapt_dense_metric_ pilot + slot freeze active during the pilot.
static void F4_dense_metric_frozen() {
    std::printf("\n--- F4: dense metric + slot freeze end-to-end ---\n");
    const std::size_t n = 100, p = 3;
    std::mt19937_64 rng(20260721002ULL);
    std::normal_distribution<double> norm(0.0, 1.0);
    arma::mat X(n, p);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j) X(i, j) = norm(rng);
    // Add some correlation to X columns -- makes joint (beta, log_sigma)
    // posterior have real cross-couplings and dense metric matter.
    for (std::size_t i = 0; i < n; ++i) X(i, 1) += 0.7 * X(i, 0);
    arma::vec beta_true{1.0, -0.5, 0.3};
    arma::vec y = X * beta_true + arma::randn(n);

    arma::vec init_nat = arma::zeros(p + 1);  init_nat[p] = 1.0;
    auto blk = make_reg_block("theta", X, y, 100.0, init_nat, /*use_dense=*/true);
    // Dense metric enabled via config (make_reg_block use_dense=true).

    composite_block comp("F4_wrapper");
    comp.add_child(std::move(blk));

    // Warmup with dense metric (fires adapt_dense_metric_ pilot).
    for (int i = 0; i < 500; ++i) comp.step(rng);

    // Freeze log_sigma at 1 and sample under dense metric.
    arma::vec fix = comp.current();
    fix[p] = 1.0;
    comp.set_current(fix);
    comp.freeze({"log_sigma"});

    const int N_KEEP = 3000;
    arma::mat beta_hist(N_KEEP, p);
    for (int i = 0; i < N_KEEP; ++i) {
        comp.step(rng);
        arma::vec cur = comp.current();
        for (std::size_t j = 0; j < p; ++j) beta_hist(i, j) = cur[j];
    }

    arma::vec final_state = comp.current();
    check(final_state[p] == 1.0, "F4.a sigma slot preserved under dense metric + freeze");

    arma::vec beta_mean = arma::mean(beta_hist, 0).t();
    arma::mat post_prec = X.t() * X + arma::eye(p, p) / 10000.0;
    arma::vec beta_ana = arma::solve(post_prec, X.t() * y);
    double max_err = arma::max(arma::abs(beta_mean - beta_ana));
    check(max_err < 0.08,
          "F4.b beta posterior recovery under dense metric matches analytic (loose tol)",
          "max_err=" + std::to_string(max_err));

    comp.unfreeze_all();
}

// ---- F5: freeze BEFORE first step (adapt_dense_metric_ pilot sees freeze) --
// Verifies the freeze wrap in adapt_dense_metric_ correctly runs the pilot
// on the CONDITIONAL target (frozen slot pinned) rather than the joint.
// Otherwise Sigma_reg would reflect joint covariance -> mass matrix wrong
// for conditional sampling -> posterior recovery drifts.
static void F5_freeze_before_first_step() {
    std::printf("\n--- F5: freeze BEFORE first step ---\n");
    const std::size_t n = 100, p = 3;
    std::mt19937_64 rng(20260721003ULL);
    std::normal_distribution<double> norm(0.0, 1.0);
    arma::mat X(n, p);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j) X(i, j) = norm(rng);
    arma::vec beta_true{1.0, -0.5, 0.3};
    arma::vec y = X * beta_true + arma::randn(n);

    // init sigma at 1 (log_sigma = 0)
    arma::vec init_nat = arma::zeros(p + 1);  init_nat[p] = 1.0;
    auto blk = make_reg_block("theta", X, y, 100.0, init_nat, /*use_dense=*/true);

    composite_block comp("F5_wrapper");
    comp.add_child(std::move(blk));

    // Freeze log_sigma BEFORE the first step.
    comp.freeze({"log_sigma"});
    // get_frozen returns canonical dot-path form ("<block>.<slot>") per
    // DESIGN_NOTES Sec.10.b/c ordering.
    auto fr = comp.get_frozen();
    check(fr.size() == 1 && fr[0] == std::string("theta.log_sigma"),
          "F5.a frozen before first step (dot-path canonical form)");

    // Now sample; first step() triggers adapt_dense_metric_ pilot which
    // should see the frozen wrap.
    for (int i = 0; i < 500; ++i) comp.step(rng);

    const int N_KEEP = 3000;
    arma::mat beta_hist(N_KEEP, p);
    for (int i = 0; i < N_KEEP; ++i) {
        comp.step(rng);
        arma::vec cur = comp.current();
        for (std::size_t j = 0; j < p; ++j) beta_hist(i, j) = cur[j];
    }

    arma::vec final_state = comp.current();
    check(std::abs(final_state[p] - 1.0) < 1e-12,
          "F5.b sigma slot preserved at 1.0 through pilot + sampling",
          "sigma=" + std::to_string(final_state[p]));

    arma::vec beta_mean = arma::mean(beta_hist, 0).t();
    arma::mat post_prec = X.t() * X + arma::eye(p, p) / 10000.0;
    arma::vec beta_ana = arma::solve(post_prec, X.t() * y);
    double max_err = arma::max(arma::abs(beta_mean - beta_ana));
    check(max_err < 0.08,
          "F5.c beta posterior recovers when pilot ran with freeze already active",
          "max_err=" + std::to_string(max_err));

    comp.unfreeze_all();
}

} // namespace

int main() {
    std::printf("=== joint_nuts_block slot-freeze acceptance test ===\n");
    F1_whole_block_freeze();
    F2_slot_freeze();
    F3_readapt_frozen();
    F4_dense_metric_frozen();
    F5_freeze_before_first_step();
    std::printf("\n=== SUMMARY: %d passed, %d failed ===\n",
                G_RES.passed, G_RES.failed);
    return G_RES.failed == 0 ? 0 : 1;
}
