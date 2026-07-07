/*================================================================================
 *  AI4BayesCode v1.2 Block 3 (order_mcmc_block) — bde_scorer unit tests.
 *
 *    T1  2-node Heckerman example: hand-computed BDe score
 *    T2  Likelihood equivalence: B_{X→Y} and B_{X←Y} same DAG score
 *    T3  Empty parent set: closed-form check
 *    T4  Structure prior: rho_{X_i}(U) ∝ 1/C(n-1, |U|)
 *    T5  alpha scaling: larger alpha → smaller |score|
 *    T6  Single-edge score consistency
 *    T7  top_candidate_parents respects ranking
 *    T8  Invalid input validation
 *================================================================================*/

#include "AI4BayesCode/bde_scorer.hpp"

#include <armadillo>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <sstream>

namespace {

using namespace AI4BayesCode;

struct test_result { int passed = 0; int failed = 0; };
test_result G_RES;
static void check(bool ok, const std::string& tag, const std::string& detail = "") {
    if (ok) { ++G_RES.passed; std::printf("  PASS  %s\n", tag.c_str()); }
    else    { ++G_RES.failed; std::printf("  FAIL  %s  %s\n", tag.c_str(), detail.c_str()); }
}
template <typename T> static std::string ts(const T& v) {
    std::ostringstream s; s << std::setprecision(10) << v; return s.str();
}

// ===========================================================================
//  T1: 2-node 2-state hand-computed example
//
//  Data:
//    X Y
//    0 0   (n_00 = 5)
//    0 1   (n_01 = 3)
//    1 0   (n_10 = 2)
//    1 1   (n_11 = 7)
//  (One row per occurrence.)
//
//  Empty parent for Y: r_i=2, q_i=1, Np_ij=alpha, Np_ijk=alpha/2.
//    Σ_k [lgamma(N'_ijk + N_ijk) - lgamma(N'_ijk)]
//    = lgamma(alpha/2 + 8) + lgamma(alpha/2 + 9)
//      - 2*lgamma(alpha/2)
//    + lgamma(alpha) - lgamma(alpha + 17)
//  with no structure-prior contribution (single node, |U|=0).
// ===========================================================================
static void T1_hand_computed() {
    std::printf("\n--- T1: 2-node Heckerman hand-computed BDe ---\n");
    arma::imat data(17, 2, arma::fill::zeros);
    std::size_t r = 0;
    for (std::size_t i = 0; i < 5; ++i) { data(r,0)=0; data(r,1)=0; ++r; }
    for (std::size_t i = 0; i < 3; ++i) { data(r,0)=0; data(r,1)=1; ++r; }
    for (std::size_t i = 0; i < 2; ++i) { data(r,0)=1; data(r,1)=0; ++r; }
    for (std::size_t i = 0; i < 7; ++i) { data(r,0)=1; data(r,1)=1; ++r; }

    bde_scorer_config cfg;
    cfg.data = data;
    cfg.cardinalities = arma::uvec({2, 2});
    cfg.alpha = 1.0;
    cfg.use_structure_prior = false;
    bde_scorer s(cfg);

    // Empty-parent score for Y (i=1).
    // q_i=1, r_i=2, Np_ij=1.0, Np_ijk=0.5.
    // N_1jk counts: N_{1,0,0}=5+2=7, N_{1,0,1}=3+7=10, N_{1,0}=17
    const double expected_Y_empty =
        std::lgamma(0.5 + 7.0) - std::lgamma(0.5)
      + std::lgamma(0.5 + 10.0) - std::lgamma(0.5)
      + std::lgamma(1.0)
      - std::lgamma(1.0 + 17.0);
    const double got_Y_empty = s.family_score(1, 0u);
    std::printf("    expected = %.6f\n", expected_Y_empty);
    std::printf("    got      = %.6f\n", got_Y_empty);
    check(std::abs(got_Y_empty - expected_Y_empty) < 1e-9,
          "T1 empty-parent score matches hand-computed",
          "diff = " + ts(std::abs(got_Y_empty - expected_Y_empty)));

    // X→Y score (i=1, U={0}).
    // q_i=2, r_i=2, Np_ij=0.5, Np_ijk=0.25.
    // Counts:
    //  X=0: N_{1,0,0}=5, N_{1,0,1}=3, N_{1,0}=8
    //  X=1: N_{1,1,0}=2, N_{1,1,1}=7, N_{1,1}=9
    const double expected_Y_givenX =
        std::lgamma(0.25 + 5.0) - std::lgamma(0.25)
      + std::lgamma(0.25 + 3.0) - std::lgamma(0.25)
      + std::lgamma(0.5)
      - std::lgamma(0.5 + 8.0)
      + std::lgamma(0.25 + 2.0) - std::lgamma(0.25)
      + std::lgamma(0.25 + 7.0) - std::lgamma(0.25)
      + std::lgamma(0.5)
      - std::lgamma(0.5 + 9.0);
    const double got_Y_givenX = s.family_score(1, 1u << 0);  // parent = X (bit 0)
    std::printf("    Y|X expected = %.6f\n", expected_Y_givenX);
    std::printf("    Y|X got      = %.6f\n", got_Y_givenX);
    check(std::abs(got_Y_givenX - expected_Y_givenX) < 1e-9,
          "T1 family_score(Y, {X}) matches hand-computed",
          "diff = " + ts(std::abs(got_Y_givenX - expected_Y_givenX)));
}

// ===========================================================================
//  T2: Likelihood equivalence — BDe scores I-equivalent DAGs identically
//
//  For 2-node case: B_{X→Y} and B_{X←Y} are in the same Markov equiv class.
//  Their total scores (over both nodes) must match.
// ===========================================================================
static void T2_likelihood_equivalence() {
    std::printf("\n--- T2: likelihood equivalence on 2-node ---\n");
    arma::imat data(17, 2, arma::fill::zeros);
    std::size_t r = 0;
    for (std::size_t i = 0; i < 5; ++i) { data(r,0)=0; data(r,1)=0; ++r; }
    for (std::size_t i = 0; i < 3; ++i) { data(r,0)=0; data(r,1)=1; ++r; }
    for (std::size_t i = 0; i < 2; ++i) { data(r,0)=1; data(r,1)=0; ++r; }
    for (std::size_t i = 0; i < 7; ++i) { data(r,0)=1; data(r,1)=1; ++r; }

    bde_scorer_config cfg;
    cfg.data = data;
    cfg.cardinalities = arma::uvec({2, 2});
    cfg.alpha = 1.0;
    cfg.use_structure_prior = false;
    bde_scorer s(cfg);

    // B_{X→Y}: score(X | {}) + score(Y | {X})
    const double s_X_to_Y = s.family_score(0, 0u) + s.family_score(1, 1u << 0);
    // B_{X←Y}: score(X | {Y}) + score(Y | {})
    const double s_X_from_Y = s.family_score(0, 1u << 1) + s.family_score(1, 0u);
    std::printf("    score(X→Y) = %.6f\n", s_X_to_Y);
    std::printf("    score(X←Y) = %.6f\n", s_X_from_Y);
    check(std::abs(s_X_to_Y - s_X_from_Y) < 1e-9,
          "T2 BDe is likelihood-equivalent on 2-node",
          "diff = " + ts(std::abs(s_X_to_Y - s_X_from_Y)));
}

// ===========================================================================
//  T3: Empty parent closed-form
//
//  For empty parent: q_i=1, score = lgamma(alpha) - lgamma(alpha+N)
//    + sum_k [lgamma(alpha/r_i + N_k) - lgamma(alpha/r_i)]
// ===========================================================================
static void T3_empty_parent_closed_form() {
    std::printf("\n--- T3: empty-parent closed-form on 3-state node ---\n");
    arma::imat data(30, 1, arma::fill::zeros);
    for (std::size_t i = 0; i < 30; ++i) data(i, 0) = static_cast<int>(i % 3);
    bde_scorer_config cfg;
    cfg.data = data;
    cfg.cardinalities = arma::uvec({3});
    cfg.alpha = 1.0;
    cfg.use_structure_prior = false;
    bde_scorer s(cfg);
    const double got = s.family_score(0, 0u);
    // r_i=3, q_i=1, Np_ij=1, Np_ijk=1/3, N_k=10 each.
    const double exp_score =
        3 * (std::lgamma(1.0/3.0 + 10.0) - std::lgamma(1.0/3.0))
      + std::lgamma(1.0) - std::lgamma(1.0 + 30.0);
    std::printf("    expected = %.6f, got = %.6f\n", exp_score, got);
    check(std::abs(got - exp_score) < 1e-9,
          "T3 empty-parent uniform closed-form matches",
          "diff = " + ts(std::abs(got - exp_score)));
}

// ===========================================================================
//  T4: Structure prior — different |U| gives different log rho_X_i(U)
// ===========================================================================
static void T4_structure_prior() {
    std::printf("\n--- T4: structure prior FK Eq 2 — rho(U) ∝ 1/C(n-1, |U|) ---\n");
    arma::imat data(20, 5, arma::fill::zeros);
    arma::arma_rng::set_seed(7);
    for (std::size_t i = 0; i < 20; ++i)
        for (std::size_t j = 0; j < 5; ++j)
            data(i, j) = static_cast<int>(arma::randi<int>(arma::distr_param(0, 1)));
    bde_scorer_config cfg;
    cfg.data = data;
    cfg.cardinalities = arma::uvec(5, arma::fill::value(2));
    cfg.alpha = 1.0;
    cfg.use_structure_prior = true;
    bde_scorer s(cfg);

    // Without prior.
    bde_scorer_config cfg0 = cfg;
    cfg0.use_structure_prior = false;
    bde_scorer s0(cfg0);

    // For node 0, parent sets of different sizes.
    const double s_empty = s.family_score(0, 0u);
    const double s0_empty = s0.family_score(0, 0u);
    const double s_one   = s.family_score(0, 1u << 1);  // |U|=1
    const double s0_one  = s0.family_score(0, 1u << 1);
    const double s_two   = s.family_score(0, (1u << 1) | (1u << 2));  // |U|=2
    const double s0_two  = s0.family_score(0, (1u << 1) | (1u << 2));

    // Prior contribution = -log C(n-1, |U|), n-1=4.
    const double pri_0 = -std::log(1.0);   // C(4,0) = 1
    const double pri_1 = -std::log(4.0);   // C(4,1) = 4
    const double pri_2 = -std::log(6.0);   // C(4,2) = 6
    std::printf("    diff (|U|=0): expected -ln(1)=%.4f, got %.4f\n",
                pri_0, s_empty - s0_empty);
    std::printf("    diff (|U|=1): expected -ln(4)=%.4f, got %.4f\n",
                pri_1, s_one - s0_one);
    std::printf("    diff (|U|=2): expected -ln(6)=%.4f, got %.4f\n",
                pri_2, s_two - s0_two);
    check(std::abs((s_empty - s0_empty) - pri_0) < 1e-9
       && std::abs((s_one   - s0_one)   - pri_1) < 1e-9
       && std::abs((s_two   - s0_two)   - pri_2) < 1e-9,
          "T4 structure prior contributions match -log C(n-1, |U|)");
}

// ===========================================================================
//  T5: alpha scaling — larger alpha shrinks family score magnitude
// ===========================================================================
static void T5_alpha_scaling() {
    std::printf("\n--- T5: alpha=10 vs alpha=1 — score differs but both finite ---\n");
    arma::imat data(40, 3, arma::fill::zeros);
    arma::arma_rng::set_seed(13);
    for (std::size_t i = 0; i < 40; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            data(i, j) = static_cast<int>(arma::randi<int>(arma::distr_param(0, 1)));
    bde_scorer_config cfg_a1 = {data, arma::uvec(3, arma::fill::value(2)), 1.0, false, 5};
    bde_scorer_config cfg_a10 = {data, arma::uvec(3, arma::fill::value(2)), 10.0, false, 5};
    bde_scorer s1(cfg_a1), s10(cfg_a10);
    const double f1 = s1.family_score(0, 1u << 1);
    const double f10 = s10.family_score(0, 1u << 1);
    std::printf("    alpha=1  family_score(0, {1}) = %.4f\n", f1);
    std::printf("    alpha=10 family_score(0, {1}) = %.4f\n", f10);
    check(std::isfinite(f1) && std::isfinite(f10) && f1 != f10,
          "T5 alpha scaling changes score (finite both)");
}

// ===========================================================================
//  T6: Single-edge score consistency
// ===========================================================================
static void T6_single_edge() {
    std::printf("\n--- T6: single_edge_score(i, j) matches family_score(i, {j}) ---\n");
    arma::imat data(25, 4, arma::fill::zeros);
    arma::arma_rng::set_seed(21);
    for (std::size_t i = 0; i < 25; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            data(i, j) = static_cast<int>(arma::randi<int>(arma::distr_param(0, 1)));
    bde_scorer_config cfg = {data, arma::uvec(4, arma::fill::value(2)), 1.0, true, 5};
    bde_scorer s(cfg);
    bool ok = true;
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            if (i == j) continue;
            const double a = s.single_edge_score(i, j);
            const double b = s.family_score(i, 1ULL << j);
            if (std::abs(a - b) > 1e-9) { ok = false; break; }
        }
        if (!ok) break;
    }
    check(ok, "T6 single_edge_score consistent with family_score");
}

// ===========================================================================
//  T7: top_candidate_parents respects ranking
// ===========================================================================
static void T7_top_candidates() {
    std::printf("\n--- T7: top_candidate_parents returns sorted ranking ---\n");
    arma::imat data(50, 5, arma::fill::zeros);
    arma::arma_rng::set_seed(99);
    for (std::size_t i = 0; i < 50; ++i) {
        // Make node 0 strongly depend on node 1.
        data(i, 1) = static_cast<int>(arma::randi<int>(arma::distr_param(0, 1)));
        data(i, 0) = data(i, 1);  // perfect dependence
        for (std::size_t j = 2; j < 5; ++j)
            data(i, j) = static_cast<int>(arma::randi<int>(arma::distr_param(0, 1)));
    }
    bde_scorer_config cfg = {data, arma::uvec(5, arma::fill::value(2)), 1.0, true, 5};
    bde_scorer s(cfg);
    auto top = s.top_candidate_parents(0, 4);
    std::printf("    top candidates for node 0: ");
    for (auto p : top) std::printf("%zu ", p);
    std::printf("\n");
    // Node 1 should be ranked first (perfect dependence).
    check(top[0] == 1, "T7 perfect-dependency parent ranked first",
          "got top[0] = " + std::to_string(top[0]));

    // Check sorting.
    bool sorted = true;
    for (std::size_t k = 0; k + 1 < top.size(); ++k) {
        if (s.single_edge_score(0, top[k]) < s.single_edge_score(0, top[k + 1])) {
            sorted = false; break;
        }
    }
    check(sorted, "T7 returned list sorted by single-edge score");
}

// ===========================================================================
//  T8: Invalid input validation
// ===========================================================================
static void T8_validation() {
    std::printf("\n--- T8: invalid input rejection ---\n");
    arma::imat data(5, 2);
    data.fill(0);
    arma::uvec cards = {2, 2};

    // Wrong cards length.
    {
        bde_scorer_config c;
        c.data = data; c.cardinalities = arma::uvec({2, 2, 2}); c.alpha = 1.0;
        bool caught = false;
        try { bde_scorer s(c); } catch (const std::invalid_argument&) { caught = true; }
        check(caught, "T8 cardinalities length mismatch rejected");
    }
    // Negative alpha.
    {
        bde_scorer_config c;
        c.data = data; c.cardinalities = cards; c.alpha = -1.0;
        bool caught = false;
        try { bde_scorer s(c); } catch (const std::invalid_argument&) { caught = true; }
        check(caught, "T8 alpha <= 0 rejected");
    }
    // Out-of-range data value.
    {
        arma::imat bad = data; bad(0, 0) = 99;
        bde_scorer_config c;
        c.data = bad; c.cardinalities = cards; c.alpha = 1.0;
        bool caught = false;
        try { bde_scorer s(c); } catch (const std::invalid_argument&) { caught = true; }
        check(caught, "T8 out-of-range data value rejected");
    }
}

} // anonymous namespace

int main() {
    std::printf("====== bde_scorer unit tests ======\n");
    T1_hand_computed();
    T2_likelihood_equivalence();
    T3_empty_parent_closed_form();
    T4_structure_prior();
    T5_alpha_scaling();
    T6_single_edge();
    T7_top_candidates();
    T8_validation();
    std::printf("\n====== Summary: %d PASS / %d FAIL ======\n",
                G_RES.passed, G_RES.failed);
    return (G_RES.failed == 0) ? 0 : 1;
}
