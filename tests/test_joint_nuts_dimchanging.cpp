/*================================================================================
 *  joint_nuts_block DIMENSION-CHANGING constraint-type tests (2026-06-17).
 *
 *  Verifies the 6 dimension-changing per-slice kinds wired into joint_nuts_block
 *  via the DUAL-OFFSET scheme (unconstrained slice width != natural slice width):
 *  SIMPLEX, SUM_TO_ZERO, CHOLESKY_CORR, CHOLESKY_FACTOR_COV, CORR_MATRIX,
 *  COV_MATRIX. Transforms live in constraints.hpp; these target the joint-block
 *  wiring: dual nat/unc offsets, metric sizing in UNCONSTRAINED space, the
 *  matrix-type AUTO-DIAGONAL default, and end-to-end recovery.
 *
 *    D1  FD gradient of eval_log_density_unc per dim-changing kind (rel err).
 *    D2  FD all-in-one mixed block REAL+SIMPLEX+COV (dual-offset dispatch).
 *    D3  Round-trip set_current(nat) -> current() == nat (nat<->unc inverse).
 *    D4  SIMPLEX recovery: Dirichlet posterior mean == alpha/alpha0.
 *    D5  COV_MATRIX recovery: MVN N=500, posterior mean Sigma ~ empirical S/M
 *        (auto-diagonal metric; REGRESSION GUARD for the 2026-06-17 metric-dim
 *        crash: dense/diag adaptation sized at total_dim_ vs total_unc_dim_).
 *    D6  Determinism (same seed -> identical sequence) for a dim-changing block.
 *    D7  Constructor guards: non-PD cov / off-manifold simplex / sum!=0 throw.
 *================================================================================*/

#include "AI4BayesCode/joint_nuts_block.hpp"
#include "AI4BayesCode/constraints.hpp"

#include <armadillo>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
using namespace AI4BayesCode;

struct test_result { int passed = 0; int failed = 0; };
test_result G_RES;
static void check(bool ok, const std::string& tag, const std::string& detail = "") {
    if (ok) { ++G_RES.passed; std::printf("  PASS  %s\n", tag.c_str()); }
    else    { ++G_RES.failed; std::printf("  FAIL  %s  %s\n", tag.c_str(), detail.c_str()); }
}

// MVN log-lik on flat col-major KxK Sigma; grad = -0.5 M P + 0.5 P S P.
static double mvn_on_sigma(const arma::vec& Sf, int K, const arma::mat& S, int M,
                           arma::mat* G, double* lp) {
    arma::mat Sig(K, K);
    for (int j = 0; j < K; ++j) for (int i = 0; i < K; ++i) Sig(i, j) = Sf[i + j * K];
    Sig = 0.5 * (Sig + Sig.t());
    arma::mat P;
    if (!arma::inv_sympd(P, Sig)) { *lp = -1e300; if (G) G->zeros(K, K); return *lp; }
    double ld, sg; arma::log_det(ld, sg, Sig);
    *lp = -0.5 * M * ld - 0.5 * arma::trace(P * S);
    if (G) *G = -0.5 * M * P + 0.5 * P * S * P;
    return *lp;
}

// FD-check eval_log_density_unc gradient at a random unconstrained point.
static double fd_relerr(joint_nuts_block& blk, const arma::vec& y) {
    arma::vec ga; blk.eval_log_density_unc(y, &ga);
    double maxe = 0, h = 1e-6;
    for (std::size_t k = 0; k < y.n_elem; ++k) {
        arma::vec yp = y; yp[k] += h; double lpp = blk.eval_log_density_unc(yp, nullptr);
        arma::vec ym = y; ym[k] -= h; double lpm = blk.eval_log_density_unc(ym, nullptr);
        maxe = std::max(maxe, std::abs((lpp - lpm) / (2 * h) - ga[k]));
    }
    return maxe / (arma::abs(ga).max() + 1e-12);
}

static auto noop = [](const arma::vec&, const block_context&, arma::vec* g) -> double {
    if (g) g->zeros(); return 0.0;
};

// ---- D1: FD per dim-changing kind --------------------------------------------
void D1() {
    std::printf("\n[D1] FD gradient per dim-changing kind\n");
    std::mt19937_64 g(1); std::normal_distribution<double> nz(0, 1);
    int K = 3, M = 400;
    arma::mat A(K, K); for (int i = 0; i < K; ++i) for (int j = 0; j < K; ++j) A(i, j) = nz(g);
    arma::mat St = A * A.t() / K + arma::eye(K, K), Lt = arma::chol(St, "lower"), S(K, K, arma::fill::zeros);
    for (int n = 0; n < M; ++n) { arma::vec z(K); for (int i = 0; i < K; ++i) z[i] = nz(g); arma::vec y = Lt * z; S += y * y.t(); }

    struct Case { const char* name; joint_constraint c; std::size_t nat; std::size_t ud;
                  std::function<double(const arma::vec&, arma::vec*)> in; };
    auto cov_in = [&](const arma::vec& Sf, arma::vec* gr) { arma::mat G; double l; mvn_on_sigma(Sf, K, S, M, gr ? &G : nullptr, &l); if (gr) *gr = arma::vectorise(G); return l; };
    auto chf_in = [&](const arma::vec& Lf, arma::vec* gr) { arma::mat L(K, K); for (int j = 0; j < K; ++j) for (int i = 0; i < K; ++i) L(i, j) = Lf[i + j * K]; arma::vec Sf = arma::vectorise(L * L.t()); arma::mat G; double l; mvn_on_sigma(Sf, K, S, M, &G, &l); if (gr) { arma::mat dL = (G + G.t()) * L; *gr = arma::vectorise(dL); } return l; };
    auto crr_in = [&](const arma::vec& Rf, arma::vec* gr) { arma::mat G; double l; mvn_on_sigma(Rf, K, S, M, gr ? &G : nullptr, &l); if (gr) *gr = arma::vectorise(G); return l; };
    arma::vec alpha = {2, 3, 4}, tvec = {0.5, -0.3, -0.2};
    auto simplex_in = [&](const arma::vec& x, arma::vec* gr) { double v = 0; arma::vec gg(3); for (int i = 0; i < 3; ++i) { double xi = std::max(x[i], 1e-12); v += (alpha[i] - 1) * std::log(xi); gg[i] = (alpha[i] - 1) / xi; } if (gr) *gr = gg; return v; };
    auto sumzero_in = [&](const arma::vec& x, arma::vec* gr) { double v = 0; arma::vec gg(3); for (int i = 0; i < 3; ++i) { double d = x[i] - tvec[i]; v += -0.5 * d * d; gg[i] = -d; } if (gr) *gr = gg; return v; };

    std::vector<Case> cases = {
        {"SIMPLEX",            joint_constraint::SIMPLEX,             3, 2, simplex_in},
        {"SUM_TO_ZERO",        joint_constraint::SUM_TO_ZERO,         3, 2, sumzero_in},
        {"CHOLESKY_CORR",      joint_constraint::CHOLESKY_CORR,       9, 3, crr_in},
        {"CHOLESKY_FACTOR_COV",joint_constraint::CHOLESKY_FACTOR_COV, 9, 6, chf_in},
        {"CORR_MATRIX",        joint_constraint::CORR_MATRIX,         9, 3, crr_in},
        {"COV_MATRIX",         joint_constraint::COV_MATRIX,          9, 6, cov_in},
    };
    for (auto& c : cases) {
        joint_nuts_block_config cfg; cfg.name = c.name;
        joint_nuts_sub_param sp; sp.name = "x"; sp.dim = c.nat; sp.constraint = c.c; cfg.sub_params = {sp};
        cfg.log_density_grad = [&c](const arma::vec& z, const block_context&, arma::vec* gr) { return c.in(z, gr); };
        std::mt19937_64 gi(11); std::normal_distribution<double> z(0, 0.4); arma::vec u(c.ud); for (auto& v : u) v = z(gi);
        if (c.c == joint_constraint::SIMPLEX) cfg.initial_cat = constraints::simplex::constrain(u);
        else if (c.c == joint_constraint::SUM_TO_ZERO) cfg.initial_cat = constraints::sum_to_zero::constrain(u);
        else if (c.c == joint_constraint::CHOLESKY_CORR) cfg.initial_cat = constraints::cholesky_corr::constrain(u);
        else if (c.c == joint_constraint::CHOLESKY_FACTOR_COV) cfg.initial_cat = constraints::cholesky_factor_cov::constrain(u);
        else if (c.c == joint_constraint::CORR_MATRIX) cfg.initial_cat = constraints::corr_matrix::constrain(u);
        else cfg.initial_cat = constraints::cov_matrix::constrain(u);
        joint_nuts_block blk(cfg); block_context ctx; blk.set_context(ctx);
        std::mt19937_64 gy(7); std::normal_distribution<double> zy(0, 0.5); arma::vec y(c.ud); for (auto& v : y) v = zy(gy);
        double rel = fd_relerr(blk, y);
        check(rel < 1e-5, std::string("FD ") + c.name, "relErr=" + std::to_string(rel));
    }
}

// ---- D2: FD all-in-one mixed block REAL+SIMPLEX+COV --------------------------
void D2() {
    std::printf("\n[D2] FD mixed REAL(2)+SIMPLEX(3)+COV(2x2) — dual-offset dispatch\n");
    int K = 2, M = 300; std::mt19937_64 g(2); std::normal_distribution<double> nz(0, 1);
    arma::mat St = {{1.0, 0.4}, {0.4, 1.3}}, Lt = arma::chol(St, "lower"), S(K, K, arma::fill::zeros);
    for (int n = 0; n < M; ++n) { arma::vec z(K); for (int i = 0; i < K; ++i) z[i] = nz(g); arma::vec y = Lt * z; S += y * y.t(); }
    arma::vec btar = {0.6, -1.1}, alpha = {2, 3, 4};
    auto lp = [&](const arma::vec& th, const block_context&, arma::vec* gr) -> double {
        arma::vec beta = th.subvec(0, 1), x = th.subvec(2, 4), Sf = th.subvec(5, 8);
        double v = 0; arma::vec gb(2), gx(3);
        for (int i = 0; i < 2; ++i) { double d = beta[i] - btar[i]; v += -0.5 * d * d; gb[i] = -d; }
        for (int i = 0; i < 3; ++i) { double xi = std::max(x[i], 1e-12); v += (alpha[i] - 1) * std::log(xi); gx[i] = (alpha[i] - 1) / xi; }
        arma::mat G; double ls; mvn_on_sigma(Sf, K, S, M, gr ? &G : nullptr, &ls); v += ls;
        if (gr) { gr->set_size(9); (*gr)[0] = gb[0]; (*gr)[1] = gb[1]; for (int i = 0; i < 3; ++i) (*gr)[2 + i] = gx[i]; arma::vec gs = arma::vectorise(G); for (int k = 0; k < 4; ++k) (*gr)[5 + k] = gs[k]; }
        return v;
    };
    joint_nuts_block_config cfg; cfg.name = "mix";
    cfg.sub_params = {{"beta", 2, joint_constraint::REAL}, {"w", 3, joint_constraint::SIMPLEX}, {"S", 4, joint_constraint::COV_MATRIX}};
    cfg.log_density_grad = [&lp](const arma::vec& z, const block_context& c, arma::vec* gr) { return lp(z, c, gr); };
    arma::vec init(9); init[0] = 0.1; init[1] = -0.1;
    arma::vec sx = constraints::simplex::constrain(arma::vec({0.2, -0.1})); for (int k = 0; k < 3; ++k) init[2 + k] = sx[k];
    arma::vec Sf = constraints::cov_matrix::constrain(arma::vec({0.1, 0.2, 0.0})); for (int k = 0; k < 4; ++k) init[5 + k] = Sf[k];
    cfg.initial_cat = init;
    joint_nuts_block blk(cfg); block_context ctx; blk.set_context(ctx);
    std::mt19937_64 gy(3); std::normal_distribution<double> zy(0, 0.4); arma::vec y(7); for (auto& v : y) v = zy(gy);  // unc dim = 2+2+3 = 7
    double rel = fd_relerr(blk, y);
    check(rel < 1e-5, "FD mixed REAL+SIMPLEX+COV", "relErr=" + std::to_string(rel));
}

// ---- D3: round-trip set_current(nat) -> current() == nat ---------------------
void D3() {
    std::printf("\n[D3] round-trip set_current(nat) -> current() == nat\n");
    struct RT { const char* name; joint_constraint c; std::size_t nat; std::size_t ud; };
    std::vector<RT> rts = {
        {"SIMPLEX", joint_constraint::SIMPLEX, 4, 3},
        {"SUM_TO_ZERO", joint_constraint::SUM_TO_ZERO, 4, 3},
        {"CHOLESKY_CORR", joint_constraint::CHOLESKY_CORR, 9, 3},
        {"COV_MATRIX", joint_constraint::COV_MATRIX, 9, 6},
        {"CORR_MATRIX", joint_constraint::CORR_MATRIX, 9, 3},
    };
    for (auto& r : rts) {
        joint_nuts_block_config cfg; cfg.name = r.name;
        joint_nuts_sub_param sp; sp.name = "x"; sp.dim = r.nat; sp.constraint = r.c; cfg.sub_params = {sp};
        cfg.log_density_grad = noop;
        std::mt19937_64 gi(20); std::normal_distribution<double> z(0, 0.5); arma::vec u(r.ud); for (auto& v : u) v = z(gi);
        arma::vec nat;
        if (r.c == joint_constraint::SIMPLEX) nat = constraints::simplex::constrain(u);
        else if (r.c == joint_constraint::SUM_TO_ZERO) nat = constraints::sum_to_zero::constrain(u);
        else if (r.c == joint_constraint::CHOLESKY_CORR) nat = constraints::cholesky_corr::constrain(u);
        else if (r.c == joint_constraint::COV_MATRIX) nat = constraints::cov_matrix::constrain(u);
        else nat = constraints::corr_matrix::constrain(u);
        cfg.initial_cat = nat;
        joint_nuts_block blk(cfg); block_context ctx; blk.set_context(ctx);
        blk.set_current(nat);
        double err = arma::abs(blk.current() - nat).max();
        check(err < 1e-9, std::string("round-trip ") + r.name, "maxErr=" + std::to_string(err));
    }
}

// ---- D4: SIMPLEX Dirichlet recovery ------------------------------------------
void D4() {
    std::printf("\n[D4] SIMPLEX recovery: Dirichlet posterior mean\n");
    std::size_t K = 4; arma::vec a = {6, 12, 4, 18}; double a0 = arma::accu(a);
    joint_nuts_block_config cfg; cfg.name = "dir";
    cfg.sub_params = {{"x", K, joint_constraint::SIMPLEX}};
    cfg.log_density_grad = [a, K](const arma::vec& x, const block_context&, arma::vec* gr) -> double {
        double v = 0; arma::vec gg(K); for (std::size_t i = 0; i < K; ++i) { double xi = std::max(x[i], 1e-12); v += (a[i] - 1) * std::log(xi); gg[i] = (a[i] - 1) / xi; } if (gr) *gr = gg; return v; };
    cfg.initial_cat = constraints::simplex::constrain(arma::vec({0.0, 0.0, 0.0}));
    joint_nuts_block blk(cfg); block_context ctx; blk.set_context(ctx); std::mt19937_64 rng(42);
    arma::vec mean(K, arma::fill::zeros); int keep = 6000;
    for (int s = 0; s < 1500; ++s) blk.step(rng);
    for (int s = 0; s < keep; ++s) { blk.step(rng); mean += blk.current(); }
    mean /= keep;
    double err = arma::abs(mean - a / a0).max();
    check(err < 0.01, "SIMPLEX Dirichlet recovery", "maxErr=" + std::to_string(err));
}

// ---- D5: COV_MATRIX recovery (auto-diagonal; metric-crash regression guard) --
void D5() {
    std::printf("\n[D5] COV_MATRIX recovery MVN N=500 (auto-diagonal metric)\n");
    int K = 3, M = 500; std::mt19937_64 g(2024); std::normal_distribution<double> nz(0, 1);
    arma::mat St = {{1.0, 0.5, -0.3}, {0.5, 2.0, 0.4}, {-0.3, 0.4, 0.8}}, Lt = arma::chol(St, "lower"), S(K, K, arma::fill::zeros);
    for (int n = 0; n < M; ++n) { arma::vec z(K); for (int i = 0; i < K; ++i) z[i] = nz(g); arma::vec y = Lt * z; S += y * y.t(); }
    arma::vec Semp = arma::vectorise(S / M);
    joint_nuts_block_config cfg; cfg.name = "cov";
    cfg.sub_params = {{"S", 9, joint_constraint::COV_MATRIX}};   // NO metric flag -> auto-diagonal must fire
    cfg.log_density_grad = [K, &S, M](const arma::vec& Sf, const block_context&, arma::vec* gr) -> double {
        arma::mat G; double l; mvn_on_sigma(Sf, K, S, M, gr ? &G : nullptr, &l); if (gr) *gr = arma::vectorise(G); return l; };
    cfg.initial_cat = constraints::cov_matrix::constrain(arma::vec(6, arma::fill::zeros));
    joint_nuts_block blk(cfg); block_context ctx; blk.set_context(ctx); std::mt19937_64 rng(101);
    arma::vec mean(9, arma::fill::zeros); int keep = 3000;
    for (int s = 0; s < 1000; ++s) blk.step(rng);
    for (int s = 0; s < keep; ++s) { blk.step(rng); mean += blk.current(); }
    mean /= keep;
    double err = arma::abs(mean - Semp).max();
    // Also assert the chain MOVED (regression vs the identity-metric freeze).
    double moved = arma::abs(blk.current() - cfg.initial_cat).max();
    check(err < 0.08 && moved > 1e-6, "COV_MATRIX recovery + auto-diagonal moved",
          "maxErr=" + std::to_string(err) + " moved=" + std::to_string(moved));
}

// ---- D6: determinism ---------------------------------------------------------
void D6() {
    std::printf("\n[D6] determinism (same seed -> identical)\n");
    int K = 2, M = 300; std::mt19937_64 g(5); std::normal_distribution<double> nz(0, 1);
    arma::mat St = {{1.0, 0.4}, {0.4, 1.2}}, Lt = arma::chol(St, "lower"), S(K, K, arma::fill::zeros);
    for (int n = 0; n < M; ++n) { arma::vec z(K); for (int i = 0; i < K; ++i) z[i] = nz(g); arma::vec y = Lt * z; S += y * y.t(); }
    auto lp = [&](const arma::vec& Sf, const block_context&, arma::vec* gr) -> double { arma::mat G; double l; mvn_on_sigma(Sf, K, S, M, gr ? &G : nullptr, &l); if (gr) *gr = arma::vectorise(G); return l; };
    auto run = [&]() {
        joint_nuts_block_config cfg; cfg.name = "d";
        cfg.sub_params = {{"S", 4, joint_constraint::COV_MATRIX}};
        cfg.log_density_grad = [&lp](const arma::vec& z, const block_context& c, arma::vec* gr) { return lp(z, c, gr); };
        cfg.initial_cat = constraints::cov_matrix::constrain(arma::vec(3, arma::fill::zeros));
        joint_nuts_block blk(cfg); block_context ctx; blk.set_context(ctx); std::mt19937_64 rng(12345);
        arma::mat out(150, 4); for (int s = 0; s < 200; ++s) blk.step(rng);
        for (int s = 0; s < 150; ++s) { blk.step(rng); out.row(s) = blk.current().t(); } return out;
    };
    arma::mat a = run(), b = run();
    double md = arma::abs(a - b).max();
    check(md == 0.0, "determinism bit-identical", "maxDiff=" + std::to_string(md));
}

// ---- D7: constructor guards --------------------------------------------------
void D7() {
    std::printf("\n[D7] constructor guards (bad inits throw)\n");
    auto throws = [](joint_constraint c, std::size_t nat, const arma::vec& bad) -> bool {
        joint_nuts_block_config cfg; cfg.name = "g";
        joint_nuts_sub_param sp; sp.name = "x"; sp.dim = nat; sp.constraint = c; cfg.sub_params = {sp};
        cfg.log_density_grad = noop; cfg.initial_cat = bad;
        try { joint_nuts_block blk(cfg); } catch (const std::exception&) { return true; } return false;
    };
    check(throws(joint_constraint::COV_MATRIX, 4, arma::vec({1, 2, 2, 1})), "non-PD cov throws");
    check(throws(joint_constraint::SIMPLEX, 3, arma::vec({0.5, 0.5, 0.5})), "off-manifold simplex throws");
    check(throws(joint_constraint::SUM_TO_ZERO, 3, arma::vec({1, 1, 1})), "sum!=0 throws");
}

}  // namespace

int main() {
    std::printf("=== joint_nuts_block dimension-changing constraint tests ===\n");
    D1(); D2(); D3(); D4(); D5(); D6(); D7();
    std::printf("\n====== Summary: %d PASS / %d FAIL ======\n", G_RES.passed, G_RES.failed);
    return G_RES.failed > 0 ? 1 : 0;
}
