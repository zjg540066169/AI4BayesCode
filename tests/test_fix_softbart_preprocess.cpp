// test_fix_softbart_preprocess: softbart_block normalises its inputs as
// SoftBart does (X mapped to [0, 1] by the ranks of the distinct training
// values, Y rescaled by a fixed y_scale) and returns f, sigma and
// predictions on the scale of Y.
//
//  A. x_to_unit() of the training X equals SoftBart's rank transform
//     (trank: (rank among the distinct values - 1) / (m - 1)) to 1e-15, with ties;
//     a constant column maps to 0.5; new X between, below and above the
//     training values maps linearly, to 0 and to 1.
//  B. A strictly increasing transform of one column of X (x -> x^3 + 2x)
//     leaves every draw of f, sigma and the trees unchanged bit for bit.
//  C. Y and the context sigma multiplied by 4 (center_Y): every f and sigma
//     is 4 times that of the original chain bit for bit, and the trees are
//     the same.
//  D. predict() at the training X equals current() after every sweep.
//  E. The default sigma_hat (sigma_key empty: the first sweep starts from
//     it) is sqrt(RSS / n) of the least-squares fit of y on [1, X].
// B fails when the forest is given X unchanged (the split points are drawn
// in [0, 1]); C and E fail when y_scale or sigma_hat are not applied.

#include "AI4BayesCode/softbart_block.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

using AI4BayesCode::block_context;
using AI4BayesCode::softbart_block;
using AI4BayesCode::softbart_block_config;

static int g_fail = 0;
static void check(bool ok, const std::string& what) {
    std::printf("  %-96s %s\n", what.c_str(), ok ? "PASS" : "FAIL");
    if (!ok) ++g_fail;
}

static void make_data(std::size_t n, arma::mat& X, arma::vec& y, std::uint64_t seed) {
    std::mt19937_64 g(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::normal_distribution<double> z(0.0, 0.5);
    X.set_size(n, 3); y.set_size(n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < 3; ++j) X(i, j) = u(g);
        if (i % 7 == 0) X(i, 1) = X(0, 1);                  // ties in column 1
        y[i] = std::sin(3.0 * X(i, 0)) + 0.5 * X(i, 1) * X(i, 1) - X(i, 2) + z(g);
    }
}

// SoftBart's trank of one column.
static arma::vec trank(const arma::vec& x) {
    std::vector<double> u(x.begin(), x.end());
    std::sort(u.begin(), u.end());
    u.erase(std::unique(u.begin(), u.end()), u.end());
    arma::vec t(x.n_elem);
    for (arma::uword i = 0; i < x.n_elem; ++i) {
        const double r = static_cast<double>(std::lower_bound(u.begin(), u.end(), x[i]) - u.begin());
        t[i] = (r / u.size()) / ((u.size() - 1.0) / u.size());
    }
    return t;
}

struct Run { std::vector<arma::vec> f; std::vector<double> sigma; std::vector<std::string> trees; int predict_mismatch = 0; };

static Run run(const arma::mat& X, const arma::vec& y, double sigma_mult, bool routed_sigma, int sweeps) {
    bart_rng::set_seed(7);
    softbart_block_config cfg;
    cfg.x_train = X; cfg.y_init = y; cfg.ntrees = 20;
    cfg.sigma_key = routed_sigma ? "sigma" : "";
    softbart_block b(cfg);
    Run out;
    std::mt19937_64 unused(1);
    for (int s = 0; s < sweeps; ++s) {
        block_context ctx;
        ctx["softbart_target"] = y;
        if (routed_sigma) ctx["sigma"] = arma::vec{sigma_mult * (0.4 + 0.02 * (s % 5))};
        b.set_context(ctx);
        b.step(unused);
        out.f.push_back(b.current());
        out.sigma.push_back(b.current_sigma());
        out.trees.push_back(b.get_tree());
        if (!arma::all(b.predict(X) == b.current())) ++out.predict_mismatch;
    }
    return out;
}

int main() {
    std::printf("=== test_fix_softbart_preprocess ===\n");
    arma::mat X; arma::vec y; make_data(300, X, y, 11);

    std::printf("A. map of X to [0, 1]\n");
    {
        arma::mat Xc = X; Xc.col(2).fill(1.25);
        softbart_block_config cfg; cfg.x_train = Xc; cfg.y_init = y; cfg.ntrees = 5;
        softbart_block b(cfg);
        const arma::mat T = b.x_to_unit(Xc);
        const double e0 = arma::abs(T.col(0) - trank(Xc.col(0))).max(), e1 = arma::abs(T.col(1) - trank(Xc.col(1))).max();
        check(e0 <= 1e-15 && e1 <= 1e-15,
              "training columns equal SoftBart's trank to 1e-15 (column 1 has ties)");
        check(arma::all(T.col(2) == 0.5), "a constant column maps to 0.5");
        const double lo = Xc.col(0).min(), hi = Xc.col(0).max();
        arma::mat Xn(3, 3, arma::fill::zeros);
        Xn(0, 0) = lo - 1.0; Xn(1, 0) = hi + 1.0;
        std::vector<double> u(Xc.colptr(0), Xc.colptr(0) + Xc.n_rows);
        std::sort(u.begin(), u.end());
        Xn(2, 0) = 0.5 * (u[10] + u[11]);
        const arma::mat Tn = b.x_to_unit(Xn);
        const double want = (10.0 + (Xn(2, 0) - u[10]) / (u[11] - u[10])) / (u.size() - 1.0);
        check(Tn(0, 0) == 0.0 && Tn(1, 0) == 1.0 && std::fabs(Tn(2, 0) - want) < 1e-15,
              "new X: 0 below, 1 above, linear between the training values");
    }

    std::printf("B. strictly increasing transform of a column\n");
    {
        arma::mat Xt = X;
        for (arma::uword i = 0; i < X.n_rows; ++i) Xt(i, 0) = X(i, 0) * X(i, 0) * X(i, 0) + 2.0 * X(i, 0);
        const Run a = run(X, y, 1.0, true, 40), b = run(Xt, y, 1.0, true, 40);
        int diff = 0;
        for (std::size_t s = 0; s < a.f.size(); ++s)
            if (!arma::all(a.f[s] == b.f[s]) || a.sigma[s] != b.sigma[s] || a.trees[s] != b.trees[s]) ++diff;
        check(diff == 0, "x -> x^3 + 2x in column 0: f, sigma and trees equal in all 40 sweeps (" +
                         std::to_string(diff) + " differ)");
    }

    std::printf("C. Y and sigma multiplied by 4\n");
    {
        const Run a = run(X, y, 1.0, true, 40), b = run(X, arma::vec(4.0 * y), 4.0, true, 40);
        int diff = 0;
        for (std::size_t s = 0; s < a.f.size(); ++s)
            if (!arma::all(4.0 * a.f[s] == b.f[s]) || 4.0 * a.sigma[s] != b.sigma[s] || a.trees[s] != b.trees[s]) ++diff;
        check(diff == 0, "f and sigma 4 times the original, same trees, in all 40 sweeps (" +
                         std::to_string(diff) + " differ)");
    }

    std::printf("D. predict() at the training X\n");
    {
        const Run a = run(X, y, 1.0, false, 30);
        check(a.predict_mismatch == 0, "predict(x_train) equals current() after every sweep (" +
                                       std::to_string(a.predict_mismatch) + " differ)");
    }

    std::printf("E. default sigma_hat\n");
    {
        arma::mat A = arma::join_rows(arma::ones<arma::vec>(X.n_rows), X);
        const arma::vec beta = arma::solve(A, y);
        const double rmse = std::sqrt(arma::accu(arma::square(y - A * beta)) / X.n_rows);
        softbart_block_config cfg; cfg.x_train = X; cfg.y_init = y; cfg.ntrees = 5; cfg.sigma_key = "";
        softbart_block b(cfg);
        char msg[160];
        std::snprintf(msg, sizeof msg, "current_sigma() before the first sweep %.12g, sqrt(RSS / n) %.12g",
                      b.current_sigma(), rmse);
        check(std::fabs(b.current_sigma() - rmse) < 1e-10 * rmse, msg);
    }

    std::printf("%s\n", g_fail ? "SOME FAILED" : "ALL PASS");
    return g_fail ? 1 : 0;
}
