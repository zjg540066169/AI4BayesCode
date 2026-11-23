// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_dp-mixture_dpgamma_wiring.cpp
//
//  The DPGammaMixture wrapper of examples/DPGammaMixture.cpp, as shipped, on
//  the data of its standalone demo: n = 200 draws from two Gamma groups,
//  mean 3 (shape 9, rate 3) and mean 20 (shape 40, rate 2). stick_label_swap
//  rewrites the atom keys "shape" and "rate", so cluster_params must start
//  each update from the shared atoms. The checks use label-invariant
//  summaries of the kept history (draws 300..999 of 1000):
//    - the smallest and the largest observation share a component in < 5%
//      of the draws (the two groups are far apart);
//    - on average < 2% of the observations sit in a component whose mean
//      shape/rate lies on the other side of 10 from their generating group;
//    - the share of observations in components with mean < 10 is within
//      0.03 of the share drawn from the mean-3 group.
// ============================================================================
#define main dpgamma_example_main
#include "../examples/DPGammaMixture.cpp"
#undef main

#include <cmath>
#include <cstdio>
#include <vector>

namespace {
int g_fail = 0;
void check(bool ok, const char* what, double value, double bound) {
    std::printf("  [%s] %-62s %.4f (bound %.4f)\n", ok ? "PASS" : "FAIL", what, value, bound);
    if (!ok) ++g_fail;
}
}  // namespace

int main() {
    std::printf("=== test_fix_dp-mixture_dpgamma_wiring ===\n");
    // The demo's data (examples/DPGammaMixture.cpp, int main()).
    std::mt19937_64 gen(7u);
    const std::size_t n = 200;
    arma::vec y(n);
    std::vector<int> high(n);
    std::bernoulli_distribution grp(0.4);
    std::gamma_distribution<double> g1(9.0, 1.0 / 3.0), g2(40.0, 1.0 / 2.0);
    for (std::size_t i = 0; i < n; ++i) {
        high[i] = grp(gen) ? 1 : 0;
        y[i] = high[i] ? g2(gen) : g1(gen);
    }
    double share_low_true = 0.0;
    for (int h : high) share_low_true += h ? 0.0 : 1.0 / double(n);
    const arma::uword i_min = y.index_min(), i_max = y.index_max();

    DPGammaMixture m(y, 12, 1, true);
    m.step(1000);
    const auto H = m.get_history();
    const arma::mat& Z = H.at("z");
    const arma::mat& SH = H.at("shape");
    const arma::mat& RT = H.at("rate");
    const std::size_t burn = 300, D = Z.n_rows;

    double same_minmax = 0.0, wrong_side = 0.0, share_low = 0.0;
    for (std::size_t d = burn; d < D; ++d) {
        auto comp_mean = [&](std::size_t i) {
            const std::size_t k = std::size_t(std::llround(Z(d, i))) - 1;
            return SH(d, k) / RT(d, k);
        };
        same_minmax += (Z(d, i_min) == Z(d, i_max)) ? 1.0 : 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const bool low = comp_mean(i) < 10.0;
            if (low) share_low += 1.0 / double(n);
            if (low == bool(high[i])) wrong_side += 1.0 / double(n);
        }
    }
    const double kept = double(D - burn);
    same_minmax /= kept; wrong_side /= kept; share_low /= kept;
    std::printf("  kept draws %zu; share of observations from the mean-3 group %.3f\n",
                D - burn, share_low_true);
    check(same_minmax < 0.05, "P(smallest and largest y in one component)", same_minmax, 0.05);
    check(wrong_side < 0.02, "mean share of y in a component on the wrong side of 10", wrong_side, 0.02);
    check(std::fabs(share_low - share_low_true) < 0.03,
          "|share in components with mean < 10 - share of mean-3 group|",
          std::fabs(share_low - share_low_true), 0.03);
    std::printf("\n=== SUMMARY: %d failed ===\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
