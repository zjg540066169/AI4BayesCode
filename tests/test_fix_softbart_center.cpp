// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_softbart_center.cpp
//
//  softbart_block fits Z = (Y - m) / y_scale and returns f = m + y_scale f_Z
//  (f_Z the fit of the forest): with center_Y, m is the mean of the working
//  response passed last; without it, the midrange of y_init.
//
//  A. The composite holds a block "c" that draws a shift c ~ N(0, 1.5^2)
//     every sweep and a softbart_block "f" (internal sigma) whose working
//     response is y - c, so m changes from sweep to sweep. K sweeps with
//     keep_history and keep_tree:
//      1. after every sweep, current() = m + y_scale * (the kernel's fit at
//         the training X) bit for bit, m the mean of that sweep's working
//         response;
//      2. the kernel stores one m per stored forest, equal bit for bit to the
//         mean of the working response of that sweep; the values span > 1;
//      3. row d of predict_history(X_new) = m_d + y_scale * (forest d, parsed
//         here from the stored string, at x_to_unit(X_new)), bit for bit;
//      4. row d = predict(X_new) computed right after sweep d (to 1e-12);
//      5. at the training X, row d = fit row d of get_history() (to 1e-12).
//  B. y on a grid of multiples of 2^-10 with N = 64, and y + 3 (both exact,
//     so Z is the same bit for bit; sigma_hat set): in every sweep the trees
//     and sigma are the same, and every fit row and every row of
//     predict_history(X_new) is 3 larger (to 1e-12); the mean of the fit
//     rows is within 0.25 of mean(y + 3).
//  C. center_Y = false: every stored m is the midrange of y_init and
//     current() = midrange + y_scale * (the kernel's fit), bit for bit.
//  D. set_Y and set_data between sweeps: m becomes the mean of the loaded y
//     (kernel center() and current() = m + y_scale * kernel fit, bit for
//     bit), and the next stored m is the mean of the next working response.
//  E. The clear_history() check of clear_history_contract.hpp on the model
//     of A, with the stored m among the per-draw buffers.
// ============================================================================
#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif
#include <armadillo>
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/softbart_block.hpp"

#include "clear_history_contract.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

namespace ab  = AI4BayesCode;
namespace chk = clear_history_check;

constexpr int K = 40, K1 = 40, K2 = 30;

// A scalar block "c" drawn from N(0, 1.5^2) at every step, independently of
// the context.
class shift_block : public ab::block_sampler {
public:
    void set_context(const ab::block_context&) override {}
    void step(std::mt19937_64& rng) override {
        std::normal_distribution<double> nd(0.0, 1.5);
        c_[0] = nd(rng);
        if (keep_history_) hist_.push_back(c_);
    }
    const arma::vec& current() const override { return c_; }
    void set_current(const arma::vec& v) override { c_ = v; }
    const std::string& name() const noexcept override { return name_; }
    std::size_t dim() const noexcept override { return 1; }
    ab::history_map get_history() const override {
        return ab::detail::make_history_map(name_, hist_, c_);
    }
    std::size_t history_size() const noexcept override {
        return hist_.empty() ? 1 : hist_.size();
    }
    void clear_history() override { hist_.clear(); }
    void record_held_history() override { if (keep_history_) hist_.push_back(c_); }

private:
    std::string name_ = "c";
    arma::vec c_ = arma::vec{0.0};
    std::vector<arma::vec> hist_;
};

struct data_t {
    arma::mat X, X_new;
    arma::vec y;
};

data_t make_data() {
    const std::size_t N = 60, p = 2, N_new = 15;
    std::mt19937_64 g(211);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::normal_distribution<double> nd(0.0, 0.3);
    data_t d;
    d.X.set_size(N, p);
    d.y.set_size(N);
    for (std::size_t i = 0; i < N; ++i) {
        d.X(i, 0) = U(g); d.X(i, 1) = U(g);
        d.y[i] = 2.0 + std::sin(4.0 * d.X(i, 0)) + d.X(i, 1) + nd(g);
    }
    d.X_new.set_size(N_new, p);
    for (std::size_t i = 0; i < N_new; ++i) { d.X_new(i, 0) = U(g); d.X_new(i, 1) = U(g); }
    return d;
}

// The composite of part A, keep_history and keep_tree on; the kernel of the
// softbart_block draws from the thread-local engine bart_rng, seeded here.
std::shared_ptr<ab::composite_block> make_root(const data_t& d, ab::softbart_block** sb) {
    bart_rng::set_seed(223);
    auto root = std::make_shared<ab::composite_block>("root");
    root->data().set("y", d.y);
    root->data().set("c", arma::vec{0.0});
    root->data().set("softbart_target", d.y);
    root->data().register_refresher(
        "softbart_target", [](const ab::shared_data_t& s) -> arma::vec {
            return s.get("y") - s.get("c")[0];
        });
    root->data().declare_invalidates("c", {"softbart_target"});
    root->data().declare_dependencies("f", {"softbart_target"});
    root->add_child(std::make_unique<shift_block>());
    ab::softbart_block_config c;
    c.name = "f"; c.x_train = d.X; c.y_init = d.y;
    c.working_response_key = "softbart_target"; c.sigma_key = "";
    c.ntrees = 10;
    auto b = std::make_unique<ab::softbart_block>(std::move(c));
    *sb = b.get();
    root->add_child(std::move(b));
    root->set_keep_history(true);
    root->set_keep_tree(true);
    return root;
}

bool same_bits(const arma::vec& a, const arma::vec& b) {
    return chk::same_bits(arma::mat(a), arma::mat(b));
}

// m + y_scale * (forest string s at x_to_unit(X)), as the block computes it.
arma::vec eval_forest(const ab::softbart_block& sb, const std::string& s,
                      double m, const arma::mat& X) {
    std::vector<::Node*> trees = softbart::deserialize_forest_from_str(s);
    const arma::vec fz = softbart::predict_softbart_forest_(trees, sb.x_to_unit(X));
    softbart::free_forest_(trees);
    return arma::vec(m + sb.y_scale() * fz);
}

// current() = m + y_scale * (the kernel's fit at the training X: the fit kept
// by the last sweep, or the forest evaluated at the mapped X), bit for bit.
bool fit_is_m_plus_kernel_fit(const ab::softbart_block& sb, double m, const arma::mat& X) {
    const arma::vec* kept = sb.underlying()->train_fit();
    const arma::vec fz = kept ? *kept : sb.underlying()->predict(sb.x_to_unit(X));
    return same_bits(sb.current(), arma::vec(m + sb.y_scale() * fz));
}

void partA() {
    std::printf("A. softbart_block, working response y - c with c drawn every sweep (K = %d)\n", K);
    const data_t d = make_data();
    ab::softbart_block* sb = nullptr;
    auto root = make_root(d, &sb);
    std::mt19937_64 rng(227);
    std::vector<double> wr_mean(K);
    std::vector<arma::vec> live(K);
    bool fit_ok = true;
    for (int s = 0; s < K; ++s) {
        root->step(rng);
        wr_mean[s] = arma::mean(root->data().get("softbart_target"));
        live[s] = sb->predict(d.X_new);
        fit_ok = fit_ok && sb->underlying()->train_fit() != nullptr &&
                 fit_is_m_plus_kernel_fit(*sb, wr_mean[s], d.X) &&
                 sb->underlying()->center() == wr_mean[s];
    }
    chk::check(fit_ok, "1. after every sweep current() = m + y_scale * kernel fit, "
                       "m the mean of the working response");
    const softbart::SoftbartHistory h = sb->get_full_history();

    bool ok = h.tree_history.size() == static_cast<std::size_t>(K) &&
              h.center_history.n_elem == static_cast<arma::uword>(K);
    for (int s = 0; ok && s < K; ++s)
        ok = same_bits(arma::vec{h.center_history[s]}, arma::vec{wr_mean[s]});
    chk::check(ok, "2. kernel center_history: " + std::to_string(h.center_history.n_elem) +
                   " values, each the mean of the working response of its sweep");
    const auto mm = std::minmax_element(wr_mean.begin(), wr_mean.end());
    char buf[200];
    std::snprintf(buf, sizeof buf, "   the values of m span %.3f (> 1) across the sweeps",
                  *mm.second - *mm.first);
    chk::check(*mm.second - *mm.first > 1.0, buf);

    const arma::mat H = sb->predict_history(d.X_new);
    ok = H.n_rows == static_cast<arma::uword>(K) && H.n_cols == d.X_new.n_rows;
    for (int s = 0; ok && s < K; ++s)
        ok = same_bits(arma::vec(H.row(s).t()),
                       eval_forest(*sb, h.tree_history[s], wr_mean[s], d.X_new));
    chk::check(ok, "3. predict_history(X_new) " + chk::shape(H) +
                   ": row d = m_d + y_scale * forest d at X_new, bit for bit");

    double gap = 0.0;
    for (int s = 0; s < K && H.n_rows == static_cast<arma::uword>(K); ++s)
        gap = std::max(gap, arma::abs(arma::vec(H.row(s).t()) - live[s]).max());
    std::snprintf(buf, sizeof buf, "4. row d = predict(X_new) computed right after sweep d "
                  "(max |difference| %.3g < 1e-12)", gap);
    chk::check(H.n_rows == static_cast<arma::uword>(K) && gap < 1e-12, buf);

    const arma::mat Ht = sb->predict_history(d.X);
    const ab::history_map hist = root->get_history();
    const arma::mat& F = hist.at("f");
    const bool same_shape = Ht.n_rows == F.n_rows && Ht.n_cols == F.n_cols;
    const double gap_t = same_shape ? arma::abs(Ht - F).max() : -1.0;
    std::snprintf(buf, sizeof buf, "5. predict_history(X_train) = get_history() fit rows "
                  "(max |difference| %.3g < 1e-12)", gap_t);
    chk::check(same_shape && gap_t >= 0.0 && gap_t < 1e-12, buf);
}

// One block driven by set_context + step, keep_history and keep_tree on.
struct block_run {
    std::vector<arma::vec> fit;
    std::vector<double> sigma;
    std::vector<std::string> trees;
    arma::mat H;
};

block_run run_block(const arma::mat& X, const arma::vec& y, const arma::mat& X_new,
                    bool center, int sweeps) {
    bart_rng::set_seed(229);
    ab::softbart_block_config c;
    c.x_train = X; c.y_init = y; c.sigma_key = ""; c.ntrees = 10;
    c.sigma_hat = 0.3; c.center_Y = center;
    ab::softbart_block b(c);
    b.set_keep_history(true);
    b.set_keep_tree(true);
    std::mt19937_64 unused(1);
    ab::block_context ctx;
    ctx["softbart_target"] = y;
    block_run out;
    for (int s = 0; s < sweeps; ++s) {
        b.set_context(ctx);
        b.step(unused);
        out.fit.push_back(b.current());
        out.sigma.push_back(b.current_sigma());
        out.trees.push_back(b.get_tree());
    }
    out.H = b.predict_history(X_new);
    return out;
}

void partB() {
    std::printf("B. y and y + 3 on a grid of multiples of 2^-10 (N = 64)\n");
    const std::size_t N = 64, p = 2;
    std::mt19937_64 g(233);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::normal_distribution<double> nd(0.0, 0.3);
    arma::mat X(N, p), X_new(10, p);
    arma::vec y(N);
    for (std::size_t i = 0; i < N; ++i) {
        X(i, 0) = U(g); X(i, 1) = U(g);
        y[i] = std::round(1024.0 * (std::sin(4.0 * X(i, 0)) + X(i, 1) + nd(g))) / 1024.0;
    }
    for (arma::uword i = 0; i < X_new.n_rows; ++i) { X_new(i, 0) = U(g); X_new(i, 1) = U(g); }
    const arma::vec y3 = y + 3.0;
    const bool exact = arma::mean(y3) == arma::mean(y) + 3.0 &&
                       arma::all((y3 - arma::mean(y3)) == (y - arma::mean(y)));
    chk::check(exact, "   setup: mean(y + 3) = mean(y) + 3 and y + 3 - mean(y + 3) = y - mean(y), exactly");
    const int S = 60;
    const block_run a = run_block(X, y, X_new, true, S), b = run_block(X, y3, X_new, true, S);
    bool same = a.trees.size() == b.trees.size();
    double gap = 0.0, level = 0.0;
    for (std::size_t s = 0; same && s < a.trees.size(); ++s) {
        same = a.trees[s] == b.trees[s] && a.sigma[s] == b.sigma[s];
        gap = std::max(gap, arma::abs(b.fit[s] - a.fit[s] - 3.0).max());
        level += arma::mean(b.fit[s]) / static_cast<double>(S);
    }
    const bool h_ok = a.H.n_rows == static_cast<arma::uword>(S) && b.H.n_rows == a.H.n_rows;
    const double gap_h = h_ok ? arma::abs(b.H - a.H - 3.0).max() : -1.0;
    chk::check(same, "1. the trees and sigma of every sweep are the same for y and y + 3");
    char buf[200];
    std::snprintf(buf, sizeof buf, "2. fit rows for y + 3 = fit rows for y, plus 3 (max "
                  "|difference| %.3g < 1e-12)", gap);
    chk::check(same && gap < 1e-12, buf);
    std::snprintf(buf, sizeof buf, "3. predict_history(X_new) rows for y + 3 = rows for y, "
                  "plus 3 (max |difference| %.3g < 1e-12)", gap_h);
    chk::check(h_ok && gap_h >= 0.0 && gap_h < 1e-12, buf);
    std::snprintf(buf, sizeof buf, "4. mean of the fit rows %.4f, mean(y + 3) %.4f "
                  "(difference < 0.25)", level, arma::mean(y3));
    chk::check(std::fabs(level - arma::mean(y3)) < 0.25, buf);
}

void partC() {
    std::printf("C. center_Y = false\n");
    const data_t d = make_data();
    bart_rng::set_seed(239);
    ab::softbart_block_config c;
    c.x_train = d.X; c.y_init = d.y; c.sigma_key = ""; c.ntrees = 10; c.center_Y = false;
    ab::softbart_block b(c);
    b.set_keep_tree(true);
    const double mid = 0.5 * (d.y.max() + d.y.min());
    std::mt19937_64 unused(1);
    ab::block_context ctx;
    bool fit_ok = true;
    for (int s = 0; s < 20; ++s) {
        ctx["softbart_target"] = arma::vec(d.y + 0.1 * s);
        b.set_context(ctx);
        b.step(unused);
        fit_ok = fit_ok && fit_is_m_plus_kernel_fit(b, mid, d.X);
    }
    const softbart::SoftbartHistory h = b.get_full_history();
    bool ok = h.center_history.n_elem == 20;
    for (arma::uword s = 0; ok && s < h.center_history.n_elem; ++s) ok = h.center_history[s] == mid;
    chk::check(fit_ok, "1. current() = midrange(y_init) + y_scale * kernel fit after every sweep");
    chk::check(ok, "2. every stored m is the midrange of y_init");
}

void partD() {
    std::printf("D. set_Y and set_data between sweeps\n");
    const data_t d = make_data();
    bart_rng::set_seed(241);
    ab::softbart_block_config c;
    c.x_train = d.X; c.y_init = d.y; c.sigma_key = ""; c.ntrees = 10;
    ab::softbart_block b(c);
    b.set_keep_tree(true);
    std::mt19937_64 unused(1);
    ab::block_context ctx;
    ctx["softbart_target"] = d.y;
    for (int s = 0; s < 5; ++s) { b.set_context(ctx); b.step(unused); }
    const arma::vec y2 = d.y + 2.5;
    b.set_Y(y2);
    const double m2 = arma::mean(y2);
    chk::check(b.underlying()->center() == m2 && fit_is_m_plus_kernel_fit(b, m2, d.X),
               "1. after set_Y(y + 2.5): m = mean(y + 2.5), current() = m + y_scale * kernel fit");
    const arma::vec y3 = d.y - 4.0;
    ctx["softbart_target"] = y3;
    b.set_context(ctx); b.step(unused);
    const double m3 = arma::mean(y3);
    softbart::SoftbartHistory h = b.get_full_history();
    chk::check(h.center_history.n_elem == 6 && h.center_history[5] == m3 &&
               h.center_history[4] == arma::mean(d.y) && fit_is_m_plus_kernel_fit(b, m3, d.X),
               "2. the next sweep stores the mean of its working response (y - 4)");
    arma::mat X2 = d.X;
    X2.col(0) = 1.0 - X2.col(0);
    const arma::vec y4 = d.y * 2.0 + 1.0;
    b.set_data(X2, y4);
    const double m4 = arma::mean(y4);
    chk::check(b.underlying()->center() == m4 && fit_is_m_plus_kernel_fit(b, m4, X2),
               "3. after set_data(X2, 2y + 1): m = mean(2y + 1), current() = m + y_scale * kernel fit");
}

// E. The model of part A for the clear_history() check.
chk::model contract_model() {
    const data_t d = make_data();
    ab::softbart_block* sb = nullptr;
    auto root = make_root(d, &sb);
    const arma::mat X_new = d.X_new;
    return chk::composite_model(root, 227, [sb, X_new](chk::buffers& b) {
        const softbart::SoftbartHistory h = sb->get_full_history();
        b.strings["f: kernel tree_history"] = h.tree_history;
        b.mats["f: kernel center_history"] = arma::mat(h.center_history);
        b.mats["f: kernel sigma_history"] = arma::mat(h.sigma_history);
        b.mats["f: predict_history(X_new)"] = sb->predict_history(X_new);
    });
}

}  // namespace

int main() {
    std::printf("=== test_fix_softbart_center ===\n\n");
    partA();
    partB();
    partC();
    partD();
    chk::run("E. clear_history() with the stored values of m", contract_model, K1, K2);
    return chk::finish("test_fix_softbart_center");
}
