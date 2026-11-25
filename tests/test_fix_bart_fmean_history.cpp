// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_bart_fmean_history.cpp
//
//  bart_block::predict_history(X_new) adds to stored forest d the fmean of the
//  sweep that drew forest d. The composite here holds a block "c" that draws a
//  shift c ~ N(0, 1.5^2) every sweep and a bart_block "f" whose working
//  response is y - c, so fmean = mean(y - c) changes from sweep to sweep, as
//  for the partial residual of an ensemble in VCBart or of BART with a random
//  intercept. After K sweeps with keep_history and keep_tree:
//   1. the kernel stores one fmean per stored forest, equal bit for bit to the
//      mean of the working response of that sweep, and these values span more
//      than 1;
//   2. row d of predict_history(X_new) equals, bit for bit, forest d (parsed
//      here from the stored string and evaluated with the stored cutpoints)
//      at X_new plus the mean of the working response of sweep d;
//   3. row d equals predict(X_new) (the live forest plus the kernel's current
//      fmean) computed right after sweep d;
//   4. at the training X, row d equals fit row d of get_history() to 1e-9
//      (the fit row is the kernel's incrementally updated fit);
//   5. the clear_history() check of clear_history_contract.hpp on this model,
//      with the stored fmean values among the per-draw buffers.
// ============================================================================
#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif
#include <armadillo>
#include "AI4BayesCode/bart_block.hpp"
#include "AI4BayesCode/composite_block.hpp"

#include "clear_history_contract.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <sstream>
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
    std::mt19937_64 g(127);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::normal_distribution<double> nd(0.0, 0.3);
    data_t d;
    d.X.set_size(N, p);
    d.y.set_size(N);
    for (std::size_t i = 0; i < N; ++i) {
        d.X(i, 0) = U(g); d.X(i, 1) = U(g);
        d.y[i] = (d.X(i, 0) > 0.5 ? 1.0 : -1.0) + d.X(i, 1) + nd(g);
    }
    d.X_new.set_size(N_new, p);
    for (std::size_t i = 0; i < N_new; ++i) { d.X_new(i, 0) = U(g); d.X_new(i, 1) = U(g); }
    return d;
}

// The composite of the file header, keep_history and keep_tree on; the kernel
// of the bart_block draws from the thread-local engine bart_rng, seeded here.
std::shared_ptr<ab::composite_block> make_root(const data_t& d, ab::bart_block** bb) {
    bart_rng::set_seed(131);
    auto root = std::make_shared<ab::composite_block>("root");
    root->data().set("y", d.y);
    root->data().set("c", arma::vec{0.0});
    root->data().set("bart_target", d.y);
    root->data().register_refresher(
        "bart_target", [](const ab::shared_data_t& s) -> arma::vec {
            return s.get("y") - s.get("c")[0];
        });
    root->data().declare_invalidates("c", {"bart_target"});
    root->data().declare_dependencies("f", {"bart_target"});
    root->add_child(std::make_unique<shift_block>());
    ab::bart_block_config c;
    c.name = "f"; c.x_train = d.X; c.y_init = d.y;
    c.working_response_key = "bart_target"; c.sigma_key = "";
    c.ntrees = 10; c.numcut = 20;
    auto b = std::make_unique<ab::bart_block>(std::move(c));
    *bb = b.get();
    root->add_child(std::move(b));
    root->set_keep_history(true);
    root->set_keep_tree(true);
    return root;
}

// Forest string s ("T\n" then each tree followed by "---") at the rows of X:
// per row, the leaf values of the trees summed in tree order from 0.
arma::vec eval_forest(const std::string& s,
                      const std::vector<std::vector<double>>& xi,
                      const arma::mat& X) {
    std::istringstream is(s);
    std::size_t T = 0;
    is >> T;
    std::vector<tree> trees(T);
    for (tree& t : trees) { is >> t; std::string sep; is >> sep; }
    arma::vec out(X.n_rows);
    for (arma::uword i = 0; i < X.n_rows; ++i) {
        double f = 0.0;
        for (tree& t : trees) {
            tree* node = &t;
            while (node->getl() != 0) {
                const std::size_t v = node->getv(), c = node->getc();
                node = (X(i, v) < xi[v][c]) ? node->getl() : node->getr();
            }
            f += node->gettheta();
        }
        out[i] = f;
    }
    return out;
}

bool same_bits(const arma::vec& a, const arma::vec& b) {
    return chk::same_bits(arma::mat(a), arma::mat(b));
}

void fmean_checks() {
    std::printf("bart_block, working response y - c with c drawn every sweep (K = %d)\n", K);
    const data_t d = make_data();
    ab::bart_block* bb = nullptr;
    auto root = make_root(d, &bb);
    std::mt19937_64 rng(137);
    std::vector<double> wr_mean(K);
    std::vector<arma::vec> live(K);
    for (int s = 0; s < K; ++s) {
        root->step(rng);
        wr_mean[s] = arma::mean(root->data().get("bart_target"));
        live[s] = bb->predict(d.X_new);
    }
    const stdbart::BartHistory h = bb->get_full_history();

    // 1. one fmean per stored forest, the mean of the working response.
    bool ok = h.tree_history.size() == static_cast<std::size_t>(K) &&
              h.fmean_history.n_elem == static_cast<arma::uword>(K);
    for (int s = 0; ok && s < K; ++s)
        ok = same_bits(arma::vec{h.fmean_history[s]}, arma::vec{wr_mean[s]});
    chk::check(ok, "kernel fmean_history: " + std::to_string(h.fmean_history.n_elem) +
                   " values, each the mean of the working response of its sweep");
    const auto mm = std::minmax_element(wr_mean.begin(), wr_mean.end());
    char buf[160];
    std::snprintf(buf, sizeof buf, "the fmean values span %.3f (> 1) across the sweeps",
                  *mm.second - *mm.first);
    chk::check(*mm.second - *mm.first > 1.0, buf);

    // 2. predict_history(X_new) = forest d at X_new + fmean of sweep d.
    const arma::mat H = bb->predict_history(d.X_new);
    ok = H.n_rows == static_cast<arma::uword>(K) && H.n_cols == d.X_new.n_rows;
    for (int s = 0; ok && s < K; ++s) {
        const arma::vec expect = eval_forest(h.tree_history[s], h.xinfo, d.X_new) + wr_mean[s];
        ok = same_bits(arma::vec(H.row(s).t()), expect);
    }
    chk::check(ok, "predict_history(X_new) " + chk::shape(H) +
                   ": row d = forest d at X_new + fmean of sweep d, bit for bit");

    // The rows the former code returned: forest d + the fmean of the last sweep.
    double shift = 0.0;
    for (int s = 0; s < K; ++s) shift = std::max(shift, std::fabs(wr_mean[K - 1] - wr_mean[s]));
    std::snprintf(buf, sizeof buf, "adding the last sweep's fmean to every forest would "
                  "shift a row by up to %.3f", shift);
    chk::check(shift > 0.5, buf);

    // 3. = predict(X_new) right after sweep d.
    ok = true;
    for (int s = 0; ok && s < K; ++s) ok = same_bits(arma::vec(H.row(s).t()), live[s]);
    chk::check(ok, "row d = predict(X_new) computed right after sweep d, bit for bit");

    // 4. at the training X, = fit row d.
    const arma::mat Ht = bb->predict_history(d.X);
    const ab::history_map hist = root->get_history();
    const arma::mat& F = hist.at("f");
    const bool same_shape = Ht.n_rows == F.n_rows && Ht.n_cols == F.n_cols;
    const double gap = same_shape ? arma::abs(Ht - F).max() : -1.0;
    std::snprintf(buf, sizeof buf, "predict_history(X_train) = get_history() fit rows "
                  "(max |difference| %.3g < 1e-9)", gap);
    chk::check(same_shape && gap >= 0.0 && gap < 1e-9, buf);
}

// 5. The model of fmean_checks() for the clear_history() check.
chk::model contract_model() {
    const data_t d = make_data();
    ab::bart_block* bb = nullptr;
    auto root = make_root(d, &bb);
    const arma::mat X_new = d.X_new;
    return chk::composite_model(root, 137, [bb, X_new](chk::buffers& b) {
        const stdbart::BartHistory h = bb->get_full_history();
        b.strings["f: kernel tree_history"] = h.tree_history;
        b.mats["f: kernel fmean_history"] = arma::mat(h.fmean_history);
        b.mats["f: kernel sigma_history"] = arma::mat(h.sigma_history);
        b.mats["f: predict_history(X_new)"] = bb->predict_history(X_new);
    });
}

}  // namespace

int main() {
    std::printf("=== test_fix_bart_fmean_history ===\n\n");
    fmean_checks();
    chk::run("clear_history() with the stored fmean values", contract_model, K1, K2);
    return chk::finish("test_fix_bart_fmean_history");
}
