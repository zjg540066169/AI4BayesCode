/*
 *  Copyright (C) 2017-2018 Robert McCulloch, Rodney Sparapani
 *                          and Charles Spanbauer
 *  Copyright (C) 2024-2026 Jungang Zou
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, a copy is available at
 *  https://www.R-project.org/Licenses/GPL-2
 */

/*
 * bart_model.h — standard BART (Chipman et al. 2010) model wrapper.
 *
 * PURE C++ / ARMADILLO PORT (2026-06-21).  This is the R-free core: it
 * depends only on <armadillo> and the (NoRcpp) BART tree kernel.  No Rcpp,
 * no R headers.  R and Python wrappers attach on top of this arma API:
 *   - data in / out are arma::mat / arma::vec;
 *   - batch MCMC returns a plain `stdbart::BartFit` struct (arma members);
 *   - RNG is the seedable std::mt19937_64 stream in BART/r_compat.h
 *     (call bart_rng::set_seed(seed) before sampling; replaces set.seed()).
 *
 * The class name, method names, and sampling semantics are unchanged from
 * the prior Rcpp version (DART, keep_history, set_tree validation,
 * weighted/heteroscedastic update, serialized-forest predict are all
 * preserved); only the container types changed Rcpp -> Armadillo.
 */

// Activate the kernel's standalone (R-free) path before pulling it in.
// R module build: use R's Rmath/RNG, not r_compat, to avoid redefining R's namespace-R symbols.
#if !defined(NoRcpp) && !defined(AI4BAYESCODE_RCPP_MODULE)
#define NoRcpp
#endif

#ifndef BART_MODEL_H_
#define BART_MODEL_H_

#include <armadillo>
#include "BART/tree.h"
#include "BART/treefuns.h"
#include "BART/info.h"
#include "BART/bartfuns.h"
#include "BART/bd.h"
#include "BART/bart.h"
#include "BART/heterbart.h"
#include "BART/lambda.h"
#include "BART/bart_model_matrix.h"   // pure C++ cutpoint construction

#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <limits>
#include <cmath>

namespace stdbart {

// ---- batch-MCMC result (replaces the old Rcpp::List) --------------------
struct BartFit {
  arma::mat  yhat_train;        // (n_draws x n) per-draw f(X_train) (+ mu)
  arma::vec  yhat_train_mean;   // (n)  posterior mean of f(X_train) (+ mu)
  arma::vec  sigma_draws;       // (n_draws)  residual-sd draw per kept iter
  arma::umat varcount;          // (n_draws x p)  variable usage counts
  arma::mat  varprob;           // (n_draws x p)  DART s draws
  double     mu = 0.0;          // centring constant (fmean)
  // Serialized posterior forest + cutpoints, for predict() on new X.
  std::string                      trees;       // "ndraws ntrees p\n" + trees
  std::vector<std::vector<double>> cutpoints;   // xinfo
};

// ---- optional posterior-history bundle (keep_history) -------------------
struct BartHistory {
  std::vector<std::string>          tree_history;   // per kept iter, "---"-sep
  arma::vec                         sigma_history;
  arma::umat                        var_counts_history;
  arma::mat                         var_probs_history;
  std::vector<std::vector<double>>  xinfo;
  long                              ntrees = 0;
  long                              p      = 0;
  double                            fmean  = 0.0;
};

class bart_model {
public:
  bart_model() {}

  // X is N x p (observations in rows).  sigmaf: pass NaN for "unset".
  bart_model(const arma::mat& x_train, const arma::vec& y_train,
             long numcut = 100L, bool usequants = false, bool cont = false,
             bool rm_const = false, int ntrees = 300,
             double sigmaf = std::numeric_limits<double>::quiet_NaN(),
             double k = 2.0, double power = 2.0, double base = 0.95,
             double nu = 3.0, bool binary = false)
  {
    this->usequants = usequants;
    this->cont      = cont;
    this->alpha     = base;
    this->mybeta    = power;
    this->sigma     = 1.0;
    this->nu        = nu;
    this->dart      = false;
    this->aug       = false;
    this->center_y  = true;

    n = (long)y_train.n_elem;
    if ((long)x_train.n_rows != n)
      throw std::invalid_argument(
        "bart_model: nrow(x_train) must equal length(y_train)");

    // ---- cutpoints (pure C++) ----
    bart_pp::BmmResult bmm =
        bart_pp::compute_bart_model_matrix(x_train, (int)numcut, usequants,
                                           /*type=*/7, cont);
    this->numcut_ = bmm.numcut;
    this->cutpoints_init_ = bmm.xinfo;

    // ---- store X as p x n (obs-contiguous in column-major memory) ----
    this->stored_x_ = x_train.t();
    this->ix = this->stored_x_.memptr();
    p = (long)this->stored_x_.n_rows;

    // ---- sigma prior calibration ----
    double sigest = (n > 1) ? arma::stddev(y_train) : 1.0;   // R sd: /(n-1)
    if (p < n) {
      arma::mat xr = x_train;
      bool has_const = false;
      if (!rm_const)
        for (arma::uword j = 0; j < xr.n_cols; ++j)
          if (arma::all(xr.col(j) == 1.0)) { has_const = true; break; }
      if (!has_const) xr.insert_cols(0, arma::ones<arma::vec>(n));
      arma::vec coef  = arma::solve(xr, y_train);
      arma::vec resid = y_train - xr * coef;
      double sig2 = arma::dot(resid, resid) / (double)(n - p);
      if (sig2 > 0) { sigest = std::sqrt(sig2); sigma = sigest; }
    }
    // R::qchisq(1-0.9, nu).  Under the R (Rcpp) build bart_sf (r_compat.h)
    // is compiled out, so use Rcpp's R::qchisq (= ::Rf_qchisq) directly;
    // under NoRcpp use the pure-C++ bart_sf shim.  Same value either way.
#if defined(NoRcpp)
    double qchi  = bart_sf::qchisq(0.1, nu);
#else
    double qchi  = R::qchisq(0.1, nu, /*lower=*/1, /*logp=*/0);
#endif
    this->lambda = (sigest * sigest * qchi) / nu;

    this->ntrees = ntrees;
    this->fmean  = arma::mean(y_train);

    // ---- centred (and offset-adjusted) response ----
    this->offset_   = arma::zeros<arma::vec>(n);
    this->stored_y_ = y_train - this->fmean;
    this->iy = this->stored_y_.memptr();

    if (std::isnan(sigmaf)) {
      if (!binary)
        tau = (this->stored_y_.max() - this->stored_y_.min())
              / (2.0 * k * std::sqrt((double)ntrees));
      else
        tau = 3.0 / (k * std::sqrt((double)ntrees));
    } else {
      this->sigmaf = sigmaf / std::sqrt((double)ntrees);
      tau = this->sigmaf;
    }

    bm = heterbart(ntrees);

    // install cutpoints (xinfo) from the precomputed ragged vector
    {
      xinfo xi_;
      xi_.resize(p);
      for (long i = 0; i < p; ++i) {
        xi_[i].resize(numcut_[i]);
        for (int j = 0; j < numcut_[i]; ++j) xi_[i][j] = cutpoints_init_[i][j];
      }
      bm.setxinfo(xi_);
    }

    bm.setprior(alpha, mybeta, tau);
    bm.setdata(p, n, ix, iy, numcut_.data());
    bm.setdart(/*a=*/0.5, /*b=*/1.0, /*rho=*/(double)p, aug, dart,
               /*theta=*/0.0, /*omega=*/1.0);
  }

  // ===================================================================
  //  Single-step Gibbs primitives (plug-in / outer-Gibbs use)
  // ===================================================================
  void update_step() {
    bm.draw(this->sigma, gen);
    double rss = 0.0;
    for (long kk = 0; kk < n; ++kk) { double r = iy[kk] - bm.f(kk); rss += r * r; }
    this->sigma = get_invchi(n, rss);
    record_history_();
  }
  void update_step(double sigma_external) {
    this->sigma = sigma_external;
    bm.draw(this->sigma, gen);
    record_history_();
  }
  // Per-observation noise sd (weighted / heteroscedastic BART).
  void update_step_weighted(const std::vector<double>& sigma_per_obs) {
    if ((long)sigma_per_obs.size() != n)
      throw std::invalid_argument(
        "bart_model::update_step_weighted: sigma_per_obs length != n");
    bm.draw(const_cast<double*>(sigma_per_obs.data()), gen);
    record_history_();
  }

  // ===================================================================
  //  History recording
  // ===================================================================
  void set_history(bool on)   { history_recording_ = on; }
  bool history_recording() const { return history_recording_; }
  void clear_history() {
    tree_history_.clear(); sigma_history_.clear();
    var_counts_history_.clear(); var_probs_history_.clear();
  }

  // Snapshot the current forest as a single string ("ntrees\n tree --- ...").
  std::string get_tree() const {
    std::ostringstream os;
    os << std::setprecision(17) << ntrees << "\n";
    for (std::size_t t = 0; t < (std::size_t)ntrees; ++t) {
      os << const_cast<heterbart&>(bm).gettree(t);
      os << "---\n";
    }
    return os.str();
  }

  // Replace the live forest with one parsed from `s` (validated).  Throws
  // std::invalid_argument on any structural problem (mirrors the old
  // Rcpp::stop checks 1-14).
  void set_tree(const std::string& str) {
    if (str.empty())
      throw std::invalid_argument("bart_model::set_tree: input is empty");
    std::istringstream is(str);
    long T_signed = -1;
    if (!(is >> T_signed))
      throw std::invalid_argument("bart_model::set_tree: failed to parse tree count");
    if (T_signed < 0)
      throw std::invalid_argument("bart_model::set_tree: tree count is negative");
    if (T_signed != ntrees)
      throw std::invalid_argument("bart_model::set_tree: tree count != ntrees");
    const std::size_t T = (std::size_t)T_signed;
    const xinfo& xi = bm.getxinfo();
    const int depth_max = 50;
    for (std::size_t t = 0; t < T; ++t) {
      if (is.eof())
        throw std::invalid_argument("bart_model::set_tree: stream ended early");
      tree& current = bm.gettree(t);
      if (!(is >> current))
        throw std::invalid_argument("bart_model::set_tree: failed to parse a tree");
      std::string sep;
      if (!(is >> sep) || sep != "---")
        throw std::invalid_argument("bart_model::set_tree: expected '---' separator");
      tree::npv nodes;
      current.getnodes(nodes);
      for (tree::tree_p node : nodes) {
        if ((int)node->depth() > depth_max)
          throw std::invalid_argument("bart_model::set_tree: tree too deep");
        const bool has_l = node->getl() != 0;
        const bool has_r = node->getr() != 0;
        if (has_l != has_r)
          throw std::invalid_argument("bart_model::set_tree: node has exactly one child");
        if (has_l) {
          if (node->getl() == node->getr())
            throw std::invalid_argument("bart_model::set_tree: identical children");
          const std::size_t v = node->getv();
          const std::size_t c = node->getc();
          if ((long)v >= p)
            throw std::invalid_argument("bart_model::set_tree: var index out of range");
          if (xi.empty() || v >= xi.size() || xi[v].empty() || c >= xi[v].size())
            throw std::invalid_argument("bart_model::set_tree: cutpoint index out of range");
        } else {
          if (!std::isfinite(node->gettheta()))
            throw std::invalid_argument("bart_model::set_tree: non-finite leaf theta");
        }
      }
    }
    std::string trailing;
    if (is >> trailing)
      throw std::invalid_argument("bart_model::set_tree: unexpected trailing content");
    bm.refresh_allfit();
  }

  BartHistory get_history() const {
    const std::size_t n_iter = sigma_history_.size();
    BartHistory h;
    h.tree_history = tree_history_;
    h.sigma_history = arma::vec(sigma_history_.size());
    for (std::size_t i = 0; i < sigma_history_.size(); ++i) h.sigma_history(i) = sigma_history_[i];
    h.var_counts_history.set_size(n_iter, p);
    h.var_probs_history.set_size(n_iter, p);
    for (std::size_t i = 0; i < n_iter; ++i)
      for (long j = 0; j < p; ++j) {
        h.var_counts_history(i, j) = (j < (long)var_counts_history_[i].size())
                                       ? (arma::uword)var_counts_history_[i][j] : 0u;
        h.var_probs_history(i, j)  = (j < (long)var_probs_history_[i].size())
                                       ? var_probs_history_[i][j] : (double)(1.0 / p);
      }
    const xinfo& xi = bm.getxinfo();
    h.xinfo.resize(xi.size());
    for (std::size_t j = 0; j < xi.size(); ++j) h.xinfo[j] = xi[j];
    h.ntrees = ntrees; h.p = p; h.fmean = fmean;
    return h;
  }

private:
  void record_history_() {
    if (!history_recording_) return;
    std::ostringstream os;
    os << std::setprecision(17) << ntrees << "\n";
    for (std::size_t t = 0; t < (std::size_t)ntrees; ++t) { os << bm.gettree(t); os << "---\n"; }
    tree_history_.push_back(os.str());
    sigma_history_.push_back(this->sigma);
    std::vector<size_t> nv = bm.getnv();
    std::vector<int> ci(p);
    for (long j = 0; j < p; ++j) ci[j] = (j < (long)nv.size()) ? (int)nv[j] : 0;
    var_counts_history_.push_back(std::move(ci));
    std::vector<double> pv = bm.getpv();
    std::vector<double> pp(p);
    for (long j = 0; j < p; ++j) pp[j] = (j < (long)pv.size()) ? pv[j] : 1.0 / (double)p;
    var_probs_history_.push_back(std::move(pp));
  }

public:
  // Manually activate DART (no-op if dart=false at construction).
  void startdart() { if (dart) bm.startdart(); }

  // Replace X only (keep current Y / fmean / sigma).  x_new is N x p.
  void set_X(const arma::mat& x_new) {
    if ((long)x_new.n_rows != n)
      throw std::invalid_argument("bart_model::set_X: nrow(X) != n");
    if ((long)x_new.n_cols != p)
      throw std::invalid_argument("bart_model::set_X: ncol(X) != p");
    this->stored_x_ = x_new.t();
    this->ix = this->stored_x_.memptr();
    bm.setdata(p, n, this->ix, this->iy, numcut_.data());
  }

  // Additive offset (length n).  Set BEFORE set_Y / set_data.
  void set_offset(const arma::vec& off) {
    if (off.n_elem == 0) this->offset_ = arma::zeros<arma::vec>(n);
    else {
      if ((long)off.n_elem != n)
        throw std::invalid_argument("bart_model::set_offset: length != n");
      this->offset_ = off;
    }
  }

  void set_data(const arma::mat& x_new, const arma::vec& y_new, bool center_Y = true) {
    if ((long)x_new.n_rows != (long)y_new.n_elem)
      throw std::invalid_argument("bart_model::set_data: nrow(X) != length(Y)");
    n = (long)y_new.n_elem;
    center_y = center_Y;
    this->fmean = center_y ? arma::mean(y_new) : 0.0;
    if ((long)this->offset_.n_elem != n) this->offset_ = arma::zeros<arma::vec>(n);
    this->stored_y_ = y_new - this->fmean - this->offset_;
    this->iy = this->stored_y_.memptr();
    this->stored_x_ = x_new.t();
    this->ix = this->stored_x_.memptr();
    p = (long)this->stored_x_.n_rows;
    bm.setdata(p, n, this->ix, this->iy, numcut_.data());
  }

  void set_Y(const arma::vec& y_new, bool center_Y = true) {
    if ((long)y_new.n_elem != n)
      throw std::invalid_argument("bart_model::set_Y: length(Y) != n");
    center_y = center_Y;
    this->fmean = center_y ? arma::mean(y_new) : 0.0;
    if ((long)this->offset_.n_elem != n) this->offset_ = arma::zeros<arma::vec>(n);
    this->stored_y_ = y_new - this->fmean - this->offset_;
    this->iy = this->stored_y_.memptr();
    bm.setdata(p, n, this->ix, this->iy, numcut_.data());
  }

  // Posterior predictive on new X (N_new x p): returns (n_draws x N_new),
  // on the Y scale (mu added back).  Uses the serialized forest from the
  // most recent update(); empty if update() has not been called.
  arma::mat predict(const arma::mat& x_predict) const {
    if (last_trees_.empty()) return arma::mat();
    if ((long)x_predict.n_cols != p)
      throw std::invalid_argument("bart_model::predict: ncol(X) != p");
    return eval_serialized_forest_(last_trees_, last_cutpoints_, x_predict, fmean);
  }

  double get_sigma() const     { return this->sigma; }
  double current_sigma() const { return this->sigma; }
  bool   get_usequants() const { return this->usequants; }
  double get_invchi(long n_, double rss) {
    return std::sqrt((nu * lambda + rss) / gen.chi_square((double)(n_ + nu)));
  }
  double get_lambda() const { return lambda; }
  double get_nu()     const { return nu; }

  // Live cached fit on stored x_train (O(n)); mu added back.
  arma::vec current_fit_train() const {
    arma::vec out(n);
    for (long i = 0; i < n; ++i) out(i) = bm.f(i) + fmean;
    return out;
  }
  arma::vec current_var_probs() {
    std::vector<double>& pv = bm.getpv();
    return arma::vec(pv.data(), pv.size());
  }
  arma::uvec current_var_counts() {
    std::vector<size_t>& nv = bm.getnv();
    arma::uvec out(nv.size());
    for (size_t j = 0; j < nv.size(); ++j) out(j) = (arma::uword)nv[j];
    return out;
  }
  // DART weights (canonical interface).
  arma::vec get_s() {
    std::vector<double>& pv = bm.getpv();
    return arma::vec(pv.data(), pv.size());
  }
  void set_s(const arma::vec& s) {
    std::vector<double>& pv = bm.getpv();
    if ((long)s.n_elem != (long)pv.size())
      throw std::invalid_argument("bart_model::set_s: length(s) != p");
    for (size_t j = 0; j < pv.size(); ++j) pv[j] = s(j);
  }

  long N()      const { return n; }
  long P()      const { return p; }
  long n_trees() const { return ntrees; }

  // ===================================================================
  //  Batch MCMC (returns a plain struct).  Three overloads mirror the
  //  prior Rcpp version: internal-sigma, fixed-sigma, weighted-sigma.
  // ===================================================================
  BartFit update(long nburn, long npost, int skip = 1) {
    return run_(/*mode=*/0, 0.0, nullptr, nburn, npost, skip);
  }
  BartFit update(double sigma_fixed, long nburn, long npost, int skip = 1) {
    return run_(/*mode=*/1, sigma_fixed, nullptr, nburn, npost, skip);
  }
  BartFit update(double sigma_scale, const arma::vec& w,
                 long nburn, long npost, int skip = 1) {
    return run_(/*mode=*/2, sigma_scale, &w, nburn, npost, skip);
  }

private:
  BartFit run_(int mode, double sigma_arg, const arma::vec* w,
               long nburn, long npost, int skip) {
    if (mode != 0) this->sigma = sigma_arg;
    const long n_keep = npost / skip;

    BartFit fit;
    fit.yhat_train.zeros(n_keep, n);
    fit.yhat_train_mean.zeros(n);
    fit.sigma_draws.zeros(n_keep);
    fit.varcount.zeros(n_keep, p);
    fit.varprob.zeros(n_keep, p);
    fit.mu = fmean;

    std::vector<double> svec;
    if (mode == 2) {
      svec.resize(n);
      for (long i = 0; i < n; ++i) svec[i] = (*w)(i) * sigma_arg;
    }

    std::ostringstream treess;
    treess.precision(10);
    treess << n_keep << " " << ntrees << " " << p << "\n";

    size_t trcnt = 0;
    for (long it = 0; it < nburn + npost; ++it) {
      if (mode == 2) bm.draw(svec.data(), gen);
      else           bm.draw(this->sigma, gen);

      if (mode == 0) {   // internal-sigma Gibbs update
        double rss = 0.0;
        for (long kk = 0; kk < n; ++kk) { double r = iy[kk] - bm.f(kk); rss += r * r; }
        this->sigma = get_invchi(n, rss);
      }

      if (it >= nburn) {
        for (long kk = 0; kk < n; ++kk) fit.yhat_train_mean(kk) += bm.f(kk);
        const bool keep = (((it - nburn + 1) % skip) == 0);
        if (keep) {
          for (long kk = 0; kk < n; ++kk) fit.yhat_train(trcnt, kk) = bm.f(kk) + fmean;
          fit.sigma_draws(trcnt) = this->sigma;
          for (long j = 0; j < ntrees; ++j) treess << bm.gettree(j);
          std::vector<size_t> nv = bm.getnv();
          std::vector<double> pv = bm.getpv();
          for (long j = 0; j < p; ++j) {
            fit.varcount(trcnt, j) = (arma::uword)nv[j];
            fit.varprob(trcnt, j)  = pv[j];
          }
          ++trcnt;
        }
      }
    }
    fit.yhat_train_mean = fit.yhat_train_mean / (double)npost + fmean;

    // store cutpoints + serialized forest for predict()
    const xinfo& xi = bm.getxinfo();
    fit.cutpoints.resize(xi.size());
    for (std::size_t j = 0; j < xi.size(); ++j) fit.cutpoints[j] = xi[j];
    fit.trees = treess.str();

    last_trees_     = fit.trees;
    last_cutpoints_ = fit.cutpoints;
    return fit;
  }

  // Evaluate a serialized "ndraws ntrees p\n"+trees forest on X_new
  // (N_new x p).  Returns (ndraws x N_new) on the Y scale (mu added).
  static arma::mat eval_serialized_forest_(
      const std::string& trees_str,
      const std::vector<std::vector<double>>& cuts,
      const arma::mat& X_new, double mu)
  {
    std::istringstream is(trees_str);
    long ndraws = 0, nt = 0, pp = 0;
    is >> ndraws >> nt >> pp;
    const long Nt = (long)X_new.n_rows;
    arma::mat out(ndraws, Nt, arma::fill::zeros);
    std::vector<double> xrow(pp);
    for (long d = 0; d < ndraws; ++d) {
      std::vector<tree> forest(nt);
      for (long t = 0; t < nt; ++t) is >> forest[t];
      for (long r = 0; r < Nt; ++r) {
        for (long j = 0; j < pp; ++j) xrow[j] = X_new(r, j);
        double f = 0.0;
        for (auto& t : forest) {
          tree* node = &t;
          while (node->getl() != 0) {
            const std::size_t v = node->getv();
            const std::size_t c = node->getc();
            node = (xrow[v] < cuts[v][c]) ? node->getl() : node->getr();
          }
          f += node->gettheta();
        }
        out(d, r) = f + mu;
      }
    }
    return out;
  }

private:
  // ---- persistent storage (column-major; obs-contiguous as p x n) ----
  arma::mat stored_x_;      // p x n
  arma::vec stored_y_;      // n
  arma::vec offset_;        // n
  std::vector<int> numcut_; // p
  std::vector<std::vector<double>> cutpoints_init_;

  bool usequants = false, cont = false, dart = false, aug = false;

  long n = 0, p = 0, ntrees = 0;
  double sigmaf = 0.0, tau = 0.0;
  double *ix = nullptr, *iy = nullptr;
  double alpha = 0.95, mybeta = 2.0, fmean = 0.0, sigma = 1.0,
         nu = 3.0, lambda = 1.0;
  bool center_y = true;

  arn gen;            // R-named RNG, backed by r_compat.h (pure C++)
  heterbart bm;

  // serialized forest from the latest update(), for predict()
  std::string                      last_trees_;
  std::vector<std::vector<double>> last_cutpoints_;

  // history buffers
  bool history_recording_ = false;
  std::vector<std::string>          tree_history_;
  std::vector<double>               sigma_history_;
  std::vector<std::vector<int> >    var_counts_history_;
  std::vector<std::vector<double> > var_probs_history_;
};

// ---- one-shot batch entry (pure C++; R/Python wrappers can call this) ---
inline BartFit bart_mcmc(const arma::mat& X, const arma::vec& Y,
                         long nburn = 1000, long npost = 1000, int skip = 1,
                         int ntrees = 200, double k = 2.0, double power = 2.0,
                         double base = 0.95, double nu = 3.0,
                         long numcut = 100L, bool usequants = false,
                         bool cont = false, bool binary = false,
                         bool dart = false, bool center_Y = true)
{
  bart_model m(X, Y, numcut, usequants, cont, /*rm_const=*/false, ntrees,
               std::numeric_limits<double>::quiet_NaN(), k, power, base, nu, binary);
  if (!center_Y) m.set_Y(Y, false);
  if (dart) m.startdart();
  m.set_history(false);
  for (long i = 0; i < nburn; ++i) m.update_step();
  return m.update(0, npost, skip);
}

}  // namespace stdbart

#endif  // BART_MODEL_H_
