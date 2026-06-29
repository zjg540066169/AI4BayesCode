/*
 *  genbart_model.h — generalized BART (Linero 2022 RJMCMC, pluggable
 *  likelihoods).  PURE C++ / ARMADILLO PORT (2026-06-21): R-free core for
 *  R + Python wrappers.  Depends only on <armadillo> + the (NoRcpp) genBART
 *  kernel.  Class/method names + sampling semantics unchanged from the Rcpp
 *  version; only container types changed Rcpp -> Armadillo, and batch /
 *  history returns became plain structs.  RNG = seedable std::mt19937_64
 *  (bart_rng::set_seed); the genBART kernel + 10 likelihoods are untouched.
 *
 *  Copyright (C) 2024-2026 Jungang Zou.  GPL-2-or-later.
 */

// R module build: use R's Rmath/RNG, not r_compat, to avoid redefining R's namespace-R symbols.
#if !defined(NoRcpp) && !defined(AI4BAYESCODE_RCPP_MODULE)
#define NoRcpp
#endif

#ifndef GENBART_MODEL_H_
#define GENBART_MODEL_H_

#include <armadillo>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "GENBART/BART/bart_model_matrix.h"
#include "GENBART/BART/info.h"
#include "GENBART/BART/rand_draws.h"
#include "GENBART/BART/rn.h"
#include "GENBART/BART/tree.h"
#include "GENBART/BART/treefuns.h"
#include "GENBART/BART/randomkit.h"
#include "GENBART/BART/rtnorm.h"
#include "GENBART/BART/rtgamma.h"
#include "GENBART/BART/common.h"
#include "GENBART/likelihood_interface.h"
#include "GENBART/laplace.h"
#include "GENBART/rjmcmc_bd.h"

#include "GENBART/likelihoods/normal.h"
#include "GENBART/likelihoods/logistic.h"
#include "GENBART/likelihoods/poisson.h"
#include "GENBART/likelihoods/heteroscedastic.h"
#include "GENBART/likelihoods/aft_log_logistic.h"
#include "GENBART/likelihoods/aft_generalized_gamma.h"
#include "GENBART/likelihoods/gamma_shape.h"
#include "GENBART/likelihoods/beta.h"
#include "GENBART/likelihoods/beta_binomial.h"
#include "GENBART/likelihoods/negative_binomial.h"

namespace genbart {

// ---- move-statistics aggregator ----
struct move_stats {
  std::size_t birth_prop = 0, birth_acc = 0;
  std::size_t death_prop = 0, death_acc = 0;
  std::size_t change_prop = 0, change_acc = 0;
  void tally(const move_result& r) {
    if (r.move == 'B') { ++birth_prop; if (r.accepted) ++birth_acc; }
    if (r.move == 'D') { ++death_prop; if (r.accepted) ++death_acc; }
    if (r.move == 'C') { ++change_prop; if (r.accepted) ++change_acc; }
  }
  arma::mat to_matrix() const {       // rows BIRTH/DEATH/CHANGE; cols prop/acc/rate
    arma::mat m(3, 3, arma::fill::zeros);
    m(0,0)=birth_prop;  m(0,1)=birth_acc;
    m(1,0)=death_prop;  m(1,1)=death_acc;
    m(2,0)=change_prop; m(2,1)=change_acc;
    m(0,2)=birth_prop  >0 ? (double)birth_acc /birth_prop  : 0.0;
    m(1,2)=death_prop  >0 ? (double)death_acc /death_prop  : 0.0;
    m(2,2)=change_prop >0 ? (double)change_acc/change_prop : 0.0;
    return m;
  }
};

// ---- history / batch result structs (replace Rcpp::List) ----
struct GenbartHistory {
  std::vector<std::string>          tree_history;
  arma::vec                         sigma_history;
  arma::vec                         sigma_mu_history;
  arma::mat                         s_history;            // n_iter x p
  arma::umat                        var_counts_history;   // n_iter x p
  std::vector<std::vector<double>>  xinfo;
  std::size_t ntrees = 0, p = 0;
};
struct GenbartFit {
  arma::mat f_train;          // npost x N
  arma::mat f_test;           // npost x Nt
  arma::mat nuisance;         // npost x n_nuis
  arma::vec sigma_mu;         // npost
  arma::vec dart_theta;       // npost
  arma::mat var_probs;        // npost x (p or 0)
  arma::mat move_stats;       // 3 x 3
  bool          has_history = false;
  GenbartHistory history;
};

// ---------------------------------------------------------------------------
class genbart_model {
public:
  // X: N x p; Y: length N; offset: length N (empty -> zeros).
  // lik: non-owning likelihood pointer (must outlive this model).
  genbart_model(const arma::mat& X, const arma::vec& Y, const arma::vec& offset,
                likelihood* lik, rjmcmc_hypers hypers, std::size_t ntrees,
                int numcut = 100, bool usequants = false, bool cont = false)
      : N_((std::size_t)X.n_rows), p_((std::size_t)X.n_cols), ntrees_(ntrees),
        hypers_(std::move(hypers)), lik_(lik), trees_(ntrees),
        f_per_tree_train_(ntrees, std::vector<double>(X.n_rows, 0.0)),
        f_train_(X.n_rows, 0.0), xi_sigma_mu_(1.0)
  {
    if (lik_ == nullptr) throw std::invalid_argument("genbart_model: likelihood is null");
    if (ntrees_ == 0)    throw std::invalid_argument("genbart_model: ntrees must be >= 1");
    if (hypers_.half_cauchy_c <= 0.0) hypers_.half_cauchy_c = 1.0/std::sqrt((double)ntrees_);
    xi_sigma_mu_ = 1.0/(hypers_.sigma_mu*hypers_.sigma_mu);
    if (hypers_.dart_rho <= 0.0) hypers_.dart_rho = (double)X.n_cols;
    dart_theta_ = hypers_.dart_theta_init;
    hypers_.dart_active = false;
    log_s_.assign((std::size_t)X.n_cols, -std::log((double)X.n_cols));
    ::genbart_pp::BmmResult bmm = ::genbart_pp::compute_bart_model_matrix(
        X, numcut, usequants, /*type=*/7, cont);
    set_data_internal(bmm);
    set_Y(Y);
    set_offset(offset);
  }

  // ---- plug-in hooks ----
  void set_Y(const arma::vec& Y_new) {
    if ((std::size_t)Y_new.n_elem != N_) throw std::invalid_argument("set_Y: length mismatch");
    Y_ = std::vector<double>(Y_new.begin(), Y_new.end());
    if (lik_ != nullptr) lik_->prepare(Y_);
  }
  void set_offset(const arma::vec& offset_new) {
    if (offset_new.n_elem == 0) { offset_.assign(N_, 0.0); }
    else {
      if ((std::size_t)offset_new.n_elem != N_) throw std::invalid_argument("set_offset: length mismatch");
      offset_ = std::vector<double>(offset_new.begin(), offset_new.end());
    }
  }
  void set_X(const arma::mat& X_new) {
    if ((std::size_t)X_new.n_rows != N_) throw std::invalid_argument("set_X: N mismatch");
    if ((std::size_t)X_new.n_cols != p_) throw std::invalid_argument("set_X: p mismatch");
    for (std::size_t i = 0; i < N_; ++i)
      for (std::size_t j = 0; j < p_; ++j) X_row_[i*p_ + j] = X_new(i, j);
    std::fill(f_train_.begin(), f_train_.end(), 0.0);
    for (std::size_t t = 0; t < ntrees_; ++t)
      for (std::size_t i = 0; i < N_; ++i) {
        tree::tree_p leaf = trees_[t].bn(const_cast<double*>(X_row_.data()) + i*p_, xi_);
        const double v = leaf->gettheta();
        f_per_tree_train_[t][i] = v; f_train_[i] += v;
      }
  }
  void set_data(const arma::mat& X_new, const arma::vec& Y_new) { set_X(X_new); set_Y(Y_new); }

  // ---- sampler step ----
  move_stats update_step() { return update_step(gen_); }
  move_stats update_step(rn& gen) {
    move_stats stats;
    std::vector<double> lam_minus_t(N_, 0.0), new_f(N_);
    for (std::size_t t = 0; t < ntrees_; ++t) {
      for (std::size_t i = 0; i < N_; ++i)
        lam_minus_t[i] = f_train_[i] - f_per_tree_train_[t][i] + offset_[i];
      move_result mr = draw_tree(trees_[t], X_row_.data(), Y_.data(), lam_minus_t.data(),
          N_, p_, xi_, hypers_, *lik_, gen, /*do_M_step=*/true, /*f_tree_out=*/new_f.data());
      stats.tally(mr);
      std::vector<double>& fti = f_per_tree_train_[t];
      for (std::size_t i = 0; i < N_; ++i) { f_train_[i] += new_f[i]-fti[i]; fti[i] = new_f[i]; }
    }
    std::vector<double> lam_full(N_);
    for (std::size_t i = 0; i < N_; ++i) lam_full[i] = f_train_[i] + offset_[i];
    lik_->update_nuisance(Y_, lam_full, gen);

    if (hypers_.adaptive_sigma_mu) {
      double sum_mu2 = 0.0; std::size_t n_leaves = 0;
      for (std::size_t t = 0; t < ntrees_; ++t) {
        tree::npv leaves; trees_[t].getbots(leaves);
        for (std::size_t k = 0; k < leaves.size(); ++k) { const double m=leaves[k]->gettheta(); sum_mu2+=m*m; ++n_leaves; }
      }
      const double c2 = hypers_.half_cauchy_c*hypers_.half_cauchy_c;
      const double a1 = 0.5*((double)n_leaves+1.0), b1 = 1.0/xi_sigma_mu_ + 0.5*sum_mu2;
      double g1 = gen.gamma(a1, b1);
      if (g1 > 0.0 && std::isfinite(g1)) { const double s2=1.0/g1; if (std::isfinite(s2)&&s2>0.0) hypers_.sigma_mu=std::sqrt(s2); }
      const double b2 = 1.0/(hypers_.sigma_mu*hypers_.sigma_mu) + 1.0/c2;
      double g2 = gen.gamma(1.0, b2);
      if (g2 > 0.0 && std::isfinite(g2)) xi_sigma_mu_ = 1.0/g2;
    }
    if (hypers_.dart_active) { update_dart_s(gen); update_dart_theta(gen); }
    record_history_();
    return stats;
  }

  // ---- history ----
  void set_history(bool on) { history_recording_ = on; }
  bool history_recording() const { return history_recording_; }
  void clear_history() {
    tree_history_.clear(); sigma_history_.clear(); sigma_mu_history_.clear();
    s_history_.clear(); var_counts_history_.clear();
  }
  std::string get_tree() const {
    std::ostringstream os; os << std::setprecision(17) << ntrees_ << "\n";
    for (std::size_t t = 0; t < ntrees_; ++t) { os << trees_[t]; os << "---\n"; }
    return os.str();
  }
  void set_tree(const std::string& str) {
    if (str.empty()) throw std::invalid_argument("genbart_model::set_tree: input is empty");
    std::istringstream is(str);
    long T_signed = -1;
    if (!(is >> T_signed)) throw std::invalid_argument("genbart_model::set_tree: failed to parse tree count");
    if (T_signed < 0) throw std::invalid_argument("genbart_model::set_tree: tree count is negative");
    const std::size_t T = (std::size_t)T_signed;
    if (T != ntrees_) throw std::invalid_argument("genbart_model::set_tree: tree count != ntrees");
    std::vector<tree> new_trees(T);
    const int depth_max = 50;
    for (std::size_t t = 0; t < T; ++t) {
      if (is.eof()) throw std::invalid_argument("genbart_model::set_tree: stream ended early");
      if (!(is >> new_trees[t])) throw std::invalid_argument("genbart_model::set_tree: failed to parse a tree");
      std::string sep;
      if (!(is >> sep) || sep != "---") throw std::invalid_argument("genbart_model::set_tree: expected '---' separator");
      tree::npv nodes; new_trees[t].getnodes(nodes);
      for (tree::tree_p node : nodes) {
        if ((int)node->depth() > depth_max) throw std::invalid_argument("genbart_model::set_tree: tree too deep");
        const bool hl = node->getl()!=0, hr = node->getr()!=0;
        if (hl != hr) throw std::invalid_argument("genbart_model::set_tree: node has exactly one child");
        if (hl) {
          if (node->getl()==node->getr()) throw std::invalid_argument("genbart_model::set_tree: identical children");
          const std::size_t v=node->getv(), c=node->getc();
          if (v >= p_) throw std::invalid_argument("genbart_model::set_tree: var index out of range");
          if (xi_[v].empty() || c >= xi_[v].size()) throw std::invalid_argument("genbart_model::set_tree: cutpoint index out of range");
        } else if (!std::isfinite(node->gettheta())) throw std::invalid_argument("genbart_model::set_tree: non-finite leaf theta");
      }
    }
    std::string trailing;
    if (is >> trailing) throw std::invalid_argument("genbart_model::set_tree: unexpected trailing content");
    for (std::size_t t = 0; t < T; ++t) trees_[t] = new_trees[t];
    std::fill(f_train_.begin(), f_train_.end(), 0.0);
    for (std::size_t t = 0; t < T; ++t)
      for (std::size_t i = 0; i < N_; ++i) {
        tree::tree_p leaf = trees_[t].bn(X_row_.data() + i*p_, xi_);
        const double v = leaf->gettheta(); f_per_tree_train_[t][i]=v; f_train_[i]+=v;
      }
  }
  GenbartHistory get_history() const {
    const std::size_t n_iter = sigma_history_.size();
    GenbartHistory h;
    h.tree_history = tree_history_;
    h.sigma_history = arma::vec(sigma_history_.size());
    h.sigma_mu_history = arma::vec(sigma_mu_history_.size());
    for (std::size_t i=0;i<sigma_history_.size();++i) h.sigma_history(i)=sigma_history_[i];
    for (std::size_t i=0;i<sigma_mu_history_.size();++i) h.sigma_mu_history(i)=sigma_mu_history_[i];
    h.s_history.set_size(n_iter, p_); h.var_counts_history.set_size(n_iter, p_);
    for (std::size_t i=0;i<n_iter;++i)
      for (std::size_t j=0;j<p_;++j) {
        h.s_history(i,j) = (j<s_history_[i].size())? s_history_[i][j] : (double)(1.0/p_);
        h.var_counts_history(i,j) = (j<var_counts_history_[i].size())? (arma::uword)var_counts_history_[i][j] : 0u;
      }
    h.xinfo.resize(xi_.size());
    for (std::size_t j=0;j<xi_.size();++j) h.xinfo[j]=xi_[j];
    h.ntrees=ntrees_; h.p=p_;
    return h;
  }

private:
  void record_history_() {
    if (!history_recording_) return;
    std::ostringstream os; os << std::setprecision(17) << ntrees_ << "\n";
    for (std::size_t t=0;t<ntrees_;++t) { os << trees_[t]; os << "---\n"; }
    tree_history_.push_back(os.str());
    std::vector<double> nu = (lik_!=nullptr)? lik_->nuisance_snapshot() : std::vector<double>{};
    sigma_history_.push_back(nu.empty()? 0.0 : nu[0]);
    sigma_mu_history_.push_back(hypers_.sigma_mu);
    std::vector<double> s(p_, 1.0/(double)p_);
    if (hypers_.var_probs.size()==p_) for (std::size_t j=0;j<p_;++j) s[j]=hypers_.var_probs[j];
    s_history_.push_back(std::move(s));
    std::vector<std::size_t> nv(p_, 0);
    const_cast<genbart_model*>(this)->count_branch_variables(nv);
    std::vector<int> ci(p_); for (std::size_t j=0;j<p_;++j) ci[j]=(int)nv[j];
    var_counts_history_.push_back(std::move(ci));
  }
public:
  double current_sigma_mu() const { return hypers_.sigma_mu; }
  double current_sigma() const {
    if (lik_==nullptr) return 0.0;
    std::vector<double> nu = lik_->nuisance_snapshot();
    return nu.empty()? 0.0 : nu[0];
  }
  double get_invchi(long n_obs, double rss) {
    double nu_loc=3.0, lambda_loc=1.0;
    if (lik_!=nullptr) { std::vector<double> hp=lik_->nuisance_snapshot(); if (hp.size()>=3){nu_loc=hp[1];lambda_loc=hp[2];} }
    const double chi2 = gen_.chi_square(n_obs + nu_loc);
    return std::sqrt((nu_loc*lambda_loc + rss)/chi2);
  }
  arma::vec get_s() const {
    arma::vec out(p_); out.fill(1.0/(double)p_);
    if (hypers_.var_probs.size()==p_) for (std::size_t j=0;j<p_;++j) out(j)=hypers_.var_probs[j];
    return out;
  }
  void set_s(const arma::vec& s) {
    if ((std::size_t)s.n_elem != p_) throw std::invalid_argument("genbart_model::set_s: length(s) != p");
    if (hypers_.var_probs.size()!=p_) hypers_.var_probs.assign(p_, 0.0);
    if (log_s_.size()!=p_) log_s_.assign(p_, 0.0);
    for (std::size_t j=0;j<p_;++j) { const double sj=std::max(s(j),1e-300); hypers_.var_probs[j]=sj; log_s_[j]=std::log(sj); }
  }
  arma::vec  current_var_probs() const { return get_s(); }
  arma::uvec current_var_counts() {
    std::vector<std::size_t> nv(p_, 0); count_branch_variables(nv);
    arma::uvec out(p_); for (std::size_t j=0;j<p_;++j) out(j)=(arma::uword)nv[j];
    return out;
  }

  void update_dart_s(rn& gen) {
    std::vector<std::size_t> nv(p_, 0); count_branch_variables(nv);
    std::vector<double> alpha(p_); const double base = dart_theta_/(double)p_;
    for (std::size_t j=0;j<p_;++j) alpha[j]=base+(double)nv[j];
    log_s_ = gen.log_dirichlet(alpha);
    if (hypers_.var_probs.size()!=p_) hypers_.var_probs.assign(p_, 0.0);
    for (std::size_t j=0;j<p_;++j) hypers_.var_probs[j]=std::exp(log_s_[j]);
  }
  void update_dart_theta(rn& gen) {
    if (hypers_.dart_const_theta) return;
    const std::size_t G = 1000; const double rho=hypers_.dart_rho, a=hypers_.dart_a, b=hypers_.dart_b;
    double sum_log_s = 0.0; for (std::size_t j=0;j<p_;++j) sum_log_s+=log_s_[j];
    std::vector<double> lwt(G), theta_g(G);
    double mx = -std::numeric_limits<double>::infinity();
    for (std::size_t k=0;k<G;++k) {
      const double lambda=(double)(k+1)/(double)(G+1), t=lambda*rho/(1.0-lambda); theta_g[k]=t;
      const double log_lik = std::lgamma(t) - (double)p_*std::lgamma(t/(double)p_) + (t/(double)p_)*sum_log_s;
      const double log_prior = (a-1.0)*std::log(lambda) + (b-1.0)*std::log(1.0-lambda);
      lwt[k]=log_lik+log_prior; if (lwt[k]>mx) mx=lwt[k];
    }
    double sm=0.0; for (std::size_t k=0;k<G;++k) sm+=std::exp(lwt[k]-mx);
    const double lse=mx+std::log(sm);
    std::vector<double> wts(G); for (std::size_t k=0;k<G;++k) wts[k]=std::exp(lwt[k]-lse);
    gen.set_wts(wts); dart_theta_ = theta_g[gen.discrete()];
  }
  void count_branch_variables(std::vector<std::size_t>& nv) {
    std::fill(nv.begin(), nv.end(), 0);
    for (std::size_t t=0;t<ntrees_;++t) {
      tree::npv nodes; trees_[t].getnodes(nodes);
      for (std::size_t k=0;k<nodes.size();++k) if (nodes[k]->getl()!=0) nv[nodes[k]->getv()]+=1;
    }
  }
  double current_dart_theta() const { return dart_theta_; }

  // ---- readout ----
  arma::vec current_f_train() const { return arma::vec(f_train_.data(), f_train_.size()); }
  double current_f_train(std::size_t i) const { return f_train_.at(i); }
  arma::vec predict_once(const arma::mat& X_test) const {
    const std::size_t Nt = X_test.n_rows;
    if ((std::size_t)X_test.n_cols != p_) throw std::invalid_argument("predict_once: p mismatch");
    std::vector<double> Xr(Nt*p_);
    for (std::size_t i=0;i<Nt;++i) for (std::size_t j=0;j<p_;++j) Xr[i*p_+j]=X_test(i,j);
    arma::vec out(Nt, arma::fill::zeros);
    for (std::size_t t=0;t<ntrees_;++t)
      for (std::size_t i=0;i<Nt;++i) {
        tree::tree_p leaf = const_cast<tree&>(trees_[t]).bn(Xr.data()+i*p_, const_cast<xinfo&>(xi_));
        out(i) += leaf->gettheta();
      }
    return out;
  }

  // ---- full MCMC (returns a struct) ----
  GenbartFit run_mcmc(std::size_t nburn, std::size_t npost, const arma::mat& X_test,
                      long dart_start = -1, bool history = false) {
    return run_mcmc(nburn, npost, X_test, gen_, dart_start, history);
  }
  GenbartFit run_mcmc(std::size_t nburn, std::size_t npost, const arma::mat& X_test,
                      rn& gen, long dart_start = -1, bool history = false) {
    const std::size_t Nt = (X_test.n_rows > 0)? (std::size_t)X_test.n_rows : 0;
    const std::size_t n_nuis = lik_->num_nuisance();
    GenbartFit fit;
    fit.f_train.set_size(npost, N_);
    fit.f_test.set_size(npost, Nt);
    if (n_nuis>0) fit.nuisance.set_size(npost, n_nuis);
    fit.sigma_mu.set_size(npost);
    fit.dart_theta.set_size(npost);
    const std::size_t vp_cols = hypers_.dart_requested ? p_ : 0;
    fit.var_probs.set_size(npost, vp_cols);
    move_stats cumstats;

    history_recording_ = false;
    for (std::size_t it=0; it<nburn+npost; ++it) {
      if (history && it==nburn) history_recording_ = true;
      if (hypers_.dart_requested && !hypers_.dart_active && dart_start>=0 && (long)it>=dart_start) startdart();
      move_stats s = update_step(gen);
      cumstats.birth_prop+=s.birth_prop; cumstats.birth_acc+=s.birth_acc;
      cumstats.death_prop+=s.death_prop; cumstats.death_acc+=s.death_acc;
      cumstats.change_prop+=s.change_prop; cumstats.change_acc+=s.change_acc;
      if (it >= nburn) {
        const std::size_t k = it - nburn;
        for (std::size_t i=0;i<N_;++i) fit.f_train(k,i)=f_train_[i];
        if (Nt>0) { arma::vec r_t=predict_once(X_test); for (std::size_t i=0;i<Nt;++i) fit.f_test(k,i)=r_t(i); }
        if (n_nuis>0) { std::vector<double> nu=lik_->nuisance_snapshot(); for (std::size_t j=0;j<n_nuis && j<nu.size();++j) fit.nuisance(k,j)=nu[j]; }
        fit.sigma_mu(k)=hypers_.sigma_mu; fit.dart_theta(k)=dart_theta_;
        if (hypers_.dart_active && vp_cols>0) for (std::size_t j=0;j<p_;++j) fit.var_probs(k,j)=hypers_.var_probs[j];
      }
    }
    fit.move_stats = cumstats.to_matrix();
    if (history) { fit.has_history=true; fit.history=get_history(); }
    return fit;
  }

  void startdart() { if (hypers_.dart_requested) hypers_.dart_active = true; }

  std::size_t N() const { return N_; }
  std::size_t p() const { return p_; }
  std::size_t ntrees() const { return ntrees_; }
  const std::vector<tree>& trees() const { return trees_; }

private:
  void set_data_internal(const ::genbart_pp::BmmResult& bmm) {
    const arma::mat& X = bmm.X;
    X_row_.assign(N_*p_, 0.0);
    for (std::size_t i=0;i<N_;++i) for (std::size_t j=0;j<p_;++j) X_row_[i*p_+j]=X(i,j);
    xi_.resize(p_);
    for (std::size_t j=0;j<p_;++j) {
      const int nc = bmm.numcut[j];
      xi_[j].resize(nc);
      for (int c=0;c<nc;++c) xi_[j][c]=bmm.xinfo[j][c];
    }
  }

  std::size_t N_, p_, ntrees_;
  rjmcmc_hypers hypers_;
  likelihood*   lik_;
  std::vector<tree> trees_;
  xinfo xi_;
  std::vector<double> X_row_, Y_, offset_;
  std::vector<std::vector<double> > f_per_tree_train_;
  std::vector<double> f_train_;
  double xi_sigma_mu_;
  double dart_theta_;
  std::vector<double> log_s_;
  arn gen_;
  bool history_recording_ = false;
  std::vector<std::string>          tree_history_;
  std::vector<double>               sigma_history_;
  std::vector<double>               sigma_mu_history_;
  std::vector<std::vector<double> > s_history_;
  std::vector<std::vector<int> >    var_counts_history_;
};

// walk a deserialized forest at one X row
inline double genbart_walk_forest_row_(std::vector<tree>& trees,
    const std::vector<double>& x_row, const xinfo& xi) {
  double f = 0.0;
  for (auto& t : trees) {
    tree* node = &t;
    while (node->getl() != 0) {
      const std::size_t v=node->getv(), c=node->getc();
      node = (x_row[v] < xi[v][c]) ? node->getl() : node->getr();
    }
    f += node->gettheta();
  }
  return f;
}

// pure posterior predictive from a recorded history (replaces the Rcpp export)
inline arma::mat genbart_predict_posterior(const GenbartHistory& h, const arma::mat& X_new) {
  const std::size_t n_iter = h.tree_history.size();
  const std::size_t Nt = X_new.n_rows;
  const std::size_t p  = X_new.n_cols;
  xinfo xi(p);
  for (std::size_t j=0;j<p && j<h.xinfo.size();++j) xi[j] = h.xinfo[j];
  arma::mat out(n_iter, Nt, arma::fill::zeros);
  std::vector<double> x_row(p);
  for (std::size_t i=0;i<n_iter;++i) {
    std::istringstream is(h.tree_history[i]);
    std::size_t T=0; is >> T;
    std::vector<tree> trees(T);
    for (std::size_t t=0;t<T;++t) { is >> trees[t]; std::string sep; is >> sep; }
    for (std::size_t row=0; row<Nt; ++row) {
      for (std::size_t j=0;j<p;++j) x_row[j]=X_new(row,j);
      out(i,row) = genbart_walk_forest_row_(trees, x_row, xi);
    }
  }
  return out;
}

}  // namespace genbart
#endif  // GENBART_MODEL_H_
