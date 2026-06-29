/*
 *  softbart_model.h -- Soft BART (Linero & Yang 2018) wrapper.
 *  PURE C++ / ARMADILLO PORT (2026-06-21): R-free core for R + Python
 *  wrappers.  Depends only on <armadillo> + the (NoRcpp) vendored SoftBart
 *  Forest.  Hypers/Opts are built by direct field assignment (the pure path,
 *  via Hypers::finalize_from_group() + Forest(const Hypers&, const Opts&));
 *  bare unif_rand/norm_rand/Rf_lgammafn resolve to the r_compat.h shims.
 *  RNG = seedable std::mt19937_64 (bart_rng::set_seed).
 *
 *  Vendored SoftBart sources (SOFTBART_VENDOR/) by Antonio R. Linero, GPL-2.
 *  Copyright (C) 2024-2026 Jungang Zou.  GPL-2-or-later.
 */

// R module build: use R's Rmath/RNG, not r_compat, to avoid redefining R's namespace-R symbols.
#if !defined(NoRcpp) && !defined(AI4BAYESCODE_RCPP_MODULE)
#define NoRcpp
#endif

#ifndef SOFTBART_MODEL_H_
#define SOFTBART_MODEL_H_

#include <armadillo>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cmath>

#include "SOFTBART_VENDOR/soft_bart.h"

namespace softbart {

// ---- soft-tree serialization (pure; unchanged from Rcpp version) ----
inline void serialize_node_(std::ostream& os, const ::Node* n) {
  if (n == nullptr) return;
  if (n->is_leaf) { os << "L " << std::setprecision(17) << n->mu << "\n"; return; }
  os << "B " << n->var << " " << std::setprecision(17) << n->val << " "
     << std::setprecision(17) << n->tau << "\n";
  serialize_node_(os, n->left); serialize_node_(os, n->right);
}
inline ::Node* deserialize_node_(std::istream& is) {
  std::string token;
  if (!(is >> token)) return nullptr;
  if (token == "L") { ::Node* leaf = new ::Node(); leaf->is_leaf=true; leaf->is_root=false; is >> leaf->mu; return leaf; }
  else if (token == "B") {
    ::Node* br = new ::Node(); br->is_leaf=false; br->is_root=false;
    is >> br->var >> br->val >> br->tau;
    br->left = deserialize_node_(is); br->right = deserialize_node_(is);
    if (br->left) br->left->parent=br; if (br->right) br->right->parent=br;
    return br;
  }
  return nullptr;
}
inline std::string serialize_forest_to_str(const std::vector<::Node*>& trees) {
  std::ostringstream os; os << trees.size() << "\n";
  for (auto* t : trees) { serialize_node_(os, t); os << "---\n"; }
  return os.str();
}
inline std::vector<::Node*> deserialize_forest_from_str(const std::string& s) {
  std::istringstream is(s); std::size_t n_trees=0; is >> n_trees;
  std::vector<::Node*> trees; trees.reserve(n_trees);
  for (std::size_t t=0;t<n_trees;++t) {
    ::Node* root = deserialize_node_(is);
    if (root) { root->is_root=true; root->parent=nullptr; }
    trees.push_back(root); std::string sep; is >> sep;
  }
  return trees;
}
inline void free_forest_(std::vector<::Node*>& trees) { for (auto* t : trees) delete t; trees.clear(); }
inline double soft_activation_(double x, double c, double tau) { return 1.0/(1.0+std::exp((x-c)/tau)); }
inline double soft_node_predict_(const ::Node* n, const double* x_row, double weight) {
  if (n->is_leaf) return weight * n->mu;
  const double pl = soft_activation_(x_row[n->var], n->val, n->tau);
  return soft_node_predict_(n->left, x_row, weight*pl) + soft_node_predict_(n->right, x_row, weight*(1.0-pl));
}
inline bool validate_soft_node_(const ::Node* n, int p_hint, int depth, int depth_max, double tau_max, std::string& msg) {
  if (n == nullptr) { msg="null node"; return false; }
  if (depth > depth_max) { msg="tree depth exceeds "+std::to_string(depth_max); return false; }
  if (n->is_leaf) {
    if (n->left||n->right) { msg="leaf has non-null children"; return false; }
    if (!std::isfinite(n->mu)) { msg="non-finite leaf mu"; return false; }
    return true;
  }
  if (!n->left||!n->right) { msg="branch missing child"; return false; }
  if (n->left==n->right) { msg="branch identical children"; return false; }
  if (n->var<0 || n->var>=p_hint) { msg="var out of range"; return false; }
  if (!std::isfinite(n->val)) { msg="non-finite val"; return false; }
  if (!std::isfinite(n->tau)||!(n->tau>0.0)||!(n->tau<=tau_max)) { msg="tau out of range"; return false; }
  return validate_soft_node_(n->left,p_hint,depth+1,depth_max,tau_max,msg)
      && validate_soft_node_(n->right,p_hint,depth+1,depth_max,tau_max,msg);
}
inline arma::vec predict_softbart_forest_(const std::vector<::Node*>& trees, const arma::mat& X) {
  const arma::uword n=X.n_rows; arma::vec out(n, arma::fill::zeros);
  std::vector<double> x_row(X.n_cols);
  for (arma::uword i=0;i<n;++i) { for (arma::uword j=0;j<X.n_cols;++j) x_row[j]=X(i,j);
    double f=0.0; for (const auto* t : trees) f += soft_node_predict_(t, x_row.data(), 1.0); out[i]=f; }
  return out;
}

// ---- history bundle (replaces Rcpp::List) ----
struct SoftbartHistory {
  std::vector<std::string> tree_history;
  arma::vec  sigma_history, sigma_mu_history;
  arma::mat  s_history;            // n_iter x p
  arma::umat var_counts_history;   // n_iter x p
  std::size_t p = 0;
};

// =====================================================================
class softbart_model {
 public:
  softbart_model(const arma::mat& X, const arma::vec& Y,
                 int ntrees=50, double k=2.0, double sigma_hat=-1.0,
                 double alpha=1.0, double beta=2.0, double gamma=0.95,
                 double width=0.1, double shape=1.0, double tau_rate=10.0,
                 double alpha_scale_=-1.0, double alpha_shape_1=0.5,
                 double alpha_shape_2=1.0, bool dart=false, bool verbose=false)
      : N_((std::size_t)X.n_rows), p_((std::size_t)X.n_cols), ntrees_(ntrees),
        dart_(dart), forest_(nullptr), history_recording_(false)
  {
    (void)verbose;
    X_arma_ = X; Y_arma_ = Y; offset_arma_ = arma::zeros<arma::vec>(N_);
    if (sigma_hat <= 0.0)    sigma_hat    = arma::stddev(Y);
    if (alpha_scale_ <= 0.0) alpha_scale_ = (double)p_;

    ::Hypers h;
    h.alpha=alpha; h.beta=beta; h.gamma=gamma;
    h.sigma=sigma_hat; h.sigma_hat=sigma_hat;
    h.sigma_mu = 0.5/(k*std::sqrt((double)ntrees));
    h.shape=shape; h.width=width; h.num_tree=ntrees;
    h.tau_rate=tau_rate; h.num_tree_prob=2.0/(double)ntrees; h.temperature=1.0;
    h.alpha_scale=alpha_scale_; h.alpha_shape_1=alpha_shape_1; h.alpha_shape_2=alpha_shape_2;
    h.group = arma::regspace<arma::uvec>(0, (arma::uword)p_-1);
    h.finalize_from_group();

    ::Opts o;  // default ctor sets sane defaults, then override:
    o.update_sigma_mu=true; o.update_s=dart; o.update_alpha=dart;
    o.update_beta=false; o.update_gamma=false; o.update_tau=true;
    o.update_tau_mean=false; o.update_num_tree=false; o.update_sigma=true;
    o.num_burn=(dart?0:1); o.num_thin=1; o.num_save=1; o.num_print=100; o.cache_trees=false;

    forest_.reset(new ::Forest(h, o));
  }
  ~softbart_model() = default;
  softbart_model(const softbart_model&) = delete;
  softbart_model& operator=(const softbart_model&) = delete;

  // ---- plug-in API ----
  void set_X(const arma::mat& X_new) {
    if ((std::size_t)X_new.n_cols != p_) throw std::invalid_argument("softbart::set_X: ncol != p");
    if ((std::size_t)X_new.n_rows != N_) throw std::invalid_argument("softbart::set_X: nrow != N");
    X_arma_ = X_new;
  }
  void set_Y(const arma::vec& Y_new, bool center_Y=false) {
    if ((std::size_t)Y_new.n_elem != N_) throw std::invalid_argument("softbart::set_Y: length != N");
    Y_arma_ = center_Y ? (Y_new - arma::mean(Y_new)) : Y_new;
  }
  void set_data(const arma::mat& X_new, const arma::vec& Y_new, bool center_Y=false) { set_X(X_new); set_Y(Y_new, center_Y); }
  void set_offset(const arma::vec& off) {
    if (off.n_elem == 0) { offset_arma_ = arma::zeros<arma::vec>(N_); return; }
    if ((std::size_t)off.n_elem != N_) throw std::invalid_argument("softbart::set_offset: length != N");
    offset_arma_ = off;
  }
  arma::vec get_s() { return forest_->get_s(); }
  void set_s(const arma::vec& s) {
    if ((std::size_t)s.n_elem != p_) throw std::invalid_argument("softbart::set_s: length != p");
    arma::vec sa = s; forest_->set_s(sa);
  }
  void startdart() { /* DART set at construction */ }

  void update_step() {
    arma::mat X_dummy(0, p_); arma::vec Y_work = Y_arma_ - offset_arma_;
    forest_->do_gibbs(X_arma_, Y_work, X_dummy, 1);
    record_history_();
  }
  void update_step(double sigma_external) {
    forest_->set_sigma(sigma_external);
    arma::mat X_dummy(0, p_); arma::vec Y_work = Y_arma_ - offset_arma_;
    forest_->do_gibbs(X_arma_, Y_work, X_dummy, 1);
    forest_->set_sigma(sigma_external);
    record_history_();
  }

  void set_history(bool on) { history_recording_ = on; }
  bool history_recording() const { return history_recording_; }
  void clear_history() {
    tree_history_.clear(); sigma_history_.clear(); sigma_mu_history_.clear();
    s_history_.clear(); var_counts_history_.clear();
  }
  SoftbartHistory get_history() const {
    const std::size_t n_iter = sigma_history_.size();
    SoftbartHistory h; h.p = p_;
    h.tree_history = tree_history_;
    h.sigma_history = arma::vec(sigma_history_.size());
    h.sigma_mu_history = arma::vec(sigma_mu_history_.size());
    for (std::size_t i=0;i<sigma_history_.size();++i) h.sigma_history(i)=sigma_history_[i];
    for (std::size_t i=0;i<sigma_mu_history_.size();++i) h.sigma_mu_history(i)=sigma_mu_history_[i];
    h.s_history.set_size(n_iter, p_); h.var_counts_history.set_size(n_iter, p_);
    for (std::size_t i=0;i<n_iter;++i) for (std::size_t j=0;j<p_;++j) {
      h.s_history(i,j) = (j<s_history_[i].size())? s_history_[i][j] : 0.0;
      h.var_counts_history(i,j) = (j<var_counts_history_[i].size())? (arma::uword)var_counts_history_[i][j] : 0u;
    }
    return h;
  }
  std::string get_tree() const { return serialize_forest_to_str(forest_->get_trees()); }
  void set_tree(const std::string& str) {
    if (str.empty()) throw std::invalid_argument("softbart::set_tree: input is empty");
    std::istringstream is(str);
    long T_signed=-1;
    if (!(is >> T_signed)) throw std::invalid_argument("softbart::set_tree: failed to parse tree count");
    if (T_signed < 0) throw std::invalid_argument("softbart::set_tree: tree count negative");
    const std::size_t T=(std::size_t)T_signed;
    if (T != (std::size_t)ntrees_) throw std::invalid_argument("softbart::set_tree: tree count != ntrees");
    std::vector<::Node*> new_trees; new_trees.reserve(T);
    auto cleanup=[&](){ for (auto* n : new_trees) delete n; };
    for (std::size_t t=0;t<T;++t) {
      if (is.eof()) { cleanup(); throw std::invalid_argument("softbart::set_tree: stream ended early"); }
      ::Node* root = deserialize_node_(is);
      if (!root) { cleanup(); throw std::invalid_argument("softbart::set_tree: failed to parse a tree"); }
      root->is_root=true; root->parent=nullptr; root->current_weight=1.0;
      std::string sep;
      if (!(is>>sep)||sep!="---") { delete root; cleanup(); throw std::invalid_argument("softbart::set_tree: expected '---'"); }
      new_trees.push_back(root);
    }
    const int depth_max=50; const double tau_max=1e10;
    for (std::size_t t=0;t<T;++t) { std::string msg;
      if (!validate_soft_node_(new_trees[t],(int)p_,0,depth_max,tau_max,msg)) { cleanup(); throw std::invalid_argument("softbart::set_tree: invalid tree -- "+msg); } }
    std::string trailing;
    if (is>>trailing) { cleanup(); throw std::invalid_argument("softbart::set_tree: unexpected trailing content"); }
    forest_->set_trees(std::move(new_trees));
  }
  arma::vec predict(const arma::mat& X_test) const {
    if ((std::size_t)X_test.n_cols != p_) throw std::invalid_argument("softbart::predict: ncol != p");
    arma::mat Xt = X_test;
    return forest_->do_predict(Xt);
  }
  double current_sigma() const    { return forest_->get_sigma(); }
  double current_sigma_mu() const { return forest_->get_sigma_mu(); }
  arma::vec  current_var_probs()  { return forest_->get_s(); }
  arma::uvec current_var_counts() { return forest_->get_counts(); }
  std::size_t N() const { return N_; }
  std::size_t p() const { return p_; }
  std::size_t ntrees() const { return ntrees_; }
  ::Forest* forest() { return forest_.get(); }

 private:
  void record_history_() {
    if (!history_recording_) return;
    tree_history_.push_back(serialize_forest_to_str(forest_->get_trees()));
    sigma_history_.push_back(forest_->get_sigma());
    sigma_mu_history_.push_back(forest_->get_sigma_mu());
    arma::vec s = forest_->get_s(); s_history_.emplace_back(s.begin(), s.end());
    arma::uvec cnt = forest_->get_counts();
    std::vector<int> ci(cnt.size()); for (std::size_t j=0;j<cnt.size();++j) ci[j]=(int)cnt[j];
    var_counts_history_.push_back(std::move(ci));
  }

  std::size_t N_, p_;
  int ntrees_; bool dart_;
  std::unique_ptr<::Forest> forest_;
  arma::mat X_arma_; arma::vec Y_arma_; arma::vec offset_arma_;
  bool history_recording_;
  std::vector<std::string> tree_history_;
  std::vector<double> sigma_history_, sigma_mu_history_;
  std::vector<std::vector<double> > s_history_;
  std::vector<std::vector<int> > var_counts_history_;
};

}  // namespace softbart
#endif  // SOFTBART_MODEL_H_
