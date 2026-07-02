// genBART build file — single sourceCpp entry point.
// Defines R-facing wrappers for the built-in likelihoods.

// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

#include "src/genbart_model.h"

using namespace Rcpp;

// genbart_normal_MCMC: Gaussian regression via RJMCMC.
// Sparsity (DART): set dart=TRUE to enable Linero 2018 Dirichlet prior on split
// variable probabilities.  By default theta is updated via grid sampling; set
// dart_const_theta=TRUE to hold it at dart_theta_init.  Sparsity kicks in at
// dart_start_iter (default: nburn/2, matching Linero 2018 warm-up).
// [[Rcpp::export]]
List genbart_normal_MCMC(NumericMatrix X, NumericVector Y,
                  Nullable<NumericMatrix> X_test_ = R_NilValue,
                  Nullable<NumericVector> offset_ = R_NilValue,
                  std::size_t nburn = 1000,
                  std::size_t npost = 1000,
                  std::size_t ntrees = 50,
                  double sigma_init = -1.0,  // -1 -> sd(Y)
                  double nu = 3.0,
                  double q  = 0.90,
                  double sigma_mu_init = -1.0,
                  bool   adaptive_sigma_mu = true,
                  double k_half_cauchy = 1.0,
                  double gamma_depth = 0.95,
                  double beta_depth  = 2.0,
                  int    min_leaf_n  = 5,
                  int    max_depth   = 20,
                  bool   dart            = false,
                  double dart_a          = 0.5,
                  double dart_b          = 1.0,
                  double dart_rho        = -1.0,  // <=0 -> p
                  double dart_theta_init = 1.0,
                  bool   dart_const_theta= false,

                  int    numcut      = 100,
                  bool   usequants   = false,
                  bool   verbose     = false,
                  std::size_t print_every = 100)
{
  if (sigma_init <= 0.0) sigma_init = Rcpp::sd(Y);
  const double c_half_cauchy = k_half_cauchy / std::sqrt((double)ntrees);
  if (sigma_mu_init <= 0.0) {
    double ymin = Rcpp::min(Y), ymax = Rcpp::max(Y);
    double bart_default = (ymax - ymin) / (4.0 * std::sqrt((double)ntrees));
    sigma_mu_init = std::max(c_half_cauchy, bart_default);
  }
  NumericVector qch = NumericVector::create(1.0 - q);
  double qchi = Rcpp::qchisq(qch, nu, true, false)[0];
  double lambda = (sigma_init * sigma_init * qchi) / nu;

  genbart::lik::normal_lik lik(sigma_init, nu, lambda);

  genbart::rjmcmc_hypers h;
  h.gamma_depth        = gamma_depth;
  h.beta_depth         = beta_depth;
  h.sigma_mu           = sigma_mu_init;
  h.adaptive_sigma_mu  = adaptive_sigma_mu;
  h.half_cauchy_c      = c_half_cauchy;
  h.min_leaf_n         = min_leaf_n;
  h.max_depth          = max_depth;
  h.dart_requested    = dart;
  h.dart_a             = dart_a;
  h.dart_b             = dart_b;
  h.dart_rho           = dart_rho;
  h.dart_theta_init    = dart_theta_init;
  h.dart_const_theta   = dart_const_theta;

  // Offset.
  NumericVector off = offset_.isNotNull() ? NumericVector(offset_) : NumericVector(0);

  // Model.
  genbart::genbart_model model(X, Y, off, &lik, h, ntrees, numcut, usequants, false);

  // Test set.
  NumericMatrix X_test = X_test_.isNotNull()
      ? NumericMatrix(X_test_) : NumericMatrix(0, X.ncol());

  List out = model.run_mcmc(nburn, npost, X_test, verbose, print_every, (long)nburn / 2);

  // Tag the model hyperparameters for diagnostics.
  out["sigma_init"]        = sigma_init;
  out["sigma_mu_init"]     = sigma_mu_init;
  out["half_cauchy_c"]     = c_half_cauchy;
  out["adaptive_sigma_mu"] = adaptive_sigma_mu;
  out["lambda"]            = lambda;
  out["nu"]                = nu;
  out["dart"]              = dart;
  return out;
}

// genbart_logistic_MCMC: Bernoulli(sigma(r(X))) classification via RJMCMC.
// Y must be 0/1.  Supports half-Cauchy(0, c) hyperprior on sigma_mu and
// optional DART sparsity prior (Linero 2018 / paper §3.3).
// [[Rcpp::export]]
List genbart_logistic_MCMC(NumericMatrix X, NumericVector Y,
                    Nullable<NumericMatrix> X_test_ = R_NilValue,
                    Nullable<NumericVector> offset_ = R_NilValue,
                    std::size_t nburn = 1000,
                    std::size_t npost = 1000,
                    std::size_t ntrees = 50,
                    double sigma_mu_init = -1.0,
                    bool   adaptive_sigma_mu = true,
                    double k_half_cauchy = 1.0,
                    bool   dart            = false,
                    double dart_a          = 0.5,
                    double dart_b          = 1.0,
                    double dart_rho        = -1.0,
                    double dart_theta_init = 1.0,
                    bool   dart_const_theta= false,

                    int    numcut      = 100,
                    bool   usequants   = false,
                    bool   verbose     = false,
                    std::size_t print_every = 100)
{
  for (int i = 0; i < Y.size(); ++i) {
    if (Y[i] != 0.0 && Y[i] != 1.0)
      Rcpp::stop("genbart_logistic_MCMC: Y must be 0 or 1");
  }
  const double c = k_half_cauchy / std::sqrt((double)ntrees);
  if (sigma_mu_init <= 0.0) sigma_mu_init = std::max(c, 0.1);

  genbart::lik::logistic_lik lik;
  genbart::rjmcmc_hypers h;
  h.sigma_mu           = sigma_mu_init;
  h.adaptive_sigma_mu  = adaptive_sigma_mu;
  h.half_cauchy_c      = c;
  h.min_leaf_n         = 5;
  h.dart_requested    = dart;
  h.dart_a             = dart_a;
  h.dart_b             = dart_b;
  h.dart_rho           = dart_rho;
  h.dart_theta_init    = dart_theta_init;
  h.dart_const_theta   = dart_const_theta;
  // Clamp Fisher info floor so Laplace variance doesn't blow up when
  // lam -> large and sigma(1-sigma) -> 0.
  h.lap_opts.v_precision_lb = 1e-4;

  NumericVector off = offset_.isNotNull() ? NumericVector(offset_) : NumericVector(0);
  genbart::genbart_model model(X, Y, off, &lik, h, ntrees, numcut, usequants, false);
  NumericMatrix X_test = X_test_.isNotNull()
      ? NumericMatrix(X_test_) : NumericMatrix(0, X.ncol());
  List out = model.run_mcmc(nburn, npost, X_test, verbose, print_every, (long)nburn / 2);
  out["sigma_mu_init"] = sigma_mu_init;
  out["half_cauchy_c"] = c;
  out["dart"] = dart;
  out["link"] = "logistic";
  return out;
}

// genbart_poisson_MCMC: Y ~ Poisson(exp(r(X))).
// Full paper-faithful defaults: Normal(0, sigma_mu^2) leaf prior with half-
// Cauchy(0, c) hyperprior (c = k/sqrt(ntrees)), and optional DART (Linero 2018)
// sparsity prior on split variables.
// [[Rcpp::export]]
List genbart_poisson_MCMC(NumericMatrix X, NumericVector Y,
                   Nullable<NumericMatrix> X_test_ = R_NilValue,
                   Nullable<NumericVector> offset_ = R_NilValue,
                   std::size_t nburn = 1000,
                   std::size_t npost = 1000,
                   std::size_t ntrees = 50,
                   double sigma_mu_init = -1.0,
                   bool   adaptive_sigma_mu = true,
                   double k_half_cauchy = 1.0,
                   bool   dart            = false,
                   double dart_a          = 0.5,
                   double dart_b          = 1.0,
                   double dart_rho        = -1.0,
                   double dart_theta_init = 1.0,
                   bool   dart_const_theta= false,

                   int    numcut = 100,
                   bool   usequants = false,
                   bool   verbose = false)
{
  for (int i = 0; i < Y.size(); ++i) {
    if (Y[i] < 0 || Y[i] != std::floor(Y[i]))
      Rcpp::stop("genbart_poisson_MCMC: Y must be non-negative integer counts");
  }
  const double c = k_half_cauchy / std::sqrt((double)ntrees);
  if (sigma_mu_init <= 0.0) sigma_mu_init = c;

  genbart::lik::poisson_lik lik;
  genbart::rjmcmc_hypers h;
  h.sigma_mu           = sigma_mu_init;
  h.adaptive_sigma_mu  = adaptive_sigma_mu;
  h.half_cauchy_c      = c;
  h.min_leaf_n         = 5;
  h.dart_requested    = dart;
  h.dart_a             = dart_a;
  h.dart_b             = dart_b;
  h.dart_rho           = dart_rho;
  h.dart_theta_init    = dart_theta_init;
  h.dart_const_theta   = dart_const_theta;

  NumericVector off = offset_.isNotNull() ? NumericVector(offset_) : NumericVector(0);
  genbart::genbart_model model(X, Y, off, &lik, h, ntrees, numcut, usequants, false);

  NumericMatrix X_test = X_test_.isNotNull()
      ? NumericMatrix(X_test_) : NumericMatrix(0, X.ncol());

  List out = model.run_mcmc(nburn, npost, X_test, verbose, 100, (long)nburn / 2);
  out["sigma_mu_init"]      = sigma_mu_init;
  out["half_cauchy_c"]      = c;
  out["adaptive_sigma_mu"]  = adaptive_sigma_mu;
  out["dart"]               = dart;
  out["link"]               = "log-Poisson";
  return out;
}

// genbart_nb_MCMC: Y ~ NegBin(mu=exp(r(X)), kappa).  Overdispersed count regression.
// [[Rcpp::export]]
List genbart_nb_MCMC(NumericMatrix X, NumericVector Y,
              Nullable<NumericMatrix> X_test_ = R_NilValue,
              Nullable<NumericVector> offset_ = R_NilValue,
              std::size_t nburn = 1000,
              std::size_t npost = 1000,
              std::size_t ntrees = 50,
              double sigma_mu_init = -1.0,
              bool   adaptive_sigma_mu = true,
              double k_half_cauchy = 1.0,
              double kappa_init = 1.0,
              double prior_a_kappa = 0.01,
              double prior_b_kappa = 0.01,
              bool   dart            = false,
              double dart_a          = 0.5,
              double dart_b          = 1.0,
              double dart_rho        = -1.0,
              double dart_theta_init = 1.0,
              bool   dart_const_theta= false,

              int    numcut = 100,
              bool   usequants = false,
              bool   verbose = false)
{
  for (int i = 0; i < Y.size(); ++i) {
    if (Y[i] < 0 || Y[i] != std::floor(Y[i]))
      Rcpp::stop("genbart_nb_MCMC: Y must be non-negative integer counts");
  }
  const double c = k_half_cauchy / std::sqrt((double)ntrees);
  if (sigma_mu_init <= 0.0) sigma_mu_init = c;

  genbart::lik::negative_binomial_lik lik(kappa_init, 0.3, prior_a_kappa, prior_b_kappa);
  genbart::rjmcmc_hypers h;
  h.sigma_mu           = sigma_mu_init;
  h.adaptive_sigma_mu  = adaptive_sigma_mu;
  h.half_cauchy_c      = c;
  h.min_leaf_n         = 5;
  h.dart_requested    = dart;
  h.dart_a             = dart_a;
  h.dart_b             = dart_b;
  h.dart_rho           = dart_rho;
  h.dart_theta_init    = dart_theta_init;
  h.dart_const_theta   = dart_const_theta;

  NumericVector off = offset_.isNotNull() ? NumericVector(offset_) : NumericVector(0);
  genbart::genbart_model model(X, Y, off, &lik, h, ntrees, numcut, usequants, false);

  NumericMatrix X_test = X_test_.isNotNull()
      ? NumericMatrix(X_test_) : NumericMatrix(0, X.ncol());
  List out = model.run_mcmc(nburn, npost, X_test, verbose, 100, (long)nburn / 2);
  out["sigma_mu_init"]  = sigma_mu_init;
  out["half_cauchy_c"]  = c;
  out["dart"]           = dart;
  out["link"]           = "log-NegBin";
  return out;
}

// genbart_heteroscedastic_MCMC: Y ~ Normal(exp(r(X)), phi * exp(r(X))) -- paper §4.2.
// [[Rcpp::export]]
List genbart_heteroscedastic_MCMC(NumericMatrix X, NumericVector Y,
                           Nullable<NumericMatrix> X_test_ = R_NilValue,
                           Nullable<NumericVector> offset_ = R_NilValue,
                           std::size_t nburn = 1000,
                           std::size_t npost = 1000,
                           std::size_t ntrees = 50,
                           double phi_init = 1.0,
                           double a0 = 1.0,
                           double b0 = 1.0,
                           double sigma_mu_init = -1.0,
                           bool   adaptive_sigma_mu = true,
                           double k_half_cauchy = 1.0,
                           bool   dart            = false,
                           double dart_a          = 0.5,
                           double dart_b          = 1.0,
                           double dart_rho        = -1.0,
                           double dart_theta_init = 1.0,
                           bool   dart_const_theta= false,

                           int    numcut = 100,
                           bool   usequants = false,
                           bool   verbose = false)
{
  const double c = k_half_cauchy / std::sqrt((double)ntrees);
  if (sigma_mu_init <= 0.0) sigma_mu_init = std::max(c, 0.05);

  genbart::lik::heteroscedastic_lik lik(phi_init, a0, b0);
  genbart::rjmcmc_hypers h;
  h.sigma_mu           = sigma_mu_init;
  h.adaptive_sigma_mu  = adaptive_sigma_mu;
  h.half_cauchy_c      = c;
  h.min_leaf_n         = 5;
  h.dart_requested    = dart;
  h.dart_a             = dart_a;
  h.dart_b             = dart_b;
  h.dart_rho           = dart_rho;
  h.dart_theta_init    = dart_theta_init;
  h.dart_const_theta   = dart_const_theta;

  NumericVector off = offset_.isNotNull() ? NumericVector(offset_) : NumericVector(0);
  genbart::genbart_model model(X, Y, off, &lik, h, ntrees, numcut, usequants, false);
  NumericMatrix X_test = X_test_.isNotNull()
      ? NumericMatrix(X_test_) : NumericMatrix(0, X.ncol());
  List out = model.run_mcmc(nburn, npost, X_test, verbose, 100, (long)nburn / 2);
  out["sigma_mu_init"] = sigma_mu_init;
  out["half_cauchy_c"] = c;
  out["dart"] = dart;
  out["model"] = "heteroscedastic Normal (V(m)=m)";
  return out;
}

// ---------------------------------------------------------------------------
// AFT log-logistic (paper §4.3). y_log = log(observed time), delta = event.
// [[Rcpp::export]]
List genbart_aft_log_logistic_MCMC(NumericMatrix X, NumericVector y_log,
                            NumericVector delta,
                            Nullable<NumericMatrix> X_test_ = R_NilValue,
                            Nullable<NumericVector> offset_ = R_NilValue,
                            std::size_t nburn = 1000,
                            std::size_t npost = 1000,
                            std::size_t ntrees = 50,
                            double sigma_init = 1.0,
                            double sigma_mu_init = -1.0,
                            bool   adaptive_sigma_mu = true,
                            double k_half_cauchy = 1.0,
                            bool   dart            = false,
                            double dart_a          = 0.5,
                            double dart_b          = 1.0,
                            double dart_rho        = -1.0,
                            double dart_theta_init = 1.0,
                            bool   dart_const_theta= false,

                            int    numcut = 100,
                            bool   verbose = false)
{
  if (delta.size() != y_log.size())
    Rcpp::stop("aft_log_logistic: delta length must match y_log");
  const double c = k_half_cauchy / std::sqrt((double)ntrees);
  if (sigma_mu_init <= 0.0) {
    double range_y = Rcpp::max(y_log) - Rcpp::min(y_log);
    sigma_mu_init = std::max(c, range_y / (4.0 * std::sqrt((double)ntrees)));
  }

  genbart::lik::aft_log_logistic_lik lik(sigma_init);
  std::vector<double> delta_vec(delta.begin(), delta.end());
  lik.set_delta(delta_vec);

  genbart::rjmcmc_hypers h;
  h.sigma_mu           = sigma_mu_init;
  h.adaptive_sigma_mu  = adaptive_sigma_mu;
  h.half_cauchy_c      = c;
  h.min_leaf_n         = 5;
  h.dart_requested    = dart;
  h.dart_a             = dart_a;
  h.dart_b             = dart_b;
  h.dart_rho           = dart_rho;
  h.dart_theta_init    = dart_theta_init;
  h.dart_const_theta   = dart_const_theta;

  NumericVector off = offset_.isNotNull() ? NumericVector(offset_) : NumericVector(0);
  genbart::genbart_model model(X, y_log, off, &lik, h, ntrees, numcut, false, false);
  NumericMatrix X_test = X_test_.isNotNull()
      ? NumericMatrix(X_test_) : NumericMatrix(0, X.ncol());
  List out = model.run_mcmc(nburn, npost, X_test, verbose, 100, (long)nburn / 2);
  out["sigma_mu_init"] = sigma_mu_init;
  out["half_cauchy_c"] = c;
  out["dart"] = dart;
  out["model"] = "AFT log-logistic";
  return out;
}

// ---------------------------------------------------------------------------
// AFT generalized gamma (paper §4.3).  2 nuisance: sigma, alpha.
// [[Rcpp::export]]
List genbart_aft_generalized_gamma_MCMC(NumericMatrix X, NumericVector y_log,
                                 NumericVector delta,
                                 Nullable<NumericMatrix> X_test_ = R_NilValue,
                                 Nullable<NumericVector> offset_ = R_NilValue,
                                 std::size_t nburn = 1000,
                                 std::size_t npost = 1000,
                                 std::size_t ntrees = 50,
                                 double sigma_init = 1.0,
                                 double alpha_init = 1.0,
                                 double alpha_max  = 40.0,
                                 double sigma_mu_init = -1.0,
                                 bool   adaptive_sigma_mu = true,
                                 double k_half_cauchy = 1.0,
                                 bool   dart            = false,
                                 double dart_a          = 0.5,
                                 double dart_b          = 1.0,
                                 double dart_rho        = -1.0,
                                 double dart_theta_init = 1.0,
                                 bool   dart_const_theta= false,

                                 int    numcut = 100,
                                 bool   verbose = false)
{
  if (delta.size() != y_log.size())
    Rcpp::stop("aft_gen_gamma: delta length must match y_log");
  const double c = k_half_cauchy / std::sqrt((double)ntrees);
  if (sigma_mu_init <= 0.0) {
    double range_y = Rcpp::max(y_log) - Rcpp::min(y_log);
    sigma_mu_init = std::max(c, range_y / (4.0 * std::sqrt((double)ntrees)));
  }
  genbart::lik::aft_generalized_gamma_lik lik(sigma_init, alpha_init, alpha_max);
  std::vector<double> delta_vec(delta.begin(), delta.end());
  lik.set_delta(delta_vec);

  genbart::rjmcmc_hypers h;
  h.sigma_mu           = sigma_mu_init;
  h.adaptive_sigma_mu  = adaptive_sigma_mu;
  h.half_cauchy_c      = c;
  h.min_leaf_n         = 5;
  h.dart_requested    = dart;
  h.dart_a             = dart_a;
  h.dart_b             = dart_b;
  h.dart_rho           = dart_rho;
  h.dart_theta_init    = dart_theta_init;
  h.dart_const_theta   = dart_const_theta;

  NumericVector off = offset_.isNotNull() ? NumericVector(offset_) : NumericVector(0);
  genbart::genbart_model model(X, y_log, off, &lik, h, ntrees, numcut, false, false);
  NumericMatrix X_test = X_test_.isNotNull()
      ? NumericMatrix(X_test_) : NumericMatrix(0, X.ncol());
  List out = model.run_mcmc(nburn, npost, X_test, verbose, 100, (long)nburn / 2);
  out["sigma_mu_init"] = sigma_mu_init;
  out["half_cauchy_c"] = c;
  out["dart"] = dart;
  out["alpha_max"]     = alpha_max;
  out["model"]         = "AFT generalized gamma";
  return out;
}

// ---------------------------------------------------------------------------
// Gamma shape regression (paper §4.4).  alpha(X) = exp(r(X)), beta global.
// [[Rcpp::export]]
List genbart_gamma_shape_MCMC(NumericMatrix X, NumericVector Y,
                       Nullable<NumericMatrix> X_test_ = R_NilValue,
                       Nullable<NumericVector> offset_ = R_NilValue,
                       std::size_t nburn = 1000,
                       std::size_t npost = 1000,
                       std::size_t ntrees = 50,
                       double beta_init = 1.0,
                       double a0 = 1.0,
                       double b0 = 1.0,
                       double sigma_mu_init = -1.0,
                       bool   adaptive_sigma_mu = true,
                       double k_half_cauchy = 1.0,
                       bool   dart            = false,
                       double dart_a          = 0.5,
                       double dart_b          = 1.0,
                       double dart_rho        = -1.0,
                       double dart_theta_init = 1.0,
                       bool   dart_const_theta= false,

                       int    numcut = 100,
                       bool   verbose = false)
{
  for (int i = 0; i < Y.size(); ++i) if (Y[i] <= 0.0)
    Rcpp::stop("genbart_gamma_shape_MCMC: Y must be strictly positive");
  const double c = k_half_cauchy / std::sqrt((double)ntrees);
  if (sigma_mu_init <= 0.0) {
    double log_range = std::log(Rcpp::max(Y) + 1.0) - std::log(Rcpp::min(Y) + 1e-6);
    sigma_mu_init = std::max(c, log_range / (4.0 * std::sqrt((double)ntrees)));
  }
  genbart::lik::gamma_shape_lik lik(beta_init, a0, b0);
  genbart::rjmcmc_hypers h;
  h.sigma_mu           = sigma_mu_init;
  h.adaptive_sigma_mu  = adaptive_sigma_mu;
  h.half_cauchy_c      = c;
  h.dart_requested    = dart;
  h.dart_a             = dart_a;
  h.dart_b             = dart_b;
  h.dart_rho           = dart_rho;
  h.dart_theta_init    = dart_theta_init;
  h.dart_const_theta   = dart_const_theta;

  NumericVector off = offset_.isNotNull() ? NumericVector(offset_) : NumericVector(0);
  genbart::genbart_model model(X, Y, off, &lik, h, ntrees, numcut, false, false);
  NumericMatrix X_test = X_test_.isNotNull()
      ? NumericMatrix(X_test_) : NumericMatrix(0, X.ncol());
  List out = model.run_mcmc(nburn, npost, X_test, verbose, 100, (long)nburn / 2);
  out["sigma_mu_init"] = sigma_mu_init;
  out["half_cauchy_c"] = c;
  out["dart"] = dart;
  out["model"] = "Gamma shape regression";
  return out;
}

// ---------------------------------------------------------------------------
// Beta regression (paper Discussion).  mu_i = sigma(r(X)), phi global.
// [[Rcpp::export]]
List genbart_beta_MCMC(NumericMatrix X, NumericVector Y,
                Nullable<NumericMatrix> X_test_ = R_NilValue,
                Nullable<NumericVector> offset_ = R_NilValue,
                std::size_t nburn = 1000,
                std::size_t npost = 1000,
                std::size_t ntrees = 50,
                double phi_init = 10.0,
                double sigma_mu_init = -1.0,
                bool   adaptive_sigma_mu = true,
                double k_half_cauchy = 1.0,
                bool   dart            = false,
                double dart_a          = 0.5,
                double dart_b          = 1.0,
                double dart_rho        = -1.0,
                double dart_theta_init = 1.0,
                bool   dart_const_theta= false,

                int    numcut = 100,
                bool   verbose = false)
{
  for (int i = 0; i < Y.size(); ++i) if (Y[i] <= 0.0 || Y[i] >= 1.0)
    Rcpp::stop("genbart_beta_MCMC: Y must be in (0, 1)");
  const double c = k_half_cauchy / std::sqrt((double)ntrees);
  if (sigma_mu_init <= 0.0) sigma_mu_init = std::max(c, 0.3);

  genbart::lik::beta_lik lik(phi_init);
  genbart::rjmcmc_hypers h;
  h.sigma_mu           = sigma_mu_init;
  h.adaptive_sigma_mu  = adaptive_sigma_mu;
  h.half_cauchy_c      = c;
  h.dart_requested    = dart;
  h.dart_a             = dart_a;
  h.dart_b             = dart_b;
  h.dart_rho           = dart_rho;
  h.dart_theta_init    = dart_theta_init;
  h.dart_const_theta   = dart_const_theta;
  // Fisher info can be tiny at extreme lam; raise precision floor.
  h.lap_opts.v_precision_lb = 1e-4;

  NumericVector off = offset_.isNotNull() ? NumericVector(offset_) : NumericVector(0);
  genbart::genbart_model model(X, Y, off, &lik, h, ntrees, numcut, false, false);
  NumericMatrix X_test = X_test_.isNotNull()
      ? NumericMatrix(X_test_) : NumericMatrix(0, X.ncol());
  List out = model.run_mcmc(nburn, npost, X_test, verbose, 100, (long)nburn / 2);
  out["sigma_mu_init"] = sigma_mu_init;
  out["half_cauchy_c"] = c;
  out["dart"] = dart;
  out["model"] = "Beta regression";
  return out;
}

// ---------------------------------------------------------------------------
// Beta-Binomial regression (paper Discussion).  Y_i ~ BetaBin(n_i, mu_i, M).
// [[Rcpp::export]]
List genbart_beta_binomial_MCMC(NumericMatrix X, NumericVector Y, NumericVector n_trials,
                         Nullable<NumericMatrix> X_test_ = R_NilValue,
                         Nullable<NumericVector> offset_ = R_NilValue,
                         std::size_t nburn = 1000,
                         std::size_t npost = 1000,
                         std::size_t ntrees = 50,
                         double M_init = 10.0,
                         double sigma_mu_init = -1.0,
                         bool   adaptive_sigma_mu = true,
                         double k_half_cauchy = 1.0,
                         bool   dart            = false,
                         double dart_a          = 0.5,
                         double dart_b          = 1.0,
                         double dart_rho        = -1.0,
                         double dart_theta_init = 1.0,
                         bool   dart_const_theta= false,

                         int    numcut = 100,
                         bool   verbose = false)
{
  if (n_trials.size() != Y.size())
    Rcpp::stop("genbart_beta_binomial_MCMC: n_trials length must match Y");
  for (int i = 0; i < Y.size(); ++i) {
    if (Y[i] < 0 || Y[i] > n_trials[i] || Y[i] != std::floor(Y[i]))
      Rcpp::stop("genbart_beta_binomial_MCMC: Y must be integer in [0, n_trials]");
  }
  const double c = k_half_cauchy / std::sqrt((double)ntrees);
  if (sigma_mu_init <= 0.0) sigma_mu_init = std::max(c, 0.3);

  genbart::lik::beta_binomial_lik lik(M_init);
  std::vector<double> n_vec(n_trials.begin(), n_trials.end());
  lik.set_n_trials(n_vec);

  genbart::rjmcmc_hypers h;
  h.sigma_mu           = sigma_mu_init;
  h.adaptive_sigma_mu  = adaptive_sigma_mu;
  h.half_cauchy_c      = c;
  h.dart_requested    = dart;
  h.dart_a             = dart_a;
  h.dart_b             = dart_b;
  h.dart_rho           = dart_rho;
  h.dart_theta_init    = dart_theta_init;
  h.dart_const_theta   = dart_const_theta;
  h.lap_opts.v_precision_lb = 1e-4;

  NumericVector off = offset_.isNotNull() ? NumericVector(offset_) : NumericVector(0);
  genbart::genbart_model model(X, Y, off, &lik, h, ntrees, numcut, false, false);
  NumericMatrix X_test = X_test_.isNotNull()
      ? NumericMatrix(X_test_) : NumericMatrix(0, X.ncol());
  List out = model.run_mcmc(nburn, npost, X_test, verbose, 100, (long)nburn / 2);
  out["sigma_mu_init"] = sigma_mu_init;
  out["half_cauchy_c"] = c;
  out["dart"] = dart;
  out["model"] = "Beta-Binomial regression";
  return out;
}

// ---------------------------------------------------------------------------
// genbart_normal_with_missing
// Demonstration of the plug-in API (set_Y + update_step) for a hierarchical /
// missing-data scenario.  At each MCMC iter:
//   1. update_step()         -- one Gibbs sweep on the tree ensemble
//   2. for i with NA(Y_i):   -- impute Y_i ~ Normal(f(X_i) + offset, sigma)
//   3. set_Y(Y_imputed)      -- plug the imputed response back in
// Returns posterior draws of r on all points and the imputed Y on missing ones.
// [[Rcpp::export]]
List genbart_normal_with_missing(
    NumericMatrix X,
    NumericVector Y_obs,       // Y_i or NA_REAL if missing
    Nullable<NumericMatrix> X_test_ = R_NilValue,
    std::size_t nburn = 1000,
    std::size_t npost = 1000,
    std::size_t ntrees = 50,
    double sigma_init = -1.0,
    double nu = 3.0,
    double q  = 0.90,
    int    numcut = 100,
    bool   verbose = false)
{
  const std::size_t N = X.nrow();

  // Split observed vs missing; initial imputation = mean of observed.
  std::vector<std::size_t> missing_idx;
  double y_mean = 0.0; std::size_t n_obs = 0;
  for (std::size_t i = 0; i < N; ++i) {
    if (R_IsNA(Y_obs[i]) || R_IsNaN(Y_obs[i])) missing_idx.push_back(i);
    else { y_mean += Y_obs[i]; ++n_obs; }
  }
  if (n_obs == 0) Rcpp::stop("Y_obs: all values are NA");
  y_mean /= n_obs;
  if (sigma_init <= 0.0) {
    double ss = 0.0;
    for (std::size_t i = 0; i < N; ++i)
      if (!(R_IsNA(Y_obs[i]) || R_IsNaN(Y_obs[i])))
        ss += (Y_obs[i] - y_mean) * (Y_obs[i] - y_mean);
    sigma_init = std::sqrt(ss / (n_obs - 1));
  }

  // Initial imputed Y: observed kept, missing set to y_mean.
  NumericVector Y_cur(N);
  for (std::size_t i = 0; i < N; ++i) {
    if (R_IsNA(Y_obs[i]) || R_IsNaN(Y_obs[i])) Y_cur[i] = y_mean;
    else                                       Y_cur[i] = Y_obs[i];
  }

  NumericVector qch = NumericVector::create(1.0 - q);
  double qchi = Rcpp::qchisq(qch, nu, true, false)[0];
  double lambda = (sigma_init * sigma_init * qchi) / nu;
  double ymin = Rcpp::min(Y_cur), ymax = Rcpp::max(Y_cur);
  double sigma_mu = (ymax - ymin) / (4.0 * std::sqrt((double)ntrees));
  if (sigma_mu <= 0) sigma_mu = 0.1;

  genbart::lik::normal_lik lik(sigma_init, nu, lambda);
  genbart::rjmcmc_hypers h;
  h.sigma_mu = sigma_mu;
  h.min_leaf_n = 5;

  NumericVector off(0);
  genbart::genbart_model model(X, Y_cur, off, &lik, h, ntrees, numcut, false, false);

  NumericMatrix X_test = X_test_.isNotNull()
      ? NumericMatrix(X_test_) : NumericMatrix(0, X.ncol());
  const std::size_t Nt = (std::size_t)X_test.nrow();

  NumericMatrix f_train_draws(npost, N);
  NumericMatrix f_test_draws (npost, Nt);
  NumericMatrix Y_imp_draws  (npost, missing_idx.size());
  NumericVector sigma_draws  (npost);

  genbart::arn gen_local;

  for (std::size_t it = 0; it < nburn + npost; ++it) {
    if (verbose && it % 100 == 0) Rcpp::Rcout << "iter " << it << std::endl;

    // Step 1: tree Gibbs + nuisance (sigma) update
    model.update_step();

    // Step 2: impute missing Y | f(X), sigma
    Rcpp::NumericVector f_now = model.current_f_train();
    const double sigma = lik.sigma();
    for (std::size_t k = 0; k < missing_idx.size(); ++k) {
      const std::size_t i = missing_idx[k];
      Y_cur[i] = f_now[i] + sigma * gen_local.normal();
    }
    // Step 3: plug back in
    model.set_Y(Y_cur);

    // Record
    if (it >= nburn) {
      const std::size_t k = it - nburn;
      for (std::size_t i = 0; i < N; ++i) f_train_draws(k, i) = f_now[i];
      if (Nt > 0) {
        NumericVector ft = model.predict_once(X_test);
        for (std::size_t i = 0; i < Nt; ++i) f_test_draws(k, i) = ft[i];
      }
      for (std::size_t j = 0; j < missing_idx.size(); ++j) {
        Y_imp_draws(k, j) = Y_cur[missing_idx[j]];
      }
      sigma_draws[k] = sigma;
    }
  }

  // Report which indices were imputed.
  IntegerVector miss_idx_R(missing_idx.size());
  for (std::size_t j = 0; j < missing_idx.size(); ++j)
    miss_idx_R[j] = (int)missing_idx[j] + 1;  // 1-based for R

  return List::create(
      Named("f_train")       = f_train_draws,
      Named("f_test")        = f_test_draws,
      Named("Y_imputed")     = Y_imp_draws,
      Named("sigma")         = sigma_draws,
      Named("missing_index") = miss_idx_R,
      Named("n_missing")     = (int)missing_idx.size());
}
