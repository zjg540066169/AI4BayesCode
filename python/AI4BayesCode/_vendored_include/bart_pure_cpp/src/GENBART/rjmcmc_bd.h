/*
 *  Copyright (C) 2026 AI4BayesCode
 *
 *  GPL v2 or later.  See LICENSE for full text.
 */

/*
 * rjmcmc_bd.h -- BIRTH / DEATH / CHANGE tree moves for generalized BART,
 *                following Linero (2022) Proposition 1 and Algorithm 2.
 *
 * This file contains:
 *   - partition helpers (associate observations to leaves)
 *   - leaf-prior log-density helpers
 *   - three MH moves (propose_birth, propose_death, propose_change)
 *   - the dispatch function draw_tree() that picks one at random
 *
 * Design principles:
 *   - All ratios computed in LOG space.
 *   - Laplace proposals via laplace_leaf() with step-halving & NaN guards.
 *   - Every move returns an `accepted` flag; the caller updates the tree
 *     in-place only on accept (via tree::birthp / deathp).
 *   - NaN/infinity anywhere in log_ratio -> automatic reject.  Never let
 *     a bad ratio silently accept.
 */

#ifndef GENBART_RJMCMC_BD_H_
#define GENBART_RJMCMC_BD_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include "BART/tree.h"
#include "BART/info.h"
#include "BART/rn.h"
#include "laplace.h"
#include "likelihood_interface.h"

namespace genbart {

// ---------------------------------------------------------------------------
// Hyperparameters controlling tree prior and move mechanics.
// Follows the Chipman 2010 defaults: gamma=0.95, beta=2, T in {50, 200}.
// ---------------------------------------------------------------------------
struct rjmcmc_hypers {
  double gamma_depth  = 0.95;   // tree-prior rho_d = gamma * (1+d)^{-beta}
  double beta_depth   = 2.0;
  double sigma_mu     = 0.1;    // leaf-prior sd: Normal(0, sigma_mu^2).
                                // When `adaptive_sigma_mu` is true, this is
                                // MUTATED by Gibbs each iter; pass the INITIAL
                                // value here.
  double p_birth      = 0.25;   // P(propose BIRTH)
  double p_death      = 0.25;   // P(propose DEATH)
  double p_change     = 0.5;    // P(propose CHANGE) -- rest
  int    min_leaf_n   = 5;      // minimum observations per leaf after BIRTH
  int    max_depth    = 20;     // absolute cap to prevent runaway
  // ----- half-Cauchy hyperprior on sigma_mu (Linero 2022 §3.3) -----
  // If true, sigma_mu is sampled each iter from its full conditional given
  // the scale mixture representation
  //   sigma_mu^2 | xi ~ Inv-Gamma(1/2, 1/xi),  xi ~ Inv-Gamma(1/2, 1/c^2)
  // which is equivalent to sigma_mu ~ half-Cauchy(0, c).
  // Set c = k / sqrt(ntrees) with k = 1 (default) or k = 0.1.
  bool   adaptive_sigma_mu = true;
  double half_cauchy_c     = -1.0;  // <=0 -> filled at setup = 1/sqrt(ntrees)
  // ----- DART sparsity prior on split variables (Linero 2018, paper §3.3) -----
  // s_j ~ Dirichlet(theta/P + n_j, ...), updated each iter when dart_active.
  // theta reparametrised: theta/(theta + rho) ~ Beta(a, b).
  //   Default (a, b, rho) = (0.5, 1, P) -> moderate sparsity.
  bool   dart_requested    = false;  // user opted in at construction time
  bool   dart_active       = false;  // flipped by genbart_model::startdart()
  double dart_a            = 0.5;
  double dart_b            = 1.0;
  double dart_rho          = 0.0;    // <=0 -> filled at setup = p
  double dart_theta_init   = 1.0;    // initial concentration
  bool   dart_const_theta  = false;  // if true, keep theta fixed (don't update)
  // Current split-variable probabilities.  Managed by genbart_model.  If empty,
  // uniform 1/P is used.  DART populates this each iter.
  std::vector<double> var_probs;
  laplace_opts lap_opts{};
};

// ---------------------------------------------------------------------------
// Compute depth prior rho_d.  Log-safe helper also provided.
// ---------------------------------------------------------------------------
inline double rho_depth(const rjmcmc_hypers& h, int d) {
  return h.gamma_depth * std::pow(1.0 + d, -h.beta_depth);
}
inline double log_rho_depth(const rjmcmc_hypers& h, int d) {
  return std::log(h.gamma_depth) - h.beta_depth * std::log(1.0 + d);
}
inline double log1m_rho_depth(const rjmcmc_hypers& h, int d) {
  return std::log1p(-rho_depth(h, d));  // log(1 - rho_d) via log1p for precision
}

// ---------------------------------------------------------------------------
// Given a tree and the full training data (X, lambda_minus_t), associate each
// observation to a leaf pointer by traversing the tree.  This is O(N * depth).
//
// Inputs:
//   t     : the tree (const)
//   X     : N x p column-major design matrix (row i, column j accessed as X[i + j*N])
//           We follow the heterbart convention: column-major.
//   N, p  : dimensions
//   xi    : cutpoint info
//
// Output:
//   obs_of_leaf[leaf_ptr] -> vector of observation indices.
// ---------------------------------------------------------------------------
inline void partition_by_leaf(
    tree& t,
    const double* X_row_major,   // N x p, observation-contiguous
    std::size_t   N,
    std::size_t   p,
    xinfo&        xi,
    std::map<tree::tree_p, std::vector<std::size_t> >& obs_of_leaf)
{
  obs_of_leaf.clear();
  for (std::size_t i = 0; i < N; ++i) {
    // observation i's features are contiguous at X_row_major[i*p ... i*p + p - 1]
    tree::tree_p leaf = t.bn(const_cast<double*>(X_row_major) + i * p, xi);
    obs_of_leaf[leaf].push_back(i);
  }
}

// ---------------------------------------------------------------------------
// Sample a splitting rule (variable index v, cutpoint index c) at a given
// node.  The node's hyperrectangle [L_j, U_j] per variable is computed via
// tree::rg, which returns inclusive cut indices.  We follow Chipman 2010.
//
// Returns true if a valid rule was found; fills v_out, c_out.
// Returns false if no variable has a non-degenerate range (in which case the
// BIRTH move must be rejected).
// ---------------------------------------------------------------------------
inline bool sample_splitting_rule(
    tree::tree_p           leaf,
    std::size_t            p,
    xinfo&                 xi,
    const rjmcmc_hypers&   h,
    rn&                    gen,
    std::size_t&           v_out,
    std::size_t&           c_out)
{
  // Collect valid variables (those with at least one valid cutpoint at this
  // node, given the ancestor constraints).
  std::vector<std::size_t> valid_vars;
  std::vector<std::pair<int, int> > valid_ranges;
  valid_vars.reserve(p);
  valid_ranges.reserve(p);

  for (std::size_t j = 0; j < p; ++j) {
    int L = 0, U = (int)xi[j].size() - 1;
    leaf->rg((int)j, &L, &U);
    if (L <= U) {
      valid_vars.push_back(j);
      valid_ranges.push_back(std::make_pair(L, U));
    }
  }
  if (valid_vars.empty()) return false;

  // Draw variable.  If DART weights supplied, use them over valid_vars
  // (renormalised); else uniform.
  std::size_t pick = 0;
  if (h.var_probs.size() == p) {
    std::vector<double> w(valid_vars.size());
    double wsum = 0.0;
    for (std::size_t k = 0; k < valid_vars.size(); ++k) {
      w[k] = h.var_probs[valid_vars[k]];
      wsum += w[k];
    }
    if (wsum <= 0.0) {
      pick = (std::size_t)(gen.uniform() * valid_vars.size());
    } else {
      double u = gen.uniform() * wsum;
      double cum = 0.0;
      pick = valid_vars.size() - 1;
      for (std::size_t k = 0; k < valid_vars.size(); ++k) {
        cum += w[k];
        if (u < cum) { pick = k; break; }
      }
    }
  } else {
    pick = (std::size_t)(gen.uniform() * valid_vars.size());
    if (pick >= valid_vars.size()) pick = valid_vars.size() - 1;
  }
  v_out = valid_vars[pick];

  // Draw cutpoint uniformly from [L, U].
  const int L = valid_ranges[pick].first;
  const int U = valid_ranges[pick].second;
  c_out = L + (std::size_t)(gen.uniform() * (U - L + 1));
  if ((int)c_out > U) c_out = (std::size_t)U;

  return true;
}

// ---------------------------------------------------------------------------
// Result reporter for diagnostics.
// ---------------------------------------------------------------------------
struct move_result {
  bool   accepted   = false;
  double log_ratio  = std::numeric_limits<double>::quiet_NaN();
  char   move       = '?';   // 'B', 'D', 'C'
};

// ---------------------------------------------------------------------------
// BIRTH move on tree t.  Picks a random leaf, tries to split it.
// Updates `t` in place if accepted.
// ---------------------------------------------------------------------------
inline move_result propose_birth(
    tree&                 t,
    const double*         X_row_major,
    const double*         Y,
    const double*         lam_minus_t,   // lambda without this tree's contribution
    std::size_t           N,
    std::size_t           p,
    xinfo&                xi,
    const rjmcmc_hypers&  h,
    const likelihood&     lik,
    rn&                   gen)
{
  move_result res; res.move = 'B';

  // --- 1. pick a leaf uniformly -----------------------------------------
  tree::npv leaves;
  t.getbots(leaves);
  const std::size_t n_leaves = leaves.size();
  if (n_leaves == 0) return res;
  tree::tree_p ell = leaves[(std::size_t)(gen.uniform() * n_leaves)];
  const int d = (int)ell->depth();
  if (d >= h.max_depth) return res;  // hard cap

  // --- 2. sample a splitting rule ---------------------------------------
  std::size_t v_new, c_new;
  if (!sample_splitting_rule(ell, p, xi, h, gen, v_new, c_new)) return res;

  // --- 3. partition this leaf's observations into L, R ------------------
  // Temporarily treat ell as if it had been split by (v_new, c_new).
  std::map<tree::tree_p, std::vector<std::size_t> > part;
  partition_by_leaf(t, X_row_major, N, p, xi, part);
  const std::vector<std::size_t>& idx_ell = part[ell];

  std::vector<std::size_t> idx_L, idx_R;
  idx_L.reserve(idx_ell.size());
  idx_R.reserve(idx_ell.size());
  const double cut_val = xi[v_new][c_new];
  for (std::size_t k = 0; k < idx_ell.size(); ++k) {
    const std::size_t i = idx_ell[k];
    const double x_ij = X_row_major[i * p + v_new];
    if (x_ij < cut_val) idx_L.push_back(i);
    else                idx_R.push_back(i);
  }
  if ((int)idx_L.size() < h.min_leaf_n || (int)idx_R.size() < h.min_leaf_n) {
    return res;  // reject silently; tree unchanged
  }

  // --- 4. Laplace proposals for mu_ellL, mu_ellR, and existing mu_ell ---
  // Assemble the per-leaf (y, lambda_minus_t) slices.
  auto slice = [&](const std::vector<std::size_t>& idx,
                   std::vector<double>& ys, std::vector<double>& lams) {
    ys.resize(idx.size());
    lams.resize(idx.size());
    for (std::size_t k = 0; k < idx.size(); ++k) {
      ys[k]   = Y[idx[k]];
      lams[k] = lam_minus_t[idx[k]];
    }
  };
  std::vector<double> yL, lamL, yR, lamR, yAll, lamAll;
  slice(idx_L, yL, lamL);
  slice(idx_R, yR, lamR);
  slice(idx_ell, yAll, lamAll);

  // Paper Algorithm 3: for BIRTH moves, initialise Fisher scoring at the
  // PARENT node's mu (the leaf being split).  This is the current mu of `ell`.
  laplace_opts opts_birth = h.lap_opts;
  opts_birth.init_mu = ell->gettheta();
  const laplace_proposal qL = laplace_leaf(
      yL.data(), lamL.data(), idx_L.data(), yL.size(), h.sigma_mu, lik, opts_birth);
  const laplace_proposal qR = laplace_leaf(
      yR.data(), lamR.data(), idx_R.data(), yR.size(), h.sigma_mu, lik, opts_birth);
  // For the reverse-DEATH proposal density (merged leaf), init at current mu_ell.
  laplace_opts opts_death_rev = h.lap_opts;
  opts_death_rev.init_mu = ell->gettheta();
  const laplace_proposal qO = laplace_leaf(   // O = 'old' single leaf
      yAll.data(), lamAll.data(), idx_ell.data(), yAll.size(), h.sigma_mu, lik, opts_death_rev);

  // --- 5. sample mu_ellL', mu_ellR' from Gaussian proposals -------------
  const double mu_Lp = rnorm_from_laplace(qL, gen);
  const double mu_Rp = rnorm_from_laplace(qR, gen);
  const double mu_ell_current = ell->gettheta();

  // --- 6. log-likelihood ratio (data term) ------------------------------
  double logLik_new = 0.0, logLik_old = 0.0;
  for (std::size_t k = 0; k < yL.size(); ++k) {
    logLik_new += lik.log_f(yL[k], lamL[k] + mu_Lp, idx_L[k]);
  }
  for (std::size_t k = 0; k < yR.size(); ++k) {
    logLik_new += lik.log_f(yR[k], lamR[k] + mu_Rp, idx_R[k]);
  }
  for (std::size_t k = 0; k < yAll.size(); ++k) {
    logLik_old += lik.log_f(yAll[k], lamAll[k] + mu_ell_current, idx_ell[k]);
  }
  const double log_lik_ratio = logLik_new - logLik_old;
  if (!std::isfinite(log_lik_ratio)) {
    res.log_ratio = log_lik_ratio;
    return res;  // reject
  }

  // --- 7. log-prior ratio -----------------------------------------------
  //   rho_d (1 - rho_{d+1})^2 / (1 - rho_d)  [tree-structure prior]
  //   * pi_mu(mu_Lp) * pi_mu(mu_Rp) / pi_mu(mu_ell_current)
  const double lp_rho = log_rho_depth(h, d)
                      + 2.0 * log1m_rho_depth(h, d + 1)
                      - log1m_rho_depth(h, d);
  const double sigma_mu = h.sigma_mu;
  const double lp_mu = log_dnorm(mu_Lp, 0.0, sigma_mu)
                     + log_dnorm(mu_Rp, 0.0, sigma_mu)
                     - log_dnorm(mu_ell_current, 0.0, sigma_mu);
  const double log_prior_ratio = lp_rho + lp_mu;

  // --- 8. log-proposal ratio --------------------------------------------
  //   p_DEATH(T') * |NOG(T')|^{-1}         G_DEATH(mu_ell_current | T, mu_L, mu_R)
  //   --------------------------    *   -------------------------------------------
  //   p_BIRTH(T)  * |L(T)|^{-1}          G_BIRTH(mu_Lp, mu_Rp | T, mu_ell_current)
  //
  // G_BIRTH is independent Gaussian (qL, qR); G_DEATH is qO.
  //
  // After the proposed BIRTH, the newly-created branch (was a leaf) becomes
  // a NOG node.  |NOG(T')| is therefore (old NOG count) + 1 (new nog) minus
  // any NOGs that disappeared (none, because we only added leaves).
  // Concretely: we compute it after the move would be applied, in closed form.
  //
  // Easier: we know the new tree would have
  //   |L(T')|   = |L(T)| + 1   (one leaf split into two)
  //   |NOG(T')| = (previous NOG count of T) + 1 if the parent-of-ell was not
  //               itself a NOG... wait, NOG just means a branch whose both
  //               children are leaves.  The new branch created here has both
  //               children as leaves -> it IS NOG.  But the parent of ell
  //               was NOG iff ell's sibling is a leaf.  If so, after the
  //               split, the parent has one branch (newly created) and one
  //               leaf sibling -> no longer NOG.  Net:
  //
  //               delta_NOG = +1 - (parent_was_NOG ? 1 : 0)
  //
  // This is the classic BART accounting; cf. Chipman 1998.
  const std::size_t num_leaves_T  = n_leaves;
  tree::npv nogs_T;
  t.getnogs(nogs_T);
  const std::size_t num_nogs_T = nogs_T.size();
  const bool parent_was_NOG = (ell->getp() != 0) &&
      (std::find(nogs_T.begin(), nogs_T.end(), ell->getp()) != nogs_T.end());
  const std::size_t num_nogs_Tp = num_nogs_T + 1 - (parent_was_NOG ? 1 : 0);
  (void)num_nogs_T;  // num_nogs_T used below only via num_nogs_Tp, no direct refs

  const double log_G_birth = log_dnorm(mu_Lp, qL.m, qL.v)
                           + log_dnorm(mu_Rp, qR.m, qR.v);
  const double log_G_death = log_dnorm(mu_ell_current, qO.m, qO.v);

  const double log_prop_ratio
      = std::log((double)h.p_death / (double)h.p_birth)
      + std::log((double)num_leaves_T / (double)num_nogs_Tp)
      + log_G_death - log_G_birth;

  // --- 9. total MH log ratio --------------------------------------------
  const double log_R = log_lik_ratio + log_prior_ratio + log_prop_ratio;
  res.log_ratio = log_R;
  if (!std::isfinite(log_R)) return res;  // reject

  if (std::log(gen.uniform()) < log_R) {
    // ACCEPT: apply the move.
    // Use birthp(np, v, c, thetaL, thetaR).  First set ell's mu to whatever;
    // birthp creates two children with the given thetas.  We also want to
    // update ell itself to be a branch (BART's tree code handles that via
    // birthp).  The split rule (v_new, c_new) goes into ell.
    t.birthp(ell, v_new, c_new, mu_Lp, mu_Rp);
    res.accepted = true;
  }
  return res;
}

// ---------------------------------------------------------------------------
// DEATH move.  Picks a random NOG branch and collapses it into a leaf.
// ---------------------------------------------------------------------------
inline move_result propose_death(
    tree&                 t,
    const double*         X_row_major,
    const double*         Y,
    const double*         lam_minus_t,
    std::size_t           N,
    std::size_t           p,
    xinfo&                xi,
    const rjmcmc_hypers&  h,
    const likelihood&     lik,
    rn&                   gen)
{
  (void)p;  // unused (we partition by leaf not by NOG directly)
  move_result res; res.move = 'D';

  // --- 1. pick a NOG branch ---------------------------------------------
  tree::npv nogs;
  t.getnogs(nogs);
  const std::size_t num_nogs_T = nogs.size();
  if (num_nogs_T == 0) return res;
  tree::tree_p b = nogs[(std::size_t)(gen.uniform() * num_nogs_T)];
  const int d = (int)b->depth();

  // Gather observations of the two children leaves.
  std::map<tree::tree_p, std::vector<std::size_t> > part;
  partition_by_leaf(t, X_row_major, N, p, xi, part);
  const std::vector<std::size_t>& idx_L = part[b->getl()];
  const std::vector<std::size_t>& idx_R = part[b->getr()];

  // Union for the merged leaf.
  std::vector<std::size_t> idx_M;
  idx_M.reserve(idx_L.size() + idx_R.size());
  idx_M.insert(idx_M.end(), idx_L.begin(), idx_L.end());
  idx_M.insert(idx_M.end(), idx_R.begin(), idx_R.end());

  // --- 2. Laplace proposals ---------------------------------------------
  auto slice = [&](const std::vector<std::size_t>& idx,
                   std::vector<double>& ys, std::vector<double>& lams) {
    ys.resize(idx.size()); lams.resize(idx.size());
    for (std::size_t k = 0; k < idx.size(); ++k) {
      ys[k] = Y[idx[k]]; lams[k] = lam_minus_t[idx[k]];
    }
  };
  std::vector<double> yL, lamL, yR, lamR, yM, lamM;
  slice(idx_L, yL, lamL);
  slice(idx_R, yR, lamR);
  slice(idx_M, yM, lamM);

  // Paper Algorithm 3: for DEATH, init the MERGED-leaf Fisher scoring at the
  // mid-point of the two children mus that are being collapsed.
  const double mu_L_now = b->getl()->gettheta();
  const double mu_R_now = b->getr()->gettheta();
  laplace_opts opts_death = h.lap_opts;
  opts_death.init_mu = 0.5 * (mu_L_now + mu_R_now);
  // Reverse-BIRTH proposal evaluates qL, qR at CURRENT children; init there.
  laplace_opts opts_L = h.lap_opts; opts_L.init_mu = mu_L_now;
  laplace_opts opts_R = h.lap_opts; opts_R.init_mu = mu_R_now;
  const laplace_proposal qL = laplace_leaf(yL.data(), lamL.data(), idx_L.data(),
                                           yL.size(), h.sigma_mu, lik, opts_L);
  const laplace_proposal qR = laplace_leaf(yR.data(), lamR.data(), idx_R.data(),
                                           yR.size(), h.sigma_mu, lik, opts_R);
  const laplace_proposal qM = laplace_leaf(yM.data(), lamM.data(), idx_M.data(),
                                           yM.size(), h.sigma_mu, lik, opts_death);

  // --- 3. sample mu_b' from q_M -----------------------------------------
  const double mu_bp = rnorm_from_laplace(qM, gen);
  const double mu_L_current = b->getl()->gettheta();
  const double mu_R_current = b->getr()->gettheta();

  // --- 4. log-likelihood ratio (L_new - L_old) --------------------------
  double logLik_new = 0.0, logLik_old = 0.0;
  for (std::size_t k = 0; k < yM.size(); ++k) {
    logLik_new += lik.log_f(yM[k], lamM[k] + mu_bp, idx_M[k]);
  }
  for (std::size_t k = 0; k < yL.size(); ++k) {
    logLik_old += lik.log_f(yL[k], lamL[k] + mu_L_current, idx_L[k]);
  }
  for (std::size_t k = 0; k < yR.size(); ++k) {
    logLik_old += lik.log_f(yR[k], lamR[k] + mu_R_current, idx_R[k]);
  }
  const double log_lik_ratio = logLik_new - logLik_old;
  if (!std::isfinite(log_lik_ratio)) { res.log_ratio = log_lik_ratio; return res; }

  // --- 5. log-prior ratio (inverse of BIRTH's prior ratio) --------------
  const double lp_rho_inv = -log_rho_depth(h, d)
                          - 2.0 * log1m_rho_depth(h, d + 1)
                          + log1m_rho_depth(h, d);
  const double sigma_mu = h.sigma_mu;
  const double lp_mu_inv = log_dnorm(mu_bp, 0.0, sigma_mu)
                         - log_dnorm(mu_L_current, 0.0, sigma_mu)
                         - log_dnorm(mu_R_current, 0.0, sigma_mu);
  const double log_prior_ratio = lp_rho_inv + lp_mu_inv;

  // --- 6. log-proposal ratio --------------------------------------------
  // After DEATH, tree T' has:
  //   |L(T')|   = |L(T)| - 1
  //   |NOG(T')| = |NOG(T)| - 1 + (grandparent_becomes_NOG ? 1 : 0)
  // where grandparent_becomes_NOG iff b's sibling (in T) is a leaf.
  tree::npv leaves;
  t.getbots(leaves);
  const std::size_t num_leaves_T  = leaves.size();
  const std::size_t num_leaves_Tp = num_leaves_T - 1;
  (void)num_leaves_T;
  // Note: num_nogs_Tp (after death) not needed here; cf. propose_death
  // log_prop_ratio uses num_nogs_T (before) and num_leaves_Tp (after).

  const double log_G_birth = log_dnorm(mu_L_current, qL.m, qL.v)
                           + log_dnorm(mu_R_current, qR.m, qR.v);
  const double log_G_death = log_dnorm(mu_bp, qM.m, qM.v);

  const double log_prop_ratio
      = std::log((double)h.p_birth / (double)h.p_death)
      + std::log((double)num_nogs_T / (double)num_leaves_Tp)
      + log_G_birth - log_G_death;

  // --- 7. total MH log ratio --------------------------------------------
  const double log_R = log_lik_ratio + log_prior_ratio + log_prop_ratio;
  res.log_ratio = log_R;
  if (!std::isfinite(log_R)) return res;

  if (std::log(gen.uniform()) < log_R) {
    t.deathp(b, mu_bp);
    res.accepted = true;
  }
  return res;
}

// ---------------------------------------------------------------------------
// CHANGE move.  Picks a NOG branch, changes its splitting rule to a freshly-
// sampled one, and re-samples mu_L, mu_R from their Laplace conditionals.
// ---------------------------------------------------------------------------
inline move_result propose_change(
    tree&                 t,
    const double*         X_row_major,
    const double*         Y,
    const double*         lam_minus_t,
    std::size_t           N,
    std::size_t           p,
    xinfo&                xi,
    const rjmcmc_hypers&  h,
    const likelihood&     lik,
    rn&                   gen)
{
  move_result res; res.move = 'C';

  // --- 1. pick a NOG ----------------------------------------------------
  tree::npv nogs;
  t.getnogs(nogs);
  if (nogs.empty()) return res;
  tree::tree_p b = nogs[(std::size_t)(gen.uniform() * nogs.size())];

  // Save old split rule (for reject path).
  const std::size_t v_old = b->getv();
  const std::size_t c_old = b->getc();

  // --- 2. sample new splitting rule -------------------------------------
  std::size_t v_new, c_new;
  if (!sample_splitting_rule(b, p, xi, h, gen, v_new, c_new)) return res;

  // Partition observations under b.  We compute for both (old rule) and
  // (new rule) -- old rule is the current partition of the tree.
  std::map<tree::tree_p, std::vector<std::size_t> > part_old;
  partition_by_leaf(t, X_row_major, N, p, xi, part_old);
  const std::vector<std::size_t>& idx_L_old = part_old[b->getl()];
  const std::vector<std::size_t>& idx_R_old = part_old[b->getr()];
  std::vector<std::size_t> idx_all;
  idx_all.reserve(idx_L_old.size() + idx_R_old.size());
  idx_all.insert(idx_all.end(), idx_L_old.begin(), idx_L_old.end());
  idx_all.insert(idx_all.end(), idx_R_old.begin(), idx_R_old.end());

  // Apply new rule tentatively: partition idx_all by X[i, v_new] < xi[v_new][c_new].
  std::vector<std::size_t> idx_L_new, idx_R_new;
  idx_L_new.reserve(idx_all.size());
  idx_R_new.reserve(idx_all.size());
  const double cut_val = xi[v_new][c_new];
  for (std::size_t k = 0; k < idx_all.size(); ++k) {
    const std::size_t i = idx_all[k];
    if (X_row_major[i * p + v_new] < cut_val) idx_L_new.push_back(i);
    else                                      idx_R_new.push_back(i);
  }
  if ((int)idx_L_new.size() < h.min_leaf_n ||
      (int)idx_R_new.size() < h.min_leaf_n) {
    return res;  // reject
  }

  // --- 3. Laplace proposals for both old and new partitions --------------
  auto slice = [&](const std::vector<std::size_t>& idx,
                   std::vector<double>& ys, std::vector<double>& lams) {
    ys.resize(idx.size()); lams.resize(idx.size());
    for (std::size_t k = 0; k < idx.size(); ++k) {
      ys[k] = Y[idx[k]]; lams[k] = lam_minus_t[idx[k]];
    }
  };
  std::vector<double> yLo, lamLo, yRo, lamRo, yLn, lamLn, yRn, lamRn;
  slice(idx_L_old, yLo, lamLo);
  slice(idx_R_old, yRo, lamRo);
  slice(idx_L_new, yLn, lamLn);
  slice(idx_R_new, yRn, lamRn);

  // CHANGE: paper doesn't pin an init; use current children mus (warm-start).
  laplace_opts opts_Lo = h.lap_opts; opts_Lo.init_mu = b->getl()->gettheta();
  laplace_opts opts_Ro = h.lap_opts; opts_Ro.init_mu = b->getr()->gettheta();
  laplace_opts opts_Ln = h.lap_opts; opts_Ln.init_mu = b->getl()->gettheta();
  laplace_opts opts_Rn = h.lap_opts; opts_Rn.init_mu = b->getr()->gettheta();
  const laplace_proposal qLo = laplace_leaf(yLo.data(), lamLo.data(), idx_L_old.data(),
                                            yLo.size(), h.sigma_mu, lik, opts_Lo);
  const laplace_proposal qRo = laplace_leaf(yRo.data(), lamRo.data(), idx_R_old.data(),
                                            yRo.size(), h.sigma_mu, lik, opts_Ro);
  const laplace_proposal qLn = laplace_leaf(yLn.data(), lamLn.data(), idx_L_new.data(),
                                            yLn.size(), h.sigma_mu, lik, opts_Ln);
  const laplace_proposal qRn = laplace_leaf(yRn.data(), lamRn.data(), idx_R_new.data(),
                                            yRn.size(), h.sigma_mu, lik, opts_Rn);

  // --- 4. sample new mus --------------------------------------------------
  const double muL_new = rnorm_from_laplace(qLn, gen);
  const double muR_new = rnorm_from_laplace(qRn, gen);
  const double muL_old = b->getl()->gettheta();
  const double muR_old = b->getr()->gettheta();

  // --- 5. log-likelihood ratio -------------------------------------------
  double logLik_new = 0.0, logLik_old = 0.0;
  for (std::size_t k = 0; k < yLn.size(); ++k) logLik_new += lik.log_f(yLn[k], lamLn[k] + muL_new, idx_L_new[k]);
  for (std::size_t k = 0; k < yRn.size(); ++k) logLik_new += lik.log_f(yRn[k], lamRn[k] + muR_new, idx_R_new[k]);
  for (std::size_t k = 0; k < yLo.size(); ++k) logLik_old += lik.log_f(yLo[k], lamLo[k] + muL_old, idx_L_old[k]);
  for (std::size_t k = 0; k < yRo.size(); ++k) logLik_old += lik.log_f(yRo[k], lamRo[k] + muR_old, idx_R_old[k]);
  const double log_lik_ratio = logLik_new - logLik_old;
  if (!std::isfinite(log_lik_ratio)) { res.log_ratio = log_lik_ratio; return res; }

  // --- 6. log-prior ratio for the mus ------------------------------------
  const double sigma_mu = h.sigma_mu;
  const double lp_mu_ratio = log_dnorm(muL_new, 0.0, sigma_mu)
                           + log_dnorm(muR_new, 0.0, sigma_mu)
                           - log_dnorm(muL_old, 0.0, sigma_mu)
                           - log_dnorm(muR_old, 0.0, sigma_mu);
  // NOTE: split-rule prior cancels because the proposal draws from the prior.

  // --- 7. log-proposal ratio ---------------------------------------------
  const double log_G_new = log_dnorm(muL_new, qLn.m, qLn.v)
                         + log_dnorm(muR_new, qRn.m, qRn.v);
  const double log_G_old = log_dnorm(muL_old, qLo.m, qLo.v)
                         + log_dnorm(muR_old, qRo.m, qRo.v);
  const double log_prop_ratio = log_G_old - log_G_new;

  const double log_R = log_lik_ratio + lp_mu_ratio + log_prop_ratio;
  res.log_ratio = log_R;
  if (!std::isfinite(log_R)) return res;

  if (std::log(gen.uniform()) < log_R) {
    b->setv(v_new);
    b->setc(c_new);
    b->getl()->settheta(muL_new);
    b->getr()->settheta(muR_new);
    res.accepted = true;
  } else {
    (void)v_old; (void)c_old;  // nothing to restore
  }
  return res;
}

// ---------------------------------------------------------------------------
// Dispatch: one tree-update step.  Picks a move at random.
// After the move, also does one Gibbs update on all leaf mus (M-step) by
// sampling from their Laplace-approximated full conditional (slice sampling
// would be more correct, but Laplace is cheap and matches the paper for
// informative priors; if mixing suffers, replace with slice).
// ---------------------------------------------------------------------------
// If f_tree_out != nullptr, it's filled with the per-observation leaf theta
// values after the M-step (saving a subsequent full tree traversal in the
// caller).  If you pass a valid pointer, it MUST be a length-N array.
inline move_result draw_tree(
    tree&                 t,
    const double*         X_row_major,
    const double*         Y,
    const double*         lam_minus_t,
    std::size_t           N,
    std::size_t           p,
    xinfo&                xi,
    const rjmcmc_hypers&  h,
    const likelihood&     lik,
    rn&                   gen,
    bool                  do_M_step = true,
    double*               f_tree_out = nullptr)
{
  // --- normalise p_birth, p_death, p_change --------------------------------
  const double s = h.p_birth + h.p_death + h.p_change;
  const double pb = h.p_birth / s;
  const double pd = (h.p_birth + h.p_death) / s;
  const double u = gen.uniform();

  move_result mr;
  if (u < pb) {
    mr = propose_birth(t, X_row_major, Y, lam_minus_t, N, p, xi, h, lik, gen);
  } else if (u < pd) {
    mr = propose_death(t, X_row_major, Y, lam_minus_t, N, p, xi, h, lik, gen);
  } else {
    mr = propose_change(t, X_row_major, Y, lam_minus_t, N, p, xi, h, lik, gen);
  }

  // --- M-step: refresh all leaf mu via Laplace-approximate Gibbs ----------
  // While we're traversing the tree partition, optionally fill f_tree_out[i]
  // with each obs's leaf theta (saves a redundant full traversal later).
  if (do_M_step) {
    std::map<tree::tree_p, std::vector<std::size_t> > part;
    partition_by_leaf(t, X_row_major, N, p, xi, part);
    tree::npv leaves;
    t.getbots(leaves);
    std::vector<double> ys, lams;
    for (std::size_t k = 0; k < leaves.size(); ++k) {
      const std::vector<std::size_t>& idx = part[leaves[k]];
      ys.resize(idx.size()); lams.resize(idx.size());
      for (std::size_t j = 0; j < idx.size(); ++j) {
        ys[j]   = Y[idx[j]];
        lams[j] = lam_minus_t[idx[j]];
      }
      laplace_opts opts_m = h.lap_opts; opts_m.init_mu = leaves[k]->gettheta();
      laplace_proposal q = laplace_leaf(ys.data(), lams.data(), idx.data(),
                                        ys.size(), h.sigma_mu, lik, opts_m);
      const double new_mu = rnorm_from_laplace(q, gen);
      leaves[k]->settheta(new_mu);
      if (f_tree_out != nullptr) {
        for (std::size_t j = 0; j < idx.size(); ++j) f_tree_out[idx[j]] = new_mu;
      }
    }
  } else if (f_tree_out != nullptr) {
    // No M-step: fill f_tree_out via a single partition so caller doesn't have
    // to do it separately.
    std::map<tree::tree_p, std::vector<std::size_t> > part;
    partition_by_leaf(t, X_row_major, N, p, xi, part);
    for (auto& kv : part) {
      const double mu_k = kv.first->gettheta();
      for (std::size_t j : kv.second) f_tree_out[j] = mu_k;
    }
  }
  return mr;
}

}  // namespace genbart

#endif  // GENBART_RJMCMC_BD_H_
