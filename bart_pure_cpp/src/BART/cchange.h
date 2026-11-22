/*
 *  Copyright (C) 2026 AI4BayesCode
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
 *  cchange.h -- the COLLAPSED CHANGE move for BART (Gaussian leaves).
 *
 *  After BIRTH/DEATH, every internal node of the tree is visited once, in pre-order. At node eta with rule (v, c):
 *    1. propose v' ~ s, the split probabilities over all p variables (pv);
 *    2. accept v' with probability min(1, M(v') / M(v)), where M(u) is the sum, over the admissible cutpoints c of u,
 *       of the full conditional weight of (u, c) divided by s_u (M(u) = 0 when u has no admissible rule at eta);
 *    3. draw the cutpoint of the resulting variable from its full conditional.
 *  Steps 1-2 are a Metropolis-Hastings update of v from its marginal pi(v | rest) prop s_v M(v) with proposal s, so the
 *  s factors cancel; step 3 draws c from pi(c | v, rest). The leaf values are integrated out throughout and are drawn
 *  afterwards by drmu / heterdrmu, as after BIRTH/DEATH.
 *
 *  Target: the law that BIRTH/DEATH (bd.h, heterbd.h) leave invariant with aug = false,
 *    prod_{internal d} pgrow(d) s_{v_d} / N_d(v_d)  x  prod_{leaves l} (1 - pgrow(l))  x  prod_l L(b_l, M_l)
 *    x  1{every leaf holds >= 5 observations},
 *  where pgrow(node) = alpha / (1 + depth)^beta if some variable has an admissible cutpoint in the node's region and 0
 *  otherwise, N_d(v) = the number of admissible cutpoints of v in d's region, and
 *    L(b, M) = -0.5 log(1 + b tau^2) + 0.5 M^2 tau^2 / (1 + b tau^2),  b = sum_i w_i,  M = sum_i w_i r_i
 *  over the leaf's observations, r the partial residual and w_i = 1 / sigma_i^2 (heterlh, heterbartfuns.h). With one
 *  sigma for all observations, L differs from lh() in bd.h by log(sigma) per leaf; the number of leaves does not
 *  change in this move, so that term cancels.
 *
 *  Full conditional of eta's rule (every factor outside eta's subtree cancels):
 *    P(v, c) prop s_v / N_eta(v)  x  prod_{internal d below eta} 1{c_d admissible} / N_d(v, c)
 *                 x  prod_{leaves l below eta} (1 - pgrow_l(v, c))  x  prod_l L(b_l(v, c), M_l(v, c))
 *                 x  1{n_l(v, c) >= 5 for every leaf l below eta}.
 *  pgrow of an internal descendant does not depend on (v, c): a node that holds an admissible rule can split.
 *  Each observation of eta's region is routed once through the left subtree and once through the right subtree,
 *  whose rules are fixed: it reaches leaf la(i) if it goes left at eta and ra(i) if it goes right, and it goes left
 *  under (v, c) iff kidx[v n + i] <= c (x < xi[v][c], tree::bn). The leaf statistics for all cutpoints of v then follow
 *  from prefix sums over the cutpoint index, so a node costs O(observations in its region) per variable evaluated plus
 *  O(cutpoints x leaves below it); only the current and the proposed variable are evaluated.
 *
 *  bart::draw and heterbart::draw run one scan per tree in every draw and require aug = false.
 */
#ifndef GUARD_bart_cchange_h
#define GUARD_bart_cchange_h

#include "tree.h"
#include "info.h"
#include "rn.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace bart_cchange {

// Negative controls for the exactness test only (ctl argument, 0 = the correct move): 1 drops the cutpoint-count
// factors of the internal descendants, 2 drops 1/N_eta(v), 3 drops the leaf (1 - pgrow) factors, 4 keeps the s
// factors in the acceptance ratio, 5 draws the cutpoint uniformly instead of from its full conditional.

// admissible cutpoint range [L, U] of v in nd's region; tree::rg only tightens the bounds it receives, so they are
// reset to the full range on every call
inline int range(tree::tree_p nd, size_t v, xinfo& xi, int& L, int& U)
{
  L = 0; U = (int)xi[v].size() - 1;
  nd->rg(v, &L, &U);
  return U - L + 1;
}

inline bool has_cut(tree::tree_p nd, size_t v, xinfo& xi) { int L, U; return range(nd, v, xi, L, U) >= 1; }

inline double psplit(const pinfo& pi, size_t depth) { return pi.alpha / std::pow(1.0 + (double)depth, pi.mybeta); }

// Gaussian leaf likelihood with the leaf value integrated out; the same expression as heterlh (heterbartfuns.h)
inline double leaf_lik(double b, double M, double tau)
{
  const double t2 = tau * tau, k = b * t2 + 1.0;
  return -0.5 * std::log(k) + 0.5 * M * M * t2 / k;
}

// one draw from unnormalised log weights
inline size_t draw_log(const std::vector<double>& lw, rn& gen)
{
  double mx = lw[0];
  for (size_t k = 1; k < lw.size(); k++) if (lw[k] > mx) mx = lw[k];
  double s = 0.0; std::vector<double> w(lw.size());
  for (size_t k = 0; k < lw.size(); k++) { w[k] = std::exp(lw[k] - mx); s += w[k]; }
  const double u = gen.uniform() * s; double c = 0.0;
  for (size_t k = 0; k < w.size(); k++) { c += w[k]; if (u < c) return k; }
  size_t k = w.size() - 1; while (k > 0 && w[k] <= 0.0) k--; return k;
}

// kidx[v n + i] = number of cutpoints of v that are <= x_iv, so observation i goes left under (v, c) iff kidx <= c.
// Stored variable-major, so the scan over one variable reads it contiguously.
inline void make_kidx(xinfo& xi, dinfo& di, std::vector<int>& kidx)
{
  kidx.resize(di.n * di.p);
  for (size_t v = 0; v < di.p; v++) for (size_t i = 0; i < di.n; i++) {
    const double xv = di.x[i * di.p + v];
    if (!std::isfinite(xv))
      throw std::invalid_argument("BART: X must be finite (the collapsed change move routes every observation by its "
                                  "cutpoint index)");
    kidx[v * di.n + i] = (int)(std::upper_bound(xi[v].begin(), xi[v].end(), xv) - xi[v].begin());
  }
}

// leaf of every observation and the observations of every leaf, kept up to date during a scan. Leaves are numbered
// in getbots order (the topology does not change during a scan).
struct Route {
  tree::npv bots;
  std::vector<std::vector<size_t> > mem;
  int id_of(tree::tree_p b) const { for (size_t a = 0; a < bots.size(); a++) if (bots[a] == b) return (int)a; return -1; }
  void build(tree& x, xinfo& xi, dinfo& di) {
    bots.clear(); x.getbots(bots); mem.assign(bots.size(), std::vector<size_t>());
    for (size_t i = 0; i < di.n; i++) mem[id_of(x.bn(di.x + i * di.p, xi))].push_back(i);
  }
  void region_of(tree::tree_p eta, std::vector<size_t>& reg) const {
    tree::npv eb; eta->getbots(eb); reg.clear();
    for (size_t a = 0; a < eb.size(); a++) { const std::vector<size_t>& m = mem[id_of(eb[a])]; reg.insert(reg.end(), m.begin(), m.end()); }
  }
};

// Unnormalised log full conditional of the rule of internal node eta over every admissible (v, c), for the variables
// flagged in `only` (all variables when only is null): lw[k] is the weight of rule (cv[k], cc[k]) and carries log s_v.
// R = the observations of eta's region, w = the observation weights 1 / sigma_i^2, di.y = the partial residuals.
// la/ra/leaves are returned for the re-routing after a draw. eta holds its entry rule on return.
inline void rule_weights(tree::tree_p eta, const std::vector<size_t>& R, xinfo& xi, dinfo& di, pinfo& pi,
                         std::vector<double>& pv, const int* kidx, const double* w,
                         std::vector<double>& lw, std::vector<size_t>& cv, std::vector<size_t>& cc,
                         std::vector<int>& la, std::vector<int>& ra, std::vector<tree::tree_p>& leaves,
                         const std::vector<char>* only = nullptr, int ctl = 0)
{
  const size_t MINLEAF = 5;   // bd.h / heterbd.h accept a birth only if both new leaves hold >= 5 observations
  const size_t p = pv.size(), ne = R.size(), n = di.n;
  const size_t v0 = eta->getv(), c0 = eta->getc();
  tree::tree_p TL = eta->getl(), TR = eta->getr();
  tree::npv bl, br; TL->getbots(bl); TR->getbots(br);
  const size_t nl = bl.size(), nb = nl + br.size();
  leaves.clear(); leaves.insert(leaves.end(), bl.begin(), bl.end()); leaves.insert(leaves.end(), br.begin(), br.end());
  la.resize(ne); ra.resize(ne);
  for (size_t k = 0; k < ne; k++) {
    double* xx = di.x + R[k] * di.p;
    tree::tree_p a = TL->bn(xx, xi), b = TR->bn(xx, xi);
    int ia = 0; while (bl[ia] != a) ia++;
    int ib = 0; while (br[ib] != b) ib++;
    la[k] = ia; ra[k] = (int)nl + ib;
  }
  // both children of eta are leaves: every observation reaches leaf 0 on the left and leaf 1 on the right, so the two
  // histograms are the same; build one and read it for both (row[a] = the histogram row of leaf a)
  const bool nog = (nb == 2 && nl == 1);
  const size_t nrows = nog ? 1 : nb;
  std::vector<size_t> row(nb); for (size_t a = 0; a < nb; a++) row[a] = nog ? 0 : a;
  std::vector<double> rk(ne), sk(ne);
  for (size_t k = 0; k < ne; k++) { const size_t i = R[k]; rk[k] = w[i]; sk[k] = w[i] * di.y[i]; }
  tree::npv desc; { tree::npv sub; eta->getnodes(sub);
    for (size_t a = 0; a < sub.size(); a++) if (sub[a] != eta && sub[a]->getl() != 0) desc.push_back(sub[a]); }
  std::vector<double> l1pg(nb);                      // log(1 - pgrow) of each leaf, applied when the leaf can split
  for (size_t a = 0; a < nb; a++) l1pg[a] = std::log1p(-psplit(pi, leaves[a]->depth()));
  std::vector<double> lcache(nb); std::vector<char> dirty(nb);
  std::vector<int> L(p, 0), U(p, -1);
  for (size_t u = 0; u < p; u++) if (!only || (*only)[u]) range(eta, u, xi, L[u], U[u]);
  // Witnesses: with eta on v, a variable u != v is not constrained by eta, so a leaf's range for u is its range with
  // eta ignored (eta's variable set to the out-of-range index p, which no tree::rg comparison matches). A leaf can
  // split under eta = v iff some u != v has a cutpoint there; the first TWO such variables (w1 < w2, p = none) decide
  // it for every v.
  std::vector<size_t> w1(nb, p), w2(nb, p);
  eta->setv(p);
  for (size_t a = 0; a < nb; a++)
    for (size_t u = 0; u < p && w2[a] == p; u++)
      if (has_cut(leaves[a], u, xi)) { if (w1[a] == p) w1[a] = u; else w2[a] = u; }
  eta->setv(v0);
  lw.clear(); cv.clear(); cc.clear();
  std::vector<double> H, Hr, Hs, pn(nb), pr(nb), ps(nb), tn(nb), tr(nb), ts(nb);
  std::vector<tree::tree_p> dv; std::vector<size_t> lnw;
  for (size_t v = 0; v < p; v++) {
    if (only && !(*only)[v]) continue;
    if (U[v] < L[v]) continue;                         // no admissible cutpoint at eta: prior mass 0
    if (!(pv[v] > 0.0)) continue;                      // s_v = 0: prior mass 0
    const size_t nc1 = xi[v].size() + 1;
    // eta splits on v: below eta the ranges of every other variable no longer depend on c
    eta->setv(v); eta->setc((size_t)L[v]);
    double cst = std::log(pv[v]);
    if (ctl != 2) cst -= std::log((double)(U[v] - L[v] + 1));
    bool okv = true; dv.clear();
    for (size_t a = 0; a < desc.size() && okv; a++) {
      tree::tree_p d = desc[a];
      if (d->getv() == v) { dv.push_back(d); continue; }   // depends on c: checked per cutpoint below
      int Ld, Ud; range(d, d->getv(), xi, Ld, Ud);
      if ((int)d->getc() < Ld || (int)d->getc() > Ud) okv = false;
      else if (ctl != 1) cst -= std::log((double)(Ud - Ld + 1));
    }
    if (!okv) continue;
    // a leaf with a witness (a variable other than v with an admissible cutpoint there) can split whatever c is
    lnw.clear();
    for (size_t a = 0; a < nb; a++) {
      const bool wit = (w1[a] < p && w1[a] != v) || w2[a] < p;
      if (wit) { if (ctl != 3) cst += l1pg[a]; } else lnw.push_back(a);
    }
    // histograms over the cutpoint index of the observations routed to each leaf: left leaves take the prefix b <= c,
    // right leaves the suffix b > c
    H.assign(nrows * nc1, 0.0); Hr.assign(nrows * nc1, 0.0); Hs.assign(nrows * nc1, 0.0);
    const int* kv = kidx + v * n;
    if (nog) {
      for (size_t k = 0; k < ne; k++) { const int b = kv[R[k]]; H[b] += 1.0; Hr[b] += rk[k]; Hs[b] += sk[k]; }
    } else {
      for (size_t k = 0; k < ne; k++) {
        const int b = kv[R[k]];
        size_t q = la[k] * nc1 + b; H[q] += 1.0; Hr[q] += rk[k]; Hs[q] += sk[k];
        q = ra[k] * nc1 + b;        H[q] += 1.0; Hr[q] += rk[k]; Hs[q] += sk[k];
      }
    }
    for (size_t a = 0; a < nb; a++) {
      const size_t o = row[a] * nc1;
      pn[a] = pr[a] = ps[a] = tn[a] = tr[a] = ts[a] = 0.0;
      for (size_t b = 0; b < nc1; b++) { tn[a] += H[o + b]; tr[a] += Hr[o + b]; ts[a] += Hs[o + b]; }
      for (int b = 0; b < L[v]; b++) { pn[a] += H[o + b]; pr[a] += Hr[o + b]; ps[a] += Hs[o + b]; }
    }
    std::fill(dirty.begin(), dirty.end(), (char)1);
    for (int c = L[v]; c <= U[v]; c++) {
      for (size_t a = 0; a < nb; a++) {
        const size_t q = row[a] * nc1 + c;
        if (H[q] > 0.0) { pn[a] += H[q]; pr[a] += Hr[q]; ps[a] += Hs[q]; dirty[a] = 1; }
      }
      bool ok = true;
      for (size_t a = 0; a < nb && ok; a++) { const double nn = (a < nl) ? pn[a] : tn[a] - pn[a]; if (nn < (double)MINLEAF) ok = false; }
      if (!ok) continue;
      eta->setc((size_t)c);
      double wt = cst;
      for (size_t q = 0; q < dv.size() && ok; q++) {
        int Ld, Ud; range(dv[q], v, xi, Ld, Ud);
        if ((int)dv[q]->getc() < Ld || (int)dv[q]->getc() > Ud) ok = false;
        else if (ctl != 1) wt -= std::log((double)(Ud - Ld + 1));
      }
      if (!ok) continue;
      if (ctl != 3) for (size_t q = 0; q < lnw.size(); q++) if (has_cut(leaves[lnw[q]], v, xi)) wt += l1pg[lnw[q]];
      // a leaf's statistics change only at a cutpoint whose bin holds some of its observations: recompute it then only
      for (size_t a = 0; a < nb; a++) {
        if (dirty[a]) {
          const double b_ = (a < nl) ? pr[a] : tr[a] - pr[a], M_ = (a < nl) ? ps[a] : ts[a] - ps[a];
          lcache[a] = leaf_lik(b_, M_, pi.tau);
          dirty[a] = 0;
        }
        wt += lcache[a];
      }
      lw.push_back(wt); cv.push_back(v); cc.push_back((size_t)c);
    }
  }
  eta->setv(v0); eta->setc(c0);
}

// log M(v') - log M(v) from the weights lw (which carry log s_u) of the candidates of v and v'. -inf when v' has no
// admissible rule at eta, +inf when v has none (a zero-mass current state, e.g. s_v = 0).
inline double log_ratio(const std::vector<double>& lw, const std::vector<size_t>& cv, size_t v, size_t vp,
                        const std::vector<double>& pv, int ctl = 0)
{
  const double ninf = -std::numeric_limits<double>::infinity();
  double mv = ninf, mvp = ninf;
  for (size_t k = 0; k < lw.size(); k++) { if (cv[k] == v) mv = std::max(mv, lw[k]); if (cv[k] == vp) mvp = std::max(mvp, lw[k]); }
  if (mvp == ninf) return ninf;
  if (mv == ninf) return std::numeric_limits<double>::infinity();
  double zv = 0.0, zvp = 0.0;
  for (size_t k = 0; k < lw.size(); k++) { if (cv[k] == v) zv += std::exp(lw[k] - mv); if (cv[k] == vp) zvp += std::exp(lw[k] - mvp); }
  double lr = (mvp + std::log(zvp)) - (mv + std::log(zv));
  if (ctl != 4) lr += std::log(pv[v]) - std::log(pv[vp]);
  return lr;
}

// One collapsed-change scan over every internal node of tree x (pre-order). w = observation weights 1 / sigma_i^2,
// di.y = the partial residuals of this tree. nv follows the drawn variables. Returns the number of rules changed.
inline int cchange_all(tree& x, xinfo& xi, dinfo& di, pinfo& pi, std::vector<size_t>& nv, std::vector<double>& pv,
                       const int* kidx, const double* w, rn& gen, int ctl = 0)
{
  if (x.treesize() == 1) return 0;
  const size_t p = pv.size();
  Route Rt; Rt.build(x, xi, di);
  tree::npv all; x.getnodes(all);
  std::vector<size_t> reg, cv, cc; std::vector<double> lw; std::vector<int> la, ra; std::vector<tree::tree_p> leaves;
  std::vector<char> only(p, 0);
  double ssum = 0.0; for (size_t u = 0; u < p; u++) ssum += pv[u];
  int changed = 0;
  for (size_t a = 0; a < all.size(); a++) {
    tree::tree_p eta = all[a];
    if (eta->getl() == 0) continue;
    const size_t v = eta->getv();
    // 1. proposal v' ~ s (inverse cdf, one uniform)
    size_t vp = p - 1;
    { const double u = gen.uniform() * ssum; double cum = 0.0;
      for (size_t q = 0; q < p; q++) { cum += pv[q]; if (u < cum) { vp = q; break; } } }
    Rt.region_of(eta, reg);
    only[v] = 1; only[vp] = 1;
    rule_weights(eta, reg, xi, di, pi, pv, kidx, w, lw, cv, cc, la, ra, leaves, &only, ctl);
    only[v] = 0; only[vp] = 0;
    // 2. Metropolis-Hastings on the variable
    size_t vn = v;
    if (vp != v) {
      const double lr = log_ratio(lw, cv, v, vp, pv, ctl);
      if (lr == std::numeric_limits<double>::infinity()) vn = vp;
      else if (lr > -std::numeric_limits<double>::infinity() && std::log(gen.uniform()) < lr) vn = vp;
    }
    // 3. the cutpoint of vn from its full conditional
    std::vector<double> lwc; std::vector<size_t> ccc;
    for (size_t k = 0; k < lw.size(); k++) if (cv[k] == vn) { lwc.push_back(ctl == 5 ? 0.0 : lw[k]); ccc.push_back(cc[k]); }
    if (lwc.empty()) continue;                       // not reached from a valid state: its own rule is admissible
    const size_t cnew = ccc[draw_log(lwc, gen)];
    if (vn == eta->getv() && cnew == eta->getc()) continue;
    changed++;
    nv[eta->getv()]--; nv[vn]++; eta->setv(vn); eta->setc(cnew);
    std::vector<int> lid(leaves.size());
    for (size_t b = 0; b < leaves.size(); b++) { lid[b] = Rt.id_of(leaves[b]); Rt.mem[lid[b]].clear(); }
    const int* kv = kidx + vn * di.n;
    for (size_t k = 0; k < reg.size(); k++) {
      const size_t i = reg[k];
      Rt.mem[lid[(kv[i] <= (int)cnew) ? la[k] : ra[k]]].push_back(i);
    }
  }
  return changed;
}

}  // namespace bart_cchange
#endif
