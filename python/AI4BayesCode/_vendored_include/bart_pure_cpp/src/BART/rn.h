/*
 *  BART: Bayesian Additive Regression Trees
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
 *  Modifications by Jungang Zou, 2024.
 *  - To make sure the random seeds set in R function work, we deleted two implementation of arn.
 *
 *  Modifications by Jungang Zou, 2026.
 *  - set_wts keeps the normalised weights, and discrete() keeps the binomial parameters of the categorical draw
 *    while the weights are unchanged.
 *
 *  These modifications comply with the terms of the GNU General Public License 
 *  version 2 (GPL-2).
 */


#ifndef GUARD_rn_h
#define GUARD_rn_h

#include <cmath>
#include <cstring>
#include <vector>

// Pure-C++ (R-free) build: pull in the standalone R-API shim so that
// arn's R::norm_rand() / R::rgamma() / R::rmultinom() etc. resolve to
// the seedable std::mt19937_64-backed implementations in r_compat.h.
// Under the Rcpp build (NoRcpp undefined) this include is a no-op and
// the real R API is used instead — dual-mode preserved.
#ifdef NoRcpp
#include "../r_compat.h"
#endif
// double log_sum_exp(std::vector<double>& v){
//   double mx=v[0],sm=0.;
//   for(size_t i=0;i<v.size();i++) if(v[i]>mx) mx=v[i];
//   for(size_t i=0;i<v.size();i++){
//     sm += exp(v[i]-mx);
//   }
//   return mx+log(sm);
// }

//pure virtual base class for random numbers
class rn
{
 public:
  rn() {}
  virtual double normal() = 0; //standard normal
  virtual double uniform() = 0; //uniform(0,1)
  virtual double chi_square(double df) = 0; //chi-square
  virtual double exp() = 0; //exponential
  virtual double log_gamma(double shape) = 0; 
  virtual double gamma(double shape, double rate) = 0; 
  virtual double beta(double a, double b) = 0; 
  virtual size_t discrete() = 0; //discrete (categorical) distribution
  virtual size_t geometric(double p) = 0; //geometric distribution
  virtual void set_wts(std::vector<double>& _wts) = 0;
  virtual std::vector<double> log_dirichlet(std::vector<double>& alpha) = 0; 
  virtual ~rn() {}
};

//abstract random number generator based on R/Rcpp
class arn: public rn
{
 public:
  //constructor
 //arn():df(1) {}
 arn() {}
  //virtual
  virtual ~arn() {}
  virtual double normal() {return R::norm_rand();}
  virtual double uniform() { return R::unif_rand();}
  virtual double chi_square(double df) {return R::rchisq(df);}
  virtual double exp() {return R::exp_rand();}
  virtual double log_gamma(double shape) {
    double y=log(R::rgamma(shape+1., 1.)), z=log(this->uniform())/shape;
    return y+z; 
  }
  virtual double gamma(double shape, double rate) {
    if(shape<0.01) return ::exp(this->log_gamma(shape))/rate;
    else return R::rgamma(shape, 1.)/rate; 
  } 
/*
  virtual double gamma_left(double shape, double rate, double a) {
// left truncated gamma: shape must be greater than 1!
// Random Number Generation and Monte Carlo Methods, Second Edition, pp. 180-1
    if(shape<=1.) return -1.;
    double y, x=1., c=2., a_scale=a*rate, shape_shift=shape-1.,
    lambda=0.5*(a_scale-shape+sqrt(pow(a_scale-shape, 2.)+4.*a_scale))/a_scale,
      lambda_shift=1.-lambda, 
      C=1.+log(lambda_shift/shape_shift);
    while(c>x) { // do at least once
      x=R::exp_rand(); 
      y=a_scale+R::exp_rand()/lambda; 
      c=lambda_shift*y-shape_shift*(log(y)+C);
    }
    return y/rate;
  } 
*/
  virtual double beta(double a, double b) {
    double x1=this->gamma(a, 1.), x2=this->gamma(b, 1.);
    return x1/(x1+x2);
  } 
 virtual size_t discrete() {
    size_t p=wts.size(), x=0;
#ifdef NoRcpp
    if(p>0) return discrete_table_();
#endif
    std::vector<int> vOut (p,0);
    R::rmultinom(1,&wts[0],p,&vOut[0]); 
    if(vOut[0]==0) for(size_t j=1;j<p;j++) x += j*vOut[j]; 
    return x;
  }
/*
    size_t p=wts.size();
    std::vector<int> vOut (p,0); // vector of multionomial output
    std::vector<int> seq (p,0); 
    for(size_t j=0;j<p;j++) seq[j]=j;
    R::rmultinom(1,&wts[0],p,&vOut[0]); // vector vOut contains selected component
    return std::inner_product(vOut.begin(),vOut.end(),seq.begin(),0);
  }
*/
  virtual size_t geometric(double p) {return R::rgeom(p);}
  virtual void set_wts(std::vector<double>& _wts) {
    // wts and the draw table depend on _wts only: kept when _wts equals, bit for bit, the previous input
    if(wts_in_.size()==_wts.size() &&
       (_wts.empty() || std::memcmp(wts_in_.data(), _wts.data(), _wts.size()*sizeof(double))==0)) return;
    double smw=0.;
    wts.clear();
    for(size_t j=0;j<_wts.size();j++) smw+=_wts[j];
    for(size_t j=0;j<_wts.size();j++) wts.push_back(_wts[j]/smw);
    wts_in_.assign(_wts.begin(), _wts.end());
#ifdef NoRcpp
    tab_.clear(); tab_ok_=false;
#endif
  }
  virtual std::vector<double> log_dirichlet(std::vector<double>& alpha){
    size_t k=alpha.size();
    std::vector<double> draw(k);
    double lse;
    for(size_t j=0;j<k;j++) draw[j]=this->log_gamma(alpha[j]);
    double mx=draw[0],sm=0.;
    for(size_t i=0;i<draw.size();i++) if(draw[i]>mx) mx=draw[i];
    for(size_t i=0;i<draw.size();i++){
      sm += std::exp(draw[i]-mx);
    }
    lse= mx+log(sm);
    for(size_t j=0;j<k;j++) {
      draw[j] -= lse;
      //draw[j]=::exp(draw[j]);
    }
    return draw;
  }
 private:
  std::vector<double> wts;
  std::vector<double> wts_in_;   // the weights last passed to set_wts
#ifdef NoRcpp
  // The draw of R::rmultinom(1, &wts[0], p, .) (bart_rng::rmultinom) followed by the scan in discrete(), with the same
  // engine calls: category j is tried with binomial(1, wts[j] / rem_j), rem_j = sum of wts minus wts[0..j-1], in
  // order until the first success; none tried once rem_j <= 0; category p-1 otherwise. tab_[j] holds the binomial
  // parameters of category j, built in order up to the last category reached and kept until set_wts changes wts.
  std::vector<std::binomial_distribution<int>::param_type> tab_;
  double tab_rem_ = 0.0;         // rem_j for j = tab_.size()
  bool tab_ok_ = false;
  size_t discrete_table_() {
    const size_t p=wts.size();
    if(!tab_ok_) {
      double norm=0.0;
      for(size_t j=0;j<p;j++) norm+=wts[j];
      tab_.clear(); tab_rem_=norm; tab_ok_=true;
    }
    for(size_t j=0;j+1<p;j++) {
      if(j==tab_.size()) {
        if(tab_rem_<=0.0) break;
        double pj=wts[j]/tab_rem_;
        if(pj>1.0) pj=1.0;
        if(pj<0.0) pj=0.0;
        tab_.push_back(std::binomial_distribution<int>::param_type(1, pj));
        tab_rem_-=wts[j];
      }
      std::binomial_distribution<int> d(tab_[j]);
      if(d(bart_rng::engine())!=0) return j;
    }
    return p-1;
  }
#endif
#ifndef NoRcpp
  Rcpp::RNGScope RNGstate;   // R-build only; the pure-C++ RNG (r_compat.h)
                             // manages its own thread-local engine state.
#endif
};

#endif 
