# Makefile for block_mcmc end-to-end smoke tests.
#
# Requires:
#   - a C++17-capable compiler (clang++ or g++)
#   - Armadillo installed and findable by the compiler
#
# Usage:
#   make                 # build and run the Gaussian linear test
#   make build           # build only
#   make clean           # remove built binaries
#
# The Makefile assumes it is invoked from the tests/ directory.

INC_DIR   := ../include
# mcmclib's own internal includes expect its directory to be on the
# search path (so that "BaseMatrixOps/include/BaseMatrixOps.hpp" resolves).
MCMCLIB_INC := ../include/mcmclib
BMO_INC     := ../include/mcmclib/BaseMatrixOps/include
# Vendored Eigen 3.4 (used by gmrf_precision_block for sparse Cholesky).
EIGEN_INC   := ../include/eigen

# Use RcppArmadillo's bundled Armadillo headers so we don't need a
# separate libarmadillo install. This works on macOS whenever the user
# has RcppArmadillo installed in R.
R_ARMA_INC := /Library/Frameworks/R.framework/Versions/Current/Resources/library/RcppArmadillo/include

CXX      ?= clang++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter
CPPFLAGS ?= -I$(INC_DIR) -I$(MCMCLIB_INC) -I$(BMO_INC) -I$(EIGEN_INC) -I$(R_ARMA_INC) \
            -DMCMC_ENABLE_ARMA_WRAPPERS \
            -DARMA_DONT_USE_WRAPPER
LDFLAGS  ?=
# On macOS, link directly against Accelerate for BLAS/LAPACK.
LDLIBS   ?= -framework Accelerate

TARGETS  := test_constraints test_binary_gibbs test_dirichlet_gibbs test_categorical_gibbs test_gaussian_linear test_simplex_dirichlet test_nuts_variance test_nuts_block_stress test_vi_mean_field_gaussian test_vi_correlated_gaussian test_vi_hybrid_composition test_vi_glmm_logistic test_vi_bnn_regression test_vi_full_rank_correlated test_vi_bnn_full_rank test_readapt_NUTS test_ising_cluster_block test_ising_cluster_block_diagnostics test_ising_sw_vs_single_site test_ising_cluster_block_stress test_gmrf_precision_block test_gmrf_precision_block_diagnostics test_gmrf_precision_block_stress test_hmm_block_stress test_pg_logistic_block_stress test_joint_nuts_block_stress test_joint_nuts_block_mixed_stress test_rjmcmc_block_stress test_bde_scorer test_score_cache test_order_mcmc_block test_order_mcmc_block_stress test_order_mcmc_block_diagnostics test_bge_scorer test_partition_geometry test_partition_mcmc_diagnostics test_partition_mcmc_rhat test_order_mcmc_block_partition test_categorical_meanfield_vi_block test_categorical_meanfield_vi_chain test_categorical_meanfield_vi_cavi_cross test_categorical_meanfield_vi_stress test_structured_meanfield_vi_block test_structured_meanfield_vi_chain test_structured_meanfield_vi_cavi_cross test_structured_meanfield_vi_stress

.PHONY: all build run clean

all: run

build: $(TARGETS)

run: build
	@echo "=== Running test_constraints ==="
	./test_constraints
	@echo
	@echo "=== Running test_binary_gibbs ==="
	./test_binary_gibbs
	@echo
	@echo "=== Running test_dirichlet_gibbs ==="
	./test_dirichlet_gibbs
	@echo
	@echo "=== Running test_categorical_gibbs ==="
	./test_categorical_gibbs
	@echo
	@echo "=== Running test_gaussian_linear ==="
	./test_gaussian_linear
	@echo
	@echo "=== Running test_simplex_dirichlet ==="
	./test_simplex_dirichlet
	@echo
	@echo "=== Running test_nuts_variance ==="
	./test_nuts_variance
	@echo
	@echo "=== Running test_nuts_block_stress ==="
	./test_nuts_block_stress
	@echo
	@echo "=== Running test_vi_mean_field_gaussian ==="
	./test_vi_mean_field_gaussian
	@echo
	@echo "=== Running test_vi_correlated_gaussian ==="
	./test_vi_correlated_gaussian
	@echo
	@echo "=== Running test_vi_hybrid_composition ==="
	./test_vi_hybrid_composition
	@echo
	@echo "=== Running test_vi_glmm_logistic ==="
	./test_vi_glmm_logistic
	@echo
	@echo "=== Running test_vi_bnn_regression ==="
	./test_vi_bnn_regression
	@echo
	@echo "=== Running test_vi_full_rank_correlated ==="
	./test_vi_full_rank_correlated
	@echo
	@echo "=== Running test_vi_bnn_full_rank ==="
	./test_vi_bnn_full_rank
	@echo
	@echo "=== Running test_readapt_NUTS ==="
	./test_readapt_NUTS
	@echo
	@echo "=== Running test_ising_cluster_block ==="
	./test_ising_cluster_block
	@echo
	@echo "=== Running test_ising_cluster_block_diagnostics ==="
	./test_ising_cluster_block_diagnostics
	@echo
	@echo "=== Running test_ising_sw_vs_single_site ==="
	./test_ising_sw_vs_single_site
	@echo
	@echo "=== Running test_ising_cluster_block_stress ==="
	./test_ising_cluster_block_stress
	@echo
	@echo "=== Running test_gmrf_precision_block ==="
	./test_gmrf_precision_block
	@echo
	@echo "=== Running test_gmrf_precision_block_diagnostics ==="
	./test_gmrf_precision_block_diagnostics
	@echo
	@echo "=== Running test_gmrf_precision_block_stress ==="
	./test_gmrf_precision_block_stress
	@echo
	@echo "=== Running test_hmm_block_stress ==="
	./test_hmm_block_stress
	@echo
	@echo "=== Running test_pg_logistic_block_stress ==="
	./test_pg_logistic_block_stress
	@echo
	@echo "=== Running test_joint_nuts_block_stress ==="
	./test_joint_nuts_block_stress
	@echo
	@echo "=== Running test_joint_nuts_block_mixed_stress ==="
	./test_joint_nuts_block_mixed_stress
	@echo
	@echo "=== Running test_rjmcmc_block_stress ==="
	./test_rjmcmc_block_stress
	@echo
	@echo "=== Running test_bde_scorer ==="
	./test_bde_scorer
	@echo
	@echo "=== Running test_score_cache ==="
	./test_score_cache
	@echo
	@echo "=== Running test_order_mcmc_block ==="
	./test_order_mcmc_block
	@echo
	@echo "=== Running test_order_mcmc_block_stress ==="
	./test_order_mcmc_block_stress
	@echo
	@echo "=== Running test_order_mcmc_block_diagnostics ==="
	./test_order_mcmc_block_diagnostics
	@echo
	@echo "=== Running test_categorical_meanfield_vi_block ==="
	./test_categorical_meanfield_vi_block
	@echo
	@echo "=== Running test_categorical_meanfield_vi_chain ==="
	./test_categorical_meanfield_vi_chain
	@echo
	@echo "=== Running test_categorical_meanfield_vi_cavi_cross ==="
	./test_categorical_meanfield_vi_cavi_cross
	@echo
	@echo "=== Running test_categorical_meanfield_vi_stress ==="
	./test_categorical_meanfield_vi_stress
	@echo
	@echo "=== Running test_structured_meanfield_vi_block ==="
	./test_structured_meanfield_vi_block
	@echo
	@echo "=== Running test_structured_meanfield_vi_chain ==="
	./test_structured_meanfield_vi_chain
	@echo
	@echo "=== Running test_structured_meanfield_vi_cavi_cross ==="
	./test_structured_meanfield_vi_cavi_cross
	@echo
	@echo "=== Running test_structured_meanfield_vi_stress ==="
	./test_structured_meanfield_vi_stress

AI4BAYESCODE_HEADERS := \
  $(INC_DIR)/AI4BayesCode/block_sampler.hpp \
  $(INC_DIR)/AI4BayesCode/shared_data.hpp \
  $(INC_DIR)/AI4BayesCode/nuts_block.hpp \
  $(INC_DIR)/AI4BayesCode/composite_block.hpp \
  $(INC_DIR)/AI4BayesCode/constraints.hpp \
  $(INC_DIR)/AI4BayesCode/binary_gibbs_block.hpp \
  $(INC_DIR)/AI4BayesCode/dirichlet_gibbs_block.hpp \
  $(INC_DIR)/AI4BayesCode/categorical_gibbs_block.hpp

test_constraints: test_constraints.cpp $(AI4BAYESCODE_HEADERS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

test_binary_gibbs: test_binary_gibbs.cpp $(AI4BAYESCODE_HEADERS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

test_dirichlet_gibbs: test_dirichlet_gibbs.cpp $(AI4BAYESCODE_HEADERS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

test_categorical_gibbs: test_categorical_gibbs.cpp $(AI4BAYESCODE_HEADERS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

test_gaussian_linear: test_gaussian_linear.cpp $(AI4BAYESCODE_HEADERS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

test_simplex_dirichlet: test_simplex_dirichlet.cpp $(AI4BAYESCODE_HEADERS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

test_nuts_variance: test_nuts_variance.cpp $(AI4BAYESCODE_HEADERS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

test_nuts_block_stress: test_nuts_block_stress.cpp $(AI4BAYESCODE_HEADERS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

# Block 3: order_mcmc_block (Friedman-Koller 2003)
AI4BAYESCODE_ORDER_MCMC_HEADERS := \
  $(INC_DIR)/AI4BayesCode/block_sampler.hpp \
  $(INC_DIR)/AI4BayesCode/i_scorer.hpp \
  $(INC_DIR)/AI4BayesCode/bde_scorer.hpp \
  $(INC_DIR)/AI4BayesCode/bge_scorer.hpp \
  $(INC_DIR)/AI4BayesCode/score_cache.hpp \
  $(INC_DIR)/AI4BayesCode/partition_state.hpp \
  $(INC_DIR)/AI4BayesCode/order_mcmc_block.hpp

test_bde_scorer: test_bde_scorer.cpp $(INC_DIR)/AI4BayesCode/bde_scorer.hpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

test_score_cache: test_score_cache.cpp $(INC_DIR)/AI4BayesCode/bde_scorer.hpp $(INC_DIR)/AI4BayesCode/score_cache.hpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

test_order_mcmc_block: test_order_mcmc_block.cpp $(AI4BAYESCODE_ORDER_MCMC_HEADERS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

test_order_mcmc_block_stress: test_order_mcmc_block_stress.cpp $(AI4BAYESCODE_ORDER_MCMC_HEADERS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

test_order_mcmc_block_diagnostics: test_order_mcmc_block_diagnostics.cpp $(AI4BAYESCODE_ORDER_MCMC_HEADERS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

# Block 3 v1.2.1: BGe score + partition MCMC
test_bge_scorer: test_bge_scorer.cpp $(INC_DIR)/AI4BayesCode/bge_scorer.hpp $(INC_DIR)/AI4BayesCode/i_scorer.hpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

test_partition_geometry: test_partition_geometry.cpp $(AI4BAYESCODE_ORDER_MCMC_HEADERS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

test_partition_mcmc_diagnostics: test_partition_mcmc_diagnostics.cpp $(AI4BAYESCODE_ORDER_MCMC_HEADERS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

test_partition_mcmc_rhat: test_partition_mcmc_rhat.cpp $(AI4BAYESCODE_ORDER_MCMC_HEADERS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

test_order_mcmc_block_partition: test_order_mcmc_block_partition.cpp $(AI4BAYESCODE_ORDER_MCMC_HEADERS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

# Data + AI-sample generator for the BiDAG partition cross-check (run the R
# script audit_PartitionMCMCBN_vs_BiDAG.R on its output). Not a self-checking
# unit test, so it is NOT in TARGETS / the run panel.
partition_crosscheck_gen: partition_crosscheck_gen.cpp $(AI4BAYESCODE_ORDER_MCMC_HEADERS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

# VI-family headers (Phase 2 / 3 / 4 of the VI track)
AI4BAYESCODE_VI_HEADERS := \
  $(INC_DIR)/AI4BayesCode/vi_block.hpp \
  $(INC_DIR)/AI4BayesCode/vi_optimizer.hpp \
  $(INC_DIR)/AI4BayesCode/mean_field_gaussian_vi_block.hpp \
  $(INC_DIR)/AI4BayesCode/full_rank_gaussian_vi_block.hpp \
  $(INC_DIR)/AI4BayesCode/block_sampler.hpp \
  $(INC_DIR)/AI4BayesCode/composite_block.hpp

test_vi_mean_field_gaussian: test_vi_mean_field_gaussian.cpp $(AI4BAYESCODE_VI_HEADERS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

test_vi_correlated_gaussian: test_vi_correlated_gaussian.cpp $(AI4BAYESCODE_VI_HEADERS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

test_vi_hybrid_composition: test_vi_hybrid_composition.cpp $(AI4BAYESCODE_VI_HEADERS) \
                            $(INC_DIR)/AI4BayesCode/nuts_block.hpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

test_vi_glmm_logistic: test_vi_glmm_logistic.cpp $(AI4BAYESCODE_VI_HEADERS) \
                       $(INC_DIR)/AI4BayesCode/nuts_block.hpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

test_vi_bnn_regression: test_vi_bnn_regression.cpp $(AI4BAYESCODE_VI_HEADERS) \
                        $(INC_DIR)/AI4BayesCode/nuts_block.hpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

test_vi_full_rank_correlated: test_vi_full_rank_correlated.cpp $(AI4BAYESCODE_VI_HEADERS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

test_vi_bnn_full_rank: test_vi_bnn_full_rank.cpp $(AI4BAYESCODE_VI_HEADERS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

# readapt_NUTS — 7th-method kernel tuning (NUTS-family only)
AI4BAYESCODE_READAPT_HEADERS := \
  $(INC_DIR)/AI4BayesCode/block_sampler.hpp \
  $(INC_DIR)/AI4BayesCode/nuts_block.hpp \
  $(INC_DIR)/AI4BayesCode/joint_nuts_block.hpp \
  $(INC_DIR)/AI4BayesCode/composite_block.hpp \
  $(INC_DIR)/AI4BayesCode/binary_gibbs_block.hpp \
  $(INC_DIR)/AI4BayesCode/constraints.hpp

test_readapt_NUTS: test_readapt_NUTS.cpp $(AI4BAYESCODE_READAPT_HEADERS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

# Block 1 (v1.2 specialized blocks): ising_cluster_block (Swendsen-Wang)
AI4BAYESCODE_ISING_HEADERS := \
  $(INC_DIR)/AI4BayesCode/block_sampler.hpp \
  $(INC_DIR)/AI4BayesCode/ising_cluster_block.hpp

test_ising_cluster_block: test_ising_cluster_block.cpp $(AI4BAYESCODE_ISING_HEADERS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

test_ising_cluster_block_diagnostics: test_ising_cluster_block_diagnostics.cpp $(AI4BAYESCODE_ISING_HEADERS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

test_ising_sw_vs_single_site: test_ising_sw_vs_single_site.cpp $(AI4BAYESCODE_ISING_HEADERS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

test_ising_cluster_block_stress: test_ising_cluster_block_stress.cpp $(AI4BAYESCODE_ISING_HEADERS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

# Block 2 (v1.2 specialized blocks): gmrf_precision_block (Rue 2001 sparse Cholesky)
AI4BAYESCODE_GMRF_HEADERS := \
  $(INC_DIR)/AI4BayesCode/block_sampler.hpp \
  $(INC_DIR)/AI4BayesCode/gmrf_precision_block.hpp

test_gmrf_precision_block: test_gmrf_precision_block.cpp $(AI4BAYESCODE_GMRF_HEADERS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

test_gmrf_precision_block_diagnostics: test_gmrf_precision_block_diagnostics.cpp $(AI4BAYESCODE_GMRF_HEADERS) $(INC_DIR)/AI4BayesCode/ising_cluster_block.hpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

test_gmrf_precision_block_stress: test_gmrf_precision_block_stress.cpp $(AI4BAYESCODE_GMRF_HEADERS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

# HMM block stress (parallels Block 1-5 R-pattern)
AI4BAYESCODE_HMM_HEADERS := \
  $(INC_DIR)/AI4BayesCode/block_sampler.hpp \
  $(INC_DIR)/AI4BayesCode/hmm_block.hpp

test_hmm_block_stress: test_hmm_block_stress.cpp $(AI4BAYESCODE_HMM_HEADERS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

# pg_logistic_block stress (parallels Block 1-5 R-pattern)
AI4BAYESCODE_PGLOGISTIC_HEADERS := \
  $(INC_DIR)/AI4BayesCode/block_sampler.hpp \
  $(INC_DIR)/AI4BayesCode/pg_logistic_block.hpp

test_pg_logistic_block_stress: test_pg_logistic_block_stress.cpp $(AI4BAYESCODE_PGLOGISTIC_HEADERS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

# joint_nuts_block stress (parallels Block 1-5 R-pattern)
AI4BAYESCODE_JOINT_NUTS_HEADERS := \
  $(INC_DIR)/AI4BayesCode/block_sampler.hpp \
  $(INC_DIR)/AI4BayesCode/nuts_block.hpp \
  $(INC_DIR)/AI4BayesCode/joint_nuts_block.hpp

test_joint_nuts_block_stress: test_joint_nuts_block_stress.cpp $(AI4BAYESCODE_JOINT_NUTS_HEADERS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

# joint_nuts_block_mixed stress
AI4BAYESCODE_JOINT_NUTS_MIXED_HEADERS := \
  $(INC_DIR)/AI4BayesCode/block_sampler.hpp \
  $(INC_DIR)/AI4BayesCode/nuts_block.hpp \
  $(INC_DIR)/AI4BayesCode/joint_nuts_block_mixed.hpp

test_joint_nuts_block_mixed_stress: test_joint_nuts_block_mixed_stress.cpp $(AI4BAYESCODE_JOINT_NUTS_MIXED_HEADERS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

# rjmcmc_block stress
AI4BAYESCODE_RJMCMC_HEADERS := \
  $(INC_DIR)/AI4BayesCode/block_sampler.hpp \
  $(INC_DIR)/AI4BayesCode/rjmcmc_block.hpp

test_rjmcmc_block_stress: test_rjmcmc_block_stress.cpp $(AI4BAYESCODE_RJMCMC_HEADERS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

# Block 4 (v1.2 specialized blocks): categorical_meanfield_vi_block
AI4BAYESCODE_CATVI_HEADERS := \
  $(INC_DIR)/AI4BayesCode/block_sampler.hpp \
  $(INC_DIR)/AI4BayesCode/vi_block.hpp \
  $(INC_DIR)/AI4BayesCode/vi_optimizer.hpp \
  $(INC_DIR)/AI4BayesCode/categorical_meanfield_vi_block.hpp

test_categorical_meanfield_vi_block: test_categorical_meanfield_vi_block.cpp $(AI4BAYESCODE_CATVI_HEADERS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

test_categorical_meanfield_vi_chain: test_categorical_meanfield_vi_chain.cpp $(AI4BAYESCODE_CATVI_HEADERS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

test_categorical_meanfield_vi_cavi_cross: test_categorical_meanfield_vi_cavi_cross.cpp $(AI4BAYESCODE_CATVI_HEADERS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

test_categorical_meanfield_vi_stress: test_categorical_meanfield_vi_stress.cpp $(AI4BAYESCODE_CATVI_HEADERS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

# Block 5 (v1.2 specialized blocks): structured_meanfield_vi_block
AI4BAYESCODE_STRUCTVI_HEADERS := \
  $(INC_DIR)/AI4BayesCode/block_sampler.hpp \
  $(INC_DIR)/AI4BayesCode/vi_block.hpp \
  $(INC_DIR)/AI4BayesCode/vi_optimizer.hpp \
  $(INC_DIR)/AI4BayesCode/categorical_meanfield_vi_block.hpp \
  $(INC_DIR)/AI4BayesCode/structured_meanfield_vi_block.hpp

test_structured_meanfield_vi_block: test_structured_meanfield_vi_block.cpp $(AI4BAYESCODE_STRUCTVI_HEADERS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

test_structured_meanfield_vi_chain: test_structured_meanfield_vi_chain.cpp $(AI4BAYESCODE_STRUCTVI_HEADERS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

test_structured_meanfield_vi_cavi_cross: test_structured_meanfield_vi_cavi_cross.cpp $(AI4BAYESCODE_STRUCTVI_HEADERS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

test_structured_meanfield_vi_stress: test_structured_meanfield_vi_stress.cpp $(AI4BAYESCODE_STRUCTVI_HEADERS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

clean:
	rm -f $(TARGETS)
