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

# Headers the joint_nuts_block freeze regression tests guard -- listed as build
# prerequisites so editing any of them rebuilds the tests (a header-only change
# otherwise leaves the stale test binary in place).
FREEZE_HDRS := $(INC_DIR)/AI4BayesCode/joint_nuts_block.hpp \
               $(INC_DIR)/AI4BayesCode/composite_block.hpp \
               $(INC_DIR)/AI4BayesCode/block_sampler.hpp \
               $(INC_DIR)/AI4BayesCode/shared_data.hpp \
               $(INC_DIR)/AI4BayesCode/constraints.hpp
# mcmclib's own internal includes expect its directory to be on the
# search path (so that "BaseMatrixOps/include/BaseMatrixOps.hpp" resolves).
MCMCLIB_INC := ../include/mcmclib
BMO_INC     := ../include/mcmclib/BaseMatrixOps/include
# Vendored Eigen 3.4 (used by gmrf_precision_block for sparse Cholesky).
EIGEN_INC   := ../include/eigen

# Armadillo headers. Preference order, so the same Makefile works on a dev
# Mac and on a bare Linux CI runner:
#   1. ARMA_INC=... passed in explicitly
#   2. RcppArmadillo's bundled copy, asked for via Rscript (any platform, any
#      R library path -- the old hard-coded /Library/Frameworks path only ever
#      worked on macOS)
#   3. the system libarmadillo-dev headers, which need no R at all
ARMA_INC ?= $(shell Rscript -e 'cat(system.file("include", package="RcppArmadillo"))' 2>/dev/null)
ARMA_FLAG := $(if $(strip $(ARMA_INC)),-I$(ARMA_INC),)

UNAME_S := $(shell uname -s)

CXX      ?= c++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter
CPPFLAGS ?= -I$(INC_DIR) -I$(MCMCLIB_INC) -I$(BMO_INC) -I$(EIGEN_INC) $(ARMA_FLAG) \
            -DMCMC_ENABLE_ARMA_WRAPPERS \
            -DARMA_DONT_USE_WRAPPER
LDFLAGS  ?=
# macOS: Accelerate provides BLAS/LAPACK. Elsewhere: the usual pair.
ifeq ($(UNAME_S),Darwin)
LDLIBS   ?= -framework Accelerate
else
LDLIBS   ?= -lblas -llapack
endif

TARGETS  := test_constraints test_binary_gibbs test_dirichlet_gibbs test_categorical_gibbs test_gaussian_linear test_simplex_dirichlet test_nuts_variance test_vi_mean_field_gaussian test_vi_correlated_gaussian test_vi_hybrid_composition test_vi_glmm_logistic test_vi_bnn_regression test_vi_full_rank_correlated test_vi_bnn_full_rank test_readapt_NUTS test_target_accept_config test_joint_nuts_block_mixed_stress test_bde_scorer test_score_cache test_order_mcmc_block test_order_mcmc_block_stress test_order_mcmc_block_diagnostics test_bge_scorer test_partition_geometry test_partition_mcmc_diagnostics test_partition_mcmc_rhat test_order_mcmc_block_partition test_freeze_predict_history test_freeze_element test_freeze_dense_pilot test_freeze_joint_nuts_slot test_ode_rk45_inplace test_dag_key_validation test_genbart_slice_mstep test_predict_dag_depth test_history_alignment test_wiring_guards test_composite_bare_name_slot_fallback test_freeze_normal_gamma_cluster test_freeze_rjmcmc_subkey test_joint_nuts_block_constraints test_joint_nuts_dimchanging test_vi_publish_and_history

# The VI publish/history regression guards four VI headers plus the composite
# freeze path -- a header-only edit otherwise leaves a stale binary in place.
test_vi_publish_and_history: $(INC_DIR)/AI4BayesCode/vi_block.hpp \
    $(INC_DIR)/AI4BayesCode/mean_field_gaussian_vi_block.hpp \
    $(INC_DIR)/AI4BayesCode/full_rank_gaussian_vi_block.hpp \
    $(INC_DIR)/AI4BayesCode/structured_categorical_vi_block.hpp \
    $(INC_DIR)/AI4BayesCode/mean_field_categorical_vi_block.hpp \
    $(INC_DIR)/AI4BayesCode/composite_block.hpp

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
	@echo "=== Running test_target_accept_config ==="
	./test_target_accept_config
	@echo
	@echo "=== Running test_joint_nuts_block_mixed_stress ==="
	./test_joint_nuts_block_mixed_stress
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
	@echo "=== Running test_bge_scorer ==="
	./test_bge_scorer
	@echo
	@echo "=== Running test_partition_geometry ==="
	./test_partition_geometry
	@echo
	@echo "=== Running test_partition_mcmc_diagnostics ==="
	./test_partition_mcmc_diagnostics
	@echo
	@echo "=== Running test_partition_mcmc_rhat ==="
	./test_partition_mcmc_rhat
	@echo
	@echo "=== Running test_order_mcmc_block_partition ==="
	./test_order_mcmc_block_partition
	@echo
	@echo "=== Running test_freeze_predict_history ==="
	./test_freeze_predict_history
	@echo
	@echo "=== Running test_freeze_element ==="
	./test_freeze_element
	@echo
	@echo "=== Running test_freeze_dense_pilot ==="
	./test_freeze_dense_pilot
	@echo
	@echo "=== Running test_freeze_joint_nuts_slot ==="
	./test_freeze_joint_nuts_slot
	@echo
	@echo "=== Running test_ode_rk45_inplace ==="
	./test_ode_rk45_inplace
	@echo
	@echo "=== Running test_dag_key_validation ==="
	./test_dag_key_validation
	@echo
	@echo "=== Running test_genbart_slice_mstep ==="
	./test_genbart_slice_mstep
	@echo
	@echo "=== Running test_history_alignment ==="
	./test_history_alignment
	@echo
	@echo "=== Running test_predict_dag_depth ==="
	./test_predict_dag_depth
	@echo
	@echo "=== Running test_wiring_guards ==="
	./test_wiring_guards
	@echo
	@echo "=== Running test_composite_bare_name_slot_fallback ==="
	./test_composite_bare_name_slot_fallback
	@echo
	@echo "=== Running test_freeze_normal_gamma_cluster ==="
	./test_freeze_normal_gamma_cluster
	@echo
	@echo "=== Running test_freeze_rjmcmc_subkey ==="
	./test_freeze_rjmcmc_subkey
	@echo
	@echo "=== Running test_joint_nuts_block_constraints ==="
	./test_joint_nuts_block_constraints
	@echo
	@echo "=== Running test_joint_nuts_dimchanging ==="
	./test_joint_nuts_dimchanging
	@echo
	@echo "=== Running test_vi_publish_and_history ==="
	./test_vi_publish_and_history

# These three had no rule, so make's built-in %: %.cpp applied and a
# header-only edit left the previous binary in place, still reporting PASS.
test_predict_dag_depth: test_predict_dag_depth.cpp \
    $(INC_DIR)/AI4BayesCode/shared_data.hpp \
    $(INC_DIR)/AI4BayesCode/block_sampler.hpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

test_history_alignment: test_history_alignment.cpp \
    $(INC_DIR)/AI4BayesCode/composite_block.hpp \
    $(INC_DIR)/AI4BayesCode/inv_gamma_gibbs_block.hpp \
    $(INC_DIR)/AI4BayesCode/gamma_gibbs_block.hpp \
    $(INC_DIR)/AI4BayesCode/block_sampler.hpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

test_wiring_guards: test_wiring_guards.cpp \
    $(INC_DIR)/AI4BayesCode/shared_data.hpp \
    $(INC_DIR)/AI4BayesCode/composite_block.hpp \
    $(INC_DIR)/AI4BayesCode/nuts_block.hpp \
    $(INC_DIR)/AI4BayesCode/piecewise_exponential_gibbs_block.hpp \
    $(INC_DIR)/AI4BayesCode/gamma_gibbs_block.hpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

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

# FORK MARKER (2026-07-26 restore) [target_accept API expose, default=0.55]
# New test verifying the wrapper-layer target_accept_rate API surface (14
# assertions). Uses the same header set as test_readapt_NUTS.
test_target_accept_config: test_target_accept_config.cpp $(AI4BAYESCODE_READAPT_HEADERS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

test_joint_nuts_block_mixed_stress: test_joint_nuts_block_mixed_stress.cpp $(FREEZE_HDRS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

test_freeze_predict_history: test_freeze_predict_history.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

test_freeze_element: test_freeze_element.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

test_freeze_dense_pilot: test_freeze_dense_pilot.cpp $(FREEZE_HDRS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

test_freeze_joint_nuts_slot: test_freeze_joint_nuts_slot.cpp $(FREEZE_HDRS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

test_ode_rk45_inplace: test_ode_rk45_inplace.cpp $(INC_DIR)/AI4BayesCode/ode_rk45.hpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

test_dag_key_validation: test_dag_key_validation.cpp \
                         $(INC_DIR)/AI4BayesCode/composite_block.hpp \
                         $(INC_DIR)/AI4BayesCode/shared_data.hpp \
                         $(INC_DIR)/AI4BayesCode/block_sampler.hpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

test_genbart_slice_mstep: test_genbart_slice_mstep.cpp \
                          ../bart_pure_cpp/src/GENBART/slice.h \
                          ../bart_pure_cpp/src/GENBART/rjmcmc_bd.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

clean:
	# The build also leaves .rebuilt / .rb / .new / .builderr / .be markers;
	# without these a stale one can outlive the binary it describes.
	rm -f $(TARGETS) *.rebuilt *.rb *.new *.builderr *.be
