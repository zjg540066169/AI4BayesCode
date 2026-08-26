# ----------------------------------------------------------------------------
# Pin the native BLAS to one thread, as early as possible.
#
# ai4bayescode_run_chains() parallelises with parallel::mclapply(), i.e.
# fork(). A multithreaded BLAS -- macOS Accelerate/vecLib above all, but also
# OpenBLAS with pthreads and MKL -- keeps a worker-thread pool; fork() copies
# only the calling thread, so the child inherits a pool whose threads do not
# exist and the first heavy BLAS call there dies. On macOS the crash is inside
# libdispatch (`_dispatch_root_queue_push` under a LAPACK entry point,
# reported as "crashed on child side of fork pre-exec"), and it aborts the
# whole R session -- it cannot be caught and retried.
#
# The cap has to be in place BEFORE the first BLAS call, because that is when
# Accelerate decides whether to use its dispatch queues. Setting it later --
# inside ai4bayescode_run_chains(), say -- has no effect whatsoever. Hence:
# here, at load.
#
# Only variables the caller has not already set are touched.
#
# The cap costs nothing in practice: the chains are separate processes
# already, so mc.cores workers x k BLAS threads each would only oversubscribe
# the machine. Measured on the GP examples, per-chain wall time is the same or
# slightly better at one thread, and the parallel speedup is a clean 4.2x on
# four cores.
.ai4b_pin_blas_threads <- function() {
    vars <- c("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS",
              "OMP_NUM_THREADS", "MKL_NUM_THREADS")
    unset <- vars[is.na(Sys.getenv(vars, unset = NA))]
    if (length(unset)) {
        do.call(Sys.setenv, as.list(stats::setNames(rep("1", length(unset)), unset)))
    }
    invisible(NULL)
}

.onLoad <- function(libname, pkgname) {
    .ai4b_pin_blas_threads()
}

.onAttach <- function(libname, pkgname) {
    v <- utils::packageVersion(pkgname)
    msg <- c(
        sprintf("AI4BayesCode %s -- stateful modular MCMC + AI codegen skill chain", v),
        "  ai4bayescode_example('GaussianLocationScale')   # try a built-in",
        "  ai4bayescode_source('./my_model.cpp')           # compile your own .cpp"
    )
    # Only nudge to set a key when none is configured yet (env var / set_key);
    # users who already have a key set don't see this line.
    if (!nzchar(.ai4b_provider_key("anthropic")) && !nzchar(.ai4b_provider_key("openai")))
        msg <- c(msg,
            "  ai4bayescode_set_key('sk-ant-...', 'anthropic') # do this 1st -- replace 'sk-ant-...' with YOUR real key")
    msg <- c(msg,
        "  ai4bayescode_generate('describe your model')    # NL -> validated sampler",
        "  ai4bayescode_skills_path()                       # for AI agents",
        ""
    )
    packageStartupMessage(paste(msg, collapse = "\n"))
}
