# Compile and run a sampler when the R library and the temp folder both live
# under paths with a space and non-ASCII characters, as they do for a Windows
# user whose account name has them. The Windows CI job sets AI4B_LIB and TMP.

lib <- Sys.getenv("AI4B_LIB")
stopifnot(nzchar(lib), dir.exists(lib))
.libPaths(c(lib, .libPaths()))
suppressPackageStartupMessages(library(AI4BayesCode))
cat("package:", find.package("AI4BayesCode"), "\n")
cat("tempdir:", tempdir(), "\n")

ai4bayescode_example("GaussianLocationScale")
set.seed(1)
m <- new(GaussianLocationScale, rnorm(200, 3, 2), rng_seed = 7L)
m$step(2000L)
cur <- m$get_current()
print(cur)
stopifnot(abs(cur$mu - 3) < 1, abs(cur$sigma - 2) < 1)

# A sampler given as a file whose folder and name contain spaces (Rcpp itself
# refuses a file name with a space on Windows).
src <- file.path(tempdir(), "my model", "my model.cpp")
dir.create(dirname(src), recursive = TRUE, showWarnings = FALSE)
file.copy(ai4bayescode_examples_path("GaussianLocationScale.cpp"), src)
ai4bayescode_source(src, rebuild = TRUE)
cat("PASS\n")
