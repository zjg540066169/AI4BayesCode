#!/usr/bin/env bash
# Build every shipped example as a plain C++ binary and run it. Without the R
# or Python module macros each example compiles its own main(), which
# simulates data from known parameters and checks that the sampler recovers
# them. The Windows CI job runs this under MSYS2 to test the library with the
# MinGW-w64 toolchain, independently of the R and Python packages.
#
# Usage: bash tools/ci/run_examples_standalone.sh [log_dir]
# Environment: CXX (default g++), LIBS (default -lopenblas, -lopenblas_64 on
# Windows), EXTRA_CXXFLAGS,
# TIMEOUT (seconds)
set -u
out=${1:-example-logs}
mkdir -p "$out"
inc=r-pkg/inst/include
flags=(-std=c++17 -O2
       -I"$inc" -I"$inc/mcmclib" -I"$inc/mcmclib/BaseMatrixOps/include"
       -I"$inc/eigen" -I"$inc/celerite/include" -I"$inc/libgp_kernels"
       -DMCMC_ENABLE_ARMA_WRAPPERS -DARMA_DONT_USE_WRAPPER=)
# Windows (MSYS2): the 64-bit-integer OpenBLAS that MSYS2's Armadillo expects,
# a bigger object-file section table, and no SuperLU or ARPACK.
case "$(uname -s)" in
  MINGW*|MSYS*|UCRT*)
    flags+=(-Wa,-mbig-obj -DARMA_DONT_USE_SUPERLU -DARMA_DONT_USE_ARPACK)
    default_libs=-lopenblas_64 ;;
  *) default_libs=-lopenblas ;;
esac
read -r -a libs <<< "${LIBS:-$default_libs}"
read -r -a extra <<< "${EXTRA_CXXFLAGS:-}"
cxx=${CXX:-g++}
"$cxx" --version | head -1

n=0; bad=0
for f in examples/*.cpp; do
  name=$(basename "$f" .cpp)
  n=$((n + 1))
  start=$(date +%s)
  if ! "$cxx" "${flags[@]}" ${extra[@]+"${extra[@]}"} "$f" -o "$out/$name.exe" "${libs[@]}" \
       > "$out/$name.build.txt" 2>&1; then
    status="BUILD-FAIL"
  elif timeout "${TIMEOUT:-1200}" "$out/$name.exe" > "$out/$name.run.txt" 2>&1; then
    status="PASS"
  else
    status="RUN-FAIL($?)"
  fi
  printf '%-34s %-14s %5ss\n' "$name" "$status" "$(( $(date +%s) - start ))"
  if [ "$status" != "PASS" ]; then
    bad=$((bad + 1))
    log="$out/$name.build.txt"; [ "$status" = "BUILD-FAIL" ] || log="$out/$name.run.txt"
    grep -m 15 -iE 'error|fail|undefined' "$log" | sed 's/^/    | /'
    tail -n 5 "$log" | sed 's/^/    | /'
  fi
  rm -f "$out/$name.exe"
done
echo
echo "$((n - bad)) of $n examples passed"
[ "$bad" -eq 0 ]
