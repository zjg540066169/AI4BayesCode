#!/usr/bin/env bash
# Run BART-family diagnostic tests and collect pass/fail summary.
# Must be executed from AI4BayesCode/examples/.
#
# The genBART-family diagnostic scripts live under ../tests_autodiff/gbart_*.R.

set -u

TESTS=(
    "test_bart_warmstart.R"
    "test_bart_multichain.R"
    "test_bart_holdout.R"
    "test_bart_reference.R"
)

declare -a STATUS
declare -a RUNTIME

t_start=$(date +%s)

for t in "${TESTS[@]}"; do
    echo "=============================================================="
    echo "=== Running $t"
    echo "=============================================================="
    ts=$(date +%s)
    if Rscript "$t" 2>&1 | tail -5; then
        s="PASS"
    else
        s="FAIL"
    fi
    te=$(date +%s)
    STATUS+=("$s")
    RUNTIME+=("$((te - ts))")
    echo "--- $t : $s ($((te - ts))s)"
    echo
done

t_end=$(date +%s)
echo
echo "=============================================================="
echo "SUMMARY"
echo "=============================================================="
for i in "${!TESTS[@]}"; do
    printf "  %-30s  %s  (%ds)\n" "${TESTS[$i]}" "${STATUS[$i]}" "${RUNTIME[$i]}"
done
echo "  ---"
echo "  total time: $((t_end - t_start))s"

all_pass=0
for s in "${STATUS[@]}"; do
    if [ "$s" != "PASS" ]; then all_pass=1; fi
done
if [ $all_pass -eq 0 ]; then
    echo "  ALL PASS"
else
    echo "  SOME FAILURES"
fi
exit $all_pass
