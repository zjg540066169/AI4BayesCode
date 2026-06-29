#!/bin/bash
# Drives extended audit phases F1-F5 in order.
# Each phase writes its own log + rds; failure does not abort remaining phases.

set -u
cd "$(dirname "$0")"
DIR="$(pwd)"
LOG="${DIR}/audit_xl_ALL.log"

echo "=== EXTENDED AUDIT START: $(date -Iseconds) ===" > "$LOG"

run_phase() {
    local name="$1"; shift
    local script="$1"; shift
    local phase_log="${DIR}/${name}.log"
    echo ""                                        >> "$LOG"
    echo "### $(date -Iseconds)  $name — $script" >> "$LOG"
    Rscript "$script" > "$phase_log" 2>&1
    local rc=$?
    echo "   exit_code=$rc  (log: $phase_log)"    >> "$LOG"
    tail -60 "$phase_log"                         >> "$LOG"
    return $rc
}

# F5 is cheapest — run first so shape errors surface early
run_phase audit_compile_smoke audit_compile_smoke.R  || echo "F4 FAILED" >> "$LOG"
run_phase audit_predict_at    audit_predict_at.R    || echo "F5 FAILED" >> "$LOG"
run_phase audit_coverage      audit_coverage.R      || echo "F2 FAILED" >> "$LOG"
run_phase audit_joint_dim     audit_joint_dim.R     || echo "F3 FAILED" >> "$LOG"
run_phase audit_xl            audit_xl.R            || echo "F1 FAILED" >> "$LOG"

echo "" >> "$LOG"
echo "=== EXTENDED AUDIT END: $(date -Iseconds) ===" >> "$LOG"
