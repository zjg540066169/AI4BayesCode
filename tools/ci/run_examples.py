"""Run the runnable ``// @example:python`` block of every shipped example.

Each example runs in a fresh Python process with its own working directory,
and the script reports which ones fail. The Windows CI job uses this to check
that generated-style samplers compile and run end to end.

Usage: python tools/ci/run_examples.py [--part P] [--n-parts N]
                                       [--log-dir DIR] [--timeout S] [--limit K]
"""
from __future__ import annotations

import argparse
import csv
import platform
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import AI4BayesCode
from AI4BayesCode.doc import _parse_example


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, default=1)
    ap.add_argument("--n-parts", type=int, default=1)
    ap.add_argument("--log-dir", default="example-logs")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    log_dir = Path(a.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    names = AI4BayesCode.list_examples()
    mine = [n for i, n in enumerate(names) if i % a.n_parts == a.part - 1]
    if a.limit:
        mine = mine[: a.limit]
    print(f"Python {platform.python_version()} on {platform.platform()} | "
          f"part {a.part} of {a.n_parts} | {len(mine)} examples", flush=True)

    rows = []
    for name in mine:
        src = Path(AI4BayesCode.examples_path(name)).read_text(encoding="utf-8")
        code = _parse_example(src, "python")
        if not code:
            rows.append((name, "NO-EXAMPLE", 0.0))
            continue
        wd = log_dir / f"{name}_wd"
        wd.mkdir(exist_ok=True)
        script = log_dir / f"{name}.py"
        # A non-interactive backend, so diagnose() writes plots and never blocks.
        script.write_text("import matplotlib\nmatplotlib.use('Agg')\n"
                          + textwrap.dedent("\n".join(code)) + "\n",
                          encoding="utf-8")
        t0 = time.time()
        with open(log_dir / f"{name}.out.txt", "w", encoding="utf-8") as out:
            try:
                rc = subprocess.run([sys.executable, str(script)], cwd=wd,
                                    stdout=out, stderr=subprocess.STDOUT,
                                    timeout=a.timeout).returncode
                status = "PASS" if rc == 0 else f"FAIL({rc})"
            except subprocess.TimeoutExpired:
                status = "TIMEOUT"
        secs = round(time.time() - t0, 1)
        rows.append((name, status, secs))
        print(f"{name:<34} {status:<10} {secs:7.1f}s", flush=True)
        if status != "PASS":
            tail = (log_dir / f"{name}.out.txt").read_text(
                encoding="utf-8", errors="replace").splitlines()[-25:]
            print("\n".join("    | " + ln for ln in tail), flush=True)

    with open(log_dir / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["example", "status", "seconds"])
        w.writerows(rows)
    bad = sum(1 for _, s, _ in rows if s != "PASS")
    print(f"\n{len(rows) - bad} of {len(rows)} examples passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
