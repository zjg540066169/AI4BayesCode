#!/usr/bin/env python3
"""
Migrate examples/*.cpp to inherit AI4BayesCode::kernel_control_mixin<Derived>
and bind the freeze/unfreeze/get_frozen category via macros.

DRY-RUN by default. Pass --apply to write.

Skips ARDLasso.cpp (hand-Gibbs, no composite_block impl_; manual migration).
"""
import argparse
import re
import sys
from pathlib import Path

EX = Path(__file__).resolve().parents[1] / "examples"
SKIP = {"ARDLasso.cpp"}

INCLUDE_LINE = '#include "AI4BayesCode/kernel_control_mixin.hpp"\n'
INC_RE = re.compile(r'^#include "AI4BayesCode/[^"]+\.hpp"\s*$')


def migrate(path: Path):
    name = path.stem
    src = path.read_text()
    if "kernel_control_mixin" in src:
        return ("SKIP", "already-migrated", src, src)

    lines = src.splitlines(keepends=True)

    # Locate `class <Name> {` (top-level class declaration, no existing base).
    class_re = re.compile(rf'^class {re.escape(name)}\s*\{{\s*$')
    class_idx = next((i for i, L in enumerate(lines) if class_re.match(L)), None)
    if class_idx is None:
        return ("FAIL", f"no `class {name} {{` line found", src, src)

    # A. Insert include line after the LAST AI4BayesCode/*.hpp include that
    # appears BEFORE the class declaration.
    last_inc = None
    for i in range(class_idx):
        if INC_RE.match(lines[i]):
            last_inc = i
    if last_inc is None:
        return ("FAIL", "no AI4BayesCode include before class", src, src)
    lines.insert(last_inc + 1, INCLUDE_LINE)
    class_idx += 1

    # B. Add CRTP base to class declaration.
    lines[class_idx] = (
        f"class {name} : public AI4BayesCode::kernel_control_mixin<{name}> "
        "{\n"
    )

    # C. RCPP_MODULE macro insertion.
    rcpp_open = next(
        (i for i, L in enumerate(lines) if L.startswith("RCPP_MODULE(")),
        None)
    if rcpp_open is None:
        return ("FAIL", "no top-level `RCPP_MODULE(`", src, src)
    rcpp_close = next(
        (i for i in range(rcpp_open + 1, len(lines))
         if lines[i].rstrip() == "}"),
        None)
    if rcpp_close is None:
        return ("FAIL", "no `}` after RCPP_MODULE", src, src)
    term_re = re.compile(r'\)\s*;\s*(//.*)?$')
    term_idx = None
    for i in range(rcpp_close - 1, rcpp_open, -1):
        if term_re.search(lines[i].rstrip()):
            term_idx = i
            break
    if term_idx is None:
        return ("FAIL", "no chain-terminator `);` in RCPP block", src, src)
    lines[term_idx] = re.sub(r'\)\s*;', r')', lines[term_idx], count=1)
    lines.insert(term_idx + 1,
                 f"        AI4BAYESCODE_BIND_KERNEL_CONTROL({name});\n")

    # D. PYBIND11_MODULE macro insertion (chain-embedded via .def).
    pyb_open = next(
        (i for i, L in enumerate(lines) if L.startswith("PYBIND11_MODULE(")),
        None)
    if pyb_open is None:
        return ("FAIL", "no top-level `PYBIND11_MODULE(`", src, src)
    pyb_close = next(
        (i for i in range(pyb_open + 1, len(lines))
         if lines[i].rstrip() == "}"),
        None)
    if pyb_close is None:
        return ("FAIL", "no `}` after PYBIND11_MODULE", src, src)
    term_idx = None
    for i in range(pyb_close - 1, pyb_open, -1):
        if term_re.search(lines[i].rstrip()):
            term_idx = i
            break
    if term_idx is None:
        return ("FAIL", "no chain-terminator `);` in PYBIND block", src, src)
    lines[term_idx] = re.sub(r'\)\s*;', r')', lines[term_idx], count=1)
    lines.insert(term_idx + 1,
                 f"        AI4BAYESCODE_PYBIND_KERNEL_CONTROL({name});\n")

    return ("OK", "migrated", src, "".join(lines))


def verify(name: str, text: str) -> list[str]:
    checks = [
        (INCLUDE_LINE.strip(), 1),
        (f"class {name} : public AI4BayesCode::kernel_control_mixin<{name}>", 1),
        (f"AI4BAYESCODE_BIND_KERNEL_CONTROL({name})", 1),
        (f"AI4BAYESCODE_PYBIND_KERNEL_CONTROL({name})", 1),
    ]
    problems = []
    for needle, want in checks:
        got = text.count(needle)
        if got != want:
            problems.append(f"  {needle!r}: got {got}, want {want}")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="write files (default: dry-run)")
    args = ap.parse_args()

    ok = fail = skipped = 0
    for f in sorted(EX.glob("*.cpp")):
        if f.name in SKIP:
            print(f"[SKIP-MANUAL] {f.name}")
            skipped += 1
            continue
        status, msg, _src, new = migrate(f)
        if status == "OK":
            problems = verify(f.stem, new)
            if problems:
                status = "FAIL"
                msg = "verify-fail"
        tag = status
        print(f"[{tag}] {f.name}: {msg}")
        if status == "OK":
            ok += 1
            if args.apply:
                f.write_text(new)
        elif status == "SKIP":
            skipped += 1
        else:
            fail += 1
            for p in (verify(f.stem, new) if 'new' in locals() else []):
                print(p)

    print(f"\n=== summary: {ok} ok, {skipped} skipped, {fail} failed ===")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
