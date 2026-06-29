# CRAN submission comments

## v1.0.0 — first submission

This is the first CRAN submission for AI4BayesCode.

## R CMD check --as-cran results

0 errors, 0 warnings, 3 notes:

### Note 1: New submission

```
* checking CRAN incoming feasibility ... NOTE
Maintainer: 'Jungang Zou <jungang.zou@gmail.com>'
New submission
```

Standard for first-time submission.

### Note 2: Pragmas suppressing diagnostics in vendored Eigen

```
* checking pragmas in C/C++ headers and code ... NOTE
File which contains pragma(s) suppressing diagnostics:
  'inst/include/eigen/Eigen/src/Core/util/DisableStupidWarnings.h'
```

This is from the vendored Eigen 3.4 library (MPL-2.0) bundled under
`inst/include/eigen/`. The pragmas are upstream Eigen code, unmodified
by us — Eigen suppresses some compiler warnings on certain code paths
that are intentional. Other CRAN packages that vendor Eigen (e.g.
RcppEigen) carry the same NOTE.

### Note 3: HTML Tidy not recent enough on submitter's machine

```
* checking HTML version of manual ... NOTE
Skipping checking HTML validation: 'tidy' doesn't look like recent
enough HTML Tidy.
```

Local tool issue on submitter's macOS; CRAN's build farm has a current
tidy. No package content issue.

## Package size

Installed size ~10.7 MB (just over the 10 MB soft threshold). The bulk
(~9.2 MB) is the vendored header-only dependencies:

- Eigen 3.4 (MPL-2.0) — 6.6 MB
- mcmclib (patched, Apache-2.0) — 1.2 MB
- bart kernels (GPL-2.0-or-later) — 0.6 MB
- BaseMatrixOps (Apache-2.0) — 0.5 MB
- autodiff (MIT, gen-time use only) — 0.3 MB
- libgp kernels (BSD-3) — 0.15 MB
- celerite (MIT) — 0.07 MB

A future minor release will trim the Eigen subset to only the parts our
headers transitively need (estimated 6.6 MB → ~1 MB), bringing total
installed size to ~5 MB.

## Tested on

- R 4.5.x on macOS 14 (arm64) — local dev machine
- Win-builder and r-hub matrix runs in progress; will update if issues found.

## Reverse dependencies

None — first submission.

## Vendored / GPL combination notes

The combined work is licensed GPL-3.0-or-later (declared in DESCRIPTION).
The vendored Apache-2.0 mcmclib / BaseMatrixOps is one-way compatible
with GPL-3 but not with GPL-2-only — hence the GPL-3+ choice. The
vendored BART kernel is GPL-2.0-or-later, which the "or later" clause
resolves to GPL-3 in the combined work. Full per-component analysis is
shipped in the source tree's `THIRD_PARTY_LICENSES.md`.
