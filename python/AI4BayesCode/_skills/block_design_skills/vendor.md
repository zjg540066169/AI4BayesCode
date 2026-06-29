# module: vendor.md — block_design VENDOR sub-skill — adapt an external kernel to AI4BayesCode's
# STATEFUL block contract (the stateful modifications), with minimal-diff staging + license preservation.
# loaded: LAZILY, only when the block VENDORS an external kernel (decided at INTAKE Step 5). Not a phase
# of its own — a sub-skill DESIGN pulls in while implementing, and VALIDATE checks. Cites intake.md Step 5
# (license / attribution), validate.md (the stateful-compatibility check), system_design §1 (the block
# contract); does NOT restate them.

# VENDOR — make the borrowed kernel behave inside the stateful-block contract

## 0. When this loads

Load this ONLY if INTAKE Step 5 decided the block VENDORS an external kernel / library (a k-d tree, an
ODE integrator, a special-function or sampler routine, …). If the block has NO vendored code, skip this
module entirely. The license-gate + attribution were settled at INTAKE Step 5; the GPL-3 agreement at
`00_flow.md §0.5`. **This module is the ENGINEERING:** making borrowed code correct inside AI4BayesCode's
STATEFUL block contract (`system_design §1`: `step(rng)` / `set_context` / `current` / `set_current`,
instance-owned state, deterministic given the seed).

## 1. The principle — vendor = ADAPT, with the SMALLEST diff

External code was NOT written for this contract, so a borrowed kernel usually needs MODIFICATION to fit
— but modify it as LITTLE as possible:

- **WRAP, don't rewrite.** Prefer owning the kernel as a member + a thin adapter over editing its
  source. The less you touch upstream, the cheaper a future version swap.
- **Every modification is documented + reproducible** (`vendor/<lib>/PATCHES.md`), so an upstream bump
  can re-apply them.
- **Upstream LICENSE header stays VERBATIM** (`intake.md` Step 5): never relicense the vendored source.
  YOUR adapter code is GPL-3.0-or-later; the vendored source keeps its own license. (`§5c` verifies both.)

## 2. State audit — find the state FIRST

Before adapting, find every piece of state the kernel holds — the same four the VALIDATE
stateful-compatibility check will test (`validate.md` §1 vendored-kernel check):

1. **Global / `static` mutable state** — non-const `static` locals, global / namespace-scope mutables,
   mutable singletons, thread-locals.
2. **An internal RNG** — `rand()` / `srand`, a `static` / global generator, a self-seeded engine.
3. **Data-derived caches** — anything precomputed from the inputs (a k-d tree built on locations, a
   factorization, a lookup table).
4. **Hidden handles** — file / socket / GPU handles, allocators, pools held across calls.

Fast first pass: `grep -nE "static |rand\(|srand|mt19937|::generator|singleton|thread_local"` the
headers you actually `#include`.

## 3. The STATEFUL modifications (the heart of vendoring)

For each kind of state found, apply the matching fix.

### 3.1 Global / `static` mutable → INSTANCE-OWNED
The contract is one-state-per-instance. Any global/static mutable is SHARED across block instances —
block A's `step` corrupts block B. Move it into a member the block owns (or a wrapper struct the block
holds). If the state is buried deep in the library and cannot be hoisted, you have two honest options:
(a) wrap the kernel so each block owns its OWN instance of the stateful object; (b) if even that is
impossible, the kernel is UNUSABLE as a reusable primitive — say so and STOP (do not ship a block that
silently only works as a singleton). Never paper over it.

### 3.2 Internal RNG → the BLOCK's `rng`
The block is deterministic given the seed: `step(std::mt19937_64& rng)`. A kernel with its own RNG breaks
reproducibility. Replace it:
- Kernel takes a generator / seed argument → pass the block's `rng` (or a kernel-RNG seeded
  deterministically FROM it each call).
- Kernel calls `rand()` / a global generator internally → you MUST modify that call site to take the
  block's `rng` (a documented `PATCHES.md` entry), or wrap it. **Never leave `rand()` in a vendored
  sampling path.**
- Confirmed by the VALIDATE same-seed-determinism test: same seed → same output.

### 3.3 Data-derived cache → REBUILD on `set_context` (and NEVER hold a raw pointer to shared_data)
If the kernel precomputes from the inputs (a nanoflann k-d tree on the locations; a Cholesky; a table),
build it in / after `set_context`, and REBUILD it whenever `set_context` swaps the data. Two hard rules:
- **Never retain a raw pointer / reference into `shared_data`** inside the kernel — copy what you need
  into block-owned storage in `set_context` (the block hard-rule against pointers into shared_data).
- **Invalidate on refresh.** The clean pattern: a dirty-flag set in `set_context`, a lazy rebuild on the
  next `step`. The VALIDATE `set_context(A)→step→set_context(B≠A)→step` test catches a stale cache.

### 3.4 Instance isolation
Two block instances stepped interleaved must NOT interfere; a fresh block must have fresh kernel state.
This falls out of 3.1–3.3 done right; the VALIDATE two-instance test confirms it.

## 4. Staging the vendored code

- `blocks_local/<Block>/vendor/<lib>/` — the (minimally-modified) source + its LICENSE (single-header:
  the in-file license header suffices, no separate LICENSE needed; multi-file: copy its LICENSE too).
  Block-vendored code lives INSIDE the block's self-contained bundle — NOT in the shared system include
  tree (that is for the FIXED system libs — mcmclib / Eigen / celerite / libgp — which every block just
  `#include`s and never re-vendors). **WHY in the block's own folder: so the whole `<Block>/` folder is
  ONE uploadable / downloadable unit** — the FUTURE registry ships exactly this folder, so a downloader
  gets the block AND its borrowed kernel together, nothing to fetch separately. Until the bundle
  auto-delivers (the move on all-pass), it sits in the STAGING copy
  (`.block_design_staging/<Block>/vendor/<lib>/`), per the TOP RULE.
- **How the block finds it (include path):** the block `.hpp` includes the vendored header by its
  bundle-relative path — `#include "vendor/<lib>/<header>.hpp"`. No extra `-I` is needed: the compile
  already puts the bundle root (`blocks_local/<Block>/`, or the staging copy during VALIDATE) on `-I`,
  so the relative `vendor/...` path resolves.
- `vendor/<lib>/PATCHES.md` — EVERY modification you made (what + why), so an upstream swap is
  reproducible. A pure WRAP with no upstream edits → note "no upstream source modified; adapted via
  wrapper `<file>`".
- Manifest `Vendored:` records the lib + its upstream license (provenance).

## 5. What VALIDATE then checks (your §3 work is what makes it pass)

For a vendored kernel, VALIDATE runs BOTH:
- the **vendored-correctness regime** — the routine standalone vs KNOWN values (`validate.md` §1), AND
- the **vendored-kernel STATEFUL-compatibility check** (`validate.md` §1 / §2) — no hidden global state,
  determinism under the block's `rng`, cache rebuild on `set_context`, instance isolation.

Your §3 modifications are exactly what make that check pass. The `§5c` final license check then confirms
your adapter is `GPL-3.0-or-later` and the vendored source kept its upstream license UNCHANGED.
