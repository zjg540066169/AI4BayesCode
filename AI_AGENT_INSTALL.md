# AI Agent Installation Guide — AI4BayesCode

This file is for an AI coding agent installing this repository as an agent skill. Keep
installation safe and non-destructive. After install, the user activates the skill by typing
`/AI4BayesCode`, which loads `SKILL.md` and bootstraps `start.md`.

## 1. Verify package root

Run from the package root. This directory MUST contain `SKILL.md` directly, plus the router and
the bundled library:

```bash
test -f SKILL.md
test -f start.md
test -d skills
test -d include            # the C++ library + built-in block headers
```

If `SKILL.md` is missing, stop and ask the user for the correct package directory.

## 2. Dependencies

The skill itself is markdown + headers; it needs no install. The GENERATED samplers compile with:

```bash
c++ --version || clang++ --version || g++ --version   # a C++17 compiler is REQUIRED to compile generated code
Rscript --version || true                              # optional — only for the R (Rcpp) runner
python3 --version || true                              # optional — only for the Python (pybind11) runner
```

Notes:
- A C++17 compiler (clang++/g++) is required to compile generated samplers. On macOS, also
  `-framework Accelerate`; on Linux, `-lblas -llapack`.
- R + `RcppArmadillo` are needed ONLY if the user picks the R backend; Python ≥ 3.11 + `pybind11`
  ONLY for the Python backend. Neither is needed to install or to use the C++ backend.

## 3. Choose install target

If the user has not specified, ask:

```markdown
Install target:
- A. Claude Code
- B. Codex
- C. Cursor project rule
- D. All available
```

## 4. Install for Claude Code

```bash
CLAUDE_SKILLS_DIR="$HOME/.claude/skills"
TARGET="$CLAUDE_SKILLS_DIR/AI4BayesCode"
mkdir -p "$CLAUDE_SKILLS_DIR"

# Back up any existing install OUTSIDE the skills dir, so the backup is not
# itself rescanned as a bogus "AI4BayesCode.backup.*" skill.
if [ -e "$TARGET" ]; then
  BACKUP_DIR="$HOME/.claude/skill-backups"; mkdir -p "$BACKUP_DIR"
  mv "$TARGET" "$BACKUP_DIR/AI4BayesCode.backup.$(date +%Y%m%d-%H%M%S)"
fi

mkdir -p "$TARGET"
if command -v rsync >/dev/null 2>&1; then
  rsync -a --exclude '.git/' --exclude '.block_design_staging/' --exclude 'generated/' \
        --exclude '*.bak*' --exclude '__pycache__/' --exclude '.DS_Store' ./ "$TARGET"/
else
  cp -R . "$TARGET"/
fi

test -f "$TARGET/SKILL.md" && test -d "$TARGET/include"
```

Tell the user to restart or reload Claude Code, then type `/AI4BayesCode`.

## 5. Install for Codex

Same as §4 but with the Codex skills dir:

```bash
CODEX_SKILLS_DIR="${CODEX_HOME:-$HOME/.codex}/skills"
TARGET="$CODEX_SKILLS_DIR/AI4BayesCode"
mkdir -p "$CODEX_SKILLS_DIR"
# Back up outside the skills dir (see §4) so the backup is not rescanned as a skill.
if [ -e "$TARGET" ]; then
  BACKUP_DIR="${CODEX_HOME:-$HOME/.codex}/skill-backups"; mkdir -p "$BACKUP_DIR"
  mv "$TARGET" "$BACKUP_DIR/AI4BayesCode.backup.$(date +%Y%m%d-%H%M%S)"
fi
mkdir -p "$TARGET"
if command -v rsync >/dev/null 2>&1; then
  rsync -a --exclude '.git/' --exclude '.block_design_staging/' --exclude 'generated/' \
        --exclude '*.bak*' --exclude '__pycache__/' --exclude '.DS_Store' ./ "$TARGET"/
else
  cp -R . "$TARGET"/
fi
test -f "$TARGET/SKILL.md" && test -d "$TARGET/include"
```

Tell the user to restart Codex after installation.

## 6. Install for Cursor project rule

Run inside the user's target project, not this package:

```bash
mkdir -p .cursor/rules
cat > .cursor/rules/ai4bayescode.mdc <<'RULE'
---
description: Use the installed AI4BayesCode skill to generate/design Bayesian MCMC samplers.
alwaysApply: false
---

When the user asks to generate an MCMC sampler for a Bayesian model, or to design a new sampler
block, use the installed AI4BayesCode skill. Find its entry at one of:
- a local cloned package containing `SKILL.md`
- `~/.claude/skills/AI4BayesCode/SKILL.md`
- `~/.codex/skills/AI4BayesCode/SKILL.md`

Load `SKILL.md` first, then `start.md`, then only the phase sub-skills needed. Generated user code
goes in the user's own project, never inside the installed skill.
RULE
```

## 7. Final install report

Report:

```markdown
- Package path:
- Installed target(s):
- Skill entry path(s):     (e.g. ~/.claude/skills/AI4BayesCode/SKILL.md)
- Library path for codegen: (the installed skill root; its include/ holds the built-in blocks)
- C++ compiler detected:
- R / Python backends available:
- Restart/reload needed:    (yes — then `/AI4BayesCode`)
```
