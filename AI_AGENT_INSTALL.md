# AI Agent Installation Guide — AI4BayesCode

This file is for an AI coding agent installing this repository as an agent skill. Keep
installation safe and non-destructive. After install, the user activates the skill by typing
`/AI4BayesCode` in Claude Code, or `$AI4BayesCode` in Codex, which loads `SKILL.md` and
bootstraps `start.md`.

Every command below runs unchanged in bash, zsh, Git Bash, Windows PowerShell 5.1 and
PowerShell 7. None of them works in the Windows Command Prompt (cmd.exe), which does not
expand `$HOME`. Keep the double quotes, and write `$HOME`, never `~`: PowerShell before 7.6
hands `~` to git unexpanded, and git then silently creates a folder literally named `~`.

## 1. Prerequisites

The skill itself is markdown + headers and needs only `git` to install:

```bash
git --version
```

If git is missing, ask the user to install it. Windows: Git for Windows, for example
`winget install --id Git.Git -e --source winget`, then open a new terminal. macOS:
`xcode-select --install`. Linux: the distribution's package manager.

The GENERATED samplers are compiled through the R or the Python package. One of them is
needed before the first sampler can be validated, not for the install:

- **R backend:** R with `Rcpp` and `RcppArmadillo`. On Windows also Rtools for the installed
  R version (https://cran.r-project.org/bin/windows/Rtools/). The Windows R installer does not
  put `Rscript.exe` on PATH. If `Rscript --version` fails there, use
  `C:\Program Files\R\R-<version>\bin\Rscript.exe` (the newest installed version).
- **Python backend:** Python 3.11 or newer with the AI4BayesCode package, a C++17 compiler and
  the Armadillo headers. macOS: `brew install armadillo`. Debian/Ubuntu:
  `apt-get install libarmadillo-dev`. Windows: install MSYS2 (https://www.msys2.org, default
  folder `C:\msys64`), then in the "MSYS2 UCRT64" terminal run
  `pacman -S --needed mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-armadillo mingw-w64-ucrt-x86_64-openblas`.

## 2. Choose install target

If the user has not specified, ask:

```markdown
Install target:
- A. Claude Code
- B. Codex
- C. Cursor project rule
- D. All available
```

## 3. Install

The skill is the whole repository, cloned straight into the agent's skills folder.
`git clone` creates the missing parent folders, so no `mkdir` is needed.

Claude Code:

```bash
git clone --depth 1 https://github.com/zjg540066169/AI4BayesCode.git "$HOME/.claude/skills/AI4BayesCode"
```

Codex:

```bash
git clone --depth 1 https://github.com/zjg540066169/AI4BayesCode.git "$HOME/.agents/skills/AI4BayesCode"
```

Before cloning, check the target folder:

- If `CLAUDE_CONFIG_DIR` is set, Claude Code reads skills from `$CLAUDE_CONFIG_DIR/skills`.
  Clone there instead.
- If the target folder already exists, find out whether it is a clone of its own:
  `git -C "<target>" rev-parse --show-toplevel` must print the target folder itself. If it
  does, update it (section 4) and do not clone. If it does not, it is an older copy made
  without git. Ask the user before removing it, remove it, then clone. Removing it in bash:
  `rm -rf "<target>"`. In PowerShell: `Remove-Item -Recurse -Force "<target>"`.
- Codex: earlier instructions installed into `$HOME/.codex/skills/AI4BayesCode`. Codex still
  reads that folder, so a copy there makes the skill appear twice. Ask the user, then remove
  it the same way.
- If Claude Code or Codex runs inside WSL, run the commands inside WSL. A clone made from
  Windows PowerShell lands in the Windows profile, which an agent inside WSL never reads.

Afterwards tell the user to restart Claude Code (then type `/AI4BayesCode`) or Codex (then type
`$AI4BayesCode` or pick it from `/skills`).

## 4. Update

```bash
git -C "$HOME/.claude/skills/AI4BayesCode" pull --ff-only
```

For Codex, use `$HOME/.agents/skills/AI4BayesCode`. Run it only after the check in section 3
shows that the folder is a clone of its own. Otherwise git searches the parent folders for a
repository and would pull into that one instead, for example a dotfiles repository that
tracks the home folder. If the pull stops with "Not possible to fast-forward", the upstream
history was rewritten. Ask the user, remove the folder and clone again.

## 5. Install for Cursor project rule

Inside the user's target project (not this package), create the file
`.cursor/rules/ai4bayescode.mdc` with your file-writing tool, with this content:

```markdown
---
description: Use the installed AI4BayesCode skill to generate/design Bayesian MCMC samplers.
alwaysApply: false
---

When the user asks to generate an MCMC sampler for a Bayesian model, or to design a new sampler
block, use the installed AI4BayesCode skill. Find its entry at one of:
- a local cloned package containing `SKILL.md`
- `$HOME/.claude/skills/AI4BayesCode/SKILL.md`
- `$HOME/.agents/skills/AI4BayesCode/SKILL.md`

Load `SKILL.md` first, then `start.md`, then only the phase sub-skills needed. Generated user code
goes in the user's own project, never inside the installed skill.
```

## 6. Final install report

Report:

```markdown
- Package path:
- Installed target(s):
- Skill entry path(s):     (e.g. $HOME/.claude/skills/AI4BayesCode/SKILL.md)
- Library path for codegen: (the installed skill root; its include/ holds the built-in blocks)
- R / Python backends available (and, on Windows, Rtools or MSYS2 UCRT64 present):
- Restart needed:          (yes, then /AI4BayesCode in Claude Code or $AI4BayesCode in Codex)
```
