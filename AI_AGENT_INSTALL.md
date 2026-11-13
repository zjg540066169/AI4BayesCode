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
`xcode-select --install`. Linux: the distribution's package manager. Alternatively, if the
AI4BayesCode R or Python package is installed, it can install the skill without git:
`Rscript -e "AI4BayesCode::ai4bayescode_install_skill()"` or
`python -c "import AI4BayesCode; AI4BayesCode.install_skill()"`. Both work on any OS and
install for both agents. Pass `"claude"` or `"codex"` to install for one of them only.

The GENERATED samplers are compiled through the R or the Python package. One of them is
needed before the first sampler can be validated, not for the install:

- **R backend:** R with `Rcpp` and `RcppArmadillo`. On Windows also Rtools for the installed
  R version (https://cran.r-project.org/bin/windows/Rtools/). The Windows R installer does not
  put `Rscript.exe` on PATH. If `Rscript --version` fails there, use the newest
  `C:\Program Files\R\R-<version>\bin\Rscript.exe` (a per-user install is under
  `%LOCALAPPDATA%\Programs\R\`). In PowerShell, run a quoted path with `&`, for example
  `& "C:\Program Files\R\R-<version>\bin\Rscript.exe" --version`.
- **Python backend:** Python 3.11 or newer with the AI4BayesCode package, a C++17 compiler and
  the Armadillo headers. macOS: `brew install armadillo`. Debian/Ubuntu:
  `apt-get install libarmadillo-dev`. Windows: Rtools44 or newer (the Rtools of R 4.4 or later)
  already holds everything. Otherwise install MSYS2 (https://www.msys2.org, default folder `C:\msys64`),
  then in the "MSYS2 UCRT64" terminal run
  `pacman -S --needed mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-armadillo mingw-w64-ucrt-x86_64-openblas64`.

## 2. Choose install target

If the user has not specified, ask:

```markdown
Install target:
- A. Claude Code
- B. Codex
- C. Cursor project rule
- D. All available
```

## 3. Install or update

The skill is the whole repository, fetched straight into the agent's skills folder. One
command does both jobs: the first run installs, and running it again updates to the
newest version. `git init` creates the missing folders and is safe to repeat.

Claude Code:

```bash
git init -q "$HOME/.claude/skills/AI4BayesCode"; git -C "$HOME/.claude/skills/AI4BayesCode" pull -q --ff-only https://github.com/zjg540066169/AI4BayesCode.git main
```

Codex:

```bash
git init -q "$HOME/.agents/skills/AI4BayesCode"; git -C "$HOME/.agents/skills/AI4BayesCode" pull -q --ff-only https://github.com/zjg540066169/AI4BayesCode.git main
```

- If `CLAUDE_CONFIG_DIR` is set, Claude Code reads skills from `$CLAUDE_CONFIG_DIR/skills`.
  Use that folder instead. In PowerShell write it `$env:CLAUDE_CONFIG_DIR`: a bare
  `$CLAUDE_CONFIG_DIR` there is an unset PowerShell variable.
- If the pull stops with "untracked working tree files would be overwritten", the folder
  holds an older copy made without git. If it stops with "Not possible to fast-forward",
  the upstream history was rewritten. In both cases ask the user, remove the folder and run
  the command again. Removing it in bash: `rm -rf "<target>"`. In PowerShell:
  `Remove-Item -Recurse -Force "<target>"`.
- Codex: earlier instructions installed into `$CODEX_HOME/skills/AI4BayesCode` (`CODEX_HOME`
  defaults to `$HOME/.codex`). Codex still reads that folder, so a copy there makes the skill
  appear twice. Ask the user, then remove it the same way.
- If Claude Code or Codex runs inside WSL, run the command inside WSL. A copy made from
  Windows PowerShell lands in the Windows profile, which an agent inside WSL never reads.
- In Git Bash, `$HOME` can differ from the profile folder that Claude Code and Codex use: it
  follows a `HOME` variable the user set, and without one Git for Windows takes
  `%HOMEDRIVE%%HOMEPATH%` (a network drive on some work and school PCs) when that folder
  exists. If `cygpath -u "$USERPROFILE"` prints a different folder than `echo "$HOME"`, write
  `$USERPROFILE` in place of `$HOME` in the command. PowerShell's `$HOME` is always the
  profile folder.

Afterwards tell the user to restart Claude Code (then type `/AI4BayesCode`) or Codex (then type
`$AI4BayesCode` or pick it from `/skills`).

## 4. Install for Cursor project rule

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

## 5. Final install report

Report:

```markdown
- Package path:
- Installed target(s):
- Skill entry path(s):     (e.g. $HOME/.claude/skills/AI4BayesCode/SKILL.md)
- Library path for codegen: (the installed skill root; its include/ holds the built-in blocks)
- R / Python backends available (and, on Windows, Rtools or MSYS2 UCRT64 present):
- Restart needed:          (yes, then /AI4BayesCode in Claude Code or $AI4BayesCode in Codex)
```
