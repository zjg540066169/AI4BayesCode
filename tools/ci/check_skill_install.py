"""Check that a skill install landed where Claude Code and Codex look.

Run on CI after the install commands from the website or after
ai4bayescode_install_skill() / AI4BayesCode.install_skill(). For each agent
folder it checks that SKILL.md exists, that git checked it out with LF line
endings, and that the frontmatter parses as strict YAML with the expected
name and description.

Usage: python tools/ci/check_skill_install.py [--git-head SHA] [--report-only DIR ...]
With --git-head, each agent folder must also be a clean git checkout of SHA, as
the install command on the website leaves it. Folders given with --report-only
are checked and printed but never fail the run.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml


def frontmatter(text: str) -> str:
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        raise ValueError("SKILL.md does not start with a --- frontmatter line")
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return "\n".join(lines[1:i])
    raise ValueError("frontmatter has no closing --- line")


def check(folder: Path) -> list[str]:
    skill = folder / "SKILL.md"
    if not skill.is_file():
        return [f"{skill} is missing"]
    problems = []
    raw = skill.read_bytes()
    if b"\r\n" in raw:
        problems.append("SKILL.md has CRLF line endings")
    if (folder / ".git").exists():
        eol = subprocess.run(["git", "-C", str(folder), "ls-files", "--eol", "SKILL.md"],
                             capture_output=True, text=True).stdout.strip()
        print(f"    git ls-files --eol: {eol}")
    try:
        meta = yaml.safe_load(frontmatter(raw.decode("utf-8")))
    except Exception as exc:  # noqa: BLE001
        problems.append("frontmatter is not valid YAML: " + str(exc).splitlines()[0])
    else:
        if meta.get("name") != "AI4BayesCode":
            problems.append(f"unexpected name {meta.get('name')!r}")
        if not str(meta.get("description", "")).startswith("Build Bayesian MCMC samplers"):
            problems.append("description did not parse as one string")
    stray = [p for p in (Path.cwd() / "$HOME", Path.cwd() / "~") if p.exists()]
    for p in stray:
        problems.append(f"a literal folder {p} was created (the shell did not expand the path)")
    return problems


def check_git(folder: Path, head: str) -> list[str]:
    if not (folder / ".git").exists():
        return ["not a git checkout"]
    git = ["git", "-C", str(folder)]
    got = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    problems = [] if got == head else [f"HEAD is {got[:12]}, expected {head[:12]}"]
    dirty = subprocess.run(git + ["status", "--porcelain"], capture_output=True, text=True).stdout
    if dirty.strip():
        problems.append("the working tree is not clean:\n" + dirty.rstrip())
    return problems


def main(argv: list[str]) -> int:
    head = None
    if "--git-head" in argv:
        i = argv.index("--git-head")
        head = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    report_only = []
    if "--report-only" in argv:
        report_only = [Path(p) for p in argv[argv.index("--report-only") + 1:]]
    # The profile folder on Windows (%USERPROFILE%), $HOME elsewhere: where
    # Claude Code and Codex look.
    home = Path.home()
    targets = {
        "Claude Code": home / ".claude" / "skills" / "AI4BayesCode",
        "Codex": home / ".agents" / "skills" / "AI4BayesCode",
    }
    failed = False
    for label, folder in targets.items():
        print(f"{label}: {folder}")
        problems = check(folder) + (check_git(folder, head) if head else [])
        for p in problems:
            print(f"    FAIL {p}")
        print("    OK" if not problems else "")
        failed |= bool(problems)
    for folder in report_only:
        print(f"report only: {folder}")
        for p in check(folder) or ["no problems"]:
            print(f"    {p}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
