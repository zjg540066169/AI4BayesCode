"""Install the AI4BayesCode agent skill for Claude Code or Codex.

Mirror of the R ``ai4bayescode_install_skill()``. The skill is the whole
GitHub repository; it is downloaded as a zip archive, so neither git nor any
particular shell is needed, and the call is the same on Windows, macOS and
Linux::

    import AI4BayesCode
    AI4BayesCode.install_skill()            # Claude Code and Codex
    AI4BayesCode.install_skill("claude")    # Claude Code only
"""
from __future__ import annotations

import os
import shutil
import stat
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from ._console import safe_print as print  # noqa: A001 (see _console.py)

__all__ = ["install_skill"]

_SKILL_REPO = "zjg540066169/AI4BayesCode"
_AGENTS = ("claude", "codex")


def _skill_dest(agent: str) -> Path:
    """Where each agent reads personal skills from. Path.home() is the
    profile folder on Windows (%USERPROFILE%), which is where both look."""
    if agent == "claude":
        cfg = os.environ.get("CLAUDE_CONFIG_DIR")
        base = Path(cfg) if cfg else Path.home() / ".claude"
        return base / "skills" / "AI4BayesCode"
    return Path.home() / ".agents" / "skills" / "AI4BayesCode"


def _is_skill_dir(d: Path) -> bool:
    """True when ``d`` holds an AI4BayesCode skill (or nothing), so replacing
    it cannot destroy anything else."""
    if not any(d.iterdir()):
        return True
    f = d / "SKILL.md"
    if not f.is_file():
        return False
    head = f.read_text(encoding="utf-8", errors="replace").splitlines()[:5]
    return any(line.strip() == "name: AI4BayesCode" for line in head)


def _remove_tree(d: Path) -> None:
    """rmtree that also clears the read-only bit git sets on pack files,
    which otherwise stops the deletion on Windows."""
    def clear_and_retry(func, path, _exc):
        os.chmod(path, stat.S_IWRITE)
        func(path)
    if sys.version_info >= (3, 12):
        shutil.rmtree(d, onexc=clear_and_retry)
    else:
        shutil.rmtree(d, onerror=clear_and_retry)


def install_skill(agent=_AGENTS, ref: str = "main", quiet: bool = False) -> dict:
    """Install the AI4BayesCode skill where the coding agent looks for it.

    Puts the skill in ``~/.claude/skills/AI4BayesCode`` for Claude Code
    (``$CLAUDE_CONFIG_DIR/skills/AI4BayesCode`` when that variable is set) and
    in ``~/.agents/skills/AI4BayesCode`` for Codex. Run it again to update.
    An existing copy at the destination is replaced; a folder there that does
    not hold the AI4BayesCode skill is never touched, the call raises instead.

    Parameters
    ----------
    agent:
        ``"claude"``, ``"codex"``, or a sequence of both (the default).
    ref:
        Git branch or tag to install (default ``"main"``).
    quiet:
        Suppress the progress messages.

    Returns
    -------
    dict
        The installed folder for each agent.
    """
    agents = (agent,) if isinstance(agent, str) else tuple(agent)
    bad = [a for a in agents if a not in _AGENTS]
    if not agents or bad:
        raise ValueError(f"agent must be 'claude', 'codex' or both; got {agent!r}")
    if not isinstance(ref, str) or not ref:
        raise ValueError("ref must be a branch or tag name, e.g. 'main'")
    dests = {a: _skill_dest(a) for a in agents}
    for d in dests.values():
        if d.is_dir() and not _is_skill_dir(d):
            raise FileExistsError(
                f"{d} already exists and does not hold the AI4BayesCode skill. "
                "Move it away, then run this again.")

    url = f"https://github.com/{_SKILL_REPO}/archive/{ref}.zip"
    with tempfile.TemporaryDirectory(prefix="ai4b_skill_") as tmp:
        tmp = Path(tmp)
        archive = tmp / "skill.zip"
        if not quiet:
            print(f"Downloading the AI4BayesCode skill ({ref}) from GitHub ...", flush=True)
        try:
            with urllib.request.urlopen(url, timeout=600) as resp, open(archive, "wb") as fh:
                shutil.copyfileobj(resp, fh)
        except OSError as exc:
            raise RuntimeError(
                f"Could not download {url}\n  {exc}\n"
                "  Check the network connection (and any proxy), then retry.") from exc
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(tmp / "x")
        tops = [p for p in (tmp / "x").iterdir() if p.is_dir()]
        if len(tops) != 1 or not (tops[0] / "SKILL.md").is_file():
            raise RuntimeError(f"The download from {url} does not contain the skill.")
        for d in dests.values():
            if d.exists():
                _remove_tree(d)
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(tops[0], d)

    if not quiet:
        print("Installed the AI4BayesCode skill:")
        if "claude" in dests:
            print(f"  Claude Code: {dests['claude']}\n"
                  "    restart Claude Code, then type /AI4BayesCode")
        if "codex" in dests:
            print(f"  Codex:       {dests['codex']}\n"
                  "    restart Codex, then type $AI4BayesCode or pick it from /skills")
            legacy = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex") \
                / "skills" / "AI4BayesCode"
            if legacy.is_dir():
                print(f"  Note: an older copy is still in {legacy}. Codex reads that "
                      "folder too and would list the skill twice, so delete it.")
        print("Run AI4BayesCode.install_skill() again to update.")
    return {a: str(d) for a, d in dests.items()}
