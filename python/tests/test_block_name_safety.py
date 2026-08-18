"""`remove_block` must never delete anything but one block directory.

`blocks_path(name)` is `os.path.join(library_root, name)`, and os.path.join
has two behaviours that turn a bad name into a destructive path:

  * `os.path.join(root, "")` returns the library ROOT. It passes `isdir()`,
    so `remove_block("")` used to rmtree every installed block.
  * `os.path.join(root, "/abs/path")` DISCARDS the root entirely, so an
    absolute name reached straight out of the library.

`"../x"` escapes by the ordinary route. Run:

    python -m pytest python/tests/test_block_name_safety.py
"""

from __future__ import annotations

import os

import pytest

from AI4BayesCode.install_block import (_check_block_name, blocks_path,
                                        installed_blocks, remove_block)


@pytest.fixture()
def library(tmp_path, monkeypatch):
    """A block library with two blocks, plus a sibling directory that no block
    operation is ever allowed to touch."""
    root = tmp_path / "data_home"
    for b in ("blockA", "blockB"):
        (root / "blocks_download" / b).mkdir(parents=True)
        (root / "blocks_download" / b / f"{b}.hpp").write_text("// x\n")
    (root / "PRECIOUS_USER_DATA").mkdir(parents=True)
    (root / "PRECIOUS_USER_DATA" / "keepme.txt").write_text("do not delete\n")
    monkeypatch.setenv("AI4BAYESCODE_DATA_HOME", str(root))
    # installed_blocks() reports BOTH tiers, and the local tier is relative to
    # the working directory -- so the cwd has to be isolated too, or the repo's
    # own ./blocks_local/ leaks into every assertion here.
    monkeypatch.chdir(tmp_path)
    return root


# Every one of these resolved to a real directory outside (or above) the single
# block it names.
BAD_NAMES = [
    "",                     # -> the library root itself
    ".",
    "..",                   # -> the data home, i.e. the library's parent
    "../PRECIOUS_USER_DATA",
    "a/b",
    "a\\b",
    None,
    42,
    ["blockA"],
]


@pytest.mark.parametrize("name", BAD_NAMES)
def test_remove_block_rejects_non_names(library, name):
    with pytest.raises((ValueError, TypeError)):
        remove_block(name)
    assert sorted(installed_blocks()) == ["blockA", "blockB"]
    assert (library / "PRECIOUS_USER_DATA" / "keepme.txt").exists()


def test_remove_block_rejects_an_absolute_path(library):
    """os.path.join(root, "/abs") == "/abs" -- the library prefix vanishes, so
    without the guard this deleted a directory with no relation to the library."""
    victim = str(library / "PRECIOUS_USER_DATA")
    with pytest.raises(ValueError):
        remove_block(victim)
    assert os.path.isdir(victim)


# None is NOT a bad name for blocks_path -- it is the documented "give me the
# library root" form. It IS rejected by remove_block, which needs a block.
@pytest.mark.parametrize("name",
                         [n for n in BAD_NAMES if n is not None] + ["/abs/path"])
def test_blocks_path_rejects_non_names(library, name):
    """The guard sits in blocks_path, so install_block and any future caller
    inherit it rather than each needing their own check."""
    with pytest.raises((ValueError, TypeError)):
        blocks_path(name)


def test_blocks_path_with_no_name_is_still_the_library_root(library):
    assert blocks_path() == str(library / "blocks_download")


def test_a_real_block_still_removes(library):
    assert remove_block("blockA") is True
    assert sorted(installed_blocks()) == ["blockB"]
    assert remove_block("blockA") is False        # already gone, not an error


def test_check_block_name_accepts_ordinary_names():
    for ok in ("particle_gibbs_block", "nngp_gaussian_gibbs_block", "b1", "A-B"):
        assert _check_block_name(ok) == ok
