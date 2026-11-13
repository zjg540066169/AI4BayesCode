"""Compile and run a sampler when the build cache and temp folder live under
paths with a space and non-ASCII characters, as they do for a Windows user
whose account name has them. The Windows CI job sets LOCALAPPDATA and TMP.
"""
import shutil
import tempfile
from pathlib import Path

import numpy as np

import AI4BayesCode
from AI4BayesCode.source import _user_cache_dir

print("cache:", _user_cache_dir())
print("temp:", tempfile.gettempdir())

mod = AI4BayesCode.example("GaussianLocationScale", rebuild=True)
y = np.random.default_rng(1).normal(3.0, 2.0, 200)
m = mod.GaussianLocationScale(y, 7, False)
m.step(2000)
cur = m.get_current()
print(cur)
assert abs(cur["mu"][0] - 3) < 1 and abs(cur["sigma"][0] - 2) < 1

# A sampler given as a file in a folder with the same kind of name, and the
# same source passed as a string (source() then writes a temp .cpp).
src = Path(tempfile.gettempdir()) / "my model" / "Gauss.cpp"
src.parent.mkdir(parents=True, exist_ok=True)
shutil.copy(AI4BayesCode.examples_path("GaussianLocationScale"), src)
AI4BayesCode.source(src, rebuild=True, quiet=True)
AI4BayesCode.source(src.read_text(encoding="utf-8"), rebuild=True, quiet=True)

# Rebuilding a module that is already loaded: Windows locks the loaded file.
AI4BayesCode.example("GaussianLocationScale", rebuild=True)
print("PASS")
