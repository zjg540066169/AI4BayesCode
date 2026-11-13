"""Compile and run a sampler when the build cache and the temp folder live under
paths with a space and non-ASCII characters, as they do for a Windows user
whose account name has them. The Windows CI job sets LOCALAPPDATA and TMP.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import AI4BayesCode
from AI4BayesCode.source import _user_cache_dir

# This script prints the Chinese folder names it tests; the console code page
# of the CI runner cannot show them.
sys.stdout.reconfigure(errors="backslashreplace")
print("cache:", _user_cache_dir())
print("temp:", tempfile.gettempdir())

mod = AI4BayesCode.example("GaussianLocationScale", rebuild=True)
y = np.random.default_rng(1).normal(3.0, 2.0, 200)
m = mod.GaussianLocationScale(y, 7, False)
m.step(2000)
cur = m.get_current()
print(cur)
assert abs(cur["mu"][0] - 3) < 1 and abs(cur["sigma"][0] - 2) < 1

# A changed build of the same module needs a fresh interpreter (pybind11
# registers a module's classes once per process), so these run in children:
# a source file in a folder whose name has a space and Chinese characters, and
# the same source passed as text (source() then writes it under TMP).
src = Path(tempfile.gettempdir()) / "my model" / "my model.cpp"
src.parent.mkdir(parents=True, exist_ok=True)
text = Path(AI4BayesCode.examples_path("GaussianLocationScale")).read_text(encoding="utf-8")
src.write_text(text + "\n// a changed copy\n", encoding="utf-8")
child = ("import sys, AI4BayesCode\n"
         "arg = sys.argv[1]\n"
         "code = arg if sys.argv[2] == 'file' else open(arg, encoding='utf-8').read() + '// text\\n'\n"
         "mod = AI4BayesCode.source(code, rebuild=True, quiet=True)\n"
         "print('child built', mod.__file__)\n")
for kind in ("file", "text"):
    subprocess.run([sys.executable, "-c", child, str(src), kind], check=True)

# Asking again in this process for the build already loaded returns it.
assert AI4BayesCode.example("GaussianLocationScale") is mod
print("PASS")
