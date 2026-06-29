"""setuptools build script for AI4BayesCode Python helper.

The helper itself is pure Python (compiles user `.cpp` files on demand
via `sourceCpp`). No C++ extension modules ship with the package —
each user's model compiles at runtime against their local AI4BayesCode
header tree.

Install:
    pip install .          # from the `python/` folder
    pip install -e .       # dev install, editable

Once installed:
    import AI4BayesCode
    mod = AI4BayesCode.sourceCpp("MyModel.cpp", ai4bayescode_path=".../AI4BayesCode")
"""

from setuptools import setup, find_packages

setup(
    name="AI4BayesCode",
    version="1.0.0",
    description="Python helper for the AI4BayesCode MCMC library",
    long_description=(
        "Compile-on-demand pybind11 wrapper around user-written AI4BayesCode "
        "`.cpp` files, plus DAG visualization, parallel-chain orchestration, "
        "and convergence diagnostics. Analogous to the R-side AI4BayesCode_helpers.R."
    ),
    author="AI4BayesCode authors",
    author_email="jungang.zou@gmail.com",
    url="https://github.com/zjg540066169/AI4BayesCode",
    license="GPL-3.0-or-later",
    python_requires=">=3.11",
    packages=find_packages(include=["AI4BayesCode", "AI4BayesCode.*"]),
    # Ship the vendored AI4BayesCode C++ header tree so source()
    # is self-contained (no checkout needed). See MANIFEST.in (graft).
    include_package_data=True,
    package_data={"AI4BayesCode": ["start.md", "_examples/**/*", "_vendored_include/**/*", "_skills/**/*"]},
    install_requires=[
        "numpy>=1.24",
        "pybind11>=2.11",
    ],
    extras_require={
        "viz": ["networkx>=3.0", "matplotlib>=3.7"],
        "diagnostics": ["arviz>=0.16"],
        "generate": ["anthropic>=0.40"],
        "dev": ["pytest>=7.0", "pytest-cov"],
    },
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
