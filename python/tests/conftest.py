"""Shared pytest configuration for the AI4BayesCode Python tests."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "compile: compiles a shipped C++ example (uses a C++ compiler and "
        "Armadillo; deselect with -m 'not compile')")
