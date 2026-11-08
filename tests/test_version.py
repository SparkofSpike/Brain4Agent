"""The version lives in two files, so a test keeps them honest.

`pyproject.toml` is what the package metadata says; `__init__.py` is what
`brain4agent.__version__` reports at runtime. A release that bumps one and not
the other produces a wheel labelled 0.2.0 that prints 0.1.0 -- and PyPI will not
let you fix it, because the filename is already taken.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import brain4agent

ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_and_package_agree() -> None:
    with (ROOT / "pyproject.toml").open("rb") as fh:
        declared = tomllib.load(fh)["project"]["version"]

    assert brain4agent.__version__ == declared, (
        f"pyproject says {declared}, package reports {brain4agent.__version__}"
    )


def test_version_is_semver() -> None:
    parts = brain4agent.__version__.split(".")
    assert len(parts) == 3, brain4agent.__version__
    assert all(p.isdigit() for p in parts), brain4agent.__version__
