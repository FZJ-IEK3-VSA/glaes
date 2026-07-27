"""MkDocs hook to stage example notebooks (and their data) for the docs build.

The whole ``Examples/`` tree is copied to ``docs/notebooks/``, which is what the
nav refers to. The notebooks are executed beforehand, in ``Examples/`` itself, by
``docs/execute_notebooks.py`` (a docs pre-build step; see ``.readthedocs.yaml``),
so the copy carries their outputs and mkdocs-jupyter — configured with
``execute: false`` — simply renders them.

The copy is still whole-tree rather than notebooks-only: the ``data/`` inputs and
the ``tmp/`` scratch folder must sit next to the notebooks for the relative paths
inside cells to resolve, both during execution and for any file a rendered
notebook links to.
"""

import shutil
from pathlib import Path


def on_pre_build(config, **kwargs):
    src = Path("Examples")
    dst = Path("docs/notebooks")
    if not src.exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
