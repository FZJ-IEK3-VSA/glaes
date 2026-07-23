"""MkDocs hook to stage example notebooks (and their data) for the docs build.

The whole ``Examples/`` tree is copied to ``docs/notebooks/`` so that mkdocs-jupyter
can execute the notebooks in place: the copied ``data/`` inputs and ``tmp/`` output
folder sit next to the notebooks, so their relative paths resolve during execution.
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
