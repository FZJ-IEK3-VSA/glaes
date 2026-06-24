"""MkDocs hook to copy example notebooks and data from Examples/ to docs/notebooks/."""

import shutil
from pathlib import Path


def on_pre_build(config, **kwargs):
    src = Path("Examples")
    dst = Path("docs/notebooks")
    if src.exists():
        dst.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            dest_path = dst / item.name
            if item.is_file():
                shutil.copy2(item, dest_path)
