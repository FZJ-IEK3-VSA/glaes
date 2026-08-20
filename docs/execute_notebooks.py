"""Execute the example notebooks in parallel and write outputs in place.

This is a documentation **pre-build** step (see ``.readthedocs.yaml``).
``mkdocs-jupyter`` is configured with ``execute: false``, so it only renders the
outputs produced here. Running execution as a dedicated, parallel step is faster
than mkdocs-jupyter's sequential in-build execution, and it keeps a notebook
failure out of the middle of the site build, where the traceback is buried in
MkDocs logging.

Notebooks are executed in ``Examples/`` — the source of truth — and *not* in the
staged ``docs/notebooks/`` copy. Two reasons: ``Examples/`` is where the
``data/`` inputs and the ``tmp/`` scratch directory live, so the relative paths
inside cells resolve; and ``docs/notebooks/`` does not exist until
``docs/copy_examples.py`` runs as an ``on_pre_build`` hook, i.e. once
``mkdocs build`` has already started. That hook copies the whole tree, so the
outputs written here are carried into the site automatically.

The notebooks are output-free in git. Executing them therefore dirties the
working tree; use ``--strip`` to clean up before committing.

Usage:
    python docs/execute_notebooks.py                  # default: CPU-1 workers
    python docs/execute_notebooks.py --workers 2
    python docs/execute_notebooks.py --timeout 1200
    python docs/execute_notebooks.py --exclude 02_extended_methods.ipynb
    python docs/execute_notebooks.py --strip          # clear outputs, no execute

Exit code is non-zero if any notebook fails. A cell error aborts that notebook
but does not stop the rest of the batch; the summary at the end lists failures.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


@dataclass
class Result:
    path: Path
    elapsed: float
    error: str | None


def execute_one(path: Path, timeout: int) -> Result:
    """Execute one notebook in its own kernel, writing outputs back in place.

    Each notebook gets its own kernel subprocess, so a ThreadPoolExecutor is
    sufficient — the GIL is not the bottleneck. ``resources.metadata.path`` sets
    the kernel's working directory to the notebook's own folder, so relative
    paths inside cells behave as they do in Jupyter Lab.
    """
    start = time.perf_counter()
    try:
        nb = nbformat.read(path, as_version=4)
        client = NotebookClient(
            nb,
            timeout=timeout,
            kernel_name="python3",
            resources={"metadata": {"path": str(path.parent)}},
        )
        client.execute()
        nbformat.write(nb, path)
        return Result(path, time.perf_counter() - start, None)
    except CellExecutionError as exc:
        first_line = str(exc).splitlines()[0] if str(exc) else "CellExecutionError"
        return Result(path, time.perf_counter() - start, first_line)
    except Exception as exc:  # noqa: BLE001 - report, don't abort the batch
        return Result(path, time.perf_counter() - start, f"{type(exc).__name__}: {exc}")


def strip_outputs(path: Path) -> bool:
    """Clear all code-cell outputs and execution counts in place.

    Returns True if the file changed. Uses stdlib json with nbformat's on-disk
    style (indent=1) so diffs stay minimal.
    """
    nb = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        meta = cell.get("metadata", {})
        if cell.get("outputs") or cell.get("execution_count") is not None or "execution" in meta:
            cell["outputs"] = []
            cell["execution_count"] = None
            meta.pop("execution", None)  # volatile per-cell run timestamps
            changed = True
    if changed:
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return changed


def discover(notebooks_dir: Path, exclude: list[str]) -> list[Path]:
    return sorted(
        p for p in notebooks_dir.rglob("*.ipynb") if p.name not in exclude and ".ipynb_checkpoints" not in p.parts
    )


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--notebooks-dir",
        type=Path,
        default=Path("Examples"),
        help="Directory searched recursively for notebooks (default: Examples)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Number of parallel workers (default: CPU count - 1)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Per-cell timeout in seconds (default: 1800)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="FILENAME",
        help="Notebook filename to skip (repeat for multiple)",
    )
    parser.add_argument(
        "--strip",
        action="store_true",
        help="Clear outputs from the notebooks instead of executing them",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.notebooks_dir.is_dir():
        print(f"error: {args.notebooks_dir} is not a directory", file=sys.stderr)
        return 2

    notebooks = discover(args.notebooks_dir, args.exclude)
    if not notebooks:
        print(f"No notebooks found in {args.notebooks_dir}", file=sys.stderr)
        return 1

    if args.strip:
        stripped = [nb for nb in notebooks if strip_outputs(nb)]
        if stripped:
            print(f"Stripped outputs from {len(stripped)} notebook(s):")
            for nb in stripped:
                print(f"  {nb}")
        else:
            print("No outputs to strip; all notebooks are already clean.")
        return 0

    print(
        f"Executing {len(notebooks)} notebooks with {args.workers} worker(s), per-cell timeout {args.timeout}s",
        flush=True,
    )

    overall_start = time.perf_counter()
    results: list[Result] = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(execute_one, nb, args.timeout): nb for nb in notebooks}
        for fut in as_completed(futures):
            r = fut.result()
            tag = "FAIL" if r.error else "OK  "
            print(f"  [{tag}] {r.path.name:<40} {r.elapsed:6.1f}s", flush=True)
            results.append(r)

    total = time.perf_counter() - overall_start
    failures = [r for r in results if r.error]
    print(f"\nTotal: {total:.1f}s ({len(results) - len(failures)}/{len(results)} succeeded)")
    if failures:
        print("\nFailures:")
        for r in failures:
            print(f"  {r.path}: {r.error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
