#!/usr/bin/env python3
"""Execute one notebook in the locked environment and retain visible outputs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    args = parser.parse_args()
    path = args.notebook.resolve()
    notebook = nbformat.read(path, as_version=4)
    executed = NotebookClient(
        notebook,
        timeout=900,
        kernel_name="chronoclone",
        resources={"metadata": {"path": str(path.parents[1])}},
    ).execute()
    executed.metadata.setdefault("execution", {})["status"] = "ok"
    executed.metadata["execution"]["completed_at_utc"] = datetime.now(
        timezone.utc
    ).isoformat()
    nbformat.write(executed, path)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
