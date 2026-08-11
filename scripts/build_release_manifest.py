#!/usr/bin/env python3
"""Build a deterministic checksum manifest for a source release."""

import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
from typing import Dict, Iterable, Sequence


DEFAULT_PATHS = (
    ".gitignore",
    "Dockerfile",
    "Makefile",
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "results/validation/gsm8k_structure.json",
    "configs",
    "docs",
    "scripts",
    "slurm",
    "src",
    "tests",
)
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def release_files(root: Path, includes: Sequence[str]) -> Iterable[Path]:
    files = set()
    for include in includes:
        path = root / include
        if not path.exists():
            raise FileNotFoundError(f"Release input does not exist: {path}")
        candidates = path.rglob("*") if path.is_dir() else (path,)
        for candidate in candidates:
            relative = candidate.relative_to(root)
            if (
                candidate.is_file()
                and not EXCLUDED_PARTS.intersection(relative.parts)
                and candidate.suffix not in EXCLUDED_SUFFIXES
            ):
                files.add(candidate)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def _git_value(root: Path, arguments: Sequence[str]) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def build_manifest(root: Path, includes: Sequence[str] = DEFAULT_PATHS) -> Dict:
    root = root.resolve()
    files = release_files(root, includes)
    revision = _git_value(root, ("rev-parse", "HEAD"))
    status = _git_value(root, ("status", "--porcelain"))
    if revision is None or status is None:
        raise RuntimeError(
            f"Release root must be a Git worktree with readable provenance: {root}"
        )
    return {
        "schema_version": 1,
        "git_revision": revision,
        "git_dirty": bool(status),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "implementation": platform.python_implementation(),
        },
        "files": [
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--include", action="append", dest="includes")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = build_manifest(args.root, args.includes or DEFAULT_PATHS)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Manifested {len(manifest['files'])} files: {args.output}")


if __name__ == "__main__":
    main()