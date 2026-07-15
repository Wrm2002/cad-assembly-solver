"""Build a ZIP whose entry names are portable to Linux."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not args.source.is_dir():
        raise FileNotFoundError(args.source)
    if args.output.exists():
        raise FileExistsError(args.output)

    files = sorted(path for path in args.source.rglob("*") if path.is_file())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        args.output,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        for path in files:
            archive.write(path, path.relative_to(args.source).as_posix())

    report = {
        "source": str(args.source.resolve()),
        "output": str(args.output.resolve()),
        "entry_count": len(files),
        "size_bytes": args.output.stat().st_size,
        "sha256": sha256(args.output),
        "path_separator": "/",
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
