"""Run STEP simplification in isolated workers without touching source files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
CASE_IDS = tuple(str(number) for number in range(1, 6))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def source_files(source_root: Path, case_ids: tuple[str, ...]):
    for case_id in case_ids:
        case_dir = source_root / case_id
        for path in sorted(case_dir.iterdir()):
            if (
                path.is_file()
                and path.suffix.lower() in {".step", ".stp"}
                and not path.name.lower().startswith("assembly")
            ):
                yield case_id, path.resolve()


def load_successful_audit(
    audit: Path, output: Path, expected_source_hash: str
) -> dict[str, Any] | None:
    if not audit.is_file() or not output.is_file():
        return None
    try:
        report = json.loads(audit.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        report.get("status") == "success"
        and report.get("source_sha256") == expected_source_hash
        and report.get("checks", {}).get("accepted") is True
    ):
        return report
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(HERE))
    parser.add_argument(
        "--output-root",
        default=str(HERE / "real_simplified_20260703"),
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=CASE_IDS,
        default=list(CASE_IDS),
    )
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--acknowledge-native-crash-risk",
        action="store_true",
    )
    args = parser.parse_args()
    if not args.acknowledge_native_crash_risk:
        parser.error(
            "native geometry simplification is quarantined; the real "
            "fan-cage model crashes OCCT TKG2d.dll with 0xc0000005"
        )
    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()
    protected_case_roots = [
        (source_root / case_id).resolve() for case_id in CASE_IDS
    ]
    if any(
        output_root == case_root or case_root in output_root.parents
        for case_root in protected_case_roots
    ):
        raise ValueError(
            "output root must not be inside the source case directories"
        )
    case_ids = tuple(args.cases)
    sources = list(source_files(source_root, case_ids))
    initial_hashes = {str(path): sha256_file(path) for _, path in sources}
    write_json(
        output_root / "source_hashes_before.json",
        {
            "source_root": str(source_root),
            "files": initial_hashes,
        },
    )
    reports = []
    for position, (case_id, source) in enumerate(sources, start=1):
        relative = Path(case_id) / source.name
        output = output_root / "models" / relative
        audit = output_root / "audits" / case_id / f"{source.name}.json"
        log_dir = output_root / "logs" / case_id
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / f"{source.name}.stdout.log"
        stderr_path = log_dir / f"{source.name}.stderr.log"
        source_hash = initial_hashes[str(source)]
        cached = None if args.overwrite else load_successful_audit(
            audit, output, source_hash
        )
        print(
            f"[{position}/{len(sources)}] case {case_id}: {source.name}",
            flush=True,
        )
        if cached is not None:
            reports.append({**cached, "batch_status": "resumed"})
            print("  resumed", flush=True)
            continue
        command = [
            sys.executable,
            str(HERE / "step_simplifier.py"),
            str(source),
            str(output),
            "--audit",
            str(audit),
            "--acknowledge-native-crash-risk",
        ]
        if args.overwrite:
            command.append("--overwrite")
        started = time.perf_counter()
        try:
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=args.timeout,
            )
            stdout_path.write_text(process.stdout, encoding="utf-8")
            stderr_path.write_text(process.stderr, encoding="utf-8")
            report = (
                json.loads(audit.read_text(encoding="utf-8"))
                if process.returncode == 0 and audit.is_file()
                else {
                    "status": "worker_failed",
                    "source": str(source),
                    "output": str(output),
                    "source_sha256": source_hash,
                    "returncode": process.returncode,
                    "stderr_tail": process.stderr[-4000:],
                }
            )
        except subprocess.TimeoutExpired as exc:
            stdout_path.write_text(
                exc.stdout or "" if isinstance(exc.stdout, str) else "",
                encoding="utf-8",
            )
            stderr_path.write_text(
                exc.stderr or "" if isinstance(exc.stderr, str) else "",
                encoding="utf-8",
            )
            report = {
                "status": "timeout",
                "source": str(source),
                "output": str(output),
                "source_sha256": source_hash,
                "timeout_seconds": args.timeout,
            }
        report["case_id"] = case_id
        report["batch_elapsed_seconds"] = time.perf_counter() - started
        report["batch_status"] = report["status"]
        reports.append(report)
        print(f"  {report['batch_status']}", flush=True)
        if sha256_file(source) != source_hash:
            raise RuntimeError(f"source hash changed during worker: {source}")
    final_hashes = {str(path): sha256_file(path) for _, path in sources}
    changed_sources = sorted(
        path for path, digest in initial_hashes.items()
        if final_hashes[path] != digest
    )
    rows = []
    for report in reports:
        before = report.get("before", {}).get("topology", {})
        after = report.get("after_roundtrip", {}).get("topology", {})
        rows.append({
            "case_id": report.get("case_id"),
            "source": Path(report["source"]).name,
            "status": report.get("batch_status"),
            "faces_before": before.get("faces"),
            "faces_after": after.get("faces"),
            "solids_before": before.get("solids"),
            "solids_after": after.get("solids"),
            "face_reduction_fraction": report.get(
                "face_reduction_fraction"
            ),
            "file_size_reduction_fraction": report.get(
                "file_size_reduction_fraction"
            ),
            "elapsed_seconds": report.get("elapsed_seconds"),
        })
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "simplification_results.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": "1.0.0",
        "method": (
            "lossless-first per-solid same-domain topology unification"
        ),
        "source_root": str(source_root),
        "output_root": str(output_root),
        "source_file_count": len(sources),
        "success_count": sum(
            row["status"] in {"success", "resumed"} for row in rows
        ),
        "failure_count": sum(
            row["status"] not in {"success", "resumed"} for row in rows
        ),
        "source_hashes_unchanged": not changed_sources,
        "changed_sources": changed_sources,
        "results": rows,
    }
    write_json(output_root / "worker_reports.json", reports)
    write_json(output_root / "simplification_summary.json", summary)
    if changed_sources:
        raise RuntimeError(
            "source immutability check failed: "
            + ", ".join(changed_sources)
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if summary["failure_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
