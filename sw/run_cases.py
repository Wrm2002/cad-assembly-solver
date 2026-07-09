"""Run the unchanged legacy manifest/build commands over a case directory."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


STEP_SUFFIXES = {".step", ".stp"}
MATCH_COUNT_RE = re.compile(r"Feature matches:\s*(\d+)")


@dataclass
class CaseResult:
    case_id: str
    num_parts: int
    mode: str
    manifest_generated: bool
    assembly_generated: bool
    num_matches: int | None
    runtime_sec: float
    error_message: str
    manifest_size_bytes: int | None
    assembly_size_bytes: int | None
    compute_returncode: int | None
    build_returncode: int | None


def _step_inputs(case_dir: Path) -> list[Path]:
    return sorted(
        p for p in case_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in STEP_SUFFIXES
        and not p.name.lower().startswith("assembly")
    )


def discover_cases(dataset_root: Path) -> list[Path]:
    candidates = [dataset_root] if _step_inputs(dataset_root) else []
    candidates.extend(
        p for p in dataset_root.iterdir()
        if p.is_dir() and _step_inputs(p)
    )
    candidates.extend(
        p / "step"
        for p in dataset_root.iterdir()
        if p.is_dir() and (p / "step").is_dir() and _step_inputs(p / "step")
    )
    return sorted(
        set(candidates),
        key=lambda p: (0, int(p.name)) if p.name.isdigit() else (1, p.name.lower()),
    )


def _case_id(case_dir: Path) -> str:
    return case_dir.parent.name if case_dir.name.lower() == "step" else case_dir.name


def _run(command: list[str], cwd: Path, timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def _write_log(path: Path, command: list[str], result: subprocess.CompletedProcess[str]) -> None:
    path.write_text(
        f"COMMAND: {subprocess.list2cmdline(command)}\n"
        f"RETURN CODE: {result.returncode}\n\n"
        f"--- STDOUT ---\n{result.stdout}\n"
        f"--- STDERR ---\n{result.stderr}\n",
        encoding="utf-8",
    )


def run_case(
    case_dir: Path,
    project_dir: Path,
    output_root: Path,
    decompose: bool,
    use_parents: bool,
    compute_only: bool,
    solver: str,
    enable_scoring: bool,
    enable_pruning: bool,
    beam_width: int,
    min_score: float,
    max_neighbors: int,
    timeout: float,
) -> CaseResult:
    started = time.perf_counter()
    case_id = _case_id(case_dir)
    inputs = _step_inputs(case_dir)
    mode = f"{solver}_decompose" if decompose else solver
    if enable_pruning:
        mode += "_scoring_pruning"
    elif enable_scoring:
        mode += "_scoring"
    manifest = case_dir / "assembly_manifest.json"
    assembly = case_dir / "assembly.step"
    compute_result = None
    build_result = None
    errors: list[str] = []
    num_matches = None

    compute_command = [sys.executable, str(project_dir / "compute_manifest.py"), str(case_dir)]
    compute_command.extend(["--solver", solver])
    if enable_scoring:
        compute_command.append("--enable-scoring")
    if enable_pruning:
        compute_command.append("--enable-pruning")
    compute_command.extend([
        "--beam-width", str(beam_width),
        "--min-score", str(min_score),
        "--max-neighbors", str(max_neighbors),
    ])
    if decompose:
        compute_command.append("--decompose")

    try:
        compute_result = _run(compute_command, project_dir, timeout)
        _write_log(output_root / "logs" / f"{case_id}_compute.log", compute_command, compute_result)
        match = MATCH_COUNT_RE.search(compute_result.stdout)
        if match:
            num_matches = int(match.group(1))
        if compute_result.returncode != 0:
            errors.append(f"compute_manifest exit {compute_result.returncode}")
    except subprocess.TimeoutExpired:
        errors.append(f"compute_manifest timeout after {timeout:g}s")
    except Exception as exc:
        errors.append(f"compute_manifest error: {exc}")

    manifest_generated = (
        compute_result is not None and compute_result.returncode == 0 and manifest.is_file()
    )

    if manifest_generated and not compute_only:
        build_command = [sys.executable, str(project_dir / "build_assembly.py"), str(case_dir)]
        if use_parents:
            build_command.append("--use-parents")
        try:
            build_result = _run(build_command, project_dir, timeout)
            _write_log(output_root / "logs" / f"{case_id}_build.log", build_command, build_result)
            if build_result.returncode != 0:
                errors.append(f"build_assembly exit {build_result.returncode}")
        except subprocess.TimeoutExpired:
            errors.append(f"build_assembly timeout after {timeout:g}s")
        except Exception as exc:
            errors.append(f"build_assembly error: {exc}")
    elif not manifest_generated:
        errors.append("build skipped because this run did not generate a manifest")
    elif compute_only:
        errors.append("build skipped by --compute-only")

    assembly_generated = (
        build_result is not None and build_result.returncode == 0 and assembly.is_file()
    )
    result = CaseResult(
        case_id=case_id,
        num_parts=len(inputs),
        mode=mode,
        manifest_generated=manifest_generated,
        assembly_generated=assembly_generated,
        num_matches=num_matches,
        runtime_sec=round(time.perf_counter() - started, 3),
        error_message="; ".join(errors),
        manifest_size_bytes=manifest.stat().st_size if manifest_generated else None,
        assembly_size_bytes=assembly.stat().st_size if assembly_generated else None,
        compute_returncode=compute_result.returncode if compute_result else None,
        build_returncode=build_result.returncode if build_result else None,
    )
    (output_root / "reports" / f"{case_id}.json").write_text(
        json.dumps(asdict(result), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root")
    parser.add_argument(
        "--project-dir",
        default=str(Path(__file__).resolve().parent),
        help="Directory containing compute_manifest.py and build_assembly.py.",
    )
    parser.add_argument("--solver", choices=["bfs", "reliable"], default="bfs")
    parser.add_argument("--enable-scoring", action="store_true")
    parser.add_argument("--enable-pruning", action="store_true")
    parser.add_argument("--beam-width", type=int, default=20)
    parser.add_argument("--min-score", type=float, default=0.5)
    parser.add_argument("--max-neighbors", type=int, default=4)
    parser.add_argument("--decompose", action="store_true")
    parser.add_argument("--use-parents", action="store_true")
    parser.add_argument(
        "--compute-only",
        action="store_true",
        help="Generate manifests without invoking the native STEP writer.",
    )
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument(
        "--cases",
        nargs="+",
        help="Only run the named case directories, for example: --cases 1 2.",
    )
    parser.add_argument(
        "--output-dir",
        help="Default: <dataset_root>/baseline_results.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    project_dir = Path(args.project_dir).resolve()
    output_root = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else dataset_root / "baseline_results"
    )
    (output_root / "logs").mkdir(parents=True, exist_ok=True)
    (output_root / "reports").mkdir(parents=True, exist_ok=True)

    cases = discover_cases(dataset_root)
    if args.cases:
        selected = set(args.cases)
        cases = [case for case in cases if _case_id(case) in selected]
        missing = sorted(selected - {_case_id(case) for case in cases})
        if missing:
            parser.error(f"requested cases not found: {', '.join(missing)}")
    if not cases:
        parser.error(f"no case directories containing STEP inputs under {dataset_root}")

    results = []
    for index, case_dir in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {_case_id(case_dir)}", flush=True)
        result = run_case(
            case_dir,
            project_dir,
            output_root,
            args.decompose,
            args.use_parents,
            args.compute_only,
            args.solver,
            args.enable_scoring,
            args.enable_pruning,
            args.beam_width,
            args.min_score,
            args.max_neighbors,
            args.timeout,
        )
        results.append(result)
        state = "OK" if not result.error_message else result.error_message
        print(f"  {state} ({result.runtime_sec:.3f}s)", flush=True)

    fieldnames = list(CaseResult.__dataclass_fields__)
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)

    failures = sum(bool(result.error_message) for result in results)
    print(f"Summary: {output_root / 'summary.csv'}")
    print(f"Cases: {len(results)}, failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
