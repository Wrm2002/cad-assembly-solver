"""Audit completeness and ground-truth structure of a synthetic dataset."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


def _step_readable(path):
    try:
        from OCC.Core.IFSelect import IFSelect_RetDone
        from OCC.Core.STEPControl import STEPControl_Reader

        reader = STEPControl_Reader()
        if reader.ReadFile(str(path)) != IFSelect_RetDone:
            return False, "STEPControl_Reader.ReadFile failed"
        transferred = reader.TransferRoots()
        if transferred <= 0 or reader.OneShape().IsNull():
            return False, "STEP contains no transferable shape"
        return True, None
    except Exception as exc:
        return False, str(exc)


def _geometry_consistency(case, gt, relative_tolerance=1e-3):
    from part_index import index_part
    from placement_validation import transformed_bbox

    assembly = index_part(case / "step" / "assembly_gt.step", "assembly_gt")
    union_min = [float("inf")] * 3
    union_max = [float("-inf")] * 3
    volume_sum = 0.0
    for name in gt["parts"]:
        part = index_part(case / "step" / name, Path(name).stem)
        placement = gt.get("placements", {}).get(name, {})
        box = transformed_bbox(
            {"bbox": {"min": part.bbox.minimum, "max": part.bbox.maximum}},
            placement,
        )
        for axis in range(3):
            union_min[axis] = min(union_min[axis], box["min"][axis])
            union_max[axis] = max(union_max[axis], box["max"][axis])
        if part.volume is not None:
            volume_sum += part.volume
    assembly_min, assembly_max = assembly.bbox.minimum, assembly.bbox.maximum
    diagonal = sum(
        (assembly_max[axis] - assembly_min[axis]) ** 2 for axis in range(3)
    ) ** 0.5
    tolerance = max(0.1, diagonal * relative_tolerance)
    bbox_error = max(
        abs(union_min[axis] - assembly_min[axis])
        for axis in range(3)
    )
    bbox_error = max(
        bbox_error,
        max(abs(union_max[axis] - assembly_max[axis]) for axis in range(3)),
    )
    volume_error_ratio = None
    if assembly.volume and volume_sum:
        volume_error_ratio = abs(assembly.volume - volume_sum) / max(
            assembly.volume, volume_sum
        )
    passed = bbox_error <= tolerance and (
        volume_error_ratio is None or volume_error_ratio <= relative_tolerance
    )
    return {
        "case_id": case.name,
        "passed": passed,
        "expected_union_bbox": {"min": union_min, "max": union_max},
        "assembly_bbox": {"min": assembly_min, "max": assembly_max},
        "bbox_max_absolute_error_mm": bbox_error,
        "bbox_tolerance_mm": tolerance,
        "volume_relative_error": volume_error_ratio,
    }


def _isolated_probe(case, geometry=False, timeout=60):
    worker = Path(__file__).with_name("step_audit_worker.py")
    command = [sys.executable, str(worker), str(case)]
    if geometry:
        command.append("--geometry")
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    if result.returncode not in (0, 1) or not result.stdout.strip():
        return {
            "case_id": case.name,
            "step_errors": [{
                "file": None,
                "error": (
                    f"isolated worker returncode={result.returncode}; "
                    f"stderr={result.stderr[-1000:]}"
                ),
            }],
            "geometry": None,
        }
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except Exception as exc:
        return {
            "case_id": case.name,
            "step_errors": [{
                "file": None,
                "error": f"invalid worker JSON: {exc}",
            }],
            "geometry": None,
        }


def audit(
    root,
    expected_per_group=None,
    verify_step=False,
    geometry_samples_per_group=0,
    probe_retries=2,
    group_sizes=None,
):
    root = Path(root).resolve()
    group_sizes = tuple(group_sizes or range(1, 7))
    counts = Counter()
    failures = []
    complete_cases = {size: [] for size in range(1, 7)}
    geometry_checks = []
    for case in sorted(path for path in root.iterdir() if path.is_dir()):
        try:
            gt = json.loads((case / "gt.json").read_text(encoding="utf-8"))
            size = int(gt["group_size"])
            expected_parts = [f"part_{index:02d}" for index in range(1, size + 1)]
            required = [
                case / "generated_spec.json",
                case / "native" / "assembly.sldasm",
                case / "step" / "assembly_gt.step",
                *[case / "native" / f"{stem}.sldprt" for stem in expected_parts],
                *[case / "step" / f"{stem}.step" for stem in expected_parts],
            ]
            missing = [
                str(path.relative_to(case))
                for path in required
                if not path.is_file() or path.stat().st_size == 0
            ]
            if len(gt.get("parts", [])) != size:
                missing.append("gt.json: parts length mismatch")
            if set(gt.get("placements", {})) != set(gt.get("parts", [])):
                missing.append("gt.json: placements do not match parts")
            if missing:
                failures.append({"case_id": case.name, "problems": missing})
            else:
                if verify_step:
                    run_geometry = (
                        len(complete_cases[size]) < geometry_samples_per_group
                    )
                    probe = _isolated_probe(case, geometry=run_geometry)
                    for _ in range(probe_retries):
                        if not probe.get("step_errors"):
                            break
                        probe = _isolated_probe(case, geometry=run_geometry)
                    read_errors = [
                        f"{item.get('file')}: {item.get('error')}"
                        for item in probe.get("step_errors", [])
                    ]
                    if read_errors:
                        failures.append({
                            "case_id": case.name,
                            "problems": read_errors,
                        })
                        continue
                    if probe.get("geometry") is not None:
                        geometry_checks.append(probe["geometry"])
                counts[size] += 1
                complete_cases[size].append((case, gt))
        except Exception as exc:
            failures.append({"case_id": case.name, "problems": [str(exc)]})
    missing_groups = {}
    if expected_per_group is not None:
        missing_groups = {
            str(size): expected_per_group - counts[size]
            for size in group_sizes
            if counts[size] != expected_per_group
        }
    if not verify_step and geometry_samples_per_group:
        for size in group_sizes:
            for case, _ in complete_cases[size][:geometry_samples_per_group]:
                probe = _isolated_probe(case, geometry=True)
                geometry_checks.append(
                    probe.get("geometry") or {
                        "case_id": case.name,
                        "passed": False,
                        "error": probe.get("step_errors"),
                    }
                )
    geometry_failures = [
        item for item in geometry_checks if not item.get("passed", False)
    ]
    return {
        "root": str(root),
        "valid_cases": sum(counts.values()),
        "valid_cases_by_group": {str(size): counts[size] for size in range(1, 7)},
        "invalid_cases": len(failures),
        "missing_or_extra_by_group": missing_groups,
        "failures": failures,
        "step_reopen_verified": verify_step,
        "geometry_consistency_checks": geometry_checks,
        "geometry_consistency_failures": len(geometry_failures),
        "status": (
            "success"
            if not failures and not missing_groups and not geometry_failures
            else "incomplete"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root")
    parser.add_argument("--expected-per-group", type=int)
    parser.add_argument("--group-size", nargs="+", type=int)
    parser.add_argument("--verify-step", action="store_true")
    parser.add_argument("--geometry-samples-per-group", type=int, default=0)
    parser.add_argument("--probe-retries", type=int, default=2)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = audit(
        args.dataset_root,
        args.expected_per_group,
        verify_step=args.verify_step,
        geometry_samples_per_group=args.geometry_samples_per_group,
        probe_retries=args.probe_retries,
        group_sizes=args.group_size,
    )
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        Path(args.output).resolve().write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
