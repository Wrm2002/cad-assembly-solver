"""Generate legacy geometry-smoke SolidWorks datasets.

These primitive cases are explicitly ineligible as functional positives.
Use ``functional_dataset_generator.py`` for D0 data.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from sw_dataset_generator.generate_assembly import generate_assembly
    from sw_dataset_generator.generate_parts import generate_parts
    from sw_dataset_generator.sw_api import SolidWorksSession
    from sw_dataset_generator.templates import build_case_spec
    from sw_dataset_generator.write_ground_truth import write_ground_truth
else:
    from .generate_assembly import generate_assembly
    from .generate_parts import generate_parts
    from .sw_api import SolidWorksSession
    from .templates import build_case_spec
    from .write_ground_truth import write_ground_truth


def generate_case(output_root, group_size, index, seed, session=None, dry_run=False):
    case_id = f"group_{group_size}_{index:06d}"
    case_dir = Path(output_root) / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    spec = build_case_spec(group_size, seed)
    gt_path = write_ground_truth(case_dir, spec, case_id)
    (case_dir / "generated_spec.json").write_text(
        json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if dry_run:
        return {"case_id": case_id, "gt": str(gt_path), "status": "dry_run"}
    if session is None:
        raise ValueError("SolidWorks session required unless dry_run")
    parts = generate_parts(session, case_dir, spec)
    assembly = generate_assembly(session, case_dir, parts, spec)
    # SolidWorks AddComponent5 uses an insertion location, not the translation
    # of the local STEP origin. Repair and verify this semantic distinction in
    # an isolated OCCT process so future GT is correct by construction.
    worker = Path(__file__).with_name("gt_repair_worker.py")
    repair = subprocess.run(
        [sys.executable, str(worker), str(case_dir), "--apply"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )
    if repair.returncode != 0:
        raise RuntimeError(f"ground-truth transform repair failed: {repair.stderr[-1000:]}")
    return {
        "case_id": case_id,
        "gt": str(gt_path),
        "parts": [str(item["step"]) for item in parts],
        "assembly": str(assembly["step"]),
        "status": "generated",
    }


def _complete_case(output_root, group_size, index):
    case_id = f"group_{group_size}_{index:06d}"
    case_dir = Path(output_root) / case_id
    parts = [case_dir / "step" / f"part_{number:02d}.step"
             for number in range(1, group_size + 1)]
    required = [
        case_dir / "gt.json",
        case_dir / "generated_spec.json",
        case_dir / "native" / "assembly.sldasm",
        case_dir / "step" / "assembly_gt.step",
        *parts,
    ]
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        return None
    return {
        "case_id": case_id,
        "gt": str((case_dir / "gt.json").resolve()),
        "parts": [str(path.resolve()) for path in parts],
        "assembly": str((case_dir / "step" / "assembly_gt.step").resolve()),
        "status": "resumed_existing",
    }


def _checkpoint(path, results):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-size", type=int, nargs="+", required=True)
    parser.add_argument("--num-cases", type=int, default=1)
    parser.add_argument("--output-root", default="synthetic_dataset")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--visible", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--session-batch-size", type=int, default=5,
        help="Restart SolidWorks after this many generated cases (default: 5).",
    )
    args = parser.parse_args()
    session = None
    generated_in_session = 0
    results = []
    output = Path(args.output_root).resolve() / "generation_summary.json"
    try:
        for size in args.group_size:
            for index in range(1, args.num_cases + 1):
                result = _complete_case(args.output_root, size, index) if args.resume else None
                try:
                    if result is None:
                        if not args.dry_run and (
                            session is None
                            or generated_in_session >= args.session_batch_size
                        ):
                            if session is not None:
                                try:
                                    session.quit()
                                except Exception:
                                    pass
                            session = SolidWorksSession(visible=args.visible)
                            generated_in_session = 0
                        result = generate_case(
                            args.output_root,
                            size,
                            index,
                            args.seed + size * 100000 + index,
                            session=session,
                            dry_run=args.dry_run,
                        )
                        generated_in_session += 1
                except Exception as exc:
                    result = {
                        "case_id": f"group_{size}_{index:06d}",
                        "status": "failed",
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                    if session is not None:
                        try:
                            session.quit()
                        except Exception:
                            pass
                        session = None
                    generated_in_session = 0
                    if not args.continue_on_error:
                        results.append(result)
                        _checkpoint(output, results)
                        raise
                results.append(result)
                print(f"{result['case_id']}: {result['status']}", flush=True)
                _checkpoint(output, results)
    finally:
        if session is not None:
            try:
                session.quit()
            except Exception:
                pass
    _checkpoint(output, results)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
