"""Run the generic three-stage B-Rep/manifold/global-pose pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for root in (PROJECT_ROOT, PROJECT_ROOT / "sw"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from learned_joint import solve_manifold_pose_graph  # noqa: E402
from learned_joint.mesh_residuals import MeshContactResidualProvider  # noqa: E402
from learned_joint.report_adapter import build_manifold_pools  # noqa: E402


def step_to_stl(step_path: Path, output_dir: Path) -> Path:
    """Tessellate without importing the Torch/JoinABLe inference runtime."""

    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.StlAPI import StlAPI_Writer

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{step_path.stem}.stl"
    if output.exists() and output.stat().st_mtime_ns >= step_path.stat().st_mtime_ns:
        return output
    reader = STEPControl_Reader()
    if reader.ReadFile(str(step_path)) != IFSelect_RetDone:
        raise RuntimeError(f"STEP read failed: {step_path}")
    reader.TransferRoots()
    shape = reader.OneShape()
    mesh = BRepMesh_IncrementalMesh(shape, 0.2, False, 0.5)
    mesh.Perform()
    writer = StlAPI_Writer()
    writer.SetASCIIMode(False)
    writer.Write(shape, str(output))
    return output


def _part(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--part requires PART_ID=STEP_PATH")
    part_id, raw_path = value.split("=", 1)
    path = Path(raw_path)
    if not part_id or not path.is_file():
        raise argparse.ArgumentTypeError("--part requires an existing STEP path")
    return part_id, path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", action="append", required=True, type=_part)
    parser.add_argument("--pair-frontier-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-pair-candidates", type=int, default=4)
    parser.add_argument(
        "--max-compound-pair-candidates",
        type=int,
        default=0,
        help=(
            "Additive geometry-only multi-interface budget. Analytic baseline "
            "and learned sidecar budgets remain protected."
        ),
    )
    parser.add_argument(
        "--max-learned-pair-candidates", type=int, default=0,
        help="Additive learned sidecar budget; baseline candidates remain protected.",
    )
    parser.add_argument("--max-topologies", type=int, default=6)
    parser.add_argument("--max-hypotheses", type=int, default=24)
    parser.add_argument("--max-nfev", type=int, default=120)
    parser.add_argument(
        "--closure-enumeration-limit",
        type=int,
        default=50000,
        help=(
            "Maximum small discrete product ranked by multi-edge closure "
            "before the fixed hypothesis budget is optimized."
        ),
    )
    parser.add_argument("--translation-scale-mm", type=float, default=2.0)
    parser.add_argument("--rotation-scale-degrees", type=float, default=5.0)
    parser.add_argument("--sdf-samples", type=int, default=384)
    parser.add_argument("--selected-overlap-weight", type=float, default=16.0)
    parser.add_argument("--selected-contact-weight", type=float, default=2.0)
    parser.add_argument("--selected-distance-weight", type=float, default=0.5)
    parser.add_argument("--nonedge-overlap-weight", type=float, default=25.0)
    parser.add_argument("--local-patch-gap-weight", type=float, default=0.0)
    parser.add_argument("--local-patch-normal-weight", type=float, default=0.0)
    parser.add_argument("--exact-top-n", type=int, default=8)
    parser.add_argument(
        "--learned-pose-prior-weight", type=float, default=0.0,
        help="Optional soft weight for learned full-Pose candidates; zero preserves prior behaviour.",
    )
    parser.add_argument("--no-mesh-residuals", action="store_true")
    args = parser.parse_args()
    if (
        args.max_pair_candidates < 0
        or args.max_compound_pair_candidates < 0
        or args.max_learned_pair_candidates < 0
    ):
        parser.error("candidate budgets must be non-negative")
    if (
        args.max_pair_candidates
        + args.max_compound_pair_candidates
        + args.max_learned_pair_candidates
        < 1
    ):
        parser.error("at least one candidate channel must be enabled")

    part_sources = dict(args.part)
    if not 2 <= len(part_sources) <= 5:
        parser.error("known group must contain 2..5 unique parts")
    manifest = json.loads(args.pair_frontier_manifest.read_text(encoding="utf-8"))
    records = [
        row for row in manifest.get("records") or []
        if row.get("status") == "success"
        and row.get("source") in part_sources
        and row.get("target") in part_sources
    ]
    pools, pool_audit = build_manifold_pools(
        records,
        maximum_candidates_per_pair=args.max_pair_candidates,
        maximum_compound_candidates_per_pair=(
            args.max_compound_pair_candidates
        ),
        maximum_learned_candidates_per_pair=args.max_learned_pair_candidates,
    )
    solver_candidate_limit = (
        args.max_pair_candidates
        + args.max_compound_pair_candidates
        + args.max_learned_pair_candidates
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    geometry_provider = None
    mesh_sources = {}
    if not args.no_mesh_residuals:
        mesh_dir = args.output.parent / "meshes"
        mesh_sources = {}
        for part, path in part_sources.items():
            sibling = path.with_suffix(".stl")
            mesh_sources[part] = (
                sibling if sibling.is_file() else step_to_stl(path, mesh_dir)
            )
        geometry_provider = MeshContactResidualProvider(
            mesh_sources,
            sample_count=args.sdf_samples,
            selected_overlap_weight=args.selected_overlap_weight,
            selected_contact_weight=args.selected_contact_weight,
            selected_distance_weight=args.selected_distance_weight,
            nonedge_overlap_weight=args.nonedge_overlap_weight,
            local_patch_gap_weight=args.local_patch_gap_weight,
            local_patch_normal_weight=args.local_patch_normal_weight,
        )
    validator = None
    result = solve_manifold_pose_graph(
        list(part_sources),
        pools,
        max_candidates_per_pair=solver_candidate_limit,
        max_topologies=args.max_topologies,
        max_hypotheses=args.max_hypotheses,
        translation_scale_mm=args.translation_scale_mm,
        rotation_scale_degrees=args.rotation_scale_degrees,
        max_nfev=args.max_nfev,
        translation_bound_mm=(
            max(10.0, geometry_provider.scale * 1.5)
            if geometry_provider is not None else 250.0
        ),
        rotation_bound_degrees=180.0,
        geometry_residual_provider=geometry_provider,
        learned_pose_prior_weight=args.learned_pose_prior_weight,
        exact_validator=validator,
        validate_top_n=0,
        closure_enumeration_limit=args.closure_enumeration_limit,
    )
    result["input_audit"] = {
        "part_sources": {key: str(path.resolve()) for key, path in part_sources.items()},
        "mesh_sources": {key: str(path.resolve()) for key, path in mesh_sources.items()},
        "pair_pool_audit": pool_audit,
        "model_and_solver_feature_policy": (
            "Only B-Rep topology/geometry, learned scores, local frames, free DOFs, "
            "sampled SDF and OCCT geometry reach inference. Paths and IDs are runner bookkeeping."
        ),
        "case_specific_override": False,
        "named_mechanical_factor": False,
        "learned_pose_prior_weight": args.learned_pose_prior_weight,
        "baseline_candidate_budget_per_pair": args.max_pair_candidates,
        "compound_candidate_budget_per_pair": (
            args.max_compound_pair_candidates
        ),
        "learned_sidecar_budget_per_pair": args.max_learned_pair_candidates,
        "baseline_protected": True,
        "compound_geometry_additive": True,
        "exact_validation_deferred": args.exact_top_n > 0,
        "closure_enumeration_limit": args.closure_enumeration_limit,
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": result["status"],
        "hypothesis_count": len(result.get("hypotheses") or []),
        "valid_exact_count": sum(
            row.get("exact_validation", {}).get("status") == "valid"
            for row in result.get("hypotheses") or []
        ),
        "output": str(args.output.resolve()),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
