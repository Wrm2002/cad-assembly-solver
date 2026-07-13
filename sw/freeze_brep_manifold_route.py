"""Freeze the generic three-stage route before opening SolidWorks exams."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE_FILES = (
    ROOT / "sw" / "learned_joint" / "fusion_contract.py",
    ROOT / "sw" / "learned_joint" / "hypothesis.py",
    ROOT / "sw" / "learned_joint" / "manifold_solver.py",
    ROOT / "sw" / "learned_joint" / "mesh_residuals.py",
    ROOT / "sw" / "learned_joint" / "report_adapter.py",
    ROOT / "sw" / "joinable_e2e.py",
)
FORBIDDEN_PRODUCTION_TOKENS = (
    "keyslotfactor",
    "flangefactor",
    "flange_part_a",
    "flange_part_b",
    "shaft_with_keyway",
    "fan-cage-module",
    "case_specific_override = true",
)


def main() -> int:
    audit_path = ROOT / "reports" / "pure_brep_contract_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    token_hits = []
    hashes = {}
    for path in CORE_FILES:
        content = path.read_bytes()
        text = content.decode("utf-8").lower()
        hashes[str(path.relative_to(ROOT))] = sha256(content).hexdigest()
        for token in FORBIDDEN_PRODUCTION_TOKENS:
            if token in text:
                token_hits.append({"file": str(path.relative_to(ROOT)), "token": token})
    passed = bool(audit.get("passed")) and not token_hits
    result = {
        "schema_version": "three_stage_brep_manifold_freeze.v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_before_case_exam": True,
        "passed": passed,
        "pure_brep_contract": {
            "record_count": audit.get("record_count"),
            "joint_supervision_count": audit.get("joint_supervision_count"),
            "forbidden_model_key_count": audit.get("forbidden_model_key_count"),
            "split_group_overlap_count": audit.get("split_group_overlap_count"),
        },
        "core_sha256": hashes,
        "forbidden_production_token_hits": token_hits,
        "frozen_solver_policy": {
            "learned_signal": "JoinABLe top-k B-Rep entity-pair logits",
            "pair_representation": "local interface frame + free-DOF manifold + symmetry class",
            "global_optimization": "bounded discrete topology/candidate search + scipy least_squares on SE(3)",
            "geometry_terms": "projected manifold residual + sampled SDF contact/penetration + OCCT top-N",
            "named_mechanical_factors": False,
            "file_or_case_features": False,
            "fixed_pair_transform_decode": False,
            "automatic_semantic_acceptance": False,
        },
    }
    output = ROOT / "reports" / "three_stage_brep_manifold_freeze.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": passed, "output": str(output), "hash_count": len(hashes)}, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
