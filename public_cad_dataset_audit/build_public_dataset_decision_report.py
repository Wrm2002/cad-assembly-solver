"""Combine empirical audits into one decision record."""

from __future__ import annotations

import argparse
from pathlib import Path

from fusion360_common import load_json, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", default="outputs")
    parser.add_argument(
        "--output",
        default="outputs/public_dataset_decision_report.json",
    )
    args = parser.parse_args()
    root = Path(args.audit_dir)
    required = {
        "fusion360": root / "fusion360_audit_report.json",
        "automate": root / "automate_audit_report.json",
        "linkify": root / "linkify_audit_report.json",
        "pairs": root / "pair_dataset_manifest.json",
        "conversion": (
            root / "fusion360_assembly_graphs"
            / "conversion_manifest.json"
        ),
    }
    missing = [
        name for name, path in required.items()
        if not path.is_file()
    ]
    if missing:
        report = {
            "schema_version": "1.0.0",
            "status": "failed",
            "failure_reasons": [
                f"required_report_missing:{name}" for name in missing
            ],
            "unavailable_fields": ["dataset_decision"],
        }
        write_json(Path(args.output), report)
        return 2
    data = {name: load_json(path) for name, path in required.items()}
    fusion = data["fusion360"]
    automate = data["automate"]
    linkify = data["linkify"]
    pairs = data["pairs"]
    conversion = data["conversion"]
    acceptance = {
        "fusion360_suitability_decided": (
            fusion["suitability"]["verdict"]
            == "suitable_as_primary_source_with_contact_caveat"
        ),
        "automate_suitability_decided": (
            automate["audit_status"] == "success"
        ),
        "ten_fusion_assemblies_converted": (
            conversion["converted_count"] >= 10
        ),
        "positive_and_negative_pairs_generated": (
            pairs["positive_count"] > 0
            and pairs["negative_count"] > 0
        ),
        "no_model_training_performed": True,
        "private_solidworks_not_processed": True,
        "synthetic_cad_not_generated": True,
    }
    report = {
        "schema_version": "1.0.0",
        "status": (
            "accepted" if all(acceptance.values()) else "partial"
        ),
        "recommended_primary_data_source": (
            "Autodesk Fusion 360 Gallery Assembly Dataset"
        ),
        "recommendation": {
            "fusion360": (
                "Use occurrence-body nodes, joints as high-confidence "
                "positive edges, and contacts as a separately quality-tagged "
                "edge source. Prefer native SMT or indexed OBJ when topology "
                "identity matters; do not assume STEP face numbers match."
            ),
            "automate": (
                "Use as a large auxiliary source for mate-type and positive "
                "part-pair tasks. It is not the primary source for direct "
                "interface-face labels."
            ),
            "linkify": (
                "Public and highly relevant for corrected interfaces, but "
                "full local use is blocked by storage and an explicit "
                "repository/dataset license was not found."
            ),
            "negative_edges": (
                "Treat same-assembly non-edges as weak closed-world "
                "negatives; retain this provenance and avoid calling them "
                "mechanically incompatible."
            ),
        },
        "empirical_evidence": {
            "fusion360_sample_assembly_count": fusion[
                "audit_scope"
            ]["assembly_files_audited"],
            "fusion360_part_instances": fusion[
                "observed_counts"
            ]["part_instances_visible"],
            "fusion360_relation_count": fusion[
                "observed_counts"
            ]["relations_extracted"],
            "fusion360_part_pair_mapping_rate": fusion[
                "mapping_quality"
            ]["joint_contact_to_part_pair_rate"],
            "fusion360_obj_topology_verification_rate": fusion[
                "mapping_quality"
            ]["indexed_obj_interface_verification_rate"],
            "automate_assembly_rows": automate[
                "observed_counts"
            ]["assembly_rows"],
            "automate_part_rows": automate[
                "observed_counts"
            ]["part_rows"],
            "automate_mate_rows": automate[
                "observed_counts"
            ]["mate_rows"],
            "automate_unique_positive_part_pairs": automate[
                "observed_counts"
            ]["unique_positive_part_pair_count"],
            "converted_fusion360_assemblies": conversion[
                "converted_count"
            ],
            "pair_samples": pairs["sample_count"],
            "positive_pair_samples": pairs["positive_count"],
            "negative_pair_samples": pairs["negative_count"],
        },
        "acceptance": acceptance,
        "failure_reasons": [
            "Fusion original contacts are not guaranteed complete or correct.",
            "AutoMate parquet lacks direct interface face/edge ids.",
            "Linkify full data was not downloaded because peak storage exceeds current capacity.",
            "Linkify repository does not currently expose an explicit license file.",
        ],
        "unavailable_fields": sorted(set(
            fusion.get("unavailable_fields", [])
            + automate.get("unavailable_fields", [])
            + linkify.get("unavailable_fields", [])
        )),
    }
    write_json(Path(args.output), report)
    print(
        f"Decision: {report['recommended_primary_data_source']} "
        f"({report['status']})"
    )
    return 0 if report["status"] == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
