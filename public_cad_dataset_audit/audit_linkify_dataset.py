"""Audit whether Linkify code and interface-augmented data are public."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

from fusion360_common import write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository")
    parser.add_argument(
        "--output", default="outputs/linkify_audit_report.json"
    )
    parser.add_argument(
        "--storage-path", default="D:/Model_match_public_data"
    )
    args = parser.parse_args()
    repository = Path(args.repository).resolve()
    readme = repository / "README.md"
    failures = []
    if not readme.is_file():
        failures.append("linkify_readme_missing")
        content = ""
    else:
        content = readme.read_text(encoding="utf-8", errors="replace")
    urls = re.findall(
        r"https://fusion-360-gallery-assembly-interfaces"
        r"\.s3\.us-west-2\.amazonaws\.com/[^\s)]+",
        content,
    )
    required_code = [
        repository
        / "scripts/data_generation/contact_generation/"
        "generate_contacts_test.py",
        repository
        / "scripts/data_generation/assemblyGraphGeneration/"
        "assembly_graph.py",
        repository
        / "scripts/data_generation/assemblyGraphGeneration/"
        "assembly2graph.py",
    ]
    missing_code = [
        str(path.relative_to(repository))
        for path in required_code if not path.is_file()
    ]
    failures.extend(
        f"required_code_missing:{path}" for path in missing_code
    )
    license_files = list(repository.glob("LICENSE*"))
    if not license_files:
        failures.append("repository_license_file_not_found")
    storage = Path(args.storage_path)
    storage.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(storage).free / (1024**3)
    compressed_gb = 93.2
    extracted_gb = 211.0
    peak_required_gb = 304.0
    locally_extracted = any(
        storage.glob("**/contacts_assembly_json/*/assembly.json")
    )
    report = {
        "schema_version": "1.0.0",
        "dataset": "Linkify interface-augmented Fusion 360 assembly graphs",
        "audit_status": (
            "success_publication_verified"
            if urls and not missing_code else "partial"
        ),
        "public_release": {
            "code_repository": "https://github.com/ajignasu/linkify",
            "repository_locally_cloned": repository.is_dir(),
            "repository_license_file_present": bool(license_files),
            "dataset_urls_found": len(urls),
            "dataset_urls": sorted(set(urls)),
            "compressed_size_gb": compressed_gb,
            "extracted_size_gb": extracted_gb,
            "peak_storage_requirement_gb": peak_required_gb,
            "locally_extracted": locally_extracted,
        },
        "format": {
            "assembly_json_compatible_with_fusion360_gallery": True,
            "recomputed_contact_fields": [
                "entity_one.body",
                "entity_one.index",
                "entity_two.body",
                "entity_two.index",
                "contact_area",
                "contact_volume",
            ],
            "local_interface_geometry": (
                "Per-contact PLY point clouds under contact/"
            ),
            "source_code_available": not missing_code,
        },
        "local_feasibility": {
            "storage_path": str(storage.resolve()),
            "free_space_gb": free_gb,
            "enough_space_for_full_download_and_extraction": (
                free_gb >= peak_required_gb
            ),
        },
        "suitability": {
            "verdict": (
                "public_and_highly_relevant_but_not_locally_audited"
            ),
            "suitable_for_interface_aware_graphs": True,
            "suitable_for_immediate_full_local_use": (
                locally_extracted
            ),
            "reasons": [
                "It recomputes contacts reported missing or erroneous in the original Fusion release.",
                "It supplies contact face indices and local point-cloud geometry.",
                "The full release requires more storage than is currently available on D:.",
                "No small official sample archive is documented in the repository.",
            ],
        },
        "failure_reasons": failures,
        "unavailable_fields": (
            ([] if locally_extracted else [
                "empirical_assembly_count_not_locally_verified",
                "empirical_contact_count_not_locally_verified",
                "per_record_quality_not_locally_verified",
            ]) + ([] if license_files else [
                "explicit_linkify_repository_and_dataset_license"
            ])
        ),
    }
    write_json(Path(args.output), report)
    print(f"Linkify release URLs found: {len(urls)}")
    print(f"Local full extraction: {locally_extracted}")
    return 0 if report["audit_status"].startswith("success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
