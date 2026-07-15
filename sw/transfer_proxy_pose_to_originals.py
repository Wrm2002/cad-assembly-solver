"""Transfer a proxy assembly pose to unmodified source STEP components.

Only the rigid placement is copied.  The source STEP geometry is never
rewritten.  The emitted manifest is explicitly review-only because proxy
collision/contact success is not evidence that the original B-Rep is valid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def transfer(proxy_manifest: Path, original_dir: Path, output_manifest: Path) -> dict[str, Any]:
    manifest = json.loads(proxy_manifest.read_text(encoding="utf-8"))
    originals = {
        path.name: path
        for path in [*original_dir.glob("*.step"), *original_dir.glob("*.stp")]
        if path.stem.lower() != "assembly"
    }
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    audit_components: list[dict[str, Any]] = []
    for component in manifest.get("components", []):
        proxy_name = Path(component["source"]).name
        source = originals.get(proxy_name)
        if source is None:
            raise FileNotFoundError(f"no matching original component for proxy {proxy_name}")
        component["source"] = str(Path(__import__("os").path.relpath(source, output_manifest.parent)).as_posix())
        audit_components.append(
            {
                "proxy_component": proxy_name,
                "original_source": str(source.resolve()),
                "original_sha256_before": _sha256(source),
                "placement_copied": component.get("placement", {}),
                "original_sha256_after": _sha256(source),
            }
        )
    manifest["assembly_name"] = f"{manifest.get('assembly_name', 'proxy')}_original_geometry_review"
    manifest["proxy_pose_transfer"] = {
        "status": "review_only",
        "reason": "Rigid pose proposed by low-complexity interface proxy; source STEP topology remains unmodified.",
    }
    output_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    audit = {
        "schema": "proxy_pose_transfer/v1",
        "proxy_manifest": str(proxy_manifest.resolve()),
        "output_manifest": str(output_manifest.resolve()),
        "decision": "review_only",
        "components": audit_components,
        "source_geometry_modified": False,
    }
    output_manifest.with_name("proxy_pose_transfer_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("proxy_manifest", type=Path)
    parser.add_argument("original_dir", type=Path)
    parser.add_argument("output_manifest", type=Path)
    args = parser.parse_args()
    audit = transfer(args.proxy_manifest, args.original_dir, args.output_manifest)
    print(json.dumps({"decision": audit["decision"], "components": len(audit["components"])}, indent=2))


if __name__ == "__main__":
    main()
