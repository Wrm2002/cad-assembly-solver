"""Isolated detailed candidate indexing for the real-world mixed pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from validate_real_cases import build_index_from_features


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pool")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    pool = Path(args.pool).resolve()
    feature_files = sorted((pool / "index" / "parts").glob("*.json"))
    features = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in feature_files
    ]
    config = json.loads(
        Path(args.config).resolve().read_text(encoding="utf-8")
    )
    outputs = build_index_from_features(pool, features, config)
    print(
        json.dumps(
            {
                "parts": len(outputs["part_features.json"]),
                "screened_pairs": outputs["screening_audit.json"][
                    "accepted_pairs"
                ],
                "generated_candidates": len(
                    outputs["geometry_candidates.json"]
                ),
                "kept_candidates": len(
                    outputs["pruned_candidates.json"]
                ),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
