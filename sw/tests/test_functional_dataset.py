import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_functional_pools import build_pools
from constraints import match_features
from functional_dataset_generator import (
    FAMILIES,
    generate_dataset,
    validate_metadata,
)
from sw_dataset_generator.templates.library import build_case_spec


class FunctionalDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.dataset = Path(cls.temporary.name) / "dataset"
        cls.pools = Path(cls.temporary.name) / "pools"
        generate_dataset(
            cls.dataset, variants_per_family=3, render=False
        )
        build_pools(cls.dataset, cls.pools)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_three_functional_families_emit_valid_metadata(self):
        found = set()
        for metadata_path in self.dataset.glob("*/metadata.json"):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            found.add(metadata["assembly_family"])
            self.assertEqual(
                validate_metadata(metadata, metadata_path.parent), []
            )
            self.assertEqual(metadata["truth_basis"], "functional_validity")
            self.assertFalse(metadata["source_id_is_production_truth"])
            self.assertEqual(len(metadata["negative_groups"]), 3)
            self.assertTrue(
                all(part["part_role"] for part in metadata["parts"])
            )
            self.assertTrue(
                all(
                    len(mate["independent_evidence"]) >= 2
                    for mate in metadata["functional_mates"]
                )
            )
        self.assertEqual(found, set(FAMILIES))

    def test_mixed_pool_truth_is_functional_and_split_is_disjoint(self):
        manifest = json.loads(
            (self.pools / "mixed_pool_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(manifest["truth_basis"], "functional_validity")
        self.assertFalse(manifest["source_id_is_production_truth"])
        self.assertTrue(
            all(
                not row["source_case_overlap_with_other_splits"]
                for row in manifest["pools"]
            )
        )
        for pool in self.pools.glob("functional_pool_*"):
            gt = json.loads(
                (pool / "pool_gt.json").read_text(encoding="utf-8")
            )
            self.assertTrue(
                all(
                    group["truth_basis"] == "functional_validity"
                    for group in gt["true_groups"]
                )
            )

    def test_legacy_generator_is_never_a_functional_positive(self):
        spec = build_case_spec(4, 1234, family="cover_base")
        self.assertEqual(spec["dataset_intended_use"], "geometry_smoke_only")
        self.assertFalse(spec["functional_positive_eligible"])

    def test_small_component_localization_recalls_key_faces(self):
        parts = {
            "large": {
                "filepath": "large.step",
                "bbox": {"min": [-30, -30, 0], "max": [30, 30, 70]},
                "cylinders": [],
                "planes": [
                    {
                        "area": 100.0,
                        "normal": [1, 0, 0],
                        "position": [10, 0, 30],
                    }
                ],
                "cones": [],
                "torii": [],
                "spheres": [],
            },
            "key": {
                "filepath": "key.step",
                "bbox": {"min": [-3, -3, -9], "max": [3, 3, 9]},
                "cylinders": [],
                "planes": [
                    {
                        "area": 108.0,
                        "normal": [1, 0, 0],
                        "position": [3, 0, 0],
                    }
                ],
                "cones": [],
                "torii": [],
                "spheres": [],
            },
        }
        matches = match_features(
            parts,
            {
                "minimum_plane_area_mm2": 250.0,
                "minimum_local_plane_area_mm2": 20.0,
                "local_component_diagonal_mm": 40.0,
                "minimum_plane_area_ratio": 0.1,
            },
        )
        planar = [row for row in matches if row["type"] == "planar_mate"]
        self.assertTrue(planar)
        self.assertEqual(
            planar[0]["candidate_origin"],
            "localized_small_component_planar",
        )


if __name__ == "__main__":
    unittest.main()
