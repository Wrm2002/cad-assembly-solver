from __future__ import annotations

from dataclasses import dataclass
import sys
import unittest
from pathlib import Path

import numpy as np
import trimesh


SW_DIR = Path(__file__).resolve().parents[1]
if str(SW_DIR) not in sys.path:
    sys.path.insert(0, str(SW_DIR))

from learned_joint.mesh_residuals import MeshContactResidualProvider  # noqa: E402


@dataclass
class _Factor:
    source: str
    target: str


class MeshResidualTests(unittest.TestCase):
    def test_contact_is_distinguished_from_penetration_and_separation(self):
        meshes = {
            "a": trimesh.creation.box(extents=[2, 2, 2]),
            "b": trimesh.creation.box(extents=[2, 2, 2]),
        }
        provider = MeshContactResidualProvider(meshes, sample_count=256, seed=11)
        factor = [_Factor("a", "b")]

        def poses(x):
            moving = np.eye(4)
            moving[0, 3] = x
            return {"a": np.eye(4), "b": moving}

        contact = provider.audit(poses(2.0), factor)["pair_scores"][0]
        overlap = provider.audit(poses(1.0), factor)["pair_scores"][0]
        separated = provider.audit(poses(4.0), factor)["pair_scores"][0]
        self.assertLess(contact["overlap"], overlap["overlap"])
        self.assertGreater(contact["contact"], separated["contact"])
        self.assertLess(contact["closest_distance"], separated["closest_distance"])
        self.assertLess(contact["contact_gap_normalized"], separated["contact_gap_normalized"])
        self.assertGreater(overlap["penetration_depth_normalized"], contact["penetration_depth_normalized"])


if __name__ == "__main__":
    unittest.main()
