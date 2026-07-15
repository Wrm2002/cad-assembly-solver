from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from known_group_assembly import (  # noqa: E402
    _carrier_open_side_consistent_candidates,
    _carrier_thin_side,
)


def _placement(z: float) -> dict:
    return {"translate": [0.0, 0.0, float(z)]}


class CarrierOpenSideConsistencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.features = {
            "carrier.stp": {
                "obb": {
                    "center": [0.0, 0.0, 0.0],
                    "axes": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    "dimensions": [100.0, 80.0, 4.0],
                }
            },
            "slot_leaf.stp": {
                "obb": {
                    "center": [0.0, 0.0, 0.0],
                    "axes": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    "dimensions": [8.0, 20.0, 1.0],
                }
            },
            "footprint_leaf.stp": {
                "obb": {
                    "center": [0.0, 0.0, 0.0],
                    "axes": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    "dimensions": [20.0, 20.0, 2.0],
                }
            },
        }
        self.edge_placements = {
            "carrier.stp": _placement(0.0),
            "slot_leaf.stp": _placement(8.0),
            "footprint_leaf.stp": _placement(0.0),
        }
        self.edge_candidate = {
            "placements": self.edge_placements,
            "total_score": 3.0,
            "edge_slot_interface": {
                "connection_id": "edge-1",
                "stationary_part": "carrier.stp",
                "movable_part": "slot_leaf.stp",
                "insertion_axis": [0.0, 0.0, 1.0],
            },
            "edge_slot_history": [{
                "connection_id": "edge-1",
                "stationary_part": "carrier.stp",
                "movable_part": "slot_leaf.stp",
                "insertion_axis": [0.0, 0.0, 1.0],
            }],
        }

    def _footprint_candidate(self, z: float, polarity: int) -> dict:
        return {
            "placements": {
                "carrier.stp": _placement(0.0),
                "slot_leaf.stp": _placement(0.0),
                "footprint_leaf.stp": _placement(z),
            },
            "total_score": 2.0,
            "planar_footprint": {
                "connection_id": "footprint-1",
                "stationary_part": "carrier.stp",
                "movable_part": "footprint_leaf.stp",
                "support_polarity": polarity,
                "normal_sign": 1,
            },
        }

    def test_composes_only_same_side_and_preserves_slot_leaf(self) -> None:
        candidates = _carrier_open_side_consistent_candidates(
            [self.edge_candidate],
            [self._footprint_candidate(-7.0, -1), self._footprint_candidate(7.0, 1)],
            self.features,
            max_candidates=2,
        )

        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        detail = candidate["carrier_open_side_consistency"]
        self.assertTrue(detail["supported"])
        self.assertEqual(detail["reference_side_sign"], detail["target_side_sign"])
        self.assertEqual(candidate["placements"]["slot_leaf.stp"], self.edge_placements["slot_leaf.stp"])
        self.assertEqual(candidate["placements"]["footprint_leaf.stp"], _placement(7.0))
        self.assertTrue(candidate["proposal_only"])
        self.assertTrue(candidate["review_required"])
        self.assertFalse(candidate["can_auto_accept"])

    def test_midplane_reference_abstains(self) -> None:
        placements = dict(self.edge_placements)
        placements["slot_leaf.stp"] = _placement(0.1)
        side = _carrier_thin_side(
            self.features,
            placements,
            carrier="carrier.stp",
            moving="slot_leaf.stp",
        )
        self.assertIsNotNone(side)
        self.assertEqual(side["side_sign"], 0)
        ambiguous_edge = {
            **self.edge_candidate,
            "placements": placements,
            "edge_slot_interface": {
                key: value
                for key, value in self.edge_candidate["edge_slot_interface"].items()
                if key != "insertion_axis"
            },
            "edge_slot_history": [{
                key: value
                for key, value in self.edge_candidate["edge_slot_history"][0].items()
                if key != "insertion_axis"
            }],
        }
        candidates = _carrier_open_side_consistent_candidates(
            [ambiguous_edge],
            [self._footprint_candidate(7.0, 1)],
            self.features,
            max_candidates=2,
        )
        self.assertEqual(candidates, [])


if __name__ == "__main__":
    unittest.main()
