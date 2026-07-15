"""Discover exterior openings that can carry the EAR cross-section.

This is deliberately an *opening-first* recall stage.  It does not use EAR
or chassis hole correspondence to generate placements.  A read-only triangle
proxy of the original chassis is ray-tested from the exterior along the
proposed insertion direction.  A rectangular EAR envelope is a conservative
necessary condition: an opening may only be proposed when every envelope
sample is clear to the required insertion depth.  Passing openings are merely
handed to the later exact-profile, flange/hole, and OCCT solid-collision gates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def first_hits_y(mesh, origins: np.ndarray) -> np.ndarray:
    """Return the first +Y hit for every exterior ray (or inf if no hit)."""
    directions = np.tile([0.0, 1.0, 0.0], (len(origins), 1))
    locations, index_ray, _ = mesh.ray.intersects_location(
        ray_origins=origins, ray_directions=directions, multiple_hits=True
    )
    first = np.full(len(origins), np.inf)
    if len(locations):
        np.minimum.at(first, index_ray, locations[:, 1])
    return first


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    out = args.folder

    import trimesh

    mesh = trimesh.load(
        out / "opening_mesh_proxy" / "part_00_chassis_fixed.stl",
        force="mesh", process=False,
    )

    # Orientation is inferred upstream from the EAR's exterior I/O face and
    # mounting-flange normals.  For the left external face this yields +Y
    # insertion.  Only its dimensions are borrowed here; no hole-centre pose
    # is used as an opening proposal.
    seed = json.loads((out / "ear_folded_flange_candidates.json").read_text())["candidates"][0]
    seed_lo = np.asarray(seed["bbox_mm"]["min"], dtype=float)
    seed_hi = np.asarray(seed["bbox_mm"]["max"], dtype=float)
    # The oriented seed bbox is already in chassis coordinates: X/Z form the
    # entrance cross-section, and Y is the straight insertion depth.
    seed_extent = seed_hi - seed_lo
    width_x = seed_extent[0]
    height_z = seed_extent[2]
    required_depth = seed_extent[1]

    # The EAR-L semantic side prior limits recall to the chassis' left exterior
    # strip.  Positions on that strip are scanned independently of all holes.
    x_centres = np.arange(-210.0, -194.9, 5.0)
    z_centres = np.arange(-20.0 + height_z / 2.0, 776.0 - height_z / 2.0, 10.0)
    depth_samples = np.arange(0.25, required_depth - 0.25, 5.0)
    sample_x = np.linspace(-width_x / 2.0 + 1.5, width_x / 2.0 - 1.5, 6)
    sample_z = np.linspace(-height_z / 2.0 + 1.5, height_z / 2.0 - 1.5, 18)
    dx, dz = np.meshgrid(sample_x, sample_z)
    local = np.column_stack([dx.ravel(), dz.ravel()])
    n_samples = len(local)

    centres = np.array([(x, z) for x in x_centres for z in z_centres], dtype=float)
    origins = np.empty((len(centres) * n_samples, 3), dtype=float)
    origins[:, 0] = np.repeat(centres[:, 0], n_samples) + np.tile(local[:, 0], len(centres))
    origins[:, 1] = -25.0
    origins[:, 2] = np.repeat(centres[:, 1], n_samples) + np.tile(local[:, 1], len(centres))
    first = first_hits_y(mesh, origins).reshape(len(centres), n_samples)

    candidates = []
    score_grid = np.zeros((len(x_centres), len(z_centres)), dtype=float)
    for i, (centre_x, centre_z) in enumerate(centres):
        # A chassis hit at/before a depth blocks the material cross-section.
        clear = first[i, :, None] > (depth_samples[None, :] + 0.25)
        per_depth = clear.mean(axis=0)
        full_corridor = bool(np.all(per_depth == 1.0))
        max_free_depth = float(depth_samples[np.where(per_depth == 1.0)[0][-1]]) if np.any(per_depth == 1.0) else 0.0
        min_clearance = float(per_depth.min())
        score_grid[np.where(x_centres == centre_x)[0][0], np.where(z_centres == centre_z)[0][0]] = min_clearance
        candidates.append({
            "opening_id": f"LEFT_Y_{i:03d}",
            "entrance_center_mm": [round(float(centre_x), 3), 0.0, round(float(centre_z), 3)],
            "insertion_axis": "+Y",
            "conservative_required_cross_section_mm": {
                "width_x": round(float(width_x), 3), "height_z": round(float(height_z), 3),
            },
            "required_insertion_depth_mm": round(float(required_depth), 3),
            "minimum_clear_fraction": round(min_clearance, 4),
            "max_fully_clear_depth_mm": round(max_free_depth, 3),
            "envelope_corridor_passed": full_corridor,
        })

    passed = [c for c in candidates if c["envelope_corridor_passed"]]
    best = sorted(candidates, key=lambda c: (c["minimum_clear_fraction"], c["max_fully_clear_depth_mm"]), reverse=True)[:20]
    result = {
        "method": "read_only_mesh_exterior_opening_recall_before_holes",
        "proxy": "original_chassis_STEP_tessellation_only",
        "semantic_side_prior": "EAR-L left exterior strip",
        "insertion_axis": "+Y",
        "hole_arrays_used_for_discovery": False,
        "grid": {"x_centres_mm": x_centres.tolist(), "z_step_mm": 10.0, "sample_count_per_cross_section": n_samples},
        "conservative_envelope": {"width_x_mm": float(width_x), "height_z_mm": float(height_z), "required_depth_mm": float(required_depth)},
        "opening_count_passing_envelope": len(passed),
        "passing_openings": passed,
        "best_nonpassing_openings": best,
        "next_gate": (
            "exact EAR profile cross-section -> flange/stop/hole locking -> original B-Rep OCCT collision"
            if passed else
            "no opening proposal may enter hole locking; broaden exterior direction/orientation recall or request CAD metadata"
        ),
        "status": "opening_candidates_ready_for_exact_profile" if passed else "no_conservative_opening_found",
    }
    (out / "ear_opening_discovery.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4.8))
    im = ax.imshow(score_grid.T, origin="lower", aspect="auto", vmin=0, vmax=1,
                   extent=[x_centres[0] - 2.5, x_centres[-1] + 2.5, z_centres[0] - 5, z_centres[-1] + 5], cmap="viridis")
    ax.contour(x_centres, z_centres, score_grid.T, levels=[0.999], colors=["#ff5a36"], linewidths=1.5)
    ax.set(title="EAR conservative cross-section clearance through left exterior (+Y insertion)",
           xlabel="chassis X (mm)", ylabel="chassis Z (mm)")
    fig.colorbar(im, ax=ax, label="minimum clear sample fraction across insertion depth")
    fig.tight_layout()
    fig.savefig(out / "ear_opening_discovery_map.png", dpi=180)
    print(json.dumps({"passing_openings": len(passed), "best": best[:3], "status": result["status"]}, indent=2))


if __name__ == "__main__":
    main()
