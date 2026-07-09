"""
compute_manifest.py — Universal assembly manifest generator.

Four-phase architecture:
  1. features.py          — geometry extraction (cylinders, planes, bbox)
  2. constraints.py       — feature-space matching (who pairs with whom)
  3. coordinate_solver.py — global-frame solving (how to place)
  4. refinement.py        — strategy-based fine-tuning

Optional --decompose mode:
  Decomposes multi-solid STEP files into sub-parts before matching,
  uses cross-group prescreening to avoid combinatorial explosion.

Zero hard-coded strategies — works for any mechanical parts.

Usage:
  python compute_manifest.py <folder_path>
  python compute_manifest.py <folder_path> --decompose
Output: <folder_path>/assembly_manifest.json
"""

import argparse
import json, math, os, sys, copy

from features import extract_features
from constraints import match_features, COAXIAL, CLEARANCE, PLANAR_MATE
from coordinate_solver import solve_in_global_frame, placements_to_manifest
from refinement import refine_placements


def _print_matches(matches):
    for m in matches:
        a, b = m['parts']
        extra = ""
        if m['type'] == COAXIAL:
            extra = f" radius_match={m['radius_match']:.2f}"
        elif m['type'] == CLEARANCE:
            extra = f" gap={m['gap']:.2f}"
        elif m['type'] in (PLANAR_MATE, 'planar_align'):
            extra = f" distance={m['distance']:.2f}"
        print(f"  {m['type']:12s}: {a} <-> {b}{extra}")


def _print_placements(components, label):
    print(f"\n{label}:")
    for c in components:
        rot = c['placement'].get('rotate_sequence', [])
        trans = c['placement'].get('translate', [0, 0, 0])
        print(f"  [{c['id']}] {c['source']}: translate={trans}, rotations={len(rot)}")


def _run_standard(folder, step_files, solver="bfs", enable_scoring=False,
                  enable_pruning=False,
                  beam_width=20, min_score=0.5, max_neighbors=4):
    """Standard pipeline: each STEP file = one part."""
    parts = {}
    for f in step_files:
        fp = os.path.join(folder, f)
        parts[f] = extract_features(fp)
        nc = len(parts[f]['cylinders'])
        np = len(parts[f]['planes'])
        print(f"  {f}: {nc} cylinders, {np} planes")

    matches = match_features(parts)
    print(f"\nPhase 1 — Feature matches: {len(matches)}")
    _print_matches(matches)

    if solver == "reliable" or enable_scoring or enable_pruning:
        from match_pruning import prune_match_graph, write_pruning_logs
        from match_scoring import score_matches
        matches = score_matches(matches, parts)
        matches.sort(key=lambda match: float(match["score"]), reverse=True)
        if solver == "reliable" or enable_pruning:
            matches, removed = prune_match_graph(
                matches, min_score=min_score, max_neighbors=max_neighbors
            )
            write_pruning_logs(folder, matches, removed)
            print(f"Scoring/pruning: kept={len(matches)}, removed={len(removed)}")
        else:
            print(f"Scoring: ranked={len(matches)}")

    if solver == "reliable":
        from small_assembly_solver import solve_small_assembly
        search = solve_small_assembly(parts, matches, beam_width=beam_width)
        components = search.pop("components")
        with open(os.path.join(folder, "search_report.json"), "w", encoding="utf-8") as f:
            json.dump(search, f, indent=2, ensure_ascii=False)
        _print_placements(components, "Phase 2 - Reliable search placements")
        print(f"Search: status={search['status']}, expanded={search['expanded_states']}")
        return components

    solved = solve_in_global_frame(parts, matches)
    components = placements_to_manifest(parts, solved)
    _print_placements(components, "Phase 2 — Global-frame placements")

    components = refine_placements(components, matches, parts, folder)
    _print_placements(components, "Phase 3 — After refinement")
    return components


def _run_decomposed(folder, step_files):
    """Decompose multi-solid files, prescreen, then match sub-parts."""
    from decompose_step import decompose_file
    from prescreen import prescreen_candidates

    decomp_dir = os.path.join(folder, '_decomposed')

    # ── Step 1: Decompose ──
    sub_parts_by_parent = {}
    for f in step_files:
        fp = os.path.join(folder, f)
        parent = os.path.splitext(f)[0]
        subs = decompose_file(fp, decomp_dir)
        sub_parts_by_parent[parent] = subs
        decomp_flag = "decomposed" if len(subs) > 1 else "single"
        print(f"  {f}: {len(subs)} sub-parts ({decomp_flag})")

    total = sum(len(v) for v in sub_parts_by_parent.values())
    cross_pairs = 0
    parents = list(sub_parts_by_parent.keys())
    for i in range(len(parents)):
        for j in range(i + 1, len(parents)):
            cross_pairs += len(sub_parts_by_parent[parents[i]]) * len(sub_parts_by_parent[parents[j]])
    print(f"  Total: {total} sub-parts, {cross_pairs} cross-parent pairs")

    # ── Step 2: Prescreen + force-include single-solid parents ──
    candidates = prescreen_candidates(sub_parts_by_parent)
    print(f"\nPrescreened candidates: {len(candidates)}")
    for sa, sb, score in candidates[:10]:
        print(f"  {score:.3f} | {sa['parent']}/sub_{sa['index']:03d}"
              f" <-> {sb['parent']}/sub_{sb['index']:03d}")

    # Single-solid parents can't be decomposed — their bbox may not
    # resemble any sub-part of the mating parent (e.g. DDR stick vs DIMM slot).
    # Force-include all cross-pairs involving single-solid parents.
    single_parents = [p for p, subs in sub_parts_by_parent.items() if len(subs) == 1]
    if single_parents:
        seen_pairs = set()
        for sa, sb, _ in candidates:
            seen_pairs.add((sa['parent'], sa['index'], sb['parent'], sb['index']))
        extra = 0
        for sp in single_parents:
            sp_sub = sub_parts_by_parent[sp][0]
            sp_sub['parent'] = sp
            for other_p, other_subs in sub_parts_by_parent.items():
                if other_p == sp:
                    continue
                for other_sub in other_subs:
                    other_sub['parent'] = other_p
                    key = (sp, sp_sub['index'], other_p, other_sub['index'])
                    rev = (other_p, other_sub['index'], sp, sp_sub['index'])
                    if key not in seen_pairs and rev not in seen_pairs:
                        seen_pairs.add(key)
                        # score not meaningful here — forced inclusion
                        candidates.append((sp_sub, other_sub, 0.0))
                        extra += 1
        if extra:
            print(f"  + {extra} forced pairs from single-solid parents")

    # Build reverse mapping: sub-part key → original parent filename
    sub_to_parent = {}
    for f in step_files:
        parent = os.path.splitext(f)[0]
        fp_abs = os.path.join(folder, f)
        for s in sub_parts_by_parent.get(parent, []):
            if os.path.abspath(s['path']) != fp_abs:
                sub_to_parent[os.path.relpath(s['path'], folder)] = f

    # ── Step 3: Per-pair feature extraction + matching ──
    # Match each candidate pair individually — no same-parent matches possible.
    feat_cache = {}
    all_matches = []
    for sa, sb, score in candidates:
        key_a = os.path.relpath(sa['path'], folder)
        key_b = os.path.relpath(sb['path'], folder)

        if key_a not in feat_cache:
            feat_cache[key_a] = extract_features(sa['path'])
            nc = len(feat_cache[key_a]['cylinders'])
            np = len(feat_cache[key_a]['planes'])
            print(f"  features: {key_a}: {nc}c/{np}p")
        if key_b not in feat_cache:
            feat_cache[key_b] = extract_features(sb['path'])
            nc = len(feat_cache[key_b]['cylinders'])
            np = len(feat_cache[key_b]['planes'])
            print(f"  features: {key_b}: {nc}c/{np}p")

        pair_parts = {key_a: feat_cache[key_a], key_b: feat_cache[key_b]}
        pair_matches = match_features(pair_parts)
        all_matches.extend(pair_matches)

    parts = feat_cache
    matches = all_matches

    # ── Step 4: Solve → Refine (same as standard) ──
    print(f"\nPhase 1 — Feature matches: {len(matches)} (cross-parent only)")
    _print_matches(matches)

    solved = solve_in_global_frame(parts, matches)
    components = placements_to_manifest(parts, solved)
    _print_placements(components, "Phase 2 — Global-frame placements")

    # Attach parent_source so build_assembly can optionally use full parent files
    for c in components:
        if c['source'] in sub_to_parent:
            c['parent_source'] = sub_to_parent[c['source']]

    components = refine_placements(components, matches, parts, folder)
    _print_placements(components, "Phase 3 — After refinement")
    return components


def generate_manifest(folder_path, decompose=False, write_diagnostics=False,
                      solver="bfs", enable_scoring=False, enable_pruning=False, beam_width=20,
                      min_score=0.5, max_neighbors=4):
    """Main pipeline: discover STEP files, match, solve, refine, output."""
    folder = os.path.abspath(folder_path)
    if not os.path.isdir(folder):
        raise NotADirectoryError(folder)

    step_files = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(('.step', '.stp'))
        and not f.lower().startswith('assembly')
    ])

    if not step_files:
        raise FileNotFoundError(f"No STEP files in {folder}")

    print(f"Folder: {folder}")
    print(f"Parts: {len(step_files)}")
    if decompose:
        print("Mode: decompose (multi-solid → sub-parts)")
    print()

    if decompose and solver != "bfs":
        raise ValueError("--decompose currently supports --solver bfs only")
    if decompose:
        components = _run_decomposed(folder, step_files)
    else:
        components = _run_standard(
            folder, step_files, solver, enable_scoring, enable_pruning,
            beam_width, min_score, max_neighbors
        )

    # ── Build manifest ──
    manifest = {
        "__description": f"Auto-generated assembly manifest for {os.path.basename(folder)}",
        "assembly_name": os.path.basename(folder),
        "global_units": "mm",
        "components": components
    }

    out_path = os.path.join(folder, 'assembly_manifest.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nManifest: {out_path}")
    if write_diagnostics:
        from diagnostics import write_diagnostics as _write_diagnostics

        diagnostics_path, report_path = _write_diagnostics(folder)
        print(f"Diagnostics: {diagnostics_path}")
        print(f"Report: {report_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("--decompose", action="store_true")
    parser.add_argument("--write-diagnostics", action="store_true")
    parser.add_argument("--solver", choices=["bfs", "reliable"], default="bfs")
    parser.add_argument("--enable-scoring", action="store_true")
    parser.add_argument("--enable-pruning", action="store_true")
    parser.add_argument("--beam-width", type=int, default=20)
    parser.add_argument("--min-score", type=float, default=0.5)
    parser.add_argument("--max-neighbors", type=int, default=4)
    args = parser.parse_args()
    generate_manifest(
        args.folder,
        decompose=args.decompose,
        write_diagnostics=args.write_diagnostics,
        solver=args.solver,
        enable_scoring=args.enable_scoring,
        enable_pruning=args.enable_pruning,
        beam_width=args.beam_width,
        min_score=args.min_score,
        max_neighbors=args.max_neighbors,
    )
