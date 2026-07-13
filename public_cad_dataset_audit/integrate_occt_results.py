"""Integrate OCCT pose validation results into solidworks_assembly_plan.json.

Reads known_group_output from each case and produces final accepted/review/rejected edges.
"""
import json, sys
from pathlib import Path
from typing import Any

sys.path.insert(0, 'sw')

CASES = ['1', '2', '3', '4', '5']
SW_DIR = Path('sw')
OUT_DIR = Path('public_cad_dataset_audit/outputs/step34_solidworks_plan')
LABELS_DIR = SW_DIR / 'phase5_annotation_pack'

# Human labels (scoring only!)
def load_human_labels(case_id: str) -> dict:
    p = LABELS_DIR / f'case_{case_id}' / 'human_labels.json'
    if p.exists():
        return json.loads(p.read_text(encoding='utf-8'))
    return {}

def mat_identity():
    return [[1.0,0,0,0],[0,1.0,0,0],[0,0,1.0,0],[0,0,0,1.0]]

def read_occt_output(case_id: str) -> dict:
    """Read known_group_output for a case."""
    kg_dir = SW_DIR / case_id / 'known_group_output'
    rel_path = kg_dir / 'assembly_relations.json'
    pose_path = kg_dir / 'pose_validation.json'
    manifest_path = kg_dir / 'assembly_manifest.json'

    result = {'case_id': case_id, 'found': False}

    if rel_path.exists():
        rel = json.loads(rel_path.read_text(encoding='utf-8'))
        result['pose_status'] = rel.get('pose_status', 'unknown')
        result['parts'] = rel.get('parts', [])
        result['direct_connections'] = rel.get('direct_connections', [])
        result['found'] = True

    if pose_path.exists():
        pose = json.loads(pose_path.read_text(encoding='utf-8'))
        result['pose_audit'] = pose.get('pose_audit', [])
        if pose.get('pose_audit'):
            best = pose['pose_audit'][0]
            result['collision_status'] = best.get('collision_status', 'unknown')
            result['collision_count'] = best.get('collision_count', 0)

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        result['placements'] = {}
        for comp in manifest.get('components', []):
            label = comp.get('label', '')
            placement = comp.get('placement', {})
            translate = placement.get('translate', [0,0,0])
            # Build 4x4 from translate (simplified)
            T = mat_identity()
            for i in range(3):
                T[i][3] = translate[i] if i < len(translate) else 0.0
            result['placements'][label] = T

    return result


def build_final_plan(case_id: str, occt: dict) -> dict:
    """Build solidworks_assembly_plan.json from OCCT results."""
    parts = occt.get('parts', [])
    human = load_human_labels(case_id)
    true_pairs = set()
    for rel in human.get('pass_1_direct_relations', []):
        true_pairs.add(tuple(sorted(rel['parts'])))

    # Determine accepted edges from OCCT direct_connections
    accepted_edges = []
    review_edges = []
    rejected_edges = []
    unresolved = []

    pose_status = occt.get('pose_status', 'unknown')
    connections = occt.get('direct_connections', [])
    placements = occt.get('placements', {})
    collision_status = occt.get('collision_status', 'unknown')

    # Build edges from direct connections
    accepted_parts = set()
    for conn in connections:
        pair = conn['parts']
        pair_tuple = tuple(sorted(pair))

        edge = {
            'pair_id': f"case_{case_id}_conn_{hash(pair_tuple) & 0xffff:04x}",
            'parts': pair,
            'relation_type': conn.get('primary_relation_type', 'unknown'),
            'supporting_types': conn.get('supporting_relation_types', []),
            'constraint_closed': conn.get('constraint_closed_in_selected_pose', False),
            'relative_transform': conn.get('relative_transform_a_to_b'),
            'confidence': conn.get('confidence', 'low'),
            'review_required': conn.get('review_required', True),
        }

        # OCCT acceptance logic
        if (pose_status == 'valid'
            and collision_status == 'success'
            and conn.get('constraint_closed_in_selected_pose')
            and not conn.get('review_required', True)):
            edge['decision'] = 'accepted'
            edge['decision_reason'] = 'occt_pose_valid_collision_free_constraint_closed'
            accepted_edges.append(edge)
            accepted_parts.update(pair)
        elif pose_status == 'failed':
            edge['decision'] = 'rejected'
            edge['decision_reason'] = f'occt_pose_{pose_status}'
            rejected_edges.append(edge)
        else:
            edge['decision'] = 'review'
            edge['decision_reason'] = f'occt_pose_{pose_status}_needs_review'
            review_edges.append(edge)

    # Unresolved parts
    for p in parts:
        if p not in accepted_parts:
            unresolved.append(p)

    # Build plan
    plan = {
        'schema_version': '1.0',
        'case_id': f'sw_case_{case_id}',
        'input_parts': [
            {
                'part_id': f'part_{i:03d}',
                'file_path': str((SW_DIR / case_id / p).resolve()),
                'file_name': p,
                'unit': 'mm',
            }
            for i, p in enumerate(parts)
        ],
        'placements': [],
        'accepted_edges': [e for e in accepted_edges],
        'review_edges': [e for e in review_edges],
        'rejected_edges': [e for e in rejected_edges],
        'unresolved_parts': unresolved,
        'generation_policy': {
            'used_human_labels': False,
            'used_case_specific_rules': False,
            'used_filename_answer_hardcoding': False,
            'acceptance_mode': 'conservative',
            'occt_validation': True,
            'semantic_reranking_enabled': False,
        },
    }

    # Fill placements
    for i, p in enumerate(parts):
        T = placements.get(p, mat_identity())
        plan['placements'].append({
            'part_id': f'part_{i:03d}',
            'part_name': p,
            'transform_world_from_part': T,
            'fixed': (i == 0),
            'source': 'root_part' if i == 0 else 'occt_pose_solver',
        })

    return plan


def main():
    print("=" * 60)
    print("OCCT Integration Pipeline")
    print("=" * 60)

    summary = {'cases': {}}

    for case_id in CASES:
        print(f"\n--- Case {case_id} ---")
        occt = read_occt_output(case_id)

        if not occt['found']:
            print(f"  No OCCT results found (may still be running)")
            summary['cases'][case_id] = {'status': 'no_occt_results'}
            continue

        pose_status = occt.get('pose_status', '?')
        connections = occt.get('direct_connections', [])
        collision = occt.get('collision_status', '?')

        print(f"  pose={pose_status}, collision={collision}, connections={len(connections)}")

        plan = build_final_plan(case_id, occt)

        n_accepted = len(plan['accepted_edges'])
        n_review = len(plan['review_edges'])
        n_rejected = len(plan['rejected_edges'])
        n_unresolved = len(plan['unresolved_parts'])

        print(f"  accepted={n_accepted}, review={n_review}, rejected={n_rejected}, unresolved={n_unresolved}")

        # Save
        case_out = OUT_DIR / f'case_{case_id}'
        case_out.mkdir(parents=True, exist_ok=True)
        plan_path = case_out / 'solidworks_assembly_plan.json'
        plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

        # Score against human labels
        human = load_human_labels(case_id)
        true_pairs = set()
        for rel in human.get('pass_1_direct_relations', []):
            true_pairs.add(tuple(sorted(rel['parts'])))

        accepted_pairs = {tuple(sorted(e['parts'])) for e in plan['accepted_edges']}
        tp = len(accepted_pairs & true_pairs)
        fp = len(accepted_pairs - true_pairs)
        fn = len(true_pairs - accepted_pairs)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        print(f"  scoring: TP={tp}, FP={fp}, FN={fn}, P={precision:.2f}, R={recall:.2f}")

        summary['cases'][case_id] = {
            'pose_status': pose_status,
            'collision_status': collision,
            'connection_count': len(connections),
            'accepted': n_accepted,
            'review': n_review,
            'rejected': n_rejected,
            'unresolved': n_unresolved,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': precision,
            'recall': recall,
        }

    # Overall
    total_tp = sum(c.get('tp', 0) for c in summary['cases'].values())
    total_fp = sum(c.get('fp', 0) for c in summary['cases'].values())
    total_fn = sum(c.get('fn', 0) for c in summary['cases'].values())
    total_accepted = sum(c.get('accepted', 0) for c in summary['cases'].values())

    overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    summary['overall'] = {
        'total_tp': total_tp, 'total_fp': total_fp, 'total_fn': total_fn,
        'precision': overall_p, 'recall': overall_r,
        'total_accepted': total_accepted, 'false_positive_count': total_fp,
    }

    print(f"\n{'='*60}")
    print(f"OVERALL: TP={total_tp}, FP={total_fp}, FN={total_fn}")
    print(f"Precision={overall_p:.1%}, Recall={overall_r:.1%}")
    print(f"Auto-accepted: {total_accepted}, False Positives: {total_fp}")

    summary_path = OUT_DIR / 'occt_integration_summary.json'
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f"\nSummary: {summary_path}")


if __name__ == '__main__':
    main()
