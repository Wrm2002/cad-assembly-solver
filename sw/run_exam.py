"""Exam runner: run known_group_assembly on cases 1-5 and score results."""
import sys, json, time, traceback
sys.path.insert(0, 'sw')
from pathlib import Path
from known_group_assembly import run_known_group_assembly

SW_DIR = Path('sw')

CASES = [
    {
        'case_id': 'case_1',
        'dir': SW_DIR / '1',
        'beam': 20,
        'expected_edges': [('flange_part_a.step', 'flange_part_b.step')],
        'expected_non_edges': [],
        'desc': '2 flanges face-to-face',
    },
    {
        'case_id': 'case_2',
        'dir': SW_DIR / '2',
        'beam': 20,
        # Ground truth: shaft connects to both flanges + key
        'expected_edges': [
            ('shaft_with_keyway.step', 'flange_a_pipe_X_fan15deg.step'),
            ('shaft_with_keyway.step', 'flange_b_pipe_Y_fan20deg.step'),
            ('shaft_with_keyway.step', 'key.step'),
        ],
        'expected_non_edges': [
            ('flange_a_pipe_X_fan15deg.step', 'flange_b_pipe_Y_fan20deg.step'),
            ('flange_a_pipe_X_fan15deg.step', 'key.step'),
            ('flange_b_pipe_Y_fan20deg.step', 'key.step'),
        ],
        'desc': 'shaft + 2 flanges + key',
    },
    {
        'case_id': 'case_3',
        'dir': SW_DIR / '3',
        'beam': 20,
        'expected_edges': [
            ('01_FAN-CAGE-MODULE-R620-NH.stp', '01_FAN-MODULE-SUNON-6056(1).stp'),
        ],
        'expected_non_edges': [],
        'desc': 'fan cage + fan module',
    },
    {
        'case_id': 'case_4',
        'dir': SW_DIR / '4_lightweight',
        'beam': 20,
        'expected_edges': [
            ('01-62DC24-MLB-PCBA.stp', '5-rd_rc_a_2rx4_1_ddrv.stp'),
            ('01-62DC24-MLB-PCBA.stp', 'Hygon-7400-3D.stp'),
        ],
        'expected_non_edges': [
            ('5-rd_rc_a_2rx4_1_ddrv.stp', 'Hygon-7400-3D.stp'),
        ],
        'desc': 'PCBA + memory + CPU',
    },
    {
        'case_id': 'case_5',
        'dir': SW_DIR / '5_lightweight',
        'beam': 20,
        'expected_edges': [
            ('01-ASSY-CHASSIS-MODULE-R6250H0.stp', '01-ASSY-CHASSIS-EAR-L-R620.stp'),
            ('01-ASSY-CHASSIS-MODULE-R6250H0.stp', '5-CRPS1300NC.stp'),
        ],
        'expected_non_edges': [
            ('01-ASSY-CHASSIS-EAR-L-R620.stp', '5-CRPS1300NC.stp'),
        ],
        'desc': 'chassis + ear + PSU',
    },
]


def normalize_parts(parts_tuple):
    """Normalize part names for comparison."""
    return tuple(sorted(str(p) for p in parts_tuple))


def eval_case(info):
    case_id = info['case_id']
    case_dir = info['dir']
    beam = info['beam']
    expected = {normalize_parts(e) for e in info['expected_edges']}
    expected_non = {normalize_parts(e) for e in info['expected_non_edges']}

    print(f'\n{"="*60}')
    print(f'Exam: {case_id} — {info["desc"]}')
    print(f'Parts: {case_dir}')
    print(f'Expected edges: {len(expected)}, Expected non-edges: {len(expected_non)}')
    print(f'{"="*60}', flush=True)

    result = {'case_id': case_id, 'passed': False, 'errors': []}

    try:
        t0 = time.time()
        output = run_known_group_assembly(case_dir, beam_width=beam)
        elapsed = time.time() - t0

        pose_status = output.get('pose_status', '?')
        connections = output.get('direct_connections', [])

        # Extract actual edges
        actual_edges = set()
        for conn in connections:
            actual_edges.add(normalize_parts(conn['parts']))

        # Check pose validation
        pv_path = case_dir / 'known_group_output' / 'pose_validation.json'
        pv = {}
        collision_count = None
        if pv_path.exists():
            pv = json.loads(pv_path.read_text(encoding='utf-8'))
            if pv.get('pose_audit'):
                best = pv['pose_audit'][0]
                collision_count = best.get('collision_count', 0)
                closure = best.get('constraint_closure', {})
                closed = closure.get('closed_connection_count', 0)
                total_c = closure.get('connection_count', 0)
                result['closure'] = f'{closed}/{total_c}'
                result['collision_count'] = collision_count
                result['score'] = best.get('total_score')

        result['pose_status'] = pose_status
        result['runtime_s'] = round(elapsed, 1)
        result['num_conn'] = len(connections)
        result['actual_edges'] = [list(e) for e in actual_edges]

        # Compute edge-level metrics
        tp_set = actual_edges & expected
        fp_set = actual_edges - expected
        fn_set = expected - actual_edges
        tp = len(tp_set)
        fp = len(fp_set)
        fn = len(fn_set)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Check non-edges (false positives from expected_non)
        fp_from_non = actual_edges & expected_non
        fn_wrong = expected - actual_edges

        result['tp'] = tp
        result['fp'] = fp
        result['fn'] = fn
        result['precision'] = round(precision, 3)
        result['recall'] = round(recall, 3)
        result['f1'] = round(f1, 3)
        result['fp_from_expected_non'] = [list(e) for e in fp_from_non]
        result['fn_missing'] = [list(e) for e in fn_wrong]

        # Pass criteria: no false positives AND all true edges found
        collision_ok = collision_count is None or collision_count == 0
        no_fp = fp == 0
        no_fn = fn == 0
        result['passed'] = no_fp and no_fn

        status_icon = '✅' if result['passed'] else '❌'
        print(f'\n  {status_icon} Pose: {pose_status}, Runtime: {elapsed:.1f}s')
        print(f'  Edges: {len(actual_edges)} (expected {len(expected)})')
        print(f'  TP={tp}, FP={fp}, FN={fn}')
        print(f'  Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}')
        if collision_count is not None:
            print(f'  Collisions: {collision_count}')
        if result.get('closure'):
            print(f'  Constraint closure: {result["closure"]}')
        if result.get('score'):
            print(f'  Score: {result["score"]:.3f}')
        if fp_from_non:
            print(f'  ❌ FALSE POSITIVE (expected non-edge): {fp_from_non}')
        if fn_wrong:
            print(f'  ❌ MISSING (expected edge): {fn_wrong}')
        if not collision_ok:
            print(f'  ⚠️  COLLISIONS present')

    except Exception as e:
        result['error'] = str(e)
        result['traceback'] = traceback.format_exc()
        print(f'\n  ❌ ERROR: {e}', flush=True)
        traceback.print_exc()

    return result


def main():
    all_results = []
    for case_info in CASES:
        r = eval_case(case_info)
        all_results.append(r)

    # Summary
    print(f'\n\n{"="*60}')
    print('EXAM SUMMARY')
    print(f'{"="*60}')
    passed = sum(1 for r in all_results if r['passed'])
    failed = sum(1 for r in all_results if not r['passed'] and 'error' not in r)
    errors = sum(1 for r in all_results if 'error' in r)
    total = len(all_results)

    print(f'\n  Total cases: {total}')
    print(f'  ✅ Passed: {passed}')
    print(f'  ❌ Failed: {failed}')
    print(f'  💥 Errors: {errors}')
    print()

    for r in all_results:
        icon = '✅' if r['passed'] else ('💥' if 'error' in r else '❌')
        status = f'pose={r.get("pose_status","?")}, P={r.get("precision",0):.3f}, R={r.get("recall",0):.3f}, F1={r.get("f1",0):.3f}'
        print(f'  {icon} {r["case_id"]}: {status}')

    # Write detailed results
    out_path = SW_DIR / 'exam_results.json'
    out_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\nDetailed results: {out_path}')

    return 0 if failed == 0 and errors == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
