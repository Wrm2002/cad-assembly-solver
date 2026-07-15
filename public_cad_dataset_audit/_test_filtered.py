"""Quick test: case 2 without flange-flange coaxial constraints."""
import sys, json, time
sys.path.insert(0, 'sw')

# Monkey-patch: wrap match_features to filter flange-flange coaxial
import constraints as _c
_orig_match_features = _c.match_features

def _filtered_match_features(parts_features, thresholds=None):
    raw = _orig_match_features(parts_features, thresholds)
    filtered = []
    for m in raw:
        p0, p1 = m['parts']
        both_flange = ('flange' in p0.lower()) and ('flange' in p1.lower())
        is_coaxial = m.get('type') == 'coaxial'
        if both_flange and is_coaxial:
            continue
        filtered.append(m)
    return filtered

_c.match_features = _filtered_match_features

# Also patch in known_group_assembly's imported version
import known_group_assembly as kga
kga.match_features = _filtered_match_features

from pathlib import Path
case_dir = Path('_audit_sw_origin/sw/2')
print('Running case 2 WITHOUT flange-flange coaxial...', flush=True)
t0 = time.time()
result = kga.run_known_group_assembly(case_dir, beam_width=20)
elapsed = time.time() - t0

status = result.get('pose_status', '?')
conn_count = len(result.get('direct_connections', []))
print(f'DONE in {elapsed:.0f}s: pose={status}, connections={conn_count}', flush=True)

pv_path = case_dir / 'known_group_output' / 'pose_validation.json'
if pv_path.exists():
    pv = json.loads(pv_path.read_text(encoding='utf-8'))
    best = pv['pose_audit'][0]
    cc = best['constraint_closure']
    print(f'Score: {best["total_score"]:.3f}, Collisions: {best["collision_count"]}', flush=True)
    print(f'Closure: {cc["closed_connection_count"]}/{cc["connection_count"]}', flush=True)
    for c in cc['connections']:
        print(f'  {c["parts"]}: closed={c["closed"]}', flush=True)
    for c in best.get('collisions', []):
        print(f'  COLLISION: {c["parts"]}: {c.get("intersection_volume_mm3",0):.0f} mm3', flush=True)
    if status == 'valid':
        print('SUCCESS!', flush=True)
