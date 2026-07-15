import sys, json, time
sys.path.insert(0, 'sw')
from known_group_assembly import run_known_group_assembly
from pathlib import Path

case_dir = Path('_audit_sw_origin/sw/2')
print('Running improved solver on case 2...', flush=True)
t0 = time.time()
result = run_known_group_assembly(case_dir, beam_width=20)
elapsed = time.time() - t0

status = result.get('pose_status', '?')
conns = result.get('direct_connections', [])
print(f'DONE in {elapsed:.0f}s: pose={status}, connections={len(conns)}', flush=True)

pv_path = case_dir / 'known_group_output' / 'pose_validation.json'
pv = json.loads(pv_path.read_text(encoding='utf-8'))
best = pv['pose_audit'][0]
cc = best['constraint_closure']
print(f'Score: {best["total_score"]:.3f}, Collisions: {best["collision_count"]}', flush=True)
print(f'Closure: {cc["closed_connection_count"]}/{cc["connection_count"]}', flush=True)
for c in cc['connections']:
    print(f'  {c["parts"]}: closed={c["closed"]}', flush=True)
for c in best.get('collisions', []):
    print(f'  COLLISION: {c["parts"]}: {c["intersection_volume_mm3"]:.0f} mm3', flush=True)

if status == 'valid':
    print('SUCCESS! Case 2 passes!', flush=True)
else:
    print(f'Failed: pose={status}', flush=True)
