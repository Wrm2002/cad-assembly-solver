import sys, json, time
sys.path.insert(0, 'sw')
from pathlib import Path
from known_group_assembly import run_known_group_assembly

d = Path('sw/3')
print('Starting case 3...', flush=True)
t0 = time.time()
out = run_known_group_assembly(d, beam_width=20)
elapsed = time.time() - t0
conns = out.get('direct_connections', [])
ps = out.get('pose_status', '?')
print(f'Done {elapsed:.0f}s: pose={ps}, edges={len(conns)}', flush=True)
for c in conns:
    print(f'  Edge: {c["parts"]}', flush=True)
