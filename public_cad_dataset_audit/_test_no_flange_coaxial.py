import sys, json, time, copy
sys.path.insert(0, 'sw')
from known_group_assembly import run_known_group_assembly
from constraints import match_features
from features import extract_features
from pathlib import Path

case_dir = Path('_audit_sw_origin/sw/2')
step_files = [f for f in case_dir.glob('*.step') if not f.name.lower().startswith('assembly')]
parts = {f.name: extract_features(str(f)) for f in step_files}

# Get raw matches and filter: drop coaxial between flange-flange
raw = match_features(parts, {"preserve_cylindrical_face_hypotheses": True})
filtered = []
dropped = 0
for m in raw:
    p0, p1 = m['parts']
    both_flange = 'flange' in p0.lower() and 'flange' in p1.lower()
    is_coaxial = m['type'] in ('coaxial',)
    if both_flange and is_coaxial:
        dropped += 1
        continue
    filtered.append(m)

print(f'Raw edges: {len(raw)}, after drop flange-flange coaxial: {len(filtered)} (dropped {dropped})', flush=True)

# Now run solver with filtered matches
# We need to monkey-patch match_features... easier to just modify known_group_assembly
# Let me directly call the solver internals
from known_group_assembly import build_constraint_graph_from_matches, solve_assembly_pose

# Build graph from filtered matches
graph = build_constraint_graph_from_matches(parts, filtered)

# Solve
print('Solving...', flush=True)
t0 = time.time()
result = solve_assembly_pose(parts, graph, beam_width=20)
elapsed = time.time() - t0

status = result.get('pose_status', '?')
print(f'DONE in {elapsed:.0f}s: pose={status}', flush=True)
print(f'Connections: {len(result.get("direct_connections", []))}', flush=True)

# Check collisions
collisions = result.get('collisions', [])
if collisions:
    print(f'Collisions: {len(collisions)}', flush=True)
    for c in collisions:
        print(f'  {c["parts"]}: {c.get("intersection_volume_mm3",0):.0f} mm3', flush=True)
else:
    print('NO COLLISIONS!', flush=True)

if status == 'valid':
    print('SUCCESS! Case 2 passes with flange-flange coaxial removed!', flush=True)
else:
    print(f'Failed: pose={status}', flush=True)
