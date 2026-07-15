"""JoinABLe pipeline: B-Rep graph extraction → JoinABLe → OCCT pose solve.
Python 3.11 env with torch_geometric required.
"""
import sys, json, time
from pathlib import Path

sys.path.insert(0, 'sw')
sys.path.insert(0, 'cad_assembly_agent/tools/joinable_interface_predictor')
sys.path.insert(0, 'joinable_gpu_reproduction')
sys.path.insert(0, 'joinable_step4_bundle_20260705')
sys.path.insert(0, 'cad_assembly_agent/tools/brep_graph_extractor')
sys.path.insert(0, 'joinable_migration_audit')

CKPT = Path('joinable_step4_bundle_20260705/joinable_migration_audit/vendor/JoinABLe/pretrained/paper/last_run_0.ckpt')
SW = Path('sw')

def extract_graphs(case_dir, graph_dir):
    from step_to_brep_graph_probe import extract_graph
    files = sorted(list(case_dir.glob('*.step')) + list(case_dir.glob('*.stp')) + list(case_dir.glob('*.STP')))
    files = [f for f in files if not f.stem.lower().startswith('assembly')]
    seen = set(); unique = []
    for f in files:
        if f.stem.lower() not in seen: seen.add(f.stem.lower()); unique.append(f)
    graphs = {}
    for f in unique:
        gp = graph_dir / (f.stem + '_graph.json')
        if gp.exists():
            graphs[f.name] = gp; continue
        print(f'  [graph] {f.name}...', flush=True)
        try:
            g = extract_graph(f)
            gp.write_text(json.dumps(g, ensure_ascii=False)+'\n', encoding='utf-8')
            s = g.get('metadata',{}).get('extraction_status','?')
            print(f'    {s} ({len(g.get("nodes",[]))} nodes)', flush=True)
            graphs[f.name] = gp
        except Exception as e:
            print(f'    FAILED: {e}', flush=True)
    return graphs

def run_joinable(ga, gb, out):
    from pretrained_joinable_predictor import predict
    try:
        ret = predict(ga, gb, CKPT, out, 10, 'cpu', 0)
        if ret == 0 and out.exists():
            return json.loads(out.read_text(encoding='utf-8'))
    except Exception as e:
        print(f'      err: {e}', flush=True)
    return None

def process_case(case_id):
    from known_group_assembly import run_known_group_assembly
    from integrate_occt_results import build_final_plan

    case_dir = (SW / case_id).resolve()
    graph_dir = case_dir / '_brep_graphs311'
    graph_dir.mkdir(exist_ok=True)
    jdir = case_dir / '_joinable'
    jdir.mkdir(exist_ok=True)

    graphs = extract_graphs(case_dir, graph_dir)
    if len(graphs) < 2:
        return {'error': f'only {len(graphs)} graphs'}

    parts = list(graphs.keys())
    pool = case_dir.name
    pairs = []
    for i in range(len(parts)):
        for j in range(i+1, len(parts)):
            a, b = parts[i], parts[j]
            out = jdir / f'pred_{i:02d}_{j:02d}.json'
            if out.exists():
                r = json.loads(out.read_text(encoding='utf-8'))
            else:
                print(f'  [joinable] {a} + {b}...', flush=True)
                r = run_joinable(graphs[a], graphs[b], out)
                if r is None: r = {}
            pairs.append({
                'pool_id': pool, 'pair_id': f'j_{i:02d}_{j:02d}',
                'part_a': a, 'part_b': b,
                'status': 'success' if r.get('candidates') else 'failed',
                'candidates': r.get('candidates', []),
                'pair_features': r.get('pair_features', {}),
            })

    rp = jdir / 'report.json'
    rp.write_text(json.dumps({'pairs': pairs}, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    n_ok = sum(1 for p in pairs if p['status']=='success')
    n_cand = sum(len(p.get('candidates',[])) for p in pairs)
    print(f'  Report: {n_ok}/{len(pairs)} pairs OK, {n_cand} candidates', flush=True)

    print(f'  [pose] Solving...', flush=True)
    t0 = time.time()
    result = run_known_group_assembly(case_dir, beam_width=20, joinable_report=str(rp))
    elapsed = time.time()-t0
    status = result.get('pose_status','?')
    conns = result.get('direct_connections',[])
    print(f'  [pose] {elapsed:.0f}s: pose={status}, connections={len(conns)}', flush=True)
    for c in conns:
        if 'joinable' in str(c.get('providers',[])):
            print(f'    JoinABLe: {c["parts"]}', flush=True)

    kg = case_dir/'known_group_output'
    occt = {'case_id': case_id, 'found': True, 'pose_status': status,
            'parts': result.get('parts',[]), 'direct_connections': conns}
    pv = kg/'pose_validation.json'
    if pv.exists():
        pvj = json.loads(pv.read_text(encoding='utf-8'))
        if pvj.get('pose_audit'):
            occt['collision_status'] = pvj['pose_audit'][0].get('collision_status')
            occt['collision_count'] = pvj['pose_audit'][0].get('collision_count',0)

    plan = build_final_plan(case_id, occt)
    od = Path(f'public_cad_dataset_audit/outputs/step34_solidworks_plan/case_{case_id}')
    od.mkdir(parents=True, exist_ok=True)
    (od/'solidworks_assembly_plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

    return {'pose': status, 'conns': len(conns), 'accepted': len(plan['accepted_edges']),
            'rejected': len(plan['rejected_edges']), 'j_ok': n_ok, 'j_cand': n_cand}

if __name__ == '__main__':
    import torch
    print(f'Torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
    print(f'CKPT: {CKPT.name} ({CKPT.stat().st_size/1024/1024:.0f}MB)\n')
    for cid in ['1','2','3']:
        print(f'{"="*60}\nCase {cid}\n{"="*60}')
        r = process_case(cid)
        print(f'  → {r}\n')
    print('Done')
