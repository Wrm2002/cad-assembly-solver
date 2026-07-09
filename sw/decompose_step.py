"""Save sub-solids as individual STEP files in named folders."""
import sys, os, math, json
from OCC.Core.STEPControl import STEPControl_Reader, STEPControl_Writer
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_FACE
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRep import BRep_Builder
from OCC.Core.TopoDS import TopoDS_Compound


def _try_load_cache(filepath, out_dir, min_vol):
    """Return cached sub-part list if valid, else None."""
    cache_path = os.path.join(out_dir, '_cache.json')
    if not os.path.isfile(cache_path):
        return None
    try:
        if os.path.getmtime(cache_path) <= os.path.getmtime(filepath):
            return None  # source file newer than cache
        with open(cache_path, 'r') as f:
            cached = json.load(f)
        if cached.get('min_vol') != min_vol:
            return None
        if cached.get('source') != os.path.abspath(filepath):
            return None
        # Verify all cached files still exist
        for s in cached['subs']:
            if not os.path.isfile(s['path']):
                return None
        return cached['subs']
    except Exception:
        return None


def _save_cache(filepath, out_dir, min_vol, subs):
    """Save sub-part list to cache for fast re-load."""
    cache_path = os.path.join(out_dir, '_cache.json')
    try:
        os.makedirs(out_dir, exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump({
                'source': os.path.abspath(filepath),
                'min_vol': min_vol,
                'subs': subs,
            }, f)
    except Exception:
        pass


def decompose_file(filepath, output_dir, min_vol=100):
    """
    Decompose a single STEP file into sub-solids.

    Returns a list of dicts, one per sub-solid:
        {index, bbox: (dx,dy,dz), path: str, faces: int}
    For single-solid files (no decomposition needed), returns one entry
    with the original filepath — no temp files are written.

    Multi-solid files are saved as sub_XXX.stp under output_dir/<parent_name>/.
    Cached via _cache.json — re-running on unchanged files skips OCCT entirely.
    """
    parent_name = os.path.splitext(os.path.basename(filepath))[0]
    out_dir = os.path.join(output_dir, parent_name)

    # ── Cache hit: return cached results without touching OCCT ──
    cached = _try_load_cache(filepath, out_dir, min_vol)
    if cached is not None:
        return cached

    reader = STEPControl_Reader()
    if reader.ReadFile(filepath) != IFSelect_RetDone:
        raise RuntimeError(f"Failed to read STEP file: {filepath}")
    reader.TransferRoots()
    shape = reader.OneShape()

    # Collect valid solids
    solids = []
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    while exp.More():
        s = exp.Current()
        bb = Bnd_Box(); bb.SetGap(0.0); brepbndlib.Add(s, bb)
        x1, y1, z1, x2, y2, z2 = bb.Get()
        sz = (x2 - x1, y2 - y1, z2 - z1)
        if sz[0] * sz[1] * sz[2] > min_vol:
            solids.append((s, sz))
        exp.Next()

    # Build result list
    if len(solids) == 1:
        results = [{'index': 0, 'bbox': solids[0][1], 'path': filepath, 'faces': None}]
    elif len(solids) == 0:
        bb = Bnd_Box(); bb.SetGap(0.0); brepbndlib.Add(shape, bb)
        x1, y1, z1, x2, y2, z2 = bb.Get()
        sz = (x2 - x1, y2 - y1, z2 - z1)
        results = [{'index': 0, 'bbox': sz, 'path': filepath, 'faces': None}]
    else:
        # Multiple solids — save each as individual STEP
        os.makedirs(out_dir, exist_ok=True)
        results = []
        for i, (sld, sz) in enumerate(solids):
            comp = TopoDS_Compound()
            builder = BRep_Builder()
            builder.MakeCompound(comp)
            builder.Add(comp, sld)

            writer = STEPControl_Writer()
            writer.Transfer(comp, 1)
            out_path = os.path.join(out_dir, f'sub_{i:03d}.stp')
            writer.Write(out_path)

            fc = 0
            exp2 = TopExp_Explorer(comp, TopAbs_FACE)
            while exp2.More():
                fc += 1
                exp2.Next()

            results.append({'index': i, 'bbox': sz, 'path': out_path, 'faces': fc})

    _save_cache(filepath, out_dir, min_vol, results)
    return results


def decompose_to_step_files(input_folder, output_root, min_vol=100):
    """
    Decompose all STEP files in input_folder into sub-solids.
    Each file's sub-solids are saved as individual STEP files under:
        output_root/<parent_name>/
    """
    step_files = sorted([f for f in os.listdir(input_folder)
                         if f.lower().endswith(('.step', '.stp'))
                         and 'assembly' not in f.lower()])

    for fname in step_files:
        fp = os.path.join(input_folder, fname)
        fsize_mb = os.path.getsize(fp) / 1024 / 1024
        print(f'{fname} ({fsize_mb:.0f}MB)...')

        try:
            subs = decompose_file(fp, output_root, min_vol=min_vol)
        except RuntimeError as e:
            print(f'  READ FAILED: {e}')
            continue
        for s in subs:
            fc_str = f'{s["faces"]} faces' if s['faces'] is not None else '? faces'
            sz = s['bbox']
            path_size = os.path.getsize(s['path']) / 1024 if os.path.exists(s['path']) else 0
            print(f'  sub_{s["index"]:03d}: ({sz[0]:.0f},{sz[1]:.0f},{sz[2]:.0f})mm  {fc_str}  {path_size:.0f}KB')
        print(f'  -> {len(subs)} sub-parts')

if __name__ == '__main__':
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else '4'
    folder = os.path.abspath(folder)
    output = os.path.join(folder, 'sub_parts')
    decompose_to_step_files(folder, output)
    print(f'\nDone. Sub-parts in: {output}')
