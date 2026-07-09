"""Render case 2 assembly — exports STLs + renders 4 views."""
import sys, os, math
sys.path.insert(0, 'sw')
from pathlib import Path

def export_stls(step_path, out_dir):
    """Export each solid body as a separate STL file."""
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_SOLID
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.StlAPI import StlAPI_Writer

    reader = STEPControl_Reader()
    if reader.ReadFile(str(step_path)) != IFSelect_RetDone:
        raise RuntimeError(f"Failed to read {step_path}")
    reader.TransferRoots()
    shape = reader.OneShape()

    solids = []
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    while exp.More():
        solids.append(exp.Current())
        exp.Next()

    if not solids:
        raise RuntimeError("No solids found in assembly")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    stl_files = []
    for i, solid in enumerate(solids):
        mesh = BRepMesh_IncrementalMesh(solid, 0.3, True, 0.3)
        mesh.Perform()
        stl_path = out_dir / f'component_{i:02d}.stl'
        writer = StlAPI_Writer()
        writer.Write(solid, str(stl_path))
        stl_files.append(stl_path)
        size_kb = stl_path.stat().st_size / 1024
        print(f'  Component {i}: {stl_path.name} ({size_kb:.0f} KB)')
    
    return stl_files


def render_views(stl_files, out_dir):
    """Render 4 views using matplotlib from STL files."""
    import numpy as np
    from stl import mesh as stl_mesh
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    colors = [
        (0.2, 0.5, 1.0),   # blue - shaft
        (0.2, 0.8, 0.3),   # green - flange  
        (0.9, 0.6, 0.1),   # orange - flange
        (0.7, 0.2, 0.7),   # purple - key
    ]

    # Load meshes
    meshes = []
    for stl_path in stl_files:
        m = stl_mesh.Mesh.from_file(str(stl_path))
        meshes.append(m)

    views = [
        ("isometric", 25, -40),
        ("front_XY", 0, -90),
        ("side_YZ", 0, 0),
        ("top_XZ", 90, 0),
    ]

    fig = plt.figure(figsize=(16, 12))
    
    for vi, (name, elev, azim) in enumerate(views):
        ax = fig.add_subplot(2, 2, vi + 1, projection='3d')
        
        for i, m in enumerate(meshes):
            c = colors[i % len(colors)]
            faces_count = len(m.vectors)
            step3 = max(1, faces_count // 3000)
            pc = Poly3DCollection(
                m.vectors[::step3], 
                alpha=0.80, 
                facecolor=c,
                edgecolor='none',
                linewidth=0,
            )
            ax.add_collection3d(pc)
        
        # Auto-scale
        all_verts = np.concatenate([m.vectors.reshape(-1, 3) for m in meshes])
        center = (all_verts.max(axis=0) + all_verts.min(axis=0)) / 2
        radius = (all_verts.max(axis=0) - all_verts.min(axis=0)).max() / 2 * 1.2
        
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(name, fontsize=14)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_box_aspect([1, 1, 1])

    out_path = Path(out_dir) / 'case2_assembly_render.png'
    fig.suptitle('Case 2: Shaft + 2 Flanges + Key Assembly\n'
                 '(blue=shaft, green=flange_a, orange=flange_b, purple=key)',
                 fontsize=13)
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=120, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f'Saved: {out_path}')
    return out_path


if __name__ == '__main__':
    step = Path('sw/2/known_group_output/assembly.step')
    out = Path('sw/2/known_group_output')
    
    print('Exporting STLs...')
    stls = export_stls(step, out)
    
    print('\nRendering views...')
    try:
        png = render_views(stls, out)
        print(f'\nDone! Open: {png}')
    except ImportError as e:
        print(f'numpy-stl not available ({e})')
        print(f'STLs exported to: {out}')
        print('Open them in any STL viewer (e.g. 3D Viewer, MeshLab, Fusion 360)')
