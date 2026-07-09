"""Render case 2 assembly with colored components for semantic inspection."""
import sys
sys.path.insert(0, 'sw')
from pathlib import Path

def render_assembly(step_path: str, output_dir: str):
    """Render each component in a different color using OCCT + mesh export."""
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_COMPOUND
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.StlAPI import StlAPI_Writer
    from OCC.Core.TopoDS import TopoDS_Compound
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib
    import numpy as np
    import trimesh

    reader = STEPControl_Reader()
    if reader.ReadFile(step_path) != IFSelect_RetDone:
        raise RuntimeError(f"Failed to read {step_path}")
    reader.TransferRoots()
    shape = reader.OneShape()

    # Collect all solids
    solids = []
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    while exp.More():
        solids.append(exp.Current())
        exp.Next()
    
    if not solids:
        # Try compound
        exp = TopExp_Explorer(shape, TopAbs_COMPOUND)
        while exp.More():
            solids.append(exp.Current())
            exp.Next()

    print(f"Found {len(solids)} solid bodies")

    # Render with trimesh
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axis3D

    # Colors for parts
    colors = [
        [0.2, 0.6, 1.0],   # blue - shaft
        [0.2, 0.8, 0.3],   # green - flange_a
        [0.9, 0.6, 0.1],   # orange - flange_b
        [0.7, 0.2, 0.7],   # purple - key
    ]
    
    fig = plt.figure(figsize=(16, 10))
    
    views = [
        ("Isometric", (30, -45)),
        ("Front XY", (0, -90)),
        ("Side YZ", (0, 0)),
        ("Top XZ", (90, 0)),
    ]
    
    meshes = []
    for i, solid in enumerate(solids):
        # Mesh the solid
        mesh = BRepMesh_IncrementalMesh(solid, 0.5, True, 0.5)
        mesh.Perform()
        
        # Export to temp STL
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp:
            stl_path = tmp.name
        writer = StlAPI_Writer()
        writer.Write(solid, stl_path)
        
        # Load with trimesh
        try:
            tm = trimesh.load(stl_path)
            meshes.append(tm)
        finally:
            os.unlink(stl_path)
    
    for vi, (title, (elev, azim)) in enumerate(views):
        ax = fig.add_subplot(2, 2, vi + 1, projection='3d')
        
        for i, tm in enumerate(meshes):
            color = colors[i % len(colors)]
            # Get vertices
            verts = tm.vertices
            faces = tm.faces
            ax.plot_trisurf(
                verts[:, 0], verts[:, 1], verts[:, 2],
                triangles=faces,
                color=color,
                alpha=0.85,
                shade=True,
                edgecolor='none',
                linewidth=0,
                antialiased=True,
            )
        
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect([1, 1, 1])
    
    out_path = Path(output_dir) / 'case2_assembly_render.png'
    fig.suptitle('Case 2: Shaft + 2 Flanges + Key Assembly', fontsize=14, fontweight='bold')
    fig.legend(
        [plt.Line2D([0], [0], color=c, lw=8) for c in colors[:len(meshes)]],
        ['shaft_with_keyway', 'flange_a (15°)', 'flange_b (20°)', 'key'],
        loc='lower center', ncol=4, fontsize=10
    )
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


if __name__ == '__main__':
    step_path = 'sw/2/known_group_output/assembly.step'
    render_assembly(step_path, 'sw/2/known_group_output')
