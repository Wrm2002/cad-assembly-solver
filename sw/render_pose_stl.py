"""Render transformed STL files with the stable non-OCCT Python runtime."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('stl_dir',type=Path); parser.add_argument('output',type=Path); parser.add_argument('--title',default='CAD Pose')
    args=parser.parse_args(); import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; import trimesh; from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    files=sorted(args.stl_dir.glob('*.stl')); meshes=[(p.stem,trimesh.load_mesh(p,force='mesh')) for p in files]
    points=np.concatenate([m.vertices for _,m in meshes]); center=(points.min(axis=0)+points.max(axis=0))/2; radius=max((points.max(axis=0)-points.min(axis=0)).max()*0.58,1.0)
    colors=[(0.15,0.43,0.85),(0.16,0.76,0.30),(0.94,0.48,0.08),(0.62,0.20,0.72),(0.10,0.65,0.70)]; views=[(25,-45,'isometric'),(0,-90,'front XY'),(0,0,'side YZ'),(90,0,'top XZ')]
    figure=plt.figure(figsize=(15,12))
    for i,(el,az,name) in enumerate(views,1):
        axis=figure.add_subplot(2,2,i,projection='3d')
        for k,(_,mesh) in enumerate(meshes):
            faces=mesh.triangles; step=max(1,len(faces)//3500); axis.add_collection3d(Poly3DCollection(faces[::step],facecolor=colors[k%len(colors)],edgecolor='none',alpha=.82))
        axis.set_xlim(center[0]-radius,center[0]+radius); axis.set_ylim(center[1]-radius,center[1]+radius); axis.set_zlim(center[2]-radius,center[2]+radius); axis.set_box_aspect((1,1,1)); axis.view_init(el,az); axis.set_axis_off(); axis.set_title(name)
    figure.suptitle(args.title+'\n'+', '.join(f'{name}=color{i+1}' for i,(name,_) in enumerate(meshes))); args.output.parent.mkdir(parents=True,exist_ok=True); figure.tight_layout(); figure.savefig(args.output,dpi=150,bbox_inches='tight',facecolor='white'); plt.close(figure)
    print(json.dumps({'output':str(args.output.resolve()),'parts':[name for name,_ in meshes]})); return 0
if __name__=='__main__': raise SystemExit(main())
