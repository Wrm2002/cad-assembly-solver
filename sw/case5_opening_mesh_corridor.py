"""Mesh-proxy opening corridor test for a straight folded-ear insertion.

The mesh is read-only geometry derived from the original chassis STEP.  Rays
start outside the chassis and travel along the proposed insertion direction;
an EAR cross-section sample is blocked if a chassis triangle is encountered
before the sample depth.  Exact OCCT solid collision remains the final gate.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();o=a.folder
    cand=json.loads((o/'ear_folded_flange_candidates.json').read_text())['candidates'][0]
    lo=np.asarray(cand['bbox_mm']['min'],float);hi=np.asarray(cand['bbox_mm']['max'],float);step=5.0
    import trimesh
    mesh=trimesh.load(o/'opening_mesh_proxy'/'part_00_chassis_fixed.stl',force='mesh',process=False)
    xs=np.arange(lo[0]+step/2,hi[0],step);zs=np.arange(lo[2]+step/2,hi[2],step);ys=np.arange(lo[1]+step/2,hi[1],step)
    XX,ZZ=np.meshgrid(xs,zs);orig=np.column_stack([XX.ravel(),np.full(XX.size,lo[1]-25),ZZ.ravel()]);directions=np.tile([0.,1.,0.],(len(orig),1))
    grids=[];slices=[]
    for y in ys:
        locations,index_ray,_=mesh.ray.intersects_location(ray_origins=orig,ray_directions=directions,multiple_hits=True)
        first=np.full(len(orig),np.inf)
        if len(locations): np.minimum.at(first,index_ray,locations[:,1])
        blocked=first<y-0.25;grid=blocked.reshape(len(zs),len(xs));grids.append(grid)
        slices.append({'y_mm':round(float(y),3),'blocked_samples':int(grid.sum()),'sample_count':int(grid.size),'free_fraction':round(float(1-grid.mean()),4),'envelope_clear':not bool(grid.any())})
    passed=all(x['envelope_clear'] for x in slices)
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    cols=4;rows=int(np.ceil(len(grids)/cols));fig,ax=plt.subplots(rows,cols,figsize=(4*cols,3*rows),squeeze=False)
    for i,a0 in enumerate(ax.ravel()):
        if i>=len(grids):a0.axis('off');continue
        a0.imshow(grids[i],origin='lower',extent=[lo[0],hi[0],lo[2],hi[2]],cmap='Reds',vmin=0,vmax=1,aspect='auto');a0.set_title(f'y={ys[i]:.1f}; blocked={grids[i].sum()}');a0.set_xlabel('X');a0.set_ylabel('Z')
    fig.suptitle('Exterior-to-interior opening corridor audit (mesh proxy)\nred: a ray from exterior intersects chassis before the EAR envelope sample');fig.tight_layout();fig.savefig(o/'ear_opening_mesh_corridor_slices.png',dpi=160);plt.close(fig)
    data={'method':'read_only_chassis_triangle_mesh_exterior_ray_corridor','mesh_deflection_mm':1.5,'candidate_source':'ear_folded_flange_candidates.json','insertion_axis':'chassis +Y','sample_step_mm':step,'cross_section_envelope_mm':{'x':lo[[0]].tolist()+hi[[0]].tolist(),'z':lo[[2]].tolist()+hi[[2]].tolist()},'slices':slices,'corridor_passed':passed,'status':'eligible_for_hole_locking' if passed else 'rejected_before_hole_locking','limitation':'mesh proxy only; original B-Rep solid collision remains final validation'}
    (o/'ear_opening_mesh_corridor_audit.json').write_text(json.dumps(data,indent=2),encoding='utf-8');print(json.dumps({'corridor_passed':passed,'blocked_slices':sum(not x['envelope_clear'] for x in slices),'slices':len(slices)},indent=2))
if __name__=='__main__':main()
