"""Conservative B-Rep opening/corridor audit for a folded-insertion pose.

It samples the chassis' solid topology on a sequence of planes normal to the
candidate insertion direction.  A candidate can proceed to hole locking only
when the moving component's *conservative cross-section envelope* remains in
the exterior-reachable free region at every sampled depth.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument('folder',type=Path); args=parser.parse_args(); out=args.folder
    candidate=json.loads((out/'ear_folded_flange_candidates.json').read_text())['candidates'][0]
    lo=np.asarray(candidate['bbox_mm']['min'],float); hi=np.asarray(candidate['bbox_mm']['max'],float)
    # The new folded proposal explicitly maps I/O +Z to chassis -Y; sample
    # the full candidate y-span as the inward insertion path.
    insertion_axis=1; step=4.0
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_IN, TopAbs_ON
    from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.gp import gp_Pnt
    root=Path(__file__).resolve().parents[1]; path=root/'sw'/'5'/'01-ASSY-CHASSIS-MODULE-R6250H0.stp'
    reader=STEPControl_Reader();
    if reader.ReadFile(str(path))!=IFSelect_RetDone: raise RuntimeError(f'cannot load {path}')
    reader.TransferRoots(); shape=reader.OneShape(); explorer=TopExp_Explorer(shape,TopAbs_SOLID); solids=[]
    while explorer.More():
        solid=explorer.Current(); explorer.Next(); box=Bnd_Box(); box.SetGap(0); brepbndlib.Add(solid,box); solids.append((solid,np.asarray(box.Get(),float)))
    xs=np.arange(lo[0]+step/2,hi[0],step); zs=np.arange(lo[2]+step/2,hi[2],step); ys=np.arange(lo[1]+step/2,hi[1],step)
    slices=[]; occupancy=[]
    for y in ys:
        grid=np.zeros((len(zs),len(xs)),dtype=np.uint8)
        # only solids overlapping this local slab are relevant to a point test
        relevant=[solid for solid,b in solids if b[1]-0.05<=y<=b[4]+0.05]
        for iz,z in enumerate(zs):
            for ix,x in enumerate(xs):
                point=gp_Pnt(float(x),float(y),float(z))
                for solid in relevant:
                    state=BRepClass3d_SolidClassifier(solid,point,0.05).State()
                    if state in (TopAbs_IN,TopAbs_ON): grid[iz,ix]=1; break
        occupied=int(grid.sum()); total=int(grid.size); occupancy.append(grid)
        slices.append({'y_mm':round(float(y),3),'occupied_samples':occupied,'sample_count':total,'free_fraction':round(1-occupied/max(total,1),4),'envelope_clear':occupied==0})
    passed=all(row['envelope_clear'] for row in slices)
    # A high-resolution image is a direct diagnostic of the opening contour
    # in the actual B-Rep solid topology, not an inferred bbox cavity.
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    cols=4; rows=int(np.ceil(len(occupancy)/cols)); fig,axes=plt.subplots(rows,cols,figsize=(4*cols,3*rows),squeeze=False)
    for i,ax in enumerate(axes.ravel()):
        if i>=len(occupancy): ax.axis('off'); continue
        ax.imshow(occupancy[i],origin='lower',extent=[lo[0],hi[0],lo[2],hi[2]],cmap='Reds',vmin=0,vmax=1,aspect='auto')
        ax.set_title(f'y={ys[i]:.1f} mm; occupied={int(occupancy[i].sum())}'); ax.set_xlabel('chassis X');ax.set_ylabel('chassis Z')
    fig.suptitle('EAR conservative cross-section envelope vs chassis solid occupancy\nred = chassis solid; any red sample blocks insertion corridor');fig.tight_layout();fig.savefig(out/'ear_opening_corridor_slices.png',dpi=160);plt.close(fig)
    result={'method':'BRep_solid_cross_section_free_space_audit','candidate_source':'ear_folded_flange_candidates.json','insertion_axis':'chassis +Y','cross_section_envelope_mm':{'x':[float(lo[0]),float(hi[0])],'z':[float(lo[2]),float(hi[2])]},'sample_step_mm':step,'solid_count':len(solids),'slices':slices,'corridor_passed':passed,'status':'eligible_for_hole_locking' if passed else 'rejected_before_hole_locking','reason':'all sampled conservative envelopes are free' if passed else 'at least one sampled cross-section intersects chassis solid topology'}
    (out/'ear_opening_corridor_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'solid_count':len(solids),'slice_count':len(slices),'corridor_passed':passed,'blocked_slices':sum(not x['envelope_clear'] for x in slices)},indent=2))
if __name__=='__main__':main()
