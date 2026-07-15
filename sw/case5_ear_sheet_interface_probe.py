"""Generate EAR candidates from local sheet-metal interfaces before holes.

Unlike the old hole-first probe, broad source I/O/flange faces are matched to
bounded left-side chassis facets.  The pose is seeded by opposing normals and
in-plane footprint compatibility; nearby holes are only scored afterwards.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any
import numpy as np
from proxy_insertion_pose import _rotation_axis_angle


def load(p: Path) -> Any: return json.loads(p.read_text(encoding="utf-8"))
def save(p: Path, x: Any) -> None: p.write_text(json.dumps(x, indent=2), encoding="utf-8")
def unit(x: Any) -> np.ndarray:
    x=np.asarray(x,float); return x/np.linalg.norm(x)
def basis(plane: dict[str, Any]) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    n=unit(plane['normal']); u=unit(plane['axis_u']); v=unit(plane['axis_v'])
    if np.linalg.det(np.column_stack([u,v,n]))<0: v=-v
    return u,v,n
def rigid_from_frames(s: dict[str,Any], t: dict[str,Any]) -> np.ndarray:
    su,sv,sn=basis(s); tu,tv,tn=basis(t)
    # source outward normal must oppose chassis wall normal; retain a proper frame.
    A=np.column_stack([su,sv,sn]); B=np.column_stack([tu,-tv,-tn])
    R=B@A.T
    if np.linalg.det(R)<0: B[:,1]*=-1; R=B@A.T
    return R
def bbox(b: dict[str,Any],R:np.ndarray,t:np.ndarray):
    lo=np.asarray(b['bbox_min'],float);hi=np.asarray(b['bbox_max'],float)
    pts=np.asarray([[x,y,z]for x in[lo[0],hi[0]]for y in[lo[1],hi[1]]for z in[lo[2],hi[2]]])@R.T+t
    return pts.min(0),pts.max(0)
def main():
 p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();o=a.folder
 ep=load(o/'ear_planes_raw.json'); cp=load(o/'chassis_planes_raw.json'); eb=load(o/'ear_bbox.json'); ch=load(o/'chassis_bbox.json')
 # EAR outer plates: broad but not enclosure-scale, with 20×80-ish footprint.
 src=[x for x in ep['planes'] if x['area_proxy']>700 and min(x['extent_u'],x['extent_v'])>14 and max(x['extent_u'],x['extent_v'])>45]
 # bounded left-wall facets, excluding the full-height rail that generated the false match.
 dst=[x for x in cp['planes'] if x['normal'][0]>0.98 and x['origin'][0]<-205 and x['area_proxy']>500 and min(x['extent_u'],x['extent_v'])>10 and max(x['extent_u'],x['extent_v'])<180]
 cand=[]
 for s in src:
  ds=sorted([s['extent_u'],s['extent_v']])
  for q in dst:
   dq=sorted([q['extent_u'],q['extent_v']]); ratio=sum(abs(np.log(max(a,b)/min(a,b))) for a,b in zip(ds,dq))
   if ratio>0.72: continue
   R=rigid_from_frames(s,q); t=np.asarray(q['origin'])-R@np.asarray(s['origin'])
   lo,hi=bbox(eb,R,t); c_lo=np.asarray(ch['bbox_min']);c_hi=np.asarray(ch['bbox_max'])
   # I/O face must reside on exterior side; do not accept the old interior-facing orientation.
   source_n=R@unit(s['normal']); face_x=float((R@np.asarray(s['origin'])+t)[0])
   ext=face_x<c_lo[0]+2.0 and source_n[0]<-0.98
   if not ext: continue
   # preserve a viable local vertical/long-span overlap with the target facet.
   overlap_y=max(0,min(hi[1],q['origin'][1]+max(dq)/2)-max(lo[1],q['origin'][1]-max(dq)/2))
   score=np.exp(-ratio)*min(1,overlap_y/max(min(ds),1))
   cand.append({'R':R.round(8).tolist(),'t_mm':t.round(6).tolist(),'axis_angle':_rotation_axis_angle(R),'ear_face':s['face_index'],'chassis_face':q['face_index'],'source_dimensions_mm':ds,'target_dimensions_mm':dq,'dimension_log_error':round(float(ratio),5),'bbox_mm':{'min':lo.round(5).tolist(),'max':hi.round(5).tolist()},'io_face_exterior':True,'sheet_interface_score':round(float(score),5)})
 cand.sort(key=lambda x:-x['sheet_interface_score']); save(o/'ear_sheet_interface_candidates.json',{'method':'bounded_sheet_face_opposing_normal_and_footprint','candidate_count':len(cand),'candidates':cand[:20]})
 if cand:
  root=Path(__file__).resolve().parents[1]
  top=cand[0]
  save(o/'ear_sheet_interface_manifest.json',{'assembly_name':'case5_ear_sheet_interface_probe','components':[{'id':'chassis_fixed','source':str((root/'sw'/'5'/'01-ASSY-CHASSIS-MODULE-R6250H0.stp').resolve()),'placement':{'translate':[0,0,0]}},{'id':'EAR_sheet_interface','source':str((root/'sw'/'5'/'01-ASSY-CHASSIS-EAR-L-R620.stp').resolve()),'placement':{'rotate_sequence':[{'axis_angle':top['axis_angle']}],'translate':top['t_mm']}}]})
 print(json.dumps({'candidate_count':len(cand),'top':cand[:3]},indent=2))
if __name__=='__main__':main()
