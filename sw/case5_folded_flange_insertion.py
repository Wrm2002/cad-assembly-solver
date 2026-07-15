"""Folded-flange EAR insertion proposal: guide/flange first, holes last."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any
import numpy as np
from proxy_insertion_pose import _rotation_axis_angle

def load(p:Path)->Any:return json.loads(p.read_text(encoding='utf-8'))
def save(p:Path,x:Any)->None:p.write_text(json.dumps(x,indent=2),encoding='utf-8')
def u(a): a=np.asarray(a,float);return a/np.linalg.norm(a)
def unique(rows,t=.35):
 out=[]
 for r in rows:
  if not any(np.linalg.norm(np.asarray(r['centre'])-np.asarray(q['centre']))<t and abs(r['radius']-q['radius'])<.2 for q in out):out.append(r)
 return out
def bbox(payload,R,t):
 lo=np.asarray(payload['bbox_min'],float);hi=np.asarray(payload['bbox_max'],float);p=np.asarray([[x,y,z]for x in[lo[0],hi[0]]for y in[lo[1],hi[1]]for z in[lo[2],hi[2]]])@R.T+t;return p.min(0),p.max(0)
def main():
 a=argparse.ArgumentParser();a.add_argument('folder',type=Path);args=a.parse_args();o=args.folder
 E=load(o/'ear_holes_raw.json')['holes'];C=load(o/'chassis_holes_raw.json')['holes']; ep=load(o/'ear_planes_raw.json')['planes']; cp=load(o/'chassis_planes_raw.json')['planes']; eb=load(o/'ear_bbox.json');cb=load(o/'chassis_bbox.json')
 # A flange is a boundary-hosted hole group on a broad folded side face.  It
 # is detected, not named: the group has three+ small holes aligned along its
 # longest in-plane direction.
 src=unique([h for h in E if h['host_normal'][0]<-.98 and h['host_centre'][0]<-15 and h['radius']<2.5])
 targets=unique([h for h in C if h['host_normal'][0]>.98 and h['host_centre'][0]<-205 and h['radius']<2.5 and (h['host_bbox_max'][2]-h['host_bbox_min'][2])<150])
 # I/O exterior face: a broad plane normal +Z with a high aspect ratio.  The
 # carrier global thin dimension is Y, so an installed I/O face must face -Y.
 io=max([p for p in ep if p['normal'][2]>.98 and p['area_proxy']>500],key=lambda p:p['area_proxy'])
 # source flange normal -X -> target +X opposing contact, source long y ->
 # chassis z, and I/O +Z -> exterior -Y. This is a proper rigid rotation.
 R=np.array([[1,0,0],[0,0,-1],[0,1,0.]],float)
 P=np.asarray([R@np.asarray(h['centre'])for h in src]);Q=np.asarray([h['centre']for h in targets]);rp=np.asarray([h['radius']for h in src]);rq=np.asarray([h['radius']for h in targets])
 candidates=[]
 for i,p in enumerate(P):
  for j,q in enumerate(Q):
   if abs(rp[i]-rq[j])>.65:continue
   t=q-p;lo,hi=bbox(eb,R,t)
   io_y=float((R@np.asarray(io['origin'])+t)[1]); cmin=np.asarray(cb['bbox_min']);cmax=np.asarray(cb['bbox_max'])
   # external I/O but body can enter the chassis through the opening.
   if not (-3.0<=io_y<=3.0 and lo[1]<cmin[1]+2 and hi[1]<cmax[1]+3):continue
   D=np.linalg.norm(P[:,None,:]+t-Q[None,:,:],axis=2);M=(D<1.6)&(abs(rp[:,None]-rq[None,:])<.65);pairs=[(x,int(np.where(M[x])[0][0]),float(D[x,np.where(M[x])[0][0]]))for x in np.where(M.any(1))[0]];un=len(set(y for _,y,_ in pairs))
   # guide/stop evidence: a chassis Z-normal plane just beyond the inserted
   # long edge is a stop; this is independent of the holes.
   stops=[p for p in cp if abs(p['normal'][2])>.98 and p['area_proxy']>100 and hi[2]-2<=p['origin'][2]<=hi[2]+30]
   stop=min(stops,key=lambda p:abs(p['origin'][2]-hi[2])) if stops else None
   score=.35*min(1,un/3)+.30*max(0,1-np.mean([z for _,_,z in pairs])/2 if pairs else 0)+.20*(1 if stop else 0)+.15*max(0,1-abs(io_y)/3)
   candidates.append({'R':R.tolist(),'t_mm':t.round(6).tolist(),'axis_angle':_rotation_axis_angle(R),'flange_source_hole_count':len(src),'matched_holes':[{'ear_face':src[x]['face_index'],'chassis_face':targets[y]['face_index'],'residual_mm':round(z,4)}for x,y,z in pairs],'unique_target_hole_count':un,'mean_hole_residual_mm':round(float(np.mean([z for _,_,z in pairs])),4) if pairs else None,'io_face':io['face_index'],'io_face_y_mm':round(io_y,4),'bbox_mm':{'min':lo.round(4).tolist(),'max':hi.round(4).tolist()},'stop_face':stop['face_index'] if stop else None,'stop_gap_mm':round(float(stop['origin'][2]-hi[2]),4) if stop else None,'score':round(float(score),5)})
 candidates.sort(key=lambda x:(-x['unique_target_hole_count'],-x['score'],x['mean_hole_residual_mm'] or 9));save(o/'ear_folded_flange_candidates.json',{'method':'folded_flange_plus_external_io_plus_stop_then_holes','candidate_count':len(candidates),'candidates':candidates[:20]})
 if candidates:
  r=Path(__file__).resolve().parents[1];top=candidates[0];save(o/'ear_folded_flange_manifest.json',{'assembly_name':'case5_folded_flange_insertion_review','components':[{'id':'chassis_fixed','source':str((r/'sw'/'5'/'01-ASSY-CHASSIS-MODULE-R6250H0.stp').resolve()),'placement':{'translate':[0,0,0]}},{'id':'EAR_folded_flange','source':str((r/'sw'/'5'/'01-ASSY-CHASSIS-EAR-L-R620.stp').resolve()),'placement':{'rotate_sequence':[{'axis_angle':top['axis_angle']}],'translate':top['t_mm']}}]})
 print(json.dumps({'candidate_count':len(candidates),'top':candidates[:2]},indent=2))
if __name__=='__main__':main()
