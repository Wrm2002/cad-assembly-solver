"""Freeze manifest transforms and OCCT closest-support interface candidates.

The output is deliberately a *geometry-derived support set*, not a claim that
the selected supports are the designer's original mate entities.
"""
from __future__ import annotations
import argparse,json,subprocess,sys,warnings
from pathlib import Path
warnings.simplefilter("ignore")
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/"sw"))
def dump(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def load_shape(p):
 from OCC.Core.STEPControl import STEPControl_Reader
 r=STEPControl_Reader()
 if int(r.ReadFile(str(p)))!=1:raise RuntimeError("STEP_read_failed")
 r.TransferRoots();return r.OneShape()
def worker(a,b,pa,pb,out,placement_source="unknown"):
 try:
  from build_assembly import build_transform
  from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
  from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
  from OCC.Core.TopTools import TopTools_IndexedMapOfShape
  from OCC.Core.TopExp import topexp
  from OCC.Core.TopAbs import TopAbs_FACE,TopAbs_EDGE,TopAbs_VERTEX
  # Copy geometry so subshape maps and extrema supports share one coordinate
  # frame.  Located (copy=False) transformed STEP compounds can return support
  # vertices whose coordinates do not round-trip through IndexedMap.
  sa=BRepBuilderAPI_Transform(load_shape(a),build_transform(pa),True).Shape()
  sb=BRepBuilderAPI_Transform(load_shape(b),build_transform(pb),True).Shape()
  fm1,fm2,em1,em2,vm1,vm2=[TopTools_IndexedMapOfShape() for _ in range(6)]
  for s,t,m in (
   (sa,TopAbs_FACE,fm1),(sb,TopAbs_FACE,fm2),
   (sa,TopAbs_EDGE,em1),(sb,TopAbs_EDGE,em2),
   (sa,TopAbs_VERTEX,vm1),(sb,TopAbs_VERTEX,vm2),
  ):topexp.MapShapes(s,t,m)
  adj=[]
  def adjacency(shape,fm,em,vm):
   from OCC.Core.TopExp import TopExp_Explorer
   edge_to_faces={};vertex_to_edges={};vertex_to_faces={}
   for fi in range(1,fm.Size()+1):
    ex=TopExp_Explorer(fm.FindKey(fi),TopAbs_EDGE)
    while ex.More():
     ei=em.FindIndex(ex.Current())
     if ei:edge_to_faces.setdefault(ei,set()).add(fi)
     ex.Next()
    ex=TopExp_Explorer(fm.FindKey(fi),TopAbs_VERTEX)
    while ex.More():
     vi=vm.FindIndex(ex.Current())
     if vi:vertex_to_faces.setdefault(vi,set()).add(fi)
     ex.Next()
   for ei in range(1,em.Size()+1):
    ex=TopExp_Explorer(em.FindKey(ei),TopAbs_VERTEX)
    while ex.More():
     vi=vm.FindIndex(ex.Current())
     if vi:vertex_to_edges.setdefault(vi,set()).add(ei)
     ex.Next()
   return edge_to_faces,vertex_to_edges,vertex_to_faces
  ef1,ve1,vf1=adjacency(sa,fm1,em1,vm1)
  ef2,ve2,vf2=adjacency(sb,fm2,em2,vm2)
  dist=BRepExtrema_DistShapeShape(sa,sb);dist.Perform()
  if not dist.IsDone():raise RuntimeError("distance_solver_not_done")
  faces1=set();faces2=set();edges1=set();edges2=set();vertices1=set();vertices2=set();solutions=[]
  def nearest_vertex_index(s,vm,point=None):
   from OCC.Core.BRep import BRep_Tool
   def located_point(v):
    p=BRep_Tool.Pnt(v)
    return p.Transformed(v.Location().Transformation()) if not v.Location().IsIdentity() else p
   p=point or located_point(s);best=0;best_d2=float("inf")
   for vi in range(1,vm.Size()+1):
    q=located_point(vm.FindKey(vi))
    d2=(p.X()-q.X())**2+(p.Y()-q.Y())**2+(p.Z()-q.Z())**2
    if d2<best_d2:best,best_d2=vi,d2
   return best if best_d2 <= 1e-12 else 0
  def collect(t,s,point,fm,em,vm,ef,ve,vf,faces,edges,vertices):
   if t==2:
    fi=fm.FindIndex(s)
    if fi:faces.add(fi)
   elif t==1:
    ei=em.FindIndex(s)
    if ei:
     edges.add(ei);faces.update(ef.get(ei,()))
   elif t==0:
    vi=vm.FindIndex(s) or nearest_vertex_index(s,vm,point)
    if vi:
     vertices.add(vi);edges.update(ve.get(vi,()));faces.update(vf.get(vi,()))
    elif fm.Size()+em.Size() <= 5000:
     # Some OCCT STEP transforms return a generated extrema vertex that is
     # not IsSame() as any indexed topology vertex.  Recover incident
     # topology from the exact solution point instead of dropping the side.
     from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
     probe=BRepBuilderAPI_MakeVertex(point).Vertex()
     for ei in range(1,em.Size()+1):
      local=BRepExtrema_DistShapeShape(probe,em.FindKey(ei));local.Perform()
      if local.IsDone() and local.Value() <= 1e-6:
       edges.add(ei);faces.update(ef.get(ei,()))
     for fi in range(1,fm.Size()+1):
      local=BRepExtrema_DistShapeShape(probe,fm.FindKey(fi));local.Perform()
      if local.IsDone() and local.Value() <= 1e-6:faces.add(fi)
  for i in range(1,dist.NbSolution()+1):
   t1,t2=int(dist.SupportTypeShape1(i)),int(dist.SupportTypeShape2(i));s1=dist.SupportOnShape1(i);s2=dist.SupportOnShape2(i)
   p1,p2=dist.PointOnShape1(i),dist.PointOnShape2(i)
   collect(t1,s1,p1,fm1,em1,vm1,ef1,ve1,vf1,faces1,edges1,vertices1)
   collect(t2,s2,p2,fm2,em2,vm2,ef2,ve2,vf2,faces2,edges2,vertices2)
   solutions.append({"support_type_a":t1,"support_type_b":t2,"point_a":list(p1.Coord()),"point_b":list(p2.Coord())})
  r={"schema_version":"1.0.0","part_a":a.name,"part_b":b.name,"minimum_distance":dist.Value(),"solution_count":dist.NbSolution(),
     "candidate_interface_a":{"face_ids":sorted(x for x in faces1 if x),"edge_ids":sorted(x for x in edges1 if x),"vertex_ids":sorted(x for x in vertices1 if x)},
     "candidate_interface_b":{"face_ids":sorted(x for x in faces2 if x),"edge_ids":sorted(x for x in edges2 if x),"vertex_ids":sorted(x for x in vertices2 if x)},
     "solutions":solutions,"label_semantics":"geometry-derived closest/contact support; not designer-selected",
     "transform_a":pa,"transform_b":pb,"placement_source":placement_source,
     "pose_reference_status":"contact_or_overlap" if dist.Value() <= 1e-4 else "separated_not_interface_truth",
     "failure_reasons":[] if dist.Value() <= 1e-4 else [f"expected_pose_parts_separated_by_{dist.Value():.6g}_mm"],
     "unavailable_fields":["designer_selected_interface_id"]}
  dump(out,r);return 0
 except Exception as e:
  dump(out,{"failure_reasons":[f"{type(e).__name__}:{e}"],"unavailable_fields":["interface_candidates"]});return 2
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--benchmark",required=True);ap.add_argument("--source-root",required=True);ap.add_argument("--out-dir",required=True)
 ap.add_argument("--worker",action="store_true");ap.add_argument("--a");ap.add_argument("--b");ap.add_argument("--pa");ap.add_argument("--pb");ap.add_argument("--out")
 ap.add_argument("--placement-source",default="unknown");ap.add_argument("--case");ap.add_argument("--pair-index",type=int);ap.add_argument("--timeout",type=int,default=900);x=ap.parse_args()
 if x.worker:return worker(Path(x.a),Path(x.b),json.loads(x.pa),json.loads(x.pb),Path(x.out),x.placement_source)
 b=Path(x.benchmark);src=Path(x.source_root);out=Path(x.out_dir);rows=[]
 for tp in sorted((b/"manual_interfaces").glob("*.json")):
  d=json.loads(tp.read_text(encoding="utf-8"));case=d["case_id"]
  if x.case and case != str(x.case):continue
  manifest=src/case/"assembly_manifest.json";placements={};placement_source="identity_fallback"
  if manifest.exists():
   placement_source="assembly_manifest"
   for c in json.loads(manifest.read_text(encoding="utf-8"))["components"]:placements[c["source"]]=c.get("placement",{})
  elif case=="4":
   placement_source="recovered_from_baseline_build_log_identity"
  for name in d["parts"]:placements.setdefault(name,{"translate":[0,0,0]})
  for n,pair in enumerate(d["positive_part_pairs"],1):
   if x.pair_index and n != x.pair_index:continue
   a,bn=pair["parts"];dest=out/f"case_{case}"/f"pair_{n:02d}.json"
   cmd=[sys.executable,str(Path(__file__).resolve()),"--benchmark",str(b),"--source-root",str(src),"--out-dir",str(out),"--worker","--a",str(src/case/a),"--b",str(src/case/bn),"--pa",json.dumps(placements[a]),"--pb",json.dumps(placements[bn]),"--out",str(dest),"--placement-source",placement_source]
   try:
    cp=subprocess.run(cmd,timeout=x.timeout);ok=cp.returncode==0;reason=None if ok else f"worker_exit_{cp.returncode}"
   except subprocess.TimeoutExpired:
    ok=False;reason=f"worker_timeout_{x.timeout}s"
    dump(dest,{"schema_version":"1.0.0","part_a":a,"part_b":bn,"placement_source":placement_source,
      "failure_reasons":[reason],"unavailable_fields":["minimum_distance","interface_candidates"]})
   rows.append({"case_id":case,"parts":[a,bn],"output":str(dest.resolve()),"status":"success" if ok else "failed","failure_reason":reason})
 dump(out/"interface_candidate_summary.json",{"pair_count":len(rows),"success_count":sum(r["status"]=="success" for r in rows),"pairs":rows,
      "failure_reasons":[f"failed:{r['case_id']}:{r['parts']}" for r in rows if r["status"]!="success"],"unavailable_fields":["designer_selected_interface_id"]})
 print("pair truth",sum(r["status"]=="success" for r in rows),"/",len(rows))
if __name__=="__main__":raise SystemExit(main())
