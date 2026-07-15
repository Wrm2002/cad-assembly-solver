"""Isolated STEP to auditable B-Rep graph and shape-map extractor."""
from __future__ import annotations
import argparse,json,math,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"joinable_migration_audit"))
from step_to_brep_graph_probe import extract_graph

def write(p,d):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def vec(x):return [float(v) for v in x.Coord()]
def worker(src,graph_out,map_out,feature_mode="full"):
    try:
        g=extract_graph(src)
        from OCC.Core.STEPControl import STEPControl_Reader
        from OCC.Core.TopTools import TopTools_IndexedMapOfShape
        from OCC.Core.TopExp import topexp_MapShapes
        from OCC.Core.TopAbs import TopAbs_FACE,TopAbs_EDGE,TopAbs_REVERSED,TopAbs_IN,TopAbs_ON
        from OCC.Core.TopoDS import topods
        from OCC.Core.BRepAdaptor import BRepAdaptor_Surface,BRepAdaptor_Curve
        from OCC.Core.BRepTools import breptools_UVBounds
        from OCC.Core.BRepLProp import BRepLProp_SLProps
        from OCC.Core.BRepClass import BRepClass_FaceClassifier
        from OCC.Core.gp import gp_Pnt2d
        from OCC.Core.Bnd import Bnd_Box
        from OCC.Core.BRepBndLib import brepbndlib_Add
        r=STEPControl_Reader();r.ReadFile(str(src));r.TransferRoots();shape=r.OneShape()
        fm=TopTools_IndexedMapOfShape();em=TopTools_IndexedMapOfShape()
        topexp_MapShapes(shape,TopAbs_FACE,fm);topexp_MapShapes(shape,TopAbs_EDGE,em)
        def bbox(s):
            b=Bnd_Box();brepbndlib_Add(s,b);x1,y1,z1,x2,y2,z2=b.Get()
            return {"min":[x1,y1,z1],"max":[x2,y2,z2]}
        for node in g["nodes"]:
            idx=node["occt_topology_index"]
            if node["entity_type"]=="face":
                f=topods.Face(fm.FindKey(idx));a=BRepAdaptor_Surface(f,True);node["bbox"]=bbox(f)
                pts=[];norms=[];mask=[];u1,u2,v1,v2=breptools_UVBounds(f)
                if feature_mode=="full" and all(math.isfinite(x) for x in (u1,u2,v1,v2)):
                    for iu in range(10):
                        u=u1+(u2-u1)*(iu+.5)/10
                        for iv in range(10):
                            v=v1+(v2-v1)*(iv+.5)/10;p=a.Value(u,v);pts+=vec(p)
                            cl=BRepClass_FaceClassifier(f,gp_Pnt2d(u,v),1e-7);mask.append(1 if cl.State() in (TopAbs_IN,TopAbs_ON) else 0)
                            pr=BRepLProp_SLProps(a,u,v,1,1e-7)
                            n=vec(pr.Normal()) if pr.IsNormalDefined() else [0,0,0]
                            if f.Orientation()==TopAbs_REVERSED:n=[-x for x in n]
                            norms+=n
                node["points"]=pts;node["normals"]=norms;node["trimming_mask"]=mask
                node["feature_sampling_status"]="success" if len(mask)==100 else "deferred_compact_mode"
            else:
                e=topods.Edge(em.FindKey(idx));a=BRepAdaptor_Curve(e);node["bbox"]=bbox(e)
                lo,hi=a.FirstParameter(),a.LastParameter();pts=[];tans=[]
                if feature_mode=="full" and math.isfinite(lo) and math.isfinite(hi):
                    for i in range(10):
                        u=lo+(hi-lo)*(i+.5)/10;pts+=vec(a.Value(u))
                        try:
                            t=vec(a.DN(u,1));ln=math.sqrt(sum(x*x for x in t));t=[x/ln for x in t] if ln else [0,0,0]
                        except Exception:t=[0,0,0]
                        tans+=t
                node["points"]=pts;node["tangents"]=tans;node.setdefault("dihedral_angle",None)
                node.setdefault("convexity","unknown_requires_local_adjacent_normals")
                node["feature_sampling_status"]="success" if len(pts)==30 else "deferred_compact_mode"
        shape_map={"schema_version":"1.0.0","part_id":g["part_id"],"source_step_path":g["source_step_path"],
          "source_geometry_sha256":g["source_geometry_sha256"],"id_stability_scope":"same SHA256 + OCCT build + import settings",
          "reverse_lookup":"reimport then IndexedMap.FindKey(occt_topology_index)",
          "entities":[{"node_id":n["node_id"],"entity_type":n["entity_type"],"occt_topology_index":n["occt_topology_index"],
                       "geometry_signature":n["geometry_signature"]} for n in g["nodes"]],
          "failure_reasons":[],"unavailable_fields":["serialized_TopoDS_Shape_handle"]}
        g["metadata"]["joinable_feature_contract"]={"face_grid_size":10,"edge_sample_count":10,
          "feature_mode":feature_mode,"points":feature_mode=="full","normals":feature_mode=="full",
          "trimming_mask":feature_mode=="full","tangents":feature_mode=="full",
          "exact_convexity":False,"exact_dihedral_angle":False}
        g["unavailable_fields"]=sorted(set(g["unavailable_fields"]+["exact_edge_convexity","exact_dihedral_angle"]))
        write(graph_out,g);write(map_out,shape_map);return 0
    except Exception as e:
        fail={"schema_version":"1.0.0","source_step_path":str(src.resolve()),"failure_reasons":[f"{type(e).__name__}:{e}"],"unavailable_fields":["brep_graph","shape_map"]}
        write(graph_out,fail);write(map_out,fail);return 2
def main():
    ap=argparse.ArgumentParser();ap.add_argument("inputs",nargs="*");ap.add_argument("--out-root",required=True)
    ap.add_argument("--report",default="brep_extraction_report.json");ap.add_argument("--worker",action="store_true")
    ap.add_argument("--worker-input");ap.add_argument("--worker-graph");ap.add_argument("--worker-map")
    ap.add_argument("--feature-mode",choices=["full","compact"],default="full");a=ap.parse_args()
    if a.worker:return worker(Path(a.worker_input),Path(a.worker_graph),Path(a.worker_map),a.feature_mode)
    out=Path(a.out_root);rows=[]
    for src in map(Path,a.inputs):
        case=src.parent.name;dst=out/f"case_{case}";graph=dst/f"{src.stem}.brep_graph.json";smap=dst/f"{src.stem}.shape_map.json"
        try:cp=subprocess.run([sys.executable,str(Path(__file__).resolve()),"--out-root",str(out),"--worker","--worker-input",str(src),"--worker-graph",str(graph),"--worker-map",str(smap)],timeout=900)
        except subprocess.TimeoutExpired:cp=None
        ok=graph.exists() and not json.loads(graph.read_text(encoding="utf-8")).get("failure_reasons")
        rows.append({"source":str(src.resolve()),"graph":str(graph.resolve()),"shape_map":str(smap.resolve()),"status":"success" if ok else "failed","failure_reasons":[] if ok else ["worker_timeout_or_failure"],"unavailable_fields":[] if ok else ["graph"]})
    rep={"schema_version":"1.0.0","attempted":len(rows),"success_count":sum(x["status"]=="success" for x in rows),"results":rows,
         "failure_reasons":[z for x in rows for z in x["failure_reasons"]],"unavailable_fields":sorted({z for x in rows for z in x["unavailable_fields"]})}
    write(a.report,rep);print(f"brep graphs={rep['success_count']}/{len(rows)}");return 0 if rep["success_count"]==len(rows) else 2
if __name__=="__main__":raise SystemExit(main())
