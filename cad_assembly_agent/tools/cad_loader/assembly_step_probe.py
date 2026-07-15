"""Probe assembly hierarchy and occurrence locations from STEP/XCAF in workers."""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path

def write(p,d): Path(p).parent.mkdir(parents=True,exist_ok=True); Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def matrix(tr):
    m=tr.VectorialPart(); t=tr.TranslationPart()
    return [[m.Value(r,c) for c in range(1,4)]+[t.Coord()[r-1]] for r in range(1,4)]+[[0,0,0,1]]
def worker(src,out):
    try:
        from OCC.Core.STEPCAFControl import STEPCAFControl_Reader
        from OCC.Core.TDocStd import TDocStd_Document
        from OCC.Core.XCAFDoc import XCAFDoc_DocumentTool
        from OCC.Core.TDF import TDF_LabelSequence,TDF_Label
        doc=TDocStd_Document("assembly-probe"); tool=XCAFDoc_DocumentTool.ShapeTool(doc.Main())
        r=STEPCAFControl_Reader(); r.SetNameMode(True); status=r.ReadFile(str(src))
        if int(status)!=1 or not r.Transfer(doc): raise RuntimeError(f"xcaf_read_or_transfer_failed:{status}")
        roots=TDF_LabelSequence(); tool.GetFreeShapes(roots); comps=[]
        def walk(label,parent,path):
            seq=TDF_LabelSequence(); tool.GetComponents(label,seq)
            if tool.IsAssembly(label):
                for i in range(1,seq.Length()+1):
                    c=seq.Value(i); loc=tool.GetLocation(c); ref=TDF_Label(); referred=None
                    if tool.IsReference(c) and tool.GetReferredShape(c,ref): referred=ref
                    tr=matrix(loc.Transformation()); identity=all(abs(tr[i][j]-(1 if i==j else 0))<1e-10 for i in range(4) for j in range(4))
                    row={"component_id":"/".join(path+[str(i)]),"name":c.GetLabelName(),"parent":parent,
                         "location_matrix":tr,"location_is_identity":identity,
                         "is_reference":bool(tool.IsReference(c)),"failure_reasons":[],
                         "unavailable_fields":["source_part_filename"] if not c.GetLabelName() else []}
                    comps.append(row)
                    if referred is not None: walk(referred,row["component_id"],path+[str(i)])
        for i in range(1,roots.Length()+1): walk(roots.Value(i),None,[str(i)])
        all_identity=all(c["location_is_identity"] for c in comps)
        result={"schema_version":"1.0.0","source":str(src.resolve()),"free_shape_count":roots.Length(),
                "component_count":len(comps),"top_level_component_count":sum(c["parent"] is None for c in comps),"components":comps,
                "expected_pose_assessment":"geometry_baked_pose_only" if comps and all_identity else "occurrence_transforms_available",
                "can_use_as_expected_geometry":bool(comps),"can_recover_original_part_transforms_directly":bool(comps) and not all_identity,
                "failure_reasons":[],"unavailable_fields":["original_part_to_assembly_transform"] if all_identity else []}
        write(out,result); return 0
    except Exception as e:
        write(out,{"schema_version":"1.0.0","source":str(src.resolve()),"component_count":0,
                   "failure_reasons":[f"{type(e).__name__}:{e}"],"unavailable_fields":["assembly_hierarchy","component_transforms"]}); return 2
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("inputs",nargs="*"); ap.add_argument("--out-dir",default="assembly_step_probes")
    ap.add_argument("--report",default="assembly_step_probe_report.json"); ap.add_argument("--worker",action="store_true")
    ap.add_argument("--worker-input");ap.add_argument("--worker-output");a=ap.parse_args()
    if a.worker:return worker(Path(a.worker_input),Path(a.worker_output))
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True); rows=[]
    for i,p in enumerate(map(Path,a.inputs),1):
        dst=out/f"case_{i}.json"; cp=subprocess.run([sys.executable,str(Path(__file__).resolve()),"--worker","--worker-input",str(p),"--worker-output",str(dst)],timeout=300)
        rows.append(json.loads(dst.read_text(encoding="utf-8")) if dst.exists() else {"source":str(p),"failure_reasons":[f"worker_exit:{cp.returncode}"],"unavailable_fields":["probe"]})
    report={"schema_version":"1.0.0","case_count":len(rows),"success_count":sum(not x["failure_reasons"] for x in rows),
            "cases":rows,"failure_reasons":[r for x in rows for r in x["failure_reasons"]],
            "unavailable_fields":sorted({u for x in rows for u in x["unavailable_fields"]})}
    write(a.report,report);print(f"assembly probes={report['success_count']}/{len(rows)}");return 0 if report["success_count"]==len(rows) else 2
if __name__=="__main__":raise SystemExit(main())
