"""Apply a pose JSON to STEP parts and export transformed STL files (OCCT only)."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

def _part(value: str):
    key, raw = value.split("=", 1)
    path = Path(raw)
    if not key or not path.is_file(): raise argparse.ArgumentTypeError("invalid --part")
    return key, path

def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True); parser.add_argument('--part',action='append',type=_part,required=True)
    parser.add_argument('--output-dir',type=Path,required=True); parser.add_argument('--hypothesis-index',type=int,default=0)
    args=parser.parse_args(); data=json.loads(args.input.read_text(encoding='utf-8')); row=data['hypotheses'][args.hypothesis_index]
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.StlAPI import StlAPI_Writer
    from OCC.Core.gp import gp_Trsf
    args.output_dir.mkdir(parents=True,exist_ok=True); outputs=[]
    for key,path in dict(args.part).items():
        reader=STEPControl_Reader()
        if reader.ReadFile(str(path))!=IFSelect_RetDone: raise RuntimeError(f'cannot_read:{path}')
        reader.TransferRoots(); shape=reader.OneShape(); matrix=np.asarray(row['part_poses'][key],dtype=float)
        trsf=gp_Trsf(); trsf.SetValues(*[float(x) for x in matrix[:3,:4].reshape(-1)])
        transformed=BRepBuilderAPI_Transform(shape,trsf,True).Shape(); BRepMesh_IncrementalMesh(transformed,0.25,False,0.5,True); output=args.output_dir/f'{key}.stl'
        writer=StlAPI_Writer(); writer.SetASCIIMode(False); writer.Write(transformed,str(output)); outputs.append(str(output))
    print(json.dumps({'output_dir':str(args.output_dir.resolve()),'files':outputs,'exact_status':row.get('exact_validation',{}).get('status','not_checked')}))
    return 0
if __name__=='__main__': raise SystemExit(main())
