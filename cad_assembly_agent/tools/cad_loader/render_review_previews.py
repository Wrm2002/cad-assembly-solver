"""Render lightweight multi-view PNG previews for manual review."""
from __future__ import annotations
import argparse,subprocess,sys
from pathlib import Path

def step_to_stl(src,dst):
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.StlAPI import StlAPI_Writer
    r=STEPControl_Reader()
    if int(r.ReadFile(str(src)))!=1: raise RuntimeError("STEP read failed")
    r.TransferRoots();s=r.OneShape();BRepMesh_IncrementalMesh(s,2.0,False,0.5,True).Perform()
    dst.parent.mkdir(parents=True,exist_ok=True)
    if not StlAPI_Writer().Write(s,str(dst)): raise RuntimeError("STL write failed")
def render(stl,png):
    import vtk
    reader=vtk.vtkSTLReader();reader.SetFileName(str(stl));reader.Update()
    bounds=reader.GetOutput().GetBounds();center=[(bounds[i*2]+bounds[i*2+1])/2 for i in range(3)]
    span=max(bounds[1]-bounds[0],bounds[3]-bounds[2],bounds[5]-bounds[4],1e-6);distance=span*2.2
    window=vtk.vtkRenderWindow();window.SetOffScreenRendering(1);window.SetSize(1600,1200)
    views=[((1,1,1),(0,0,1),(0,.5,1,1)),((0,-1,0),(0,0,1),(.5,.5,1,1)),
           ((0,0,1),(0,1,0),(0,0,.5,.5)),((1,0,0),(0,0,1),(.5,0,1,.5))]
    for direction,up,viewport in views:
        mapper=vtk.vtkPolyDataMapper();mapper.SetInputConnection(reader.GetOutputPort())
        actor=vtk.vtkActor();actor.SetMapper(mapper);actor.GetProperty().SetColor(.35,.65,.88)
        ren=vtk.vtkRenderer();ren.SetViewport(*viewport);ren.SetBackground(.96,.96,.96);ren.AddActor(actor)
        cam=ren.GetActiveCamera();cam.SetFocalPoint(*center);cam.SetPosition(*[center[i]+direction[i]*distance for i in range(3)]);cam.SetViewUp(*up)
        ren.ResetCamera();window.AddRenderer(ren)
    window.Render();capture=vtk.vtkWindowToImageFilter();capture.SetInput(window);capture.Update()
    writer=vtk.vtkPNGWriter();writer.SetFileName(str(png));writer.SetInputConnection(capture.GetOutputPort());writer.Write()
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--review-root",required=True);ap.add_argument("--worker-step");ap.add_argument("--worker-stl");a=ap.parse_args()
    if a.worker_step:step_to_stl(Path(a.worker_step),Path(a.worker_stl));return
    root=Path(a.review_root)
    for n in range(1,6):
        d=root/f"第{n}组";stl=d/"assembly.stl"
        if not stl.exists():
            subprocess.run([sys.executable,str(Path(__file__).resolve()),"--review-root",str(root),"--worker-step",str(d/"assembly.step"),"--worker-stl",str(stl)],check=True,timeout=900)
        render(stl,d/f"第{n}组_装配多视图.png");print("rendered",n)
if __name__=="__main__":main()
