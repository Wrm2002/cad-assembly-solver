"""Quick STEP simplification - unify same-domain faces to reduce entity count."""
import sys, os, argparse
from pathlib import Path

def simplify_step(src: Path, dst: Path):
    """Read STEP, unify faces, write simplified STEP."""
    from OCC.Core.STEPControl import STEPControl_Reader, STEPControl_Writer, STEPControl_AsIs
    from OCC.Core.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_FACE
    from OCC.Core.BRepCheck import BRepCheck_Analyzer

    print(f"  Reading {src.name} ({src.stat().st_size/1024/1024:.1f} MB)...", flush=True)
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(src))
    if status != 1:
        print(f"    ERROR: Read failed")
        return False

    reader.TransferRoots()
    shape = reader.OneShape()

    # Count original faces
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    orig_faces = 0
    while exp.More():
        orig_faces += 1
        exp.Next()
    print(f"    Original faces: {orig_faces}", flush=True)

    # Unify same-domain faces
    print(f"    Unifying faces...", flush=True)
    unifier = ShapeUpgrade_UnifySameDomain(shape, True, True, True)
    unifier.Build()
    unified = unifier.Shape()

    # Count unified faces
    exp2 = TopExp_Explorer(unified, TopAbs_FACE)
    new_faces = 0
    while exp2.More():
        new_faces += 1
        exp2.Next()

    reduction = (1 - new_faces/orig_faces)*100 if orig_faces > 0 else 0
    print(f"    Unified faces: {new_faces} ({reduction:.0f}% reduction)", flush=True)

    # Validate
    analyzer = BRepCheck_Analyzer(unified)
    if not analyzer.IsValid():
        print(f"    WARNING: simplified shape invalid, using original")
        unified = shape

    # Write
    writer = STEPControl_Writer()
    writer.Transfer(unified, STEPControl_AsIs)
    writer.Write(str(dst))

    dst_size = dst.stat().st_size/1024/1024
    print(f"    Written: {dst.name} ({dst_size:.1f} MB)", flush=True)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('case_dir', type=Path)
    parser.add_argument('--min-mb', type=float, default=20.0, help='Only simplify files larger than this')
    args = parser.parse_args()

    case_dir = args.case_dir.resolve()
    simplified_dir = case_dir / 'simplified'
    simplified_dir.mkdir(exist_ok=True)

    step_files = sorted(
        list(case_dir.glob('*.step')) + list(case_dir.glob('*.stp')) + list(case_dir.glob('*.STP'))
    )
    step_files = [f for f in step_files if not f.stem.lower().startswith('assembly')]

    for f in step_files:
        size_mb = f.stat().st_size / 1024 / 1024
        dst = simplified_dir / (f.stem + '_simplified.step')

        if size_mb < args.min_mb:
            print(f"  {f.name}: {size_mb:.1f} MB → skipping (below {args.min_mb} MB threshold)")
            # Copy small files directly
            import shutil
            shutil.copy2(f, simplified_dir / f.name)
            continue

        if dst.exists():
            print(f"  {f.name}: already simplified, skipping")
            continue

        simplify_step(f, dst)

    print(f"\nSimplified files in: {simplified_dir}")


if __name__ == '__main__':
    main()
