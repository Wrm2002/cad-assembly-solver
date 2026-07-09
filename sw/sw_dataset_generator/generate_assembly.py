from pathlib import Path

from .sw_api import add_component, save_document


def generate_assembly(session, case_dir, part_results, spec):
    case_dir = Path(case_dir)
    doc = session.new_assembly()
    try:
        for index, result in enumerate(part_results, start=1):
            placement = spec["placements"][f"part_{index:02d}.step"]
            loaded = session.open_part(result["native"])
            try:
                session.activate(doc)
                add_component(doc, result["native"], placement["translate"])
            finally:
                session.close(loaded)
        rebuild = doc.EditRebuild3
        if callable(rebuild):
            rebuild()
        native = case_dir / "native" / "assembly.sldasm"
        step = case_dir / "step" / "assembly_gt.step"
        save_document(doc, native, step)
        return {"native": native, "step": step}
    finally:
        session.close(doc)
