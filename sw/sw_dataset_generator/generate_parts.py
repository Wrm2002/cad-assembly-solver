from pathlib import Path

from .sw_api import create_primitive, save_document


def generate_parts(session, case_dir, spec):
    case_dir = Path(case_dir)
    native_dir, step_dir = case_dir / "native", case_dir / "step"
    native_dir.mkdir(parents=True, exist_ok=True)
    step_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for index, part_spec in enumerate(spec["parts"], start=1):
        stem = f"part_{index:02d}"
        doc = session.new_part()
        try:
            create_primitive(doc, part_spec)
            native = native_dir / f"{stem}.sldprt"
            step = step_dir / f"{stem}.step"
            save_document(doc, native, step)
            results.append({"native": native, "step": step})
        finally:
            session.close(doc)
    return results
