"""Read-only SolidWorks COM bridge probe for one small STEP part.

The probe never saves a document and verifies the source hash before/after.
It exists to test the final CAD boundary without promoting any candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_faces(bodies: Any) -> tuple[int | None, int | None]:
    if bodies is None:
        return 0, 0
    try:
        body_rows = list(bodies)
    except TypeError:
        body_rows = [bodies]
    face_count = 0
    for body in body_rows:
        try:
            faces = body.GetFaces()
            face_count += len(list(faces)) if faces is not None else 0
        except Exception:
            return len(body_rows), None
    return len(body_rows), face_count


def _com_value(value: Any) -> Any:
    """Handle pywin32 members exposed as either methods or properties."""
    return value() if callable(value) else value


def probe(step_path: Path) -> dict[str, Any]:
    before = _sha256(step_path)
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "mode": "read_only_silent_probe",
        "source_path": str(step_path),
        "source_sha256_before": before,
        "solidworks_available": False,
        "solidworks_version": None,
        "document_opened": False,
        "body_count": None,
        "face_count": None,
        "source_hash_unchanged": None,
        "failure_reasons": [],
        "unavailable_fields": [],
    }
    app = None
    model = None
    launched_here = False
    title = None
    try:
        import pythoncom
        import win32com.client

        try:
            app = win32com.client.GetActiveObject(
                "SldWorks.Application"
            )
        except Exception:
            app = win32com.client.Dispatch("SldWorks.Application")
            launched_here = True
        app.Visible = False
        try:
            app.CommandInProgress = True
        except Exception:
            pass
        revision = app.RevisionNumber
        result["solidworks_version"] = str(
            revision() if callable(revision) else revision
        )
        result["solidworks_available"] = True
        # Official API guidance requires LoadFile4 for foreign STEP files.
        import_data = app.GetImportFileData(str(step_path))
        load_errors = win32com.client.VARIANT(
            pythoncom.VT_BYREF | pythoncom.VT_I4, 0
        )
        model = app.LoadFile4(
            str(step_path),
            "r",
            import_data,
            load_errors,
        )
        result["import_method"] = "ISldWorks.LoadFile4"
        result["load_errors"] = int(load_errors.value)
        if model is None:
            result["failure_reasons"].append(
                "OpenDoc6 returned no model for STEP input"
            )
        else:
            result["document_opened"] = True
            title = str(_com_value(model.GetTitle))
            bodies = model.GetBodies2(0, False)
            body_count, face_count = _count_faces(bodies)
            result["body_count"] = body_count
            result["face_count"] = face_count
            result["document_title"] = title
            result["document_type"] = int(_com_value(model.GetType))
    except Exception as exc:
        result["failure_reasons"].append(
            f"{type(exc).__name__}: {exc}"
        )
    finally:
        if app is not None:
            if title:
                try:
                    app.CloseDoc(title)
                except Exception as exc:
                    result["failure_reasons"].append(
                        f"CloseDoc failed: {type(exc).__name__}: {exc}"
                    )
            try:
                app.CommandInProgress = False
            except Exception:
                pass
            if launched_here:
                try:
                    app.ExitApp()
                except Exception as exc:
                    result["failure_reasons"].append(
                        f"ExitApp failed: {type(exc).__name__}: {exc}"
                    )
    after = _sha256(step_path)
    result["source_sha256_after"] = after
    result["source_hash_unchanged"] = before == after
    if not result["document_opened"]:
        result["unavailable_fields"].extend(
            ["body_count", "face_count"]
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "step_path",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent / "1" / "flange_part_a.step",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = probe(args.step_path.resolve())
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0 if result["solidworks_available"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
