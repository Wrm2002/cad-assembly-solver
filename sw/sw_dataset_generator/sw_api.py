"""Small, defensive wrapper around the SolidWorks COM API."""

from __future__ import annotations

from pathlib import Path

MM = 0.001
SW_DEFAULT_TEMPLATE_PART = 8
SW_DEFAULT_TEMPLATE_ASSEMBLY = 9
SW_SAVE_SILENT = 1


class SolidWorksSession:
    def __init__(self, visible=False):
        import win32com.client

        self.app = win32com.client.Dispatch("SldWorks.Application")
        self.app.Visible = bool(visible)

    def template(self, kind):
        value = self.app.GetUserPreferenceStringValue(
            SW_DEFAULT_TEMPLATE_PART if kind == "part" else SW_DEFAULT_TEMPLATE_ASSEMBLY
        )
        if not value:
            extension = "*.prtdot" if kind == "part" else "*.asmdot"
            roots = [
                Path(r"C:\ProgramData\SOLIDWORKS"),
                Path(r"C:\Program Files\SOLIDWORKS Corp"),
            ]
            candidates = [
                path
                for root in roots
                if root.exists()
                for path in root.rglob(extension)
                if "mbd" not in str(path).lower()
            ]
            value = str(candidates[0]) if candidates else ""
        if not value:
            raise RuntimeError(f"SolidWorks {kind} template was not found")
        return value

    def new_part(self):
        doc = self.app.NewDocument(self.template("part"), 0, 0.0, 0.0)
        if doc is None:
            raise RuntimeError("SolidWorks failed to create part document")
        return doc

    def new_assembly(self):
        doc = self.app.NewDocument(self.template("assembly"), 0, 0.0, 0.0)
        if doc is None:
            raise RuntimeError("SolidWorks failed to create assembly document")
        return doc

    def close(self, doc):
        if doc is not None:
            title = doc.GetTitle
            self.app.CloseDoc(title() if callable(title) else title)

    def open_part(self, path):
        import pythoncom
        import win32com.client

        errors = win32com.client.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)
        warnings = win32com.client.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)
        doc = self.app.OpenDoc6(
            str(Path(path).resolve()), 1, 1, "", errors, warnings
        )
        if doc is None:
            raise RuntimeError(f"failed to load component: errors={errors.value}")
        return doc

    def activate(self, doc):
        import pythoncom
        import win32com.client

        title = doc.GetTitle
        title = title() if callable(title) else title
        errors = win32com.client.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)
        active = self.app.ActivateDoc3(title, False, 0, errors)
        if active is None:
            raise RuntimeError(f"failed to activate {title}: errors={errors.value}")
        return active

    def quit(self):
        try:
            exit_app = self.app.ExitApp
            if callable(exit_app):
                exit_app()
        except Exception:
            # A crashed/disconnected SolidWorks server is already closed.
            pass


def _select_plane(doc):
    import pythoncom
    import win32com.client

    null_dispatch = win32com.client.VARIANT(pythoncom.VT_DISPATCH, None)
    for name in ("Front Plane", "前视基准面", "前基准面"):
        if doc.Extension.SelectByID2(
            name, "PLANE", 0.0, 0.0, 0.0, False, 0, null_dispatch, 0
        ):
            return
    raise RuntimeError("could not select the front sketch plane")


def _extrude(doc, depth_mm):
    feature = doc.FeatureManager.FeatureExtrusion2(
        True, False, False, 0, 0, float(depth_mm) * MM, 0.0,
        False, False, False, False, 0.0, 0.0,
        False, False, False, False, True, True, True, 0, 0.0, False,
    )
    if feature is None:
        raise RuntimeError("SolidWorks extrusion failed")


def create_primitive(doc, spec):
    _select_plane(doc)
    doc.SketchManager.InsertSketch(True)
    shape = spec["shape"]
    if shape == "cylinder":
        doc.SketchManager.CreateCircleByRadius(0, 0, 0, spec["radius"] * MM)
    elif shape == "ring":
        doc.SketchManager.CreateCircleByRadius(0, 0, 0, spec["outer_radius"] * MM)
        doc.SketchManager.CreateCircleByRadius(0, 0, 0, spec["inner_radius"] * MM)
    elif shape == "box":
        half_x = spec["width"] * MM / 2
        half_y = spec["height"] * MM / 2
        doc.SketchManager.CreateCenterRectangle(0, 0, 0, half_x, half_y, 0)
    elif shape == "housing_bore":
        half_x = spec["width"] * MM / 2
        half_y = spec["height"] * MM / 2
        doc.SketchManager.CreateCenterRectangle(0, 0, 0, half_x, half_y, 0)
        doc.SketchManager.CreateCircleByRadius(
            0, 0, 0, spec["bore_radius"] * MM
        )
    elif shape in {"flange", "plate"}:
        if shape == "flange":
            doc.SketchManager.CreateCircleByRadius(
                0, 0, 0, spec["outer_radius"] * MM
            )
            doc.SketchManager.CreateCircleByRadius(
                0, 0, 0, spec["inner_radius"] * MM
            )
        else:
            half_x = spec["width"] * MM / 2
            half_y = spec["height"] * MM / 2
            doc.SketchManager.CreateCenterRectangle(
                0, 0, 0, half_x, half_y, 0
            )
        import math

        for index in range(int(spec["bolt_count"])):
            angle = 2.0 * math.pi * index / int(spec["bolt_count"])
            x = spec["bolt_circle_radius"] * math.cos(angle) * MM
            y = spec["bolt_circle_radius"] * math.sin(angle) * MM
            doc.SketchManager.CreateCircleByRadius(
                x, y, 0, spec["bolt_hole_radius"] * MM
            )
    else:
        raise ValueError(f"unsupported primitive: {shape}")
    doc.SketchManager.InsertSketch(True)
    _extrude(doc, spec["depth"])
    rebuild = doc.EditRebuild3
    if callable(rebuild):
        rebuild()


def save_document(doc, native_path, step_path=None):
    import pythoncom
    import win32com.client

    def save(path):
        errors = win32com.client.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)
        warnings = win32com.client.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)
        export_data = win32com.client.VARIANT(pythoncom.VT_DISPATCH, None)
        ok = doc.Extension.SaveAs(
            str(Path(path).resolve()), 0, SW_SAVE_SILENT,
            export_data, errors, warnings
        )
        if not ok:
            raise RuntimeError(
                f"failed to save {path}: errors={errors.value}, warnings={warnings.value}"
            )

    native_path = str(Path(native_path).resolve())
    save(native_path)
    if step_path:
        step_path = str(Path(step_path).resolve())
        save(step_path)


def add_component(assembly, part_path, translation_mm):
    x, y, z = [float(value) * MM for value in translation_mm]
    component = assembly.AddComponent5(
        str(Path(part_path).resolve()), 0, "", False, "", x, y, z
    )
    if component is None:
        raise RuntimeError(f"failed to insert component {part_path}")
    return component
