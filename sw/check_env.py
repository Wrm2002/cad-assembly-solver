"""Report whether the legacy CAD assembly environment is usable."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def _import_status(module: str) -> dict[str, Any]:
    try:
        imported = importlib.import_module(module)
        return {
            "available": True,
            "version": getattr(imported, "__version__", None),
            "error": None,
        }
    except Exception as exc:
        return {"available": False, "version": None, "error": str(exc)}


def _solidworks_status(skip: bool) -> dict[str, Any]:
    if skip:
        return {"available": None, "version": None, "error": "check skipped"}
    try:
        import win32com.client
    except Exception as exc:
        return {"available": False, "version": None, "error": str(exc)}

    app = None
    launched_here = False
    try:
        try:
            app = win32com.client.GetActiveObject("SldWorks.Application")
        except Exception:
            app = win32com.client.Dispatch("SldWorks.Application")
            launched_here = True
        revision = app.RevisionNumber if app is not None else None
        version = str(revision() if callable(revision) else revision)
        return {"available": True, "version": version, "error": None}
    except Exception as exc:
        return {"available": False, "version": None, "error": str(exc)}
    finally:
        if launched_here and app is not None:
            try:
                app.ExitApp()
            except Exception:
                pass


def _gpu_status() -> dict[str, Any]:
    torch_status = _import_status("torch")
    cuda = False
    gpu_name = None
    if torch_status["available"]:
        try:
            import torch

            cuda = bool(torch.cuda.is_available())
            if cuda:
                gpu_name = torch.cuda.get_device_name(0)
        except Exception as exc:
            torch_status["error"] = str(exc)

    nvidia_smi = shutil.which("nvidia-smi")
    driver_detected = False
    if nvidia_smi:
        try:
            result = subprocess.run(
                [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            driver_detected = result.returncode == 0
            if driver_detected and not gpu_name:
                gpu_name = result.stdout.strip().splitlines()[0]
        except Exception:
            pass

    return {
        "torch": torch_status["available"],
        "torch_version": torch_status["version"],
        "cuda": cuda,
        "device": "cuda" if cuda else "cpu",
        "nvidia_driver_detected": driver_detected,
        "gpu_name": gpu_name,
    }


def _numerical_runtime_status() -> dict[str, Any]:
    """Probe a delay-loaded LAPACK routine in an isolated process."""
    code = (
        "import numpy as np; "
        "u,s,vh=np.linalg.svd(np.eye(3)); "
        "assert u.shape==(3,3) and s.shape==(3,)"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        return {
            "available": result.returncode == 0,
            "returncode": result.returncode,
            "error": (result.stderr or result.stdout).strip() or None,
        }
    except Exception as exc:
        return {"available": False, "returncode": None, "error": str(exc)}


def _workdir_status(path: Path) -> dict[str, Any]:
    readable = path.is_dir() and os.access(path, os.R_OK)
    writable = False
    error = None
    if path.is_dir():
        try:
            with tempfile.NamedTemporaryFile(dir=path, prefix=".cad_env_", delete=True):
                writable = True
        except Exception as exc:
            error = str(exc)
    else:
        error = "directory does not exist"
    return {"path": str(path), "readable": readable, "writable": writable, "error": error}


def _find_step_files(path: Path) -> list[str]:
    files = sorted(
        p for p in path.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".step", ".stp"}
        and not p.name.lower().startswith("assembly")
    )
    return [str(p.relative_to(path)) for p in files]


def collect_report(workdir: Path, skip_solidworks: bool = False) -> dict[str, Any]:
    modules = {
        name: _import_status(module)
        for name, module in {
            "pythonocc": "OCC",
            "numpy": "numpy",
            "scipy": "scipy",
            "pandas": "pandas",
            "networkx": "networkx",
            "pywin32": "win32com.client",
        }.items()
    }
    gpu = _gpu_status()
    numerical_runtime = _numerical_runtime_status()
    solidworks = _solidworks_status(skip_solidworks)
    step_files = _find_step_files(workdir) if workdir.is_dir() else []

    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "pythonocc": modules["pythonocc"]["available"],
        "numpy": modules["numpy"]["available"],
        "scipy": modules["scipy"]["available"],
        "pandas": modules["pandas"]["available"],
        "networkx": modules["networkx"]["available"],
        "pywin32": modules["pywin32"]["available"],
        "solidworks_com": solidworks["available"],
        "solidworks_version": solidworks["version"],
        "solidworks_error": solidworks["error"],
        "numerical_runtime": numerical_runtime["available"],
        "numerical_runtime_returncode": numerical_runtime["returncode"],
        "numerical_runtime_error": numerical_runtime["error"],
        **gpu,
        "workdir": _workdir_status(workdir),
        "step_file_count": len(step_files),
        "step_files": step_files,
        "module_details": modules,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "workdir",
        nargs="?",
        default=str(Path(__file__).resolve().parent),
        help="Directory searched recursively for STEP test files.",
    )
    parser.add_argument(
        "--skip-solidworks",
        action="store_true",
        help="Do not attempt a SolidWorks COM connection.",
    )
    parser.add_argument("--output", help="Optional path for the JSON report.")
    args = parser.parse_args()

    report = collect_report(Path(args.workdir).resolve(), args.skip_solidworks)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
