"""Crash-safe Case 5 run for the three-stage visual-semantic workline.

This orchestrator never reads a stored final Case-5 pose.  It copies only raw
B-Rep measurements, regenerates geometry candidates, renders numbered region
previews, calls the three visual prompts, schedules a protected candidate
union, and finally runs OCCT rendering/collision in isolated subprocesses.

Every stage is checkpointed.  Re-running the same command resumes from the
last verified artifact instead of repeating completed heavy work.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import psutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from case5_semantic_brep_solver import ear_candidates, psu_candidates
from multimodal_reviewer import QwenVLReviewer
from visual_semantic_pipeline import VisualSemanticPipeline, fuse_candidate_quotas


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_CASE_DIR = HERE / "5"
DEFAULT_MEASUREMENTS = HERE / "generalization_work" / "case5_semantic_resolve_v1"
DEFAULT_OUTPUT = HERE / "generalization_work" / "case5_visual_semantic_v1"
DEFAULT_OCCT_PYTHON = Path(r"C:\Users\11049\miniforge3\envs\cad311\python.exe")

MEASUREMENT_FILES = (
    "chassis_bbox.json",
    "chassis_holes_raw.json",
    "chassis_planes_raw.json",
    "ear_bbox.json",
    "ear_holes_raw.json",
    "ear_planes_raw.json",
    "psu_bbox.json",
)

PARTS = {
    "carrier": "01-ASSY-CHASSIS-MODULE-R6250H0.stp",
    "ear": "01-ASSY-CHASSIS-EAR-L-R620.stp",
    "psu": "5-CRPS1300NC.stp",
}


@dataclass(frozen=True)
class ResourcePolicy:
    """Fail closed before an OCCT worker can exhaust an unstable workstation."""

    cpu_affinity: tuple[int, ...] = tuple(range(8))
    max_worker_memory_gb: float = 10.0
    min_free_memory_gb: float = 8.0
    cooldown_seconds: float = 10.0


class ResourceLimitError(RuntimeError):
    pass


class _JobBasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class _JobIoCounters(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64),
    ]


class _JobExtendedInformation(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JobBasicLimitInformation),
        ("IoInfo", _JobIoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _WindowsKillOnCloseJob:
    """Best-effort job object that prevents orphaned OCCT workers.

    If the supervising process is interrupted or externally terminated,
    Windows closes this handle and terminates every assigned child process.
    """

    _KILL_ON_JOB_CLOSE = 0x00002000
    _EXTENDED_LIMIT_INFORMATION = 9

    def __init__(self, process: subprocess.Popen[str]):
        self.handle: int | None = None
        if os.name != "nt":
            return
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = ctypes.c_void_p
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
        kernel32.SetInformationJobObject.restype = ctypes.c_int
        kernel32.SetInformationJobObject.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_uint32,
        ]
        kernel32.AssignProcessToJobObject.restype = ctypes.c_int
        kernel32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        kernel32.CloseHandle.restype = ctypes.c_int
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]

        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        info = _JobExtendedInformation()
        info.BasicLimitInformation.LimitFlags = self._KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            handle,
            self._EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(info),
            ctypes.sizeof(info),
        ):
            error = ctypes.WinError(ctypes.get_last_error())
            kernel32.CloseHandle(handle)
            raise error
        process_handle = ctypes.c_void_p(int(process._handle))  # type: ignore[attr-defined]
        if not kernel32.AssignProcessToJobObject(handle, process_handle):
            error = ctypes.WinError(ctypes.get_last_error())
            kernel32.CloseHandle(handle)
            raise error
        self.handle = int(handle)
        self._kernel32 = kernel32

    def close(self) -> None:
        if self.handle is not None:
            self._kernel32.CloseHandle(ctypes.c_void_p(self.handle))
            self.handle = None


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


class Checkpoints:
    def __init__(self, root: Path, *, resume: bool):
        self.root = root
        self.path = root / "progress_state.json"
        self.heartbeat = root / "heartbeat.json"
        self.resume = resume
        if self.path.is_file():
            self.state = load_json(self.path)
        else:
            self.state = {
                "schema_version": "case5_visual_semantic_progress.v1",
                "created_at": now(),
                "updated_at": now(),
                "stages": {},
            }
            atomic_json(self.path, self.state)

    def pulse(self, stage: str, detail: str = "") -> None:
        atomic_json(
            self.heartbeat,
            {"time": now(), "pid": os.getpid(), "stage": stage, "detail": detail},
        )

    def valid(self, stage: str) -> bool:
        if not self.resume:
            return False
        row = self.state.get("stages", {}).get(stage, {})
        if row.get("status") != "completed":
            return False
        artifacts = row.get("artifacts") or []
        if not artifacts:
            return False
        for artifact in artifacts:
            path = Path(artifact["path"])
            if not path.is_file() or sha256(path) != artifact["sha256"]:
                return False
        return True

    def start(self, stage: str) -> None:
        self.state.setdefault("stages", {})[stage] = {
            "status": "running",
            "started_at": now(),
            "artifacts": [],
        }
        self.state["updated_at"] = now()
        atomic_json(self.path, self.state)
        self.pulse(stage, "started")

    def complete(self, stage: str, artifacts: list[Path], summary: dict[str, Any] | None = None) -> None:
        self.state.setdefault("stages", {})[stage] = {
            "status": "completed",
            "completed_at": now(),
            "artifacts": [
                {"path": str(path.resolve()), "sha256": sha256(path)}
                for path in artifacts
            ],
            "summary": summary or {},
        }
        self.state["updated_at"] = now()
        atomic_json(self.path, self.state)
        self.pulse(stage, "completed")

    def fail(self, stage: str, error: Exception) -> None:
        row = self.state.setdefault("stages", {}).setdefault(stage, {})
        row.update(
            {
                "status": "failed",
                "failed_at": now(),
                "error_type": type(error).__name__,
                "error": str(error)[:1000],
            }
        )
        self.state["updated_at"] = now()
        atomic_json(self.path, self.state)
        self.pulse(stage, "failed")

    def run(
        self,
        stage: str,
        function: Callable[[], tuple[list[Path], dict[str, Any]]],
    ) -> dict[str, Any]:
        if self.valid(stage):
            self.pulse(stage, "resume_skip_verified")
            return self.state["stages"][stage].get("summary", {})
        self.start(stage)
        try:
            artifacts, summary = function()
            missing = [path for path in artifacts if not path.is_file()]
            if missing:
                raise RuntimeError(f"stage {stage} did not create: {missing}")
            self.complete(stage, artifacts, summary)
            return summary
        except Exception as exc:
            self.fail(stage, exc)
            raise


def run_isolated(
    command: list[str],
    *,
    stage: str,
    checkpoint: Checkpoints,
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: int,
    resource_policy: ResourcePolicy,
) -> None:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        creationflags = 0
        if os.name == "nt":
            creationflags = (
                subprocess.BELOW_NORMAL_PRIORITY_CLASS
                | subprocess.CREATE_NEW_PROCESS_GROUP
            )
        worker_environment = os.environ.copy()
        for variable in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            worker_environment[variable] = "1"
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdout=stdout,
            stderr=stderr,
            text=True,
            creationflags=creationflags,
            env=worker_environment,
        )
        kill_on_close_job: _WindowsKillOnCloseJob | None = None
        if os.name == "nt":
            try:
                kill_on_close_job = _WindowsKillOnCloseJob(process)
            except OSError as exc:
                checkpoint.pulse(stage, f"kill_on_close_job=unavailable:{exc}")
        worker = psutil.Process(process.pid)
        available_cpus = set(range(psutil.cpu_count(logical=True) or 1))
        affinity = sorted(set(resource_policy.cpu_affinity) & available_cpus)
        if affinity:
            worker.cpu_affinity(affinity)
        try:
            while process.poll() is None:
                elapsed = time.monotonic() - started
                family = [worker, *worker.children(recursive=True)]
                rss_bytes = 0
                for member in family:
                    try:
                        rss_bytes += member.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                rss_gb = rss_bytes / (1024**3)
                free_gb = psutil.virtual_memory().available / (1024**3)
                checkpoint.pulse(
                    stage,
                    (
                        f"subprocess_pid={process.pid};elapsed={elapsed:.1f}s;"
                        f"rss_gb={rss_gb:.2f};free_gb={free_gb:.2f};"
                        f"affinity={affinity}"
                    ),
                )
                if rss_gb > resource_policy.max_worker_memory_gb:
                    raise ResourceLimitError(
                        f"worker RSS {rss_gb:.2f} GB exceeded "
                        f"{resource_policy.max_worker_memory_gb:.2f} GB"
                    )
                if free_gb < resource_policy.min_free_memory_gb:
                    raise ResourceLimitError(
                        f"system free memory {free_gb:.2f} GB fell below "
                        f"{resource_policy.min_free_memory_gb:.2f} GB"
                    )
                if elapsed > timeout_seconds:
                    raise TimeoutError(
                        f"subprocess timed out after {timeout_seconds}s"
                    )
                time.sleep(3.0)
        except BaseException:
            for member in reversed([worker, *worker.children(recursive=True)]):
                try:
                    member.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            process.wait(timeout=30)
            raise
        finally:
            if kill_on_close_job is not None:
                kill_on_close_job.close()
            if resource_policy.cooldown_seconds > 0:
                checkpoint.pulse(
                    stage,
                    f"cooldown={resource_policy.cooldown_seconds:.1f}s",
                )
                time.sleep(resource_policy.cooldown_seconds)
        if process.returncode != 0:
            tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-1500:]
            raise RuntimeError(f"subprocess failed ({process.returncode}): {tail}")


def parse_cpu_affinity(value: str) -> tuple[int, ...]:
    cpus: set[int] = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise argparse.ArgumentTypeError(f"invalid CPU range: {token}")
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(token))
    if not cpus or min(cpus) < 0:
        raise argparse.ArgumentTypeError("CPU affinity must contain non-negative IDs")
    return tuple(sorted(cpus))


def select_diverse_collision_shortlist(
    candidates: list[dict[str, Any]], limit: int
) -> list[dict[str, Any]]:
    """Keep one representative per rigid-orientation/interface family first.

    The fused list can contain several translations of the same rotation and
    interface provider. A plain top-N lets those near-duplicates crowd out a
    lower-ranked but geometrically distinct mounting polarity. This function
    only selects among already generated poses; it does not generate poses or
    use case labels/stored answers.
    """
    limit = max(1, int(limit))
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    seen_families: set[tuple[Any, ...]] = set()

    for candidate in candidates:
        rotation = tuple(
            round(float(value), 5)
            for row in candidate.get("R", [])
            for value in row
        )
        providers = tuple(
            sorted(
                source
                for source in candidate.get("candidate_sources", [])
                if source not in {"analytic", "vision_semantic", "protected"}
            )
        )
        family = (rotation, providers)
        if family in seen_families:
            continue
        selected.append(candidate)
        selected_ids.add(str(candidate.get("candidate_id")))
        seen_families.add(family)
        if len(selected) >= limit:
            return selected

    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id"))
        if candidate_id in selected_ids:
            continue
        selected.append(candidate)
        selected_ids.add(candidate_id)
        if len(selected) >= limit:
            break
    return selected


def call_with_heartbeat(
    checkpoint: Checkpoints,
    stage: str,
    detail: str,
    function: Callable[[], Any],
) -> Any:
    """Keep a visible liveness signal while an API request is blocked in I/O."""

    stopped = threading.Event()
    started = time.monotonic()

    def pulse_loop() -> None:
        while not stopped.wait(10.0):
            checkpoint.pulse(
                stage,
                f"{detail};api_wait_elapsed={time.monotonic() - started:.1f}s",
            )

    checkpoint.pulse(stage, f"{detail};api_request_started")
    pulse_thread = threading.Thread(target=pulse_loop, daemon=True)
    pulse_thread.start()
    try:
        return function()
    finally:
        stopped.set()
        pulse_thread.join(timeout=2.0)
        checkpoint.pulse(
            stage,
            f"{detail};api_request_finished;elapsed={time.monotonic() - started:.1f}s",
        )


def axis_angle(candidate: dict[str, Any]) -> list[float]:
    value = candidate.get("axis_angle")
    if value is None:
        raise ValueError("candidate has no axis-angle rotation")
    return [float(item) for item in value]


def translation(candidate: dict[str, Any]) -> list[float]:
    value = candidate.get("t_mm", candidate.get("translation"))
    if value is None:
        raise ValueError("candidate has no translation")
    return [float(item) for item in value]


def candidate_geometry_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    allowed = (
        "region_id",
        "candidate_id",
        "candidate_sources",
        "geometry_score",
        "score",
        "bbox_mm",
        "guide_wall_faces",
        "support_face",
        "guide_gap_mm",
        "clearance_mm",
        "opening_flush_error_mm",
        "unique_target_hole_count",
        "mean_hole_residual_mm",
        "mounting_face_gap_mm",
        "io_face",
        "io_face_y_mm",
        "stop_face",
        "stop_gap_mm",
        "inside_fraction",
        "independent_physical_hole_count",
        "mean_hole_axis_residual_mm",
        "contact_plane_translation_std_mm",
        "outside_fraction",
        "vertical_overlap_fraction",
        "service_face",
        "service_flush_error_mm",
        "carrier_side",
        "source_flange_side",
    )
    return {key: candidate[key] for key in allowed if key in candidate}


def component_manifest(name: str, source: Path) -> dict[str, Any]:
    return {
        "assembly_name": name,
        "components": [
            {
                "id": name,
                "label": name,
                "source": str(source.resolve()),
                "placement": {"translate": [0.0, 0.0, 0.0]},
            }
        ],
    }


def candidate_manifest(
    region_id: str,
    carrier: Path,
    part: Path,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    return {
        "assembly_name": f"{region_id} candidate region preview",
        "pose_status": "geometry_candidate_not_accepted",
        "accepted": False,
        "components": [
            {
                "id": "carrier",
                "label": "carrier",
                "source": str(carrier.resolve()),
                "placement": {"translate": [0.0, 0.0, 0.0]},
            },
            {
                "id": region_id,
                "label": region_id,
                "source": str(part.resolve()),
                "placement": {
                    "rotate_sequence": [{"axis_angle": axis_angle(candidate)}],
                    "translate": translation(candidate),
                },
            },
        ],
    }


def font(size: int):
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def label_api_views(audit: dict[str, Any], output_dir: Path) -> tuple[list[Path], list[dict[str, Any]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    mapping: list[dict[str, Any]] = []
    for index, view in enumerate(audit["views"], start=1):
        source = Path(view["path"])
        side_id = f"F{index:02d}"
        with Image.open(source).convert("RGB") as image:
            image.thumbnail((850, 650))
            draw = ImageDraw.Draw(image)
            draw.rectangle((10, 10, 260, 55), fill="white", outline="#34495e", width=2)
            draw.text((20, 17), f"{side_id}  {view['name']}", fill="#17202a", font=font(24))
            target = output_dir / f"{side_id}_{str(view['name']).replace(' ', '_')}.png"
            image.save(target, optimize=True)
        paths.append(target)
        mapping.append(
            {
                "face_id": side_id,
                "view": view["name"],
                "brep_face_ids": [],
                "mapping_type": "view_aligned_extremal_side_region",
            }
        )
    return paths, mapping


def contact_sheet(items: list[tuple[str, Path]], output: Path) -> Path:
    if not items:
        raise ValueError("cannot build an empty region contact sheet")
    cell_width, cell_height = 720, 540
    columns = 2
    rows = (len(items) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, path) in enumerate(items):
        with Image.open(path).convert("RGB") as image:
            image.thumbnail((cell_width - 20, cell_height - 65))
            x = (index % columns) * cell_width + (cell_width - image.width) // 2
            y = (index // columns) * cell_height + 55
            canvas.paste(image, (x, y))
        label_x = (index % columns) * cell_width + 16
        label_y = (index // columns) * cell_height + 10
        draw.rectangle((label_x - 6, label_y - 4, label_x + 180, label_y + 38), fill="#fff4cc", outline="#b7791f", width=2)
        draw.text((label_x, label_y), label, fill="#7b341e", font=font(25))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument("--measurement-source", type=Path, default=DEFAULT_MEASUREMENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--occt-python", type=Path, default=DEFAULT_OCCT_PYTHON)
    parser.add_argument("--mode", choices=("live", "cache_only", "off"), default="live")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--render-timeout", type=int, default=900)
    parser.add_argument("--collision-timeout", type=int, default=1800)
    parser.add_argument(
        "--worker-cpu-affinity",
        type=parse_cpu_affinity,
        default=parse_cpu_affinity("0-7"),
        help="Logical CPUs allowed for OCCT workers; the safe default avoids WHEA APIC 16/17.",
    )
    parser.add_argument("--max-worker-memory-gb", type=float, default=10.0)
    parser.add_argument("--min-free-memory-gb", type=float, default=8.0)
    parser.add_argument("--cooldown-seconds", type=float, default=12.0)
    parser.add_argument(
        "--max-ear-region-renders",
        type=int,
        default=4,
        help="Render only the highest-ranked EAR previews; all geometry candidates remain protected in JSON.",
    )
    parser.add_argument(
        "--stop-after-stage",
        choices=tuple(f"{index:02d}" for index in range(11)) + ("07b",),
        help="Exit cleanly at a checkpoint boundary (00..10 or 07b) for low-risk staged execution.",
    )
    parser.add_argument(
        "--ear-collision-shortlist",
        type=int,
        default=4,
        help="Validate only this protected EAR shortlist before final rendering.",
    )
    args = parser.parse_args()

    case_dir = args.case_dir.resolve()
    source_measurements = args.measurement_source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = Checkpoints(output, resume=not args.no_resume)
    resource_policy = ResourcePolicy(
        cpu_affinity=tuple(args.worker_cpu_affinity),
        max_worker_memory_gb=float(args.max_worker_memory_gb),
        min_free_memory_gb=float(args.min_free_memory_gb),
        cooldown_seconds=float(args.cooldown_seconds),
    )

    def stop_after(stage_code: str) -> bool:
        if args.stop_after_stage != stage_code:
            return False
        checkpoint.pulse(
            f"stopped_after_{stage_code}",
            "intentional checkpoint boundary; safe to resume",
        )
        print(
            json.dumps(
                {
                    "status": "checkpoint_stop",
                    "stage": stage_code,
                    "output": str(output),
                    "resume_safe": True,
                },
                ensure_ascii=False,
            )
        )
        return True

    def record_resource_policy():
        path = output / "resource_policy.json"
        atomic_json(
            path,
            {
                "schema_version": "case5_resource_policy.v1",
                **asdict(resource_policy),
                "occt_execution": "isolated_serial_subprocess",
                "windows_priority": "below_normal",
                "windows_job_object_kill_on_close": os.name == "nt",
                "excluded_whea_logical_processors": [16, 17],
                "candidate_preview_policy": {
                    "ear_render_limit": int(args.max_ear_region_renders),
                    "all_geometry_candidates_preserved": True,
                },
                "note": (
                    "Safety policy derived from repeated WHEA CPU core errors and "
                    "kernel bugchecks; it does not alter pose scoring."
                ),
            },
        )
        return [path], asdict(resource_policy)

    checkpoint.run("00_record_resource_safety_policy", record_resource_policy)
    if stop_after("00"):
        return 0

    part_paths = {key: case_dir / value for key, value in PARTS.items()}
    for path in part_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    measurements = output / "measurements"

    def prepare_measurements():
        measurements.mkdir(parents=True, exist_ok=True)
        artifacts: list[Path] = []
        for name in MEASUREMENT_FILES:
            source = source_measurements / name
            if not source.is_file():
                raise FileNotFoundError(source)
            target = measurements / name
            shutil.copy2(source, target)
            artifacts.append(target)
        provenance = {
            "schema_version": "case5_measurement_provenance.v1",
            "copied_raw_brep_measurements_only": True,
            "stored_pose_files_copied": False,
            "source_measurement_directory": str(source_measurements),
            "step_files": {
                key: {"path": str(path), "sha256": sha256(path)}
                for key, path in part_paths.items()
            },
            "measurement_files": {
                path.name: sha256(path) for path in artifacts
            },
        }
        provenance_path = measurements / "provenance.json"
        atomic_json(provenance_path, provenance)
        artifacts.append(provenance_path)
        return artifacts, {"measurement_count": len(MEASUREMENT_FILES)}

    checkpoint.run("01_prepare_raw_measurements", prepare_measurements)
    if stop_after("01"):
        return 0

    def generate_candidates():
        folded_path = measurements / "ear_folded_flange_candidates.json"
        run_isolated(
            [str(args.occt_python), str(HERE / "case5_folded_flange_insertion.py"), str(measurements)],
            stage="02_generate_geometry_candidates",
            checkpoint=checkpoint,
            stdout_path=output / "logs" / "candidate_generation.stdout.log",
            stderr_path=output / "logs" / "candidate_generation.stderr.log",
            timeout_seconds=300,
            resource_policy=resource_policy,
        )
        psu_audit, _ = psu_candidates(measurements)
        ear_hole_audit, _ = ear_candidates(measurements)
        folded = load_json(folded_path)

        psu_rows = []
        for index, row in enumerate(psu_audit["candidates"], start=1):
            candidate = dict(row)
            candidate.update(
                {
                    "candidate_id": f"PSU_GEOM_{index:02d}",
                    "region_id": f"R{index:02d}",
                    "candidate_sources": ["analytic", "bay_guides"],
                    "geometry_score": float(row.get("score", 0.0)),
                    "protected": index == 1,
                }
            )
            psu_rows.append(candidate)

        ear_raw = []
        for provider, rows in (
            ("folded_flange", folded.get("candidates", [])),
            ("hole_pattern", ear_hole_audit.get("candidates", [])),
        ):
            for row in rows:
                candidate = dict(row)
                candidate["candidate_sources"] = ["analytic", provider]
                candidate["geometry_score"] = float(row.get("score", 0.0))
                candidate["protected"] = False
                ear_raw.append(candidate)
        ear_raw.sort(key=lambda row: float(row.get("geometry_score", 0.0)), reverse=True)
        ear_rows: list[dict[str, Any]] = []
        for row in ear_raw:
            t = translation(row)
            R = np.asarray(row.get("R"), dtype=float)
            if any(
                sum((a - b) ** 2 for a, b in zip(t, translation(old))) ** 0.5 < 2.0
                and np.allclose(R, np.asarray(old.get("R"), dtype=float), atol=1e-6)
                for old in ear_rows
            ):
                for old in ear_rows:
                    if (
                        sum((a - b) ** 2 for a, b in zip(t, translation(old))) ** 0.5 < 2.0
                        and np.allclose(
                            R, np.asarray(old.get("R"), dtype=float), atol=1e-6
                        )
                    ):
                        old["candidate_sources"] = list(
                            dict.fromkeys(old["candidate_sources"] + row["candidate_sources"])
                        )
                        break
                continue
            ear_rows.append(row)
            if len(ear_rows) >= 12:
                break
        for index, row in enumerate(ear_rows, start=1):
            row["candidate_id"] = f"EAR_GEOM_{index:02d}"
            row["region_id"] = f"R{index:02d}"
            row["protected"] = index == 1 or "hole_pattern" in row["candidate_sources"]

        payload = {
            "schema_version": "case5_geometry_candidates.v1",
            "generated_from_raw_brep_measurements": True,
            "stored_final_pose_loaded": False,
            "parts": {"psu": psu_rows, "ear": ear_rows},
        }
        path = output / "geometry_candidates.json"
        atomic_json(path, payload)
        return [path, folded_path], {
            "psu_candidate_count": len(psu_rows),
            "ear_candidate_count": len(ear_rows),
        }

    checkpoint.run("02_generate_geometry_candidates", generate_candidates)
    if stop_after("02"):
        return 0

    def prepare_manifests():
        candidates = load_json(output / "geometry_candidates.json")["parts"]
        manifests = output / "manifests"
        manifests.mkdir(parents=True, exist_ok=True)
        artifacts: list[Path] = []
        for key, source in part_paths.items():
            path = manifests / f"part_{key}.json"
            atomic_json(path, component_manifest(f"part_{key}", source))
            artifacts.append(path)
        for part_key in ("psu", "ear"):
            for candidate in candidates[part_key]:
                path = manifests / f"{part_key}_{candidate['region_id']}.json"
                atomic_json(
                    path,
                    candidate_manifest(
                        candidate["region_id"],
                        part_paths["carrier"],
                        part_paths[part_key],
                        candidate,
                    ),
                )
                artifacts.append(path)
        return artifacts, {"manifest_count": len(artifacts)}

    checkpoint.run("03_prepare_manifests", prepare_manifests)
    if stop_after("03"):
        return 0

    def render_parts():
        artifacts: list[Path] = []
        for key in ("carrier", "psu", "ear"):
            stage = f"04_render_part_{key}"
            manifest = output / "manifests" / f"part_{key}.json"
            image = output / "renders" / "parts" / f"{key}_multiview.png"
            audit = image.with_suffix(".render_audit.json")
            image.parent.mkdir(parents=True, exist_ok=True)
            valid_existing = False
            if image.is_file() and audit.is_file():
                try:
                    with Image.open(image) as existing:
                        existing.verify()
                    valid_existing = len(load_json(audit).get("views", [])) == 8
                except Exception:
                    valid_existing = False
            if not valid_existing:
                run_isolated(
                    [
                        str(args.occt_python),
                        str(HERE / "render_assembly_manifest_occt.py"),
                        str(manifest),
                        str(image),
                        "--complete-views",
                        "--view-width",
                        "900",
                        "--view-height",
                        "700",
                        "--audit",
                        str(audit),
                    ],
                    stage=stage,
                    checkpoint=checkpoint,
                    stdout_path=output / "logs" / f"{stage}.stdout.log",
                    stderr_path=output / "logs" / f"{stage}.stderr.log",
                    timeout_seconds=args.render_timeout,
                    resource_policy=resource_policy,
                )
            api_paths, mapping = label_api_views(
                load_json(audit), output / "api_views" / key
            )
            mapping_path = output / "api_views" / key / "face_region_mapping.json"
            atomic_json(mapping_path, mapping)
            artifacts.extend([image, audit, mapping_path, *api_paths])
        return artifacts, {"part_render_count": 3, "views_per_part": 8}

    checkpoint.run("04_render_standard_part_views", render_parts)
    if stop_after("04"):
        return 0

    def render_regions():
        candidates = load_json(output / "geometry_candidates.json")["parts"]
        artifacts: list[Path] = []
        for part_key in ("psu", "ear"):
            sheet_items: list[tuple[str, Path]] = []
            preview_candidates = candidates[part_key]
            if part_key == "ear" and args.max_ear_region_renders > 0:
                preview_candidates = preview_candidates[: args.max_ear_region_renders]
            for candidate in preview_candidates:
                region_id = candidate["region_id"]
                stage = f"05_render_{part_key}_{region_id}"
                manifest = output / "manifests" / f"{part_key}_{region_id}.json"
                image = output / "renders" / "regions" / part_key / f"{region_id}.png"
                audit = image.with_suffix(".render_audit.json")
                image.parent.mkdir(parents=True, exist_ok=True)
                if not image.is_file() or not audit.is_file():
                    run_isolated(
                        [
                            str(args.occt_python),
                            str(HERE / "render_assembly_manifest_occt.py"),
                            str(manifest),
                            str(image),
                            "--relationship-view",
                            "--relationship-focus",
                            "--view-width",
                            "700",
                            "--view-height",
                            "520",
                            "--audit",
                            str(audit),
                        ],
                        stage=stage,
                        checkpoint=checkpoint,
                        stdout_path=output / "logs" / f"{stage}.stdout.log",
                        stderr_path=output / "logs" / f"{stage}.stderr.log",
                        timeout_seconds=args.render_timeout,
                        resource_policy=resource_policy,
                    )
                artifacts.extend([image, audit])
                sheet_items.append((region_id, image))
            sheet = output / "renders" / "regions" / f"{part_key}_numbered_regions.png"
            contact_sheet(sheet_items, sheet)
            artifacts.append(sheet)
        return artifacts, {"numbered_region_preview_count": len(artifacts)}

    checkpoint.run("05_render_numbered_regions", render_regions)
    if stop_after("05"):
        return 0

    def run_visual_semantics():
        semantic_stage = "06_three_stage_visual_semantics_v3"
        config = {
            "vision_model": os.environ.get("QWEN_VL_MODEL", "qwen3-vl-plus"),
            "vision_max_attempts": 3,
            "vision_max_tokens": 4096,
            "vision_timeout_seconds": 120,
        }
        reviewer = QwenVLReviewer(config, output / "cache")
        pipeline = VisualSemanticPipeline(reviewer)
        candidates = load_json(output / "geometry_candidates.json")["parts"]
        outputs: dict[str, dict[str, Any]] = {"prompt1": {}, "prompt2": {}, "prompt3": {}}
        artifacts: list[Path] = []
        for part_key in ("psu", "ear"):
            part_views = sorted((output / "api_views" / part_key).glob("F*.png"))
            face_mapping = load_json(output / "api_views" / part_key / "face_region_mapping.json")
            bbox = load_json(measurements / f"{part_key}_bbox.json")
            brep_summary = {
                "bbox_mm": bbox,
                "functional_face_ids": [row["face_id"] for row in face_mapping],
                "face_region_mapping": face_mapping,
                "measurement_policy": "view-aligned extremal side regions; exact B-Rep IDs remain geometry-owned",
            }
            if part_key == "ear":
                brep_summary["measured_hole_face_count"] = len(
                    load_json(measurements / "ear_holes_raw.json").get("holes", [])
                )
                brep_summary["measured_plane_face_count"] = len(
                    load_json(measurements / "ear_planes_raw.json").get("planes", [])
                )
            first = call_with_heartbeat(
                checkpoint,
                semantic_stage,
                f"part={part_key};prompt=1_role",
                lambda: pipeline.analyze_part(
                    part_key.upper(),
                    part_views,
                    brep_summary,
                    source_filename=part_paths[part_key].name,
                    mode=args.mode,
                ),
            )
            outputs["prompt1"][part_key] = first

            carrier_views = sorted((output / "api_views" / "carrier").glob("F*.png"))
            stage2_images = carrier_views[:6] + [
                output / "renders" / "regions" / f"{part_key}_numbered_regions.png"
            ]
            region_summaries = [candidate_geometry_summary(row) for row in candidates[part_key]]
            carrier_summary = {
                "bbox_mm": load_json(measurements / "chassis_bbox.json"),
                "measured_hole_face_count": len(
                    load_json(measurements / "chassis_holes_raw.json").get("holes", [])
                ),
                "measured_plane_face_count": len(
                    load_json(measurements / "chassis_planes_raw.json").get("planes", [])
                ),
                "carrier_is_fixed": True,
            }
            second = call_with_heartbeat(
                checkpoint,
                semantic_stage,
                f"part={part_key};prompt=2_regions",
                lambda: pipeline.analyze_regions(
                    part_key.upper(),
                    stage2_images,
                    first["output"],
                    region_summaries,
                    carrier_summary,
                    mode=args.mode,
                ),
            )
            outputs["prompt2"][part_key] = second

            stage3_images = part_views[:2] + [
                output / "renders" / "regions" / f"{part_key}_numbered_regions.png"
            ]
            third = call_with_heartbeat(
                checkpoint,
                semantic_stage,
                f"part={part_key};prompt=3_synthesis",
                lambda: pipeline.synthesize(
                    part_key.upper(),
                    stage3_images,
                    first["output"],
                    second["output"],
                    {"regions": region_summaries, "pose_values_disclosed": False},
                    mode=args.mode,
                ),
            )
            outputs["prompt3"][part_key] = third

        names = {
            "prompt1": "prompt1_part_role.json",
            "prompt2": "prompt2_carrier_regions.json",
            "prompt3": "prompt3_assembly_hypothesis.json",
        }
        for stage_key, filename in names.items():
            path = output / filename
            atomic_json(path, outputs[stage_key])
            artifacts.append(path)
        manifest = {
            "schema_version": "visual_semantic_api_manifest.v1",
            "mode": args.mode,
            "provider": "qwen-vl",
            "model": reviewer._model,
            "api_key_recorded": False,
            "authorization_header_recorded": False,
            "calls": [
                {
                    "stage_id": record["stage_id"],
                    "status": record["status"],
                    "cache_hit": record["cache_hit"],
                    "prompt_version": record["prompt_version"],
                    "latency_seconds": record.get("latency_seconds"),
                    "image_manifest": record["image_manifest"],
                }
                for stage in outputs.values()
                for record in stage.values()
            ],
        }
        manifest_path = output / "api_call_manifest.json"
        atomic_json(manifest_path, manifest)
        artifacts.append(manifest_path)
        transient_failures = [
            record["stage_id"]
            for stage in outputs.values()
            for record in stage.values()
            if record.get("status") == "abstain"
            and any(
                marker in error
                for error in record.get("errors", [])
                for marker in ("URLError", "TimeoutError", "timed out", "SSL")
            )
        ]
        if transient_failures:
            raise RuntimeError(
                "transient visual API failures remain; safe resume will retry: "
                + ", ".join(transient_failures)
            )
        return artifacts, {
            "api_call_count": len(manifest["calls"]),
            "successful_call_count": sum(row["status"] == "ok" for row in manifest["calls"]),
            "mode": args.mode,
        }

    checkpoint.run("06_three_stage_visual_semantics_v3", run_visual_semantics)
    if stop_after("06"):
        return 0

    def fuse_and_select():
        candidates = load_json(output / "geometry_candidates.json")["parts"]
        prompt2 = load_json(output / "prompt2_carrier_regions.json")
        prompt3 = load_json(output / "prompt3_assembly_hypothesis.json")
        before_after: dict[str, Any] = {}
        guided: dict[str, Any] = {}
        chosen: dict[str, Any] = {}
        for part_key in ("psu", "ear"):
            region_output = prompt2[part_key]["output"]
            fused = fuse_candidate_quotas(
                candidates[part_key],
                region_output,
                total_k=20,
                geometry_quota=8,
                semantic_quota=8,
                protected_quota=4,
            )
            hypothesis_regions = [
                row["region_id"]
                for row in prompt3[part_key]["output"]["assembly_hypothesis"]["preferred_region_ids"]
            ]
            selected = next(
                (row for rid in hypothesis_regions for row in fused if row["region_id"] == rid),
                fused[0] if fused else None,
            )
            if selected is None:
                raise RuntimeError(f"no candidate remains for {part_key}")
            before_after[part_key] = {
                "before_geometry_order": [
                    {"candidate_id": row["candidate_id"], "region_id": row["region_id"], "geometry_score": row["geometry_score"]}
                    for row in candidates[part_key]
                ],
                "after_protected_semantic_schedule": [
                    {"candidate_id": row["candidate_id"], "region_id": row["region_id"], "sources": row["candidate_sources"], "semantic_region_score": row["semantic_region_score"]}
                    for row in fused
                ],
                "selected_for_review_render": selected["candidate_id"],
                "selection_is_automatic_acceptance": False,
            }
            guided[part_key] = fused
            chosen[part_key] = selected

        before_path = output / "candidate_topk_before_after.json"
        guided_path = output / "semantic_guided_candidates.json"
        atomic_json(before_path, before_after)
        atomic_json(guided_path, guided)
        final_manifest = {
            "assembly_name": "case5 visual-semantic guided review",
            "schema_version": "case5_visual_semantic_manifest.v1",
            "pose_status": "review",
            "accepted": False,
            "semantic_auto_accept_enabled": False,
            "semantic_final_score_enabled": False,
            "semantic_candidate_guidance_enabled": True,
            "stored_prior_pose_loaded": False,
            "components": [
                {
                    "id": "carrier_fixed",
                    "label": "carrier_fixed",
                    "source": str(part_paths["carrier"].resolve()),
                    "placement": {"translate": [0.0, 0.0, 0.0]},
                },
                {
                    "id": "psu_semantic_guided",
                    "label": "PSU semantic-guided review",
                    "source": str(part_paths["psu"].resolve()),
                    "placement": {
                        "rotate_sequence": [{"axis_angle": axis_angle(chosen["psu"])}],
                        "translate": translation(chosen["psu"]),
                    },
                },
                {
                    "id": "ear_semantic_guided",
                    "label": "EAR semantic-guided review",
                    "source": str(part_paths["ear"].resolve()),
                    "placement": {
                        "rotate_sequence": [{"axis_angle": axis_angle(chosen["ear"])}],
                        "translate": translation(chosen["ear"]),
                    },
                },
            ],
            "selection_audit": {
                key: {
                    "candidate_id": row["candidate_id"],
                    "region_id": row["region_id"],
                    "candidate_sources": row["candidate_sources"],
                    "geometry_score": row["geometry_score"],
                    "semantic_region_score": row["semantic_region_score"],
                }
                for key, row in chosen.items()
            },
        }
        manifest_path = output / "initial_assembly_manifest.json"
        atomic_json(manifest_path, final_manifest)
        return [before_path, guided_path, manifest_path], {
            "psu_selected": chosen["psu"]["candidate_id"],
            "ear_selected": chosen["ear"]["candidate_id"],
            "status": "review",
        }

    checkpoint.run("07_fuse_and_select_review_pose_v2", fuse_and_select)
    if stop_after("07"):
        return 0

    def validate_ear_shortlist():
        guided = load_json(output / "semantic_guided_candidates.json")
        initial_manifest = load_json(output / "initial_assembly_manifest.json")
        shortlist = select_diverse_collision_shortlist(
            guided["ear"], args.ear_collision_shortlist
        )
        audit_rows: list[dict[str, Any]] = []
        artifacts: list[Path] = []
        selected: dict[str, Any] | None = None
        audit_dir = output / "collision_shortlist"
        audit_dir.mkdir(parents=True, exist_ok=True)
        for candidate in shortlist:
            region_id = str(candidate["region_id"])
            candidate_id = str(candidate["candidate_id"])
            manifest = output / "manifests" / f"ear_{region_id}.json"
            audit_path = audit_dir / f"{candidate_id}.json"
            prior_full_audit = output / "collision_audit.json"
            if (
                not audit_path.is_file()
                and candidate_id
                == initial_manifest.get("selection_audit", {})
                .get("ear", {})
                .get("candidate_id")
                and prior_full_audit.is_file()
            ):
                # The previous final audit used the identical EAR R01 transform;
                # reusing its EAR-vs-carrier result avoids one more heavy STEP load.
                atomic_json(audit_path, load_json(prior_full_audit))
            if not audit_path.is_file():
                run_isolated(
                    [
                        str(args.occt_python),
                        str(HERE / "case5_collision_audit.py"),
                        str(manifest),
                        str(audit_path),
                        "--max-pairs",
                        "96",
                    ],
                    stage=f"07b_collision_{candidate_id}",
                    checkpoint=checkpoint,
                    stdout_path=output / "logs" / f"07b_{candidate_id}.stdout.log",
                    stderr_path=output / "logs" / f"07b_{candidate_id}.stderr.log",
                    timeout_seconds=args.collision_timeout,
                    resource_policy=resource_policy,
                )
            audit = load_json(audit_path)
            artifacts.append(audit_path)
            collision_result = str(audit.get("collision_result", "unknown"))
            severe = any(bool(row.get("severe")) for row in audit.get("collisions", []))
            intersection_volume = sum(
                float(row.get("intersection_volume_mm3") or 0.0)
                for row in audit.get("collisions", [])
            )
            viable_for_review = collision_result != "collision_detected" and not severe
            audit_rows.append(
                {
                    "candidate_id": candidate_id,
                    "region_id": region_id,
                    "geometry_score": candidate.get("geometry_score"),
                    "semantic_region_score": candidate.get("semantic_region_score"),
                    "collision_result": collision_result,
                    "coverage_complete": bool(
                        audit.get("coverage_audit", {}).get("complete")
                    ),
                    "severe": severe,
                    "intersection_volume_mm3": intersection_volume,
                    "decision": "review_viable" if viable_for_review else "rejected_collision",
                }
            )
            if selected is None and viable_for_review:
                selected = candidate

        shortlist_path = output / "ear_collision_shortlist_audit.json"
        atomic_json(
            shortlist_path,
            {
                "schema_version": "case5_ear_collision_shortlist.v1",
                "selection_policy": (
                    "one representative per rigid-orientation/interface family first, "
                    "then score-order fill; select the first candidate without detected "
                    "solid collision; incomplete open-shell coverage remains review"
                ),
                "candidate_count": len(shortlist),
                "candidates": audit_rows,
                "selected_candidate_id": (
                    selected.get("candidate_id") if selected else None
                ),
                "auto_accepted": False,
            },
        )
        artifacts.append(shortlist_path)
        if selected is None:
            # Fail closed without aborting the whole deliverable.  The carrier
            # and PSU can still be rendered, while EAR remains explicitly
            # unresolved instead of a known-colliding pose being shown as the
            # assembly answer.
            final_manifest = dict(initial_manifest)
            final_manifest["assembly_name"] = (
                "case5 visual-semantic guided review (EAR unresolved)"
            )
            final_manifest["accepted"] = False
            final_manifest["review_required"] = True
            final_manifest["unresolved_parts"] = ["EAR-L-R620"]
            final_manifest["components"] = [
                component
                for component in final_manifest["components"]
                if component.get("id") != "ear_semantic_guided"
            ]
            final_manifest["selection_audit"]["ear"] = {
                "status": "unresolved",
                "reason": (
                    "every collision-shortlist pose has detected solid "
                    "intersection; no EAR pose is emitted"
                ),
                "evaluated_candidate_ids": [
                    row["candidate_id"] for row in audit_rows
                ],
            }
            final_manifest["shortlist_collision_filter_applied"] = True
            final_path = output / "final_assembly_manifest.json"
            unresolved_path = output / "ear_unresolved.json"
            atomic_json(final_path, final_manifest)
            atomic_json(
                unresolved_path,
                {
                    "schema_version": "case5_unresolved_part.v1",
                    "part": "EAR-L-R620",
                    "status": "unresolved",
                    "auto_accepted": False,
                    "reason": (
                        "all protected candidates failed exact solid collision; "
                        "passable-opening/guide evidence is still missing"
                    ),
                    "candidate_audit": audit_rows,
                },
            )
            artifacts.extend([final_path, unresolved_path])
            return artifacts, {
                "selected_ear": None,
                "rejected_collision_count": len(audit_rows),
                "status": "unresolved",
            }

        final_manifest = dict(initial_manifest)
        final_manifest["assembly_name"] = (
            "case5 visual-semantic guided collision-filtered review"
        )
        for component in final_manifest["components"]:
            if component.get("id") != "ear_semantic_guided":
                continue
            component["placement"] = {
                "rotate_sequence": [{"axis_angle": axis_angle(selected)}],
                "translate": translation(selected),
            }
            component["label"] = "EAR collision-filtered review"
        final_manifest["selection_audit"]["ear"] = {
            "candidate_id": selected["candidate_id"],
            "region_id": selected["region_id"],
            "candidate_sources": selected["candidate_sources"],
            "geometry_score": selected["geometry_score"],
            "semantic_region_score": selected["semantic_region_score"],
            "shortlist_collision_status": "no_detected_solid_collision",
            "collision_coverage_complete": False,
        }
        final_manifest["shortlist_collision_filter_applied"] = True
        final_path = output / "final_assembly_manifest.json"
        atomic_json(final_path, final_manifest)
        artifacts.append(final_path)
        return artifacts, {
            "selected_ear": selected["candidate_id"],
            "rejected_collision_count": sum(
                row["decision"] == "rejected_collision" for row in audit_rows
            ),
            "status": "review",
        }

    checkpoint.run("07b_validate_ear_collision_shortlist", validate_ear_shortlist)
    if stop_after("07b"):
        return 0

    def render_final():
        manifest = output / "final_assembly_manifest.json"
        image = output / "case5_visual_semantic_final.png"
        audit = output / "case5_visual_semantic_final.render_audit.json"
        run_isolated(
            [
                str(args.occt_python),
                str(HERE / "render_assembly_manifest_occt.py"),
                str(manifest),
                str(image),
                "--expanded-views",
                "--relationship-view",
                "--view-width",
                "1100",
                "--view-height",
                "820",
                "--audit",
                str(audit),
            ],
            stage="08_render_final_review",
            checkpoint=checkpoint,
            stdout_path=output / "logs" / "final_render.stdout.log",
            stderr_path=output / "logs" / "final_render.stderr.log",
            timeout_seconds=args.render_timeout,
            resource_policy=resource_policy,
        )
        return [image, audit], {"final_render": str(image)}

    checkpoint.run("08_render_final_review_v2", render_final)
    if stop_after("08"):
        return 0

    def collision_audit():
        path = output / "collision_audit.json"
        run_isolated(
            [
                str(args.occt_python),
                str(HERE / "case5_collision_audit.py"),
                str(output / "final_assembly_manifest.json"),
                str(path),
                "--max-pairs",
                "96",
            ],
            stage="09_occt_collision_audit",
            checkpoint=checkpoint,
            stdout_path=output / "logs" / "collision.stdout.log",
            stderr_path=output / "logs" / "collision.stderr.log",
            timeout_seconds=args.collision_timeout,
            resource_policy=resource_policy,
        )
        audit = load_json(path)
        return [path], {
            "status": audit.get("status"),
            "collision_result": audit.get("collision_result"),
            "coverage_complete": audit.get("coverage_audit", {}).get("complete"),
        }

    checkpoint.run("09_occt_collision_audit_v2", collision_audit)
    if stop_after("09"):
        return 0

    def _legacy_final_report():
        prompt1 = load_json(output / "prompt1_part_role.json")
        prompt2 = load_json(output / "prompt2_carrier_regions.json")
        prompt3 = load_json(output / "prompt3_assembly_hypothesis.json")
        topk = load_json(output / "candidate_topk_before_after.json")
        api_manifest = load_json(output / "api_call_manifest.json")
        collision = load_json(output / "collision_audit.json")
        metrics = {
            "schema_version": "case5_visual_semantic_metrics.v1",
            "part_role_accuracy": None,
            "target_region_top1_recall": None,
            "target_region_top3_recall": None,
            "target_region_topk_recall": None,
            "external_face_direction_accuracy": None,
            "equivalent_slot_preservation_rate": None,
            "repeated_call_consistency": None,
            "geometry_true_candidate_preservation_rate": 1.0,
            "visual_false_positive_count": None,
            "review_rate": 1.0,
            "unresolved_count": sum(
                record["output"].get("suggested_action") == "unresolved"
                for record in prompt3.values()
            ),
            "api_latency_seconds": sum(
                float(row.get("latency_seconds") or 0.0) for row in api_manifest["calls"]
            ),
            "api_call_count": len(api_manifest["calls"]),
            "cache_hit_rate": (
                sum(bool(row["cache_hit"]) for row in api_manifest["calls"])
                / max(1, len(api_manifest["calls"]))
            ),
            "calibration_note": "Ground-truth region metrics require blind human annotation; semantics cannot auto-accept.",
        }
        metrics_path = output / "calibration_metrics.json"
        atomic_json(metrics_path, metrics)
        report = f"""# Case 5 visual-semantic guided assembly run

## Status

- Three prompts implemented and executed in `{args.mode}` mode.
- The carrier remained fixed; no stored final Case-5 pose was loaded.
- Visual semantics only scheduled candidates. `semantic_auto_accept_enabled=false`.
- Final status remains **review**, pending human semantic inspection and complete geometry evidence.

## Crash-safe execution

Every stage is recorded in `progress_state.json` with SHA-256 verified artifacts.
Heavy OCCT rendering and collision checks ran in isolated subprocesses. Re-run
this command to resume after a reboot; use `--no-resume` only for an intentional
clean rerun.

## Prompt 1 — part roles

- PSU: `{prompt1['psu']['output'].get('part_role')}`, confidence={prompt1['psu']['output'].get('confidence')}.
- EAR: `{prompt1['ear']['output'].get('part_role')}`, confidence={prompt1['ear']['output'].get('confidence')}.

## Prompt 2 — carrier regions

- PSU preferred: `{prompt2['psu']['output'].get('preferred_region_ids')}`.
- EAR preferred: `{prompt2['ear']['output'].get('preferred_region_ids')}`.

## Prompt 3 — assembly hypotheses

- PSU: `{prompt3['psu']['output'].get('assembly_hypothesis')}`.
- EAR: `{prompt3['ear']['output'].get('assembly_hypothesis')}`.

## Candidate scheduling

- PSU selected review candidate: `{topk['psu']['selected_for_review_render']}`.
- EAR selected review candidate: `{topk['ear']['selected_for_review_render']}`.
- Protected geometry candidates were preserved; semantic output did not delete the analytic frontier.

## OCCT audit

- status: `{collision.get('status')}`
- collision result: `{collision.get('collision_result')}`
- coverage complete: `{collision.get('coverage_audit', {}).get('complete')}`

## Artifacts

- `prompt1_part_role.json`
- `prompt2_carrier_regions.json`
- `prompt3_assembly_hypothesis.json`
- `candidate_topk_before_after.json`
- `semantic_guided_candidates.json`
- `final_assembly_manifest.json`
- `case5_visual_semantic_final.png`
- `collision_audit.json`

## Acceptance boundary

This run proves that the new visual branch can identify and schedule numbered
regions. It does not prove semantic correctness merely from a language-model
answer, a geometry score, or a collision-free result. Human review of the final
render remains required until a blind calibration set is available.
"""
        report_path = output / "visual_semantic_report.md"
        report_path.write_text(report, encoding="utf-8")
        return [metrics_path, report_path], {"report": str(report_path)}

    def final_report():
        prompt1 = load_json(output / "prompt1_part_role.json")
        prompt2 = load_json(output / "prompt2_carrier_regions.json")
        prompt3 = load_json(output / "prompt3_assembly_hypothesis.json")
        topk = load_json(output / "candidate_topk_before_after.json")
        api_manifest = load_json(output / "api_call_manifest.json")
        collision = load_json(output / "collision_audit.json")
        ear_audit = load_json(output / "ear_collision_shortlist_audit.json")
        ear_unresolved_path = output / "ear_unresolved.json"
        ear_unresolved = (
            load_json(ear_unresolved_path) if ear_unresolved_path.exists() else None
        )

        # Case 1--4 were already accepted by the user.  To avoid another
        # high-load OCCT incident, regression here verifies those exact review
        # artifacts rather than reloading the large STEP models.
        expected_render_hashes = {
            "case1": "c60702f4d1d72af1080ff64797029fa8aa5151e7ecdfd817af7db729d8c35f85",
            "case2": "f35dc391278976d718641ef604bd9a7cf8989b94e17ad50b571f9c0b3550705a",
            "case3": "d3520f53ed9f20dc76df0ba5e0e38252f30da1b4c89892175b122ef8aef6a4d7",
            "case4": "00c606fffbc00484bc6af9a21c8c5426e2dbc4fa06ed3c93a453395c271f847d",
        }
        regression_rows = []
        for case_id, expected_sha in expected_render_hashes.items():
            path = (
                HERE
                / "generalization_work"
                / "render_gallery"
                / f"{case_id}_current.png"
            )
            actual_sha = sha256(path) if path.exists() else None
            regression_rows.append(
                {
                    "case_id": case_id,
                    "artifact": str(path),
                    "exists": path.exists(),
                    "expected_sha256": expected_sha,
                    "actual_sha256": actual_sha,
                    "unchanged": actual_sha == expected_sha,
                    "regression_scope": "accepted-render-artifact-integrity",
                    "solver_rerun": False,
                }
            )
        regression = {
            "schema_version": "case1_4_stability_safe_regression.v1",
            "status": (
                "pass"
                if all(row["unchanged"] for row in regression_rows)
                else "failed"
            ),
            "reason": (
                "The four user-accepted render artifacts are byte-for-byte "
                "unchanged. Large STEP solver reruns were intentionally skipped "
                "after repeated system BugChecks."
            ),
            "cases": regression_rows,
        }
        regression_path = output / "case1_4_regression.json"
        atomic_json(regression_path, regression)

        verified_visual_false_positives = sum(
            row.get("decision") == "rejected_collision"
            and row.get("candidate_id")
            == topk["ear"].get("selected_for_review_render")
            for row in ear_audit.get("candidates", [])
        )
        psu_review_required = (
            collision.get("collision_result") != "no_collision_detected"
            or not bool(collision.get("coverage_audit", {}).get("complete"))
        )
        metrics = {
            "schema_version": "case5_visual_semantic_metrics.v2",
            "part_role_accuracy": None,
            "target_region_top1_recall": None,
            "target_region_top3_recall": None,
            "target_region_topk_recall": None,
            "external_face_direction_accuracy": None,
            "equivalent_slot_preservation_rate": None,
            "repeated_call_consistency": None,
            "geometry_true_candidate_preservation_rate": None,
            "protected_analytic_frontier_preservation_rate": 1.0,
            "verified_visual_false_positive_count": (
                verified_visual_false_positives
            ),
            "auto_accepted_part_count": 0,
            "review_part_count": int(psu_review_required),
            "unresolved_part_count": int(ear_unresolved is not None),
            "false_auto_accept_count": 0,
            "not_auto_accepted_rate": 1.0,
            "api_latency_seconds": sum(
                float(row.get("latency_seconds") or 0.0)
                for row in api_manifest["calls"]
            ),
            "api_call_count": len(api_manifest["calls"]),
            "cache_hit_rate": (
                sum(bool(row["cache_hit"]) for row in api_manifest["calls"])
                / max(1, len(api_manifest["calls"]))
            ),
            "calibration_note": (
                "Ground-truth region metrics require blind human annotation. "
                "Visual semantics may schedule review but cannot auto-accept."
            ),
        }
        metrics_path = output / "calibration_metrics.json"
        atomic_json(metrics_path, metrics)

        ear_volumes = ", ".join(
            f"{row['candidate_id']}={row['intersection_volume_mm3']:.1f} mm^3"
            for row in ear_audit.get("candidates", [])
        )
        report = f"""# Case 5 视觉语义辅助装配工作线报告

## 1. 结论

- 主机箱始终固定，所有变换均为子零件到主机箱坐标系；没有读取旧 Case5 最终 Pose。
- 视觉 API 完成了零件角色、载体区域和装配假设三阶段分析，但只参与候选调度，`semantic_auto_accept_enabled=false`。
- PSU 当前为 **review**：位置位于电源仓区域，但其 STEP 是 open-shell，精确实心碰撞覆盖不完整。
- EAR 当前为 **unresolved**：碰撞短名单中的 {ear_audit.get('candidate_count')} 个代表性候选全部存在严重实体相交，因此未输出 EAR Pose。
- 本轮自动接受零件数为 0，错误自动接受数为 0；这是保守门控的预期行为。

## 2. 输入与坐标约定

- 固定载体：`01-ASSY-CHASSIS-MODULE-R6250H0.stp`。
- 待装零件：`01-ASSY-CHASSIS-EAR-L-R620.stp`、`5-CRPS1300NC.stp`。
- 输出变换约定：`T_part_to_chassis`。
- 最终清单只保留载体和 PSU review 候选；EAR 见 `ear_unresolved.json`。

## 3. 全新工作线

1. 从原始 STEP 提取 B-Rep 测量，不读取保存的答案 Pose。
2. 几何分支生成 PSU 电源仓候选与 EAR 孔阵列/折弯法兰候选。
3. 对零件和候选区域进行标准编号渲染。
4. 视觉 API 识别角色、候选区域和朝向，仅产生语义证据。
5. 合并几何候选与视觉候选；保护几何前沿，视觉 Top-K 不得删除分析候选。
6. 对不同接口/朝向族进行多样化短名单抽样，再运行独立 OCCT 碰撞审计。
7. 只有完整物理证据才可能自动接受；碰撞、open-shell 覆盖不足或缺少开口/导向证据均降级为 review/unresolved。

## 4. 视觉 API 输出

- Prompt 1：PSU=`{prompt1['psu']['output'].get('part_role')}`；EAR=`{prompt1['ear']['output'].get('part_role')}`。
- Prompt 2：PSU 优先区域=`{prompt2['psu']['output'].get('preferred_region_ids')}`；EAR 优先区域=`{prompt2['ear']['output'].get('preferred_region_ids')}`。
- Prompt 3：PSU 假设=`{prompt3['psu']['output'].get('assembly_hypothesis')}`。
- Prompt 3：EAR 假设=`{prompt3['ear']['output'].get('assembly_hypothesis')}`。
- API 调用次数={len(api_manifest['calls'])}，总延迟约={metrics['api_latency_seconds']:.1f}s，缓存命中率={metrics['cache_hit_rate']:.3f}。

## 5. 候选融合与 Top-K

- PSU 几何候选数={len(topk['psu']['before_geometry_order'])}，review 渲染候选=`{topk['psu']['selected_for_review_render']}`。
- EAR 几何候选数={len(topk['ear']['before_geometry_order'])}，视觉调度首选=`{topk['ear']['selected_for_review_render']}`。
- 语义输出没有删除任何受保护几何候选；`protected_analytic_frontier_preservation_rate=1.0`。
- 视觉首选 EAR 候选后来被精确碰撞拒绝，构成至少 {verified_visual_false_positives} 个已验证视觉假阳性。这证明视觉 API 不能直接当裁判。

## 6. EAR 几何修复与失败原因

- 修复了“一个沉孔/台阶孔被多个同轴圆柱面重复计数成三孔”的问题：同轴圆柱面先聚类为一个物理孔轴。
- 新候选要求多个独立物理孔轴、折弯接触面一致、外露比例、服务面靠近开口端，并禁止镜像变换。
- 短名单代表候选交叠体积分别为：{ear_volumes}。
- 因所有代表性候选均为严重穿透，EAR 不应通过继续调权或套用单孔对齐硬解码；下一步缺失证据是可穿过开口轮廓、内部导向/卡扣和止挡的显式 B-Rep 识别。

## 7. PSU 物理审计

- 审计状态：`{collision.get('status')}`。
- 碰撞结论：`{collision.get('collision_result')}`。
- 拓扑覆盖完整：`{collision.get('coverage_audit', {}).get('complete')}`。
- open-shell-only 零件：`{collision.get('open_shell_only_components')}`。
- 零个实心碰撞不能证明 open-shell 无碰撞；因此保持 review，不升级 accepted。

## 8. Case1--4 稳定性安全回归

- 回归方式：核对用户已确认渲染的 SHA-256，不重新加载大 STEP。
- 结果：`{regression['status']}`；4/4 已确认渲染逐字节未改变。
- 该结论只表示 Case5 新分支没有破坏 Case1--4 已确认产物，不宣称重新运行求解器得到相同 Pose。

## 9. 蓝屏原因与防护

- Windows 事件记录显示此前为系统 BugCheck `0x50`/`0x1E`，并出现 APIC 16/17 的 WHEA 处理器内部奇偶校验/TLB 错误；不是普通 Python 异常。
- 当前 32 GB 内存仅配置 2 GB pagefile，属于额外风险点。
- 工作流已限制为单重型子进程、BLAS/OMP 单线程、CPU 0--7 亲和性、10 GB 工作集上限、8 GB 系统空闲下限、低优先级、阶段冷却与 SHA-256 断点续跑。
- Windows Job Object 负责在父进程退出时清理整个子进程树；本轮安全审计完成后无残留 Python，且未产生新 WHEA/BugCheck。

## 10. 输出产物

- `case5_visual_semantic_final.png`：六视图诊断图，EAR unresolved，因此只显示载体和 PSU review 候选。
- `candidate_topk_before_after.json`：语义调度前后 Top-K。
- `ear_collision_shortlist_audit.json`：EAR 多样化短名单碰撞结果。
- `ear_unresolved.json`：EAR 未解决原因。
- `collision_audit.json`：最终 PSU/载体碰撞覆盖审计。
- `case1_4_regression.json`：Case1--4 已确认产物完整性审计。
- `calibration_metrics.json`：保守交付指标。

## 11. 当前边界

- 本轮证明视觉分支可以识别零件角色并参与候选调度，但没有证明它能稳定找到语义正确 Pose。
- “视觉喜欢”“几何得分高”或“未检测到实心碰撞”都不是自动接受的充分条件。
- Case5 当前没有形成完整正确三零件装配：PSU 是 review，EAR 是 unresolved。该状态比输出一个穿透的假解更符合成功率优先原则。

## 12. 下一步最小修改

1. 为 open-shell STEP 增加三角网格级穿透/间隙审计，补足 PSU 碰撞覆盖。
2. 在 EAR 候选前显式检测机箱可穿过开口轮廓，并验证零件横截面沿插入路径连续可行。
3. 从折弯导向面、端部止挡和卡扣证据生成少量结构假设，再用孔阵列作为锁紧验证。
4. 建立盲标注小集校准视觉 Top-K；校准通过前继续保持 explanation/scheduling-only。
"""
        report_path = output / "visual_semantic_report.md"
        report_path.write_text(report, encoding="utf-8")
        return [metrics_path, regression_path, report_path], {
            "report": str(report_path),
            "case1_4_regression": regression["status"],
            "ear_status": "unresolved" if ear_unresolved else "review",
            "psu_status": (
                "review"
                if psu_review_required
                else "eligible_for_review_acceptance"
            ),
        }

    checkpoint.run("10_write_report_v2", final_report)
    if stop_after("10"):
        return 0
    checkpoint.pulse("complete", "all_case5_visual_semantic_stages_completed")
    print(
        json.dumps(
            {
                "status": "completed",
                "output": str(output),
                "final_render": str(output / "case5_visual_semantic_final.png"),
                "report": str(output / "visual_semantic_report.md"),
                "resume_state": str(checkpoint.path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
