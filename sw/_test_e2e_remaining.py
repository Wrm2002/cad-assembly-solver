"""Quick test of JoinABLe E2E pipeline on remaining cases."""
import sys
sys.path.insert(0, ".")
from sw.joinable_e2e import run_pipeline
from pathlib import Path

pairs = [
    ("sw/3/01_FAN-CAGE-MODULE-R620-NH.stp", "sw/3/01_FAN-MODULE-SUNON-6056(1).stp", "sw/3/joinable_e2e_output"),
    ("sw/4/01-62DC24-MLB-PCBA.stp", "sw/4/5-rd_rc_a_2rx4_1_ddrv.stp", "sw/4/joinable_e2e_output"),
    ("sw/5/01-ASSY-CHASSIS-MODULE-R6250H0.stp", "sw/5/01-ASSY-CHASSIS-EAR-L-R620.stp", "sw/5/joinable_e2e_output"),
]

for a, b, out in pairs:
    stem_a = Path(a).stem
    stem_b = Path(b).stem
    print(f"\n===== {stem_a} + {stem_b} =====")
    try:
        r = run_pipeline(Path(a), Path(b), output_dir=Path(out), top_k=5, search_budget=30)
        s = r["summary"]
        print(f"  Top-1: p={s['top1_probability']:.4f} entities={s['top1_entities']}")
        if s.get("best_pose_overlap") is not None:
            print(f"  Pose: overlap={s['best_pose_overlap']:.4f} contact={s['best_pose_contact']:.4f} offset={s['best_pose_offset_mm']:.1f}mm")
    except Exception as e:
        print(f"  FAILED: {e}")

print("\nDone.")
