from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SW_ROOT = Path(__file__).resolve().parents[1]
if str(SW_ROOT) not in sys.path:
    sys.path.insert(0, str(SW_ROOT))

from run_joinable_pair_frontier import run_pair_frontier  # noqa: E402


def _valid_result(step_a: Path, step_b: Path) -> dict:
    return {
        "schema_version": "joinable_e2e.v2",
        "part_a_fixed": str(step_a.resolve()),
        "part_b_moving": str(step_b.resolve()),
        "gnn_inference": {},
        "joint_hypotheses": {},
        "pose_search": {},
        "acceptance_boundary": {"can_auto_accept": False},
    }


class FakeJoinableRunner:
    def __init__(self, *, write_result: bool = True, return_code: int = 0):
        self.calls: list[list[str]] = []
        self.write_result = write_result
        self.return_code = return_code

    def __call__(self, command, **kwargs):
        command = [str(value) for value in command]
        self.calls.append(command)
        if self.write_result:
            output_dir = Path(command[command.index("--output-dir") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "joinable_e2e_result.json").write_text(
                json.dumps(_valid_result(Path(command[2]), Path(command[3]))),
                encoding="utf-8",
            )
        return subprocess.CompletedProcess(
            command,
            self.return_code,
            stdout="fake stdout",
            stderr="fake stderr",
        )


class JoinablePairFrontierRunnerTests(unittest.TestCase):
    def _parts(self, root: Path, count: int):
        rows = []
        for index in range(count):
            path = root / f"part_{index}.step"
            path.write_text(f"STEP {index}", encoding="utf-8")
            rows.append((f"P{index}", path))
        return rows

    def test_enumerates_every_unordered_pair_for_four_parts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runner = FakeJoinableRunner()
            manifest = run_pair_frontier(
                self._parts(root, 4),
                root / "output",
                run_search=False,
                command_runner=runner,
            )

            self.assertEqual(manifest["expected_pair_count"], 6)
            self.assertEqual(manifest["pair_count"], 6)
            self.assertEqual(manifest["completed_count"], 6)
            self.assertTrue(manifest["pipeline_complete"])
            self.assertEqual(len(runner.calls), 6)
            self.assertEqual(
                {(row["source"], row["target"]) for row in manifest["records"]},
                {
                    ("P0", "P1"), ("P0", "P2"), ("P0", "P3"),
                    ("P1", "P2"), ("P1", "P3"), ("P2", "P3"),
                },
            )
            self.assertTrue(
                (root / "output" / "pair_frontier_manifest.json").is_file()
            )

    def test_valid_cache_is_reused_and_option_change_invalidates_it(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parts = self._parts(root, 3)
            first = FakeJoinableRunner()
            first_manifest = run_pair_frontier(
                parts,
                root / "output",
                run_search=False,
                command_runner=first,
            )
            self.assertTrue(first_manifest["pipeline_complete"])
            self.assertEqual(len(first.calls), 3)

            cached = FakeJoinableRunner(write_result=False, return_code=99)
            cached_manifest = run_pair_frontier(
                parts,
                root / "output",
                run_search=False,
                command_runner=cached,
            )
            self.assertEqual(len(cached.calls), 0)
            self.assertEqual(cached_manifest["cache_hit_count"], 3)
            self.assertTrue(cached_manifest["pipeline_complete"])
            self.assertTrue(all(
                row["status"] == "cached" for row in cached_manifest["records"]
            ))

            changed = FakeJoinableRunner()
            changed_manifest = run_pair_frontier(
                parts,
                root / "output",
                top_k=21,
                run_search=False,
                command_runner=changed,
            )
            self.assertEqual(len(changed.calls), 3)
            self.assertTrue(changed_manifest["pipeline_complete"])

    def test_missing_input_is_recorded_for_each_affected_pair(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parts = self._parts(root, 2)
            parts.append(("P2", root / "missing.step"))
            runner = FakeJoinableRunner()
            manifest = run_pair_frontier(
                parts,
                root / "output",
                run_search=False,
                command_runner=runner,
            )

            self.assertFalse(manifest["pipeline_complete"])
            self.assertEqual(manifest["completed_count"], 1)
            self.assertEqual(manifest["failed_count"], 2)
            self.assertEqual(len(runner.calls), 1)
            missing = [
                row for row in manifest["records"]
                if row["status"] == "missing_input"
            ]
            self.assertEqual(len(missing), 2)
            self.assertTrue(all(not row["pipeline_complete"] for row in missing))

    def test_zero_exit_without_result_is_an_explicit_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runner = FakeJoinableRunner(write_result=False, return_code=0)
            manifest = run_pair_frontier(
                self._parts(root, 2),
                root / "output",
                run_search=False,
                command_runner=runner,
            )

            self.assertFalse(manifest["pipeline_complete"])
            self.assertEqual(manifest["records"][0]["status"], "invalid_result")
            self.assertIn("not produced", manifest["records"][0]["error"])

    def test_nonzero_exit_cannot_reuse_a_stale_result_as_success(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parts = self._parts(root, 2)
            output = root / "output"
            good = FakeJoinableRunner()
            self.assertTrue(run_pair_frontier(
                parts, output, run_search=False, command_runner=good
            )["pipeline_complete"])

            failed = FakeJoinableRunner(write_result=False, return_code=7)
            manifest = run_pair_frontier(
                parts,
                output,
                top_k=99,
                run_search=False,
                command_runner=failed,
            )
            self.assertFalse(manifest["pipeline_complete"])
            self.assertEqual(manifest["records"][0]["status"], "failed")
            self.assertEqual(manifest["records"][0]["return_code"], 7)


if __name__ == "__main__":
    unittest.main()
