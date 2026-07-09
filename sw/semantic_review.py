"""Provider-neutral, cached semantic review with a DeepSeek implementation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Literal

from contracts import SemanticDecision


SYSTEM_PROMPT = """You are a conservative mechanical-assembly candidate reviewer.
You receive only structured geometric summaries. You do not solve placements
and you must not override collision, residual, or connectivity failures.
Judge whether all listed parts plausibly form ONE functional assembly group.
When evidence is insufficient or multiple interpretations are equally likely,
choose abstain. Output one JSON object only, with exactly this shape:
{
  "schema_version": "1.0.0",
  "proposal_id": "same id from input",
  "verdict": "accept|reject|abstain",
  "plausibility_score": 0.0,
  "confidence": 0.0,
  "reason_codes": ["short_machine_readable_code"],
  "explanation": "brief evidence-based explanation",
  "risk_flags": []
}
Never infer facts not present in the input JSON."""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _abstention(proposal_id: str, reason: str) -> dict[str, Any]:
    return SemanticDecision(
        schema_version="1.0.0",
        proposal_id=proposal_id,
        verdict="abstain",
        plausibility_score=0.5,
        confidence=0.0,
        reason_codes=[reason],
        explanation="Semantic review abstained; geometry-only behavior is preserved.",
        risk_flags=[reason],
    ).model_dump(mode="json")


class DeepSeekReviewer:
    def __init__(
        self,
        config: dict[str, Any],
        cache_dir: str | Path,
        *,
        transport: Callable[[dict[str, Any], str, float], dict[str, Any]] | None = None,
    ):
        self.config = dict(config)
        self.cache_dir = Path(cache_dir).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.transport = transport or self._http_transport

    def _request(self, summary: dict[str, Any]) -> dict[str, Any]:
        return {
            "model": self.config["model"],
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Review this candidate and return JSON:\n"
                    + json.dumps(summary, ensure_ascii=False),
                },
            ],
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
            "max_tokens": int(self.config["maximum_output_tokens"]),
            "stream": False,
        }

    def _cache_key(self, summary: dict[str, Any]) -> str:
        material = {
            "provider": "deepseek",
            "base_url": self.config["base_url"],
            "model": self.config["model"],
            "prompt_version": self.config["prompt_version"],
            "system_prompt": SYSTEM_PROMPT,
            "summary": summary,
        }
        return hashlib.sha256(_canonical(material).encode("utf-8")).hexdigest()

    def _http_transport(
        self,
        payload: dict[str, Any],
        api_key: str,
        timeout: float,
    ) -> dict[str, Any]:
        url = self.config["base_url"].rstrip("/") + "/chat/completions"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def review(
        self,
        summary: dict[str, Any],
        *,
        mode: Literal["live", "cache_only", "off"] = "live",
    ) -> dict[str, Any]:
        proposal_id = str(summary["proposal_id"])
        cache_key = self._cache_key(summary)
        cache_path = self.cache_dir / f"{cache_key}.json"
        if cache_path.is_file():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            cached["cache_hit"] = True
            return cached
        if mode == "off":
            return {
                "decision": _abstention(proposal_id, "semantic_disabled"),
                "cache_hit": False,
                "provider": "none",
            }
        if mode == "cache_only":
            return {
                "decision": _abstention(proposal_id, "cache_miss"),
                "cache_hit": False,
                "provider": "deepseek",
            }
        api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            return {
                "decision": _abstention(proposal_id, "api_key_missing"),
                "cache_hit": False,
                "provider": "deepseek",
            }

        payload = self._request(summary)
        errors = []
        for attempt in range(1, int(self.config["maximum_attempts"]) + 1):
            started = time.perf_counter()
            try:
                response = self.transport(
                    payload,
                    api_key,
                    float(self.config["timeout_seconds"]),
                )
                content = response["choices"][0]["message"].get("content")
                if not content:
                    raise ValueError("empty response content")
                decision = SemanticDecision.model_validate(
                    json.loads(content)
                )
                if decision.proposal_id != proposal_id:
                    raise ValueError("response proposal_id mismatch")
                record = {
                    "schema_version": "1.0.0",
                    "provider": "deepseek",
                    "model": response.get("model", self.config["model"]),
                    "prompt_version": self.config["prompt_version"],
                    "cache_key": cache_key,
                    "cache_hit": False,
                    "attempt": attempt,
                    "latency_seconds": time.perf_counter() - started,
                    "usage": response.get("usage", {}),
                    "finish_reason": response["choices"][0].get(
                        "finish_reason"
                    ),
                    "decision": decision.model_dump(mode="json"),
                    "request_summary": summary,
                }
                cache_path.write_text(
                    json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                return record
            except (
                KeyError,
                IndexError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
                urllib.error.URLError,
            ) as exc:
                errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
        return {
            "schema_version": "1.0.0",
            "provider": "deepseek",
            "model": self.config["model"],
            "prompt_version": self.config["prompt_version"],
            "cache_hit": False,
            "errors": errors,
            "decision": _abstention(proposal_id, "provider_failure"),
            "request_summary": summary,
        }


def connectivity_smoke(config_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    reviewer = DeepSeekReviewer(
        config["semantic_review"],
        Path(output_dir) / "semantic_cache",
    )
    summary = {
        "schema_version": "1.0.0",
        "proposal_id": "connectivity_smoke",
        "parts": [
            {
                "part_id": "anonymous_shaft_candidate",
                "bbox_size_mm": [20.0, 20.0, 80.0],
                "cylinder_radii_mm": [10.0],
            },
            {
                "part_id": "anonymous_hub_candidate",
                "bbox_size_mm": [50.0, 50.0, 30.0],
                "cylinder_radii_mm": [10.2, 25.0],
            },
        ],
        "candidate_edges": [
            {
                "type": "clearance",
                "geometry_score": 0.94,
                "radius_gap_mm": 0.2,
            }
        ],
        "hard_geometry_status": "not_yet_validated",
    }
    result = reviewer.review(summary, mode="live")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "deepseek_connectivity_smoke.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "configs" / "pool_pipeline.json"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent / "semantic_results"),
    )
    args = parser.parse_args()
    if not args.smoke:
        parser.error("--smoke is currently required")
    result = connectivity_smoke(args.config, args.output_dir)
    print(
        json.dumps(
            {
                "provider": result.get("provider"),
                "model": result.get("model"),
                "cache_hit": result.get("cache_hit"),
                "decision": result["decision"],
                "usage": result.get("usage", {}),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
