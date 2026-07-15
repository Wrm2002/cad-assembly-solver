"""Multimodal semantic reviewer using Qwen-VL (DashScope).

Sends rendered assembly preview images alongside structured semantic metadata
to a vision-language model.  This replaces the text-only anonymous-geometry
summaries that caused the DeepSeek calibration failure (AUC=0.50).
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Literal

from contracts import SemanticDecision


VISION_SYSTEM_PROMPT = """You are a conservative mechanical-assembly candidate reviewer.
You receive a rendered parts image and structured metadata.
Judge whether the parts plausibly form ONE functional engineering assembly.

Output ONLY a single JSON object.  No markdown, no preamble, no explanation outside the JSON.
The JSON must have exactly this shape:
{
  "schema_version": "1.0.0",
  "proposal_id": "same id from input",
  "verdict": "accept|reject|abstain",
  "plausibility_score": 0.0,
  "confidence": 0.0,
  "reason_codes": ["short_code"],
  "explanation": "brief explanation",
  "risk_flags": []
}

RULES:
- abstain if you cannot identify the mechanical function from the image.
- reject if the parts clearly do not form a meaningful assembly.
- accept ONLY if you can name the assembly type and the parts fit.
- confidence must reflect YOUR certainty, not the geometry score.
- plausibility_score: 0.85-1.0 for accept, 0.15-0.40 for reject, 0.45-0.55 for abstain.
"""


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
        explanation="Multimodal review abstained; geometry-only behavior is preserved.",
        risk_flags=[reason],
    ).model_dump(mode="json")


def encode_image_base64(image_path: str | Path) -> str:
    """Read a PNG image file and return a base64 data URI."""
    path = Path(image_path)
    data = path.read_bytes()
    encoded = base64.b64encode(data).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _first_environment_value(names: tuple[str, ...]) -> str:
    """Return the first non-empty environment value without logging it."""
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def _chat_completions_url(base_url: str) -> str:
    """Accept either an API root or a complete chat-completions endpoint."""
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return normalized + "/chat/completions"


def _extract_json(text: str) -> str:
    """Extract a JSON object from text that may contain preamble or markdown."""
    # Try to find JSON object boundaries
    # Remove markdown code fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    # Find first { and matching }
    start = cleaned.find("{")
    if start == -1:
        return cleaned.strip()
    depth = 0
    for i in range(start, len(cleaned)):
        if cleaned[i] == "{":
            depth += 1
        elif cleaned[i] == "}":
            depth -= 1
            if depth == 0:
                return cleaned[start : i + 1]
    return cleaned[start:].strip()


class QwenVLReviewer:
    """Multimodal reviewer using Qwen-VL via DashScope OpenAI-compatible API.

    Sends a parts-tray rendered image plus structured metadata to the vision
    model and returns a SemanticDecision.
    """

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
        self._model = (
            os.environ.get("QWEN_VL_MODEL", "").strip()
            or self.config.get("vision_model", "qwen-vl-plus")
        )
        self._base_url = (
            _first_environment_value(("QWEN_BASE_URL", "OPENAI_BASE_URL"))
            or self.config.get(
                "vision_base_url",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
        )
        self._max_tokens = int(self.config.get("vision_max_tokens", 1024))
        self._timeout = float(self.config.get("vision_timeout_seconds", 45))
        self._max_attempts = int(self.config.get("vision_max_attempts", 2))

    def _build_payload(
        self, proposal_id: str, image_paths: list[str | Path], text_context: str
    ) -> dict[str, Any]:
        """Build a multimodal chat payload with images and text."""
        user_content: list[dict[str, Any]] = []
        for img_path in image_paths:
            data_uri = encode_image_base64(img_path)
            user_content.append(
                {"type": "image_url", "image_url": {"url": data_uri}}
            )
        # Include the text context with explicit JSON-only instruction
        instruction = (
            f"{text_context}\n\n"
            f"proposal_id: {proposal_id}\n"
            "Output ONLY a JSON object. No markdown. No preamble."
        )
        user_content.append({"type": "text", "text": instruction})

        return {
            "model": self._model,
            "messages": [
                {"role": "system", "content": VISION_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": self._max_tokens,
            "stream": False,
        }

    def _cache_key(
        self, proposal_id: str, image_paths: list[str | Path], text_context: str
    ) -> str:
        image_hashes = []
        for p in image_paths:
            data = Path(p).read_bytes()
            image_hashes.append(hashlib.sha256(data).hexdigest()[:16])
        material = {
            "provider": "qwen-vl",
            "base_url": self._base_url,
            "model": self._model,
            "proposal_id": proposal_id,
            "image_hashes": sorted(image_hashes),
            "text_context": text_context,
            "system_prompt": VISION_SYSTEM_PROMPT,
            "input_policy_version": self.config.get(
                "input_policy_version", "production_fields_only_v1"
            ),
        }
        return hashlib.sha256(_canonical(material).encode("utf-8")).hexdigest()

    def _http_transport(
        self, payload: dict[str, Any], api_key: str, timeout: float
    ) -> dict[str, Any]:
        url = _chat_completions_url(self._base_url)
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
        proposal_id: str,
        image_paths: list[str | Path],
        text_context: str,
        *,
        mode: Literal["live", "cache_only", "off"] = "live",
    ) -> dict[str, Any]:
        """Review a candidate group using multimodal input.

        Args:
            proposal_id: Group proposal identifier.
            image_paths: List of PNG image paths to send (e.g., parts-tray view).
            text_context: Structured text describing parts, roles, and evidence.
            mode: "live" to call API, "cache_only" to use cache, "off" to skip.
        """
        cache_key = self._cache_key(proposal_id, image_paths, text_context)
        cache_path = self.cache_dir / f"{cache_key}.json"
        if mode == "off":
            return {
                "decision": _abstention(proposal_id, "semantic_disabled"),
                "cache_hit": False,
                "provider": "qwen-vl",
            }
        if cache_path.is_file():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            cached["cache_hit"] = True
            return cached
        if mode == "cache_only":
            return {
                "decision": _abstention(proposal_id, "cache_miss"),
                "cache_hit": False,
                "provider": "qwen-vl",
            }
        api_key = _first_environment_value(
            (
                "QWEN_API_KEY",
                "Qwen_API_KEY",
                "DASHSCOPE_API_KEY",
                "OPENAI_API_KEY",
            )
        )
        if not api_key:
            return {
                "decision": _abstention(proposal_id, "api_key_missing"),
                "cache_hit": False,
                "provider": "qwen-vl",
            }

        # Validate image paths
        valid_paths = [p for p in image_paths if Path(p).is_file()]
        if not valid_paths:
            return {
                "decision": _abstention(proposal_id, "no_valid_images"),
                "cache_hit": False,
                "provider": "qwen-vl",
            }

        payload = self._build_payload(proposal_id, valid_paths, text_context)
        errors = []
        for attempt in range(1, self._max_attempts + 1):
            started = time.perf_counter()
            try:
                response = self.transport(payload, api_key, self._timeout)
                content = response["choices"][0]["message"].get("content")
                if not content:
                    raise ValueError("empty response content")
                # Extract JSON from possibly noisy response
                json_str = _extract_json(str(content))
                decision = SemanticDecision.model_validate(json.loads(json_str))
                if decision.proposal_id != proposal_id:
                    raise ValueError("response proposal_id mismatch")
                record = {
                    "schema_version": "1.0.0",
                    "provider": "qwen-vl",
                    "model": response.get("model", self._model),
                    "cache_key": cache_key,
                    "cache_hit": False,
                    "attempt": attempt,
                    "latency_seconds": time.perf_counter() - started,
                    "usage": response.get("usage", {}),
                    "finish_reason": response["choices"][0].get("finish_reason"),
                    "decision": decision.model_dump(mode="json"),
                    "image_paths": [str(p) for p in valid_paths],
                    "text_context": text_context,
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
            "provider": "qwen-vl",
            "model": self._model,
            "cache_hit": False,
            "decision": _abstention(proposal_id, "all_attempts_failed"),
            "request_summary": {"error": "; ".join(errors)},
        }

    def structured_review(
        self,
        stage_id: str,
        image_paths: list[str | Path],
        text_context: str,
        *,
        system_prompt: str,
        prompt_version: str,
        validate_output: Callable[[Any], dict[str, Any]],
        fallback_output: dict[str, Any],
        mode: Literal["live", "cache_only", "off"] = "live",
    ) -> dict[str, Any]:
        """Run one strict-JSON multimodal stage with an isolated cache.

        The cache contains hashes and validated output only.  It deliberately
        excludes request payloads, data URIs, authorization headers and keys.
        """
        valid_paths = [Path(path).resolve() for path in image_paths if Path(path).is_file()]
        image_manifest = [
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in valid_paths
        ]
        material = {
            "stage_id": stage_id,
            "provider": "qwen-vl",
            "base_url": self._base_url,
            "model": self._model,
            "prompt_version": prompt_version,
            "system_prompt_sha256": hashlib.sha256(
                system_prompt.encode("utf-8")
            ).hexdigest(),
            "text_context_sha256": hashlib.sha256(
                text_context.encode("utf-8")
            ).hexdigest(),
            "images": image_manifest,
        }
        cache_key = hashlib.sha256(
            _canonical(material).encode("utf-8")
        ).hexdigest()
        cache_path = self.cache_dir / f"structured_{cache_key}.json"

        def abstain(reason: str, errors: list[str] | None = None) -> dict[str, Any]:
            return {
                "schema_version": "visual_semantic_stage.v1",
                "stage_id": stage_id,
                "provider": "qwen-vl",
                "model": self._model,
                "prompt_version": prompt_version,
                "cache_key": cache_key,
                "cache_hit": False,
                "status": "abstain",
                "abstention_reason": reason,
                "errors": list(errors or []),
                "image_manifest": image_manifest,
                "output": fallback_output,
            }

        if mode == "off":
            return abstain("semantic_disabled")
        if not valid_paths:
            return abstain("no_valid_images")
        if cache_path.is_file():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            cached["output"] = validate_output(cached["output"])
            cached["cache_hit"] = True
            return cached
        if mode == "cache_only":
            return abstain("cache_miss")

        api_key = _first_environment_value(
            (
                "QWEN_API_KEY",
                "Qwen_API_KEY",
                "DASHSCOPE_API_KEY",
                "OPENAI_API_KEY",
            )
        )
        if not api_key:
            return abstain("api_key_missing")

        user_content: list[dict[str, Any]] = []
        for path in valid_paths:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": encode_image_base64(path)},
                }
            )
        user_content.append(
            {
                "type": "text",
                "text": (
                    text_context
                    + "\n\nOutput ONLY one strict JSON object. "
                    "No markdown and no text outside JSON."
                ),
            }
        )
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": self._max_tokens,
            "stream": False,
        }
        errors: list[str] = []
        for attempt in range(1, self._max_attempts + 1):
            started = time.perf_counter()
            try:
                response = self.transport(payload, api_key, self._timeout)
                content = response["choices"][0]["message"].get("content")
                if isinstance(content, list):
                    content = "".join(
                        str(item.get("text", "")) if isinstance(item, dict) else str(item)
                        for item in content
                    )
                if not content:
                    raise ValueError("empty response content")
                output = validate_output(json.loads(_extract_json(str(content))))
                record = {
                    "schema_version": "visual_semantic_stage.v1",
                    "stage_id": stage_id,
                    "provider": "qwen-vl",
                    "model": response.get("model", self._model),
                    "prompt_version": prompt_version,
                    "cache_key": cache_key,
                    "cache_hit": False,
                    "status": "ok",
                    "attempt": attempt,
                    "latency_seconds": time.perf_counter() - started,
                    "usage": response.get("usage", {}),
                    "finish_reason": response["choices"][0].get("finish_reason"),
                    "image_manifest": image_manifest,
                    "input_hashes": {
                        "system_prompt_sha256": material["system_prompt_sha256"],
                        "text_context_sha256": material["text_context_sha256"],
                    },
                    "output": output,
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
                # Never retain payloads or headers in the error record.
                errors.append(f"attempt {attempt}: {type(exc).__name__}: {str(exc)[:240]}")
        return abstain("all_attempts_failed", errors)


def smoke_test() -> int:
    """Verify that the reviewer can be instantiated and handle missing API key."""
    reviewer = QwenVLReviewer(
        {"vision_model": "qwen-vl-plus"},
        Path("test_cache"),
    )
    result = reviewer.review(
        "smoke_001", [], "test smoke", mode="off"
    )
    decision = result["decision"]
    assert decision["verdict"] == "abstain", f"unexpected verdict: {decision['verdict']}"
    assert result["provider"] == "qwen-vl"
    # Cleanup test cache
    import shutil
    cache_dir = Path("test_cache")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    print("QwenVLReviewer smoke test passed.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(smoke_test())
