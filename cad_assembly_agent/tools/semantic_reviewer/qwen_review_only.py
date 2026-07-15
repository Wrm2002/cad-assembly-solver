"""Qwen multimodal review-only adapter.

The response is written as auxiliary evidence and is never allowed to alter
accepted/review/rejected geometry tiers.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path


def dump(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_json(text: str):
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    return json.loads(cleaned)


def call_qwen(api_key: str, endpoint: str, model: str, prompt: str, image: Path) -> dict:
    encoded = base64.b64encode(image.read_bytes()).decode("ascii")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a cautious CAD assembly review assistant. You do not "
                    "judge geometric feasibility and you never override OCCT. "
                    "When evidence is insufficient, abstain."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
        "temperature": 0,
        "max_completion_tokens": 800,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        body = json.loads(response.read().decode("utf-8"))
    content = body["choices"][0]["message"]["content"]
    return {
        "parsed_review": parse_json(content),
        "model": body.get("model", model),
        "usage": body.get("usage"),
        "raw_content": content,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True)
    parser.add_argument("--preview-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    graph = json.loads(Path(args.graph).read_text(encoding="utf-8"))
    preview_root = Path(args.preview_root)
    api_key = os.getenv("Qwen_API_KEY") or os.getenv("QWEN_API_KEY") or os.getenv(
        "DASHSCOPE_API_KEY"
    )
    endpoint = os.getenv(
        "QWEN_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    )
    model = os.getenv("QWEN_VL_MODEL", "qwen3-vl-plus")
    rows = []
    for case in graph["cases"]:
        case_id = str(case["case_id"])
        image = preview_root / f"第{case_id}组" / f"第{case_id}组_装配多视图.png"
        base = {
            "case_id": case_id,
            "image": str(image.resolve()),
            "review_only": True,
            "may_change_geometry_tier": False,
        }
        if not api_key:
            rows.append(
                {
                    **base,
                    "status": "unavailable",
                    "failure_reasons": ["qwen_api_key_not_configured"],
                    "unavailable_fields": ["semantic_review"],
                }
            )
            continue
        if not image.exists():
            rows.append(
                {
                    **base,
                    "status": "unavailable",
                    "failure_reasons": ["preview_image_missing"],
                    "unavailable_fields": ["semantic_review"],
                }
            )
            continue
        edge_summary = [
            {
                "parts": row["parts"],
                "pose_status": row["pose_status"],
                "decision_reason": row["decision_reason"],
            }
            for row in case["review_edges"] + case["rejected_edges"]
        ]
        prompt = (
            "Review only the visible engineering semantics. Return one JSON object "
            "with exactly these keys: semantic_validity (high|medium|low|unknown), "
            "possible_system, functional_reason, risk, review_required (boolean), "
            "suggested_action (review|abstain), visible_part_count_estimate, "
            "limitations (array). Do not output a pose and do not claim hidden "
            "interfaces are correct. File/edge evidence follows:\n"
            + json.dumps(
                {"parts": case["parts"], "candidate_edges": edge_summary},
                ensure_ascii=False,
            )
        )
        try:
            response = call_qwen(api_key, endpoint, model, prompt, image)
            rows.append(
                {
                    **base,
                    "status": "success",
                    "review": response["parsed_review"],
                    "model": response["model"],
                    "usage": response["usage"],
                    "failure_reasons": [],
                    "unavailable_fields": [],
                }
            )
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError) as exc:
            rows.append(
                {
                    **base,
                    "status": "failed",
                    "failure_reasons": [f"{type(exc).__name__}:{exc}"],
                    "unavailable_fields": ["semantic_review"],
                }
            )
    result = {
        "schema_version": "1.0.0",
        "semantic_reranking_enabled": False,
        "policy": "Qwen is explanation/review-only and cannot change graph tiers.",
        "case_count": len(rows),
        "success_count": sum(row["status"] == "success" for row in rows),
        "reviews": rows,
        "failure_reasons": [
            reason for row in rows for reason in row.get("failure_reasons", [])
        ],
        "unavailable_fields": sorted(
            {
                field
                for row in rows
                for field in row.get("unavailable_fields", [])
            }
        ),
    }
    dump(Path(args.output), result)
    print(f"Qwen review-only: {result['success_count']}/{result['case_count']}")
    return 0 if result["success_count"] == result["case_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
