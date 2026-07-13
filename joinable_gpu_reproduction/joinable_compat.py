"""Compatibility helpers for running the official JoinABLe release on modern PyTorch.

The official checkpoint and preprocessed pickles are left unchanged.  Old
PyTorch-Geometric ``Data`` objects are converted in memory to the current
storage layout, retaining only the features used by the released checkpoint.
"""

from __future__ import annotations

import gc
import json
import pickle
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch_geometric.data import Batch, Data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = PROJECT_ROOT / "joinable_migration_audit" / "vendor" / "JoinABLe"
DEFAULT_CHECKPOINT = VENDOR_ROOT / "pretrained" / "paper" / "last_run_0.ckpt"
DEFAULT_DATA_ROOT = Path(
    r"D:\Model_match_public_data\fusion360_joint\j1.0.0_preprocessed\joint"
)


@dataclass
class ModernSample:
    name: str
    graph1: Data
    graph2: Data
    joint_graph: Data


def add_vendor_to_path() -> None:
    vendor = str(VENDOR_ROOT)
    if vendor not in sys.path:
        sys.path.insert(0, vendor)


def load_checkpoint(path: Path = DEFAULT_CHECKPOINT) -> tuple[dict, object]:
    """Load the released Lightning checkpoint on CPU and return its args."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    args = checkpoint["hyper_parameters"]["args"]
    return checkpoint, args


def build_model(checkpoint: dict, args: object) -> torch.nn.Module:
    """Instantiate the released architecture and strictly load all weights."""

    add_vendor_to_path()
    from models.joinable import JoinABLe

    model = JoinABLe(
        args.hidden,
        args.input_features,
        dropout=args.dropout,
        mpn=args.mpn,
        batch_norm=args.batch_norm,
        reduction=args.reduction,
        post_net=args.post_net,
        pre_net=args.pre_net,
    )
    state_dict = {
        key.removeprefix("model."): value
        for key, value in checkpoint["state_dict"].items()
    }
    model.load_state_dict(state_dict, strict=True)
    return model


def _legacy_num_nodes(graph: object) -> int:
    attrs = graph.__dict__
    if "__num_nodes__" in attrs:
        return int(attrs["__num_nodes__"])
    if attrs.get("x") is not None:
        return int(attrs["x"].shape[0])
    raise ValueError("Legacy graph has neither __num_nodes__ nor x")


def _modernize_body_graph(graph: object) -> Data:
    attrs = graph.__dict__
    required = (
        "edge_index",
        "entity_types",
        "is_face",
        "length",
        "face_reversed",
        "edge_reversed",
    )
    missing = [key for key in required if attrs.get(key) is None]
    if missing:
        raise ValueError(f"Body graph is missing required fields: {missing}")
    return Data(
        num_nodes=_legacy_num_nodes(graph),
        edge_index=attrs["edge_index"],
        entity_types=attrs["entity_types"],
        is_face=attrs["is_face"],
        length=attrs["length"],
        face_reversed=attrs["face_reversed"],
        edge_reversed=attrs["edge_reversed"],
    )


def _remap_labels(labels: torch.Tensor, positive_codes: Iterable[int]) -> torch.Tensor:
    positive = torch.zeros_like(labels, dtype=torch.bool)
    for code in positive_codes:
        positive |= labels == code
    return positive.to(dtype=torch.long)


def _modernize_joint_graph(
    graph: object, positive_codes: Iterable[int]
) -> Data:
    attrs = graph.__dict__
    required = ("edge_index", "edge_attr", "num_nodes_graph1", "num_nodes_graph2")
    missing = [key for key in required if attrs.get(key) is None]
    if missing:
        raise ValueError(f"Joint graph is missing required fields: {missing}")
    n1 = int(attrs["num_nodes_graph1"])
    n2 = int(attrs["num_nodes_graph2"])
    return Data(
        num_nodes=n1 + n2,
        edge_index=attrs["edge_index"],
        edge_attr=_remap_labels(attrs["edge_attr"], positive_codes),
        num_nodes_graph1=n1,
        num_nodes_graph2=n2,
    )


def load_legacy_samples(
    pickle_path: Path,
    limit: int = 0,
    max_node_count: int = 950,
    label_scheme: str = "test",
    selection: str = "first",
) -> tuple[list[ModernSample], dict]:
    """Load and migrate a bounded subset of an official PyG 1.7 pickle.

    ``label_scheme='test'`` maps Joint and JointEquivalent to positives, matching
    the official test command.  ``label_scheme='train'`` keeps only Joint.
    """

    if label_scheme == "test":
        positive_codes = (1, 3)
    elif label_scheme == "train":
        positive_codes = (1,)
    else:
        raise ValueError("label_scheme must be 'test' or 'train'")
    if selection not in {"first", "uniform"}:
        raise ValueError("selection must be 'first' or 'uniform'")

    with pickle_path.open("rb") as handle:
        raw = pickle.load(handle)

    total_cached = len(raw["graphs"])
    samples: list[ModernSample] = []
    conversion_failures: list[dict] = []
    eligible_indices = []
    eligible_node_counts = []
    eligible_pair_counts = []
    for index, legacy_triplet in enumerate(raw["graphs"]):
        n1 = _legacy_num_nodes(legacy_triplet[0])
        n2 = _legacy_num_nodes(legacy_triplet[1])
        if max_node_count > 0 and n1 + n2 > max_node_count:
            continue
        eligible_indices.append(index)
        eligible_node_counts.append(n1 + n2)
        eligible_pair_counts.append(n1 * n2)

    if limit > 0 and len(eligible_indices) > limit:
        if selection == "uniform":
            positions = np.linspace(
                0, len(eligible_indices) - 1, num=limit, dtype=int
            )
            selected_indices = [eligible_indices[int(pos)] for pos in positions]
        else:
            selected_indices = eligible_indices[:limit]
    else:
        selected_indices = eligible_indices

    for index in selected_indices:
        legacy_triplet = raw["graphs"][index]
        try:
            samples.append(
                ModernSample(
                    name=str(raw["files"][index]),
                    graph1=_modernize_body_graph(legacy_triplet[0]),
                    graph2=_modernize_body_graph(legacy_triplet[1]),
                    joint_graph=_modernize_joint_graph(
                        legacy_triplet[2], positive_codes
                    ),
                )
            )
        except Exception as exc:
            conversion_failures.append(
                {
                    "index": index,
                    "file": str(raw["files"][index]),
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )

    del raw
    gc.collect()

    selected_node_counts = [
        sample.graph1.num_nodes + sample.graph2.num_nodes for sample in samples
    ]
    selected_pair_counts = [
        sample.graph1.num_nodes * sample.graph2.num_nodes for sample in samples
    ]

    def distribution(values: list[int]) -> dict:
        if not values:
            return {}
        array = np.asarray(values)
        return {
            "min": int(array.min()),
            "median": float(np.median(array)),
            "p95": float(np.percentile(array, 95)),
            "max": int(array.max()),
        }

    audit = {
        "pickle_path": str(pickle_path),
        "total_cached_graphs": total_cached,
        "eligible_graphs": len(eligible_indices),
        "converted_graphs": len(samples),
        "skipped_over_max_node_count": total_cached - len(eligible_indices),
        "max_node_count": max_node_count,
        "label_scheme": label_scheme,
        "selection": selection,
        "eligible_combined_node_distribution": distribution(eligible_node_counts),
        "eligible_candidate_pair_distribution": distribution(eligible_pair_counts),
        "selected_combined_node_distribution": distribution(selected_node_counts),
        "selected_candidate_pair_distribution": distribution(selected_pair_counts),
        "conversion_failure_count": len(conversion_failures),
        "conversion_failures": conversion_failures,
        "legacy_pickle_modified": False,
    }
    return samples, audit


def collate_samples(samples: list[ModernSample]) -> tuple[Batch, Batch, Batch]:
    return (
        Batch.from_data_list([sample.graph1 for sample in samples]),
        Batch.from_data_list([sample.graph2 for sample in samples]),
        Batch.from_data_list([sample.joint_graph for sample in samples]),
    )


def batch_to_device(
    batch: tuple[Batch, Batch, Batch], device: torch.device
) -> tuple[Batch, Batch, Batch]:
    return tuple(graph.to(device) for graph in batch)


def top_k_hits(
    logits: torch.Tensor, labels: torch.Tensor, ks: Iterable[int] = (1, 5, 10)
) -> dict[str, bool]:
    labels = labels.flatten()
    logits = logits.flatten()
    max_k = min(max(ks), logits.numel())
    indices = torch.topk(logits, k=max_k).indices
    return {
        f"top_{k}": bool(labels[indices[: min(k, max_k)]].max().item() == 1)
        for k in ks
    }


def environment_summary() -> dict:
    cuda_available = torch.cuda.is_available()
    summary = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_geometric": __import__("torch_geometric").__version__,
        "cuda_available": cuda_available,
        "torch_cuda_runtime": torch.version.cuda,
    }
    if cuda_available:
        props = torch.cuda.get_device_properties(0)
        summary.update(
            {
                "gpu": props.name,
                "gpu_compute_capability": list(torch.cuda.get_device_capability(0)),
                "gpu_vram_gib": round(props.total_memory / 1024**3, 3),
            }
        )
    return summary


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
