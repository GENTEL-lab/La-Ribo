

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch

from src.model.latent.checkpoint_utils import build_file_identity


REPO_ROOT = Path(__file__).resolve().parents[2]


_TRACKED_CODE_PATHS = {
    "latent_flow": "src/model/latent/latent_flow.py",
    "checkpoint_utils": "src/model/latent/checkpoint_utils.py",
    "pdb_writer": "src/data/basics/pdb_writer.py",
}


def resolve_sampling_device(
    requested_device: str,
    *,
    allow_device_fallback: bool,
) -> tuple[torch.device, bool]:
    if requested_device == "auto":
        if torch.cuda.is_available():
            requested_device = "cuda"
        else:
            requested_device = "cpu"
    requested = torch.device(requested_device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        if not allow_device_fallback:
            raise RuntimeError(
                f"Sampling requested {requested_device!r}, but CUDA is unavailable. "
                "Refusing a silent CPU fallback; use --allow_device_fallback only "
                "for non-publication diagnostics."
            )
        return torch.device("cpu"), True
    return requested, False


def source_tree_identity() -> dict[str, object]:
    files = []
    for path in sorted((REPO_ROOT / "src").rglob("*.py")):
        identity = build_file_identity(path)
        files.append(
            {
                "path": path.relative_to(REPO_ROOT).as_posix(),
                "size_bytes": identity["size_bytes"],
                "sha256": identity["sha256"],
            }
        )
    payload = json.dumps(files, separators=(",", ":"), sort_keys=True).encode()
    return {
        "root": str(REPO_ROOT),
        "file_count": len(files),
        "manifest_sha256": hashlib.sha256(payload).hexdigest(),
        "files": files,
    }


def sampling_code_identities(entrypoint: str) -> dict[str, dict]:
    return {
        name: build_file_identity(REPO_ROOT / relative_path)
        for name, relative_path in {"entrypoint": entrypoint, **_TRACKED_CODE_PATHS}.items()
    }


def sampling_runtime_provenance(
    *,
    entrypoint: str,
    repository_identity: dict[str, object],
    requested_device: str,
    resolved_device: torch.device,
    device_fallback_used: bool,
) -> dict[str, object]:
    cuda_device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    runtime: dict[str, object] = {
        "requested_device": requested_device,
        "resolved_device": str(resolved_device),
        "device_type": resolved_device.type,
        "device_fallback_used": device_fallback_used,
        "cuda_available": torch.cuda.is_available(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_count": cuda_device_count,
        "torch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "repository": repository_identity,
        "source_tree": source_tree_identity(),
        "code": sampling_code_identities(entrypoint),
    }
    if resolved_device.type == "cuda":
        device_index = (
            resolved_device.index
            if resolved_device.index is not None
            else torch.cuda.current_device()
        )
        properties = torch.cuda.get_device_properties(device_index)
        runtime.update(
            {
                "cuda_device_index": device_index,
                "cuda_device_name": torch.cuda.get_device_name(device_index),
                "cuda_device_capability": list(torch.cuda.get_device_capability(device_index)),
                "cuda_device_uuid": str(getattr(properties, "uuid", "")),
                "cuda_pci_bus_id": str(getattr(properties, "pci_bus_id", "")),
                "cuda_total_memory": int(properties.total_memory),
            }
        )
    return runtime
