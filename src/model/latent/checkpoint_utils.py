from __future__ import annotations


import hashlib
import json
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from src.data.basics.constants import nucleotide_constants
from src.utils.file_identity import file_identity
from src.utils.simple_config import namespace_to_dict


build_file_identity = file_identity


CHECKPOINT_METADATA_KEY = "checkpoint_metadata"
CHECKPOINT_SCHEMA_VERSION = 1
RNA_GEOMETRY_VERSION_CODE = 2
_FLOW_SOURCE_ARTIFACTS = (
    "autoencoder_checkpoint",
    "latent_stats",
    "coarse_stats",
)


def checkpoint_weight_provenance(
    checkpoint: Mapping[str, Any],
    *,
    use_inline_ema: bool,
) -> dict[str, Any]:
    if use_inline_ema:
        ema_state = checkpoint.get("ema")
        if not isinstance(ema_state, Mapping):
            raise RuntimeError(
                "Inline EMA was requested but the checkpoint has no EMA state"
            )
        return {
            "kind": "ema",
            "source": "inline_ema_shadow",
            "decay": ema_state.get("decay"),
            "num_updates": ema_state.get("num_updates"),
        }

    if checkpoint.get("averaging") is not None:
        raise RuntimeError("Only raw or inline-EMA Flow checkpoints are supported")
    return {
        "kind": "raw",
        "source": "checkpoint_model",
    }


def validate_file_identity(identity: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(identity, Mapping):
        raise RuntimeError(f"{context} must be a file-identity mapping")
    path = identity.get("path")
    size_bytes = identity.get("size_bytes")
    sha256 = identity.get("sha256")
    if not isinstance(path, str) or not path:
        raise RuntimeError(f"{context} has no path")
    if (
        isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
        or size_bytes < 0
    ):
        raise RuntimeError(f"{context} has invalid size_bytes")
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
    ):
        raise RuntimeError(f"{context} has invalid SHA-256")
    return {
        "path": path,
        "size_bytes": size_bytes,
        "sha256": sha256,
    }


def validate_finite_tensors(value: Any, *, context: str) -> None:
    def visit(item: Any, path: str) -> None:
        if isinstance(item, torch.Tensor):
            if (item.is_floating_point() or item.is_complex()) and not torch.isfinite(
                item
            ).all():
                raise RuntimeError(f"{context} contains a non-finite tensor at {path}")
            return
        if isinstance(item, Mapping):
            for key, child in item.items():
                visit(child, f"{path}.{key}")
            return
        if isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")

    visit(value, "root")


def validate_checkpoint_source_artifacts(
    checkpoint: Mapping[str, Any],
    *,
    context: str,
    current_artifacts: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:

    raw_artifacts = checkpoint.get("source_artifacts")
    if not isinstance(raw_artifacts, Mapping):
        raise RuntimeError(f"{context} has no Flow source_artifacts mapping")
    if set(raw_artifacts) != set(_FLOW_SOURCE_ARTIFACTS):
        raise RuntimeError(
            f"{context} Flow source artifact keys differ: "
            f"expected={list(_FLOW_SOURCE_ARTIFACTS)!r}, "
            f"actual={sorted(str(key) for key in raw_artifacts)!r}"
        )

    validated: dict[str, dict[str, Any]] = {}
    for name in _FLOW_SOURCE_ARTIFACTS:
        validated[name] = validate_file_identity(
            raw_artifacts[name],
            context=f"{context} source artifact {name!r}",
        )

    if current_artifacts is not None:
        if set(current_artifacts) != set(_FLOW_SOURCE_ARTIFACTS):
            raise RuntimeError(
                "Current Flow source artifact keys differ from the checkpoint contract"
            )
        mismatches = {}
        for name in _FLOW_SOURCE_ARTIFACTS:
            current = current_artifacts[name]
            expected_content = {
                "size_bytes": validated[name]["size_bytes"],
                "sha256": validated[name]["sha256"],
            }
            current_content = {
                "size_bytes": current.get("size_bytes"),
                "sha256": current.get("sha256"),
            }
            if current_content != expected_content:
                mismatches[name] = {
                    "checkpoint": expected_content,
                    "current": current_content,
                }
        if mismatches:
            raise RuntimeError(
                f"{context} Flow source artifacts differ from the supplied files: {mismatches}"
            )
    return validated


def _inference_model_config(stage: str, config: Mapping[str, Any]) -> dict[str, Any]:
    if stage == "scaffold_conditioned_autoencoder":
        return {
            key: config[key]
            for key in (
                "num_nt_types",
                "num_torsions",
                "latent_dim",
                "explicit_backbone",
                "decoder_conf",
            )
        }
    ae = _inference_model_config(
        "scaffold_conditioned_autoencoder", config["autoencoder_conf"]
    )
    matcher = config["product_flowmatcher"]
    return {
        "autoencoder_conf": ae,
        "flow_model_conf": config["flow_model_conf"],
        "product_flowmatcher": {
            "modalities": matcher["modalities"],
            "bb_coarse": {
                key: matcher["bb_coarse"][key] for key in ("coarse_mode", "dim")
            },
            "local_latents": {"dim": matcher["local_latents"]["dim"]},
        },
    }


def validate_checkpoint_metadata(
    checkpoint: Mapping[str, Any],
    *,
    expected_stage: str,
    context: str,
    expected_model_config: Any | None = None,
) -> Mapping[str, Any]:
    metadata = checkpoint.get(CHECKPOINT_METADATA_KEY)
    if not isinstance(metadata, Mapping):
        raise RuntimeError(
            f"{context} has no {CHECKPOINT_METADATA_KEY} for "
            f"{nucleotide_constants.RNA_GEOMETRY_VERSION}"
        )
    expected = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "rna_geometry_version": nucleotide_constants.RNA_GEOMETRY_VERSION,
    }
    mismatches = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"{context} metadata is incompatible: {mismatches}")
    actual_contract = metadata.get("model_contract")
    if not isinstance(actual_contract, Mapping):
        raise RuntimeError(f"{context} has no canonical semantic model contract")
    actual_config = actual_contract.get("config")
    if not isinstance(actual_config, Mapping):
        raise RuntimeError(f"{context} semantic model contract has no config mapping")
    required_keys = (
        {
            "num_nt_types",
            "num_torsions",
            "latent_dim",
            "explicit_backbone",
            "decoder_conf",
        }
        if expected_stage == "scaffold_conditioned_autoencoder"
        else {"autoencoder_conf", "flow_model_conf", "product_flowmatcher"}
    )
    if not required_keys.issubset(actual_config):
        raise RuntimeError(f"{context} has no compatible inference model contract")
    serialized = json.dumps(
        actual_config, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    if actual_contract.get("sha256") != hashlib.sha256(serialized).hexdigest():
        raise RuntimeError(f"{context} semantic model contract hash is invalid")
    if expected_model_config is not None:
        expected_config = namespace_to_dict(expected_model_config)
        actual_inference = _inference_model_config(expected_stage, actual_config)
        expected_inference = _inference_model_config(expected_stage, expected_config)
        if actual_inference != expected_inference:
            raise RuntimeError(
                f"{context} inference model contract differs from configured model: "
                f"checkpoint={actual_inference!r}, configured={expected_inference!r}"
            )
    return metadata


def validate_geometry_artifact(
    artifact: Any,
    *,
    context: str,
) -> None:
    if not isinstance(artifact, Mapping):
        raise RuntimeError(
            f"{context} must be a mapping, got {type(artifact).__name__}"
        )
    actual = artifact.get("geometry_version")
    expected = nucleotide_constants.RNA_GEOMETRY_VERSION
    if actual != expected:
        raise RuntimeError(
            f"{context} geometry_version={actual!r}, expected {expected!r}. "
            "Use statistics compatible with this checkpoint."
        )


def normalize_checkpoint_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    *,
    strip_prefixes: tuple[str, ...] = ("module.", "model."),
) -> dict[str, torch.Tensor]:
    normalized_state = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in strip_prefixes:
            new_key = new_key.removeprefix(prefix)
        if new_key in normalized_state:
            raise RuntimeError(
                "Checkpoint key normalization collision: "
                f"{key!r} and another key both normalize to {new_key!r}"
            )
        normalized_state[new_key] = value
    return normalized_state


def load_state_dict_with_policy(
    module: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    *,
    context: str = "checkpoint",
) -> None:
    validate_finite_tensors(state_dict, context=f"{context} state_dict")
    module.load_state_dict(state_dict, strict=True)
