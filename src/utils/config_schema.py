from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from src.data.basics.constants import nucleotide_constants
from src.utils.simple_config import namespace_to_dict, to_namespace


class SamplingSchedule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["linear", "power"]
    power: float | None = None

    @model_validator(mode="after")
    def validate_power(self) -> SamplingSchedule:
        if self.kind == "power" and (
            self.power is None or not math.isfinite(self.power) or self.power <= 0
        ):
            raise ValueError("Power sampling schedule requires a positive finite power")
        if self.kind == "linear" and self.power is not None:
            raise ValueError("Linear sampling schedule does not use power")
        return self


class SamplingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    self_condition: bool
    sampling_schedule: dict[Literal["bb_coarse", "local_latents"], SamplingSchedule]


class FlowModuleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    autoencoder_ckpt_path: str | None
    latent_preprocess: dict[str, Any]
    coarse_preprocess: dict[str, Any]
    autoencoder_conf: dict[str, Any]
    flow_model_conf: dict[str, Any]
    product_flowmatcher: dict[str, Any]
    sampling_conf: SamplingConfig

    @model_validator(mode="after")
    def validate_model_contract(self) -> FlowModuleConfig:
        ae = self.autoencoder_conf
        matcher = self.product_flowmatcher
        if ae.get("num_torsions") != nucleotide_constants.NUM_NA_TORSIONS:
            raise ValueError("Autoencoder torsion count differs from RNA geometry")
        if matcher.get("modalities") != ["bb_coarse", "local_latents"]:
            raise ValueError("Inference requires joint scaffold and latent modalities")
        mode = ae.get("explicit_backbone", {}).get("mode")
        if matcher.get("bb_coarse", {}).get("coarse_mode") != mode:
            raise ValueError("Flow and decoder scaffold modes differ")
        if self.flow_model_conf.get("explicit_backbone_mode") != mode:
            raise ValueError("Flow model and decoder scaffold modes differ")
        if set(self.sampling_conf.sampling_schedule) != set(matcher["modalities"]):
            raise ValueError("A sampling schedule is required for each modality")
        return self


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device: str


class FlowInferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    flow_module_conf: FlowModuleConfig
    runtime_conf: RuntimeConfig


def validate_flow_config(config: Any) -> FlowInferenceConfig:
    return FlowInferenceConfig.model_validate(namespace_to_dict(config))


def apply_paper_inference_protocol(config: Any) -> None:
    flow_conf = config.flow_module_conf
    mode = flow_conf.autoencoder_conf.explicit_backbone.mode
    flow_conf.product_flowmatcher = to_namespace(
        {
            "modalities": ["bb_coarse", "local_latents"],
            "num_timesteps": 200,
            "bb_coarse": {
                "coarse_mode": mode,
                "dim": flow_conf.flow_model_conf.coarse_dim,
                "noise_scale": 1.0,
                "zero_center_noise": True,
                "zero_center_mode": "c4_anchor",
            },
            "local_latents": {
                "dim": flow_conf.flow_model_conf.latent_dim,
                "noise_scale": 1.0,
                "zero_center_noise": False,
            },
        }
    )


def resolve_flow_artifact_paths(
    config: Any,
    *,
    autoencoder_checkpoint: str | None = None,
    latent_stats: str | None = None,
    coarse_stats: str | None = None,
) -> None:
    flow_conf = config.flow_module_conf
    if autoencoder_checkpoint is not None:
        flow_conf.autoencoder_ckpt_path = autoencoder_checkpoint
    if latent_stats is not None:
        flow_conf.latent_preprocess.stats_path = latent_stats
    if coarse_stats is not None:
        flow_conf.coarse_preprocess.stats_path = coarse_stats
    required = {
        "--autoencoder_checkpoint": getattr(flow_conf, "autoencoder_ckpt_path", None),
        "--latent_stats": getattr(flow_conf.latent_preprocess, "stats_path", None),
        "--coarse_stats": getattr(flow_conf.coarse_preprocess, "stats_path", None),
    }
    missing = [
        flag
        for flag, value in required.items()
        if not isinstance(value, str) or not value
    ]
    if missing:
        raise ValueError("Missing inference artifacts: " + ", ".join(missing))
