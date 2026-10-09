from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.model.latent.checkpoint_utils import (
    RNA_GEOMETRY_VERSION_CODE,
    build_file_identity,
    load_state_dict_with_policy,
    normalize_checkpoint_state_dict,
    validate_geometry_artifact,
)
from src.model.latent.common import (
    build_coarse_distance_features,
    build_relpos_features,
    ensure_float_mask,
    extract_c4_positions,
    get_coarse_backbone_dim,
    get_coarse_pair_feature_dim,
    get_explicit_backbone_mode,
    get_timestep_features,
)
from src.model.latent.rna_autoencoder import RNAPartialAutoencoder
from src.model.latent.transformer import PairBiasedTransformer


EPS = 1e-8


FLOW_PREPROCESS_STATE_KEYS = (
    "autoencoder.rna_geometry_version_code",
    "latent_preprocess.latent_mean",
    "latent_preprocess.latent_std",
    "latent_preprocess.latent_mode_code",
    "coarse_preprocess.coarse_scale",
    "coarse_preprocess.coarse_mode_code",
)


def validate_flow_preprocess_checkpoint_state(
    state_dict: Mapping[str, torch.Tensor],
) -> None:
    normalized = normalize_checkpoint_state_dict(state_dict)
    missing = sorted(key for key in FLOW_PREPROCESS_STATE_KEYS if key not in normalized)
    duplicate_owner_keys = sorted(
        key for key in normalized if key.startswith("flow_model.coarse_preprocess.")
    )
    if missing or duplicate_owner_keys:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if duplicate_owner_keys:
            details.append(f"duplicate_owner_keys={duplicate_owner_keys}")
        raise RuntimeError(
            "Latent-flow preprocessing state is not canonical: " + "; ".join(details)
        )
    geometry_code = int(
        torch.as_tensor(normalized["autoencoder.rna_geometry_version_code"]).reshape(())
    )
    if geometry_code != RNA_GEOMETRY_VERSION_CODE:
        raise RuntimeError(
            "Latent-flow autoencoder geometry code mismatch: "
            f"checkpoint={geometry_code}, expected={RNA_GEOMETRY_VERSION_CODE}"
        )


class LatentPreprocessor(nn.Module):
    def __init__(
        self,
        conf: Any,
        latent_dim: int,
    ):
        super().__init__()
        self.mode = getattr(conf, "mode", "none") if conf is not None else "none"
        self.eps = float(getattr(conf, "eps", 1.0e-6) if conf is not None else 1.0e-6)
        if not math.isfinite(self.eps) or self.eps <= 0.0:
            raise ValueError(
                f"latent_preprocess eps must be finite and positive, got {self.eps}"
            )
        mean = torch.zeros(1, 1, latent_dim)
        std = torch.ones(1, 1, latent_dim)
        if self.mode == "whiten":
            stats_path = getattr(conf, "stats_path", None)
            if stats_path is None:
                raise ValueError("latent_preprocess.mode=whiten requires stats_path")
            stats_file = Path(stats_path)
            if not stats_file.exists():
                raise FileNotFoundError(
                    f"Latent whitening stats not found at {stats_file}"
                )
            stats = torch.load(stats_file, map_location="cpu")
            validate_geometry_artifact(stats, context=f"latent stats {stats_file}")
            mean = torch.as_tensor(stats["mean"]).float().reshape(1, 1, -1)
            std = torch.as_tensor(stats["std"]).float().reshape(1, 1, -1)
            if mean.shape[-1] != latent_dim or std.shape[-1] != latent_dim:
                raise ValueError(
                    f"Latent stats dim mismatch: expected {latent_dim}, got mean={mean.shape[-1]} std={std.shape[-1]}"
                )
            if not torch.isfinite(mean).all():
                raise ValueError("Latent stats mean must contain only finite values")
            if not torch.isfinite(std).all() or torch.any(std <= 0):
                raise ValueError(
                    "Latent stats std must contain only finite positive values"
                )
            std = std.clamp_min(self.eps)
        elif self.mode != "none":
            raise ValueError(f"Unsupported latent preprocess mode: {self.mode}")
        self.register_buffer("latent_mean", mean)
        self.register_buffer("latent_std", std)
        self.register_buffer(
            "latent_mode_code",
            torch.tensor(0 if self.mode == "none" else 1, dtype=torch.int8),
        )

    def _load_from_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        prefix: str,
        local_metadata: Any,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        for name in ("latent_mean", "latent_std"):
            key = f"{prefix}{name}"
            if key not in state_dict:
                continue
            checkpoint_value = torch.as_tensor(state_dict[key]).detach().cpu()
            configured_value = getattr(self, name).detach().cpu()
            if checkpoint_value.shape != configured_value.shape:
                error_msgs.append(
                    f"{key} shape mismatch: checkpoint={tuple(checkpoint_value.shape)}, "
                    f"configured={tuple(configured_value.shape)}"
                )
            elif not torch.equal(
                checkpoint_value.to(configured_value.dtype),
                configured_value,
            ):
                error_msgs.append(
                    f"{key} mismatch: checkpoint values differ from configured latent stats"
                )
        mode_key = f"{prefix}latent_mode_code"
        if mode_key in state_dict:
            checkpoint_mode = torch.as_tensor(state_dict[mode_key])
            configured_mode = int(self.latent_mode_code)
            if checkpoint_mode.numel() != 1:
                error_msgs.append(
                    f"{mode_key} must be scalar, got shape {tuple(checkpoint_mode.shape)}"
                )
            elif int(checkpoint_mode.reshape(())) != configured_mode:
                error_msgs.append(
                    f"{mode_key} mismatch: checkpoint={int(checkpoint_mode.reshape(()))}, "
                    f"configured mode={self.mode!r} (code={configured_mode})"
                )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def unwhiten(self, latents: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return latents
        return latents * self.latent_std.to(
            latents.device, latents.dtype
        ) + self.latent_mean.to(
            latents.device,
            latents.dtype,
        )


class CoarsePreprocessor(nn.Module):
    def __init__(
        self,
        conf: Any,
        *,
        expected_backbone_mode: str | None = None,
        expected_zero_center: bool | None = None,
    ):
        super().__init__()
        self.mode = getattr(conf, "mode", "none") if conf is not None else "none"
        self.eps = float(getattr(conf, "eps", 1.0e-6) if conf is not None else 1.0e-6)
        if not math.isfinite(self.eps) or self.eps <= 0.0:
            raise ValueError(
                f"coarse_preprocess eps must be finite and positive, got {self.eps}"
            )
        scale = torch.ones(1, 1, 1)
        if self.mode == "global_scale":
            explicit_scale = getattr(conf, "scale", None)
            stats_path = getattr(conf, "stats_path", None)
            has_explicit_scale = explicit_scale is not None
            has_stats_path = stats_path not in {None, ""}
            if has_explicit_scale == has_stats_path:
                raise ValueError(
                    "coarse_preprocess.mode=global_scale requires exactly one of "
                    "`scale` or `stats_path`"
                )
            if has_explicit_scale:
                scale_value = float(explicit_scale)
            else:
                stats_file = Path(stats_path)
                if not stats_file.exists():
                    raise FileNotFoundError(f"Coarse stats not found at {stats_file}")
                stats = torch.load(stats_file, map_location="cpu")
                validate_geometry_artifact(stats, context=f"coarse stats {stats_file}")
                if stats.get("mode") != "global_scale":
                    raise RuntimeError(
                        f"coarse stats {stats_file} mode must be 'global_scale'"
                    )
                if expected_backbone_mode is not None:
                    stats_backbone_mode = stats.get("explicit_backbone_mode")
                    if stats_backbone_mode != expected_backbone_mode:
                        raise RuntimeError(
                            f"coarse stats {stats_file} explicit_backbone_mode differs: "
                            f"stats={stats_backbone_mode!r}, "
                            f"configured={expected_backbone_mode!r}"
                        )
                if expected_zero_center is not None:
                    stats_zero_center = stats.get("zero_center")
                    if not isinstance(stats_zero_center, bool):
                        raise RuntimeError(
                            f"coarse stats {stats_file} has no boolean zero_center"
                        )
                    if stats_zero_center != expected_zero_center:
                        raise RuntimeError(
                            f"coarse stats {stats_file} zero_center differs: "
                            f"stats={stats_zero_center!r}, "
                            f"configured={expected_zero_center!r}"
                        )
                if "scale" not in stats:
                    raise KeyError(
                        "Coarse stats file must contain a scalar `scale` tensor"
                    )
                loaded_scale = torch.as_tensor(stats["scale"]).float()
                if loaded_scale.numel() != 1:
                    raise ValueError(
                        f"Coarse scale must be scalar, got shape {tuple(loaded_scale.shape)}"
                    )
                scale_value = float(loaded_scale.reshape(()).item())
            if not math.isfinite(scale_value) or scale_value <= 0.0:
                raise ValueError("coarse_preprocess scale must be finite and positive")
            scale = torch.tensor(
                max(scale_value, self.eps), dtype=torch.float32
            ).reshape(1, 1, 1)
        elif self.mode != "none":
            raise ValueError(f"Unsupported coarse preprocess mode: {self.mode}")

        self.register_buffer("coarse_scale", scale)
        self.register_buffer(
            "coarse_mode_code",
            torch.tensor(0 if self.mode == "none" else 1, dtype=torch.int8),
        )

    def _load_from_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        prefix: str,
        local_metadata: Any,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        key = f"{prefix}coarse_scale"
        mode_key = f"{prefix}coarse_mode_code"
        if mode_key in state_dict:
            checkpoint_mode = torch.as_tensor(state_dict[mode_key])
            configured_mode = int(self.coarse_mode_code)
            if checkpoint_mode.numel() != 1:
                error_msgs.append(
                    f"{mode_key} must be scalar, got shape {tuple(checkpoint_mode.shape)}"
                )
            elif int(checkpoint_mode.reshape(())) != configured_mode:
                error_msgs.append(
                    f"{mode_key} mismatch: checkpoint={int(checkpoint_mode.reshape(()))}, "
                    f"configured mode={self.mode!r} (code={configured_mode})"
                )
        if key in state_dict:
            checkpoint_scale = torch.as_tensor(state_dict[key]).detach().cpu()
            configured_scale = self.coarse_scale.detach().cpu()
            if checkpoint_scale.shape != configured_scale.shape:
                error_msgs.append(
                    f"{key} shape mismatch: checkpoint={tuple(checkpoint_scale.shape)}, "
                    f"configured={tuple(configured_scale.shape)}"
                )
            elif not torch.equal(
                checkpoint_scale.to(configured_scale.dtype),
                configured_scale,
            ):
                error_msgs.append(
                    f"{key} mismatch: checkpoint values differ from configured coarse stats"
                )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def to_flow(self, bb_physical: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return bb_physical
        return bb_physical / self.coarse_scale.to(bb_physical.device, bb_physical.dtype)

    def to_physical(self, bb_flow: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return bb_flow
        return bb_flow * self.coarse_scale.to(bb_flow.device, bb_flow.dtype)


def build_sampling_time_grids(
    modalities: list[str],
    num_timesteps: int,
    device: torch.device,
    schedule_conf: Any | None,
) -> dict[str, torch.Tensor]:
    base = torch.linspace(0.0, 1.0, num_timesteps + 1, device=device)
    schedules = _namespace_items(schedule_conf)
    grids: dict[str, torch.Tensor] = {}
    for name in modalities:
        spec = schedules.get(name)
        if spec is None:
            grids[name] = base
            continue
        kind = getattr(spec, "kind", "linear")
        if kind == "linear":
            grid = base
        elif kind == "power":
            power = float(getattr(spec, "power", 1.0))
            if not math.isfinite(power) or power <= 0.0:
                raise ValueError(
                    f"Power sampling schedule for {name} must be finite and positive"
                )
            grid = base.pow(power)
        else:
            raise ValueError(f"Unsupported sampling schedule kind for {name}: {kind}")
        grid = grid.clone()
        grid[0] = 0.0
        grid[-1] = 1.0
        grids[name] = grid
    return grids


def _namespace_items(namespace: Any) -> dict[str, Any]:
    if namespace is None:
        return {}
    if isinstance(namespace, Mapping):
        return dict(namespace)
    return {
        key: value for key, value in vars(namespace).items() if not key.startswith("_")
    }


class EuclideanFlowMatcher:
    def __init__(self, modality_name: str, conf: Any):
        self.modality_name = modality_name
        self.dim = int(conf.dim)
        self.zero_center_noise = bool(getattr(conf, "zero_center_noise", False))
        self.zero_center_mode = getattr(conf, "zero_center_mode", "all")
        if self.zero_center_mode not in {"all", "c4_anchor"}:
            raise ValueError(f"Unsupported zero_center_mode: {self.zero_center_mode}")
        self.coarse_mode = getattr(conf, "coarse_mode", None)
        self.noise_scale = float(getattr(conf, "noise_scale", 1.0))

    def _apply_mask(
        self,
        x_flow: torch.Tensor,
        x_fixed: torch.Tensor,
        flow_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if flow_mask is None:
            return x_flow
        flow_mask = ensure_float_mask(flow_mask)
        while flow_mask.ndim < x_flow.ndim:
            flow_mask = flow_mask.unsqueeze(-1)
        return flow_mask * x_flow + (1.0 - flow_mask) * x_fixed

    def _force_zero_com(
        self, x: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        if not self.zero_center_noise:
            return x
        if mask is None:
            mask = torch.ones(x.shape[:2], device=x.device, dtype=x.dtype)
        mask = ensure_float_mask(mask)
        while mask.ndim < x.ndim:
            mask = mask.unsqueeze(-1)
        if self.zero_center_mode == "c4_anchor" and self.coarse_mode is not None:
            c4 = extract_c4_positions(x, mode=self.coarse_mode)
            center = (c4 * mask[..., :1]).sum(dim=1, keepdim=True)
            denom = mask[..., :1].sum(dim=1, keepdim=True).clamp_min(EPS)
            center = center / denom
            shift = torch.cat([center] * (x.shape[-1] // 3), dim=-1)
            return (x - shift) * mask
        center = (x * mask).sum(dim=1, keepdim=True)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(EPS)
        center = center / denom
        return (x - center) * mask

    def sample_noise(
        self,
        *,
        num_batch: int,
        num_res: int,
        device: torch.device,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        noise = (
            torch.randn(num_batch, num_res, self.dim, device=device) * self.noise_scale
        )
        noise = self._force_zero_com(noise, mask)
        return noise * ensure_float_mask(mask).unsqueeze(-1)

    def nn_out_to_clean_sample(
        self,
        *,
        x_t: torch.Tensor,
        t: torch.Tensor,
        nn_out: torch.Tensor,
        res_mask: torch.Tensor,
        flow_mask: torch.Tensor,
    ) -> torch.Tensor:
        clean = x_t + (1.0 - t[:, None, None]) * nn_out
        clean = self._force_zero_com(clean, res_mask)
        clean = self._apply_mask(clean, x_t, flow_mask)
        return clean * ensure_float_mask(res_mask).unsqueeze(-1)

    def simulation_step(
        self,
        *,
        x_t: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        nn_out: torch.Tensor,
        res_mask: torch.Tensor,
        flow_mask: torch.Tensor,
    ) -> torch.Tensor:
        del t
        x_next = x_t + dt[:, None, None] * nn_out
        x_next = self._force_zero_com(x_next, res_mask)
        x_next = self._apply_mask(x_next, x_t, flow_mask)
        return x_next * ensure_float_mask(res_mask).unsqueeze(-1)


class ProductFlowMatcher:
    def __init__(self, conf: Any):
        self.conf = conf
        members = _namespace_items(conf)
        self.num_timesteps = int(members.pop("num_timesteps", 200))
        modality_names = members.pop("modalities", None)
        if modality_names is None:
            modality_names = [
                name for name, value in members.items() if hasattr(value, "dim")
            ]
        self.modalities = list(modality_names)
        self.matchers = {
            name: EuclideanFlowMatcher(name, getattr(conf, name))
            for name in self.modalities
        }

    def sample_noise_dict(
        self,
        *,
        num_batch: int,
        num_res: int,
        device: torch.device,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {
            name: matcher.sample_noise(
                num_batch=num_batch,
                num_res=num_res,
                device=device,
                mask=mask,
            )
            for name, matcher in self.matchers.items()
        }

    def nn_out_to_clean_sample(
        self,
        *,
        x_t: dict[str, torch.Tensor],
        t: dict[str, torch.Tensor],
        model_out: dict[str, dict[str, torch.Tensor]],
        res_mask: torch.Tensor,
        flow_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {
            name: self.matchers[name].nn_out_to_clean_sample(
                x_t=x_t[name],
                t=t[name],
                nn_out=model_out[name]["v"],
                res_mask=res_mask,
                flow_mask=flow_mask,
            )
            for name in self.modalities
        }

    def step(
        self,
        *,
        x_t: dict[str, torch.Tensor],
        t: dict[str, torch.Tensor],
        dt: torch.Tensor | dict[str, torch.Tensor],
        model_out: dict[str, dict[str, torch.Tensor]],
        res_mask: torch.Tensor,
        flow_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {
            name: self.matchers[name].simulation_step(
                x_t=x_t[name],
                t=t[name],
                dt=dt[name] if isinstance(dt, Mapping) else dt,
                nn_out=model_out[name]["v"],
                res_mask=res_mask,
                flow_mask=flow_mask,
            )
            for name in self.modalities
        }


class LatentFlowModel(nn.Module):
    def __init__(
        self,
        flow_model_conf: Any,
        coarse_mode: str = "p_c4prime_n",
        coarse_to_physical: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        super().__init__()
        self.flow_model_conf = flow_model_conf
        self.coarse_mode = coarse_mode
        self.coarse_dim = get_coarse_backbone_dim(coarse_mode)

        self._coarse_to_physical = (
            coarse_to_physical
            if coarse_to_physical is not None
            else lambda value: value
        )
        self.latent_dim = int(flow_model_conf.latent_dim)
        self.relpos_dim = int(flow_model_conf.relpos_dim)
        self.dist_num_bins = int(flow_model_conf.dist_num_bins)
        self.dist_min = float(flow_model_conf.dist_min)
        self.dist_max = float(flow_model_conf.dist_max)
        self.time_emb_dim = int(flow_model_conf.time_emb_dim)
        token_input_dim = (
            self.coarse_dim + self.latent_dim + self.coarse_dim + self.latent_dim + 1
        )
        pair_input_dim = self.relpos_dim + 2 * get_coarse_pair_feature_dim(
            coarse_mode, self.dist_num_bins
        )
        cond_input_dim = 2 * self.time_emb_dim
        pair_update_conf = getattr(flow_model_conf, "pair_update", None)
        pu_enabled = pair_update_conf is not None and pair_update_conf.enabled
        if pu_enabled:
            pu_every = pair_update_conf.every
            pu_cueq = pair_update_conf.use_cuequivariance
            pu_exp = pair_update_conf.expansion
            pu_t2p = pair_update_conf.token_to_pair
            pu_opm_c = pair_update_conf.opm_c_hidden if pu_t2p == "opm" else 32
            pu_opm_chunk = (
                getattr(pair_update_conf, "opm_chunk_size", None)
                if pu_t2p == "opm"
                else None
            )
        else:
            pu_every = 1
            pu_cueq = True
            pu_exp = 2
            pu_t2p = "none"
            pu_opm_c = 32
            pu_opm_chunk = None
        self.trunk = PairBiasedTransformer(
            token_input_dim=token_input_dim,
            pair_input_dim=pair_input_dim,
            cond_input_dim=cond_input_dim,
            token_dim=int(flow_model_conf.token_dim),
            pair_dim=int(flow_model_conf.pair_dim),
            cond_dim=int(flow_model_conf.cond_dim),
            num_layers=int(flow_model_conf.num_layers),
            num_heads=int(flow_model_conf.num_heads),
            use_qkln=bool(getattr(flow_model_conf, "use_qkln", True)),
            transition_expansion=int(
                getattr(flow_model_conf, "transition_expansion", 4)
            ),
            pair_update_enabled=pu_enabled,
            pair_update_every=pu_every,
            pair_update_use_cuequivariance=pu_cueq,
            pair_update_expansion=pu_exp,
            pair_update_token_to_pair=pu_t2p,
            pair_update_opm_c_hidden=pu_opm_c,
            pair_update_opm_chunk_size=pu_opm_chunk,
        )
        hidden_dim = int(flow_model_conf.token_dim)
        self.bb_coarse_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, self.coarse_dim)
        )
        self.local_latent_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, self.latent_dim)
        )

    def forward(
        self,
        *,
        x_t: dict[str, torch.Tensor],
        t: dict[str, torch.Tensor],
        x_sc: dict[str, torch.Tensor],
        res_mask: torch.Tensor,
        flow_mask: torch.Tensor,
        res_idx: torch.Tensor,
    ) -> dict[str, dict[str, torch.Tensor]]:
        res_mask = ensure_float_mask(res_mask)
        flow_mask = ensure_float_mask(flow_mask)
        bb_t = x_t["bb_coarse"]
        bb_sc = x_sc["bb_coarse"]
        lat_t = x_t["local_latents"]
        lat_sc = x_sc["local_latents"]

        bb_t_for_pair = self._coarse_to_physical(bb_t)
        bb_sc_for_pair = self._coarse_to_physical(bb_sc)
        t_bb = t["bb_coarse"]
        t_local = t.get("local_latents", torch.zeros_like(t_bb))
        token_inputs = torch.cat(
            [
                bb_t,
                lat_t,
                bb_sc,
                lat_sc,
                flow_mask.unsqueeze(-1),
            ],
            dim=-1,
        ) * res_mask.unsqueeze(-1)
        pair_inputs = torch.cat(
            [
                build_relpos_features(res_idx, self.relpos_dim),
                build_coarse_distance_features(
                    bb_t_for_pair,
                    dist_min=self.dist_min,
                    dist_max=self.dist_max,
                    dist_num_bins=self.dist_num_bins,
                    mode=self.coarse_mode,
                ),
                build_coarse_distance_features(
                    bb_sc_for_pair,
                    dist_min=self.dist_min,
                    dist_max=self.dist_max,
                    dist_num_bins=self.dist_num_bins,
                    mode=self.coarse_mode,
                ),
            ],
            dim=-1,
        )
        cond_inputs = torch.cat(
            [
                get_timestep_features(t_bb, self.time_emb_dim, res_mask),
                get_timestep_features(t_local, self.time_emb_dim, res_mask),
            ],
            dim=-1,
        )
        token_repr = self.trunk(token_inputs, pair_inputs, cond_inputs, res_mask)
        bb_coarse_v = self.bb_coarse_head(token_repr) * res_mask.unsqueeze(-1)
        local_latents_v = self.local_latent_head(token_repr) * res_mask.unsqueeze(-1)
        return {
            "bb_coarse": {"v": bb_coarse_v},
            "local_latents": {"v": local_latents_v},
            "token_repr": token_repr,
        }


class LatentFlowModule(nn.Module):
    def __init__(self, flow_module_conf: Any):
        super().__init__()
        self.flow_module_conf = flow_module_conf
        self.coarse_mode = get_explicit_backbone_mode(flow_module_conf.autoencoder_conf)
        expected_coarse_dim = get_coarse_backbone_dim(self.coarse_mode)
        flow_model_conf = flow_module_conf.flow_model_conf
        expected_latent_dim = int(flow_module_conf.autoencoder_conf.latent_dim)
        dimension_checks = {
            "flow_model_conf.coarse_dim": (
                int(flow_model_conf.coarse_dim),
                expected_coarse_dim,
            ),
            "product_flowmatcher.bb_coarse.dim": (
                int(flow_module_conf.product_flowmatcher.bb_coarse.dim),
                expected_coarse_dim,
            ),
            "flow_model_conf.latent_dim": (
                int(flow_model_conf.latent_dim),
                expected_latent_dim,
            ),
            "product_flowmatcher.local_latents.dim": (
                int(flow_module_conf.product_flowmatcher.local_latents.dim),
                expected_latent_dim,
            ),
        }
        mismatches = {
            name: {"configured": actual, "expected": expected}
            for name, (actual, expected) in dimension_checks.items()
            if actual != expected
        }
        configured_mode = str(flow_model_conf.explicit_backbone_mode)
        if configured_mode != self.coarse_mode:
            mismatches["flow_model_conf.explicit_backbone_mode"] = {
                "configured": configured_mode,
                "expected": self.coarse_mode,
            }
        configured_flowmatcher_mode = str(
            flow_module_conf.product_flowmatcher.bb_coarse.coarse_mode
        )
        if configured_flowmatcher_mode != self.coarse_mode:
            mismatches["product_flowmatcher.bb_coarse.coarse_mode"] = {
                "configured": configured_flowmatcher_mode,
                "expected": self.coarse_mode,
            }
        if mismatches:
            raise ValueError(
                f"Latent-flow dimension/mode contract mismatch: {mismatches}"
            )
        self.autoencoder = RNAPartialAutoencoder(flow_module_conf.autoencoder_conf)
        autoencoder_checkpoint_path = getattr(
            flow_module_conf,
            "autoencoder_ckpt_path",
            None,
        )
        autoencoder_checkpoint_identity = (
            build_file_identity(autoencoder_checkpoint_path)
            if autoencoder_checkpoint_path is not None
            else None
        )
        self._load_autoencoder_checkpoint(autoencoder_checkpoint_path)
        if (
            autoencoder_checkpoint_identity is not None
            and build_file_identity(autoencoder_checkpoint_path)
            != autoencoder_checkpoint_identity
        ):
            raise RuntimeError(
                "Autoencoder checkpoint changed while constructing LatentFlowModule: "
                f"{autoencoder_checkpoint_path}"
            )
        self.coarse_preprocess = CoarsePreprocessor(
            getattr(flow_module_conf, "coarse_preprocess", None),
            expected_backbone_mode=self.coarse_mode,
            expected_zero_center=bool(
                getattr(
                    flow_module_conf.autoencoder_conf.explicit_backbone,
                    "zero_center",
                    True,
                )
            ),
        )
        self.flow_model = LatentFlowModel(
            flow_module_conf.flow_model_conf,
            coarse_mode=self.coarse_mode,
            coarse_to_physical=self.coarse_preprocess.to_physical,
        )
        self.flow_compute_precision = "fp32"
        self.flow_matcher = ProductFlowMatcher(flow_module_conf.product_flowmatcher)
        required_modalities = {"bb_coarse", "local_latents"}
        configured_modalities = set(self.flow_matcher.modalities)
        if configured_modalities != required_modalities:
            raise ValueError(
                "LatentFlowModule supports only joint bb_coarse + local_latents flow in "
                f"the mainline; got modalities={self.flow_matcher.modalities!r}."
            )
        self.sampling_conf = flow_module_conf.sampling_conf
        self.latent_preprocess = LatentPreprocessor(
            flow_module_conf.latent_preprocess,
            latent_dim=self.autoencoder.latent_dim,
        )
        self.self_condition = bool(self.sampling_conf.self_condition)
        self.sampling_schedule_conf = self.sampling_conf.sampling_schedule
        for parameter in self.autoencoder.parameters():
            parameter.requires_grad = False
        self.autoencoder.eval()

    def set_flow_compute_precision(self, precision: str) -> None:
        if precision not in {"fp32", "bf16"}:
            raise ValueError(
                f"Unsupported flow compute precision {precision!r}; "
                "expected 'fp32' or 'bf16'"
            )
        self.flow_compute_precision = precision

    def _run_flow_model(self, **kwargs: Any) -> dict[str, Any]:
        if self.flow_compute_precision == "fp32":
            return self.flow_model(**kwargs)
        sample = kwargs["x_t"]["bb_coarse"]
        if sample.device.type != "cuda":
            raise RuntimeError("Flow BF16 flow compute requires CUDA")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return self.flow_model(**kwargs)

    def preprocessing_metadata(self) -> dict[str, Any]:
        return {
            "latent": {
                "mode": self.latent_preprocess.mode,
                "eps": self.latent_preprocess.eps,
                "mean": self.latent_preprocess.latent_mean.detach()
                .cpu()
                .reshape(-1)
                .tolist(),
                "std": self.latent_preprocess.latent_std.detach()
                .cpu()
                .reshape(-1)
                .tolist(),
            },
            "coarse": {
                "mode": self.coarse_preprocess.mode,
                "eps": self.coarse_preprocess.eps,
                "scale": float(
                    self.coarse_preprocess.coarse_scale.detach().cpu().reshape(())
                ),
            },
        }

    def validate_checkpoint(
        self, checkpoint: Mapping[str, Any], *, context: str
    ) -> None:
        state_dict = checkpoint.get("model", checkpoint)
        validate_flow_preprocess_checkpoint_state(state_dict)

    def _load_autoencoder_checkpoint(self, checkpoint_path: str | None) -> None:
        if checkpoint_path is None:
            return
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, Mapping):
            raise TypeError(
                f"autoencoder checkpoint {checkpoint_path} is not a mapping"
            )
        state_dict = checkpoint.get("model", checkpoint)
        normalized_state = normalize_checkpoint_state_dict(
            state_dict,
            strip_prefixes=("module.", "model.", "autoencoder."),
        )
        if int(normalized_state["rna_geometry_version_code"]) != RNA_GEOMETRY_VERSION_CODE:
            raise RuntimeError("Autoencoder geometry version tensor is incompatible")
        decoder_state = {
            key: value
            for key, value in normalized_state.items()
            if key == "rna_geometry_version_code" or key.startswith("decoder.")
        }
        load_state_dict_with_policy(
            self.autoencoder,
            decoder_state,
            context=f"autoencoder checkpoint {checkpoint_path}",
        )

    def _zero_self_condition(
        self, x_1: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return {name: torch.zeros_like(value) for name, value in x_1.items()}

    def _bb_to_physical(self, bb_flow: torch.Tensor) -> torch.Tensor:
        return self.coarse_preprocess.to_physical(bb_flow)

    @torch.no_grad()
    def sample(
        self,
        *,
        num_batch: int,
        num_res: int,
        num_timesteps: int | None = None,
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        if num_batch <= 0:
            raise ValueError("num_batch must be positive")
        if num_res <= 0:
            raise ValueError("num_res must be positive")
        if device is None:
            device = next(self.parameters()).device
        if num_timesteps is None:
            num_timesteps = self.flow_matcher.num_timesteps
        if num_timesteps <= 0:
            raise ValueError("num_timesteps must be positive")
        res_mask = torch.ones(num_batch, num_res, device=device)
        flow_mask = torch.ones_like(res_mask)
        res_idx = torch.arange(num_res, device=device).unsqueeze(0).repeat(num_batch, 1)
        is_na_residue_mask = torch.ones_like(res_mask, dtype=torch.bool)
        x_t = self.flow_matcher.sample_noise_dict(
            num_batch=num_batch,
            num_res=num_res,
            device=device,
            mask=res_mask,
        )
        x_sc = self._zero_self_condition(x_t)
        time_grids = build_sampling_time_grids(
            self.flow_matcher.modalities,
            num_timesteps,
            device,
            self.sampling_schedule_conf,
        )
        self._last_sampling_time_grids = {
            name: grid.detach().cpu() for name, grid in time_grids.items()
        }
        for step_idx in range(num_timesteps):
            t_dict = {
                name: grid[step_idx].expand(num_batch)
                for name, grid in time_grids.items()
            }
            dt_dict = {
                name: (grid[step_idx + 1] - grid[step_idx]).expand(num_batch)
                for name, grid in time_grids.items()
            }
            model_out = self._run_flow_model(
                x_t=x_t,
                t=t_dict,
                x_sc=x_sc,
                res_mask=res_mask,
                flow_mask=flow_mask,
                res_idx=res_idx,
            )
            clean_pred = self.flow_matcher.nn_out_to_clean_sample(
                x_t=x_t,
                t=t_dict,
                model_out=model_out,
                res_mask=res_mask,
                flow_mask=flow_mask,
            )
            x_t = self.flow_matcher.step(
                x_t=x_t,
                t=t_dict,
                dt=dt_dict,
                model_out=model_out,
                res_mask=res_mask,
                flow_mask=flow_mask,
            )
            if self.self_condition:
                x_sc = {key: value.detach() for key, value in clean_pred.items()}
        bb_physical = self._bb_to_physical(x_t["bb_coarse"])
        base_output = {
            "bb_coarse": bb_physical,
            "bb_coarse_physical": bb_physical,
            "bb_coarse_flow": x_t["bb_coarse"],
            "res_mask": res_mask,
            "res_idx": res_idx,
            "is_na_residue_mask": is_na_residue_mask,
        }
        decoded_latents = self.latent_preprocess.unwhiten(x_t["local_latents"])
        decode_out = self.autoencoder.decode(
            z_latent=decoded_latents,
            bb_coarse=bb_physical,
            res_mask=res_mask,
            res_idx=res_idx,
            is_na_residue_mask=is_na_residue_mask,
            aatype_for_atoms=None,
        )
        return {
            **decode_out,
            **base_output,
            "local_latents": decoded_latents,
            "local_latents_flow": x_t["local_latents"],
        }

    @torch.no_grad()
    def sample_latents_conditioned_on_scaffold(
        self,
        *,
        bb_coarse: torch.Tensor,
        res_mask: torch.Tensor,
        res_idx: torch.Tensor,
        num_timesteps: int | None = None,
    ) -> dict[str, torch.Tensor]:

        if num_timesteps is None:
            num_timesteps = self.flow_matcher.num_timesteps
        if num_timesteps <= 0:
            raise ValueError("num_timesteps must be positive")

        res_mask = ensure_float_mask(res_mask)
        flow_mask = res_mask
        bb_physical = bb_coarse * res_mask.unsqueeze(-1)
        bb_flow = self.coarse_preprocess.to_flow(bb_physical)
        num_batch, num_res = res_mask.shape
        device = bb_coarse.device
        latent_matcher = self.flow_matcher.matchers["local_latents"]
        x_t = {
            "bb_coarse": bb_flow,
            "local_latents": latent_matcher.sample_noise(
                num_batch=num_batch,
                num_res=num_res,
                device=device,
                mask=res_mask,
            ),
        }
        x_sc = self._zero_self_condition(x_t)
        latent_grid = build_sampling_time_grids(
            self.flow_matcher.modalities,
            num_timesteps,
            device,
            self.sampling_schedule_conf,
        )["local_latents"]
        scaffold_grid = torch.ones_like(latent_grid)
        self._last_sampling_time_grids = {
            "bb_coarse": scaffold_grid.detach().cpu(),
            "local_latents": latent_grid.detach().cpu(),
        }

        scaffold_time = torch.ones(num_batch, device=device)
        for step_idx in range(num_timesteps):
            latent_time = latent_grid[step_idx].expand(num_batch)
            t_dict = {
                "bb_coarse": scaffold_time,
                "local_latents": latent_time,
            }
            model_out = self._run_flow_model(
                x_t=x_t,
                t=t_dict,
                x_sc=x_sc,
                res_mask=res_mask,
                flow_mask=flow_mask,
                res_idx=res_idx,
            )
            clean_pred = self.flow_matcher.nn_out_to_clean_sample(
                x_t=x_t,
                t=t_dict,
                model_out=model_out,
                res_mask=res_mask,
                flow_mask=flow_mask,
            )
            latent_dt = (latent_grid[step_idx + 1] - latent_grid[step_idx]).expand(
                num_batch
            )
            next_latents = latent_matcher.simulation_step(
                x_t=x_t["local_latents"],
                t=latent_time,
                dt=latent_dt,
                nn_out=model_out["local_latents"]["v"],
                res_mask=res_mask,
                flow_mask=flow_mask,
            )
            x_t = {
                "bb_coarse": bb_flow,
                "local_latents": next_latents,
            }
            if self.self_condition:
                x_sc = {key: value.detach() for key, value in clean_pred.items()}
                x_sc["bb_coarse"] = bb_flow

        decoded_latents = self.latent_preprocess.unwhiten(x_t["local_latents"])
        is_na_residue_mask = res_mask.bool()
        decode_out = self.autoencoder.decode(
            z_latent=decoded_latents,
            bb_coarse=bb_physical,
            res_mask=res_mask,
            res_idx=res_idx,
            is_na_residue_mask=is_na_residue_mask,
            aatype_for_atoms=None,
        )
        return {
            **decode_out,
            "bb_coarse": bb_physical,
            "bb_coarse_physical": bb_physical,
            "bb_coarse_flow": bb_flow,
            "res_mask": res_mask,
            "res_idx": res_idx,
            "is_na_residue_mask": is_na_residue_mask,
            "local_latents": decoded_latents,
            "local_latents_flow": x_t["local_latents"],
        }
