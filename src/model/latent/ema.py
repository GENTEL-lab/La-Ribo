from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn


class EMA:
    def __init__(self):
        self.decay = 0.999
        self.num_updates: int = 0
        self.shadow: dict[str, torch.Tensor] = {}

    @staticmethod
    def _validated_decay(decay: float) -> float:
        decay = float(decay)
        if not math.isfinite(decay) or not 0.0 <= decay < 1.0:
            raise ValueError(f"EMA decay must be finite and in [0, 1), got {decay}")
        return decay

    @torch.no_grad()
    def validate_against(self, model: nn.Module) -> None:

        parameters = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        expected = set(parameters)
        actual = set(self.shadow)
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        if missing or unexpected:
            raise RuntimeError(
                "EMA parameter keys do not match the model: "
                f"missing={missing}, unexpected={unexpected}"
            )
        for name, parameter in parameters.items():
            shadow = self.shadow[name]
            if shadow.shape != parameter.shape:
                raise RuntimeError(
                    f"EMA shape mismatch for {name}: shadow={tuple(shadow.shape)}, "
                    f"model={tuple(parameter.shape)}"
                )
            if not torch.isfinite(shadow).all():
                raise RuntimeError(f"EMA shadow for {name} contains non-finite values")

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:

        self.validate_against(model)
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                shadow = self.shadow[name]
                parameter.data.copy_(shadow.to(parameter.device, parameter.dtype))

    def load_state_dict(
        self,
        sd: Mapping,
        device: torch.device | str = "cpu",
        *,
        expected_decay: float | None = None,
    ) -> None:
        if not isinstance(sd, Mapping):
            raise TypeError("EMA state must be a mapping")
        decay = self._validated_decay(sd.get("decay"))
        if expected_decay is not None and decay != self._validated_decay(
            expected_decay
        ):
            raise RuntimeError(
                f"EMA checkpoint decay={decay} differs from configured decay={expected_decay}"
            )
        num_updates = sd.get("num_updates")
        if (
            isinstance(num_updates, bool)
            or not isinstance(num_updates, int)
            or num_updates < 0
        ):
            raise ValueError("EMA num_updates must be a non-negative integer")
        raw_shadow = sd.get("shadow")
        if not isinstance(raw_shadow, Mapping):
            raise TypeError("EMA shadow must be a mapping")
        shadow: dict[str, torch.Tensor] = {}
        for name, tensor in raw_shadow.items():
            if not isinstance(name, str) or not torch.is_tensor(tensor):
                raise TypeError("EMA shadow must map string names to tensors")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"EMA shadow for {name} contains non-finite values")
            shadow[name] = tensor.to(device)
        self.decay = decay
        self.num_updates = num_updates
        self.shadow = shadow

    @classmethod
    def from_state_dict(cls, sd: dict, device: torch.device | str = "cpu") -> EMA:

        obj = cls()
        obj.load_state_dict(sd, device=device)
        return obj
