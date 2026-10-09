


from __future__ import annotations

import torch
from torch import nn


class AddTokenToPair(nn.Module):


    def __init__(self, token_dim: int, pair_dim: int) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.pair_dim = int(pair_dim)
        self.layer_norm = nn.LayerNorm(self.token_dim)

        self.linear_x = nn.Linear(self.token_dim, 2 * self.pair_dim, bias=False)

    def extra_repr(self) -> str:
        return f"token_dim={self.token_dim}, pair_dim={self.pair_dim}"

    def forward(
        self,
        token: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:


        if node_mask is not None:
            token = token * node_mask[..., None].to(token.dtype)
        a, b = self.linear_x(self.layer_norm(token)).chunk(2, dim=-1)
        delta = a[:, None, :, :] + b[:, :, None, :]
        if node_mask is not None:
            pair_mask = node_mask[:, :, None] * node_mask[:, None, :]
            delta = delta * pair_mask[..., None].to(delta.dtype)
        return delta


class OuterProductTokenToPair(nn.Module):


    def __init__(
        self,
        token_dim: int,
        pair_dim: int,
        c_hidden: int = 32,
        chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.pair_dim = int(pair_dim)
        self.c_hidden = int(c_hidden)
        self.chunk_size = int(chunk_size) if chunk_size is not None else None
        self.layer_norm = nn.LayerNorm(self.token_dim)
        self.linear_a = nn.Linear(self.token_dim, self.c_hidden, bias=False)
        self.linear_b = nn.Linear(self.token_dim, self.c_hidden, bias=False)


        self.linear_out = nn.Linear(self.c_hidden * self.c_hidden, self.pair_dim)
        nn.init.zeros_(self.linear_out.weight)
        nn.init.zeros_(self.linear_out.bias)

    def extra_repr(self) -> str:
        return (
            f"token_dim={self.token_dim}, pair_dim={self.pair_dim}, "
            f"c_hidden={self.c_hidden}, chunk_size={self.chunk_size}"
        )

    def _opm(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:

        outer = torch.einsum("bic,bjd->bijcd", a, b)
        outer = outer.reshape(outer.shape[:-2] + (-1,))
        return self.linear_out(outer)

    def _opm_chunked(self, a: torch.Tensor, b: torch.Tensor, chunk_size: int) -> torch.Tensor:


        outs = [
            self._opm(a[:, start : start + chunk_size, :], b)
            for start in range(0, a.shape[1], chunk_size)
        ]
        return torch.cat(outs, dim=1)

    def _forward(self, token: torch.Tensor, node_mask: torch.Tensor | None) -> torch.Tensor:
        ln = self.layer_norm(token)
        a = self.linear_a(ln)
        b = self.linear_b(ln)
        if node_mask is not None:
            m = node_mask[..., None].to(a.dtype)
            a = a * m
            b = b * m
        if self.chunk_size is not None:
            outer = self._opm_chunked(a, b, self.chunk_size)
        else:
            outer = self._opm(a, b)
        if node_mask is not None:
            pair_mask = node_mask[:, :, None] * node_mask[:, None, :]
            outer = outer * pair_mask[..., None].to(outer.dtype)
        return outer

    def forward(
        self,
        token: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:


        orig_dtype = token.dtype
        device_type = "cuda" if token.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            delta = self._forward(token.float(), node_mask)
        return delta.to(orig_dtype)


def build_token_to_pair(
    mode: str,
    token_dim: int | None,
    pair_dim: int,
    opm_c_hidden: int = 32,
    opm_chunk_size: int | None = None,
) -> nn.Module | None:


    if mode == "none":
        return None
    if mode in ("add", "opm") and token_dim is None:
        raise ValueError(f"token_dim is required for token_to_pair={mode!r}")
    if mode == "add":
        return AddTokenToPair(int(token_dim), pair_dim)
    if mode == "opm":
        return OuterProductTokenToPair(
            int(token_dim), pair_dim, c_hidden=opm_c_hidden, chunk_size=opm_chunk_size
        )
    raise ValueError(f"unknown token_to_pair mode {mode!r} (expected none|add|opm)")
