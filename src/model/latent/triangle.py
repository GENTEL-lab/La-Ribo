


from __future__ import annotations

import torch
import torch.nn.functional as F


from torch import nn

from src.model.latent.token_to_pair import build_token_to_pair


class TriangleMultiplicativeUpdate(nn.Module):


    def __init__(
        self,
        dim: int,
        direction: str = "outgoing",
        use_cuequivariance: bool = True,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if direction not in ("outgoing", "incoming"):
            raise ValueError(f"direction must be 'outgoing' or 'incoming', got {direction!r}")
        self.dim = int(dim)
        self.direction = direction
        self.eps = float(eps)
        self.use_cuequivariance = bool(use_cuequivariance)
        if use_cuequivariance and self.dim % 32 != 0:
            raise ValueError(
                f"cuequivariance triangle kernel requires dim % 32 == 0, got dim={self.dim}"
            )

        D = self.dim

        self.norm_in_weight = nn.Parameter(torch.ones(D))
        self.norm_in_bias = nn.Parameter(torch.zeros(D))

        self.p_in_weight = nn.Parameter(torch.empty(2 * D, D))
        self.p_in_bias = nn.Parameter(torch.zeros(2 * D))
        self.g_in_weight = nn.Parameter(torch.empty(2 * D, D))
        self.g_in_bias = nn.Parameter(torch.zeros(2 * D))

        self.norm_out_weight = nn.Parameter(torch.ones(D))
        self.norm_out_bias = nn.Parameter(torch.zeros(D))

        self.p_out_weight = nn.Parameter(torch.zeros(D, D))
        self.p_out_bias = nn.Parameter(torch.zeros(D))

        self.g_out_weight = nn.Parameter(torch.empty(D, D))
        self.g_out_bias = nn.Parameter(torch.zeros(D))

        nn.init.xavier_uniform_(self.p_in_weight)
        nn.init.xavier_uniform_(self.g_in_weight)
        nn.init.xavier_uniform_(self.g_out_weight)

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, direction={self.direction!r}, "
            f"use_cuequivariance={self.use_cuequivariance}"
        )

    def forward(
        self,
        pair: torch.Tensor,
        pair_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:


        if self.use_cuequivariance and pair.is_cuda:
            from cuequivariance_torch import triangle_multiplicative_update as _cueq_tri_update

            kernel_mask = pair_mask
            if kernel_mask is not None and kernel_mask.dtype != pair.dtype:
                kernel_mask = kernel_mask.to(pair.dtype)
            return _cueq_tri_update(
                pair,
                direction=self.direction,
                mask=kernel_mask,
                norm_in_weight=self.norm_in_weight,
                norm_in_bias=self.norm_in_bias,
                p_in_weight=self.p_in_weight,
                p_in_bias=self.p_in_bias,
                g_in_weight=self.g_in_weight,
                g_in_bias=self.g_in_bias,
                norm_out_weight=self.norm_out_weight,
                norm_out_bias=self.norm_out_bias,
                p_out_weight=self.p_out_weight,
                p_out_bias=self.p_out_bias,
                g_out_weight=self.g_out_weight,
                g_out_bias=self.g_out_bias,
                eps=self.eps,
            )
        return self._pytorch_forward(pair, pair_mask)

    def _pytorch_forward(
        self,
        pair: torch.Tensor,
        pair_mask: torch.Tensor | None,
    ) -> torch.Tensor:

        if pair_mask is None:
            mask = pair.new_ones(pair.shape[:-1]).unsqueeze(-1)
        else:
            mask = pair_mask.to(pair.dtype).unsqueeze(-1)


        z = F.layer_norm(
            pair,
            (self.dim,),
            self.norm_in_weight,
            self.norm_in_bias,
            eps=self.eps,
        )


        p_proj = F.linear(z, self.p_in_weight, self.p_in_bias)
        g_proj = F.linear(z, self.g_in_weight, self.g_in_bias)
        a_p, b_p = p_proj.chunk(2, dim=-1)
        a_g, b_g = g_proj.chunk(2, dim=-1)
        a = a_p * torch.sigmoid(a_g) * mask
        b = b_p * torch.sigmoid(b_g) * mask


        if self.direction == "outgoing":

            x = torch.einsum("bikd,bjkd->bijd", a, b)
        else:

            x = torch.einsum("bkid,bkjd->bijd", a, b)


        x = F.layer_norm(
            x,
            (self.dim,),
            self.norm_out_weight,
            self.norm_out_bias,
            eps=self.eps,
        )
        x = F.linear(x, self.p_out_weight, self.p_out_bias)


        g_out = torch.sigmoid(F.linear(z, self.g_out_weight, self.g_out_bias))
        return x * g_out


class PairUpdateBlock(nn.Module):


    def __init__(
        self,
        dim: int,
        use_cuequivariance: bool = True,
        transition_expansion: int = 2,
        token_dim: int | None = None,
        token_to_pair: str = "none",
        opm_c_hidden: int = 32,
        opm_chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        self.dim = int(dim)

        self.token_to_pair = build_token_to_pair(
            token_to_pair,
            token_dim,
            self.dim,
            opm_c_hidden=opm_c_hidden,
            opm_chunk_size=opm_chunk_size,
        )
        self.tri_out = TriangleMultiplicativeUpdate(
            dim=dim,
            direction="outgoing",
            use_cuequivariance=use_cuequivariance,
        )
        self.tri_in = TriangleMultiplicativeUpdate(
            dim=dim,
            direction="incoming",
            use_cuequivariance=use_cuequivariance,
        )
        hidden = int(dim * transition_expansion)
        self.transition_norm = nn.LayerNorm(dim)
        self.transition_linear_in = nn.Linear(dim, hidden)
        self.transition_linear_out = nn.Linear(hidden, dim)


        nn.init.zeros_(self.transition_linear_out.weight)
        nn.init.zeros_(self.transition_linear_out.bias)

    def _pair_transition(
        self,
        pair: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        h = self.transition_norm(pair)
        h = self.transition_linear_in(h)
        h = F.silu(h)
        h = self.transition_linear_out(h)
        return h * pair_mask.unsqueeze(-1)

    def forward(
        self,
        pair: torch.Tensor,
        pair_mask: torch.Tensor,
        token: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        mask_b = pair_mask.unsqueeze(-1)


        if self.token_to_pair is not None:
            if token is None:
                raise ValueError("token must be provided when token_to_pair is enabled")
            pair = (pair + self.token_to_pair(token, node_mask)) * mask_b
        pair = pair + self.tri_out(pair, pair_mask) * mask_b
        pair = pair + self.tri_in(pair, pair_mask) * mask_b
        pair = pair + self._pair_transition(pair, pair_mask)
        return pair * mask_b
