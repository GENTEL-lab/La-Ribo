from __future__ import annotations

import torch
from einops import rearrange
from torch import Tensor, nn

from src.model.latent.triangle import PairUpdateBlock


class SwiGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return x * torch.nn.functional.silu(gate)


class AdaptiveLayerNorm(nn.Module):
    def __init__(self, dim: int, dim_cond: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm_cond = nn.LayerNorm(dim_cond)
        self.to_gamma = nn.Sequential(nn.Linear(dim_cond, dim), nn.Sigmoid())
        self.to_beta = nn.Linear(dim_cond, dim, bias=False)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        normed = self.norm(x)
        cond = self.norm_cond(cond)
        gamma = self.to_gamma(cond)
        beta = self.to_beta(cond)
        return (normed * gamma + beta) * mask[..., None]


class AdaptiveOutputScale(nn.Module):
    def __init__(self, dim: int, dim_cond: int, bias_init: float = -2.0) -> None:
        super().__init__()
        linear = nn.Linear(dim_cond, dim)
        nn.init.zeros_(linear.weight)
        nn.init.constant_(linear.bias, bias_init)
        self.to_gamma = nn.Sequential(linear, nn.Sigmoid())

    def forward(self, x: torch.Tensor, cond: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        gamma = self.to_gamma(cond)
        return x * gamma * mask[..., None]


class PairBiasAttention(nn.Module):
    def __init__(
        self,
        node_dim: int,
        dim_head: int,
        heads: int,
        dim_out: int,
        qkln: bool,
        pair_dim: int | None = None,
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.to_qkv = nn.Linear(node_dim, inner_dim * 3, bias=True)
        self.to_g = nn.Linear(node_dim, inner_dim)
        self.to_out = nn.Linear(inner_dim, dim_out)
        self.node_norm = nn.LayerNorm(node_dim)
        self.q_norm = nn.LayerNorm(inner_dim) if qkln else nn.Identity()
        self.k_norm = nn.LayerNorm(inner_dim) if qkln else nn.Identity()
        if pair_dim is not None:
            self.to_bias = nn.Linear(pair_dim, heads, bias=False)
            self.pair_norm = nn.LayerNorm(pair_dim)
        else:
            self.to_bias = None
            self.pair_norm = None

    def forward(
        self, node_feats: Tensor, pair_feats: Tensor | None, mask: Tensor | None
    ) -> Tensor:
        node_feats = self.node_norm(node_feats)
        pair_feats = self.pair_norm(pair_feats) if pair_feats is not None else None
        q, k, v = self.to_qkv(node_feats).chunk(3, dim=-1)
        q = self.q_norm(q)
        k = self.k_norm(k)
        g = self.to_g(node_feats)
        bias = (
            rearrange(self.to_bias(pair_feats), "b i j h -> b h i j")
            if pair_feats is not None
            else 0
        )
        q, k, v, g = (
            rearrange(tensor, "b n (h d) -> b h n d", h=self.heads) for tensor in (q, k, v, g)
        )
        attn_out = self._attend(q, k, v, bias, mask)
        attn_out = rearrange(torch.sigmoid(g) * attn_out, "b h n d -> b n (h d)")
        return self.to_out(attn_out)

    def _attend(
        self, q: Tensor, k: Tensor, v: Tensor, bias: Tensor | int, mask: Tensor | None
    ) -> Tensor:
        sim = (q.unsqueeze(-2) * k.unsqueeze(-3)).sum(dim=-1) * self.scale
        if mask is not None:
            mask = rearrange(mask, "b i j -> b 1 i j")
            sim = sim.masked_fill(~mask.bool(), torch.finfo(sim.dtype).min)
        attn = torch.softmax(sim + bias, dim=-1)
        return (attn.unsqueeze(-1) * v.unsqueeze(-3)).sum(dim=-2)


class MultiHeadBiasedAttentionADALN(nn.Module):
    def __init__(
        self, dim_token: int, dim_pair: int, nheads: int, dim_cond: int, use_qkln: bool
    ) -> None:
        super().__init__()
        dim_head = dim_token // nheads
        self.adaln = AdaptiveLayerNorm(dim=dim_token, dim_cond=dim_cond)
        self.attn = PairBiasAttention(
            node_dim=dim_token,
            dim_head=dim_head,
            heads=nheads,
            dim_out=dim_token,
            qkln=use_qkln,
            pair_dim=dim_pair,
        )
        self.out_scale = AdaptiveOutputScale(dim=dim_token, dim_cond=dim_cond)

    def forward(
        self, x: torch.Tensor, pair_rep: torch.Tensor, cond: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        pair_mask = mask[:, :, None] * mask[:, None, :]
        x = self.adaln(x, cond, mask)
        x = self.attn(node_feats=x, pair_feats=pair_rep, mask=pair_mask)
        x = self.out_scale(x, cond, mask)
        return x * mask[..., None]


class Transition(nn.Module):
    def __init__(self, dim: int, expansion_factor: int = 4, layer_norm: bool = False) -> None:
        super().__init__()
        dim_inner = int(dim * expansion_factor)
        self.use_layer_norm = layer_norm
        self.ln = nn.LayerNorm(dim) if layer_norm else None
        self.swiglu = nn.Sequential(nn.Linear(dim, dim_inner * 2, bias=False), SwiGLU())
        self.linear_out = nn.Linear(dim_inner, dim, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.use_layer_norm and self.ln is not None:
            x = self.ln(x)
        x = self.linear_out(self.swiglu(x))
        return x * mask[..., None]


class TransitionADALN(nn.Module):
    def __init__(self, dim: int, dim_cond: int, expansion_factor: int = 4) -> None:
        super().__init__()
        self.adaln = AdaptiveLayerNorm(dim=dim, dim_cond=dim_cond)
        self.transition = Transition(dim=dim, expansion_factor=expansion_factor, layer_norm=False)
        self.out_scale = AdaptiveOutputScale(dim=dim, dim_cond=dim_cond)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.adaln(x, cond, mask)
        x = self.transition(x, mask)
        x = self.out_scale(x, cond, mask)
        return x * mask[..., None]


class MultiheadAttnAndTransition(nn.Module):
    def __init__(
        self,
        dim_token: int,
        dim_pair: int,
        nheads: int,
        dim_cond: int,
        use_qkln: bool,
        expansion_factor: int = 4,
    ) -> None:
        super().__init__()
        self.attn = MultiHeadBiasedAttentionADALN(dim_token, dim_pair, nheads, dim_cond, use_qkln)
        self.transition = TransitionADALN(dim_token, dim_cond, expansion_factor)

    def forward(
        self, x: torch.Tensor, pair_rep: torch.Tensor, cond: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.attn(x, pair_rep, cond, mask)
        x = x + self.transition(x, cond, mask)
        return x * mask[..., None]


class MLP(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or dim_out
        self.net = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim_out),
            nn.LayerNorm(dim_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PairBiasedTransformer(nn.Module):
    def __init__(
        self,
        token_input_dim: int,
        pair_input_dim: int,
        cond_input_dim: int,
        token_dim: int,
        pair_dim: int,
        cond_dim: int,
        num_layers: int,
        num_heads: int,
        use_qkln: bool = True,
        transition_expansion: int = 4,
        pair_update_enabled: bool = False,
        pair_update_every: int = 1,
        pair_update_use_cuequivariance: bool = True,
        pair_update_expansion: int = 2,
        pair_update_token_to_pair: str = "none",
        pair_update_opm_c_hidden: int = 32,
        pair_update_opm_chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        self.token_proj = MLP(token_input_dim, token_dim)
        self.pair_proj = MLP(pair_input_dim, pair_dim)
        self.cond_proj = MLP(cond_input_dim, cond_dim)
        self.blocks = nn.ModuleList(
            [
                MultiheadAttnAndTransition(
                    dim_token=token_dim,
                    dim_pair=pair_dim,
                    nheads=num_heads,
                    dim_cond=cond_dim,
                    use_qkln=use_qkln,
                    expansion_factor=transition_expansion,
                )
                for _ in range(num_layers)
            ]
        )
        self.pair_update_enabled = pair_update_enabled
        self.pair_update_every = pair_update_every
        if self.pair_update_enabled:


            num_pair_blocks = (num_layers - 1) // self.pair_update_every
            self.pair_blocks = nn.ModuleList(
                [
                    PairUpdateBlock(
                        dim=pair_dim,
                        use_cuequivariance=bool(pair_update_use_cuequivariance),
                        transition_expansion=int(pair_update_expansion),
                        token_dim=token_dim,
                        token_to_pair=pair_update_token_to_pair,
                        opm_c_hidden=int(pair_update_opm_c_hidden),
                        opm_chunk_size=pair_update_opm_chunk_size,
                    )
                    for _ in range(num_pair_blocks)
                ]
            )
        else:
            self.pair_blocks = nn.ModuleList()

    def forward(
        self,
        token_inputs: torch.Tensor,
        pair_inputs: torch.Tensor,
        cond_inputs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        token_repr = self.token_proj(token_inputs) * mask[..., None]
        pair_mask = mask[:, :, None] * mask[:, None, :]
        pair_repr = self.pair_proj(pair_inputs) * pair_mask[..., None]
        cond_repr = self.cond_proj(cond_inputs) * mask[..., None]
        last_layer = len(self.blocks) - 1
        for i, block in enumerate(self.blocks):
            token_repr = block(token_repr, pair_repr, cond_repr, mask)


            if (
                self.pair_update_enabled
                and (i + 1) % self.pair_update_every == 0
                and i < last_layer
            ):
                pair_repr = self.pair_blocks[i // self.pair_update_every](
                    pair_repr, pair_mask, token=token_repr, node_mask=mask
                )
        return token_repr * mask[..., None]
