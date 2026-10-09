from __future__ import annotations

from typing import Any

import torch

from src.data.basics import all_atom_na
from src.data.basics.constants import nucleotide_constants
from src.model.utils import (
    calc_distogram,
    distance_to_distogram,
    get_index_embedding,
    get_timestep_embedding,
)


RNA_NUM_TORSIONS = nucleotide_constants.NUM_NA_TORSIONS
EPS = 1e-8


DEGENERATE_EPS = 1e-6


def ensure_float_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
        return mask
    return mask.float()


def canonicalize_explicit_backbone_mode(
    mode: str | None, default: str = "p_c4prime_n"
) -> str:
    resolved = default if mode is None else mode
    if resolved not in {
        "c4prime",
        "c4prime_p",
        "p_c4prime_n",
        "p_c4prime_c1prime",
    }:
        raise ValueError(f"Unsupported explicit backbone mode: {resolved}")
    return resolved


def get_explicit_backbone_mode(conf: Any, default: str = "p_c4prime_n") -> str:
    explicit_conf = getattr(conf, "explicit_backbone", None)
    if explicit_conf is None:
        return canonicalize_explicit_backbone_mode(default, default=default)
    return canonicalize_explicit_backbone_mode(
        getattr(explicit_conf, "mode", default), default=default
    )


def get_explicit_backbone_beads(mode: str) -> tuple[str, ...]:
    mode = canonicalize_explicit_backbone_mode(mode)
    if mode == "c4prime":
        return ("C4'",)
    if mode == "c4prime_p":
        return ("C4'", "P")
    if mode == "p_c4prime_n":
        return ("P", "C4'", "N")
    if mode == "p_c4prime_c1prime":
        return ("P", "C4'", "C1'")
    raise ValueError(f"Unsupported explicit backbone mode: {mode}")


def get_coarse_backbone_dim(mode: str) -> int:
    return 3 * len(get_explicit_backbone_beads(mode))


def get_coarse_pair_feature_dim(mode: str, dist_num_bins: int) -> int:
    num_beads = len(get_explicit_backbone_beads(mode))
    return num_beads * num_beads * dist_num_bins


def validate_rna_torsions(torsions: torch.Tensor) -> torch.Tensor:
    if torsions.ndim != 4 or torsions.shape[-2:] != (RNA_NUM_TORSIONS, 2):
        raise ValueError(
            f"Expected [B, N, {RNA_NUM_TORSIONS}, 2], got {tuple(torsions.shape)}"
        )
    return torsions


def normalize_torsions(
    torsions: torch.Tensor, eps: float = DEGENERATE_EPS
) -> torch.Tensor:

    norm = torch.linalg.norm(torsions, dim=-1, keepdim=True)
    valid = norm > eps
    normalized = torsions / torch.where(valid, norm, torch.ones_like(norm))
    neutral = torch.cat([torch.zeros_like(norm), torch.ones_like(norm)], dim=-1)
    return torch.where(valid, normalized, neutral)


def _unit_or_fallback(
    v: torch.Tensor,
    fallback: torch.Tensor,
    eps: float,
    scale: torch.Tensor | None = None,
) -> torch.Tensor:

    norm = torch.linalg.norm(v, dim=-1, keepdim=True)
    threshold = eps if scale is None else eps * torch.clamp(scale, min=1.0)
    ok = norm > threshold
    unit = v / torch.where(ok, norm, torch.ones_like(norm))
    return torch.where(ok, unit, fallback)


def rotation_6d_to_matrix(x: torch.Tensor, eps: float = DEGENERATE_EPS) -> torch.Tensor:

    if x.shape[-1] < 6:
        raise ValueError(
            f"rotation_6d_to_matrix expects a trailing dim of at least 6, got {tuple(x.shape)}"
        )
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]

    default_b1 = torch.zeros_like(a1)
    default_b1[..., 0] = 1.0
    b1 = _unit_or_fallback(a1, default_b1, eps)

    proj = (b1 * a2).sum(dim=-1, keepdim=True)

    b2 = _unit_or_fallback(
        a2 - proj * b1,
        _orthogonal_vector(b1),
        eps,
        scale=torch.linalg.norm(a2, dim=-1, keepdim=True),
    )

    b2 = _unit_or_fallback(
        b2 - (b1 * b2).sum(dim=-1, keepdim=True) * b1,
        _orthogonal_vector(b1),
        eps,
    )
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def safe_normalize(x: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    return x / torch.linalg.norm(x, dim=-1, keepdim=True).clamp_min(eps)


def _orthogonal_vector(x: torch.Tensor) -> torch.Tensor:
    basis_x = x.new_tensor((1.0, 0.0, 0.0)).view(*((1,) * (x.ndim - 1)), 3)
    basis_y = x.new_tensor((0.0, 1.0, 0.0)).view(*((1,) * (x.ndim - 1)), 3)
    use_x = (x[..., 0].abs() < 0.9).unsqueeze(-1)
    ref = torch.where(use_x, basis_x, basis_y)
    return safe_normalize(torch.cross(x, ref, dim=-1))


def compute_rna_atom37_from_frames(
    trans: torch.Tensor,
    rotmats: torch.Tensor,
    torsions: torch.Tensor,
    is_na_residue_mask: torch.Tensor,
    aatype: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    torsions = validate_rna_torsions(normalize_torsions(torsions))
    output_na_mask = is_na_residue_mask.bool()
    compute_na_mask = (
        torch.ones_like(output_na_mask, dtype=torch.bool)
        if output_na_mask.ndim == 2
        else output_na_mask
    )
    atom37, atom_mask, _, _, _ = all_atom_na.compute_backbone(
        bb_rigids=all_atom_na.create_rna_rigid(rotmats, trans),
        torsions=torsions,
        is_na_residue_mask=compute_na_mask,
        aatype=aatype.long(),
    )
    atom37 = atom37 * output_na_mask.unsqueeze(-1).unsqueeze(-1)
    atom_mask = atom_mask * output_na_mask.unsqueeze(-1)
    return atom37, atom_mask.float()


def extract_c4_positions(
    bb_coarse: torch.Tensor, mode: str = "p_c4prime_n"
) -> torch.Tensor:
    mode = canonicalize_explicit_backbone_mode(mode)
    if mode == "c4prime":
        return bb_coarse[..., :3]
    if mode == "c4prime_p":
        return bb_coarse[..., :3]
    if mode in {"p_c4prime_n", "p_c4prime_c1prime"}:
        return bb_coarse[..., 3:6]
    raise ValueError(f"Unsupported explicit backbone mode: {mode}")


def extract_p_positions(
    bb_coarse: torch.Tensor, mode: str = "p_c4prime_n"
) -> torch.Tensor:
    mode = canonicalize_explicit_backbone_mode(mode)
    c4 = extract_c4_positions(bb_coarse, mode)
    if mode == "c4prime":
        return c4
    if mode == "c4prime_p":
        return bb_coarse[..., 3:6]
    if mode in {"p_c4prime_n", "p_c4prime_c1prime"}:
        return bb_coarse[..., :3]
    raise ValueError(f"Unsupported explicit backbone mode: {mode}")


def extract_n_positions(
    bb_coarse: torch.Tensor, mode: str = "p_c4prime_n"
) -> torch.Tensor:
    mode = canonicalize_explicit_backbone_mode(mode)
    c4 = extract_c4_positions(bb_coarse, mode)
    if mode == "p_c4prime_n":
        return bb_coarse[..., 6:9]
    return c4


def extract_c1_positions(
    bb_coarse: torch.Tensor, mode: str = "p_c4prime_n"
) -> torch.Tensor:
    mode = canonicalize_explicit_backbone_mode(mode)
    c4 = extract_c4_positions(bb_coarse, mode)
    if mode == "p_c4prime_c1prime":
        return bb_coarse[..., 6:9]
    return c4


def coarse_to_named_coords(
    bb_coarse: torch.Tensor,
    mode: str = "p_c4prime_n",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mode = canonicalize_explicit_backbone_mode(mode)
    c4 = extract_c4_positions(bb_coarse, mode)
    p = extract_p_positions(bb_coarse, mode)
    third = (
        extract_c1_positions(bb_coarse, mode)
        if mode == "p_c4prime_c1prime"
        else extract_n_positions(bb_coarse, mode)
    )
    return c4, p, third


def _shift_points(points: torch.Tensor, shift: int) -> torch.Tensor:
    if shift == 0:
        return points
    if shift > 0:
        pad = points[:, :1].expand(-1, shift, -1)
        return torch.cat([pad, points[:, :-shift]], dim=1)
    shift = -shift
    pad = points[:, -1:].expand(-1, shift, -1)
    return torch.cat([points[:, shift:], pad], dim=1)


def _shift_mask(mask: torch.Tensor, shift: int) -> torch.Tensor:
    if shift == 0:
        return mask
    if shift > 0:
        pad = mask.new_zeros(mask.shape[0], shift)
        return torch.cat([pad, mask[:, :-shift]], dim=1)
    shift = -shift
    pad = mask.new_zeros(mask.shape[0], shift)
    return torch.cat([mask[:, shift:], pad], dim=1)


def _chain_tangent(
    c4: torch.Tensor, res_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = ensure_float_mask(res_mask)
    prev_c4 = _shift_points(c4, 1)
    next_c4 = _shift_points(c4, -1)
    prev_mask = _shift_mask(mask, 1)
    next_mask = _shift_mask(mask, -1)

    fwd = next_c4 - c4
    back = c4 - prev_c4
    fwd = fwd * next_mask.unsqueeze(-1)
    back = back * prev_mask.unsqueeze(-1)

    tangent = fwd + back
    tangent_norm = torch.linalg.norm(tangent, dim=-1, keepdim=True)
    tangent = torch.where(tangent_norm > EPS, tangent, fwd)
    tangent_norm = torch.linalg.norm(tangent, dim=-1, keepdim=True)
    tangent = torch.where(tangent_norm > EPS, tangent, back)
    fallback = c4.new_tensor((1.0, 0.0, 0.0)).view(1, 1, 3)
    tangent = torch.where(
        torch.linalg.norm(tangent, dim=-1, keepdim=True) > EPS,
        tangent,
        fallback.expand_as(c4),
    )
    return tangent, back, fwd


def build_coarse_pseudo_frame(
    bb_coarse: torch.Tensor,
    res_mask: torch.Tensor,
    mode: str = "p_c4prime_n",
) -> tuple[torch.Tensor, torch.Tensor]:
    mode = canonicalize_explicit_backbone_mode(mode)
    mask = ensure_float_mask(res_mask)
    c4, p, n = coarse_to_named_coords(bb_coarse, mode)
    tangent, back, fwd = _chain_tangent(c4, mask)

    if mode == "c4prime":
        x_axis = safe_normalize(tangent)
        z_axis_raw = torch.cross(back, fwd, dim=-1)
        z_axis = torch.where(
            torch.linalg.norm(z_axis_raw, dim=-1, keepdim=True) > EPS,
            z_axis_raw,
            _orthogonal_vector(x_axis),
        )
        z_axis = safe_normalize(z_axis)
        y_axis = safe_normalize(torch.cross(z_axis, x_axis, dim=-1))
        z_axis = safe_normalize(torch.cross(x_axis, y_axis, dim=-1))
    elif mode == "c4prime_p":
        x_axis_raw = p - c4
        x_axis = torch.where(
            torch.linalg.norm(x_axis_raw, dim=-1, keepdim=True) > EPS,
            x_axis_raw,
            tangent,
        )
        x_axis = safe_normalize(x_axis)
        y_axis_raw = tangent - (tangent * x_axis).sum(dim=-1, keepdim=True) * x_axis
        y_axis = torch.where(
            torch.linalg.norm(y_axis_raw, dim=-1, keepdim=True) > EPS,
            y_axis_raw,
            _orthogonal_vector(x_axis),
        )
        y_axis = safe_normalize(y_axis)
        z_axis = safe_normalize(torch.cross(x_axis, y_axis, dim=-1))
        y_axis = safe_normalize(torch.cross(z_axis, x_axis, dim=-1))
    elif mode in {"p_c4prime_n", "p_c4prime_c1prime"}:
        x_axis_raw = n - c4
        x_axis = torch.where(
            torch.linalg.norm(x_axis_raw, dim=-1, keepdim=True) > EPS,
            x_axis_raw,
            tangent,
        )
        x_axis = safe_normalize(x_axis)
        y_seed = p - c4
        y_axis_raw = y_seed - (y_seed * x_axis).sum(dim=-1, keepdim=True) * x_axis
        y_axis = torch.where(
            torch.linalg.norm(y_axis_raw, dim=-1, keepdim=True) > EPS,
            y_axis_raw,
            tangent - (tangent * x_axis).sum(dim=-1, keepdim=True) * x_axis,
        )
        y_axis = torch.where(
            torch.linalg.norm(y_axis, dim=-1, keepdim=True) > EPS,
            y_axis,
            _orthogonal_vector(x_axis),
        )
        y_axis = safe_normalize(y_axis)
        z_axis = safe_normalize(torch.cross(x_axis, y_axis, dim=-1))
        y_axis = safe_normalize(torch.cross(z_axis, x_axis, dim=-1))
    else:
        raise ValueError(f"Unsupported explicit backbone mode: {mode}")

    rotmats = torch.stack([x_axis, y_axis, z_axis], dim=-1)
    eye = torch.eye(3, device=bb_coarse.device, dtype=bb_coarse.dtype).view(1, 1, 3, 3)
    rotmats = torch.where(mask[:, :, None, None].bool(), rotmats, eye)
    trans = c4 * mask.unsqueeze(-1)
    return trans, rotmats


def compose_rotations(
    base_rotmats: torch.Tensor, residual_rot6d: torch.Tensor
) -> torch.Tensor:
    residual_rotmats = rotation_6d_to_matrix(residual_rot6d)
    return base_rotmats @ residual_rotmats


def get_timestep_features(
    t: torch.Tensor, emb_dim: int, mask: torch.Tensor
) -> torch.Tensor:
    if t.ndim == 2:
        t = t[:, 0]
    emb = get_timestep_embedding(t, emb_dim)[:, None, :].repeat(1, mask.shape[1], 1)
    return emb * ensure_float_mask(mask).unsqueeze(-1)


def build_relpos_features(res_idx: torch.Tensor, relpos_dim: int) -> torch.Tensor:
    rel = res_idx[:, :, None] - res_idx[:, None, :]
    return get_index_embedding(rel, relpos_dim, max_len=4096)


def pairwise_distogram(
    x: torch.Tensor, min_bin: float, max_bin: float, num_bins: int
) -> torch.Tensor:
    return calc_distogram(
        x, min_bin=min_bin, max_bin=max_bin, num_bins=num_bins
    ).float()


def cross_distogram(
    x: torch.Tensor, y: torch.Tensor, min_bin: float, max_bin: float, num_bins: int
) -> torch.Tensor:

    dists = torch.linalg.norm(x[:, :, None, :] - y[:, None, :, :], dim=-1)
    return distance_to_distogram(dists, min_bin, max_bin, num_bins, dtype=x.dtype)


def build_coarse_distance_features(
    bb_coarse: torch.Tensor,
    dist_min: float,
    dist_max: float,
    dist_num_bins: int,
    mode: str = "p_c4prime_n",
) -> torch.Tensor:
    mode = canonicalize_explicit_backbone_mode(mode)
    bead_names = get_explicit_backbone_beads(mode)
    c4, p, n = coarse_to_named_coords(bb_coarse, mode)
    c1 = extract_c1_positions(bb_coarse, mode)
    bead_map = {"C4'": c4, "P": p, "N": n, "C1'": c1}
    features: list[torch.Tensor] = []
    for i, name_i in enumerate(bead_names):
        xi = bead_map[name_i]
        for j, name_j in enumerate(bead_names):
            xj = bead_map[name_j]
            if i == j:
                feat = pairwise_distogram(
                    xi, min_bin=dist_min, max_bin=dist_max, num_bins=dist_num_bins
                )
            else:
                feat = cross_distogram(
                    xi, xj, min_bin=dist_min, max_bin=dist_max, num_bins=dist_num_bins
                )
            features.append(feat)
    return torch.cat(features, dim=-1)


def build_coarse_pair_features(
    bb_coarse: torch.Tensor,
    res_idx: torch.Tensor,
    relpos_dim: int,
    dist_min: float,
    dist_max: float,
    dist_num_bins: int,
    mode: str = "p_c4prime_n",
) -> torch.Tensor:
    relpos = build_relpos_features(res_idx, relpos_dim)
    distance_feats = build_coarse_distance_features(
        bb_coarse,
        dist_min,
        dist_max,
        dist_num_bins,
        mode=mode,
    )
    return torch.cat([relpos, distance_feats], dim=-1)


def zero_condition(mask: torch.Tensor) -> torch.Tensor:
    return (
        ensure_float_mask(mask).unsqueeze(-1).new_zeros(mask.shape[0], mask.shape[1], 1)
    )
