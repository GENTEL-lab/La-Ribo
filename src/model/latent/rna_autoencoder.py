from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.model.latent.checkpoint_utils import RNA_GEOMETRY_VERSION_CODE
from src.model.latent.common import (
    RNA_NUM_TORSIONS,
    build_coarse_pair_features,
    build_coarse_pseudo_frame,
    compose_rotations,
    compute_rna_atom37_from_frames,
    ensure_float_mask,
    get_coarse_backbone_dim,
    get_coarse_pair_feature_dim,
    get_explicit_backbone_mode,
    normalize_torsions,
    zero_condition,
)
from src.model.latent.transformer import PairBiasedTransformer


class RNADecoder(nn.Module):
    def __init__(self, ae_conf: Any):
        super().__init__()
        self.num_nt_types = int(ae_conf.num_nt_types)
        self.num_torsions = int(ae_conf.num_torsions)
        self.latent_dim = int(ae_conf.latent_dim)
        self.coarse_mode = get_explicit_backbone_mode(ae_conf)
        self.coarse_dim = get_coarse_backbone_dim(self.coarse_mode)
        dec_conf = ae_conf.decoder_conf
        self.relpos_dim = int(dec_conf.relpos_dim)
        self.dist_num_bins = int(dec_conf.dist_num_bins)
        self.dist_min = float(dec_conf.dist_min)
        self.dist_max = float(dec_conf.dist_max)
        token_input_dim = self.latent_dim + self.coarse_dim
        pair_input_dim = self.relpos_dim + get_coarse_pair_feature_dim(
            self.coarse_mode,
            self.dist_num_bins,
        )
        self.trunk = PairBiasedTransformer(
            token_input_dim=token_input_dim,
            pair_input_dim=pair_input_dim,
            cond_input_dim=1,
            token_dim=int(dec_conf.token_dim),
            pair_dim=int(dec_conf.pair_dim),
            cond_dim=int(dec_conf.cond_dim),
            num_layers=int(dec_conf.num_layers),
            num_heads=int(dec_conf.num_heads),
            use_qkln=bool(getattr(dec_conf, "use_qkln", True)),
        )
        hidden_dim = int(dec_conf.token_dim)
        self.seq_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, self.num_nt_types)
        )
        self.frame_rot6d_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 6)
        )
        self.torsion_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, self.num_torsions * 2)
        )

    def forward_heads(
        self,
        *,
        z_latent: torch.Tensor,
        bb_coarse: torch.Tensor,
        res_mask: torch.Tensor,
        res_idx: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        res_mask = ensure_float_mask(res_mask)
        token_inputs = torch.cat([z_latent, bb_coarse], dim=-1) * res_mask.unsqueeze(-1)
        pair_inputs = build_coarse_pair_features(
            bb_coarse=bb_coarse,
            res_idx=res_idx,
            relpos_dim=self.relpos_dim,
            dist_min=self.dist_min,
            dist_max=self.dist_max,
            dist_num_bins=self.dist_num_bins,
            mode=self.coarse_mode,
        )
        token_repr = self.trunk(
            token_inputs, pair_inputs, zero_condition(res_mask), res_mask
        )
        seq_logits = self.seq_head(token_repr) * res_mask.unsqueeze(-1)
        pred_aatypes = torch.argmax(seq_logits, dim=-1)

        pred_trans, pseudo_rotmats = build_coarse_pseudo_frame(
            bb_coarse,
            res_mask,
            mode=self.coarse_mode,
        )
        pred_rotmats = compose_rotations(
            pseudo_rotmats, self.frame_rot6d_head(token_repr)
        )
        eye = torch.eye(3, device=pred_rotmats.device, dtype=pred_rotmats.dtype).view(
            1, 1, 3, 3
        )
        pred_rotmats = torch.where(res_mask[:, :, None, None].bool(), pred_rotmats, eye)
        pred_torsions = self.torsion_head(token_repr).reshape(
            token_repr.shape[0], token_repr.shape[1], self.num_torsions, 2
        )
        pred_torsions = normalize_torsions(pred_torsions) * res_mask.unsqueeze(
            -1
        ).unsqueeze(-1)
        return {
            "token_repr": token_repr,
            "seq_logits": seq_logits,
            "pred_aatypes": pred_aatypes,
            "pred_trans": pred_trans,
            "pseudo_rotmats": pseudo_rotmats,
            "pred_rotmats": pred_rotmats,
            "pred_torsions": pred_torsions,
        }

    def expand_atoms(
        self,
        decoder_heads: dict[str, torch.Tensor],
        *,
        is_na_residue_mask: torch.Tensor,
        aatype_for_atoms: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        atom_aatype = (
            decoder_heads["pred_aatypes"]
            if aatype_for_atoms is None
            else aatype_for_atoms.long()
        )
        pred_atom37, atom_mask = compute_rna_atom37_from_frames(
            trans=decoder_heads["pred_trans"],
            rotmats=decoder_heads["pred_rotmats"],
            torsions=decoder_heads["pred_torsions"],
            is_na_residue_mask=is_na_residue_mask,
            aatype=atom_aatype,
        )
        return {
            "atom_aatype": atom_aatype,
            **decoder_heads,
            "pred_atom37": pred_atom37,
            "atom_mask": atom_mask,
        }

    def forward(
        self,
        *,
        z_latent: torch.Tensor,
        bb_coarse: torch.Tensor,
        res_mask: torch.Tensor,
        res_idx: torch.Tensor,
        is_na_residue_mask: torch.Tensor,
        aatype_for_atoms: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        decoder_heads = self.forward_heads(
            z_latent=z_latent,
            bb_coarse=bb_coarse,
            res_mask=res_mask,
            res_idx=res_idx,
        )
        return self.expand_atoms(
            decoder_heads,
            is_na_residue_mask=is_na_residue_mask,
            aatype_for_atoms=aatype_for_atoms,
        )


class RNAPartialAutoencoder(nn.Module):
    def __init__(self, ae_conf: Any):
        super().__init__()
        self.register_buffer(
            "rna_geometry_version_code",
            torch.tensor(RNA_GEOMETRY_VERSION_CODE, dtype=torch.int32),
        )
        self.num_nt_types = int(ae_conf.num_nt_types)
        self.num_torsions = int(ae_conf.num_torsions)
        if self.num_torsions != RNA_NUM_TORSIONS:
            raise ValueError(f"RNA decoder requires {RNA_NUM_TORSIONS} torsions")
        self.latent_dim = int(ae_conf.latent_dim)
        self.decoder = RNADecoder(ae_conf)

    def decode(
        self,
        *,
        z_latent: torch.Tensor,
        bb_coarse: torch.Tensor,
        res_mask: torch.Tensor,
        res_idx: torch.Tensor,
        is_na_residue_mask: torch.Tensor,
        aatype_for_atoms: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.decoder(
            z_latent=z_latent,
            bb_coarse=bb_coarse,
            res_mask=res_mask,
            res_idx=res_idx,
            is_na_residue_mask=is_na_residue_mask,
            aatype_for_atoms=aatype_for_atoms,
        )
