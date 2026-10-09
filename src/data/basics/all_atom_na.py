import torch

from src.data.basics import rigid_utils_na as ru
from src.data.basics.constants import nucleotide_constants, vocabulary


IDEALIZED_NA_ATOM_POS23 = torch.tensor(
    nucleotide_constants.nttype_compact_atom_rigid_group_positions
)
IDEALIZED_NA_ATOM_POS27 = torch.tensor(
    nucleotide_constants.nttype_atom27_rigid_group_positions
)
DEFAULT_NA_RESIDUE_FRAMES = torch.tensor(
    nucleotide_constants.nttype_rigid_group_default_frame
)
NA_RESIDUE_ATOM23_MASK = torch.tensor(nucleotide_constants.nttype_compact_atom_mask)
NA_RESIDUE_GROUP_IDX = torch.tensor(
    nucleotide_constants.nttype_compact_atom_to_rigid_group
)
RNA_TORSION_ATOM37_INDICES = torch.tensor(
    nucleotide_constants.RNA_TORSION_ATOM37_INDICES,
    dtype=torch.long,
)
COMPACT_RNA_TO_NT_INDEX = torch.tensor(
    [
        nucleotide_constants.restype_order["A"],
        nucleotide_constants.restype_order["C"],
        nucleotide_constants.restype_order["G"],
        nucleotide_constants.restype_order["U"],
    ]
)
NA_ATOM23_TO_ATOM37 = torch.tensor(
    [
        [
            nucleotide_constants.atom_order[atom_name] if atom_name else -1
            for atom_name in nucleotide_constants.restype_name_to_compact_atom_names[
                restype
            ]
        ]
        for restype in nucleotide_constants.restypes
    ],
    dtype=torch.long,
)

IDX_2_NATYPE = {0: "A", 1: "C", 2: "G", 3: "U"}
FULL_RNA_TORSIONS = nucleotide_constants.NUM_NA_TORSIONS
ATOM37_WIDTH = 37
RNA_C1P_INDEX = nucleotide_constants.atom_order["C1'"]
RNA_C2P_INDEX = nucleotide_constants.atom_order["C2'"]
RNA_C3P_INDEX = nucleotide_constants.atom_order["C3'"]
RNA_C4P_INDEX = nucleotide_constants.atom_order["C4'"]
RNA_O4P_INDEX = nucleotide_constants.atom_order["O4'"]
RNA_N1_INDEX = nucleotide_constants.atom_order["N1"]
RNA_N9_INDEX = nucleotide_constants.atom_order["N9"]
TM_RIGID_GROUP_INDEX = 9
CHI_RIGID_GROUP_INDEX = 10


def _to_compact_rna_aatype(aatype: torch.Tensor) -> torch.Tensor:

    aatype = aatype.long()
    offset = vocabulary.protein_restype_num + 1
    nt_indices = [nucleotide_constants.restype_order[n] for n in IDX_2_NATYPE.values()]

    compact = torch.zeros_like(aatype)
    recognized = torch.zeros_like(aatype, dtype=torch.bool)

    is_compact = (aatype >= 0) & (aatype <= 3)
    compact = torch.where(is_compact, aatype, compact)
    recognized |= is_compact

    for compact_idx, residue_name in IDX_2_NATYPE.items():
        target = torch.full_like(compact, compact_idx)

        is_nt = aatype == nucleotide_constants.restype_order[residue_name]

        is_offset = aatype == nucleotide_constants.restype_order[residue_name] + offset
        for match in (is_nt, is_offset):
            compact = torch.where(match, target, compact)
            recognized |= match

    if not bool(recognized.all()):
        unknown = torch.unique(aatype[~recognized]).tolist()
        raise ValueError(
            f"Unsupported RNA residue ids {unknown}; expected compact 0-3, "
            f"nucleotide_constants RNA order {nt_indices}, or protein-offset "
            f"{[idx + offset for idx in nt_indices]}"
        )
    return compact


def _compact_to_nucleotide_constants_aatype(
    compact_aatype: torch.Tensor,
) -> torch.Tensor:
    mapping = COMPACT_RNA_TO_NT_INDEX.to(device=compact_aatype.device, dtype=torch.long)
    return mapping[compact_aatype.long().clamp(min=0, max=mapping.shape[0] - 1)]


def create_rna_rigid(rots, trans):
    return ru.Rigid(rots=ru.Rotation(rot_mats=rots), trans=trans)


def of_na_torsion_angles_to_frames(r, alpha, aatype, rrgdf):

    default_4x4 = rrgdf[aatype, ...]

    default_r = r.from_tensor_4x4(default_4x4)

    bb_rot1 = alpha.new_zeros((*((1,) * len(alpha.shape[:-1])), 2))
    bb_rot1[..., 1] = 1

    alpha = torch.cat(
        [
            bb_rot1.expand(*alpha.shape[:-2], -1, -1),
            alpha,
        ],
        dim=-2,
    )

    all_rots = alpha.new_zeros(default_r.get_rots().get_rot_mats().shape)
    all_rots[..., 0, 0] = 1
    all_rots[..., 1, 1] = alpha[..., 1]
    all_rots[..., 1, 2] = -alpha[..., 0]
    all_rots[..., 2, 1] = alpha[..., 0]
    all_rots[..., 2, 2] = alpha[..., 1]

    all_rots = ru.Rigid(ru.Rotation(rot_mats=all_rots), None)

    all_frames = default_r.compose(all_rots)

    backbone2_atom1_frame = all_frames[..., 1]
    backbone2_atom2_frame = all_frames[..., 2]
    backbone2_atom3_frame = all_frames[..., 3]
    delta_frame_to_frame = all_frames[..., 4]
    gamma_frame_to_frame = all_frames[..., 5]
    beta_frame_to_frame = all_frames[..., 6]
    alpha1_frame_to_frame = all_frames[..., 7]
    alpha2_frame_to_frame = all_frames[..., 8]

    backbone2_atom1_frame_to_bb = backbone2_atom1_frame
    backbone2_atom2_frame_to_bb = backbone2_atom2_frame

    backbone2_atom3_frame_to_bb = backbone2_atom2_frame.compose(backbone2_atom3_frame)
    delta_frame_to_bb = delta_frame_to_frame
    gamma_frame_to_bb = gamma_frame_to_frame
    beta_frame_to_bb = gamma_frame_to_bb.compose(beta_frame_to_frame)
    alpha1_frame_to_bb = beta_frame_to_bb.compose(alpha1_frame_to_frame)
    alpha2_frame_to_bb = beta_frame_to_bb.compose(alpha2_frame_to_frame)

    ideal_atom_pos27 = IDEALIZED_NA_ATOM_POS27.to(
        device=alpha.device,
        dtype=alpha.dtype,
    )[aatype]
    c2 = backbone2_atom1_frame_to_bb.apply(ideal_atom_pos27[..., RNA_C2P_INDEX, :])
    c1 = backbone2_atom2_frame_to_bb.apply(ideal_atom_pos27[..., RNA_C1P_INDEX, :])
    is_purine = (aatype == nucleotide_constants.restype_order["A"]) | (
        aatype == nucleotide_constants.restype_order["G"]
    )
    glycosidic_n_local = torch.where(
        is_purine.unsqueeze(-1),
        ideal_atom_pos27[..., RNA_N9_INDEX, :],
        ideal_atom_pos27[..., RNA_N1_INDEX, :],
    )
    glycosidic_n = backbone2_atom3_frame_to_bb.apply(glycosidic_n_local)

    tm_frame_to_bb = ru.Rigid.from_3_points(c1, c2, glycosidic_n).compose(
        all_rots[..., TM_RIGID_GROUP_INDEX]
    )
    chi_frame_to_bb = ru.Rigid.from_3_points(c1, glycosidic_n, c2).compose(
        all_rots[..., CHI_RIGID_GROUP_INDEX]
    )

    all_frames_to_bb = ru.Rigid.cat(
        [
            all_frames[..., 0].unsqueeze(-1),
            backbone2_atom1_frame_to_bb.unsqueeze(-1),
            backbone2_atom2_frame_to_bb.unsqueeze(-1),
            backbone2_atom3_frame_to_bb.unsqueeze(-1),
            delta_frame_to_bb.unsqueeze(-1),
            gamma_frame_to_bb.unsqueeze(-1),
            beta_frame_to_bb.unsqueeze(-1),
            alpha1_frame_to_bb.unsqueeze(-1),
            alpha2_frame_to_bb.unsqueeze(-1),
            tm_frame_to_bb.unsqueeze(-1),
            chi_frame_to_bb.unsqueeze(-1),
        ],
        dim=-1,
    )

    all_frames_to_global = r[..., None].compose(all_frames_to_bb)
    return all_frames_to_global


def na_frames_to_atom23_pos(r, aatype):

    group_mask = NA_RESIDUE_GROUP_IDX.to(r.device)[aatype, ...]

    group_mask = torch.nn.functional.one_hot(
        group_mask,
        num_classes=DEFAULT_NA_RESIDUE_FRAMES.shape[-3],
    ).to(r.device)

    t_atoms_to_global = r[..., None, :] * group_mask

    t_atoms_to_global = t_atoms_to_global.map_tensor_fn(lambda x: torch.sum(x, dim=-1))

    frame_atom_mask = NA_RESIDUE_ATOM23_MASK.to(r.device)[aatype, ...].unsqueeze(-1)

    frame_null_pos = IDEALIZED_NA_ATOM_POS23.to(r.device)[aatype, ...]
    pred_positions = t_atoms_to_global.apply(frame_null_pos)
    pred_positions = pred_positions * frame_atom_mask

    return pred_positions


def _scatter_atom23_to_atom37(
    atom23_pos: torch.Tensor,
    atom23_to_atom37: torch.Tensor,
    is_na_residue_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid_atom = (
        (atom23_to_atom37 >= 0)
        & (atom23_to_atom37 < ATOM37_WIDTH)
        & is_na_residue_mask.unsqueeze(-1)
    )
    scatter_target = torch.where(
        valid_atom,
        atom23_to_atom37,
        torch.full_like(atom23_to_atom37, ATOM37_WIDTH),
    )
    pos_buffer = atom23_pos.new_zeros(*atom23_pos.shape[:-2], ATOM37_WIDTH + 1, 3)
    mask_buffer = torch.zeros(
        *atom23_pos.shape[:-2],
        ATOM37_WIDTH + 1,
        device=atom23_pos.device,
        dtype=torch.bool,
    )
    pos_buffer.scatter_(
        dim=-2,
        index=scatter_target.unsqueeze(-1).expand(*scatter_target.shape, 3),
        src=atom23_pos,
    )
    mask_buffer.scatter_(dim=-1, index=scatter_target, src=valid_atom)
    return pos_buffer[..., :ATOM37_WIDTH, :], mask_buffer[..., :ATOM37_WIDTH]


def compute_backbone(bb_rigids, torsions, is_na_residue_mask, aatype=None):

    if not is_na_residue_mask.any().item():
        raise ValueError(
            "No NA residues in mask — compute_backbone requires nucleic acid inputs."
        )

    if torsions.ndim != 4 or torsions.shape[-2:] != (FULL_RNA_TORSIONS, 2):
        raise ValueError(
            f"Expected torsions with shape [B, N, {FULL_RNA_TORSIONS}, 2], "
            f"got {tuple(torsions.shape)}"
        )
    torsion_angles = torsions.reshape(torsions.shape[0], -1, FULL_RNA_TORSIONS, 2)

    rot_dim = (4,) if bb_rigids._rots._quats is not None else (3, 3)

    aatype = (
        aatype
        if aatype is not None
        else torch.zeros(bb_rigids.shape, device=bb_rigids.device, dtype=torch.long)
    )

    na_bb_rigids = bb_rigids[is_na_residue_mask].reshape(
        new_rots_shape=torch.Size((bb_rigids.shape[0], -1, *rot_dim)),
        new_trans_shape=torch.Size((bb_rigids.shape[0], -1, 3)),
    )

    na_torsion_angles = torsion_angles.view(
        torsion_angles.shape[0], -1, FULL_RNA_TORSIONS, 2
    )

    na_aatype = aatype[is_na_residue_mask].view(aatype.shape[0], -1).long()
    compact_na_aatype = _to_compact_rna_aatype(na_aatype)
    effective_na_aatype = _compact_to_nucleotide_constants_aatype(compact_na_aatype)

    all_na_frames = of_na_torsion_angles_to_frames(
        r=na_bb_rigids,
        alpha=na_torsion_angles,
        aatype=effective_na_aatype,
        rrgdf=DEFAULT_NA_RESIDUE_FRAMES.to(bb_rigids.device),
    )
    atom23_pos = na_frames_to_atom23_pos(all_na_frames, effective_na_aatype)

    atom23_to_atom37 = NA_ATOM23_TO_ATOM37.to(bb_rigids.device)[effective_na_aatype]
    atom37_bb_pos, atom37_bb_supervised_mask = _scatter_atom23_to_atom37(
        atom23_pos,
        atom23_to_atom37,
        is_na_residue_mask,
    )

    atom37_mask = atom37_bb_supervised_mask

    return atom37_bb_pos, atom37_mask, atom37_bb_supervised_mask, aatype, atom23_pos
