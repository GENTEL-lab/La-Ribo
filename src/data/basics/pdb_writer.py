

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.basics import complex as complex_mod
from src.data.basics.constants import complex_constants


RNA_OFFSET = complex_constants.restype_order["a"]
COMPACT_RNA_TO_COMPLEX_RESTYPE = np.array(
    [
        complex_constants.restype_order["a"],
        complex_constants.restype_order["c"],
        complex_constants.restype_order["g"],
        complex_constants.restype_order["u"],
        complex_constants.restype_order["x"],
    ],
    dtype=np.int32,
)


def _normalize_restype(restype: np.ndarray | None, num_res: int) -> np.ndarray:
    if restype is None:
        return np.full(num_res, RNA_OFFSET, dtype=np.int32)
    restype = np.asarray(restype).squeeze()
    if restype.ndim == 0:
        restype = np.full(num_res, int(restype), dtype=np.int32)
    if restype.shape[-1] != num_res:
        if restype.ndim >= 2 and restype.shape[-2] == num_res:
            restype = restype[..., 0]
        else:
            raise ValueError(
                f"Cannot align restype of shape {restype.shape} with {num_res} residues"
            )
    restype = restype.astype(np.int32)
    if restype.min() >= 0 and restype.max() <= 4:
        restype = COMPACT_RNA_TO_COMPLEX_RESTYPE[restype]
    return restype


def create_full_complex(
    atom37: np.ndarray,
    atom37_mask: np.ndarray,
    restype: np.ndarray | None = None,
    chain_index: np.ndarray | None = None,
    b_factors: np.ndarray | None = None,
    is_na_residue_mask: np.ndarray | None = None,
) -> complex_mod.Complex:


    if atom37.ndim != 3 or atom37.shape[-2:] != (37, 3):
        raise ValueError(f"Expected atom37 with shape [N, 37, 3], got {tuple(atom37.shape)}")
    n = atom37.shape[0]

    restype = _normalize_restype(restype, n)
    residue_index = np.arange(1, n + 1, dtype=np.int32)

    if chain_index is None:
        chain_index = np.zeros(n, dtype=np.int32)
    else:
        chain_index = np.asarray(chain_index, dtype=np.int32)

    if b_factors is None:
        b_factors = np.zeros([n, 37], dtype=np.float32)
    else:
        b_factors = np.asarray(b_factors, dtype=np.float32)

    residue_molecule_type = np.zeros([n, 2], dtype=np.int64)
    residue_molecule_type[:, 1] = 1
    if is_na_residue_mask is not None:
        is_na_residue_mask = np.asarray(is_na_residue_mask).astype(bool).reshape(-1)
        if is_na_residue_mask.shape[0] != n:
            raise ValueError(
                f"is_na_residue_mask shape {is_na_residue_mask.shape} does not match num_res {n}"
            )
        residue_molecule_type[:, 1] = is_na_residue_mask.astype(np.int64)

    return complex_mod.Complex(
        atom_positions=np.asarray(atom37, dtype=np.float32),
        restype=restype,
        atom_mask=np.asarray(atom37_mask, dtype=np.float32),
        residue_index=residue_index,
        chain_index=chain_index,
        b_factors=b_factors,
        residue_molecule_type=residue_molecule_type,
    )


def write_complex_to_pdbs(
    complex_pos: np.ndarray,
    output_filepath: str | Path,
    restype: np.ndarray | None,
    chain_index: np.ndarray | None = None,
    b_factors: np.ndarray | None = None,
    is_na_residue_mask: np.ndarray | None = None,
    *,
    atom37_mask: np.ndarray | None = None,
) -> str:


    if atom37_mask is None:
        raise ValueError("write_complex_to_pdbs requires atom37_mask")
    complex_pos = np.asarray(complex_pos)
    output_filepath = Path(output_filepath)
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    if output_filepath.suffix != ".pdb":
        output_filepath = output_filepath.with_suffix(".pdb")

    if complex_pos.ndim == 3:
        complex_pos = complex_pos[None, ...]
    if complex_pos.ndim != 4:
        raise ValueError(
            f"Expected complex positions with shape [T, N, 37, 3] or [N, 37, 3], got {tuple(complex_pos.shape)}"
        )
    atom37_mask = np.asarray(atom37_mask, dtype=np.float32)
    if atom37_mask.ndim == 2:
        atom37_mask = atom37_mask[None, ...]
    if atom37_mask.shape != complex_pos.shape[:-1]:
        raise ValueError(
            f"atom37_mask shape {tuple(atom37_mask.shape)} does not match "
            f"complex positions {tuple(complex_pos.shape[:-1])}"
        )

    pdb_chunks: list[str] = []
    for t, pos37 in enumerate(complex_pos, start=1):
        effective_atom37_mask = atom37_mask[t - 1] > 0.5
        if restype is not None:
            restype_arr = np.asarray(restype)
            effective_restype = restype_arr[t - 1] if restype_arr.ndim == 2 else restype_arr
            if np.asarray(effective_restype).ndim != 1:
                raise ValueError(
                    "When writing multiple complexes to PDB files, only a single sequence may be provided for each complex."
                )
        else:
            effective_restype = None
        full_complex = create_full_complex(
            atom37=pos37,
            atom37_mask=effective_atom37_mask,
            restype=effective_restype,
            chain_index=chain_index,
            b_factors=b_factors,
            is_na_residue_mask=is_na_residue_mask,
        )
        pdb_chunks.append(
            complex_mod.complex_to_pdb(
                full_complex, model=t, add_end=False, molecule_type_to_write=1
            )
        )

    output_filepath.write_text("".join(pdb_chunks) + "END\n")
    return str(output_filepath)
