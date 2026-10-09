


import dataclasses
from collections.abc import Mapping
from typing import Any

import numpy as np

from src.data.basics.constants import (
    complex_constants,
    nucleotide_constants,
    protein_constants,
)


FeatureDict = Mapping[str, np.ndarray]
ModelOutput = Mapping[str, Any]


PDB_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)


@dataclasses.dataclass(frozen=True)
class Complex:


    atom_positions: np.ndarray


    restype: np.ndarray


    atom_mask: np.ndarray


    residue_index: np.ndarray


    residue_molecule_type: np.ndarray


    chain_index: np.ndarray


    b_factors: np.ndarray

    def __post_init__(self):
        if len(np.unique(self.chain_index)) > PDB_MAX_CHAINS:
            raise ValueError(
                f"Cannot build an instance with more than {PDB_MAX_CHAINS} chains \
                because these cannot be written to PDB format."
            )


def complex_to_pdb(complex: Complex, model=1, add_end=True, molecule_type_to_write=-1) -> str:


    restype_name_to_full_atom_names = protein_constants.restype_name_to_full_atom_names.copy()
    restype_name_to_full_atom_names.update(
        nucleotide_constants.restype_name_to_full_atom_names.copy()
    )

    pdb_lines = []

    atom_positions = complex.atom_positions
    restype = complex.restype
    atom_mask = complex.atom_mask
    residue_index = complex.residue_index.astype(int)
    residue_molecule_type = complex.residue_molecule_type.astype(int)
    chain_index = complex.chain_index.astype(int)
    b_factors = complex.b_factors

    if np.any(restype > complex_constants.restype_num):
        raise ValueError("Invalid restypes.")


    chain_ids = {}
    for i in np.unique(chain_index):
        if i >= PDB_MAX_CHAINS:
            raise ValueError(f"The PDB format supports at most {PDB_MAX_CHAINS} chains.")
        chain_ids[i] = PDB_CHAIN_IDS[i]

    pdb_lines.append(f"MODEL     {model}")
    atom_index = 1
    last_chain_index = chain_index[0]
    res_name_3 = None

    for i in range(restype.shape[0]):
        current_chain_index = chain_index[i]


        res_is_gap_token = restype[i] == complex_constants.gap_token
        res_is_mask_token = (
            restype[i] == complex_constants.unknown_protein_token
            or restype[i] == complex_constants.unknown_nucleic_token
        )
        res_is_of_ignored_molecule_type = (
            residue_molecule_type[i, molecule_type_to_write] == 0
            if molecule_type_to_write in [0, 1]
            else False
        )
        if res_is_gap_token or res_is_mask_token or res_is_of_ignored_molecule_type:
            continue

        res_name_3 = complex_constants.restypes_3[restype[i]]


        if last_chain_index != chain_index[i]:
            chain_end = "TER"
            end_resname = res_name_3
            end_chain_id = chain_ids[last_chain_index]
            end_res_idx = residue_index[i] - 1
            chain_end_line = (
                f"{chain_end:<6}{atom_index!s:>5}      {end_resname:>3} "
                f"{end_chain_id:>1}{end_res_idx!s:>4}"
            )
            pdb_lines.append(chain_end_line)
            atom_index += 1

        atom_types = restype_name_to_full_atom_names[res_name_3]
        for atom_name, pos, mask, b_factor in zip(
            atom_types, atom_positions[i], atom_mask[i], b_factors[i]
        ):
            if mask < 0.5:
                continue

            record_type = "ATOM"
            name = atom_name if len(atom_name) == 4 else f" {atom_name}"
            alt_loc = ""
            insertion_code = ""
            occupancy = 1.00
            element = atom_name[0]
            charge = ""

            atom_line = (
                f"{record_type:<6}{atom_index!s:>5} {name:<4}{alt_loc:>1}"
                f"{res_name_3:>3} {chain_ids[current_chain_index]:>1}"
                f"{residue_index[i]!s:>4}{insertion_code:>1}   "
                f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
                f"{occupancy:>6.2f}{b_factor:>6.2f}          "
                f"{element:>2}{charge:>2}"
            )
            pdb_lines.append(atom_line)
            atom_index += 1

        last_chain_index = chain_index[i]


    if res_name_3 is None:

        return ""
    chain_end = "TER"
    end_resname = res_name_3
    end_chain_id = chain_ids[last_chain_index]
    end_res_idx = residue_index[i]
    chain_end_line = (
        f"{chain_end:<6}{atom_index!s:>5}      {end_resname:>3} "
        f"{end_chain_id:>1}{end_res_idx!s:>4}"
    )
    pdb_lines.append(chain_end_line)
    pdb_lines.append("ENDMDL")
    if add_end:
        pdb_lines.append("END")


    pdb_lines = [line.ljust(80) for line in pdb_lines]
    return "\n".join(pdb_lines) + "\n"
