


restypes = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M",
            "N", "P", "Q", "R", "S", "T", "V", "W", "Y", "X", "a",
            "c", "g", "t", "u", "x", "-", "_", "1", "2", "3", "4", "5"]


restype_order = {
    "A": 0,
    "C": 1,
    "D": 2,
    "E": 3,
    "F": 4,
    "G": 5,
    "H": 6,
    "I": 7,
    "K": 8,
    "L": 9,
    "M": 10,
    "N": 11,
    "P": 12,
    "Q": 13,
    "R": 14,
    "S": 15,
    "T": 16,
    "V": 17,
    "W": 18,
    "Y": 19,
    "X": 20,
    "a": 21,
    "c": 22,
    "g": 23,
    "t": 24,
    "u": 25,
    "x": 26,
    "-": 27,
    "_": 28,
    "1": 29,
    "2": 30,
    "3": 31,
    "4": 32,
    "5": 33,
}


restype_1to3 = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
    "X": "UNK",
    "a": "A",
    "c": "C",
    "g": "G",
    "t": "T",
    "u": "U",
    "x": "unk",
    "-": "GAP",
    "_": "PAD",
    "1": "SP1",
    "2": "SP2",
    "3": "SP3",
    "4": "SP4",
    "5": "SP5",
}

restype_3to1 = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
    "UNK": "X",
    "A": "a",
    "C": "c",
    "G": "g",
    "T": "t",
    "U": "u",
    "unk": "x",
    "GAP": "-",
    "PAD": "_",
    "SP1": "1",
    "SP2": "2",
    "SP3": "3",
    "SP4": "4",
    "SP5": "5",
}

restype_num = len(restypes)

alternative_restypes_map = {

    "MSE": "MET",
}

allowable_restypes = set(restypes + list(alternative_restypes_map.keys()))


def restype_to_str_sequence(butype):
    return "".join([restypes[butype[i]] for i in range(len(butype))])


protein_restypes = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
                    "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]

protein_restypes_with_x = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
                           "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y", "X"]


protein_resnames = [
    "ALA",
    "CYS",
    "ASP",
    "GLU",
    "PHE",
    "GLY",
    "HIS",
    "ILE",
    "LYS",
    "LEU",
    "MET",
    "ASN",
    "PRO",
    "GLN",
    "ARG",
    "SER",
    "THR",
    "VAL",
    "TRP",
    "TYR",
]

protein_restype_order = {
    "A": 0,
    "C": 1,
    "D": 2,
    "E": 3,
    "F": 4,
    "G": 5,
    "H": 6,
    "I": 7,
    "K": 8,
    "L": 9,
    "M": 10,
    "N": 11,
    "P": 12,
    "Q": 13,
    "R": 14,
    "S": 15,
    "T": 16,
    "V": 17,
    "W": 18,
    "Y": 19,
}

protein_restype_order_with_x = {
    "A": 0,
    "C": 1,
    "D": 2,
    "E": 3,
    "F": 4,
    "G": 5,
    "H": 6,
    "I": 7,
    "K": 8,
    "L": 9,
    "M": 10,
    "N": 11,
    "P": 12,
    "Q": 13,
    "R": 14,
    "S": 15,
    "T": 16,
    "V": 17,
    "W": 18,
    "Y": 19,
    "X": 20,
}

protein_resname_to_idx = {
    "ALA": 0,
    "CYS": 1,
    "ASP": 2,
    "GLU": 3,
    "PHE": 4,
    "GLY": 5,
    "HIS": 6,
    "ILE": 7,
    "LYS": 8,
    "LEU": 9,
    "MET": 10,
    "ASN": 11,
    "PRO": 12,
    "GLN": 13,
    "ARG": 14,
    "SER": 15,
    "THR": 16,
    "VAL": 17,
    "TRP": 18,
    "TYR": 19,
}

protein_restype_1to3 = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
}

protein_restype_3to1 = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}


protein_restype_num = len(protein_restypes)
protein_restype_num_with_x = len(protein_restypes_with_x)


nucleic_restypes = ["a", "c", "g", "t", "u"]
special_restypes = ["-", "_", "1", "2", "3", "4", "5"]
unknown_protein_restype = "X"
unknown_nucleic_restype = "x"
gap_token = restypes.index("-")
pad_token = restypes.index("_")


def is_protein_sequence(sequence: str) -> bool:


    return all(s in protein_restypes for s in sequence)


def is_nucleic_sequence(sequence: str) -> bool:


    return all(s in nucleic_restypes for s in sequence)
