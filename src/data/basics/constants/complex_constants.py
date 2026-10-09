


NUM_PROT_TORSIONS = 1
NUM_NA_TORSIONS = 10
NUM_PROT_NA_TORSIONS = 10
PYRIMIDINE_RESIDUE_TOKENS = [22, 24, 26, 28]


restypes = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
            "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y",
            "X", "da", "dc", "dg", "dt", "a", "c", "g",
            "u", "x", "-", "_", "1", "2", "3", "4", "5"]


restype_order = {restype: i for i, restype in enumerate(restypes)}
restype_num = len(restypes)

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
    "da": "DA",
    "dc": "DC",
    "dg": "DG",
    "dt": "DT",
    "a": "A",
    "c": "C",
    "g": "G",
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
restype_3to1 = {v: k for k, v in restype_1to3.items()}
restypes_3 = list(restype_3to1.keys())
restypes_1 = list(restype_1to3.keys())


protein_restypes = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
                    "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]


nucleic_restypes = ["DA", "DC", "DG", "DT", "A", "C", "G", "U"]
special_restypes = ["-", "_", "1", "2", "3", "4", "5"]

unknown_protein_restype = "X"
unknown_nucleic_restype = "x"
gap_token = restypes.index("-")
pad_token = restypes.index("_")
unknown_protein_token = restypes.index(unknown_protein_restype)
unknown_nucleic_token = restypes.index(unknown_nucleic_restype)

protein_restype_num = len(protein_restypes + [unknown_protein_restype])
na_restype_num = len(nucleic_restypes + [unknown_nucleic_restype])

protein_na_restype_num = len(
    protein_restypes
    + [unknown_protein_restype]
    + nucleic_restypes
    + [unknown_nucleic_restype]
)

default_protein_restype = restypes.index("A")
default_na_restype = restypes.index("da")

alternative_restypes_map = {

    "MSE": "MET",
}
allowable_restypes = set(restypes + list(alternative_restypes_map.keys()))
