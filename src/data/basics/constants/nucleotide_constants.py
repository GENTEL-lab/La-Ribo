

import collections
import functools
from collections.abc import Mapping
from importlib import resources

import numpy as np


NA_AATYPE6_MASK_RESIDUE_INDEX = 5
NA_AATYPE9_MASK_RESIDUE_INDEX = 8
NA_ATOM37_N1_ATOM_INDEX = 12
NA_ATOM37_N9_ATOM_INDEX = 18
NA_SUPERVISED_ATOM_N9_ATOM_INDEX = 11

NUM_BACKBONE2_CHI_ANGLES = 3
NUM_NA_TORSIONS = 10
RNA_GEOMETRY_VERSION = "rna_geometry_v2"


c4_c4 = 6.12


chi_angles_atoms = {
    "A": [
        ["C5'", "C4'", "C3'", "C2'"],
        ["C3'", "C4'", "O4'", "C1'"],
        ["C4'", "O4'", "C1'", "N9"],
        ["C5'", "C4'", "C3'", "O3'"],
        ["C3'", "C4'", "C5'", "O5'"],
        ["C4'", "C5'", "O5'", "P"],
        ["C5'", "O5'", "P", "OP1"],
        ["C5'", "O5'", "P", "OP2"],
        ["N9", "C1'", "C2'", "O2'"],
        ["C2'", "C1'", "N9", "C4"],
    ],
    "U": [
        ["C5'", "C4'", "C3'", "C2'"],
        ["C3'", "C4'", "O4'", "C1'"],
        ["C4'", "O4'", "C1'", "N1"],
        ["C5'", "C4'", "C3'", "O3'"],
        ["C3'", "C4'", "C5'", "O5'"],
        ["C4'", "C5'", "O5'", "P"],
        ["C5'", "O5'", "P", "OP1"],
        ["C5'", "O5'", "P", "OP2"],
        ["N1", "C1'", "C2'", "O2'"],
        ["C2'", "C1'", "N1", "C2"],
    ],
    "G": [
        ["C5'", "C4'", "C3'", "C2'"],
        ["C3'", "C4'", "O4'", "C1'"],
        ["C4'", "O4'", "C1'", "N9"],
        ["C5'", "C4'", "C3'", "O3'"],
        ["C3'", "C4'", "C5'", "O5'"],
        ["C4'", "C5'", "O5'", "P"],
        ["C5'", "O5'", "P", "OP1"],
        ["C5'", "O5'", "P", "OP2"],
        ["N9", "C1'", "C2'", "O2'"],
        ["C2'", "C1'", "N9", "C4"],
    ],
    "C": [
        ["C5'", "C4'", "C3'", "C2'"],
        ["C3'", "C4'", "O4'", "C1'"],
        ["C4'", "O4'", "C1'", "N1"],
        ["C5'", "C4'", "C3'", "O3'"],
        ["C3'", "C4'", "C5'", "O5'"],
        ["C4'", "C5'", "O5'", "P"],
        ["C5'", "O5'", "P", "OP1"],
        ["C5'", "O5'", "P", "OP2"],
        ["N1", "C1'", "C2'", "O2'"],
        ["C2'", "C1'", "N1", "C2"],
    ],
    "DA": [
        ["C5'", "C4'", "C3'", "C2'"],
        ["C3'", "C4'", "O4'", "C1'"],
        ["C4'", "O4'", "C1'", "N9"],
        ["C5'", "C4'", "C3'", "O3'"],
        ["C3'", "C4'", "C5'", "O5'"],
        ["C4'", "C5'", "O5'", "P"],
        ["C5'", "O5'", "P", "OP1"],
        ["C5'", "O5'", "P", "OP2"],
        None,
        ["C2'", "C1'", "N9", "C4"],
    ],
    "DT": [
        ["C5'", "C4'", "C3'", "C2'"],
        ["C3'", "C4'", "O4'", "C1'"],
        ["C4'", "O4'", "C1'", "N1"],
        ["C5'", "C4'", "C3'", "O3'"],
        ["C3'", "C4'", "C5'", "O5'"],
        ["C4'", "C5'", "O5'", "P"],
        ["C5'", "O5'", "P", "OP1"],
        ["C5'", "O5'", "P", "OP2"],
        None,
        ["C2'", "C1'", "N1", "C2"],
    ],
    "DG": [
        ["C5'", "C4'", "C3'", "C2'"],
        ["C3'", "C4'", "O4'", "C1'"],
        ["C4'", "O4'", "C1'", "N9"],
        ["C5'", "C4'", "C3'", "O3'"],
        ["C3'", "C4'", "C5'", "O5'"],
        ["C4'", "C5'", "O5'", "P"],
        ["C5'", "O5'", "P", "OP1"],
        ["C5'", "O5'", "P", "OP2"],
        None,
        ["C2'", "C1'", "N9", "C4"],
    ],
    "DC": [
        ["C5'", "C4'", "C3'", "C2'"],
        ["C3'", "C4'", "O4'", "C1'"],
        ["C4'", "O4'", "C1'", "N1"],
        ["C5'", "C4'", "C3'", "O3'"],
        ["C3'", "C4'", "C5'", "O5'"],
        ["C4'", "C5'", "O5'", "P"],
        ["C5'", "O5'", "P", "OP1"],
        ["C5'", "O5'", "P", "OP2"],
        None,
        ["C2'", "C1'", "N1", "C2"],
    ],
}


chi_angles_mask = {
    "DA": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
    "DC": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
    "DG": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
    "DT": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
    "A": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "C": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "G": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "U": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
}


chi_pi_periodic = {
    "DA": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "DC": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "DG": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "DT": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "A": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "C": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "G": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "U": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "-": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}


rigid_group_atom_positions = {
    "A": [
        ["C3'", 0, (-0.378, 1.475, 0.00)],

        ["C4'", 0, (1e-6, 0.000, 0.000)],
        ["O4'", 0, (1.450, -0.00, 0.000)],
        ["C5'", 0, (-0.508, -0.803, -1.174)],
        ["C2'", 1, (0.4258, 1.4607, 0.00)],
        ["C1'", 2, (0.4765, 1.3345, 0.000)],
        ["N9", 3, (0.4550, 1.4004, 0.000)],
        ["O3'", 4, (0.524, 1.321, 0.00)],
        ["O5'", 5, (0.511, 1.333, -0.0)],
        ["P", 6, (0.817, 1.367, -0.0)],
        ["OP1", 7, (0.470, 1.407, 0.00)],
        ["OP2", 8, (0.464, 1.409, -0.0)],
        ["O2'", 9, (0.467, 1.335, -0.0)],
        ["N1", 10, (2.807, 2.869, 0.002)],
        ["N3", 10, (0.446, 2.395, -0.008)],
        ["N6", 10, (4.438, 1.239, 0.022)],
        ["N7", 10, (2.110, -0.769, 0.013)],
        ["C2", 10, (1.510, 3.194, -0.006)],
        ["C4", 10, (0.817, 1.104, 0.000)],
        ["C5", 10, (2.108, 0.616, 0.009)],
        ["C6", 10, (3.146, 1.563, 0.011)],
        ["C8", 10, (0.838, -1.084, 0.008)],
    ],
    "U": [
        ["C3'", 0, (-0.378, 1.474, 0.00)],
        ["C4'", 0, (1e-6, 0.000, 0.000)],
        ["O4'", 0, (1.451, -0.00, 0.000)],
        ["C5'", 0, (-0.514, -0.806, -1.17)],
        ["C2'", 1, (0.4258, 1.4607, 0.00)],
        ["C1'", 2, (0.4765, 1.3345, 0.000)],
        ["N1", 3, (0.4550, 1.4004, 0.000)],
        ["O3'", 4, (0.522, 1.322, 0.00)],
        ["O5'", 5, (0.514, 1.332, -0.0)],
        ["P", 6, (0.820, 1.364, -0.0)],
        ["OP1", 7, (0.462, 1.408, -0.0)],
        ["OP2", 8, (0.466, 1.408, 0.00)],
        ["O2'", 9, (0.473, 1.333, -0.0)],
        ["N3", 10, (2.018, 1.154, -0.0)],
        ["C2", 10, (0.649, 1.221, 0.00)],
        ["C4", 10, (2.79, 0.014, -0.001)],
        ["C5", 10, (2.05, -1.21, -0.004)],
        ["C6", 10, (0.714, -1.175, -0.003)],
        ["O2", 10, (0.06, 2.288, -0.001)],
        ["O4", 10, (4.015, 0.113, 0.002)],
    ],
    "G": [
        ["C3'", 0, (-0.369, 1.476, 0.00)],
        ["C4'", 0, (1e-6, 0.000, 0.000)],
        ["O4'", 0, (1.450, -0.00, 0.000)],
        ["C5'", 0, (-0.513, -0.806, -1.171)],
        ["C2'", 1, (0.4258, 1.4607, 0.00)],
        ["C1'", 2, (0.4765, 1.3345, 0.000)],
        ["N9", 3, (0.4550, 1.4004, 0.000)],
        ["O3'", 4, (0.529, 1.319, 0.00)],
        ["O5'", 5, (0.514, 1.331, -0.0)],
        ["P", 6, (0.814, 1.367, -0.0)],
        ["OP1", 7, (0.472, 1.406, 0.00)],
        ["OP2", 8, (0.464, 1.408, -0.0)],
        ["O2'", 9, (0.472, 1.334, -0.0)],
        ["N1", 10, (2.750, 2.841, -0.006)],
        ["N2", 10, (1.216, 4.548, 0.001)],
        ["N3", 10, (0.415, 2.391, 0.005)],
        ["N7", 10, (2.096, -0.776, -0.013)],
        ["C2", 10, (1.437, 3.232, -0.0)],
        ["C4", 10, (0.818, 1.104, 0.00)],
        ["C5", 10, (2.102, 0.61, -0.007)],
        ["C6", 10, (3.186, 1.523, -0.009)],
        ["C8", 10, (0.830, -1.092, -0.01)],
        ["O6", 10, (4.394, 1.274, -0.014)],
    ],
    "C": [
        ["C3'", 0, (-0.372, 1.476, 0.00)],
        ["C4'", 0, (1e-6, 0.000, 0.000)],
        ["O4'", 0, (1.451, -0.00, 0.000)],
        ["C5'", 0, (-0.517, -0.809, -1.17)],
        ["C2'", 1, (0.4258, 1.4607, 0.00)],
        ["C1'", 2, (0.4765, 1.3345, 0.000)],
        ["N1", 3, (0.4550, 1.4004, 0.000)],
        ["O3'", 4, (0.528, 1.322, 0.00)],
        ["O5'", 5, (0.517, 1.332, -0.0)],
        ["P", 6, (0.818, 1.364, 0.0)],
        ["OP1", 7, (0.469, 1.407, -0.0)],
        ["OP2", 8, (0.469, 1.408, 0.00)],
        ["O2'", 9, (0.476, 1.333, -0.0)],
        ["N3", 10, (2.036, 1.22, 0.001)],
        ["N4", 10, (4.036, 0.115, 0.003)],
        ["C2", 10, (0.683, 1.220, 0.0)],
        ["C4", 10, (2.706, 0.067, -0.002)],
        ["C5", 10, (2.036, -1.188, -0.008)],
        ["C6", 10, (0.698, -1.175, -0.009)],
        ["O2", 10, (0.039, 2.276, 0.001)],
    ],
    "DA": [
        ["C3'", 0, (-0.378, 1.475, 0.00)],
        ["C4'", 0, (1e-6, 0.000, 0.000)],
        ["O4'", 0, (1.450, -0.00, 0.000)],
        ["C5'", 0, (-0.508, -0.803, -1.174)],
        ["C2'", 1, (0.4258, 1.4607, 0.00)],
        ["C1'", 2, (0.4765, 1.3345, 0.000)],
        ["N9", 3, (0.4550, 1.4004, 0.000)],
        ["O3'", 4, (0.524, 1.321, 0.00)],
        ["O5'", 5, (0.511, 1.333, -0.0)],
        ["P", 6, (0.817, 1.367, -0.0)],
        ["OP1", 7, (0.470, 1.407, 0.00)],
        ["OP2", 8, (0.464, 1.409, -0.0)],
        ["N1", 10, (2.807, 2.869, 0.002)],
        ["N3", 10, (0.446, 2.395, -0.008)],
        ["N6", 10, (4.438, 1.239, 0.022)],
        ["N7", 10, (2.110, -0.769, 0.013)],
        ["C2", 10, (1.510, 3.194, -0.006)],
        ["C4", 10, (0.817, 1.104, 0.000)],
        ["C5", 10, (2.108, 0.616, 0.009)],
        ["C6", 10, (3.146, 1.563, 0.011)],
        ["C8", 10, (0.838, -1.084, 0.008)],
    ],
    "DT": [
        ["C3'", 0, (-0.378, 1.474, 0.00)],
        ["C4'", 0, (1e-6, 0.000, 0.000)],
        ["O4'", 0, (1.451, -0.00, 0.000)],
        ["C5'", 0, (-0.514, -0.806, -1.17)],
        ["C2'", 1, (0.4258, 1.4607, 0.00)],
        ["C1'", 2, (0.4765, 1.3345, 0.000)],
        ["N1", 3, (0.4550, 1.4004, 0.000)],
        ["O3'", 4, (0.522, 1.322, 0.00)],
        ["O5'", 5, (0.514, 1.332, -0.0)],
        ["P", 6, (0.820, 1.364, -0.0)],
        ["OP1", 7, (0.462, 1.408, -0.0)],
        ["OP2", 8, (0.466, 1.408, 0.00)],
        ["N3", 10, (2.018, 1.154, -0.0)],
        ["C2", 10, (0.649, 1.221, 0.00)],
        ["C4", 10, (2.79, 0.014, -0.001)],
        ["C5", 10, (2.05, -1.21, -0.004)],
        ["C6", 10, (0.714, -1.175, -0.003)],
        ["O2", 10, (0.06, 2.288, -0.001)],
        ["O4", 10, (4.015, 0.113, 0.002)],
    ],
    "DG": [
        ["C3'", 0, (-0.369, 1.476, 0.00)],
        ["C4'", 0, (1e-6, 0.000, 0.000)],
        ["O4'", 0, (1.450, -0.00, 0.000)],
        ["C5'", 0, (-0.513, -0.806, -1.171)],
        ["C2'", 1, (0.4258, 1.4607, 0.00)],
        ["C1'", 2, (0.4765, 1.3345, 0.000)],
        ["N9", 3, (0.4550, 1.4004, 0.000)],
        ["O3'", 4, (0.529, 1.319, 0.00)],
        ["O5'", 5, (0.514, 1.331, -0.0)],
        ["P", 6, (0.814, 1.367, -0.0)],
        ["OP1", 7, (0.472, 1.406, 0.00)],
        ["OP2", 8, (0.464, 1.408, -0.0)],
        ["N1", 10, (2.750, 2.841, -0.006)],
        ["N2", 10, (1.216, 4.548, 0.001)],
        ["N3", 10, (0.415, 2.391, 0.005)],
        ["N7", 10, (2.096, -0.776, -0.013)],
        ["C2", 10, (1.437, 3.232, -0.0)],
        ["C4", 10, (0.818, 1.104, 0.00)],
        ["C5", 10, (2.102, 0.61, -0.007)],
        ["C6", 10, (3.186, 1.523, -0.009)],
        ["C8", 10, (0.830, -1.092, -0.01)],
        ["O6", 10, (4.394, 1.274, -0.014)],
    ],
    "DC": [
        ["C3'", 0, (-0.372, 1.476, 0.00)],
        ["C4'", 0, (1e-6, 0.000, 0.000)],
        ["O4'", 0, (1.451, -0.00, 0.000)],
        ["C5'", 0, (-0.517, -0.809, -1.17)],
        ["C2'", 1, (0.4258, 1.4607, 0.00)],
        ["C1'", 2, (0.4765, 1.3345, 0.000)],
        ["N1", 3, (0.4550, 1.4004, 0.000)],
        ["O3'", 4, (0.528, 1.322, 0.00)],
        ["O5'", 5, (0.517, 1.332, -0.0)],
        ["P", 6, (0.818, 1.364, 0.0)],
        ["OP1", 7, (0.469, 1.407, -0.0)],
        ["OP2", 8, (0.469, 1.408, 0.00)],
        ["N3", 10, (2.036, 1.22, 0.001)],
        ["N4", 10, (4.036, 0.115, 0.003)],
        ["C2", 10, (0.683, 1.220, 0.0)],
        ["C4", 10, (2.706, 0.067, -0.002)],
        ["C5", 10, (2.036, -1.188, -0.008)],
        ["C6", 10, (0.698, -1.175, -0.009)],
        ["O2", 10, (0.039, 2.276, 0.001)],
    ],
}


residue_atoms = {
    "A": [
        "C5'",
        "C4'",
        "O4'",
        "N9",
        "C1'",
        "O5'",
        "P",
        "O3'",
        "C4",
        "O2'",
        "C3'",
        "C2'",
        "OP1",
        "OP2",
        "N1",
        "N3",
        "N6",
        "N7",
        "C2",
        "C5",
        "C6",
        "C8",
    ],
    "U": [
        "C5'",
        "C4'",
        "O4'",
        "N1",
        "C1'",
        "O5'",
        "P",
        "O3'",
        "C2",
        "O2'",
        "C3'",
        "C2'",
        "OP1",
        "OP2",
        "N3",
        "C4",
        "C5",
        "C6",
        "O2",
        "O4",
    ],
    "G": [
        "C5'",
        "C4'",
        "O4'",
        "N9",
        "C1'",
        "O5'",
        "P",
        "O3'",
        "C4",
        "O2'",
        "C3'",
        "C2'",
        "OP1",
        "OP2",
        "N1",
        "N2",
        "N3",
        "N7",
        "C2",
        "C5",
        "C6",
        "C8",
        "O6",
    ],
    "C": [
        "C5'",
        "C4'",
        "O4'",
        "N1",
        "C1'",
        "O5'",
        "P",
        "O3'",
        "C2",
        "O2'",
        "C3'",
        "C2'",
        "OP1",
        "OP2",
        "N3",
        "N4",
        "C4",
        "C5",
        "C6",
        "O2",
    ],
    "DA": [
        "C5'",
        "C4'",
        "O4'",
        "N9",
        "C1'",
        "O5'",
        "P",
        "O3'",
        "C4",
        "C3'",
        "C2'",
        "OP1",
        "OP2",
        "N1",
        "N3",
        "N6",
        "N7",
        "C2",
        "C5",
        "C6",
        "C8",
    ],
    "DT": [
        "C5'",
        "C4'",
        "O4'",
        "N1",
        "C1'",
        "O5'",
        "P",
        "O3'",
        "C2",
        "C3'",
        "C2'",
        "OP1",
        "OP2",
        "N3",
        "C4",
        "C5",
        "C6",
        "O2",
        "O4",
    ],
    "DG": [
        "C5'",
        "C4'",
        "O4'",
        "N9",
        "C1'",
        "O5'",
        "P",
        "O3'",
        "C4",
        "C3'",
        "C2'",
        "OP1",
        "OP2",
        "N1",
        "N2",
        "N3",
        "N7",
        "C2",
        "C5",
        "C6",
        "C8",
        "O6",
    ],
    "DC": [
        "C5'",
        "C4'",
        "O4'",
        "N1",
        "C1'",
        "O5'",
        "P",
        "O3'",
        "C2",
        "C3'",
        "C2'",
        "OP1",
        "OP2",
        "N3",
        "N4",
        "C4",
        "C5",
        "C6",
        "O2",
    ],
}


van_der_waals_radius = {
    "C": 1.7,
    "N": 1.55,
    "O": 1.52,
    "P": 1.8,
}

Bond = collections.namedtuple(
    "Bond",
    ["atom1_name", "atom2_name", "length", "stddev"],
)

BondAngle = collections.namedtuple(
    "BondAngle",
    ["atom1_name", "atom2_name", "atom3name", "angle_rad", "stddev"],
)


@functools.cache
def load_stereo_chemical_props() -> tuple[
    Mapping[str, list[Bond]],
    Mapping[str, list[Bond]],
    Mapping[str, list[BondAngle]],
]:


    stereo_chemical_props = resources.read_text(
        "opencomplex.resources", "stereo_chemical_props_RNA.txt"
    )

    lines_iter = iter(stereo_chemical_props.splitlines())

    residue_bonds = {}
    next(lines_iter)
    for line in lines_iter:
        if line.strip() == "-":
            break
        bond, resname, length, stddev = line.split()
        atom1, atom2 = bond.split("-")
        if resname not in residue_bonds:
            residue_bonds[resname] = []
        residue_bonds[resname].append(Bond(atom1, atom2, float(length), float(stddev)))
    residue_bonds["X"] = []


    residue_bond_angles = {}
    next(lines_iter)
    next(lines_iter)
    for line in lines_iter:
        if line.strip() == "-":
            break
        bond, resname, angle_degree, stddev_degree = line.split()
        atom1, atom2, atom3 = bond.split("-")
        if resname not in residue_bond_angles:
            residue_bond_angles[resname] = []
        residue_bond_angles[resname].append(
            BondAngle(
                atom1,
                atom2,
                atom3,
                float(angle_degree) / 180.0 * np.pi,
                float(stddev_degree) / 180.0 * np.pi,
            )
        )
    residue_bond_angles["X"] = []

    def make_bond_key(atom1_name, atom2_name):

        return "-".join(sorted([atom1_name, atom2_name]))


    residue_virtual_bonds = {}
    for resname, bond_angles in residue_bond_angles.items():

        bond_cache = {}
        for b in residue_bonds[resname]:
            bond_cache[make_bond_key(b.atom1_name, b.atom2_name)] = b
        residue_virtual_bonds[resname] = []
        for ba in bond_angles:
            bond1 = bond_cache[make_bond_key(ba.atom1_name, ba.atom2_name)]
            bond2 = bond_cache[make_bond_key(ba.atom2_name, ba.atom3name)]


            gamma = ba.angle_rad
            length = np.sqrt(
                bond1.length**2
                + bond2.length**2
                - 2 * bond1.length * bond2.length * np.cos(gamma)
            )


            dl_outer = 0.5 / length
            dl_dgamma = (2 * bond1.length * bond2.length * np.sin(gamma)) * dl_outer
            dl_db1 = (2 * bond1.length - 2 * bond2.length * np.cos(gamma)) * dl_outer
            dl_db2 = (2 * bond2.length - 2 * bond1.length * np.cos(gamma)) * dl_outer
            stddev = np.sqrt(
                (dl_dgamma * ba.stddev) ** 2
                + (dl_db1 * bond1.stddev) ** 2
                + (dl_db2 * bond2.stddev) ** 2
            )
            residue_virtual_bonds[resname].append(
                Bond(ba.atom1_name, ba.atom3name, length, stddev)
            )

    return (residue_bonds, residue_virtual_bonds, residue_bond_angles)


between_res_bond_length_o3_p = 1.602
between_res_bond_length_stddev_o3_p = 0.001

between_res_cos_angles_c3_o3_p = [-0.5030, 0.9867]
between_res_cos_angles_o3_p_o5 = [-0.2352, 0.9932]
between_res_cos_angles_o4_c1_n = [-0.3322, 0.9979]
between_res_cos_angles_c1_n_c = [-0.5383, 0.9988]
between_res_cos_angles_c1_c2_c3 = [-0.2006, 0.9999]
between_res_cos_angles_c2_c3_c4 = [-0.2126, 0.9996]


atom_types = [
    "C1'",
    "C2'",
    "C3'",
    "C4'",
    "C5'",
    "O5'",
    "O4'",
    "O3'",
    "O2'",
    "P",
    "OP1",
    "OP2",
    "N1",
    "N2",
    "N3",
    "N4",
    "N6",
    "N7",
    "N9",
    "C2",
    "C4",
    "C5",
    "C6",
    "C8",
    "O2",
    "O4",
    "O6",
]

atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}
atom_type_num = len(atom_types)

restypes = [
    "DA",
    "DC",
    "DG",
    "DT",
    "A",
    "C",
    "G",
    "U",
]
RNA_RESTYPES = ("A", "C", "G", "U")
RNA_TORSION_ATOM37_INDICES = np.asarray(
    [
        [
            [atom_order[atom_name] for atom_name in torsion_atoms]
            for torsion_atoms in chi_angles_atoms[restype]
        ]
        for restype in RNA_RESTYPES
    ],
    dtype=np.int64,
)
deoxy_restypes = [
    "DA",
    "DC",
    "DG",
    "DT",
]


restype_name_to_compact_atom_names = {
    "DA": [
        "C3'",
        "C4'",
        "O4'",
        "C2'",
        "C1'",
        "C5'",
        "O3'",
        "O5'",
        "P",
        "OP1",
        "OP2",
        "N9",
        "C4",
        "N1",
        "N3",
        "N6",
        "N7",
        "C2",
        "C5",
        "C6",
        "C8",
        "",
        "",
    ],
    "DC": [
        "C3'",
        "C4'",
        "O4'",
        "C2'",
        "C1'",
        "C5'",
        "O3'",
        "O5'",
        "P",
        "OP1",
        "OP2",
        "N1",
        "C2",
        "N3",
        "N4",
        "C4",
        "C5",
        "C6",
        "O2",
        "",
        "",
        "",
        "",
    ],
    "DG": [
        "C3'",
        "C4'",
        "O4'",
        "C2'",
        "C1'",
        "C5'",
        "O3'",
        "O5'",
        "P",
        "OP1",
        "OP2",
        "N9",
        "C4",
        "N1",
        "N2",
        "N3",
        "N7",
        "C2",
        "C5",
        "C6",
        "C8",
        "O6",
        "",
    ],
    "DT": [
        "C3'",
        "C4'",
        "O4'",
        "C2'",
        "C1'",
        "C5'",
        "O3'",
        "O5'",
        "P",
        "OP1",
        "OP2",
        "N1",
        "C2",
        "N3",
        "C4",
        "C5",
        "C6",
        "O2",
        "O4",
        "",
        "",
        "",
        "",
    ],
    "A": [
        "C3'",
        "C4'",
        "O4'",
        "C2'",
        "C1'",
        "C5'",
        "O3'",
        "O5'",
        "P",
        "OP1",
        "OP2",
        "N9",
        "O2'",
        "C4",
        "N1",
        "N3",
        "N6",
        "N7",
        "C2",
        "C5",
        "C6",
        "C8",
        "",
    ],
    "C": [
        "C3'",
        "C4'",
        "O4'",
        "C2'",
        "C1'",
        "C5'",
        "O3'",
        "O5'",
        "P",
        "OP1",
        "OP2",
        "N1",
        "O2'",
        "C2",
        "N3",
        "N4",
        "C4",
        "C5",
        "C6",
        "O2",
        "",
        "",
        "",
    ],
    "G": [
        "C3'",
        "C4'",
        "O4'",
        "C2'",
        "C1'",
        "C5'",
        "O3'",
        "O5'",
        "P",
        "OP1",
        "OP2",
        "N9",
        "O2'",
        "C4",
        "N1",
        "N2",
        "N3",
        "N7",
        "C2",
        "C5",
        "C6",
        "C8",
        "O6",
    ],
    "U": [
        "C3'",
        "C4'",
        "O4'",
        "C2'",
        "C1'",
        "C5'",
        "O3'",
        "O5'",
        "P",
        "OP1",
        "OP2",
        "N1",
        "O2'",
        "C2",
        "N3",
        "C4",
        "C5",
        "C6",
        "O2",
        "O4",
        "",
        "",
        "",
    ],
}
compact_atom_type_num = 23
restype_name_to_compact_atom_order = {
    restype: {atom_type: i for i, atom_type in enumerate(compact_atom_names)}
    for restype, compact_atom_names in restype_name_to_compact_atom_names.items()
}
restype_name_atom_num = {
    restype: len(list(filter(None, restype_name_to_compact_atom_names[restype])))
    for restype in restypes
}

restype_name_to_full_atom_names = {
    restype_name: atom_types for restype_name in list(restype_name_to_compact_atom_names.keys())
}


restype_order = {restype: i for i, restype in enumerate(restypes)}
restype_num = len(restypes)

unk_restype_index = restype_num


restypes_with_x = restypes + ["X"]
restype_order_with_x = {restype: i for i, restype in enumerate(restypes_with_x)}


def _make_standard_atom_mask() -> np.ndarray:


    mask = np.zeros([restype_num + 1, atom_type_num], dtype=np.int32)
    for restype, restype_letter in enumerate(restypes):

        restype_name = restype_letter
        atom_names = residue_atoms[restype_name]
        for atom_name in atom_names:
            atom_type = atom_order[atom_name]
            mask[restype, atom_type] = 1
    return mask


STANDARD_ATOM_MASK = _make_standard_atom_mask()


def chi_angle_atom(atom_index: int) -> np.ndarray:

    chi_angles_index = {}
    one_hots = []

    for k, v in chi_angles_atoms.items():
        indices = [atom_types.index(s[atom_index]) for s in v]
        indices.extend([-1] * (NUM_NA_TORSIONS - len(indices)))
        chi_angles_index[k] = indices

    for r in restypes:


        one_hot = np.eye(atom_type_num)[chi_angles_index[r]]
        one_hots.append(one_hot)

    one_hots.append(np.zeros([NUM_NA_TORSIONS, atom_type_num]))
    one_hot = np.stack(one_hots, axis=0)
    one_hot = np.transpose(one_hot, [0, 2, 1])

    return one_hot


def _make_rigid_transformation_4x4(ex, ey, translation):


    ex_normalized = ex / np.linalg.norm(ex)


    ey_normalized = ey - np.dot(ey, ex_normalized) * ex_normalized
    ey_normalized /= np.linalg.norm(ey_normalized)


    eznorm = np.cross(ex_normalized, ey_normalized)
    mat = np.stack([ex_normalized, ey_normalized, eznorm, translation]).transpose()
    mat = np.concatenate([mat, [[0.0, 0.0, 0.0, 1.0]]], axis=0)
    return mat


nttype_atom27_to_rigid_group = np.zeros([9, 27], dtype=int)

nttype_atom27_mask = np.zeros([9, 27], dtype=np.float32)

nttype_atom27_rigid_group_positions = np.zeros([9, 27, 3], dtype=np.float32)

nttype_compact_atom_to_rigid_group = np.zeros([9, 23], dtype=int)

nttype_compact_atom_mask = np.zeros([9, 23], dtype=np.float32)

nttype_compact_atom_rigid_group_positions = np.zeros([9, 23, 3], dtype=np.float32)

nttype_rigid_group_default_frame = np.zeros([9, 11, 4, 4], dtype=np.float32)


def _make_rigid_group_constants():

    for nttype, nttype_letter in enumerate(restypes):

        ntname = nttype_letter
        for atomname, group_idx, atom_position in rigid_group_atom_positions[ntname]:
            atomtype = atom_order[atomname]
            nttype_atom27_to_rigid_group[nttype, atomtype] = group_idx
            nttype_atom27_mask[nttype, atomtype] = 1
            nttype_atom27_rigid_group_positions[nttype, atomtype, :] = atom_position

            atom23idx = restype_name_to_compact_atom_names[ntname].index(atomname)
            nttype_compact_atom_to_rigid_group[nttype, atom23idx] = group_idx
            nttype_compact_atom_mask[nttype, atom23idx] = 1
            nttype_compact_atom_rigid_group_positions[nttype, atom23idx, :] = atom_position

    for nttype, nttype_letter in enumerate(restypes):

        ntname = nttype_letter
        atom_positions = {
            name: np.array(pos) for name, _, pos in rigid_group_atom_positions[ntname]
        }


        nttype_rigid_group_default_frame[nttype, 0, :, :] = np.eye(4)


        if chi_angles_mask[ntname][0]:
            base_atom_names = chi_angles_atoms[ntname][0]
            base_atom_positions = [atom_positions[name] for name in base_atom_names]
            mat = _make_rigid_transformation_4x4(
                ex=base_atom_positions[2] - base_atom_positions[1],
                ey=base_atom_positions[0] - base_atom_positions[1],
                translation=base_atom_positions[2],
            )
            nttype_rigid_group_default_frame[nttype, 1, :, :] = mat


        if chi_angles_mask[ntname][1]:
            base_atom_names = chi_angles_atoms[ntname][1]
            base_atom_positions = [atom_positions[name] for name in base_atom_names]
            mat = _make_rigid_transformation_4x4(
                ex=base_atom_positions[2] - base_atom_positions[1],
                ey=base_atom_positions[0] - base_atom_positions[1],
                translation=base_atom_positions[2],
            )
            nttype_rigid_group_default_frame[nttype, 2, :, :] = mat


        if chi_angles_mask[ntname][2]:
            axis_end_atom_name = chi_angles_atoms[ntname][2][2]
            axis_end_atom_position = atom_positions[axis_end_atom_name]
            mat = _make_rigid_transformation_4x4(
                ex=axis_end_atom_position,
                ey=np.array([-1.0, 0.0, 0.0]),
                translation=axis_end_atom_position,
            )
            nttype_rigid_group_default_frame[nttype, 3, :, :] = mat


        if chi_angles_mask[ntname][3]:
            base_atom_names = chi_angles_atoms[ntname][3]
            base_atom_positions = [atom_positions[name] for name in base_atom_names]
            mat = _make_rigid_transformation_4x4(
                ex=base_atom_positions[2] - base_atom_positions[1],
                ey=base_atom_positions[0] - base_atom_positions[1],
                translation=base_atom_positions[2],
            )
            nttype_rigid_group_default_frame[nttype, 4, :, :] = mat


        if chi_angles_mask[ntname][4]:
            base_atom_names = chi_angles_atoms[ntname][4]
            base_atom_positions = [atom_positions[name] for name in base_atom_names]
            mat = _make_rigid_transformation_4x4(
                ex=base_atom_positions[2] - base_atom_positions[1],
                ey=base_atom_positions[0] - base_atom_positions[1],
                translation=base_atom_positions[2],
            )
            nttype_rigid_group_default_frame[nttype, 5, :, :] = mat


        if chi_angles_mask[ntname][5]:
            axis_end_atom_name = chi_angles_atoms[ntname][5][2]
            axis_end_atom_position = atom_positions[axis_end_atom_name]
            mat = _make_rigid_transformation_4x4(
                ex=axis_end_atom_position,
                ey=np.array([-1.0, 0.0, 0.0]),
                translation=axis_end_atom_position,
            )
            nttype_rigid_group_default_frame[nttype, 6, :, :] = mat


        for torsion_idx in range(6, 8):
            if chi_angles_mask[ntname][torsion_idx]:
                axis_end_atom_name = chi_angles_atoms[ntname][torsion_idx][2]
                axis_end_atom_position = atom_positions[axis_end_atom_name]
                mat = _make_rigid_transformation_4x4(
                    ex=axis_end_atom_position,
                    ey=np.array([-1.0, 0.0, 0.0]),
                    translation=axis_end_atom_position,
                )
                nttype_rigid_group_default_frame[nttype, 1 + torsion_idx, :, :] = mat


        if chi_angles_mask[ntname][8]:
            axis_end_atom_name = chi_angles_atoms[ntname][8][2]
            axis_end_atom_position = atom_positions[axis_end_atom_name]
            mat = _make_rigid_transformation_4x4(
                ex=axis_end_atom_position,
                ey=np.array([-1.0, 0.0, 0.0]),
                translation=axis_end_atom_position,
            )
            nttype_rigid_group_default_frame[nttype, 9, :, :] = mat


        if chi_angles_mask[ntname][9]:
            axis_end_atom_name = chi_angles_atoms[ntname][9][2]
            axis_end_atom_position = atom_positions[axis_end_atom_name]
            mat = _make_rigid_transformation_4x4(
                ex=axis_end_atom_position,
                ey=np.array([-1.0, 0.0, 0.0]),
                translation=axis_end_atom_position,
            )
            nttype_rigid_group_default_frame[nttype, 10, :, :] = mat


_make_rigid_group_constants()


def make_compact_atom_dists_bounds(overlap_tolerance=1.5, bond_length_tolerance_factor=15):

    restype_compact_atom_bond_lower_bound = np.zeros([9, 23, 23], np.float32)
    restype_compact_atom_bond_upper_bound = np.zeros([9, 23, 23], np.float32)
    restype_compact_atom_bond_stddev = np.zeros([9, 23, 23], np.float32)
    residue_bonds, residue_virtual_bonds, _ = load_stereo_chemical_props()
    for restype, restype_letter in enumerate(restypes):

        resname = restype_letter
        atom_list = restype_name_to_compact_atom_names[resname]


        for atom1_idx, atom1_name in enumerate(atom_list):
            if not atom1_name:
                continue
            atom1_radius = van_der_waals_radius[atom1_name[0]]
            for atom2_idx, atom2_name in enumerate(atom_list):
                if (not atom2_name) or atom1_idx == atom2_idx:
                    continue
                atom2_radius = van_der_waals_radius[atom2_name[0]]
                lower = atom1_radius + atom2_radius - overlap_tolerance
                upper = 1e10
                restype_compact_atom_bond_lower_bound[restype, atom1_idx, atom2_idx] = lower
                restype_compact_atom_bond_lower_bound[restype, atom2_idx, atom1_idx] = lower
                restype_compact_atom_bond_upper_bound[restype, atom1_idx, atom2_idx] = upper
                restype_compact_atom_bond_upper_bound[restype, atom2_idx, atom1_idx] = upper


        for b in residue_bonds[resname] + residue_virtual_bonds[resname]:
            atom1_idx = atom_list.index(b.atom1_name)
            atom2_idx = atom_list.index(b.atom2_name)
            lower = b.length - bond_length_tolerance_factor * b.stddev
            upper = b.length + bond_length_tolerance_factor * b.stddev
            restype_compact_atom_bond_lower_bound[restype, atom1_idx, atom2_idx] = lower
            restype_compact_atom_bond_lower_bound[restype, atom2_idx, atom1_idx] = lower
            restype_compact_atom_bond_upper_bound[restype, atom1_idx, atom2_idx] = upper
            restype_compact_atom_bond_upper_bound[restype, atom2_idx, atom1_idx] = upper
            restype_compact_atom_bond_stddev[restype, atom1_idx, atom2_idx] = b.stddev
            restype_compact_atom_bond_stddev[restype, atom2_idx, atom1_idx] = b.stddev
    return {
        "lower_bound": restype_compact_atom_bond_lower_bound,
        "upper_bound": restype_compact_atom_bond_upper_bound,
        "stddev": restype_compact_atom_bond_stddev,
    }


def restype_to_str_sequence(butype):
    return "".join([restypes_with_x[butype[i]] for i in range(len(butype))])
