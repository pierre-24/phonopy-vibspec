import numpy
from textwrap import dedent

from typing import TextIO, Optional, Tuple, List
from numpy.typing import NDArray

from phonopy.structure.atoms import PhonopyAtoms

from phonopy_vibspec import logger


l_logger = logger.getChild(__name__)


COVALENT_RADII = {  # from 10.1039/B801115J
    'H': 0.31,
    'He': 0.28,
    'Li': 1.28,
    'Be': 0.96,
    'B': 0.84,
    'C': 0.76,
    'N': 0.71,
    'O': 0.66,
    'F': 0.57,
    'Ne': 0.58,
    'Na': 1.66,
    'Mg': 1.41,
    'Al': 1.21,
    'Si': 1.11,
    'P': 1.07,
    'S': 1.05,
    'Cl': 1.02,
    'Ar': 1.06,
    'K': 2.03,
    'Ca': 1.76,
    'Sc': 1.70,
    'Ti': 1.60,
    'V': 1.53,
    'Cr': 1.39,
    'Mn': 1.39,
    'Fe': 1.32,
    'Co': 1.26,
    'Ni': 1.24,
    'Cu': 1.32,
    'Zn': 1.22,
    'Ga': 1.22,
    'Ge': 1.20,
    'As': 1.19,
    'Se': 1.20,
    'Br': 1.20,
    'Kr': 1.16,
    'Rb': 2.20,
    'Sr': 1.95,
    'Y': 1.90,
    'Zr': 1.75,
    'Nb': 1.64,
    'Mo': 1.54,
    'Tc': 1.47,
    'Ru': 1.46,
    'Rh': 1.42,
    'Pd': 1.39,
    'Ag': 1.45,
    'Cd': 1.44,
    'In': 1.42,
    'Sn': 1.39,
    'Sb': 1.39,
    'Te': 1.38,
    'I': 1.39,
    'Xe': 1.40,
    'Cs': 2.44,
    'Ba': 2.15,
    'La': 2.07,
    'Ce': 2.04,
    'Pr': 2.03,
    'Nd': 2.01,
    'Pm': 1.99,
    'Sm': 1.98,
    'Eu': 1.98,
    'Gd': 1.96,
    'Tb': 1.94,
    'Dy': 1.92,
    'Ho': 1.92,
    'Er': 1.89,
    'Tm': 1.90,
    'Yb': 1.87,
    'Lu': 1.87,
    'Hf': 1.75,
    'Ta': 1.70,
    'W': 1.62,
    'Re': 1.51,
    'Os': 1.44,
    'Ir': 1.41,
    'Pt': 1.36,
    'Au': 1.36,
    'Hg': 1.32,
    'Tl': 1.45,
    'Pb': 1.46,
    'Bi': 1.48,
    'Po': 1.40,
    'At': 1.50,
    'Rn': 1.50,
    'Fr': 2.60,
    'Ra': 2.21,
    'Ac': 2.15,
    'Th': 2.06,
    'Pa': 2.00,
    'U': 1.96,
    'Np': 1.90,
    'Pu': 1.87,
    'Am': 1.80,
    'Cm': 1.69,
}


class VestaVector:
    """See https://jp-minerals.org/vesta/en/doc/VESTAch9.html#x22-1140009.1 for the interface.

    Note: the orientation is actually give in modulus along crystalographic axes.
    """
    def __init__(
        # for VECTR
        self, is_polar: bool, orientation: List[float], sites: List[int],
        # for VECTT:
        radius: float = 0.15, color: Tuple[float, float, float] = (0, 255, 0),
        through_atom: bool = False, add_radius: bool = False,
    ):

        self.is_polar = is_polar
        self.orientation = orientation
        self.sites = sites

        self.radius = radius
        self.color = color
        self.through_atoms = through_atom
        self.add_radius = add_radius

    def to_vectr(self, index: int) -> str:
        r = '{:>4} {:10.6f} {:10.6f} {:10.6f} {}\n'.format(
            index + 1,
            *self.orientation,
            1 if self.is_polar else 0
        )

        for si in self.sites:
            r += '{:>6}   0    0    0    0\n'.format(si + 1)
        r += '   0 0 0 0 0\n'

        return r

    def to_vectt(self, index: int) -> str:
        return '{:4} {:10.6f} {:3} {:3} {:3} {}\n'.format(
            index + 1,
            self.radius,
            *self.color,
            (1 if self.through_atoms else 0) + (2 if self.add_radius else 0)
        )


def cell_to_cellpar(cell: NDArray[float]) -> Tuple[float, float, float, float, float]:
    """Returns the cell parameters `(a, b, c, alpha, beta, gamma)`.
    Angle are in degree.

    Original code from https://wiki.fysik.dtu.dk/ase/_modules/ase/geometry/cell.html#cell_to_cellpar
    """

    assert cell.shape == (3, 3)

    lengths = tuple(numpy.linalg.norm(v) for v in cell)
    angles = []
    for i in range(3):
        j = i - 1
        k = i - 2
        ll = lengths[j] * lengths[k]
        if ll > 1e-16:
            x = numpy.dot(cell[j], cell[k]) / ll
            angle = 180.0 / numpy.pi * numpy.arccos(x)
        else:
            angle = 90.0
        angles.append(angle)
    return lengths + tuple(angles)


def make_vesta_file(f: TextIO, structure: PhonopyAtoms, vectors: Optional[List[VestaVector]] = None, **kwargs):
    """
    Create a (minimal) VESTA (https://jp-minerals.org/vesta/) file to be visualized.
    Thanks to the ability of the VESTA format to include vectors, it is possible to see a given phonon mode.

    However, the file format is not documented :(
    Bits of information found in
    https://aiida-crystal17.readthedocs.io/en/stable/_modules/aiida_crystal17/parsers/raw/vesta.html,
    https://github.com/Stanford-MCTG/VASP-plot-modes/, and https://github.com/lucydot/vesta_vectors.

    Note: it seems that VESTA is not a column format, so the size and alignment are there for readability reasons.

    Note: there are a lot of section missing, but VESTA seems to fill the missing one by itself, so this should be
    fine.
    """

    l_logger.info('Write VESTA file')

    # header
    f.write('#VESTA_FORMAT_VERSION 3.3.0\n\nCRYSTAL\n\nTITLE\n{}\n\n'.format(kwargs.get('title', 'Unitcell')))

    # symmetry
    # Let's just use P1 for the moment!
    # TODO: other symmetries?
    f.write(
        dedent(
            """\
        GROUP
        1 1 P 1
        SYMOP
         0.000000  0.000000  0.000000  1  0  0   0  1  0   0  0  1   1
         -1.0 -1.0 -1.0  0 0 0  0 0 0  0 0 0
        TRANM 0
          0.000000  0.000000  0.000000  1  0  0   0  1  0   0  0  1
        LTRANSL
          -1
          0.000000  0.000000  0.000000  0.000000  0.000000  0.000000
        LORIENT
          -1   0   0   0   0
          1.000000  0.000000  0.000000  1.000000  0.000000  0.000000
          0.000000  0.000000  1.000000  0.000000  0.000000  1.000000
        LMATRIX
          1.000000  0.000000  0.000000  0.000000
          0.000000  1.000000  0.000000  0.000000
          0.000000  0.000000  1.000000  0.000000
          0.000000  0.000000  0.000000  1.000000
          0.000000  0.000000  0.000000\n"""
        )
    )

    # cell parameters
    cellpars = cell_to_cellpar(structure.cell)
    f.write('CELLP\n{:10.6f} {:10.6f} {:10.6f} {:10.6f} {:10.6f} {:10.6f}\n'.format(*cellpars))
    f.write('  0.000000   0.000000   0.000000   0.000000   0.000000   0.000000\n')

    # atoms
    f.write('STRUC\n')
    for i in range(structure.get_number_of_atoms()):
        # symbol and position
        f.write('  {:<2d} {:>2s} {:>12s} {:7.4f} {:10.6f} {:10.6f} {:10.6f} {:>5} -\n'.format(
            i + 1,
            structure.symbols[i],
            '{}{}'.format(structure.symbols[i], i + 1),  # label
            1.0,  # occupancy
            *structure.scaled_positions[i],
            '1'  # wyck?
        ))

        # charge, if any:
        f.write('{} 0.000000   0.000000   0.000000 {:5.2f}\n'.format(' ' * 30, 0))

    f.write('  0 0 0 0 0 0 0\n')

    # add bond search parameters the best we can (e.g., sum of covalent radii for each pair)
    f.write('SBOND\n')
    symbols_set = list(set(structure.symbols))
    idx = 0
    for i, si in enumerate(symbols_set):
        for sj in symbols_set[i:]:
            idx += 1

            f.write('{:>4} {:>4s} {:>4} {:10.6f} {:10.6f} {:2} {:2} {:2} {:2} {:2} {:10.6f}  {:10.6f}'
                    ' 127 127 127\n'.format(
                        idx + 1,
                        si,
                        sj,
                        0,  # min
                        (1 + kwargs.get('bond_tol', 0.1)) * (COVALENT_RADII[si] + COVALENT_RADII[sj]),
                        0,  # search mode?
                        1,  # bound mode?
                        0,  # show polyheadra
                        0,  # search by label?
                        1,  # ?
                        0.25,  # width
                        2.0,  # radius
                    ))

    f.write('  0 0 0 0\n')

    # repeat unit cell if any
    f.write('BOUND\n{:8} {:8} {:8} {:8} {:8} {:8}\n  0   0   0   0  0\n'.format(
        *kwargs.get('bounds', (0, 1, 0, 1, 0, 1))
    ))

    # vectors
    if vectors is not None:
        f.write('VECTR\n')
        for i, v in enumerate(vectors):
            f.write(v.to_vectr(i))

        f.write(' 0 0 0 0 0\n')

        f.write('VECTT\n')
        for i, v in enumerate(vectors):
            f.write(v.to_vectt(i))
        f.write(' 0 0 0 0 0\n')
