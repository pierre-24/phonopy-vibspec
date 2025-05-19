import pathlib
import numpy

from numpy.typing import NDArray

import phonopy
from phonopy.interface import calculator as phonopy_calculator

from typing import Optional, List, Tuple, Union

from phonopy_vibspec import logger, GetListWithinBounds
try:
    from phonopy.units import VaspToCm  # noqa
except ImportError:
    from phonopy.physical_units import get_physical_units

from phonopy_vibspec.spectra import RamanSpectrum, InfraredSpectrum
from phonopy_vibspec.vesta import VestaVector, make_vesta_file

# [(value, coef), ...]
# see https://en.wikipedia.org/wiki/Finite_difference_coefficient
TWO_POINTS_STENCIL = [(-1, -.5), (1, .5)]  # two-points, centered


HUGE_MASS = 10000  # AMU

physical_units = get_physical_units()

l_logger = logger.getChild(__name__)


class PhononsAnalyzer:
    """Use Phonopy to extract phonon frequencies and eigenmodes, as well as irreps, at a given q-point
    (default is Gamma).
    """

    DC_GEOMETRY_TEMPLATE = 'dielec_mode{:04d}_step{:02d}.vasp'
    VESTA_MODE_TEMPLATE = 'mode{:04d}.vesta'

    def __init__(
            self,
            phonon: phonopy.Phonopy,
            q: Union[NDArray, Tuple[float, float, float]] = (.0, .0, .0),
            only: Optional[str] = None
    ):
        self.phonopy = phonon
        self.q = q
        self.structure = phonon.primitive

        # set some masses to an excessive value, to silence those atoms if any
        if only is not None:
            masses = self.structure.masses
            indices = set(x - 1 for x in GetListWithinBounds(1, len(masses))(only))
            not_considered = set(range(masses.shape[0])) - indices
            for i in not_considered:
                masses[i] = HUGE_MASS

            self.structure.masses = masses

        # get eigenvalues and eigenvectors at a given q point
        # See https://github.com/phonopy/phonopy/issues/308#issuecomment-1769736200
        l_logger.info('Symmetrize force constant')
        self.phonopy.symmetrize_force_constants()

        l_logger.info('Fetch dynamical matrix at q=({})'.format(', '.join('{:.3f}'.format(x) for x in q)))
        self.phonopy.dynamical_matrix.run(self.q)
        dm = self.phonopy.dynamical_matrix.dynamical_matrix
        eigv, eigf = numpy.linalg.eigh(dm)

        self.N = self.structure.get_number_of_atoms()
        l_logger.info('Analyze {} modes (including acoustic)'.format(3 * self.N))
        self.frequencies = numpy.sqrt(numpy.abs(eigv.real)) * numpy.sign(eigv.real)
        self.frequencies *= physical_units.DefaultToTHz * physical_units.THzToCm  # in [cm⁻¹]

        if self.frequencies[0] < -30:
            l_logger.warn('The first frequency is very small: {:.3f} cm⁻¹'.format(self.frequencies[0]))

        l_logger.info('The 5 first modes are {}'.format(
            ', '.join('{:.3f}'.format(x) for x in self.frequencies[:5]))
        )

        self.eigenvectors = eigf.real.T  # in  [Å sqrt(AMU)]

        # compute displacements with Eq. 4 of 10.1039/C7CP01680H
        sqrt_masses = numpy.repeat(numpy.sqrt(self.structure.masses), 3)
        self.eigendisps = (self.eigenvectors / sqrt_masses[numpy.newaxis, :]).reshape(-1, self.N, 3)  # in [Å]

        # get irreps
        self.irrep_labels = ['A'] * (self.N * 3)

        try:
            self.phonopy.set_irreps(q)
            self.irreps = phonon.get_irreps()

            # TODO: that's internal API, so subject to change!
            for label, dgset in zip(self.irreps._get_ir_labels(), self.irreps._degenerate_sets):
                for j in dgset:
                    self.irrep_labels[j] = label
        except RuntimeError as e:
            l_logger.warn('Error while computing irreps ({}). Incorrect labels will be assigned.'.format(e))

    @classmethod
    def from_phonopy(
        cls,
        phonopy_yaml: str = 'phonopy_disp.yaml',
        force_constants_filename: str = 'force_constants.hdf5',
        born_filename: Optional[str] = None,
        q: Union[NDArray, Tuple[float, float, float]] = (.0, .0, .0),
        only: Optional[str] = None,
    ) -> 'PhononsAnalyzer':
        """
        Use the Python interface of Phonopy, see https://phonopy.github.io/phonopy/phonopy-module.html.
        """

        l_logger.info('Use `phonopy.load()`')

        return PhononsAnalyzer(phonopy.load(
            phonopy_yaml=phonopy_yaml,
            force_constants_filename=force_constants_filename,
            born_filename=born_filename,
        ), q=q, only=only)

    def infrared_spectrum(self, modes: Optional[List[int]] = None) -> InfraredSpectrum:
        """
        The `modes` is a 0-based list of mode to include.
        If `modes` is None, then all non-acoustic modes are selected.
        """

        l_logger.info('Create IR spectrum object')

        born_tensor = self.phonopy.nac_params['born']

        # select modes if any
        if modes is None:
            modes = list(range(3 if numpy.allclose(self.q, [.0, .0, .0]) else 0, 3 * self.N))
        else:
            for mode in modes:
                if mode < 0 or mode >= 3 * self.N:
                    raise IndexError(mode)

        frequencies = [self.frequencies[m] for m in modes]
        irrep_labels = [self.irrep_labels[m] for m in modes]
        disps = self.eigendisps[numpy.ix_(modes)]

        dmu_dq = numpy.einsum('ijb,jab->ia', disps, born_tensor)

        return InfraredSpectrum(modes=modes, frequencies=frequencies, irrep_labels=irrep_labels, dmu_dq=dmu_dq)

    def prepare_raman(
        self,
        directory: pathlib.Path,
        modes: Optional[List[int]] = None,
        disp: float = 1e-2,
        ref: Optional[str] = None,
        stencil: Optional[list] = None
    ) -> RamanSpectrum:
        """
        Prepare a Raman spectrum by preparing numerical differentiation, and thus creating a set of displaced
        geometries of the unitcell in `directory`.
        Degenerate modes are removed from the numerical differentiation (i.e., only the first mode of
        each degenerate set will be computed).
        The number of geometries that are generated per mode depends on the stencil.

        The `modes` is a 0-based list of mode to include.
        If `modes` is None, then all non-acoustic modes are selected.
        """

        l_logger.info('Create displaced geometries')

        stencil = TWO_POINTS_STENCIL if stencil is None else stencil

        # select modes if any
        if modes is None:
            modes = list(range(3 if numpy.allclose(self.q, [.0, .0, .0]) else 0, 3 * self.N))
        else:
            for mode in modes:
                if mode < 0 or mode >= 3 * self.N:
                    raise IndexError(mode)

        frequencies = [self.frequencies[m] for m in modes]
        irrep_labels = [self.irrep_labels[m] for m in modes]

        # create geometries
        dgsets = {}
        for dgset in self.irreps._degenerate_sets:
            for i in dgset:
                dgsets[i] = dgset[0]

        mode_equiv = []
        mode_calcs = []
        steps = []

        base_geometry = self.structure

        for mode in modes:
            if mode < 0 or mode >= 3 * self.N:
                raise IndexError(mode)

            mode_equiv.append(dgsets[mode])
            if dgsets[mode] != mode:
                continue
            else:
                mode_calcs.append(mode)

            step = disp
            if ref == 'norm':
                step = disp / float(numpy.linalg.norm(self.eigendisps[mode]))

            steps.append(step)

            for i, (value, _) in enumerate(stencil):
                displaced_geometry = base_geometry.copy()
                displaced_geometry.set_positions(
                    base_geometry.positions + value * step * self.eigendisps[mode]
                )

                path = directory / self.DC_GEOMETRY_TEMPLATE.format(mode + 1, i + 1)  # 1-based output
                l_logger.debug('Write displaced geometry for (mode={}, step={}) in `{}`'.format(mode, i, path))

                phonopy_calculator.write_crystal_structure(
                    path,
                    displaced_geometry,
                    interface_mode='vasp'
                )

        l_logger.info('{} geometries created'.format(len(stencil) * len(mode_calcs)))
        l_logger.info('Create Raman spectrum object')

        spectrum = RamanSpectrum(
            # input
            cell_volume=base_geometry.volume,
            modes=modes,
            frequencies=frequencies,
            irrep_labels=irrep_labels,
            # nd:
            nd_stencil=stencil,
            nd_mode_equiv=mode_equiv,
            nd_modes=mode_calcs,
            nd_steps=steps,
        )

        return spectrum

    def make_vesta_for_modes(
        self,
        directory: pathlib.Path,
        modes: Optional[List[int]] = None,
        scaling: float = 2.0,
        radius: float = 0.30,
        color: Tuple[int, int, int] = (0, 255, 0),
        threshold: float = 0.05,
    ):
        """Make a VESTA file for each `modes` (or all except acoustic if `mode` is None) containing a vector for
        each atom, corresponding to the eigenvector.

        Note: this use the primitive cell, which might not be equal to the unit cell, so this might be confusing.
        """

        l_logger.info('Make VESTA files for each mode')

        # select modes if any
        if modes is None:
            modes = list(range(3, 3 * self.N))

        cell = self.structure.cell
        norms = numpy.linalg.norm(cell, axis=1)
        cart_to_cell = numpy.linalg.inv(cell)

        eigv = self.eigenvectors.reshape(-1, self.N, 3)
        eigv_norms = numpy.linalg.norm(eigv, axis=2)

        for mode in modes:
            if mode < 0 or mode >= 3 * self.N:
                raise IndexError(mode)

            l_logger.info('Creating file for mode {}'.format(mode))

            max_norm = eigv_norms[mode].max()

            # convert to coordinates along cell vectors
            ceignedisps = numpy.einsum('ij,jk->ik', eigv[mode], cart_to_cell)
            ceigendisps = ceignedisps[:] * norms * scaling

            vectors = [
                VestaVector(True, ceigendisps[i], [i], radius=radius, through_atom=True, color=color, add_radius=True)
                for i in range(self.N) if eigv_norms[mode][i] > threshold * max_norm
            ]

            with (directory / self.VESTA_MODE_TEMPLATE.format(mode + 1)).open('w') as f:
                make_vesta_file(
                    f,
                    self.structure,
                    vectors,
                    title='Mode {} ({:.3f} cm⁻¹, {})'.format(mode + 1, self.frequencies[mode], self.irrep_labels[mode])
                )
