import pathlib
import numpy

import phonopy
from phonopy.interface import calculator as phonopy_calculator

from typing import Optional, List, Tuple

from phonopy_vibspec import logger
from phonopy_vibspec.spectra import RamanSpectrum, InfraredSpectrum
from phonopy_vibspec.vesta import VestaVector, make_vesta_file

THZ_TO_INV_CM = 33.35641

# [(value, coef), ...]
# see https://en.wikipedia.org/wiki/Finite_difference_coefficient
TWO_POINTS_STENCIL = [(-1, -.5), (1, .5)]  # two-points, centered


l_logger = logger.getChild(__name__)


class PhononsAnalyzer:
    """Use Phonopy to extract phonon frequencies and eigenmodes, as well as irreps
    """

    DC_GEOMETRY_TEMPLATE = 'dielec_mode{:04d}_step{:02d}.vasp'
    VESTA_MODE_TEMPLATE = 'mode{:04d}.vesta'

    def __init__(self, phonon: phonopy.Phonopy):
        self.phonotopy = phonon
        self.structure = phonon.primitive

        # get eigenvalues and eigenvectors at gamma point
        # See https://github.com/phonopy/phonopy/issues/308#issuecomment-1769736200
        l_logger.info('Symmetrize force constant')
        self.phonotopy.symmetrize_force_constants()

        l_logger.info('Run mesh')
        self.phonotopy.run_mesh([1, 1, 1], with_eigenvectors=True)

        mesh_dict = phonon.get_mesh_dict()

        self.N = self.structure.get_number_of_atoms()
        l_logger.info('Analyze {} modes (including acoustic)'.format(3 * self.N))
        self.frequencies = mesh_dict['frequencies'][0] * THZ_TO_INV_CM  # in [cm⁻¹]

        l_logger.info('The 5 first modes are {}'.format(
            ', '.join('{:.3f}'.format(x) for x in self.frequencies[:5]))
        )

        self.eigenvectors = mesh_dict['eigenvectors'][0].real.T  # in  [Å sqrt(AMU)]

        # compute displacements with Eq. 4 of 10.1039/C7CP01680H
        sqrt_masses = numpy.repeat(numpy.sqrt(self.structure.masses), 3)
        self.eigendisps = (self.eigenvectors / sqrt_masses[numpy.newaxis, :]).reshape(-1, self.N, 3)  # in [Å]

        # get irreps
        self.irrep_labels = ['A'] * (self.N * 3)

        try:
            self.phonotopy.set_irreps([0, 0, 0])
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
        born_filename: Optional[str] = None
    ) -> 'PhononsAnalyzer':
        """
        Use the Python interface of Phonopy, see https://phonopy.github.io/phonopy/phonopy-module.html.
        """

        l_logger.info('Use `phonopy.load()`')

        return PhononsAnalyzer(phonopy.load(
            phonopy_yaml=phonopy_yaml,
            force_constants_filename=force_constants_filename,
            born_filename=born_filename,
        ))

    def infrared_spectrum(self, modes: Optional[List[int]] = None) -> InfraredSpectrum:
        """
        The `modes` is a 0-based list of mode to include.
        If `modes` is None, then all non-acoustic modes are selected.
        """

        l_logger.info('Create IR spectrum object')

        born_tensor = self.phonotopy.nac_params['born']

        # select modes if any
        if modes is None:
            modes = list(range(3, 3 * self.N))
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
            modes = list(range(3, 3 * self.N))

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
        radius: float = 0.15,
        color: Tuple[int, int, int] = (0, 255, 0)
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

        for mode in modes:
            if mode < 0 or mode >= 3 * self.N:
                raise IndexError(mode)

            l_logger.info('Creating file for mode {}'.format(mode))

            # convert to coordinates along cell vectors
            ceignedisps = numpy.einsum('ij,jk->ik', eigv[mode], cart_to_cell)
            ceigendisps = ceignedisps[:] * norms * scaling

            vectors = [
                VestaVector(True, ceigendisps[i], [i], radius=radius, through_atom=True, color=color)
                for i in range(self.N)
            ]

            with (directory / self.VESTA_MODE_TEMPLATE.format(mode + 1)).open('w') as f:
                make_vesta_file(
                    f,
                    self.structure,
                    vectors,
                    title='Mode {} ({:.3f} cm⁻¹, {})'.format(mode + 1, self.frequencies[mode], self.irrep_labels[mode])
                )
