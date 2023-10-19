import pathlib
import numpy

import phonopy
from phonopy.interface import calculator as phonopy_calculator

from numpy.typing import NDArray
from typing import Optional, List

from phonopy_vibspec.spectra import RamanSpectrum, InfraredSpectrum

THZ_TO_INV_CM = 33.35641

# [(value, coef), ...]
# see https://en.wikipedia.org/wiki/Finite_difference_coefficient
TWO_POINTS_STENCIL = [(-1, -.5), (1, .5)]  # two-points, centered


class PhononsAnalyzer:
    def __init__(self, phonon: phonopy.Phonopy):
        self.phonotopy = phonon
        self.supercell = phonon.supercell

        # get eigenvalues and eigenvectors at gamma point
        # See https://github.com/phonopy/phonopy/issues/308#issuecomment-1769736200
        self.phonotopy.symmetrize_force_constants()
        self.phonotopy.run_mesh([1, 1, 1], with_eigenvectors=True)

        mesh_dict = phonon.get_mesh_dict()

        self.frequencies = mesh_dict['frequencies'][0] * THZ_TO_INV_CM  # in [cm⁻¹]
        self.N = self.supercell.get_number_of_atoms()
        self.eigenvectors = mesh_dict['eigenvectors'][0].real.T  # in  [Å sqrt(AMU)]

        # compute displacements with Eq. 4 of 10.1039/C7CP01680H
        sqrt_masses = numpy.repeat(numpy.sqrt(self.supercell.masses), 3)
        self.eigendisps = (self.eigenvectors / sqrt_masses[numpy.newaxis, :]).reshape(-1, self.N, 3)  # in [Å]

        # get irreps
        self.phonotopy.set_irreps([0, 0, 0])
        self.irreps = phonon.get_irreps()
        self.irrep_labels = [''] * (self.N * 3)

        # TODO: that's internal API, so subject to change!
        for label, dgset in zip(self.irreps._get_ir_labels(), self.irreps._degenerate_sets):
            for j in dgset:
                self.irrep_labels[j] = label

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
        born_tensor = self.phonotopy.nac_params['born']

        # select modes if any
        if modes is None:
            modes = list(range(3, 3 * self.N))

        frequencies = [self.frequencies[m] for m in modes]
        irrep_labels = [self.irrep_labels[m] for m in modes]
        disps = self.eigendisps[numpy.ix_(modes)]

        dmu_dq = numpy.einsum('ijb,jab->ia', disps, born_tensor)

        return InfraredSpectrum(modes=modes, frequencies=frequencies, irrep_labels=irrep_labels, dmu_dq=dmu_dq)

    def infrared_intensities(self, selected_modes: Optional[List[int]] = None) -> NDArray[float]:
        """Compute the infrared intensities with Eq. 7 of 10.1039/C7CP01680H.
        Intensities are given in [e²/AMU].
        """

        born_tensor = self.phonotopy.nac_params['born']

        # select modes if any
        disps = self.eigendisps
        if selected_modes:
            disps = self.eigendisps[numpy.ix_(selected_modes)]

        dipoles = numpy.einsum('ijb,jab->ia', disps, born_tensor)
        return (dipoles ** 2).sum(axis=1)

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

        base_geometry = self.phonotopy.unitcell

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

                phonopy_calculator.write_crystal_structure(
                    directory / 'unitcell_{:04d}_{:02d}.vasp'.format(mode + 1, i + 1),  # 1-based output
                    displaced_geometry,
                    interface_mode='vasp'
                )

        calculator = RamanSpectrum(
            # input
            modes=modes,
            frequencies=frequencies,
            irrep_labels=irrep_labels,
            # nd:
            nd_stencil=stencil,
            nd_mode_equiv=mode_equiv,
            nd_modes=mode_calcs,
            nd_steps=steps,
        )

        return calculator
