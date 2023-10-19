import pathlib
import yaml
import numpy
import phonopy
from phonopy.interface import calculator

from numpy.typing import NDArray
from typing import Optional, List

THZ_TO_INV_CM = 33.35641

# [(value, coef), ...]
# see https://en.wikipedia.org/wiki/Finite_difference_coefficient
TWO_POINTS_STENCIL = [[-1, -.5], [1, .5]]  # two-points, centered


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
        self.irrep_labels = [None] * (self.N * 3)

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

    def create_displaced_geometries(
        self,
        directory: pathlib.Path,
        disp: float = 1e-2,
        modes: Optional[List[int]] = None,
        ref: Optional[str] = None,
        stencil: List[List[float]] = TWO_POINTS_STENCIL
    ):
        """
        Create a set of displaced geometries of the unitcell in `directory`.
        The number of geometries that are generated depends on the stencil.

        The `modes` is a 0-based list of mode.
        If `mode` is `None`,  all mode are selected, except the acoustic ones, and only one version of degenerated ones.
        """

        # select modes
        if modes is None:
            modes = []
            for dgset in self.irreps._degenerate_sets:
                if any(x < 3 for x in dgset):  # skip acoustic phonons
                    continue

                modes.append(dgset[0])

        # create geometries
        base_geometry = self.phonotopy.unitcell
        raman_disps_info = []

        for mode in modes:
            if mode < 0 or mode >= 3 * self.N:
                raise IndexError(mode)

            step = disp
            if ref == 'norm':
                step = disp / float(numpy.linalg.norm(self.eigendisps[mode]))

            raman_disps_info.append({'mode': mode, 'step': step})

            for i, (value, _) in enumerate(stencil):
                displaced_geometry = base_geometry.copy()
                displaced_geometry.set_positions(
                    base_geometry.positions + value * step * self.eigendisps[mode]
                )

                calculator.write_crystal_structure(
                    directory / 'unitcell_{:04d}_{:02d}.vasp'.format(mode + 1, i + 1),  # 1-based output
                    displaced_geometry,
                    interface_mode='vasp'
                )

        with (directory / 'raman_disps.yml').open('w') as f:
            yaml.dump({'stencil': stencil, 'modes': raman_disps_info}, f)
