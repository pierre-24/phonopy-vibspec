import numpy
import phonopy

from numpy.typing import NDArray
from typing import Optional, List


THZ_TO_INV_CM = 33.35641


class PhonopyResults:
    def __init__(self, phonon: phonopy.Phonopy):
        self.phonon = phonon
        self.supercell = phonon.supercell

        # get eigenvalues and eigenvectors at gamma point
        # See https://github.com/phonopy/phonopy/issues/308#issuecomment-1769736200
        self.phonon.symmetrize_force_constants()
        self.phonon.run_mesh([1, 1, 1], with_eigenvectors=True)

        mesh_dict = phonon.get_mesh_dict()

        self.frequencies = mesh_dict['frequencies'][0] * THZ_TO_INV_CM  # in [cm⁻¹]
        self.N = self.supercell.get_number_of_atoms()
        self.eigenvectors = mesh_dict['eigenvectors'][0].real.T  # in  [Å sqrt(AMU)]

        # compute displacements with Eq. 4 of 10.1039/C7CP01680H
        sqrt_masses = numpy.repeat(numpy.sqrt(self.supercell.masses), 3)
        self.eigendisps = (self.eigenvectors / sqrt_masses[numpy.newaxis, :]).reshape(-1, self.N, 3)  # in [Å]

        # get irreps
        self.phonon.set_irreps([0, 0, 0])
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
        born_filename: str = 'BORN'
    ) -> 'PhonopyResults':
        """
        Use the Python interface of Phonopy, see https://phonopy.github.io/phonopy/phonopy-module.html.
        """

        return PhonopyResults(phonopy.load(
            phonopy_yaml=phonopy_yaml,
            force_constants_filename=force_constants_filename,
            born_filename=born_filename,
        ))

    def infrared_intensities(self, selected_modes: Optional[List[int]] = None) -> NDArray[float]:
        """Compute the infrared intensities with Eq. 7 of 10.1039/C7CP01680H.
        Intensities are given in [e²/AMU].
        """

        born_tensor = self.phonon.nac_params['born']

        # select modes if any
        disps = self.eigendisps
        if selected_modes:
            disps = self.eigendisps[numpy.ix_(selected_modes)]

        dipoles = numpy.einsum('ijb,jab->ia', disps, born_tensor)
        return (dipoles ** 2).sum(axis=1)
