import numpy
import phonopy

from typing import Optional, List


class PhonopyResults:
    def __init__(self, phonon: phonopy.Phonopy):
        self.phonon = phonon
        self.supercell = phonon.supercell

        # get eigenvalues and eigenvectors at gamma point
        self.phonon.run_mesh([1, 1, 1], with_eigenvectors=True)
        mesh_dict = phonon.get_mesh_dict()

        self.frequencies = mesh_dict['frequencies']  # in THz (!!)
        self.N = self.supercell.get_number_of_atoms()
        self.eigenvectors = mesh_dict['eigenvectors'][0].real  # in Å * sqrt(AMU)

        # Eq. 4 of 10.1039/C7CP01680H
        sqrt_masses = numpy.repeat(numpy.sqrt(self.supercell.masses), 3)
        self.eigendisps = (self.eigenvectors / sqrt_masses[numpy.newaxis, :]).reshape(-1, self.N, 3)  # (3N, N, 3), in Å

    @classmethod
    def from_phonopy(
        cls,
        phonopy_yaml: str = 'phonopy.yaml',
        force_constants_filename: str = 'force_constants.hdf5',
        born_filename: str = 'BORN'
    ) -> 'PhonopyResults':
        """
        Use the Python interface of Phonopy, see https://phonopy.github.io/phonopy/phonopy-module.html.
        """

        return PhonopyResults(phonopy.load(
            phonopy_yaml=phonopy_yaml,
            force_constants_filename=force_constants_filename,
            born_filename=born_filename
        ))

    def infrared_intensities(self, selected_modes: Optional[List[int]] = None):
        """Compute the infrared intensities as Eq. 7 of 10.1039/C7CP01680H
        """

        born_tensor = self.phonon.nac_params['born']

        # select modes if any
        disps = self.eigendisps
        if selected_modes:
            disps = self.eigendisps[selected_modes]

        for i in range(3 * self.N):
            iri = .0
            for a in range(3):
                sum_temp1 = .0
                for j in range(self.N):
                    sum_temp2 = .0
                    for b in range(3):
                        sum_temp2 += born_tensor[j, a, b] * disps[i, j, b]
                    sum_temp1 += sum_temp2

                iri += sum_temp1 ** 2

            print(i, iri)

        dipoles = numpy.einsum('ijb,jab->ia', disps, born_tensor)
        return (dipoles ** 2).sum(axis=1)
