import pytest

from phonopy_vibspec.phonopy_results import PhonopyResults


def test_infrared_SiO2(context_SiO2):
    results = PhonopyResults.from_phonopy(
        phonopy_yaml='phonopy_disp.yaml',
        force_constants_filename='force_constants.hdf5',
        born_filename='BORN'
    )

    # SiO2 is D3, so A2 and E mode should be active, A1 should not!
    ir_intensities = results.infrared_intensities()[3:]

    for i, label in enumerate(results.irrep_labels[3:]):
        if label in ['A2', 'E']:
            assert ir_intensities[i] != pytest.approx(.0, abs=1e-8)
        else:
            assert ir_intensities[i] == pytest.approx(.0, abs=1e-8)
