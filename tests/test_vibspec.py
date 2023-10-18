import pytest

from phonopy_vibspec.phonopy_results import PhonopyResults


def test_infrared_SiO2(context_SiO2):
    results = PhonopyResults.from_phonopy(
        phonopy_yaml='phonopy_disp.yaml',
        force_constants_filename='force_constants.hdf5',
        born_filename='BORN'
    )

    # compute intensities
    freqs = results.frequencies
    ir_intensities = results.infrared_intensities()

    # check that degenerate modes share the same frequencies and intensities
    for dgset in (results.irreps._degenerate_sets):
        if len(dgset) > 1:
            assert freqs[dgset[0]] == pytest.approx(freqs[dgset[1]], abs=1e-3)
            assert ir_intensities[dgset[0]] == pytest.approx(ir_intensities[dgset[1]], abs=1e-3)

    # SiO2 is D3, so A2 and E mode should be active, A1 should not!
    for i, label in enumerate(results.irrep_labels):
        if label in ['A2', 'E']:
            assert ir_intensities[i] != pytest.approx(.0, abs=1e-8)
        else:
            assert ir_intensities[i] == pytest.approx(.0, abs=1e-8)
