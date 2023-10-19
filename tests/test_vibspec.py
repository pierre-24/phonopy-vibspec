import numpy
import pytest

from phonopy_vibspec.phonons_analyzer import PhononsAnalyzer, TWO_POINTS_STENCIL
from phonopy_vibspec.spectra import InfraredSpectrum, RamanSpectrum


def test_infrared_SiO2(context_SiO2):
    phonons = PhononsAnalyzer.from_phonopy(
        phonopy_yaml='phonopy_disp.yaml',
        force_constants_filename='force_constants.hdf5',
        born_filename='BORN'
    )

    # get spectrum
    infrared_spectrum = phonons.infrared_spectrum()
    ir_intensities = infrared_spectrum.compute_intensities()

    assert len(infrared_spectrum.modes) == 24  # skip acoustic
    assert numpy.allclose(infrared_spectrum.frequencies, phonons.frequencies[3:])

    # check that degenerate modes share the same intensities
    for dgset in (phonons.irreps._degenerate_sets):
        if dgset[0] < 2:  # skip acoustic
            continue

        if len(dgset) > 1:
            assert ir_intensities[dgset[0] - 3] == pytest.approx(ir_intensities[dgset[1] - 3], abs=1e-3)

    # SiO2 is D3, so A2 and E mode should be active, A1 should not!
    for i, label in enumerate(infrared_spectrum.irrep_labels):
        if label in ['A2', 'E']:
            assert ir_intensities[i] != pytest.approx(.0, abs=1e-8)
        else:
            assert ir_intensities[i] == pytest.approx(.0, abs=1e-8)


def test_infrared_spectrum_save_SiO2(context_SiO2, tmp_path):
    phonons = PhononsAnalyzer.from_phonopy(
        phonopy_yaml='phonopy_disp.yaml',
        force_constants_filename='force_constants.hdf5',
        born_filename='BORN'
    )

    # get spectrum
    spectrum = phonons.infrared_spectrum()

    # save
    spectrum.to_hdf5(tmp_path / 'ir.hdf5')

    # read
    spectrum_read = InfraredSpectrum.from_hdf5(tmp_path / 'ir.hdf5')

    assert numpy.allclose(spectrum.modes, spectrum_read.modes)
    assert numpy.allclose(spectrum.frequencies, spectrum_read.frequencies)
    assert numpy.allclose(spectrum.modes, spectrum_read.modes)
    assert numpy.allclose(spectrum.dmu_dq, spectrum_read.dmu_dq)


def test_prepare_raman_SiO2(context_SiO2, tmp_path):
    phonons = PhononsAnalyzer.from_phonopy(
        phonopy_yaml='phonopy_disp.yaml',
        force_constants_filename='force_constants.hdf5'
    )

    spectrum = phonons.prepare_raman(tmp_path, ref='norm')
    assert len(spectrum.modes) == 24
    assert spectrum.stencil == TWO_POINTS_STENCIL

    for i in range(len(spectrum.modes)):
        assert spectrum.steps[i] == pytest.approx(
            0.01 / numpy.linalg.norm(phonons.eigendisps[spectrum.modes[i]])
        )


def test_prepare_raman_select_modes_SiO2(context_SiO2, tmp_path):
    requested_modes = [3, 5, 6]

    phonons = PhononsAnalyzer.from_phonopy(
        phonopy_yaml='phonopy_disp.yaml',
        force_constants_filename='force_constants.hdf5'
    )

    spectrum = phonons.prepare_raman(tmp_path, modes=requested_modes)

    assert spectrum.modes == requested_modes
    assert spectrum.steps == [1e-2] * len(requested_modes)


def test_raman_spectrum_save_SiO2(context_SiO2, tmp_path):
    phonons = PhononsAnalyzer.from_phonopy(
        phonopy_yaml='phonopy_disp.yaml',
        force_constants_filename='force_constants.hdf5'
    )

    spectrum = phonons.prepare_raman(tmp_path, ref='norm')
    spectrum.to_hdf5(tmp_path / 'raman.hdf5')

    spectrum_read = RamanSpectrum.from_hdf5(tmp_path / 'raman.hdf5')

    assert numpy.allclose(spectrum.modes, spectrum_read.modes)
    assert numpy.allclose(spectrum.frequencies, spectrum_read.frequencies)
    assert numpy.allclose(spectrum.modes, spectrum_read.modes)

    assert numpy.allclose(spectrum.stencil, spectrum_read.stencil)
    assert numpy.allclose(spectrum.steps, spectrum_read.steps)
