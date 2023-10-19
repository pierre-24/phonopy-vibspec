import pathlib

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
    spectrum = phonons.infrared_spectrum()
    ir_intensities = spectrum.compute_intensities()

    assert len(spectrum.modes) == 24  # skip acoustic
    assert numpy.allclose(spectrum.frequencies, phonons.frequencies[3:])
    assert spectrum.irrep_labels == phonons.irrep_labels[3:]

    # check that degenerate modes share the same intensities
    for dgset in (phonons.irreps._degenerate_sets):
        if dgset[0] < 2:  # skip acoustic
            continue

        if len(dgset) > 1:
            assert ir_intensities[dgset[0] - 3] == pytest.approx(ir_intensities[dgset[1] - 3], abs=1e-3)

    # SiO2 is D3, so A2 and E mode should be active, A1 should not!
    for i, label in enumerate(spectrum.irrep_labels):
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

    spectrum = phonons.prepare_raman(tmp_path)

    assert spectrum.cell_volume == phonons.phonotopy.unitcell.volume
    assert len(spectrum.modes) == 24  # skip acoustic
    assert numpy.allclose(spectrum.frequencies, phonons.frequencies[3:])
    assert spectrum.irrep_labels == phonons.irrep_labels[3:]

    assert len(spectrum.nd_modes) == 16  # remove degenerates
    assert spectrum.nd_stencil == TWO_POINTS_STENCIL
    assert spectrum.nd_steps == [1e-2] * len(spectrum.nd_modes)

    # check if files have been created
    for i in range(len(spectrum)):
        mode = spectrum.modes[i]
        f = tmp_path / 'unitcell_{:04d}_{:02d}.vasp'.format(mode + 1, 1)
        if spectrum.nd_mode_equiv[i] != mode:
            assert not f.exists()
        else:
            assert f.exists()


def test_prepare_raman_select_modes_SiO2(context_SiO2, tmp_path):
    requested_modes = [3, 5, 6]

    phonons = PhononsAnalyzer.from_phonopy(
        phonopy_yaml='phonopy_disp.yaml',
        force_constants_filename='force_constants.hdf5'
    )

    spectrum = phonons.prepare_raman(tmp_path, modes=requested_modes, ref='norm')

    assert spectrum.modes == requested_modes

    for i in range(len(spectrum.nd_modes)):
        assert spectrum.nd_steps[i] == pytest.approx(
            0.01 / numpy.linalg.norm(phonons.eigendisps[requested_modes[i]])
        )


def test_raman_spectrum_save_SiO2(context_SiO2, tmp_path):
    phonons = PhononsAnalyzer.from_phonopy(
        phonopy_yaml='phonopy_disp.yaml',
        force_constants_filename='force_constants.hdf5'
    )

    spectrum = phonons.prepare_raman(tmp_path, ref='norm')
    spectrum.to_hdf5(tmp_path / 'raman.hdf5')

    spectrum_read = RamanSpectrum.from_hdf5(tmp_path / 'raman.hdf5')

    assert spectrum.cell_volume == spectrum_read.cell_volume
    assert numpy.allclose(spectrum.modes, spectrum_read.modes)
    assert numpy.allclose(spectrum.frequencies, spectrum_read.frequencies)
    assert numpy.allclose(spectrum.modes, spectrum_read.modes)

    assert numpy.allclose(spectrum.nd_stencil, spectrum_read.nd_stencil)
    assert numpy.allclose(spectrum.nd_mode_equiv, spectrum_read.nd_mode_equiv)
    assert numpy.allclose(spectrum.nd_modes, spectrum_read.nd_modes)
    assert numpy.allclose(spectrum.nd_steps, spectrum_read.nd_steps)


def test_raman_spectrum_extract_dielectrics(context_SiO2):
    calc_directory = pathlib.Path.cwd() / 'calc_dielectrics'
    spectrum = RamanSpectrum.from_hdf5(pathlib.Path(calc_directory / 'raman.hdf5'))

    nsteps = len(spectrum.nd_stencil)
    files = []
    for mode in spectrum.nd_modes:
        for i in range(nsteps):
            files.append(calc_directory / 'unitcell_{:04d}_{:02d}'.format(mode + 1, i + 1) / 'vasprun.xml')

    spectrum.extract_dielectrics(files)

    assert spectrum.dielectrics is not None
    assert spectrum.dalpha_dq is not None
