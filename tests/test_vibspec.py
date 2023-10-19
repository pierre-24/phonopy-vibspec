import numpy
import pytest
import yaml

from phonopy_vibspec.phonons_analyzer import PhononsAnalyzer, TWO_POINTS_STENCIL


def test_infrared_SiO2(context_SiO2):
    phonons = PhononsAnalyzer.from_phonopy(
        phonopy_yaml='phonopy_disp.yaml',
        force_constants_filename='force_constants.hdf5',
        born_filename='BORN'
    )

    # compute intensities
    freqs = phonons.frequencies
    ir_intensities = phonons.infrared_intensities()

    # check that degenerate modes share the same frequencies and intensities
    for dgset in (phonons.irreps._degenerate_sets):
        if len(dgset) > 1:
            assert freqs[dgset[0]] == pytest.approx(freqs[dgset[1]], abs=1e-3)
            assert ir_intensities[dgset[0]] == pytest.approx(ir_intensities[dgset[1]], abs=1e-3)

    # SiO2 is D3, so A2 and E mode should be active, A1 should not!
    for i, label in enumerate(phonons.irrep_labels):
        if label in ['A2', 'E']:
            assert ir_intensities[i] != pytest.approx(.0, abs=1e-8)
        else:
            assert ir_intensities[i] == pytest.approx(.0, abs=1e-8)


def test_create_displaced_geometries_SiO2(context_SiO2, tmp_path):
    phonons = PhononsAnalyzer.from_phonopy(
        phonopy_yaml='phonopy_disp.yaml',
        force_constants_filename='force_constants.hdf5',
        born_filename='BORN'
    )

    phonons.create_displaced_geometries(tmp_path, ref='norm')

    # check info:
    with (tmp_path / 'raman_disps.yml').open() as f:
        raman_disps = yaml.load(f, Loader=yaml.Loader)

    assert raman_disps['stencil'] == TWO_POINTS_STENCIL

    for mode_disp in raman_disps['modes']:
        assert mode_disp['step'] == pytest.approx(0.01 / numpy.linalg.norm(phonons.eigendisps[mode_disp['mode']]))


def test_create_displaced_geometries_select_modes_SiO2(context_SiO2, tmp_path):
    requested_modes = [3, 5, 6]

    phonons = PhononsAnalyzer.from_phonopy(
        phonopy_yaml='phonopy_disp.yaml',
        force_constants_filename='force_constants.hdf5',
        born_filename='BORN'
    )

    phonons.create_displaced_geometries(tmp_path, modes=requested_modes)

    with (tmp_path / 'raman_disps.yml').open() as f:
        raman_disps = yaml.load(f, Loader=yaml.Loader)

    # check only selected modes has been created
    assert len(raman_disps['modes']) == len(requested_modes)

    for mode, mode_disp in zip(requested_modes, raman_disps['modes']):
        assert mode == mode_disp['mode']
        assert mode_disp['step'] == pytest.approx(0.01)
