import pathlib
import os

from phonopy_vibspec.phonopy_results import PhonopyResults

THZ_TO_INV_CM = 33.35641


def test_phonopy():
    dir_ = pathlib.Path(__file__).parent / 'tests_files/SiO2'
    os.chdir(dir_)

    phonon = PhonopyResults.from_phonopy()
    print(phonon.frequencies * THZ_TO_INV_CM)
    print(phonon.infrared_intensities())
