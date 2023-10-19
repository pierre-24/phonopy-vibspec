"""
Create a Raman spectrum in CSV form
"""

import argparse

from phonopy_vibspec.spectra import RamanSpectrum
from phonopy_vibspec.scripts import add_common_args_spectra


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    add_common_args_spectra(parser)
    parser.add_argument('-d', '--data', help='Raman HDF5 file', default='raman.hdf5')

    args = parser.parse_args()

    raman_spectrum = RamanSpectrum.from_hdf5(args.data)

    if raman_spectrum.dalpha_dq is None:
        raise Exception('not dα/dq found in this file, did you forget to gather Raman activities first?')

    raman_spectrum.to_csv(args.csv, [args.linewidth] * len(raman_spectrum), args.limits, args.each)


if __name__ == '__main__':
    main()
