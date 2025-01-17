"""
Create an IR spectrum in CSV form
"""

import argparse

from phonopy_vibspec import GetListWithinBounds
from phonopy_vibspec.phonons_analyzer import PhononsAnalyzer
from phonopy_vibspec.scripts import add_common_args, add_common_args_spectra


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    add_common_args_spectra(parser)

    parser.add_argument('-o', '--output', help='Output HDF5 file', default='ir.hdf5')
    parser.add_argument('-b', '--born', help='BORN file', default='BORN')

    args = parser.parse_args()

    phonons = PhononsAnalyzer.from_phonopy(
        phonopy_yaml=args.phonopy,
        force_constants_filename=args.fc,
        born_filename=args.born,
        q=args.q,
        only=args.only if args.only != '' else None
    )

    ir_spectrum = phonons.infrared_spectrum(
        [
            x - 1 for x in GetListWithinBounds(1, 3 * phonons.N)(args.modes)
        ] if len(args.modes) > 0 else None)

    ir_spectrum.to_csv(args.csv, [args.linewidth] * len(ir_spectrum), args.limits, args.each)

    ir_spectrum.to_hdf5(args.output)


if __name__ == '__main__':
    main()
