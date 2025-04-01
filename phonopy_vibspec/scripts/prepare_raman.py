"""
Create displaced geometries for Raman activity calculations
"""

import argparse
import pathlib

from phonopy_vibspec import GetListWithinBounds
from phonopy_vibspec.phonons_analyzer import PhononsAnalyzer
from phonopy_vibspec.scripts import add_common_args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)

    parser.add_argument('-o', '--output', help='Output HDF5 file', default='raman.hdf5')

    parser.add_argument('-d', '--displacement', type=float, help='Step size', default=1e-2)
    parser.add_argument('-r', '--ref', choices=['norm', 'none'], help='Reference for the step', default='none')

    args = parser.parse_args()

    phonons = PhononsAnalyzer.from_phonopy(
        phonopy_yaml=args.phonopy,
        force_constants_filename=args.fc,
        only=args.only if args.only != '' else None
    )

    raman_spectrum = phonons.prepare_raman(
        pathlib.Path.cwd(),
        disp=args.displacement,
        ref=args.ref,
        modes=[
            x - 1 for x in GetListWithinBounds(1, 3 * phonons.N)(args.modes)
        ] if len(args.modes) > 0 else None
    )

    raman_spectrum.to_hdf5(args.output)


if __name__ == '__main__':
    main()
