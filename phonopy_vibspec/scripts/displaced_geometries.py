"""
Create displaced geometries for Raman activity calculations
"""

import argparse
import pathlib

from typing import List

from phonopy_vibspec.phonons_analyzer import PhononsAnalyzer
from phonopy_vibspec.scripts import add_common_args


def valid_dir(inp: str) -> pathlib.Path:
    path = pathlib.Path(inp)
    if not path.is_dir():
        raise argparse.ArgumentTypeError('`{}` is not a valid directory'.format(inp))

    return path


def list_of_modes(inp: str) -> List[int]:
    try:
        return [int(x) - 1 for x in inp.split()]
    except ValueError:
        raise argparse.ArgumentTypeError('invalid (space-separated) list of integers `{}` for modes'.format(inp))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)

    parser.add_argument('-d', '--displacement', type=float, help='Step size', default=1e-2)
    parser.add_argument('-r', '--ref', choices=['norm', 'none'], help='Reference for the step', default='none')
    parser.add_argument('-m', '--modes', type=list_of_modes, help='List of modes (1-based)', default='')
    parser.add_argument('-o', '--output', type=valid_dir, help='Output directory', default='.')

    args = parser.parse_args()

    phonons = PhononsAnalyzer.from_phonopy(
        phonopy_yaml=args.phonopy,
        force_constants_filename=args.fc,
    )

    phonons.create_displaced_geometries(
        pathlib.Path('.'),
        disp=args.displacement,
        ref=args.ref,
        modes=args.modes if len(args.modes) > 0 else None
    )


if __name__ == '__main__':
    main()
