"""
Create a VESTA file containing the eigenvector for each mode
"""

import argparse
import pathlib

from typing import Tuple

from phonopy_vibspec.phonons_analyzer import PhononsAnalyzer
from phonopy_vibspec.scripts import add_common_args


def get_color(inp: str) -> Tuple[int, int, int]:
    e = 'invalid color: `{}`, must be 3 integers'.format(inp)

    try:
        cs = [int(x) for x in inp.split()]
    except ValueError:
        raise argparse.ArgumentTypeError(e)

    if len(cs) != 3:
        raise argparse.ArgumentTypeError(e)

    if any(x < 0 or x > 255 for x in cs):
        raise argparse.ArgumentTypeError(e)

    return tuple(cs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)

    parser.add_argument('-s', '--scaling', help='Scaling factor', type=float, default=2.0)
    parser.add_argument('-r', '--radius', help='Radius of the vectors', type=float, default=0.30)
    parser.add_argument('--color', help='Color of the vectors', type=get_color, default='0 255 0')
    parser.add_argument(
        '--threshold',
        help='Remove vectors that are below a certain percentage of the largest one',
        type=float,
        default=0.05
    )

    args = parser.parse_args()

    phonons = PhononsAnalyzer.from_phonopy(
        phonopy_yaml=args.phonopy,
        force_constants_filename=args.fc,
        only=args.only if args.only != '' else None
    )

    phonons.make_vesta_for_modes(
        pathlib.Path.cwd(),
        modes=args.modes if len(args.modes) > 0 else None,
        scaling=args.scaling,
        radius=args.radius,
        color=args.color,
        threshold=args.threshold
    )


if __name__ == '__main__':
    main()
