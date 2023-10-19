"""
Create displaced geometries for Raman activity calculations
"""

import argparse
import pathlib

from phonopy_vibspec.phonopy_phonons_analyzer import PhonopyPhononsAnalyzer
from phonopy_vibspec.scripts import add_common_args


def valid_dir(inp: str) -> pathlib.Path:
    path = pathlib.Path(inp)
    if not path.is_dir():
        raise argparse.ArgumentTypeError('`{}` is not a valid directory'.format(inp))

    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)

    parser.add_argument('-d', '--displacement', type=float, help='Step size', default=1e-2)
    parser.add_argument('-r', '--ref', choices=['norm', 'none'], help='Reference for the step', default='none')
    parser.add_argument('-o', '--output', type=valid_dir, help='Output directory', default='.')

    args = parser.parse_args()

    results = PhonopyPhononsAnalyzer.from_phonopy(
        phonopy_yaml=args.phonopy,
        force_constants_filename=args.fc,
    )

    results.create_displaced_geometries(pathlib.Path('.'), disp=args.displacement, ref=args.ref)


if __name__ == '__main__':
    main()
