"""
Gather dielectric tensors, and compute Raman activities
"""

import argparse
import pathlib

from phonopy_vibspec.spectra import RamanSpectrum


def files_list(inp):
    p = pathlib.Path(inp)
    if not p.exists():
        raise argparse.ArgumentTypeError('file `{}` does not exists'.format(inp))

    return p


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('-d', '--data', help='Raman HDF5 file', default='raman.hdf5')
    parser.add_argument(
        'files', nargs='*', type=files_list, help='vasprun.xml for all displaced geometries, in the correct order')

    args = parser.parse_args()

    # open & extract
    raman_spectrum = RamanSpectrum.from_hdf5(args.data)
    raman_spectrum.extract_dielectrics(args.files)

    # update
    raman_spectrum.to_hdf5(args.data)


if __name__ == '__main__':
    main()
