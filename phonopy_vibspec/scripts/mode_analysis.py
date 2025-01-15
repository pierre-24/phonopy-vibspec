"""
Create a VESTA file containing the eigenvector for each mode
"""

import argparse
import numpy

from phonopy_vibspec.phonons_analyzer import PhononsAnalyzer
from phonopy_vibspec.scripts import add_common_args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)

    args = parser.parse_args()

    phonons = PhononsAnalyzer.from_phonopy(
        phonopy_yaml=args.phonopy,
        force_constants_filename=args.fc,
        only=args.only if args.only != '' else None
    )

    nvecx = numpy.tile([1., 0, 0], phonons.N) / numpy.sqrt(phonons.N)
    nvecy = numpy.tile([0., 1., 0], phonons.N) / numpy.sqrt(phonons.N)
    nvecz = numpy.tile([0, 0, 1.], phonons.N) / numpy.sqrt(phonons.N)

    for mode in range(len(phonons.frequencies)):
        eigvec = phonons.eigenvectors[mode]

        stot = eigvec.dot(nvecx)**2 + eigvec.dot(nvecy)**2 + eigvec.dot(nvecz)**2
        print(mode + 1, eigvec.dot(nvecx)**2, eigvec.dot(nvecy)**2, eigvec.dot(nvecz)**2, '=', stot)


if __name__ == '__main__':
    main()
