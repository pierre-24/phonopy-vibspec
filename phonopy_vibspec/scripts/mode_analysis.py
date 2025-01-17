"""
Create a VESTA file containing the eigenvector for each mode
"""

import argparse
import numpy

from phonopy_vibspec import GetListWithinBounds
from phonopy_vibspec.phonons_analyzer import PhononsAnalyzer
from phonopy_vibspec.scripts import add_common_args


def get_com(inp: str) -> numpy.ndarray:
    chunks = inp.split()
    if len(chunks) != 3:
        raise argparse.ArgumentTypeError('invalid (space-separated) COM: `{}`, must contains 3 elements'.format(inp))

    try:
        return numpy.array(list(float(x) for x in chunks))
    except ValueError:
        raise argparse.ArgumentTypeError('invalid (space-separated) list of floats `{}` for COM'.format(inp))


def fix_structure(positions, cell) -> numpy.ndarray:
    """
    Get atoms closer.
    Assume rectangular cell.
    """
    new_positions = numpy.zeros(positions.shape)

    new_positions[0] = positions[0]
    ref = positions[0]

    for iatm in range(1, len(positions)):
        # apply minimal image convention
        pos = positions[iatm]
        dr = pos - ref
        for j in range(3):
            new_positions[iatm, j] = positions[iatm, j] - cell[j, j] * round(dr[j] / cell[j, j])

    return new_positions


def compute_com(positions, masses):
    return numpy.sum(positions * masses[:, numpy.newaxis], axis=0) / numpy.sum(masses)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)

    parser.add_argument(
        '-f', '--fix-geometry', action='store_true', help='unwrap the cell and move atoms close together')
    parser.add_argument('-C', '--center', type=get_com, help='set the center')

    args = parser.parse_args()

    phonons = PhononsAnalyzer.from_phonopy(
        phonopy_yaml=args.phonopy,
        force_constants_filename=args.fc,
        only=args.only if args.only != '' else None
    )

    # prepare analysis
    sqrt_masses = numpy.repeat(numpy.sqrt(phonons.structure.masses), 3)

    vec_dispx = numpy.tile([1., 0, 0], phonons.N) * sqrt_masses
    vec_dispx /= numpy.linalg.norm(vec_dispx)

    vec_dispy = numpy.tile([0., 1., 0], phonons.N) * sqrt_masses
    vec_dispy /= numpy.linalg.norm(vec_dispy)

    vec_dispz = numpy.tile([0, 0, 1.], phonons.N) * sqrt_masses
    vec_dispz /= numpy.linalg.norm(vec_dispz)

    proj_x = numpy.outer([1., 0, 0], [1., 0, 0])
    proj_y = numpy.outer([0, 1., 0], [0, 1., 0])
    proj_z = numpy.outer([0, 0, 1.], [0, 0, 1.])

    vec_rotx = numpy.zeros(3 * phonons.N)
    vec_roty = numpy.zeros(3 * phonons.N)
    vec_rotz = numpy.zeros(3 * phonons.N)

    positions = phonons.structure.scaled_positions

    if args.fix_geometry:
        positions = fix_structure(positions, numpy.eye(3))

        print('New geometry is:\n```\nDirect')
        for iatm in range(phonons.N):
            print('{: .7f} {: .7f} {: .7f} {}'.format(*positions[iatm], phonons.structure.symbols[iatm]))
        print('```')

    if args.center is not None:
        center = args.center
    else:
        center = compute_com(positions, phonons.structure.masses)

    print('Center (used for rotations) is', center)

    for iatm in range(phonons.N):
        r = positions[iatm] - center

        rx = r - proj_x.dot(r)
        if numpy.linalg.norm(rx) > 1e-5:
            rx /= numpy.linalg.norm(rx)
            vec_rotx[iatm * 3:(iatm + 1) * 3] = numpy.linalg.cross(rx, [1., 0, 0])

        ry = r - proj_y.dot(r)
        if numpy.linalg.norm(ry) > 1e-5:
            ry /= numpy.linalg.norm(ry)
            vec_roty[iatm * 3:(iatm + 1) * 3] = numpy.linalg.cross(ry, [0, 1., 0])

        rz = r - proj_z.dot(r)
        if numpy.linalg.norm(rz) > 1e-5:
            rz /= numpy.linalg.norm(rz)
            vec_rotz[iatm * 3:(iatm + 1) * 3] = numpy.linalg.cross(rz, [0, 0, 1.])

    vec_rotx *= sqrt_masses
    vec_rotx /= numpy.linalg.norm(vec_rotx)
    vec_roty *= sqrt_masses
    vec_roty /= numpy.linalg.norm(vec_roty)
    vec_rotz *= sqrt_masses
    vec_rotz /= numpy.linalg.norm(vec_rotz)

    # perform analysis
    modes = [
        x - 1 for x in GetListWithinBounds(1, 3 * phonons.N)(args.modes)
    ] if len(args.modes) > 0 else list(range(3 * phonons.N))

    print()
    print('Mode  Freq. (cm-1)  Translation (%)              Rotation (%)                 Vib. (%)')
    print('----  ------------  ---------------------------  ---------------------------  --------')
    print('                       a      b      c     Tot.     a      b      c     Tot.      Tot.')
    for mode in modes:
        eigvec = phonons.eigenvectors[mode]

        trans_contribs = eigvec.dot(vec_dispx)**2, eigvec.dot(vec_dispy)**2, eigvec.dot(vec_dispz)**2
        rot_contribs = eigvec.dot(vec_rotx)**2, eigvec.dot(vec_roty)**2, eigvec.dot(vec_rotz)**2

        print('{:4} {:14.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f}  {:6.2f} {:6.2f} {:6.2f} {:6.2f}  {:8.2f}   {}'.format(
            mode + 1,
            phonons.frequencies[mode],
            *(x * 100 for x in trans_contribs),
            sum(trans_contribs) * 100,
            *(x * 100 for x in rot_contribs),
            sum(rot_contribs) * 100,
            (1 - sum(trans_contribs) - sum(rot_contribs)) * 100,
            'T' if sum(trans_contribs) > .5 else ('R' if sum(rot_contribs) > .5 else 'V')
        ))


if __name__ == '__main__':
    main()
