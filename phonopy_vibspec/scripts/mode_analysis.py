"""
Analyze the translational/rotational/vibrational contribution to each mode
"""

import argparse
import numpy

from phonopy_vibspec import GetListWithinBounds
from phonopy_vibspec.phonons_analyzer import PhononsAnalyzer
from phonopy_vibspec.scripts import add_common_args, ArgGetVector


def fix_structure(positions: numpy.ndarray, cell: numpy.ndarray) -> numpy.ndarray:
    """
    Get atoms closer. Use scaled positions.
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


def compute_com(positions: numpy.ndarray, masses: numpy.ndarray) -> numpy.ndarray:
    """Compute the center of mass"""
    return numpy.sum(positions * masses[:, numpy.newaxis], axis=0) / numpy.sum(masses)


def compute_inertia(positions: numpy.ndarray, com: numpy.ndarray, masses: numpy.ndarray) -> numpy.ndarray:
    """Compute the inertia tensor"""
    p = positions - com
    inertia = numpy.zeros((3, 3))

    for i in range(3):
        for j in range(i + 1):
            if i == j:
                inertia[i, i] = numpy.sum(masses * (p[:, (i + 1) % 3]**2 + p[:, (i + 2) % 3]**2))
            else:
                inertia[i, j] = -numpy.sum(masses * p[:, i] * p[:, j])
                inertia[j, i] = inertia[i, j]

    return inertia


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)

    parser.add_argument(
        '-f', '--fix-geometry', action='store_true', help='unwrap the cell and move atoms close together')

    parser.add_argument(
        '-I', '--inertia', action='store_true', help='use Inertia tensor for rotations')
    parser.add_argument('-C', '--center', type=ArgGetVector(3), help='set the center')

    args = parser.parse_args()

    phonons = PhononsAnalyzer.from_phonopy(
        phonopy_yaml=args.phonopy,
        force_constants_filename=args.fc,
        only=args.only if args.only != '' else None
    )

    # patch up geometry if any
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

    # prepare analysis
    sqrt_masses = numpy.repeat(numpy.sqrt(phonons.structure.masses), 3)

    principal_axes = []
    if args.intertia:
        inertia = compute_inertia(positions, center, phonons.structure.masses)
        eigv = numpy.linalg.eigh(inertia).eigenvectors.T

        print('Principal (inertia) axes are::')

        for i in range(3):
            print(eigv[i])
            principal_axes.append(eigv[i])
    else:
        for i in range(3):
            z = numpy.zeros(3)
            z[i] = 1.
            principal_axes.append(z)

    # translation
    vec_disps = []
    for i in range(3):
        vec_disps.append(numpy.tile(principal_axes[i], phonons.N) * sqrt_masses)
        vec_disps[i] /= numpy.linalg.norm(vec_disps[i])

    # rotation
    rot_projs = []
    for i in range(3):
        rot_projs.append(numpy.outer(principal_axes[i], principal_axes[i]))

    vec_rots = []
    for i in range(3):
        vec_rots.append(numpy.zeros(3 * phonons.N))

    for iatm in range(phonons.N):
        r = positions[iatm] - center

        for i in range(3):
            ri = r - rot_projs[i].dot(r)
            ri /= numpy.linalg.norm(ri)
            vec_rots[i][iatm * 3:(iatm + 1) * 3] = numpy.cross(ri, principal_axes[i])

    for i in range(3):
        vec_rots[i] /= numpy.linalg.norm(vec_rots[i])

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

        trans_contribs = []
        rot_contribs = []

        for i in range(3):
            trans_contribs.append(eigvec.dot(vec_disps[i])**2)
            rot_contribs.append(eigvec.dot(vec_rots[i])**2)

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
