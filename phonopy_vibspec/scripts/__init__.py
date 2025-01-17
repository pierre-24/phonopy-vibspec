import argparse
from typing import Tuple


class ArgGetVector:
    def __init__(self, size: int):
        self.size = size

    def __call__(self, inp: str) -> tuple:
        chunks = inp.split()
        if len(chunks) != self.size:
            raise argparse.ArgumentTypeError(
                'invalid (space-separated) vector `{}`, must contains {} elements'.format(inp, self.size))

        try:
            return tuple(float(x) for x in chunks)
        except ValueError:
            raise argparse.ArgumentTypeError('invalid (space-separated) list of floats `{}`'.format(inp))


def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        '-c', '--phonopy', type=str, help='Phonopy YAML containing the cells', default='phonopy_disp.yaml')
    parser.add_argument(
        '--fc', type=str, help='Force constant filename', default='force_constants.hdf5')

    parser.add_argument('-m', '--modes', type=str, help='List of modes (1-based)', default='')

    parser.add_argument(
        '-q',
        type=ArgGetVector(3),
        help='q-point at which this should be computed (default is gamma)',
        default='0 0 0')

    parser.add_argument('-O', '--only', type=str, help='only consider certain atoms', default='')


def interval(s_interval: str) -> Tuple[float, float]:
    """get interval

    :param s_interval: string of the form `min:max`
    """

    inf = s_interval.split(':')
    if len(inf) != 2:
        raise argparse.ArgumentTypeError('limits must be `min:max`')

    try:
        min_, max_ = float(inf[0]), float(inf[1])
    except ValueError:
        raise argparse.ArgumentTypeError('min and max must be float')

    if min_ < .0 or max_ < .0:
        raise argparse.ArgumentTypeError('minimum should be larger or equal to zero')

    return min_, max_


def add_common_args_spectra(parser: argparse.ArgumentParser):
    parser.add_argument(
        '-l', '--limits', help='Limit and units of the graph: `min:max`', type=interval, default='100:2000')

    parser.add_argument(
        '-e', '--each', help='interval between points (in cm-1)', default=1, type=float)

    parser.add_argument(
        '-L', '--linewidth', help='Linewidth', default=5, type=float)

    parser.add_argument('csv', help='output CSV file', type=argparse.FileType('w'))
