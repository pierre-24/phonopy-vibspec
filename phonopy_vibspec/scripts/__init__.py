

def add_common_args(parser):
    parser.add_argument(
        '-c', '--phonopy', type=str, help='Phonopy YAML containing the cells', default='phonopy_disp.yaml')
    parser.add_argument(
        '--fc', type=str, help='Force constant filename', default='force_constants.hdf5')
