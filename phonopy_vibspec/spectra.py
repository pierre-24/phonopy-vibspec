import pathlib
import h5py
import numpy
from numpy.typing import NDArray
from typing import List, Optional, TextIO, Tuple

from phonopy.interface.vasp import VasprunxmlExpat

from phonopy_vibspec import logger


l_logger = logger.getChild(__name__)


def gen_lorentzian(x: NDArray[float], mu: float, intensity: float, linewidth: float):
    """Compute a Lorentzian.

    :param mu: center (mean value)
    :param area: total area (under the curve)
    """

    b = linewidth / 2
    return (intensity / numpy.pi) * b / ((x - mu) ** 2 + b ** 2)


def gen_spectrum(
    peaks: List[float],
    intensities: List[float],
    linewidths: List[float],
    spectrum_range: Tuple[float, float] = (400, 4000),
    spectrum_resolution: float = 1
) -> Tuple[NDArray[float], NDArray[float]]:
    """
    Create a spectrum containing Lorentzian for at each of the `peaks`.
    """

    assert len(linewidths) == len(peaks)
    assert len(intensities) == len(peaks)

    smin, smax = spectrum_range
    if smax < smin:
        smax, smin = smin, smax

    npoints = int(numpy.ceil((smax - smin) / spectrum_resolution)) + 1

    x = numpy.linspace(smin, smax, npoints, endpoint=True)
    y = numpy.zeros(npoints)

    for i in range(len(peaks)):
        y += gen_lorentzian(x, peaks[i], intensities[i], linewidths[i])

    return x, y


class InfraredSpectrum:
    def __init__(
        self,
        # input
        modes: List[int],
        frequencies: List[float],
        dmu_dq: NDArray[float],
        irrep_labels: Optional[List[str]] = None,
    ):
        assert irrep_labels is None or len(irrep_labels) == len(modes)
        assert len(frequencies) == len(modes)
        assert dmu_dq.shape[0] == len(modes)

        self.modes = modes
        self.frequencies = frequencies
        self.irrep_labels = irrep_labels
        self.dmu_dq = dmu_dq

    def __len__(self):
        return len(self.modes)

    @classmethod
    def from_hdf5(cls, path: pathlib.Path):
        l_logger.info('Read IR spectrum object from `{}`'.format(path))

        with h5py.File(path) as f:
            assert f.attrs['spectrum'] == 'IR'

            if f.attrs['version'] > 1:
                raise Exception('unsupported version')

            if 'input' not in f:
                raise Exception('missing `input` group')

            modes = f['input/modes'][:]
            frequencies = f['input/frequencies'][:]
            dmu_dq = f['input/dmu_dq'][:]
            irrep_labels = None
            if 'input/irrep_labels' in f:
                irrep_labels = [x.decode('utf-8') for x in f['input/irrep_labels']]

            return cls(modes=modes, frequencies=frequencies, irrep_labels=irrep_labels, dmu_dq=dmu_dq)

    def to_hdf5(self, path: pathlib.Path):
        l_logger.info('Write IR spectrum object in `{}`'.format(path))

        with h5py.File(path, 'w') as f:
            f.attrs['spectrum'] = 'IR'
            f.attrs['version'] = 1

            g_input = f.create_group('input')
            g_input.create_dataset('modes', data=self.modes)
            g_input.create_dataset('frequencies', data=self.frequencies)
            g_input.create_dataset('irrep_labels', data=[x.encode('utf-8') for x in self.irrep_labels])
            g_input.create_dataset('dmu_dq', data=self.dmu_dq)

    def compute_intensities(self) -> NDArray[float]:
        """Compute intensities, based on Eq. 7 of 10.1039/C7CP01680H.
        Intensities are given in [e²/AMU].
        """
        return (self.dmu_dq ** 2).sum(axis=1)

    def to_csv(
        self, f: TextIO,
        linewidths: List[float],
        spectrum_range: Tuple[float, float] = (400, 4000),
        spectrum_resolution: float = 1
    ):
        l_logger.info('Write CSV file')

        assert len(linewidths) == len(self)

        # 1. Peak data
        intensities = self.compute_intensities()

        f.write('"Mode"\t"Frequency [cm⁻¹]"\t"Irrep."\t"Intensity [e²/AMU]"\n')
        for i in range(len(self)):
            f.write('{}\t{:.6f}\t"{}"\t{:.6f}\n'.format(
                self.modes[i] + 1, self.frequencies[i], self.irrep_labels[i], intensities[i]
            ))

        f.write('\n\n')  # enough blank lines

        # 2. Graph
        f.write('"Frequency [cm⁻¹]"\t"Intensity [e²/AMU]"\n')
        x, y = gen_spectrum(self.frequencies, intensities, linewidths, spectrum_range, spectrum_resolution)
        f.write('\n'.join('{:.6f}\t{:.6f}'.format(xi, yi) for xi, yi in zip(x, y)))


class RamanSpectrum:
    def __init__(
        self,
        # input
        cell_volume: float,
        modes: List[int],
        frequencies: List[float],
        irrep_labels: Optional[List[str]] = None,
        dalpha_dq: Optional[NDArray[float]] = None,
        # nd
        nd_stencil: Optional[list] = None,
        nd_mode_equiv: Optional[List[int]] = None,
        nd_modes: Optional[List[int]] = None,
        nd_steps: List[float] = None,
        nd_dielectrics: Optional[NDArray] = None,
    ):
        assert irrep_labels is None or len(irrep_labels) == len(modes)
        assert len(frequencies) == len(modes)
        assert dalpha_dq is None or dalpha_dq.shape[0] == len(modes)

        assert nd_mode_equiv is None or len(nd_mode_equiv) == len(modes)
        assert nd_steps is None or len(nd_steps) == len(nd_modes)
        assert nd_dielectrics is None or nd_dielectrics.shape[0] == len(nd_modes)

        self.cell_volume = cell_volume
        self.modes = modes
        self.frequencies = frequencies
        self.irrep_labels = irrep_labels
        self.dalpha_dq = dalpha_dq

        self.nd_stencil = nd_stencil
        self.nd_mode_equiv = nd_mode_equiv
        self.nd_modes = nd_modes
        self.nd_steps = nd_steps
        self.dielectrics = nd_dielectrics

    def __len__(self):
        return len(self.modes)

    @classmethod
    def from_hdf5(cls, path: pathlib.Path):
        l_logger.info('Read Raman spectrum object from `{}`'.format(path))

        with h5py.File(path) as f:
            if 'input' not in f:
                raise Exception('missing `input` group')

            cell_volume = f['input/cell_volume'][0]
            modes = f['input/modes'][:]
            frequencies = f['input/frequencies'][:]
            irrep_labels = None
            dalpha_dq = None
            if 'input/irrep_labels' in f:
                irrep_labels = [x.decode('utf-8') for x in f['input/irrep_labels']]
            if 'input/dalpha_dq' in f:
                dalpha_dq = f['input/dalpha_dq'][:]

            nd_stencil = None
            nd_steps = None
            nd_dielectrics = None
            nd_mode_equiv = None
            nd_modes = None

            if 'nd' in f:
                g_nd = f['nd']

                nd_stencil = g_nd['stencil'][:]
                nd_mode_equiv = g_nd['mode_equiv'][:]
                nd_modes = g_nd['modes'][:]
                nd_steps = g_nd['steps'][:]

                if 'dielectrics' in g_nd:
                    nd_dielectrics = g_nd['dielectrics'][:]

            return cls(
                cell_volume=cell_volume,
                modes=modes, frequencies=frequencies, irrep_labels=irrep_labels, dalpha_dq=dalpha_dq,
                nd_stencil=nd_stencil, nd_mode_equiv=nd_mode_equiv, nd_modes=nd_modes, nd_steps=nd_steps,
                nd_dielectrics=nd_dielectrics
            )

    def to_hdf5(self, path: pathlib.Path):
        l_logger.info('Write Raman spectrum object in `{}`'.format(path))

        with h5py.File(path, 'w') as f:
            f.attrs['spectrum'] = 'Raman'
            f.attrs['version'] = 1

            g_input = f.create_group('input')
            g_input.create_dataset('cell_volume', data=[self.cell_volume])
            g_input.create_dataset('modes', data=self.modes)
            g_input.create_dataset('frequencies', data=self.frequencies)

            if self.irrep_labels is not None:
                g_input.create_dataset('irrep_labels', data=[x.encode('utf-8') for x in self.irrep_labels])

            if self.dalpha_dq is not None:
                g_input.create_dataset('dalpha_dq', data=self.dalpha_dq)

            g_nd = f.create_group('nd')

            if self.nd_stencil is not None:
                g_nd.create_dataset('stencil', data=numpy.array(self.nd_stencil))

            if self.nd_mode_equiv is not None:
                g_nd.create_dataset('mode_equiv', data=self.nd_mode_equiv)

            if self.nd_modes is not None:
                g_nd.create_dataset('modes', data=self.nd_modes)

            if self.nd_steps is not None:
                g_nd.create_dataset('steps', data=self.nd_steps)

            if self.dielectrics is not None:
                g_nd.create_dataset('dielectrics', data=self.dielectrics)

    def extract_dielectrics(self, files: List[pathlib.Path]):
        """Extract dielectric tensors from `files`.
        Assumes that the files are in the EXACT same order as created (i.e., increasing mode and step).
        """

        nsteps = len(self.nd_stencil)
        assert len(files) == nsteps * len(self.nd_modes)

        dielectrics = numpy.zeros((len(self.nd_modes), 2, 3, 3))

        for i, path in enumerate(files):
            imode = i // nsteps
            step = i % nsteps

            l_logger.info('Read (mode={}, step={}) from `{}`'.format(self.nd_modes[imode], step, path))

            with path.open('rb') as f:
                vasprun = VasprunxmlExpat(f)
                vasprun.parse()

                dielectric_tensor = vasprun.epsilon
                if dielectric_tensor is None:
                    raise Exception('no dielectric tensor in `{}`'.format(path))

                dielectrics[imode, step] = dielectric_tensor

        self.dielectrics = dielectrics
        self._compute_dalpha_dq()

    def _compute_dalpha_dq(self):
        assert self.dielectrics is not None

        l_logger.info('Compute dα/dq_i using numerical differentiation')

        # compute:
        dadqs = numpy.zeros((len(self.nd_modes), 3, 3))
        for i, mode in enumerate(self.nd_modes):
            dadqs[i] = numpy.sum(
                [c * d for (v, c), d in zip(self.nd_stencil, self.dielectrics[i])], axis=0) / self.nd_steps[i]

        # distribute:
        mode_to_index = dict((m, i) for i, m in enumerate(self.nd_modes))
        dalpha_dq = numpy.zeros((len(self), 3, 3))
        for i, mode in enumerate(self.modes):
            dalpha_dq[i] = dadqs[mode_to_index[self.nd_mode_equiv[i]]]

        self.dalpha_dq = dalpha_dq

    def _sq_alpha(self) -> NDArray[float]:
        """
        Compute the first Raman invariant (α²) for all modes, according to Eq. 7 of 10.1103/PhysRevB.54.7830
        """

        assert self.dalpha_dq is not None

        prefactor = self.cell_volume / (4 * numpy.pi)
        alphas = prefactor * (self.dalpha_dq[:, 0, 0] + self.dalpha_dq[:, 1, 1] + self.dalpha_dq[:, 2, 2]) / 3
        return alphas ** 2

    def _sq_beta(self) -> NDArray[float]:
        """
        Compute the second Raman invariant (β²) for all modes, according to Eq. 7 of 10.1103/PhysRevB.54.7830
        """

        assert self.dalpha_dq is not None

        prefactor = self.cell_volume / (4 * numpy.pi)

        return prefactor ** 2 * (
            (
                self.dalpha_dq[:, 0, 0] - self.dalpha_dq[:, 1, 1]
            ) ** 2 + (
                self.dalpha_dq[:, 0, 0] - self.dalpha_dq[:, 2, 2]
            ) ** 2 + (
                self.dalpha_dq[:, 1, 1] - self.dalpha_dq[:, 2, 2]
            ) ** 2 + 6 * (
                self.dalpha_dq[:, 0, 1] ** 2 + self.dalpha_dq[:, 0, 2] ** 2 + self.dalpha_dq[:, 1, 2] ** 2
            )
        )

    def compute_intensities(self) -> NDArray[float]:
        """Compute intensities.
        Based on Eq. 9 and 10 of 10.1039/C7CP01680H or alternatively Eq. 6 of 10.1103/PhysRevB.54.7830.
        Intensities are given in [Å⁴/AMU].
        """

        return 45 * self._sq_alpha() + 3.5 * self._sq_beta()

    def compute_depolarization_ratios(self) -> NDArray[float]:
        """Compute DR according to Eq. 8 of 10.1103/PhysRevB.54.7830
        """

        a = self._sq_alpha()
        b = self._sq_beta()

        return 3 * b / (45 * a + 4 * b)

    def to_csv(
        self, f: TextIO,
        linewidths: List[float],
        spectrum_range: Tuple[float, float] = (400, 4000),
        spectrum_resolution: float = 1
    ):
        l_logger.info('Write CSV file')

        assert len(linewidths) == len(self)

        # 1. Peak data
        intensities = self.compute_intensities()
        rhos = self.compute_depolarization_ratios()

        f.write('"Mode"\t"Frequency [cm⁻¹]"\t"Irrep."\t"Intensity [Å⁴/AMU]"\t"ρ"\n')
        for i in range(len(self)):
            f.write('{}\t{:.6f}\t"{}"\t{:.6f}\t{:.3f}\n'.format(
                self.modes[i] + 1, self.frequencies[i], self.irrep_labels[i], intensities[i], rhos[i]
            ))

        f.write('\n\n')  # enough blank lines

        # 2. Graph
        f.write('"Frequency [cm⁻¹]"\t"Intensity [Å⁴/AMU]"\n')
        x, y = gen_spectrum(self.frequencies, intensities, linewidths, spectrum_range, spectrum_resolution)
        f.write('\n'.join('{:.6f}\t{:.6f}'.format(xi, yi) for xi, yi in zip(x, y)))
