import pathlib
from typing import List, Optional

import h5py
import numpy
from numpy.typing import NDArray


class RamanSpectrum:
    def __init__(
        self,
        # input
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
        with h5py.File(path) as f:
            if 'input' not in f:
                raise Exception('missing `input` group')

            modes = f['input/modes'][:]
            frequencies = f['input/frequencies'][:]
            irrep_labels = None
            dalpha_dq = None
            if 'input/irrep_labels' in f:
                irrep_labels = f['input/irrep_labels'][:]
            if 'input/dalpha_dq' in f:
                dalpha_dq = f['input/dalpha_dq'][:]

            nd_stencil = None
            nd_steps = None
            nd_dielectrics = None

            if 'nd' in f:
                g_nd = f['nd']

                nd_stencil = g_nd['stencil'][:]
                nd_mode_equiv = g_nd['mode_equiv'][:]
                nd_modes = g_nd['modes'][:]
                nd_steps = g_nd['steps'][:]

                if 'dielectrics' in g_nd:
                    nd_dielectrics = g_nd['dielectrics'][:]

            return cls(
                modes=modes, frequencies=frequencies, irrep_labels=irrep_labels, dalpha_dq=dalpha_dq,
                nd_stencil=nd_stencil, nd_mode_equiv=nd_mode_equiv, nd_modes=nd_modes, nd_steps=nd_steps,
                nd_dielectrics=nd_dielectrics
            )

    def to_hdf5(self, path: pathlib.Path):
        with h5py.File(path, 'w') as f:
            f.attrs['spectrum'] = 'Raman'
            f.attrs['version'] = 1

            g_input = f.create_group('input')
            g_input.create_dataset('modes', data=self.modes)
            g_input.create_dataset('frequencies', data=self.frequencies)

            if self.irrep_labels:
                g_input.create_dataset('irrep_labels', data=self.irrep_labels)

            if self.dalpha_dq:
                g_input.create_dataset('dalpha_dq', data=self.dalpha_dq)

            g_nd = f.create_group('nd')

            if self.nd_stencil:
                g_nd.create_dataset('stencil', data=numpy.array(self.nd_stencil))

            if self.nd_mode_equiv:
                g_nd.create_dataset('mode_equiv', data=self.nd_mode_equiv)

            if self.nd_modes:
                g_nd.create_dataset('modes', data=self.nd_modes)

            if self.nd_steps:
                g_nd.create_dataset('steps', data=self.nd_steps)

            if self.dielectrics is not None:
                g_nd.create_dataset('dielectrics', data=self.dielectrics)


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
                irrep_labels = f['input/irrep_labels'][:]

            return cls(modes=modes, frequencies=frequencies, irrep_labels=irrep_labels, dmu_dq=dmu_dq)

    def to_hdf5(self, path: pathlib.Path):
        with h5py.File(path, 'w') as f:
            f.attrs['spectrum'] = 'IR'
            f.attrs['version'] = 1

            g_input = f.create_group('input')
            g_input.create_dataset('modes', data=self.modes)
            g_input.create_dataset('frequencies', data=self.frequencies)
            g_input.create_dataset('dmu_dq', data=self.dmu_dq)

    def compute_intensities(self) -> NDArray[float]:
        return (self.dmu_dq ** 2).sum(axis=1)
