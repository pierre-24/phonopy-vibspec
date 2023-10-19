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
        # calculations
        stencil: Optional[list] = None,
        steps: List[float] = None,
        dielectrics: Optional[NDArray] = None,
    ):
        assert irrep_labels is None or len(irrep_labels) == len(modes)
        assert len(frequencies) == len(modes)
        assert dalpha_dq is None or dalpha_dq.shape[0] == len(modes)

        assert steps is None or len(steps) == len(modes)
        assert dielectrics is None or dielectrics.shape[0] == len(modes)

        self.modes = modes
        self.frequencies = frequencies
        self.irrep_labels = irrep_labels
        self.dalpha_dq = dalpha_dq

        self.steps = steps
        self.dielectrics = dielectrics
        self.stencil = stencil

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

            stencil = None
            steps = None
            dielectrics = None

            if 'numerical_differentiation' in f:
                stencil = f['numerical_differentiation/stencil'][:]
                steps = f['numerical_differentiation/steps'][:]

                if 'numerical_differentiation/dielectrics' in f:
                    dielectrics = f['numerical_differentiation/dielectrics'][:]

            return cls(
                modes=modes, frequencies=frequencies, irrep_labels=irrep_labels, dalpha_dq=dalpha_dq,
                stencil=stencil, steps=steps, dielectrics=dielectrics
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

            g_calcs = f.create_group('numerical_differentiation')
            g_calcs.create_dataset('stencil', data=numpy.array(self.stencil))

            if self.steps:
                g_calcs.create_dataset('steps', data=self.steps)

            if self.dielectrics is not None:
                g_calcs.create_dataset('dielectrics', data=self.dielectrics)


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
