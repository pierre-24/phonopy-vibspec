# `phonopy-vibspec`

## Purpose

Simulate IR and Raman spectra. 
This requires the phonon eigenvalues and eigenvectors, wich are readily obtained from [`Phonopy`](https://phonopy.github.io/phonopy/index.html).
Then, calculation(s) of the dielectric matrix provide the IR and Raman intensities (see [10.1039/C7CP01680H](https://doi.org/10.1039/C7CP01680H)).

Note: this is actually a simpler (and packaged!) version of [`Phonopy-Spectroscopy`](https://github.com/skelton-group/Phonopy-Spectroscopy). The main difference with the latter is that this package does not include phonon line widths (and thus does not require `phono3py`).
If you are interested in that (or polarized Raman), use their code instead :)

## Installation, usage, & contributions

See [DOCUMENTATION.md](DOCUMENTATION.md).