# `phonopy-vibspec`

## Purpose

Simulate IR and Raman spectra. 
This requires the phonon eigenvalues and eigenvectors, wich are readily obtained from [`Phonopy`](https://phonopy.github.io/phonopy/index.html).
Then, calculation(s) of the dielectric matrix provide the IR and Raman intensities (see [10.1039/C7CP01680H](https://doi.org/10.1039/C7CP01680H)).

Note: this is actually a simpler (and packaged!) version of [`Phonopy-Spectroscopy`](https://github.com/skelton-group/Phonopy-Spectroscopy). The main difference with the latter is that this package does not include phonon line widths (and thus does not require `phono3py`).
If you are interested in that (or polarized Raman), use their code instead :)

## Install and usage

See [DOCUMENTATION.md](DOCUMENTATION.md)

## Who?

My name is [Pierre Beaujean](https://pierrebeaujean.net), and I have a Ph.D. in quantum chemistry from the [University of Namur](https://unamur.be) (Belgium).
I'm the main (and only) developer of this project, used in our lab.
I use this in the frame of my post-doctoral research in order to study batteries and solid electrolyte interphrase, and I developed this project to ease my life.

Note: due to my (quantum) chemistry background, we may speak of similar things using a different vocabulary.