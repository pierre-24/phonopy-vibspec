# `phonopy-vibspec`

## Purpose

Simulate IR and Raman spectra. 
This requires the phonon eigenvalues and eigenvectors, wich are readily obtained from [`Phonopy`](https://phonopy.github.io/phonopy/index.html).
Then, calculation(s) of the dielectric matrix provide the IR and Raman intensities (see [10.1039/C7CP01680H](https://doi.org/10.1039/C7CP01680H)).

Note: this is actually a simpler (and packaged!) version of [`Phonopy-Spectroscopy`](https://github.com/skelton-group/Phonopy-Spectroscopy). The main difference with the latter is that this package does not include phonon line widths (and thus does not require `phono3py`).
If you are interested in that (or polarized Raman), use their code instead :)

## Install

To install this package, you need a running Python 3 installation (Python >= 3.10 recommended), and

```bash
pip3 install git+https://github.com/pierre-24/just-psf.git
```

Note: as this script install programs, you might need to add their location (such as `$HOME/.local/bin`, if you use `--user`) to your `$PATH`, if any.

## Usage

T.B.C.

```
# 1. Create POSCAR:
# (from https://phonopy.github.io/phonopy/vasp-dfpt.html#vasp-dfpt-interface)
phonopy -d --dim="1 1 1" -c POSCAR-unitcell  # preferentially a larger cell

# 2. cleanup
rm POSCAR-0*
mv SPOSCAR POSCAR

# 3. Run VASP using `IBRION=8` or `IBRION=6` and appropriate `POTIM`

# 4. Extract force constants
phonopy --hdf5 --fc vasprun.xml

```

## Who?

My name is [Pierre Beaujean](https://pierrebeaujean.net), and I have a Ph.D. in quantum chemistry from the [University of Namur](https://unamur.be) (Belgium).
I'm the main (and only) developer of this project, used in our lab.
I use this in the frame of my post-doctoral research in order to study batteries and solid electrolyte interphrase (we have a strong background in vibrational spectroscopy in our lab!), and I developed this project to ease my life.