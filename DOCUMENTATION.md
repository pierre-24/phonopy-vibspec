# Documentation

## Install

To install this package, you need a running Python 3 installation (Python >= 3.10 recommended), and

```bash
pip3 install git+https://github.com/pierre-24/phonopy-vibspec.git
```

Note: as this script install programs, you might need to add their location (such as `$HOME/.local/bin`, if you use `--user`) to your `$PATH`, if any.

## Theory

For an intro to phonon calculations, see [the VASP documentation](https://www.vasp.at/wiki/index.php/Phonons:_Theory). 

Long story short, in order to get a spectra, one needs to:

1. Compute the dynamic matrix (also called mass-weighted hessian) and diagonalize it, which will provide the frequencies and modes, and
2. Compute the Born charge in order to compute the IR intensities and/or their derivatives to get the Raman intensities.

## Usage

### 1. Frequencies and normal modes

To get the force constant, you need to use [`Phonopy`](https://phonopy.github.io/phonopy/index.html) (which is installed since it is a dependency of this package) as usual.

On the one hand, if you can use [DFPT](https://phonopy.github.io/phonopy/vasp-dfpt.html#vasp-dfpt-interface) (which is recommended), the procedure is the following:

```bash
# 1. Create POSCAR of supercell:
phonopy -d --dim="1 1 1" -c unitcell.vasp 
# Note: use preferentially a larger cell!

# 2. cleanup
rm POSCAR-*
mv SPOSCAR POSCAR

# 3. Run VASP using `IBRION` (see below)
vasp_std

# 4. Extract force constants (a `force_constants.hdf5` file is created)
phonopy --hdf5 --fc vasprun.xml
```

For step 3, the [`IBRION`](https://www.vasp.at/wiki/index.php/IBRION) keyword should be set to 5, 6, 7, or 8 (more info [there](https://www.vasp.at/wiki/index.php/IBRION#Computing_the_phonon_modes)).
When using numerical differentiation (`IBRION=5` or `IBRION=6`), convergence criterion should be stricter.
For example:

```text
IBRION = 6      ! numerical differentiation, using symmetry
NFREE = 2       ! central differences for the force, should be 2 or 4
POTIM = 0.015   ! displacement, default is 0.015
EDIFF = 1.0e-08 ! stricter criterion for energy convergence
PREC = Accurate ! increase precision
```

You might also want to increase [`NELM`](https://www.vasp.at/wiki/index.php/NELM) since `EDIFF` is increased.

On the other hand, (see [there](https://phonopy.github.io/phonopy/vasp.html)):

```bash
# 1. Create POSCAR of supercell:
phonopy -d --dim="1 1 1" -c unitcell.vasp 
# Note: use preferentially a larger cell

# 2. Create folders for calculations
for i in POSCAR-*; do a=${i/POSCAR/disp}; mkdir -p $a; mv $i ${a}/POSCAR; done; 

# 3. Run VASP using `IBRION=-1` (see below)
vasp_std

# 4. Extract force sets
phonopy -f disp-*/vasprun.xml

# 5. Compute force constants (a `force_constants.hdf5` file is created)
phonopy --writefc-format HDF5 --writefc
```

Again, for step 3, you need to set:

```text
EDIFF = 1.0e-08 ! stricter criterion for energy convergence
PREC = Accurate  ! increase precision
```

### 2. Visualisation and interpretation of the normal modes (optional)

If you want, you can then create files to visualize the modes in [VESTA](http://jp-minerals.org/vesta/en/):

```bash
phonopy-vs-modes --modes="4 5 6"
```

A `modexxx.vesta` is created per mode.

Furthermore, a partial analysis of normal modes is available: it estimates the percentage of translation, rotation, and vibration of normal modes.

```bash
phonopy-vs-analyze-modes
```

However, to determine the rotation, the program needs a center of rotation.
By default, it is taken as the center of mass. You can manually set it with `-C`, *e.g.* `-C ".5 .5 .5"` for the center of the cell (this center is to be given in **fractional coordinates**).
It is also possible to "unwrap" the cell (i.e., move atoms close together), which gives better results for single molecules.

**Note:** negative vibrational contributions are sometimes reported. Keep in mind that this is an estimate.

### 3. Infrared spectrum

After obtaining the dynamic matrix, the frequencies and corresponding mode (step 1), you then need to run another calculation in order to compute the born effective charges and extract them using [a utility](https://phonopy.github.io/phonopy/auxiliary-tools.html#phonopy-vasp-born) provided by Phonopy:

```bash
# 1. Run a calculation with `LEPSILON = .TRUE.` **on the unit cell**
vasp_std

# 2. Extract Born effective charges from calculations (a `BORN` file is created)
phonopy-vasp-born vasprun.xml > BORN
```

For step 1, the `INCAR` file should use the following parameters (again, to ensure precision):

```text
LEPSILON = .TRUE. ! Compute dielectric matrix
PREC = Accurate
EDIFF = 1.0e-08
```

Then, you can create an IR spectrum:

```bash
# 3. Get IR spectrum
phonopy-vs-ir -b BORN  spectrum.csv
```

The `-b` option controls the location of the `BORN` file

The output CSV file contains two sections:

1. a list of each normal mode, its irreducible representation, and their IR activity, and
2. a spectrum.

You can control the latter using different command line options:

+ `--limit 200:2000`, which create a graph between 200 and 2000 cm⁻¹;
+ `--each=1`, the interval between each point (in cm⁻¹);
+ `--linewidth=5`, the linewidth of the Lorentzian (in cm⁻¹);

It is also possible to compute spectra at other `q` points in the Brillouin zone:

```bash
phonopy-vs-ir -q="0.5 0 0" spectrum_0.5.csv
```

Note that Phonopy is generally not able to assign symmetry labels in that case.

### 4. Raman spectrum

The procedure is more complex, since one needs the derivatives of the BORN charge with respect to polarizability (*i.e.*, the polarizability).
Once you have completed step 1, do the following:

```bash
# 1. Get displaced geometries and prepare a `raman.hdf5` file
phonopy-vs-prepare-raman

# 2. Create folders for calculations
for i in dielec_*.vasp; do a=${i%.vasp}; mkdir -p $a; cd $a; ln -s ../$i POSCAR; cd ..; done; 

# 3. Run calculations with `LEPSILON = .TRUE.` for each displaced geometry
for i in dielec_*; do if [[ -d $i ]]; then cd $i; vasp_std; cd ..; fi; done

# 4. Collect dielectric constants
phonopy-vs-gather-raman dielec_*/vasprun.xml

# 4. Get Raman spectrum
phonopy-vs-raman spectrum.csv
```

The resulting output contains the same sections as with IR (except it gives raman activities), and can be controlled using the same command line options.

## Contribute

Contributions, either with [issues](https://github.com/pierre-24/phonopy-vibspec/issues) or [pull requests](https://github.com/pierre-24/phonopy-vibspec/pulls) are welcomed.

### Install (for dev)

Rather than the installation procedure given on top if this document, if you want to contribute, this is the usual deal: 
start by [forking](https://guides.github.com/activities/forking/), then clone your fork and use the following install procedures instead.

```bash
cd phonopy-vibspec

# definitely recommended in this case: use a virtualenv!
python -m venv virtualenv
source venv/bin/activate

# install also dev dependencies
make install
```

### Tips to contribute

+ A good place to start is the [list of issues](https://github.com/pierre-24/phonopy-vibspec/issues).
  In fact, it is easier if you start by filling an issue, and if you want to work on it, says so there, so that everyone knows that the issue is handled.

+ Don't forget to work on a separate branch.
  Since this project follows the [git flow](http://nvie.com/posts/a-successful-git-branching-model/), you should base your branch on `main`, not work in it directly:

    ```bash
    git checkout -b new_branch origin/main
    ```
 
+ Don't forget to regularly run the linting and tests:

    ```bash
    make lint
    make test
    ```
    
    Indeed, the code follows the [PEP-8 style recommendations](http://legacy.python.org/dev/peps/pep-0008/), checked by [`flake8`](https://flake8.pycqa.org/en/latest/).
    Having an extensive test suite is also a good idea to prevent regressions.

+ Pull requests should be unitary, and include unit test(s) and documentation if needed. 
  The test suite and lint must succeed for the merge request to be accepted.

## Who?

My name is [Pierre Beaujean](https://pierrebeaujean.net), and I have a Ph.D. in quantum chemistry from the [University of Namur](https://unamur.be) (Belgium).
I'm the main (and only) developer of this project, used in our lab.
I use this in the frame of my post-doctoral research in order to study batteries and solid electrolyte interphrase, and I developed this project to ease my life.

Note: due to my (quantum) chemistry background, we may speak of similar things using a different vocabulary.
