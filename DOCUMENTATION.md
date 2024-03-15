# Documentation

## Install

To install this package, you need a running Python 3 installation (Python >= 3.10 recommended), and

```bash
pip3 install git+https://github.com/pierre-24/phonopy-vibspec.git
```

Note: as this script install programs, you might need to add their location (such as `$HOME/.local/bin`, if you use `--user`) to your `$PATH`, if any.

## Getting force constants

To get the force constant, you need to use [`Phonopy`](https://phonopy.github.io/phonopy/index.html) (which is installed since it is a dependency of this package) as usual.

On the one hand, if you can use [DFPT]( https://phonopy.github.io/phonopy/vasp-dfpt.html#vasp-dfpt-interface), the procedure is the following:
```bash
# 1. Create POSCAR of supercell:
phonopy -d --dim="1 1 1" -c unitcell.vasp 
# Note: use preferentially a larger cell

# 2. cleanup
rm POSCAR-*
mv SPOSCAR POSCAR

# 3. Run VASP using `IBRION=8` or `IBRION=6` with appropriate `POTIM`

# 4. Extract force constants (a `force_constants.hdf5` file is created)
phonopy --hdf5 --fc vasprun.xml
```

On the other hand, (see [there](https://phonopy.github.io/phonopy/vasp.html)):
```bash
# 1. Create POSCAR of supercell:
phonopy -d --dim="1 1 1" -c unitcell.vasp 
# Note: use preferentially a larger cell

# 2. Create folders for calculations
for i in POSCAR-*; do a=${i/POSCAR/disp}; mkdir -p $a; mv $i ${a}/POSCAR; done; 

# 3. Run VASP using `IBRION=-1`

# 4. Extract force sets
phonopy -f disp-*/vasprun.xml

# 5. Compute force constants (a `force_constants.hdf5` file is created)
phonopy --writefc-format HDF5 --writefc
```

If you want, you can then create files to vizualize the modes in [VESTA](http://jp-minerals.org/vesta/en/):

```bash
phonopy-vs-modes --modes="4 5 6"
```

## Infrared spectrum

The procedure is the following.

First, you need to compute the born effective charges and extract them using [a utility](https://phonopy.github.io/phonopy/auxiliary-tools.html#phonopy-vasp-born) provided by Phonopy:

```bash
# 1. Run a calculation with `LEPSILON = .TRUE.` **on the unit cell**

# 2. Extract Born effective charges from calculations (a `BORN` file is created)
phonopy-vasp-born vasprun.xml > BORN
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

It is also possible to compute spectra at other `q` points in the Brilouin zone:

```bash
phonopy-vs-ir -q="0.5 0 0" spectrum_0.5.csv
```

Note that Phonopy is generally not able to assign symmetry labels in that case.

## Raman spectrum

```bash
# 1. Get displaced geometries
phonopy-vs-prepare-raman

# 2. Create folders for calculations
for i in dielec-*.vasp; do a=$(i%.vasp); mkdir -p $a; cd $a; ln -s ../$i POSCAR; cd ..; done; 

# 3. Run calculations with `LEPSILON = .TRUE.` for each displaced geometry

# 4. Collect dielectric constants
phonopy-vs-gather-raman dielec-*/vasprun.xml

# 4. Get Raman spectrum
phonopy-vs-raman spectrum.csv
```

The resulting output contains the same sections as with IR (except it gives raman activities), and can be controlled using the same command line options.