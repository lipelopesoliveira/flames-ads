![GitHub-Mark-Light](docs/source/_static/img/logo_horizontal_light.png)
![GitHub-Mark-Dark ](docs/source/_static/img/logo_horizontal_dark.png)

# FLAMES - Flexible Lattice Adsorption by Monte Carlo Engine Simulation

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

The FLAMES is a general purpose adsorption simulation toolbox built around the Atomic Simulation Environment (ASE), which provides tools for molecular simulations and adsorption studies using machine learning potentials, classical force fields, and other advanced techniques.

## Requirements

0. Python >= 3.10
1. pymatgen
2. numpy
3. scipy
4. simplejson
5. ase
6. gemmi

The Python dependencies are most easily satisfied using a conda
([anaconda](https://www.anaconda.com/distribution)/[miniconda](https://docs.conda.io/en/latest/miniconda.html))
installation by running

```Shell
conda env create --file environment.yml
```

## Installation

Soon you will be able to install the FLAMES package using pip:

> [!WARNING]
>
> The code is not yet published on PyPI, so you need to import it manually from the GitHub repository.

You can use FLAMES by manually importing it using the `sys` module, as exemplified below:

```python
# importing module
import sys
 
# appending a path
sys.path.append('{PATH_TO_FLAMES}/flames')

from flames.gcmc import GCMC
```

Just remember to change the `{PATH_TO_FLAMES}` to the directory where you downloaded the FLAMES package.

## Feature Overview

The FLAMES package includes modules for performing Grand Canonical Monte Carlo (GCMC) simulations and Widom insertion tests. It is designed to work with the ASE (Atomic Simulation Environment) framework and supports various machine learning potentials.

## Usage

To use the FLAMES package, you can import the necessary classes from the `flames` module. For example, to perform GCMC simulations, you can use the `GCMC` class:

```python
import os

import ase
import torch
from ase.data import vdw_radii
from ase.io import read
from mace.calculators import mace_mp

from flames.gcmc import GCMC

device = "cuda" if torch.cuda.is_available() else "cpu"

FrameworkPath = "mg-mof-74.cif"
AdsorbatePath = "co2.xyz"

model = mace_mp(
    model="medium-0b2",
    dispersion=True,
    damping="zero",   # choices: ["zero", "bj", "zerom", "bjm"]
    dispersion_xc="pbe",
    default_dtype="float32",
    device=device,
)

# Load the framework structure
framework: ase.Atoms = read(FrameworkPath)  # type: ignore

# Load the adsorbate structure
adsorbate: ase.Atoms = read(AdsorbatePath)  # type: ignore

Temperature = 298.0  # in Kelvin
pressure = 100_000  # in Pa = 1 bar
MCSteps = 30_000


print(
    f"Running GCMC simulation for pressure: {pressure:.2f} Pa at temperature: {Temperature:.2f} K"
)

gcmc = GCMC(
    model=model,
    framework_atoms=framework,
    adsorbate_atoms=adsorbate,
    temperature=Temperature,
    pressure=pressure,
    device=device,
    vdw_radii=vdw_radii,
    vdw_factor=0.6,
    save_frequency=1,
    debug=True,
    output_to_file=True,
    criticalTemperature=304.1282,
    criticalPressure=7377300.0,
    acentricFactor=0.22394,
    move_weights={
            "insertion": 0.25,
            "deletion": 0.25,
            "translation": 0.25,
            "rotation": 0.25,
        },
    random_seed=42,
    cutoff_radius=6.0,
    automatic_supercell=True,
)


gcmc.logger.print_header()

gcmc.run(MCSteps)

gcmc.logger.print_summary()

gcmc.save_results()
```

## Examples

You can find example scripts in the [examples](https://github.com/lipelopesoliveira/flames/tree/main/examples) directory. These scripts demonstrate how to use the FLAMES package for various tasks, such as running GCMC simulations and performing Widom insertion tests, etc.

### Basic

[1. Widom Insertion](https://github.com/lipelopesoliveira/flames/tree/main/examples/Basic/1-Widom/run_widom.py)

The Widom insertion method is a powerful and computationally efficient technique in statistical mechanics used to calculate the excess chemical potential of a species at infinite dilution. The method operates by inserting a "ghost" or "test" particle at numerous random positions and orientations within a static configuration of a host system, such as a porous material. For each insertion, the interaction energy between the ghost particle and the host is calculated, but the particle is not actually added, so the host's configuration remains unchanged. By averaging the Boltzmann factor of these interaction energies over thousands or millions of trials, one can directly compute the Henry's constant, which is fundamentally related to the material's affinity for the adsorbate at low pressures. From this simulation the isosteric heat, or enthalpy of adsorption, at zero coverage can also be determined. Furthermore, the distribution of insertion energies reveals the potential energy landscape, identifying the coordinates of the most stable adsorption sites within the framework.

This makes the Widom method arguably the easiest and most direct way to test a new Machine Learning Potential (MLP) for adsorption applications. Unlike full Grand Canonical Monte Carlo simulations, it does not require lengthy equilibration or the simulation of multiple interacting guest molecules. The method yields two crucial, physically meaningful metrics: the most stable binding configurations and the enthalpy of adsorption. These two outputs can be directly compared with high-fidelity experimental data from microcalorimetry and diffraction techniques, or with results from expensive quantum mechanical calculations, providing a robust and computationally inexpensive first-pass validation of the MLP's accuracy in describing host-guest interactions.

[2. Rigid GCMC Simulation](https://github.com/lipelopesoliveira/flames/tree/main/examples/Basic/2-Rigid_GCMC/run_GCMC.py)

The Grand Canonical Monte Carlo (GCMC) method is a powerful computational technique used to simulate the adsorption of guest molecules in porous materials, such as metal-organic frameworks (MOFs), covalent organic frameworks (COFs), or zeolites. It operates under the grand canonical ensemble, allowing for the exchange of particles between the system and an ideal reservoir at a fixed temperature and chemical potential (that can be calculated from the pressure).

[3. Rigid GCMC Isotherm](https://github.com/lipelopesoliveira/flames/tree/main/examples/Basic/3-Rigid_GCMC_Isotherm/run_GCMC_Isotherm.py)

This example demonstrates how to run a GCMC simulation to generate adsorption isotherms for a given framework and adsorbate. The script allows you to specify a range of pressures and run the GCMC simulation for each pressure point, saving the results for further analysis. It starts each new pressure point from the last saved state, allowing for efficient continuation of simulations.

### Intermediate

[1. Geometry Optimization + GCMC](https://github.com/lipelopesoliveira/flames/tree/main/examples/Intermediate/1-Geometry_Optimization_GCMC/run_GCMC.py)

This example also demonstrate how to perform geometry optimization of the framework structure and adsorbate before running the GCMC simulation. The script uses the LBFGS optimizer from ASE to optimize the framework structure, but any other optimizer can be used as well. After the optimization, it runs the GCMC simulation using the optimized framework structure and the specified adsorbate.

[2. Molecular Dynamics (MD) with GCMC](https://github.com/lipelopesoliveira/flames/tree/main/examples/Intermediate/2-MD_GCMC/run_MD_GCMC.py)

This example demonstrates how to run a GCMC simulation with molecular dynamics (MD) steps. It allows you to perform GCMC simulations while also incorporating MD to explore the dynamic behavior of the system. The script initializes the GCMC simulation and runs it for a specified number of Monte Carlo steps. Them, it performs MD steps to simulate the motion of atoms in the framework and adsorbate at the specified temperature and pressure. After a few iterations of GCMC and MD, it runs a final GCMC simulation to ensure the system reaches equilibrium.
