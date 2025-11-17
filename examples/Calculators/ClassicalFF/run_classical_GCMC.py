import json

import ase
import numpy as np
from ase.calculators import mixing
from ase.data import vdw_radii
from ase.io import read
from numba import set_num_threads

from flames.calculators import CustomLennardJones, EwaldSum
from flames.gcmc import GCMC

NUM_THREADS_TO_USE = 4
set_num_threads(NUM_THREADS_TO_USE)

with open("/home/felipe/PRs/mlp_adsorption/flames/data/UFF_lj_params.json", "r") as f:
    uff_lj_params = json.loads(f.read())

with open("/home/felipe/PRs/mlp_adsorption/flames/data/TraPPE_lj_params.json", "r") as f:
    trappe_lj_params = json.loads(f.read())

ewald = EwaldSum(R_cutoff=5.5, G_cutoff_N=5, alpha=5 / 15)
lj = CustomLennardJones({**uff_lj_params, **trappe_lj_params}, vdw_cutoff=12.5)

calc = mixing.SumCalculator([lj, ewald])

FrameworkPath = "MgMOF-74_DDEC.cif"
AdsorbatePath = "co2_labels.xyz"

# Load the framework structure
framework: ase.Atoms = read(FrameworkPath, store_tags=True)  # type: ignore
framework.set_initial_charges(framework.info["_atom_site_charge"])
framework.arrays["labels"] = np.array(framework.get_chemical_symbols(), dtype=object)
framework.info = {}

# Load the adsorbate structure
adsorbate: ase.Atoms = read(AdsorbatePath)  # type: ignore

Temperature = 298.0  # in Kelvin
pressure = 100_000  # in Pa = 0.01 bar
MCSteps = 10_000


print(
    f"Running GCMC simulation for pressure: {pressure:.2f} Pa at temperature: {Temperature:.2f} K"
)

gcmc = GCMC(
    model=calc,  # type: ignore
    framework_atoms=framework,
    adsorbate_atoms=adsorbate,
    temperature=Temperature,
    pressure=pressure,
    device="cpu",
    vdw_radii=vdw_radii,
    vdw_factor=0.6,
    save_frequency=1,
    debug=False,
    output_to_file=True,
    criticalTemperature=304.1282,
    criticalPressure=7377300.0,
    acentricFactor=0.22394,
    random_seed=42,
    cutoff_radius=12.5,
    automatic_supercell=True,
)


gcmc.logger.print_header()

gcmc.run(MCSteps)

gcmc.logger.print_summary()

gcmc.save_results()
