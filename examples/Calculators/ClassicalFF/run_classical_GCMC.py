import json

import ase
from ase.calculators import mixing
from ase.data import vdw_radii
from ase.io import read
from numba import get_num_threads, set_num_threads

from flames.calculators import CustomLennardJones, EwaldSum
from flames.gcmc import GCMC
from flames.utilities import read_cif

NUM_THREADS_TO_USE = 25
set_num_threads(NUM_THREADS_TO_USE)
print(get_num_threads())

with open("/home/felipe/PRs/mlp_adsorption/flames/data/lj_params.json", "r") as f:
    lj_params = json.loads(f.read())

FrameworkPath = "mg-mof-74.cif"
AdsorbatePath = "co2.xyz"

ewald = EwaldSum(R_cutoff=5.5, G_cutoff_N=5, alpha=5 / 15)
lj = CustomLennardJones(lj_params, vdw_cutoff=12.5)

calc = mixing.SumCalculator([lj, ewald])

# Load the framework structure
framework: ase.Atoms = read_cif(FrameworkPath)  # type: ignore

# Load the adsorbate structure
adsorbate: ase.Atoms = read(AdsorbatePath)  # type: ignore

Temperature = 298.0  # in Kelvin
pressure = 1_000  # in Pa = 0.1 bar
MCSteps = 30_000


print(
    f"Running GCMC simulation for pressure: {pressure:.2f} Pa at temperature: {Temperature:.2f} K"
)

gcmc = GCMC(
    model=calc,
    framework_atoms=framework,
    adsorbate_atoms=adsorbate,
    temperature=Temperature,
    pressure=pressure,
    device="cpu",
    vdw_radii=vdw_radii,
    vdw_factor=0.6,
    save_frequency=1,
    debug=True,
    output_to_file=True,
    criticalTemperature=304.1282,
    criticalPressure=7377300.0,
    acentricFactor=0.22394,
    random_seed=42,
    cutoff_radius=10.0,
    automatic_supercell=True,
)


gcmc.logger.print_header()

gcmc.run(MCSteps)

gcmc.logger.print_summary()

gcmc.save_results()
