import os
import sys
sys.path.append('/home/felipe/PRs/mlp_adsorption')

import ase
from ase.data import vdw_radii
from ase.io import read
from flames.calculators import EwaldSum, CustomLennardJones
from ase.calculators import mixing
from flames.utilities import read_cif

from flames.widom import Widom

from numba import set_num_threads, get_num_threads

NUM_THREADS_TO_USE = 25
set_num_threads(NUM_THREADS_TO_USE)

print(get_num_threads())

import json

with open('/home/felipe/PRs/mlp_adsorption/flames/data/lj_params.json', 'r') as f:
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

Temperature = 298.0

NSteps = 30000

widom = Widom(
    model=calc,
    framework_atoms=framework,
    adsorbate_atoms=adsorbate,
    temperature=Temperature,
    device='cpu',
    vdw_radii=vdw_radii,
    debug=False,
    output_to_file=True,
    random_seed=42,
    cutoff_radius=6.5,
    automatic_supercell=True,
)

widom.logger.print_header()

widom.run(NSteps)
widom.logger.print_summary()
widom.save_results()

