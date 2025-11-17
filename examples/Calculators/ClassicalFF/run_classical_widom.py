import json

import ase
from ase.calculators import mixing
from ase.data import vdw_radii
from ase.io import read
from numba import get_num_threads, set_num_threads

from flames.calculators import CustomLennardJones, EwaldSum
from flames.utilities import read_cif
from flames.widom import Widom

NUM_THREADS_TO_USE = 25
set_num_threads(NUM_THREADS_TO_USE)

print(get_num_threads())


with open("/home/felipe/PRs/mlp_adsorption/flames/data/UFF_lj_params.json", "r") as f:
    uff_lj_params = json.loads(f.read())

with open("/home/felipe/PRs/mlp_adsorption/flames/data/TraPPE_lj_params.json", "r") as f:
    trappe_lj_params = json.loads(f.read())

FrameworkPath = "MgMOF-74_DDEC.cif"
AdsorbatePath = "co2_labels.xyz"

ewald = EwaldSum(R_cutoff=5.5, G_cutoff_N=5, alpha=5 / 15)
lj = CustomLennardJones({**uff_lj_params, **trappe_lj_params}, vdw_cutoff=12.5)

calc = mixing.SumCalculator([lj, ewald])

# Load the framework structure
framework: ase.Atoms = read_cif(FrameworkPath)  # type: ignore

# Load the adsorbate structure
adsorbate: ase.Atoms = read(AdsorbatePath)  # type: ignore

Temperature = 298.0

NSteps = 30000

widom = Widom(
    model=calc,  # type: ignore
    framework_atoms=framework,
    adsorbate_atoms=adsorbate,
    temperature=Temperature,
    device="cpu",
    vdw_radii=vdw_radii,
    debug=False,
    output_to_file=True,
    random_seed=42,
    cutoff_radius=12.5,
    automatic_supercell=True,
)

widom.logger.print_header()

widom.run(NSteps)
widom.logger.print_summary()
widom.save_results()
