import os

# Hide UserWarning and RuntimeWarning messages
import warnings

import ase
from ase.data import vdw_radii
from ase.io import read
from deepmd.calculator import DP  # type: ignore

from flames.gcmc import GCMC

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

FrameworkPath = "cubic_structure_F.cif"
AdsorbatePath = "co2_label.xyz"

# ORIGINAL MAP MLP_order_Al_C_O_H_Oco2_Cco2
model = DP(
    model="../frozen_model_compressed.pb",
    type_dict={"Al": 0, "C": 1, "O": 2, "H": 3, "Os": 4, "Co": 5},
)

# Load the framework structure
framework: ase.Atoms = read(FrameworkPath)  # type: ignore

# Load the adsorbate structure
adsorbate: ase.Atoms = read(AdsorbatePath)  # type: ignore

adsorbate.set_masses([12.011, 15.999, 15.999])  # type: ignore

Temperature = 298.0

vdw_radii = vdw_radii.copy()
vdw_radii[27] = vdw_radii[6]  # Co = C
vdw_radii[55] = vdw_radii[6]  # Cs = C
vdw_radii[76] = vdw_radii[8]  # Os = O

MCSteps = 20000
MDSteps = 20000

Pressure = 10_000  # 0.1 bar

print(
    f"Running GCMC simulation for pressure: {Pressure:.2f} Pa at temperature: {Temperature:.2f} K"
)

gcmc = GCMC(
    model=model,
    framework_atoms=framework,
    adsorbate_atoms=adsorbate,
    temperature=Temperature,
    pressure=Pressure,
    fugacity_coeff=1,
    device="cpu",
    vdw_radii=vdw_radii,
    vdw_factor=0.6,
    debug=False,
    output_to_file=True,
    criticalTemperature=304.1282,
    criticalPressure=7377300.0,
    acentricFactor=0.22394,
)

gcmc.logger.print_header()

for j in range(5):
    gcmc.run(MCSteps)
    gcmc.npt(nsteps=MDSteps, time_step=0.5, mode="aniso_flex")

gcmc.run(MCSteps * 5)
gcmc.logger.print_summary()
gcmc.save_results()
