import os

# Hide UserWarning and RuntimeWarning messages
import warnings

import ase
import torch
from ase.calculators.cp2k import CP2K
from ase.data import vdw_radii
from ase.io import read

from flames.widom import Widom

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = "cuda" if torch.cuda.is_available() else "cpu"

FrameworkPath = "MgMOF-74.cif"
AdsorbatePath = "co2.xyz"

inp = """&FORCE_EVAL
    &DFT
      CHARGE 0
      MULTIPLICITY 1
      &SCF
         SCF_GUESS ATOMIC
         EPS_SCF 1e-08
         &OT
            MINIMIZER DIIS
            N_DIIS 7
            PRECONDITIONER FULL_SINGLE_INVERSE
            LINESEARCH 3PNT
         &END OT
         &OUTER_SCF
            MAX_SCF 5
            EPS_SCF 1e-08
         &END OUTER_SCF
         &MIXING
            METHOD DIRECT_P_MIXING
            ALPHA 0.2
         &END MIXING
      &END SCF
      &PRINT
         &MULLIKEN OFF
         &END MULLIKEN
         &LOWDIN OFF
         &END LOWDIN
         &HIRSHFELD OFF
         &END HIRSHFELD
      &END PRINT
      &QS
         METHOD XTB
         &XTB
            CHECK_ATOMIC_CHARGES .FALSE.
            DO_EWALD .TRUE.
         &END XTB
      &END QS
   &END DFT
    &END FORCE_EVAL
"""

model = CP2K(print_level="MEDIUM", xc="PBE", potential_file="GTH_POTENTIALS", inp=inp, max_scf=20)

# Load the framework structure
framework: ase.Atoms = read(FrameworkPath)  # type: ignore

# Load the adsorbate structure
adsorbate: ase.Atoms = read(AdsorbatePath)  # type: ignore

Temperature = 298.0

NSteps = 50000

widom = Widom(
    model=model,
    framework_atoms=framework,
    adsorbate_atoms=adsorbate,
    temperature=Temperature,
    device=device,
    vdw_radii=vdw_radii,
    debug=False,
    output_to_file=True,
    random_seed=42,
    cutoff_radius=10.0,
    automatic_supercell=True,
)

widom.logger.print_header()

widom.run(NSteps)
widom.logger.print_summary()
