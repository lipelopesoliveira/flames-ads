import os

# Hide UserWarning and RuntimeWarning messages
import warnings

import ase
import torch
from ase import units
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
      &MGRID
        NGRIDS 5
      &END MGRID
      &SCF
         SCF_GUESS MOPAC
         EPS_SCF 1e-07
         &OT
            MINIMIZER DIIS
            N_DIIS 7
            PRECONDITIONER FULL_ALL
            STEPSIZE -1.0
         &END OT
         &OUTER_SCF
            MAX_SCF 3
            EPS_SCF 1e-07
         &END OUTER_SCF
         &MIXING
            METHOD DIRECT_P_MIXING
            ALPHA 0.4
         &END MIXING
      &END SCF
      &XC
         &VDW_POTENTIAL
            POTENTIAL_TYPE PAIR_POTENTIAL
            &PAIR_POTENTIAL
               TYPE DFTD3(BJ)
               REFERENCE_FUNCTIONAL PBE
               R_CUTOFF 16
               PARAMETER_FILE_NAME dftd3.dat
            &END PAIR_POTENTIAL
         &END VDW_POTENTIAL
      &END XC
    &END FORCE_EVAL
"""

model = CP2K(
    print_level="MEDIUM",
    xc="PBE",
    potential_file="GTH_POTENTIALS",
    basis_set_file=os.path.join(os.getcwd(), "BASIS_MOLOPT_UZH"),
    basis_set="TZVP-MOLOPT-PBE-GTH",
    cutoff=800 * units.Ry,
    inp=inp,
    max_scf=20,
)

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
