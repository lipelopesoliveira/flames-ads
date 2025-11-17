import os
from typing import TextIO, Union

import ase
import numpy as np
from ase import units
from ase.build import make_supercell
from ase.calculators import calculator
from ase.io import Trajectory
from ase.optimize import LBFGS

from flames.ase_utils import (
    crystalOptimization,
    nPT_Berendsen,
    nPT_NoseHoover,
    nVT_Berendsen,
)
from flames.utilities import (
    calculate_unit_cells,
    get_density,
    get_perpendicular_lengths,
)


class BaseSimulator:
    """
    This class handles all base parameters for the simulations.
    Separates the basic logic used in the simulations from the specific implementation details.
    """

    def __init__(
        self,
        model: calculator.Calculator,
        framework_atoms: ase.Atoms,
        adsorbate_atoms: ase.Atoms,
        temperature: float,
        pressure: float,
        device: str,
        vdw_radii: np.ndarray,
        vdw_factor: float = 0.6,
        max_deltaE: float = 1.555,
        save_frequency: int = 100,
        save_rejected: bool = False,
        output_to_file: bool = True,
        output_folder: Union[str, None] = None,
        debug: bool = False,
        fugacity_coeff: float = 1.0,
        random_seed: Union[int, None] = None,
        cutoff_radius: float = 6.0,
        automatic_supercell: bool = True,
    ):
        """
        model : ase.calculators.Calculator
            The calculator to use for energy calculations. Can be any ASE-compatible calculator.
            The outpyt of the calculator should be in eV.
        framework_atoms : ase.Atoms
            The framework structure as an ASE Atoms object.
        adsorbate_atoms : ase.Atoms
            The adsorbate structure as an ASE Atoms object.
        temperature : float
            Temperature of the ideal reservoir in Kelvin.
        pressure : float
            Pressure of the ideal reservoir in Pascal.
        device : str
            Device to run the simulation on, e.g., 'cpu' or 'cuda'.
        vdw_radii : np.ndarray
            Van der Waals radii for the atoms in the framework and adsorbate.
            Should be an array of the same length as the number of atomic numbers in ASE.
        vdw_factor : float, optional
            Factor to scale the Van der Waals radii (default is 0.6).
         max_deltaE : float, optional
            Maximum energy difference (in eV) to consider for acceptance criteria.
            This is used to avoid overflow due to problematic calculations (default is 1.555 eV / 150 kJ/mol).
        save_frequency : int, optional
            Frequency at which to save the simulation state and results (default is 100).
        save_rejected : bool, optional
            If True, saves the rejected moves in a trajectory file (default is False).
        output_to_file : bool, optional
            If True, writes the output to a file named 'GCMC_Output.out' in the 'results' directory
        output_folder : str | None, optional
            Folder to save the output files. If None, a folder named 'results_<T>_<P>' will be created
            based on the temperature and pressure (default is None).
        debug : bool, optional
            If True, prints detailed debug information during the simulation (default is False).
        fugacity_coeff : float, optional
            Fugacity coefficient to correct the pressure. Default is 1.0.
            Only used if `criticalTemperature`, `criticalPressure`, and `acentricFactor` are not provided.
        random_seed : int | None
            Random seed for reproducibility (default is None).
        cutoff_radius : float
            Interaction potential cut-off radius used to estimate the minimum unit cell (default is 6.0).
        automatic_supercell : bool
            If True, automatically creates a supercell based on the cutoff radius (default is True).
        """

        self.random_seed = random_seed if random_seed is not None else np.random.randint(1, 1000000)
        self.rnd_generator = np.random.default_rng(self.random_seed)
        self.cutoff = cutoff_radius
        self.automatic_supercell = automatic_supercell

        # -- General definitions for output --
        if output_folder is not None:
            self.out_folder = output_folder
        else:
            self.out_folder = f"results_{temperature:.2f}_{pressure:.2f}"
        os.makedirs(self.out_folder, exist_ok=True)
        os.makedirs(os.path.join(self.out_folder, "Movies"), exist_ok=True)

        if output_to_file:
            self.out_file: Union[TextIO, None] = open(
                os.path.join(self.out_folder, "Output.out"), "a"
            )
        else:
            self.out_file = None

        self.trajectory = Trajectory(
            os.path.join(self.out_folder, "Movies", "Trajectory.traj"),
            "a",
        )

        self.save_rejected = save_rejected

        if self.save_rejected:
            self.rejected_trajectory = Trajectory(
                os.path.join(self.out_folder, "Movies", "Trajectory_rejected.traj"),
                "a",
            )

        self.save_every = save_frequency
        self.debug = debug

        # -- General definitions for framework and adsorbate --

        self.model = model
        self.device = device

        self.set_framework(framework_atoms)
        self.set_adsorbate(adsorbate_atoms)

        self.current_system = self.framework.copy()
        self.current_system.calc = self.model
        self.current_total_energy = self.current_system.get_potential_energy()

        # General definitions for simulation parameters
        self.T = temperature
        self.P = pressure

        self.fugacity_coeff = fugacity_coeff

        # Fugacity from Pa (J/m^3) to eV / m^3
        self.fugacity = (self.P * self.fugacity_coeff * units.J) / units.eV

        self.beta = 1 / (units.kB * self.T)  # Boltzmann weight, 1 / [eV/K * K]

        atm2pa = 101325
        mol2cm3 = units.kB / units.J * units.mol * 273.15 / atm2pa

        self.conv_factors = {
            "nmol": 1,
            "mol/kg": (1 / units.mol) / self.get_framework_mass(),
            "mg/g": (self.adsorbate_mass * 1e3) / self.get_framework_mass(),
            "cm^3 STP/gr": mol2cm3 / units.mol / self.get_framework_mass() * 1e3,
            "cm^3 STP/cm^3": 1e6 * mol2cm3 / units.mol / (self.framework.get_volume() * (1e-8**3)),
            "% wt": (self.adsorbate_mass * 1e3) / self.get_framework_mass() * 1e-1,
        }

        # Get the ideal supercell. This will be updated by the set_framework method
        self.ideal_supercell = [1, 1, 1]

        self.vdw: np.ndarray = vdw_radii * vdw_factor  # Adjust van der Waals radii to avoid overlap

        # Replace any NaN value by 1.5 on self.vdw to avoid potential problems
        self.vdw[np.isnan(self.vdw)] = 1.5

        # Use an max deltaE to avoid errors on the model
        self.max_deltaE = max_deltaE

    def get_ideal_supercell(self) -> list[int]:
        """
        Get the ideal supercell dimensions based on the cutoff radius.

        Returns
        -------
        list[int]
            A list of three integers representing the number of unit cells in each dimension.
        """
        return calculate_unit_cells(self.framework.get_cell(), cutoff=self.cutoff)

    def get_framework_mass(self) -> float:
        """
        Calculate the mass of the framework in kg.

        Returns
        -------
        float
            The mass of the framework in kg.
        """
        return float(np.sum(self.framework.get_masses()) / units.kg)

    def set_framework(self, framework_atoms: ase.Atoms) -> None:
        """
        Set the framework structure for the simulation.

        Parameters
        ----------
        framework_atoms : ase.Atoms
            The new framework structure as an ASE Atoms object.
        """
        self.framework = framework_atoms
        self.framework.set_tags(np.zeros(len(self.framework), dtype=int))

        self.ideal_supercell = self.get_ideal_supercell()

        if self.ideal_supercell != [1, 1, 1] and self.automatic_supercell:
            self.framework = make_supercell(self.framework, np.eye(3) * self.ideal_supercell)

        self.cell = np.array(self.framework.get_cell())
        self.perpendicular_cell = get_perpendicular_lengths(self.framework.get_cell()) * np.eye(3)

        self.framework.calc = self.model
        self.framework_energy = self.framework.get_potential_energy()
        self.n_atoms_framework = len(self.framework)

        self.V = np.linalg.det(self.cell) / units.m**3
        self.framework_mass = self.get_framework_mass()

        # Get the framework density in g/cm^3
        self.framework_density = get_density(self.framework)

    def set_adsorbate(self, adsorbate_atoms: ase.Atoms) -> None:
        """
        Set the adsorbate structure for the simulation.

        Parameters
        ----------
        adsorbate_atoms : ase.Atoms
            The new adsorbate structure as an ASE Atoms object.
        """
        self.adsorbate = adsorbate_atoms
        self.adsorbate.set_tags(np.ones(len(self.adsorbate), dtype=int))
        self.adsorbate.calc = self.model
        self.adsorbate.set_cell(self.framework.get_cell())
        self.adsorbate.set_pbc([True, True, True])

        self.adsorbate_energy = self.adsorbate.get_potential_energy()
        self.n_adsorbate_atoms = len(self.adsorbate)
        self.adsorbate_mass = np.sum(self.adsorbate.get_masses()) / units.kg

    def set_state(self, state: ase.Atoms) -> None:
        """
        Set the current state of the simulation.

        Parameters
        ----------
        state : ase.Atoms
            The current state of the simulation as an ASE Atoms object.
        """
        self.current_system = state.copy()
        self.current_system.calc = self.model
        self.current_total_energy = self.current_system.get_potential_energy()

    def optimize_framework(
        self,
        max_steps: int = 1000,
        opt_cell: bool = True,
        fix_symmetry: bool = True,
        hydrostatic_strain: bool = True,
        symm_tol: float = 1e-3,
        max_force: float = 0.05,
    ) -> None:
        """
        Optimize the framework structure using the provided calculator.

        Parameters
        ----------
        max_steps : int, optional
            Maximum number of optimization steps (default is 1000).
        tol : float, optional
            Tolerance for convergence (default is 1e-5).

        Returns
        -------
        ase.Atoms
            The optimized framework structure.
        """

        print(
            """
=======================================================================================================
Start optimizing framework structure...
=======================================================================================================
              """,
            file=self.out_file,
            flush=True,
        )

        resultsDict, optFramework = crystalOptimization(
            atoms_in=self.framework,
            calculator=self.model,
            optimizer=LBFGS,  # type: ignore
            fmax=max_force,
            opt_cell=opt_cell,
            fix_symmetry=fix_symmetry,
            hydrostatic_strain=hydrostatic_strain,
            constant_volume=False,
            scalar_pressure=0,
            max_steps=max_steps,
            trajectory=self.trajectory,  # type: ignore
            verbose=True,
            symm_tol=symm_tol,
            out_file=self.out_file,  # type: ignore
        )

        # Remove any constraints from the optimized framework
        optFramework.set_constraint(None)

        self.set_framework(optFramework.copy())

    def optimize_adsorbate(self, max_steps: int = 1000, max_force: float = 0.05) -> None:
        """
        Optimize the adsorbate structure using the provided calculator.

        Parameters
        ----------
        max_steps : int, optional
            Maximum number of optimization steps (default is 1000).
        symm_tol : float, optional
            Tolerance for symmetry (default is 1e-3).
        max_force : float, optional
            Maximum force tolerance for convergence (default is 0.05 eV/Å).

        Returns
        -------
        ase.Atoms
            The optimized adsorbate structure.
        """

        print(
            """
=======================================================================================================
Start optimizing adsorbate structure...
=======================================================================================================
              """,
            file=self.out_file,
            flush=True,
        )

        resultsDict, optAdsorbate = crystalOptimization(
            atoms_in=self.adsorbate,
            calculator=self.model,
            optimizer=LBFGS,  # type: ignore
            fmax=max_force,
            opt_cell=False,
            fix_symmetry=False,
            hydrostatic_strain=True,
            constant_volume=True,
            scalar_pressure=self.P,
            max_steps=max_steps,
            trajectory="Adsorbate_Optimization.traj",
            verbose=True,
            symm_tol=1e3,
            out_file=self.out_file,  # type: ignore
        )

        self.adsorbate = optAdsorbate.copy()
        self.adsorbate.set_constraint(None)
        self.adsorbate.set_cell(self.framework.get_cell())
        self.adsorbate.calc = self.model
        self.adsorbate_energy = self.adsorbate.get_potential_energy()

    def npt(self, nsteps, time_step: float = 0.5, mode: str = "iso_shape"):
        """
        Run a NPT simulation using the Berendsen thermostat and barostat.

        Parameters
        ----------
        nsteps : int
            Number of steps to run the NPT simulation.
        time_step : float, optional
            Time step for the NPT simulation (default is 0.5 fs).
        mode : str, optional
            The mode of the NPT simulation (default is "iso_shape").
            Can be one of "iso_shape", "aniso_shape", or "aniso_flex".
        """

        allowed_modes = ["iso_shape", "aniso_shape", "aniso_flex"]
        assert mode in allowed_modes, f"Mode must be one of {allowed_modes}."

        if mode == "iso_shape" or mode == "aniso_shape":

            new_state = nPT_Berendsen(
                atoms=self.current_system,
                model=self.model,
                temperature=self.T,
                pressure=self.P * 1e-5,
                compressibility=1e-4,
                time_step=time_step,
                num_md_steps=nsteps,
                isotropic=True if mode == "iso_shape" else False,
                out_folder=self.out_folder,
                out_file=self.out_file,  # type: ignore
                trajectory=self.trajectory,
                output_interval=self.save_every,
                movie_interval=self.save_every,
            )
        else:

            new_state = nPT_NoseHoover(
                atoms=self.current_system,
                model=self.model,
                temperature=self.T,
                pressure=self.P * 1e-5,
                time_step=time_step,
                num_md_steps=nsteps,
                ttime=25.0,
                ptime=75.0,
                B_guess=30,
                out_folder=self.out_folder,
                out_file=self.out_file,  # type: ignore
                trajectory=self.trajectory,
                output_interval=self.save_every,
                movie_interval=self.save_every,
            )

        self.set_state(new_state)

        self.set_framework(new_state[: self.n_atoms_framework].copy())  # type: ignore

    def nvt(self, nsteps, time_step: float = 0.5):
        """
        Run a NVT simulation using the Berendsen thermostat.

        Parameters
        ----------
        nsteps : int
            Number of steps to run the NVT simulation.
        time_step : float, optional
            Time step for the NVT simulation (default is 0.5 fs).
        """

        new_state = nVT_Berendsen(
            atoms=self.current_system,
            model=self.model,
            temperature=self.T,
            time_step=time_step,
            num_md_steps=nsteps,
            out_folder=self.out_folder,
            out_file=self.out_file,  # type: ignore
            trajectory=self.trajectory,
            output_interval=self.save_every,
            movie_interval=self.save_every,
        )

        self.set_state(new_state)
