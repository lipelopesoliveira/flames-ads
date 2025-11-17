import datetime
import json
import os
from typing import Union

import ase
import numpy as np
import pymser
from ase import units
from ase.calculators import calculator
from ase.io import Trajectory, read
from tqdm import tqdm

from flames import VERSION
from flames.base_simulator import BaseSimulator
from flames.eos import PengRobinsonEOS
from flames.logger import GCMCLogger
from flames.operations import (
    check_overlap,
    random_mol_insertion,
    random_rotation_limited,
    random_translation,
)
from flames.utilities import check_weights


class GCMC(BaseSimulator):
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
        max_overlap_tries: int = 100,
        max_translation: float = 1.0,
        max_rotation: float = np.radians(15),
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
        criticalTemperature: Union[float, None] = None,
        criticalPressure: Union[float, None] = None,
        acentricFactor: Union[float, None] = None,
        LLM: bool = True,
        move_weights: dict = {
            "insertion": 0.25,
            "deletion": 0.25,
            "translation": 0.25,
            "rotation": 0.25,
        },
    ):
        """
        Base class for Grand Canonical Monte Carlo (GCMC) simulations using ASE.

        This clas employs Monte Carlo simulations under the grand canonical ensemble (μVT) ensemble
        to study the adsorption of molecules in a framework material. It allows for movements such as
        insertion, deletion, translation, and rotation of adsorbate molecules within the framework.

        Currently, it supports any ASE-compatible calculator for energy calculations.

        Parameters
        ----------
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
        max_deltaE : float, optional
            Maximum energy difference (in eV) to consider for acceptance criteria.
            This is used to avoid overflow due to problematic calculations (default is 1.555 eV / 150 kJ/mol).
        vdw_factor : float, optional
            Factor to scale the Van der Waals radii (default is 0.6).
        max_overlap_tries : int, optional
            Maximum number of tries to insert/move a molecule without overlap (default is 100).
        max_translation : float, optional
            Maximum translation distance (default is 1.0).
        max_rotation : float, optional
            Maximum rotation angle (in radians) (default is 15 degrees).
        save_frequency : int, optional
            Frequency at which to save the simulation state and results (default is 100).
        save_rejected : bool, optional
            If True, saves the rejected moves in a trajectory file (default is False).
        output_to_file : bool, optional
            If True, writes the output to a file named 'GCMC_Output.out' in the 'results' directory
            (default is True).
        output_folder : str | None, optional
            Folder to save the output files. If None, a folder named 'results_<T>_<P>' will be created.
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
        criticalTemperature : float, optional
            Critical temperature of the adsorbate in Kelvin.
        criticalPressure : float, optional
            Critical pressure of the adsorbate in Pascal.
        acentricFactor : float, optional
            Acentric factor of the adsorbate.
        LLM : bool, optional
            Use Leftmost Local Minima on the determination of the equilibration point. This can underestimate the
            equilibration point in some situations, but generate good averages for well-behaved scenarios.
        move_weights : dict, optional
            A dictionary containing the move weights for 'insertion', 'deletion', 'translation', and 'rotation'.
            Default is equal weights for all moves.
        """

        super().__init__(
            model=model,
            framework_atoms=framework_atoms,
            adsorbate_atoms=adsorbate_atoms,
            temperature=temperature,
            pressure=pressure,
            device=device,
            vdw_radii=vdw_radii,
            vdw_factor=vdw_factor,
            max_deltaE=max_deltaE,
            save_frequency=save_frequency,
            save_rejected=save_rejected,
            output_to_file=output_to_file,
            output_folder=output_folder,
            debug=debug,
            fugacity_coeff=fugacity_coeff,
            random_seed=random_seed,
            cutoff_radius=cutoff_radius,
            automatic_supercell=automatic_supercell,
        )

        self.logger = GCMCLogger(simulation=self, output_file=self.out_file)

        self.start_time = datetime.datetime.now()

        # Parameters for calculateing the Peng-Robinson equation of state
        self.criticalTemperature = criticalTemperature
        self.criticalPressure = criticalPressure
        self.acentricFactor = acentricFactor

        # Check if any critical parameters are not None
        if all([self.criticalTemperature, self.criticalPressure, self.acentricFactor]):
            self.eos = PengRobinsonEOS(
                temperature=self.T,
                pressure=self.P,
                criticalTemperature=self.criticalTemperature,  # type: ignore
                criticalPressure=self.criticalPressure,  # type: ignore
                acentricFactor=self.acentricFactor,  # type: ignore
                molarMass=self.adsorbate_mass,
            )
            self.fugacity_coeff = self.eos.get_fugacity_coefficient()

        # Parameters for storing the main results during the simulation
        self.N_ads: int = 0
        self.uptake_list: list[int] = []
        self.total_energy_list: list[float] = []
        self.total_ads_list: list[float] = []

        self.move_weights = check_weights(move_weights)

        self.max_translation = max_translation
        self.max_rotation = max_rotation

        self.mov_dict: dict = {"insertion": [], "deletion": [], "translation": [], "rotation": []}

        self.movements: dict = {
            "insertion": self.try_insertion,
            "deletion": self.try_deletion,
            "rotation": self.try_rotation,
            "translation": self.try_translation,
        }

        # Base iteration for restarting the simulation. This is for tracking the iteration count only
        self.base_iteration: int = 0

        # Maximum number of tries to insert a molecule without overlap
        self.max_overlap_tries = max_overlap_tries

        # Dictionary to store the equilibrated results by pyMSER
        self.equilibrated_results: dict = {}

        # Uses Leftmost Local Minima
        self.LLM = LLM

    def _save_rejected_if_enabled(self, atoms_trial: ase.Atoms) -> None:
        """
        Helper to conditionally write the rejected configuration to the trajectory.

        Parameters
        ----------
        atoms_trial : ase.Atoms
            The trial configuration that was rejected.
        """
        if self.save_rejected:
            self.rejected_trajectory.write(atoms_trial)  # type: ignore

    def restart(self) -> None:
        """
        Restart the simulation from the last state.

        This method loads the last saved state from the trajectory file and restores the simulation to that state.
        It also loads the uptake, total energy, and total adsorbates lists from the saved files if they exist.
        """

        print("Restarting simulation...")
        uptake_restart, total_energy_restart, total_ads_restart = [], [], []

        if os.path.exists(os.path.join(self.out_folder, f"uptake_{self.P:.5f}.npy")):
            uptake_restart = np.load(
                os.path.join(self.out_folder, f"uptake_{self.P:.5f}.npy")
            ).tolist()

        if os.path.exists(os.path.join(self.out_folder, f"total_energy_{self.P:.5f}.npy")):
            total_energy_restart = np.load(
                os.path.join(self.out_folder, f"total_energy_{self.P:.5f}.npy")
            ).tolist()

        if os.path.exists(os.path.join(self.out_folder, f"total_ads_{self.P:.5f}.npy")):
            total_ads_restart = np.load(
                os.path.join(self.out_folder, f"total_ads_{self.P:.5f}.npy")
            ).tolist()

        # Check if the len of all restart elements are the same:
        if not (len(uptake_restart) == len(total_energy_restart) == len(total_ads_restart)):
            raise ValueError(
                f"""
            The lengths of uptake, total energy, and total adsorbates lists do not match.
            Please check the saved files.
            Found lengths: {len(uptake_restart)}, {len(total_energy_restart)}, {len(total_ads_restart)}
            for uptake, total energy, and total ads respectively."""
            )

        self.uptake_list = uptake_restart
        self.total_energy_list = total_energy_restart
        self.total_ads_list = total_ads_restart

        # Set the base iteration to the length of the uptake list
        self.base_iteration = len(self.uptake_list)

        self.logger.print_restart_info()

        self.load_state(os.path.join(self.out_folder, "Movies", "Trajectory.traj"))

    def load_state(self, state_file: str) -> None:
        """
        Load the state of the simulation from a file.

        Parameters
        ----------
        state_file : str
            Path to the file containing the saved state of the simulation.
        """
        print(f"Loading state from {state_file}...")

        if not os.path.exists(state_file):
            raise FileNotFoundError(f"State file '{state_file}' does not exist.")

        if state_file.endswith(".traj"):
            state = Trajectory(state_file, "r")[-1]  # type: ignore
        else:
            state: ase.Atoms = read(state_file)  # type: ignore

        self.set_state(state)

        self.N_ads = int((len(state) - self.n_atoms_framework) / len(self.adsorbate))
        average_binding_energy = (
            (self.current_total_energy - self.framework_energy - self.N_ads * self.adsorbate_energy)
            / (units.kJ / units.mol)
            / self.N_ads
            if self.N_ads > 0
            else 0
        )

        self.logger.print_load_state_info(
            n_atoms=len(state), average_ads_energy=average_binding_energy
        )

    def equilibrate(
        self,
        batch_size: Union[int, bool] = False,
        run_ADF: bool = False,
        uncertainty: str = "uSD",
    ) -> None:
        """
        Use pyMSER to get the equilibrated statistics of the simulation.

        Parameters
        ----------
        LLM : bool
            If True, use the Leftmost-Local Minima (LLM) method to determine the equilibration time.
            This is only recommended for high-throughput simulations, and sometimes can underestimate
            the true equilibration point.
            Default is True.
        batch_size : int
            Batch size to use for speedup the equilibration process. Default is 100.
        run_ADF : bool
            If True, run the Augmented Dickey-Fuller (ADF) test to confirm for stationarity.
            Default is False.
        uncertainty : str
            The type of uncertainty to use for the equilibration process. Default is "uSD".
            Options are:
            - "uSD": uncorrelated Standard Deviation
            - "uSE": uncorrelated Standard Error
            - "SD": Standard Deviation
            - "SE": Standard Error

        """

        eq_results = pymser.equilibrate(
            self.uptake_list,
            LLM=self.LLM,
            batch_size=int(len(self.uptake_list) / 50) if batch_size is False else batch_size,
            ADF_test=run_ADF,
            uncertainty=uncertainty,
            print_results=False,
        )

        enthalpy, enthalpy_sd = pymser.calc_equilibrated_enthalpy(
            energy=np.array(self.total_ads_list) / units.kB,  # Convert to K
            number_of_molecules=self.uptake_list,
            temperature=self.T,
            eq_index=eq_results["t0"],
            uncertainty="SD",
            ac_time=int(eq_results["ac_time"]),
        )

        eq_results["average"] = float(eq_results["average"])
        eq_results["uncertainty"] = float(eq_results["uncertainty"])
        eq_results["ac_time"] = int(eq_results["ac_time"])
        eq_results["uncorr_samples"] = int(eq_results["uncorr_samples"])

        eq_results["equilibrated"] = eq_results["t0"] < 0.75 * len(self.uptake_list)

        eq_results["enthalpy_kJ_per_mol"] = float(enthalpy)
        eq_results["enthalpy_sd_kJ_per_mol"] = float(enthalpy_sd)

        self.equilibrated_results = eq_results

    def save_results(
        self,
        file_name: str = "GCMC_Results.json",
        batch_size: Union[int, bool] = False,
        run_ADF: bool = False,
        uncertainty: str = "uSD",
    ) -> None:
        """
        Save a json file with the main results of the simulation.

        Parameters
        ----------
        file_name : str
            Name of the output file. Default is 'GCMC_Results.json'.
        LLM : bool
            If True, use the Leftmost-Local Minima (LLM) method to determine the equilibration time.
            This is only recommended for high-throughput simulations, and sometimes can underestimate
            the true equilibration point.
            Default is True.
        batch_size : int
            Batch size to use for speedup the equilibration process.
            Default is False, which means 2% of the total number of steps.
        run_ADF : bool
            If True, run the Augmented Dickey-Fuller (ADF) test to confirm for stationarity.
            Default is False.
        uncertainty : str
            The type of uncertainty to use for the equilibration process. Default is "uSD".
            Options are:
            - "uSD": uncorrelated Standard Deviation
            - "uSE": uncorrelated Standard Error
            - "SD": Standard Deviation
            - "SE": Standard Error

        """

        self.equilibrate(batch_size=batch_size, run_ADF=run_ADF, uncertainty=uncertainty)

        results = {
            "simulation": {
                "code_version": VERSION,
                "random_seed": self.random_seed,
                "temperature_K": self.T,
                "pressure_Pa": self.P,
                "fugacity_coefficient": self.fugacity_coeff,
                "fugacity_Pa": self.fugacity_coeff * self.P,
                "move_weights": self.move_weights,
                "n_steps": len(self.uptake_list),
                "enlapsed_time_hours": (datetime.datetime.now() - self.start_time).total_seconds()
                / 3600,
            },
            "equilibration": {
                "LLM": self.LLM,
                "t0": self.equilibrated_results.get("t0", None),
                "average": self.equilibrated_results.get("average", None),
                "uncertainty": self.equilibrated_results.get("uncertainty", None),
                "equilibrated": self.equilibrated_results.get("equilibrated", None),
                "ac_time": self.equilibrated_results.get("ac_time", None),
                "uncorr_samples": self.equilibrated_results.get("uncorr_samples", None),
            },
            "enthalpy": {
                "kJ_mol": {
                    "mean": self.equilibrated_results.get("enthalpy_kJ_per_mol", None),
                    "sd": self.equilibrated_results.get("enthalpy_sd_kJ_per_mol", None),
                }
            },
        }

        # --- Uptake data (computed from conversion factors) ---
        avrg = self.equilibrated_results.get("average", 0)
        stdv = self.equilibrated_results.get("uncertainty", 0)

        results["uptake"] = {
            unit: {
                "mean": avrg * factor,
                "sd": stdv * factor,
            }
            for unit, factor in self.conv_factors.items()
        }

        with open(os.path.join(self.out_folder, file_name), "w") as f:
            json.dump(results, f, indent=4)

    def _insertion_acceptance(self, deltaE) -> bool:
        """
        Calculate the acceptance probability for insertion of an adsorbate molecule as

        Pacc (N -> N + 1) = min(1, β * V * f * exp(-β ΔE) / (N + 1))
        """

        exp_value = np.exp(-self.beta * deltaE)

        pre_factor = self.V * self.beta * self.fugacity / (self.N_ads + 1)

        acc = min(1, pre_factor * exp_value)

        rnd_number = self.rnd_generator.random()

        if self.debug:
            self.logger.print_debug_movement(
                movement="Insertion",
                deltaE=deltaE,
                prefactor=pre_factor,
                acc=acc,
                rnd_number=rnd_number,
            )

        # Apply Metropolis acceptance/rejection rule
        return rnd_number < acc

    def _deletion_acceptance(self, deltaE) -> bool:
        """
        Calculate the acceptance probability for deletion of an adsorbate molecule as

        Pdel (N -> N - 1 ) = min(1, N / (β * V * f) * exp(-β ΔE) )
        """

        exp_value = np.exp(-self.beta * deltaE)

        pre_factor = self.N_ads / (self.V * self.beta * self.fugacity)

        acc = min(1, pre_factor * exp_value)

        rnd_number = self.rnd_generator.random()

        if self.debug:
            self.logger.print_debug_movement(
                movement="Deletion",
                deltaE=deltaE,
                prefactor=pre_factor,
                acc=acc,
                rnd_number=rnd_number,
            )

        # Apply Metropolis acceptance/rejection rule
        return rnd_number < acc

    def _move_acceptance(self, deltaE, movement_name="Movement") -> bool:
        """
        Calculate the acceptance probability for translation or rotation of an adsorbate molecule as

        Pmove = min(1, exp(-β ΔE))
        """

        exp_value = np.exp(-self.beta * deltaE)
        acc = min(1, exp_value)

        rnd_number = self.rnd_generator.random()

        if self.debug:
            self.logger.print_debug_movement(
                movement=movement_name, deltaE=deltaE, prefactor=1, acc=acc, rnd_number=rnd_number
            )

        # Apply Metropolis acceptance/rejection rule
        return rnd_number < acc

    def _save_state(self, actual_iteration: int) -> None:

        if actual_iteration % self.save_every == 0:

            self.trajectory.write(self.current_system)  # type: ignore

            np.save(
                os.path.join(self.out_folder, f"uptake_{self.P:.5f}.npy"),
                np.array(self.uptake_list),
            )

            np.save(
                os.path.join(self.out_folder, f"total_energy_{self.P:.5f}.npy"),
                np.array(self.total_energy_list),
            )

            np.save(
                os.path.join(self.out_folder, f"total_ads_{self.P:.5f}.npy"),
                np.array(self.total_ads_list),
            )

    def try_insertion(self) -> bool:
        """
        Try to insert a new adsorbate molecule into the framework.
        This method randomly places the adsorbate in the framework and checks for van der Waals overlap.
        If there is no overlap, it calculates the new potential energy and decides whether to accept the insertion
        based on the acceptance criteria.
        If after a number of tries (self.max_overlap_tries) no valid position is found, the insertion is rejected.

        Returns
        -------
        bool
            True if the insertion was accepted, False otherwise.
        """

        for _ in range(self.max_overlap_tries):
            atoms_trial = random_mol_insertion(
                self.current_system, self.adsorbate, self.rnd_generator
            )

            overlaped = check_overlap(
                atoms=atoms_trial,
                group1_indices=np.arange(len(self.current_system)),
                group2_indices=np.arange(len(self.current_system), len(atoms_trial)),
                vdw_radii=self.vdw,
            )

            if not overlaped:
                break
        else:
            return False

        # Energy calculation
        atoms_trial.calc = self.model
        e_new = atoms_trial.get_potential_energy()

        deltaE = e_new - self.current_total_energy - self.adsorbate_energy

        if np.abs(deltaE) > np.abs(self.max_deltaE):
            self._save_rejected_if_enabled(atoms_trial)
            return False

        # Apply the acceptance criteria for insertion
        if self._insertion_acceptance(deltaE=deltaE):
            self.current_system = atoms_trial.copy()
            self.current_total_energy = e_new
            self.N_ads += 1
            return True

        self._save_rejected_if_enabled(atoms_trial)
        return False

    def try_deletion(self) -> bool:
        """
        Try to delete an adsorbate molecule from the framework.
        This method randomly selects an adsorbate molecule and try to apply the deletion.

        If there are no adsorbates, it returns False.

        Returns
        -------
        bool
            True if the deletion was accepted, False otherwise.
        """
        if self.N_ads == 0:
            return False

        # Randomly select an adsorbate molecule to delete
        i_ads = self.rnd_generator.integers(low=0, high=self.N_ads, size=1)[0]

        # Get the indices of the adsorbate atoms to be deleted
        i_start = self.n_atoms_framework + self.n_adsorbate_atoms * i_ads
        i_end = self.n_atoms_framework + self.n_adsorbate_atoms * (i_ads + 1)

        # Create a trial system for the deletion
        atoms_trial = self.current_system.copy()
        atoms_trial.calc = self.model  # type: ignore

        # Delete the adsorbate atoms from the trial structure
        del atoms_trial[i_start:i_end]

        # Calculate the new potential energy of the trial structure
        e_new = atoms_trial.get_potential_energy()  # type: ignore

        deltaE = e_new + self.adsorbate_energy - self.current_total_energy

        if np.abs(deltaE) > np.abs(self.max_deltaE):
            return False

        # Apply the acceptance criteria for deletion
        if self._deletion_acceptance(deltaE=deltaE):

            self.current_system = atoms_trial.copy()
            self.current_total_energy = e_new
            self.N_ads -= 1

            return True
        else:
            return False

    def try_translation(self) -> bool:
        """
        Try to translate an adsorbate molecule within the framework.
        This method randomly selects an adsorbate molecule and applies a random translation.
        It checks for van der Waals overlap and calculates the new potential energy.

        Returns
        -------
        bool
            True if the translation was accepted, False otherwise.
        """

        if self.N_ads == 0:
            return False

        for _ in range(self.max_overlap_tries):
            i_ads = self.rnd_generator.integers(low=0, high=self.N_ads, size=1)[0]
            atoms_trial = self.current_system.copy()

            pos = atoms_trial.get_positions()  # type: ignore

            i_start = self.n_atoms_framework + self.n_adsorbate_atoms * i_ads
            i_end = self.n_atoms_framework + self.n_adsorbate_atoms * (i_ads + 1)

            pos[i_start:i_end] = random_translation(
                original_position=pos[i_start:i_end],
                cell=self.current_system.cell.array,
                max_translation=self.max_translation,
                rnd_generator=self.rnd_generator,
            )

            atoms_trial.set_positions(pos)  # type: ignore

            overlaped = check_overlap(
                atoms=atoms_trial,
                group1_indices=np.concatenate(
                    [np.arange(0, i_start), np.arange(i_end, len(atoms_trial))]
                ),
                group2_indices=np.arange(i_start, i_end),
                vdw_radii=self.vdw,
            )

            if not overlaped:
                break
        else:
            return False

        atoms_trial.calc = self.model  # type: ignore
        e_trial = atoms_trial.get_potential_energy()  # type: ignore

        deltaE = e_trial - self.current_total_energy

        if np.abs(deltaE) > np.abs(self.max_deltaE):
            self._save_rejected_if_enabled(atoms_trial)
            return False

        if self._move_acceptance(deltaE=deltaE, movement_name="Translation"):
            self.current_system = atoms_trial.copy()
            self.current_total_energy = e_trial
            return True
        else:
            self._save_rejected_if_enabled(atoms_trial)
            return False

    def try_rotation(self) -> bool:
        """
        Try to rotate an adsorbate molecule within the framework.
        This method randomly selects an adsorbate molecule and applies a random rotation.
        It checks for van der Waals overlap and calculates the new potential energy.

        Returns
        -------
        bool
            True if the rotation was accepted, False otherwise.
        """

        if self.N_ads == 0:
            return False

        for _ in range(self.max_overlap_tries):

            i_ads = self.rnd_generator.integers(low=0, high=self.N_ads, size=1)[0]
            atoms_trial = self.current_system.copy()

            pos = atoms_trial.get_positions()  # type: ignore
            i_start = self.n_atoms_framework + self.n_adsorbate_atoms * i_ads
            i_end = self.n_atoms_framework + self.n_adsorbate_atoms * (i_ads + 1)

            pos[i_start:i_end] = random_rotation_limited(
                original_position=pos[i_start:i_end],
                cell=self.current_system.cell.array,
                rnd_generator=self.rnd_generator,
                theta_max=self.max_rotation,
            )
            atoms_trial.set_positions(pos)  # type: ignore

            overlaped = check_overlap(
                atoms=atoms_trial,
                group1_indices=np.concatenate(
                    [np.arange(0, i_start), np.arange(i_end, len(atoms_trial))]
                ),
                group2_indices=np.arange(i_start, i_end),
                vdw_radii=self.vdw,
            )

            if not overlaped:
                break
        else:
            return False

        atoms_trial.calc = self.model  # type: ignore
        e_trial = atoms_trial.get_potential_energy()  # type: ignore

        deltaE = e_trial - self.current_total_energy

        if np.abs(deltaE) > np.abs(self.max_deltaE):
            self._save_rejected_if_enabled(atoms_trial)
            return False

        if self._move_acceptance(deltaE=deltaE, movement_name="Rotation"):
            self.current_system = atoms_trial.copy()
            self.current_total_energy = e_trial
            return True
        else:
            self._save_rejected_if_enabled(atoms_trial)
            return False

    def _pick_random_move(self) -> str:
        """
        Randomly select a move from the `move_weights` dict.
        If there is no molecule on the system, always return insertion.

        Returns
        -------
        move: str
        """

        if self.N_ads == 0:
            move = "insertion"

        else:
            move = self.rnd_generator.choice(
                a=list(self.move_weights.keys()), p=list(self.move_weights.values())
            )

        return move

    def get_average_ads_energy(self) -> float:
        """
        Compute the average adsorption energy per adsorbed molecule.

        The adsorption energy is calculated as:
            E_ads_avg = [E_total - (N_ads * E_adsorbate) - E_framework] / N_ads

        where:
            - E_total: current total energy of the system (simulation)
            - N_ads: number of adsorbed molecules
            - E_adsorbate: energy of an isolated adsorbate molecule
            - E_framework: energy of the empty framework

        The result is converted from simulation units to kJ/mol per adsorbate.

        Returns
        -------
        float
            The average adsorption energy per molecule in kJ/mol.
            Returns 0.0 if no molecules are adsorbed (N_ads == 0).
        """

        if self.N_ads == 0:
            return 0.0

        # Total adsorption energy (system - framework - isolated adsorbates * n adsorbates)
        adsorption_energy_total = (
            self.current_total_energy - self.framework_energy - (self.N_ads * self.adsorbate_energy)
        )

        # Convert to kJ/mol and normalize per adsorbate
        E_ads_avg = adsorption_energy_total / (units.kJ / units.mol) / self.N_ads

        return E_ads_avg

    def step(self, iteration: int):

        actual_iteration = iteration + self.base_iteration

        step_time_start = datetime.datetime.now()

        # Randomly select a move based on the move weights
        move = self._pick_random_move()

        accepted = self.movements[move]()
        self.mov_dict[move].append(1 if accepted else 0)

        self.uptake_list.append(self.N_ads)
        self.total_energy_list.append(self.current_total_energy)
        self.total_ads_list.append(
            self.current_total_energy - (self.N_ads * self.adsorbate_energy) - self.framework_energy
        )

        average_ads_energy = self.get_average_ads_energy()

        self.logger.print_step_info(
            step=actual_iteration,
            average_ads_energy=average_ads_energy,
            step_time=(datetime.datetime.now() - step_time_start).total_seconds(),
        )

        self._save_state(actual_iteration)

    def run(self, N) -> None:
        """Run the Grand Canonical Monte Carlo simulation for N iterations."""

        self.logger.print_run_header()

        for iteration in tqdm(range(1, N + 1), disable=(self.out_file is None), desc="GCMC Step"):
            self.step(iteration)
