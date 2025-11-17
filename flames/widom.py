import datetime
import json
import os
from typing import Union

import ase
import numpy as np
from ase import units
from ase.calculators import calculator
from ase.io import write
from tqdm import tqdm

from flames import VERSION
from flames.base_simulator import BaseSimulator
from flames.logger import WidomLogger
from flames.operations import check_overlap, random_mol_insertion
from flames.utilities import random_n_splits


class Widom(BaseSimulator):
    def __init__(
        self,
        framework_atoms: ase.Atoms,
        adsorbate_atoms: ase.Atoms,
        temperature: float,
        model: calculator.Calculator,
        vdw_radii: np.ndarray,
        vdw_factor: float = 0.6,
        max_overlap_tries: int = 1000,
        max_deltaE: float = 1.555,
        device: str = "cpu",
        save_frequency: int = 100,
        save_rejected: bool = False,
        output_to_file: bool = True,
        output_folder: Union[str, None] = None,
        debug: bool = False,
        random_seed: Union[int, None] = None,
        cutoff_radius: float = 6.0,
        automatic_supercell: bool = True,
    ) -> None:
        """
        Base class for Widom insertion method using ASE.

        This class performs the Widom insertion method to calculate the enthalpy of adsorption
        and Henry coefficient of an adsorbate in a framework.

        Currently, it supports only one adsorbate molecule and any ASE-compatible calculator
        for energy calculations.

        Parameters
        ----------
        framework_atoms : ase.Atoms
            The empty framework structure where the adsorbate will be inserted.
        adsorbate_atoms : ase.Atoms
            The adsorbate molecule to be inserted into the framework.
        temperature : float
            Temperature of the ideal reservoir in Kelvin.
        model : ase.calculators.calculator.Calculator
            ASE-compatible calculator for energy calculations.
        vdw_radii : np.ndarray
            Van der Waals radii of the atoms in the framework and adsorbate.
        vdw_factor : float, optional
            Factor to scale the Van der Waals radii (default is 0.6).
        max_overlap_tries : int, optional
            Maximum number of tries to insert/move a molecule without overlap (default is 100).
        max_deltaE : float, optional
            Maximum energy difference (in eV) to consider for acceptance criteria.
            This is used to avoid overflow due to problematic calculations (default is 1.555 eV / 150 kJ/mol).
        device : str, optional
            Device to run the calculations on, either 'cpu' or 'cuda'. Default is 'cpu'.
        save_frequency : int, optional
            Frequency at which to save the simulation state and results (default is 100).
        save_rejected : bool, optional
            If True, saves the rejected moves in a trajectory file (default is False).
        output_to_file : bool, optional
            If True, writes the output to a file named 'Widom_Output.out' in the 'results' directory
            (default is True).
        output_folder: Union[str, None], optional
            Folder to save the output files. If None, a folder named 'results_<T>_<P>' will be created.
        debug : bool, optional
            If True, enables debug mode with more verbose output (default is False).
        random_seed : int | None
            Random seed for reproducibility (default is None).
        cutoff_radius : float
            Interaction potential cut-off radius used to estimate the minimum unit cell (default is 6.0).
        automatic_supercell : bool
            Whether to automatically create a supercell based on the cutoff radius (default is True).
        """

        super().__init__(
            model=model,
            framework_atoms=framework_atoms,
            adsorbate_atoms=adsorbate_atoms,
            temperature=temperature,
            pressure=0.0,
            device=device,
            vdw_radii=vdw_radii,
            vdw_factor=vdw_factor,
            max_deltaE=max_deltaE,
            save_frequency=save_frequency,
            save_rejected=save_rejected,
            output_to_file=output_to_file,
            output_folder=output_folder,
            debug=debug,
            fugacity_coeff=0.0,
            random_seed=random_seed,
            cutoff_radius=cutoff_radius,
            automatic_supercell=automatic_supercell,
        )

        self.logger = WidomLogger(simulation=self, output_file=self.out_file)

        self.start_time = datetime.datetime.now()

        self.minimum_configuration: ase.Atoms = self.framework.copy()
        self.minimum_energy: float = 0
        self.base_iteration = 0
        self.N_ads = 0
        self.int_energy_list = np.zeros(1, dtype=float)

        self.boltz_fac = np.exp(-self.beta * self.int_energy_list)

        # kH = β <exp(-β ΔE)> [mol kg-1 Pa-1]
        self.kH = self._compute_kH(self.boltz_fac)

        self.kH_std_dv = 0.0

        # Compute the adsorption energy (Qst)
        self.Qst = self._compute_Qst(self.int_energy_list, self.boltz_fac)

        self.Qst_std_dv = 0.0

        self.max_overlap_tries = max_overlap_tries
        self.MAX_ENERGY_ERROR = 1000.0

    def _compute_kH(self, boltz_fac: np.ndarray) -> float:
        """
        Compute the Henry coefficient (kH) using the Boltzmann factors.

        kH = β <exp(-β ΔE)> [mol kg-1 Pa-1]

        Parameters
        ----------
        boltz_fac : np.ndarray
            Array of Boltzmann factors corresponding to the integral energies.

        Returns
        -------
            float: The Henry coefficient in mol kg-1 Pa-1
        """

        return self.beta * boltz_fac.mean() * units.J / (units.mol * self.framework_density * 1e3)

    def _compute_kH_std(self, boltz_fac: np.ndarray) -> float:
        """
        Compute the standard deviation of the Henry coefficient (kH) using the Boltzmann factors.

        kH = β <exp(-β ΔE)> [mol kg-1 Pa-1]

        Parameters
        ----------
        boltz_fac : np.ndarray
            Array of Boltzmann factors corresponding to the integral energies.

        Returns
        -------
            float: The standard deviation of the Henry coefficient in mol kg-1 Pa-1
        """

        return (
            self.beta
            * boltz_fac.mean(axis=-1)
            * (units.J / units.mol)
            / (self.framework_density * 1e3)
        ).std()

    def _compute_Qst(self, int_energy_list: np.ndarray, boltz_fac: np.ndarray) -> float:
        """
        Compute the adsorption energy (Qst) using the integral energy list and Boltzmann factors.

        Qst = - < ΔE * exp(-β ΔE) > / <exp(-β ΔE)>  + kB.T # [kJ/mol]

        Parameters
        ----------
        int_energy_list : np.ndarray
            Array of integral energies from the Widom insertions.
        boltz_fac : np.ndarray
            Array of Boltzmann factors corresponding to the integral energies.

        Returns
        -------
            float: The Qst energy in kJ/mol
        """
        return ((int_energy_list * boltz_fac).mean() / boltz_fac.mean() - units.kB * self.T) / (
            units.kJ / units.mol
        )

    def _compute_Qst_std(self, int_energy_list: np.ndarray, boltz_fac: np.ndarray) -> float:
        """
        Compute the standard deviation of the adsorption energy (Qst) using the integral energy list and Boltzmann factors.

        Qst = - < ΔE * exp(-β ΔE) > / <exp(-β ΔE)>  + kB.T # [kJ/mol]

        Parameters
        ----------
        int_energy_list : np.ndarray
            Array of integral energies from the Widom insertions.
        boltz_fac : np.ndarray
            Array of Boltzmann factors corresponding to the integral energies.

        Returns
        -------
            float: The standard deviation of Qst energy in kJ/mol
        """
        return (
            (
                (int_energy_list * boltz_fac).mean(axis=-1) / boltz_fac.mean(axis=-1)
                - units.kB * self.T
            )
            / (units.kJ / units.mol)
        ).std()

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

    def _save_state(self, actual_iteration: int) -> None:
        """
        Save the current state of the simulation if the iteration matches the save frequency.

        Parameters
        ----------
        actual_iteration : int
            The current iteration number.
        """

        if actual_iteration % self.save_every == 0:

            np.save(
                os.path.join(self.out_folder, f"int_energy_{self.P:.5f}.npy"),
                np.array(self.int_energy_list),
            )

    def _save_minimum_configuration(self, deltaE: float, atoms_trial: ase.Atoms) -> None:
        """
        Save the minimum energy configuration found during the simulation if the latest insertion has a lower energy.

        Parameters
        ----------
        deltaE : float
            The change in energy associated with the latest insertion.
        atoms_trial : ase.Atoms
            The trial configuration of the latest insertion.
        """

        if deltaE < self.minimum_energy:
            self.minimum_configuration = atoms_trial.copy()
            self.minimum_energy = deltaE
            tmp_name = f"minimum_configuration_{deltaE / (units.kJ / units.mol):.2f}.cif"

            write(
                os.path.join(os.path.join(self.out_folder, "Movies", tmp_name)),
                atoms_trial,
                format="cif",
            )

    def update_statistics(self, deltaE: float) -> None:
        """
        Update the statistics of the Widom insertion method after a new insertion.

        Parameters
        ----------
        deltaE : float
            The change in energy associated with the latest insertion.
        """

        self.int_energy_list = np.append(self.int_energy_list, deltaE)

        self.boltz_fac = np.exp(-self.beta * self.int_energy_list)

        self.kH = self._compute_kH(self.boltz_fac)
        self.Qst = self._compute_Qst(self.int_energy_list, self.boltz_fac)

        # Calculate standard deviation using cross-validation
        if len(self.int_energy_list) > 5:
            cv_int_energy_list = random_n_splits(self.int_energy_list, 5, self.rnd_generator)

            cv_boltz_fac = np.exp(-self.beta * cv_int_energy_list)

            # Calculate standard deviation using cross-validation
            self.kH_std_dv = self._compute_kH_std(cv_boltz_fac)

            self.Qst_std_dv = self._compute_Qst_std(cv_int_energy_list, cv_boltz_fac)

    def save_results(self, file_name: str = "Widom_Results.json") -> None:
        """
        Save a json file with the main results of the simulation.

        Parameters
        ----------
        file_name : str, optional
            Name of the output json file (default is 'Widom_Results.json').
        """

        results = {
            "code_version": VERSION,
            "random_seed": self.random_seed,
            "enlapsed_time_hours": (datetime.datetime.now() - self.start_time).total_seconds()
            / 3600,
            "total_insertions": len(self.int_energy_list),
            "temperature_K": self.T,
            "henry_coefficient_mol_kg-1_Pa-1": self.kH,
            "henry_coefficient_std_mol_kg-1_Pa-1": self.kH_std_dv,
            "enthalpy_of_adsorption_kJ_mol-1": self.Qst,
            "enthalpy_of_adsorption_std_kJ_mol-1": self.Qst_std_dv,
        }

        with open(os.path.join(self.out_folder, file_name), "w") as f:
            json.dump(results, f, indent=4)

    def restart(self) -> None:
        """
        Restart the simulation from the last state.

        This method loads the last saved state from the trajectory file and restores the simulation to that state.
        It also loads the uptake, total energy, and total adsorbates lists from the saved files if they exist.
        """

        print("Restarting simulation...")

        self.int_energy_list = np.load(
            os.path.join(self.out_folder, f"int_energy_{self.P:.5f}.npy")
        )

        # Set the base iteration to the length of the uptake list
        self.base_iteration = len(self.int_energy_list)

        self.logger.print_restart_info()

    def try_insertion(self) -> tuple[float, ase.Atoms]:
        """
        Try to insert a new adsorbate molecule into the framework.
        This method randomly places the adsorbate in the framework and checks for van der Waals overlap.
        If there is no overlap, it calculates the new potential energy and decides whether to accept the insertion
        based on the acceptance criteria.

        Returns
        -------
        deltaE : float
            The change in energy associated with the insertion. If the insertion fails, returns a large value (1000.0).
        atoms_trial : ase.Atoms
            The trial configuration after the insertion attempt.
        """

        # Ensure atoms_trial is always defined so it can be returned in failure cases
        atoms_trial = self.framework.copy()

        for _ in range(max(self.max_overlap_tries, 1)):
            atoms_trial = random_mol_insertion(self.framework, self.adsorbate, self.rnd_generator)

            overlaped = check_overlap(
                atoms=atoms_trial,
                group1_indices=np.arange(self.n_atoms_framework),
                group2_indices=np.arange(
                    self.n_atoms_framework, self.n_atoms_framework + self.n_adsorbate_atoms
                ),
                vdw_radii=self.vdw,
            )

            if not overlaped:
                break
        else:
            # Return MAX_ENERGY_ERROR if no valid insertion found after max tries
            return self.MAX_ENERGY_ERROR, atoms_trial

        # Set the same calculator to the trial atoms
        atoms_trial.calc = self.model

        # Calculate the interaction energy of the trial configuration
        deltaE = atoms_trial.get_potential_energy() - self.framework_energy - self.adsorbate_energy

        # Add interaction energy to the info dictionary
        atoms_trial.info["interaction_energy"] = deltaE

        if np.abs(deltaE) > np.abs(self.max_deltaE):
            return self.MAX_ENERGY_ERROR, atoms_trial  # Return MAX_ENERGY_ERROR to indicate error

        return deltaE, atoms_trial

    def step(self, iteration: int) -> None:
        """
        Run a single iteration of the Widom insertion method.

        Parameters
        ----------
        iteration : int
            The current iteration number.
        """
        actual_iteration = iteration + self.base_iteration

        step_time_start = datetime.datetime.now()

        accepted = False

        # Try a valid insertion up to max_overlap_tries times until accepted to count as one Widom insertion
        for _ in range(self.max_overlap_tries):
            deltaE, atoms_trial = self.try_insertion()
            accepted = deltaE < units.kB * self.T

            if accepted:
                break

            self._save_rejected_if_enabled(atoms_trial)

        self._save_minimum_configuration(deltaE, atoms_trial)  # type: ignore

        self.trajectory.write(atoms_trial)  # type: ignore

        # Append int_energy_list
        self.update_statistics(deltaE)  # type: ignore
        self._save_state(actual_iteration)

        self.logger.print_iteration_info(
            [
                actual_iteration,
                deltaE,  # type: ignore
                deltaE / (units.kJ / units.mol),  # type: ignore
                self.kH,
                self.Qst,
                (datetime.datetime.now() - step_time_start).total_seconds(),
            ],
        )

    def run(self, N: int) -> None:

        self.logger.print_header()

        for iteration in tqdm(range(1, N + 1), disable=(self.out_file is None), desc="Widom Step"):
            self.step(iteration)
