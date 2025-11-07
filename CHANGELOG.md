# Changelog

## v[X.Y.Z] - YYYY-MM-DD (Unreleased)

### New Features 🎉

### Fixed 🐛

### Enhanced ✨

### Documentation 📖

### Removed 🗑️

## v[0.4.4] - 2025-11-04

### New Features 🎉

- GCMC RDF: The `ase_utils` now has the `rdf_gcmc`, that can calculate the Radial Distribution Function (RDF) on a trajectory with different number of atoms.
- Added the `EwaldSum` calculator to calculate the electrostatic interactions
- Added the `CustomLennardJones` calculator to calculate the Lennard-Jones interaction energy

### Fixed 🐛

- Fix the bug where the `output_folder` parameter was not being properly passed to the `BaseSimulator` on `GCMC` class.
- Fix a major bug in rotating molecules that have part of their atoms in one unit cell and parts in another.

### Enhanced ✨

- Introduced the `max_overlap_tries` parameter in `GCMC` and `Widom` classes to try `n` attempts for placing an adsorbate without van der Waals overlap. If a valid position is not found within the specified number of tries, the insertion is rejected. This applies to insertion, rotation and translation moves.
- Only create the `Trajectory_rejected.traj` file if the `save_rejected` parameter is set to `True` when initializing the `BaseSimulator` class.
- Abstracted a few methods on the `Widom` class for better code reusability.
- Refactor the results json to improve the clarity
- Both `Widom` and `GCMC` classes now saves the random seed and enlapsed time on the `results.json` file.
- Now the use of Left-most Local Minima for pyMSER equilibration can defined on the GCMC class through the `LLM` parameter. By default it is `True`.
- Now the molecular rotation are based on a unit vector on a sphere, using the method proposed by George Marsaglia in The Annals of Mathematical Statistics, 1972, Vol. 43, No. 2, 645-646.

### Documentation 📖

- Fix a bug on the isotherm simulation example.

### Removed 🗑️

## v[0.4.3] - 2025-10-15

### New Features 🎉

- Added support for custom output folders in the `GCMC` and `Widom` classes. Users can now specify a custom folder for saving output files through the `output_folder` parameter. If not provided, a default folder named 'results_<T>_<P>' will be created based on the simulation temperature and pressure.

### Fixed 🐛

### Enhanced ✨

### Documentation 📖

### Removed 🗑️

## v[0.4.2] - 2025-10-14

### New Features 🎉

- Added the `GCMC.equilibrate` method to allow users to perform equilibration analysis separately from the main simulation run. This method uses the pyMSER library to analyze the uptake data and determine when the system has reached equilibrium.
- Added the `GCMC.save_results` method to save the results of the equilibration analysis to a JSON file. This includes key metrics such as average uptake, uncertainty, and enthalpy of adsorption.
- Allow for custom move weights in the GCMC simulation. Users can now specify different probabilities for insertion, deletion, translation, and rotation moves through the `move_weights` parameter when initializing the `GCMC` class. This provides greater flexibility in tailoring the simulation to specific systems or research needs.
- Improved the logging of Widom simulation results to include standard deviations for both the Henry coefficient and the enthalpy of adsorption. The standard deviations are calculated using 5-fold cross-validation, providing a more robust measure of uncertainty in the results.
- Added the `Widom.save_results` method to include standard deviations for the Henry coefficient and enthalpy of adsorption.
- Added the `save_rejected` parameter to both the `GCMC` and `Widom` classes. When set to `True`, this parameter enables the saving of rejected moves to a separate trajectory file, allowing users to analyze these configurations post-simulation.

### Fixed 🐛

- The %wt conversion factor was incorrectly calculated using a factor of 1e-3 instead of 1e-1. This has been corrected in the `GCMC` class and related calculations to ensure accurate representation of adsorption data in weight percent.
- The restart method in the `GCMC` class was not properly resetting the simulation state. This has been fixed to ensure that all relevant attributes are correctly reinitialized when restarting a simulation.

### Enhanced ✨

- Improved the handling of tags in the ASE Atoms objects for both the framework and adsorbate molecules. The framework atoms are now tagged with `0`, and adsorbate atoms are tagged with `1`. This tagging system helps differentiate between framework and adsorbate atoms during simulations and analyses. This is also a necessary step for future implementations of multiple adsorbates.
- Now the rotation movements are restricted to an arc of -15 to +15 degrees around an random axis instead of a full random rotation. This change aims to improve the acceptance rate of rotation moves in the GCMC simulation by making smaller, more controlled adjustments to the adsorbate positions. This angle can be adjusted by the user through the `max_rotation_angle` parameter when initializing the `GCMC` class.

### Documentation 📖

- Examples have been updated to save the results of the simulations using the new `save_results` method.

### Removed 🗑️

## v[0.4.1] - 2025-09-19

### New Features 🎉

- The final name of the package is now `FLAMES - Flexible Lattice Adsorption by Monte Carlo Engine Simulation`. The code repository, documentation site, and examples have been updated to reflect this change.

### Fixed 🐛

### Enhanced ✨

- Added a `max_deltaE` parameter to the `BaseSimulator`, `GCMC`, and `Widom` classes to prevent errors from the model during energy calculations. This parameter sets a maximum threshold for energy changes, ensuring numerical stability during simulations. The default value is set to 1.555 eV (150 kJ/mol).
- Change the `random_seed` parameter to pick a random integer between 1 and 1,000,000 if not provided by the user. This ensures the use of a valid seed for the random number generator, enhancing reproducibility in simulations.

### Documentation 📖

### Removed 🗑️

## v[0.4.0] - 2025-09-04

### New Features 🎉

- Added the Peng-Robinson equation of state (EOS) to the `mlp_adsorption.eos` module.
  - This allows for the calculation of fugacity coefficients and bulk phase density using the Peng-Robinson EOS.
- Add functions to calculate perpendicular lengths and unit cell repetitions in `utilities`.
- Add a new parameter `cutoff` on the `mlp.gcmc` class for controlling the supercell check based on a cutoff radius for the potential.
- Add the `get_density` function to the `mlp_adsorption.utilities` module.
- Add a `mlp_adsorption.utilities.make_cubic` function that can create a cubic (or close to) supercell from a given structure.
- Now all movements are based on a `numpy.random.Generator` to ensure reproducibility. A `random_seed` can be provided to the `GCMC` and `Widom` classes for this purpose.
- New module for checking the overlap between atoms (`mlp_adsorption.operations.check_overlap`)
- A new `BaseSimulator` method created to abstract the system state management and general simulation logic.
- Now all the output of the simulations are managed by a `mlp_adsorption.logger` instance. It introduces:
  - `BaseLogger`: A base class for logging simulation information, including restart and iteration details.
  - `GCMCLogger`: A logger specifically for GCMC simulations, extending `BaseLogger` with additional functionality.
  - `WidomLogger`: A logger specifically for Widom simulations, extending `BaseLogger` with additional functionality.
- Now the `Widom` has a `restart` method to allow restarting a Widom simulation from a saved state. Since Widom does not insert adsorbates on the structure, it simply reads the existing list of insertion energies and use it to update the simulation statistics.
- Now the code uses [pyMSER](https://github.com/IBM/pymser) to evaluate the equilibration of the simulation and calculate the average uptake and enthalpy of adsorption.

### Fixed 🐛

- Fix the framework density calculation on `GCMC` and `Widom` class.
- Fix an error on the check if the optimization converged.

### Enhanced ✨

- Move the random operations (insertion, rotation, translation) to a separate `mlp_adsorption.operations.py` module for better organization and reusability.

### Documentation 📖

- Add the critical parameters to the examples.

### Removed 🗑️

- Old `mlp_adsorption.operations.vdw_overlap` and `mlp_adsorption.operations.vdw_overlap2` functions.

## v[0.3.2] - 2025-08-17

### New Features 🎉

- New `restart` method in the `GCMC` class to allow restarting a GCMC simulation from a saved state.
  - It reads the state from an existing `Trajectory` object, enabling the continuation of simulations without losing progress.
  - It reads the total uptake, total energy, and adsorption energy `npy` files for seamless simulation restoration.
- Now the `vdw_factor` can be set when initializing the `GCMC` class, allowing for more flexible control over the Van der Waals radii scaling. By default, it is set to 0.6.
- Add `flush=True` to print statements for immediate output in optimization and simulation logs

### Fixed 🐛

- Fixed the handling of NaN values in the Van der Waals radii in both `gcmc.py` and `widom.py` files to ensure that any NaN value is replaced by 1.5, preventing potential issues during simulations.
- Fixed unit for Henry constant on header and footer of Widom output. Now it correctly reports the unit as [mol kg-1 Pa-1].
- Change the energy return value in the `try_insertion` method of the `Widom` class to 1000 eV when there is a Van der Waals overlap and limit the number of insertion attempts to 100. This change ensures that the method prevents an infinite loop in the simulation.

### Enhanced ✨

- Restart of a GCMC simulation:
  - Now the `load_state` method in the `GCMC` class can load the state from a `Trajectory` object, allowing for restarting simulations from saved states.

- Add new examples for running GCMC simulations with different configurations, including:

  - Basic
    - Widom Insertion
    - Rigid GCMC
    - Rigid GCMC Isotherm
  - Intermediate
    - Geometry Optimization + GCMC
    - Flexible GCMC: Molecular Dynamics + GCMC
  
### Documentation 📖

- New examples are documented in the README file, providing clear instructions on how to run GCMC simulations and Widom insertion tests.

### Removed 🗑️

- Removed commented-out code for saving the system state in xyz format in the `gcmc.py` file to clean up the codebase.

## v[0.3.1] - 2025-08-05

### New Features 🎉

### Fixed 🐛

- Fixed the calculation of Henry's law constant in the Widom insertion method to ensure it is correctly computed in units of [mol kg-1 Pa-1].
- Fixed the calculation of the heat of adsorption (Qst) in the Widom insertion method to ensure it is correctly computed in units of [kJ/mol].

### Enhanced ✨

### Documentation 📖

### Removed 🗑️

## v[0.3.0] - 2025-07-31

### New Features 🎉

- Now GCMC has three different methods for NPT Molecular Dynamics (MD):
  - Isotropic volume change and fixed shape with Berendsen thermostat/barostat (`mode=iso_shape`)
  - Anisotropic volume change and fixed shape with Berendsen thermostat/barostat (`mode=aniso_shape`)
  - Anisotropic volume change and flexible shape with Nose-Hoover/Parrinello-Rahman thermostat/barostat (`mode=aniso_flex`).
- Added a new method `nvt` to the GCMC class for running NVT MD simulations.

### Fixed 🐛

[#10](https://github.com/lipelopesoliveira/mlp_adsorption/issues/10) - The cell angles are not changing during the MD simulation.

### Enhanced ✨

- Now the NPT MD simulation log also prints the cell volume.
- Now it is possible to control the type of NPT MD simulation through the use of a keyword argument on the `npt` method of GCMC simulation.

### Documentation 📖

### Removed 🗑️

## v[0.2.0] - 2025-07-12

### New Features 🎉

- Support for saving the snapshots of the simulation on the same trajectory file in the `gcmc.py` and `widom.py` codes.
- New examples for using the GCMC and Widom insertion methods in the `examples` directory.

### Fixed 🐛

- Incorrect statement on the insertion method in the `gcmc.py` file.

### Enhanced ✨

- Updated the `gcmc.py` file to use the `Trajectory.write` method instead of `write_proteindatabank`.
- Updated the `ase_utils.py` file to accept a `trajectory` parameter in the `npt_md` function.
- Replaces PDB output with ASE Trajectory for snapshot saving in `widom.py`.
- Removed the unused import of `write_proteindatabank` in `widom.py`.

### Documentation 📖

- Added docstrings to all functions and classes in the `gcmc.py` and `widom.py` files.
- Updated the README file with examples of how to use the GCMC and Widom insertion methods.

### Removed 🗑️

- Unused imports and commented-out code in `widom.py`.

## v[0.1.0] - 2025-07-12

- Initial release of the MLP Adsorption code, including the GCMC and Widom methods with basic functionality on the simulation of adsorption processes.
