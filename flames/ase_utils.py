import datetime
import os
import sys
from typing import TextIO

import ase
import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import Calculator
from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter
from ase.geometry import get_distances
from ase.io.trajectory import Trajectory
from ase.md import MDLogger
from ase.md.npt import NPT
from ase.md.nptberendsen import Inhomogeneous_NPTBerendsen, NPTBerendsen
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.optimize.optimize import Optimizer
from ase.spacegroup.symmetrize import check_symmetry
from tqdm import tqdm


def crystalOptimization(
    atoms_in: Atoms,
    calculator: Calculator,
    optimizer: Optimizer,
    out_file: TextIO,
    fmax: float = 0.005,
    opt_cell: bool = True,
    fix_symmetry: bool = False,
    hydrostatic_strain: bool = False,
    constant_volume: bool = False,
    scalar_pressure: float = 0.0,
    max_steps: int = 1000,
    trajectory: bool | str = "opt.traj",
    verbose: bool = True,
    symm_tol=1e-3,
) -> tuple[dict, Atoms]:
    """
    Optimize the cell and positions of the atoms with the given calculator.
    If fix_symmetry is True, the symmetry of the cell is fixed during the optimization.
    If hydrostatic_strain is True, the cell is allowed to change only isotropically.

    Helpful conversion table for fmax:

        - 0.05 eV/A^3 = 8 GPA = 8000 bar
        - 0.003 eV/A^3 = 0.48 GPa = 480 bar
        - 0.0006 eV/A^3 = 0.096 GPa = 96 bar
        - 0.0003 eV/A^3 = 0.048 GPa = 48 bar
        - 0.0001 eV/A^3 = 0.02 GPa = 20 bar


    Parameters
    ----------

    atoms_in: Atoms
        The atoms object to optimize.
    calculator: Calculator
        The calculator to use for the optimization.
    optimizer: Optimizer
        The optimizer to use for the optimization. Recommended: lBFGS.
    fmax: float
        The maximum force allowed during the optimization. Default: 0.005 eV/Ang.
    opt_cell: bool
        If True, the cell is optimized during the optimization.
    fix_symmetry: bool
        If True, the symmetry of the cell is fixed during the optimization.
    hydrostatic_strain: bool
        If True, the cell is allowed to change only isotropically.
    constant_volume: bool
        If True, the volume of the cell is fixed during the optimization.
    scalar_pressure: float
        The pressure to use during the optimization. Default is 0.0 GPa.
    max_steps: int
        The maximum number of steps to run the optimization for. Default is 1000.
    trajectory: str
        The name of the trajectory file to write the optimization steps to.
    verbose: bool
        If True, print the optimization steps to the console during the optimization.
    symm_tol: float
        The tolerance to use when checking the symmetry of the cell. Default is 1e-5.

    Returns
    -------

    resultsDict: dict
        A dictionary containing the results of the optimization, including:
    atoms: Aoms
        The optimized atoms object.
    """

    atoms = atoms_in.copy()

    atoms.calc = calculator

    if fix_symmetry:
        atoms.set_constraint([FixSymmetry(atoms)])

    if opt_cell:
        ecf = FrechetCellFilter(
            atoms,
            hydrostatic_strain=hydrostatic_strain,
            constant_volume=constant_volume,
            scalar_pressure=scalar_pressure,
        )

        opt = optimizer(ecf, logfile=None)  # type: ignore

    else:
        opt = optimizer(atoms, logfile=None)  # type: ignore

    opt_history = []

    start_time = datetime.datetime.now()

    def custom_ase_log():

        e_tot = atoms.get_potential_energy()
        if opt_cell:
            stress = atoms.get_stress(voigt=False)
            pressure = -1 / 3 * np.trace(stress)
        else:
            pressure = 0.0
            stress = np.zeros(6)

        forces = atoms.get_forces()
        max_force = np.linalg.norm(forces, axis=1).max()
        sum_force = np.sum(np.linalg.norm(forces, axis=1))
        rmsd_force = np.sqrt(np.mean(forces**2))

        opt_history.append(
            {
                "cellParameters": atoms.cell.cellpar().tolist(),
                "cellMatrix": np.array(atoms.cell).tolist(),
                "atomTypes": atoms.get_chemical_symbols(),
                "cartCoordinates": atoms.get_positions().tolist(),
                "energy": e_tot,
                "forces": forces.tolist(),
                "stress": stress.tolist(),
            }
        )

        line_txt = "{:5} {:>18.8f}    {:>15.8f}    {:>15.8f}     {:>15.8f}    {:>18.8f}   {:>18.8f} {:>12.2f}"

        print(
            line_txt.format(
                len(opt_history),
                e_tot,
                max_force,
                sum_force,
                rmsd_force,
                pressure * 1e5,
                atoms.get_volume() if opt_cell else 0.0,
                (datetime.datetime.now() - start_time).total_seconds() / 60,
            ),
            file=out_file,
            flush=True,
        )

    opt.attach(custom_ase_log, interval=1)

    traj = Trajectory(trajectory, "w", atoms)
    if trajectory:
        opt.attach(traj)

    headers = [
        "Step",
        "Energy (eV)",
        "Max Force (eV/A)",
        "Sum Force (eV/A)",
        "RMSD Force (eV/A)",
        "Pressure (bar)",
        "Volume (A3)",
        "Time (min)",
    ]
    print(
        "{:^5} {:^18}    {:^15}    {:^15}     {:^15}    {:^15}     {:^15}    {:^14}".format(
            *headers
        ),
        file=out_file,
        flush=True,
    )

    converged = opt.run(fmax=fmax, steps=max_steps)

    if trajectory:
        traj.close()

    print(
        "Optimization finished. Total time: {:.2f} minutes".format(
            (datetime.datetime.now() - start_time).total_seconds() / 60
        ),
        file=out_file,
        flush=True,
    )

    print(f"Optimization {'' if converged else 'did not '}converged.", file=out_file, flush=True)

    resultsDict = {
        "status": "Finished",
        "optConverged": "Yes" if converged else "No",
        "warningList": [],
        "executionTime": {
            "unit": "s",
            "value": (datetime.datetime.now() - start_time).total_seconds(),
        },
        "startTime": start_time.isoformat(),
        "endTime": datetime.datetime.now().isoformat(),
        "calculationResults": opt_history,
    }

    # Print final information about the symmetry of the cell
    if opt_cell:
        symm = check_symmetry(atoms, symprec=symm_tol, verbose=False)

        if symm is not None:
            print(
                """
    Symmetry information
    --------------------------------------------
    Space Group Number: {}
    Space Group Symbol: {}
    Lattice type: {}
    """.format(
                    symm.number, symm.international, atoms.cell.get_bravais_lattice().longname
                ),
                file=out_file,
                flush=True,
            )

        resultsDict["symmetryInformation"] = {
            "number": symm.number if symm else None,
            "international": symm.international if symm else None,
            "bravaisLattice": atoms.cell.get_bravais_lattice().longname,
        }

    return resultsDict, atoms


def nVT_Berendsen(
    atoms: ase.Atoms,
    model: Calculator,
    temperature: float,
    time_step: float = 0.5,
    num_md_steps: int = 1000000,
    output_interval: int = 100,
    movie_interval: int = 1,
    taut: float = 1.0,
    out_folder: str = ".",
    out_file: TextIO = sys.stdout,
    trajectory=None,
) -> ase.Atoms:
    """
    Run NVT molecular dynamics simulation using the Berendsen thermostat.

    The Berendsen thermostat is a deterministic type of velocity scaling method that
    adjusts the velocities of the atoms to maintain a target temperature of the entire system.
    It is not a true canonical ensemble method, but it is often used for its simplicity and speed.

    Parameters
    ----------
    atoms : ase.Atoms
        The atomic structure to simulate.
    temperature : float
        The target temperature in Kelvin.
    time_step : float, optional
        The time step for the simulation in femtoseconds (default is 0.5 fs).
    num_md_steps : int, optional
        The total number of MD steps to run (default is 1,000,000).
    output_interval : int, optional
        The interval for logging output (default is 100 steps).
    movie_interval : int, optional
        The interval for saving trajectory frames (default is 1 step).
    taut : float, optional
        The time constant for the Berendsen thermostat in femtoseconds (default is 1.0 fs).
    out_folder : str, optional
        The folder where the output files will be saved (default is the current directory).
    out_file : TextIO, optional
        The output file to write the simulation log to (default is sys.stdout).

    Returns
    -------
    ase.Atoms
        The final atomic structure after the MD simulation.
    """

    atoms.calc = model

    header = """
===========================================================================
    Starting NVT MD Simulation using Berendsen Thermostat

    Parameters:
        Temperature: {:.2f} K
        Time Step: {:.2f} fs
        Number of MD Steps: {}
        Output Interval: {} steps
        Movie Interval: {} steps
        Time Constant (taut): {:.2f} fs

===========================================================================
""".format(
        temperature, time_step, num_md_steps, output_interval, movie_interval, taut
    )

    print(header, file=out_file, flush=True)

    existing_md_traj = [
        i for i in os.listdir(out_folder) if i.startswith("NVT-Berendsen") and i.endswith(".traj")
    ]
    traj_filename = os.path.join(
        out_folder, f"NVT-Berendsen_{temperature:.2f}K_{len(existing_md_traj)}.traj"
    )
    log_filename = os.path.join(
        out_folder, f"NVT-Berendsen_{temperature:.2f}K_{len(existing_md_traj)}.log"
    )

    # Set the momenta corresponding to the given "temperature"
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, force_temp=True)
    # Set zero total momentum to avoid drifting
    Stationary(atoms)

    # run Berendsen MD
    dyn = NVTBerendsen(
        atoms=atoms,
        timestep=time_step * units.fs,
        temperature_K=temperature,
        taut=taut * units.fs,
        loginterval=output_interval,
        trajectory=trajectory if trajectory else traj_filename,
    )

    # Print statements
    def print_md_log() -> None:
        step = dyn.get_number_of_steps()
        etot = atoms.get_total_energy()
        temp_K = atoms.get_temperature()
        stress = atoms.get_stress(include_ideal_gas=True) / units.GPa
        stress_ave = (stress[0] + stress[1] + stress[2]) / 3.0
        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        print(
            "  {:>7}  | {:13.6f}  |  {:11.3f}  |  {:7.2f} | {:9.1f}".format(
                step, etot, temp_K, stress_ave, elapsed_time
            ),
            file=out_file,
            flush=True,
        )

    dyn.attach(print_md_log, interval=output_interval)
    dyn.attach(
        MDLogger(dyn, atoms, log_filename, header=True, stress=True, peratom=False, mode="a"),
        interval=movie_interval,
    )

    # Now run the dynamics
    start_time = datetime.datetime.now()
    print(
        "    Step   |  Total Energy  |  Temperature  |  Stress  | Elapsed Time ",
        file=out_file,
        flush=True,
    )
    print(
        "    [-]    |      [eV]      |      [K]      |   [GPa]  |     [s]      ",
        file=out_file,
        flush=True,
    )
    print(
        " --------- | -------------- | ------------- | -------- | -------------",
        file=out_file,
        flush=True,
    )

    dyn.run(num_md_steps)

    footer = """
======================================================================================
    NVT MD simulation completed at {}
    Log file saved to: {}
    Total simulation time: {:.2f} seconds
======================================================================================
    """.format(
        datetime.datetime.now(),
        log_filename,
        (datetime.datetime.now() - start_time).total_seconds(),
    )

    print(footer, file=out_file, flush=True)

    return atoms


def nPT_Berendsen(
    atoms: ase.Atoms,
    model: Calculator,
    temperature: float,
    pressure: float = 0.0,
    compressibility: float = 1e-3,
    time_step: float = 0.5,
    num_md_steps: int = 1000000,
    output_interval: int = 100,
    movie_interval: int = 10,
    taut: float = 10.0,
    taup: float = 500.0,
    isotropic: bool = True,
    out_folder: str = ".",
    out_file: TextIO = sys.stdout,
    trajectory=None,
) -> ase.Atoms:
    """
    Run NPT molecular dynamics simulation using the Berendsen thermostat and barostat.

    Warning: The Berendsen method does not change the shape of the simulation cell, i.e.,
    it does not change the cell angles. If you want to change the shape of the cell,
    use the NPT Nose-Hoover-Parrinello-Rahman method instead.

    Parameters
    ----------
    atoms : ase.Atoms
        The atomic structure to simulate.
    temperature : float
        The target temperature in Kelvin.
    pressure : float, optional
        The desired pressure, in bar (1 bar = 1e5 Pa).
    compressibility : float, optional
        The compressibility of the material, in bar-1.
    time_step : float, optional
        The time step for the simulation in femtoseconds (default is 0.5 fs).
    num_md_steps : int, optional
        The total number of MD steps to run (default is 1,000,000).
    output_interval : int, optional
        The interval for logging output (default is 100 steps).
    movie_interval : int, optional
        The interval for saving trajectory frames (default is 1 step).
    taut : float, optional
        The time constant for the Berendsen thermostat in femtoseconds (default is 10.0 fs).
    taup : float, optional
        The time constant for the Berendsen barostat in femtoseconds (default is 500.0 fs).
    isotropic : bool, optional
        If True, the barostat is isotropic, i.e., the unit cell changes equally in all directions
        Default is True.
    out_folder : str, optional
        The folder where the output files will be saved (default is the current directory).
    out_file : TextIO, optional
        The output file to write the simulation log to (default is sys.stdout).

    Returns
    -------
    ase.Atoms
        The final atomic structure after the MD simulation.
    """

    atoms.calc = model
    header = """
======================================================================================
    Starting NPT MD Simulation using Berendsen Thermostat/Barostat

    Parameters:
        Temperature: {:.2f} K
        Pressure: {:.2f} Pa
        Isotropic: {}
        Compressibility: {:.2f} bar-1
        Time Constant (taut): {:.2f} fs
        Time Step: {:.2f} fs
        Number of MD Steps: {}
        Output Interval: {} steps
        Movie Interval: {} steps

======================================================================================
    Step   |  Total Energy  |  Temperature  |  Stress  |   Volume    | Elapsed Time
    [-]    |      [eV]      |      [K]      |   [GPa]  |    [A^3]    |      [s]
 --------- | -------------- | ------------- | -------- | ----------- | -------------
""".format(
        temperature,
        pressure,
        isotropic,
        compressibility,
        taut,
        time_step,
        num_md_steps,
        output_interval,
        movie_interval,
    )

    print(header, file=out_file, flush=True)

    existing_md_traj = [
        i for i in os.listdir(out_folder) if i.startswith("NPT-Berendsen") and i.endswith(".traj")
    ]
    traj_filename = os.path.join(
        out_folder, f"NPT-Berendsen_{temperature:.2f}K_{len(existing_md_traj)}.traj"
    )
    log_filename = os.path.join(
        out_folder, f"NPT-Berendsen_{temperature:.2f}K_{len(existing_md_traj)}.log"
    )

    # Set the momenta corresponding to the given "temperature"
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, force_temp=True)

    # Set zero total momentum to avoid drifting
    Stationary(atoms)

    # Common parameters for both dynamics types
    dyn_params = {
        "atoms": atoms,
        "timestep": time_step * units.fs,
        "temperature_K": temperature,
        "pressure_au": pressure * units.bar,
        "compressibility_au": compressibility / units.bar,
        "taut": taut * units.fs,
        "taup": taup * units.fs,
        "loginterval": movie_interval,
        "trajectory": trajectory if trajectory else traj_filename,
    }

    # Select the appropriate dynamics class based on isotropic flag
    dyn_class = NPTBerendsen if isotropic else Inhomogeneous_NPTBerendsen
    dyn = dyn_class(**dyn_params)

    # Print statements
    def print_md_log():
        step = dyn.get_number_of_steps()
        etot = atoms.get_total_energy()
        temp_K = atoms.get_temperature()
        stress = atoms.get_stress(include_ideal_gas=True) / units.GPa
        stress_ave = (stress[0] + stress[1] + stress[2]) / 3.0
        volume = atoms.get_volume()
        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        print(
            "  {:>7}  | {:13.6f}  |  {:11.3f}  |  {:7.2f} | {:11.2f} | {:9.1f}".format(
                step, etot, temp_K, stress_ave, volume, elapsed_time
            ),
            file=out_file,
            flush=True,
        )

    dyn.attach(print_md_log, interval=output_interval)
    dyn.attach(
        MDLogger(dyn, atoms, log_filename, header=True, stress=True, peratom=False, mode="a"),
        interval=movie_interval,
    )

    # Now run the dynamics
    start_time = datetime.datetime.now()

    dyn.run(num_md_steps)

    footer = """
======================================================================================
    NPT MD simulation completed at {}
    Log file saved to: {}
    Total simulation time: {:.2f} seconds
======================================================================================
    """.format(
        datetime.datetime.now(),
        log_filename,
        (datetime.datetime.now() - start_time).total_seconds(),
    )

    print(footer, file=out_file, flush=True)

    return atoms


def nPT_NoseHoover(
    atoms: ase.Atoms,
    model: Calculator,
    temperature: float,
    pressure: float = 0.0,
    time_step: float = 0.5,
    num_md_steps: int = 1000000,
    output_interval: int = 100,
    movie_interval: int = 10,
    ttime: float = 25.0,
    ptime: float = 75.0,
    B_guess: float = 30.0,
    out_folder: str = ".",
    out_file: TextIO = sys.stdout,
    trajectory=None,
) -> ase.Atoms:
    """
    Constant pressure/stress and temperature dynamics.

    Combined Nose-Hoover and Parrinello-Rahman dynamics, creating an NPT (or N,stress,T) ensemble.

    Warning: The Nose-Hoover-Parrinello-Rahman method changes the shape of the simulation cell, i.e.,
    it changes the cell angles. If you do not want to change the shape of the cell, use the NPT-Barendsen instead.

    Parameters
    ----------
    atoms : ase.Atoms
        The atomic structure to simulate.
    temperature : float
        The target temperature in Kelvin.
    pressure : float, optional
        The desired pressure, in bar (1 bar = 1e5 Pa).
    time_step : float, optional
        The time step for the simulation in femtoseconds (default is 0.5 fs).
    num_md_steps : int, optional
        The total number of MD steps to run (default is 1,000,000).
    output_interval : int, optional
        The interval for logging output (default is 100 steps).
    movie_interval : int, optional
        The interval for saving trajectory frames (default is 1 step).
    ttime : float, optional
        Characteristic timescale of the thermostat, in femtoseconds. Set to None to disable the thermostat.
        Default is 25.0 fs.
    ptime : float, optional
        A constant in the barostat differential equation in femtoseconds. Set to None to disable the barostat.
        Default is 500.0 fs.
    out_folder : str, optional
        The folder where the output files will be saved (default is the current directory).
    out_file : TextIO, optional
        The output file to write the simulation log to (default is sys.stdout).

    Returns
    -------
    ase.Atoms
        The final atomic structure after the MD simulation.
    """

    # Calculate the pressure factor based on the time constant and bulk modulus
    pfactor = (ptime * units.fs) ** 2 * B_guess * units.GPa

    atoms.calc = model
    header = """
======================================================================================
    Starting NPT MD Simulation using Nose-Hoover/Parrinello-Rahman Thermostat/Barostat

    Parameters:
        Temperature: {:.2f} K
        Pressure: {:.2f} Pa
        Time Constant (ttime): {:.2f} fs
        Pressure Factor (pfactor): {:.2f} fs
        Guess of Bulk Modulus (B_guess): {:.2f} GPa
        Time Step: {:.2f} fs
        Number of MD Steps: {}
        Output Interval: {} steps
        Movie Interval: {} steps

======================================================================================
    Step   |  Total Energy  |  Temperature  |  Stress  |   Volume    | Elapsed Time
    [-]    |      [eV]      |      [K]      |   [GPa]  |    [A^3]    |      [s]
 --------- | -------------- | ------------- | -------- | ----------- | -------------
""".format(
        temperature,
        pressure,
        ttime,
        pfactor,
        B_guess,
        time_step,
        num_md_steps,
        output_interval,
        movie_interval,
    )

    print(header, file=out_file, flush=True)

    existing_md_traj = [
        i
        for i in os.listdir(out_folder)
        if i.startswith("NPT-Nose-Hoover-Parrinello-Rahman") and i.endswith(".traj")
    ]
    traj_filename = os.path.join(
        out_folder,
        f"NPT-Nose-Hoover-Parrinello-Rahman_{temperature:.2f}K_{len(existing_md_traj)}.traj",
    )
    log_filename = os.path.join(
        out_folder,
        f"NPT-Nose-Hoover-Parrinello-Rahman_{temperature:.2f}K_{len(existing_md_traj)}.log",
    )

    # Set the momenta corresponding to the given "temperature"
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, force_temp=True)

    # Set zero total momentum to avoid drifting
    Stationary(atoms)

    dyn = NPT(
        atoms=atoms,
        timestep=time_step * units.fs,
        temperature_K=temperature,
        ttime=ttime * units.fs,
        pfactor=pfactor,
        externalstress=pressure * units.bar,
        trajectory=trajectory if trajectory else traj_filename,
        logfile=log_filename,
        loginterval=output_interval,
        append_trajectory=True,
    )

    # Print statements
    def print_md_log():
        step = dyn.get_number_of_steps()
        etot = atoms.get_total_energy()
        temp_K = atoms.get_temperature()
        stress = atoms.get_stress(include_ideal_gas=True) / units.GPa
        stress_ave = (stress[0] + stress[1] + stress[2]) / 3.0
        volume = atoms.get_volume()
        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        print(
            "  {:>7}  | {:13.6f}  |  {:11.3f}  |  {:7.2f} | {:11.2f} | {:9.1f}".format(
                step, etot, temp_K, stress_ave, volume, elapsed_time
            ),
            file=out_file,
            flush=True,
        )

    dyn.attach(print_md_log, interval=output_interval)
    dyn.attach(
        MDLogger(dyn, atoms, log_filename, header=True, stress=True, peratom=False, mode="a"),
        interval=movie_interval,
    )

    # Now run the dynamics
    start_time = datetime.datetime.now()

    dyn.run(num_md_steps)

    footer = """
======================================================================================
    NPT MD simulation completed at {}
    Log file saved to: {}
    Total simulation time: {:.2f} seconds
======================================================================================
    """.format(
        datetime.datetime.now(),
        log_filename,
        (datetime.datetime.now() - start_time).total_seconds(),
    )

    print(footer, file=out_file, flush=True)

    return atoms


def pbc2pbc(pbc):
    """Helper function for dealing with pbc."""
    if pbc is None:
        pbc = False
    if not hasattr(pbc, "__len__"):
        pbc = (pbc,) * 3
    return np.asarray(pbc)


def complete_cell(cell):
    """Return 3x3 cell array from cell object."""
    if cell is None:
        cell = np.ones((3, 3))
    cell = np.asarray(cell)
    if cell.shape == (3,):
        cell = np.diag(cell)
    return cell


def unwrap_positions(positions, cell, pbc=True, ref_atom=0):
    """Unwrap positions relative to a reference atom.

    This function translates atoms by integer multiples of the
    lattice vectors so that they form a connected set,
    minimizing the distance to a reference atom. This is the
    reverse of wrap_positions.

    Parameters:

    positions: float ndarray of shape (n, 3)
        Positions of the atoms.
    cell: float ndarray of shape (3, 3)
        Unit cell vectors.
    pbc: one or 3 bool
        For each axis in the unit cell, decides whether
        unwrapping is applied.
    ref_atom: int
        The index of the atom to use as the reference point (default 0).
        All other atoms will be unwrapped to be as close as
        possible to this atom.
    """

    # Ensure pbc is a (3,) boolean array
    pbc = pbc2pbc(pbc)

    # Ensure cell is a (3, 3) array
    cell = complete_cell(cell)

    # Convert positions to fractional coordinates
    # We solve cell.T * f.T = p.T  =>  f = (solve(cell.T, p.T)).T
    fractional_positions = np.linalg.solve(cell.T, np.asarray(positions).T).T

    # Get the reference atom's fractional position
    ref_f_pos = fractional_positions[ref_atom]

    # Calculate fractional differences relative to the reference atom
    # deltas.shape = (n, 3)
    deltas = fractional_positions - ref_f_pos

    # Apply unwrapping logic
    # For periodic directions, find the closest image by
    # subtracting the nearest integer.
    # np.rint(x) rounds x to the nearest integer.
    for i in range(3):
        if pbc[i]:
            deltas[:, i] -= np.rint(deltas[:, i])

    # The new unwrapped fractional positions are the reference
    # position plus the "closest image" deltas
    unwrapped_fractional = ref_f_pos + deltas

    # Convert back to Cartesian coordinates
    return np.dot(unwrapped_fractional, cell)


def rdf_gcmc(
    images: list[ase.Atoms],
    atom_1: str,
    atom_2: str,
    rmax: float = 10.0,
    binwidth: float = 0.1,
    exclude_idx: list = [],
    surface: bool = False,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Calculate radial distribution function for GCMC trajectories.

    This version is modified to handle trajectories where the
    number of atoms changes per frame (e.g., GCMC).

    Parameters
    ----------
    images : list of Atoms or Trajectory
        ASE Atoms object(s).

    atom_1, atom_2 : str
        Atoms to use for the RDF calculation. MUST be element symbols
        (e.g., 'O') for GCMC. Integer indices are NOT supported as
        they are not persistent in GCMC.

    rmax : float
        Maximum radius (in Angstrom) for the RDF calculation.
        This is required for consistent binning.

    binwidth : float
        The distance increments (in Angstrom).

    exclude_idx : list of int
        Atomic indices to be ignored. Note: This assumes these
        indices are 'static' (e.g., a fixed slab) and exist
        in all frames. Use with caution in GCMC.

    surface : bool
        If True, returns the g(z) instead of the g(r).
        Normalization is 1D (per area) instead of 3D (per volume).

    show_progress: bool
        Show progress bar using tqdm library

    Returns
    -------
    g_r: ndarray
        2D (r, g_r) numpy array.

    Raises
    ------
    ValueError
    """

    # --- 1. Setup ---
    if isinstance(images, Atoms):
        images = [images]

    nimages = len(images)
    if nimages == 0:
        raise ValueError("No images provided.")

    # Ensure exclude_idx is a list of int
    if not all([isinstance(ei, int) for ei in exclude_idx]):
        raise ValueError("Parameter exclude_idx must be a list of int.")

    # Check for integer indices in selectors (not allowed for GCMC)
    for idx_spec in [atom_1, atom_2]:
        idx_list = idx_spec if isinstance(idx_spec, list) else [idx_spec]
        if any(isinstance(id_val, int) for id_val in idx_list):
            raise ValueError(
                "idx1/idx2 cannot contain integers for GCMC trajectories. "
                "Please use element symbols (e.g., 'O')."
            )

    # Setup bins based on rmax and binwidth
    dr = binwidth
    nbins = int(rmax / dr)
    if nbins == 0:
        ValueError(f"Error: rmax ({rmax}) or binwidth ({binwidth}) results in 0 bins.")

    bin_edges = np.linspace(0, rmax, nbins + 1)
    r_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bins = np.zeros(nbins, dtype=float)

    # Accumulators for average normalization
    total_volume = 0.0
    total_pairs = 0.0
    total_area = 0.0  # For g(z)

    # Check if this is a self-comparison (e.g., 'O' vs 'O')
    is_self_comparison = atom_1 == atom_2

    # --- 2. Loop over all images ---
    for atoms in tqdm(images, disable=not show_progress):
        if len(atoms) == 0:
            continue  # Skip empty frames

        # --- 2a. Get frame-specific properties ---
        vol = atoms.get_volume()
        cell = atoms.cell.array
        pos = atoms.get_positions()
        symbols = atoms.get_chemical_symbols()
        allElements = set(symbols)

        total_volume += vol
        if surface:
            # Area is volume projected onto XY plane (assuming Z is the surface normal)
            area = np.linalg.norm(np.cross(cell[0], cell[1]))
            total_area += area

        # --- 2b. Get atom indices for THIS frame ---
        current_idx: list = [[], []]
        for i, idx_spec in enumerate([atom_1, atom_2]):
            # Ensure idx_spec is a list (e.g., 'O' -> ['O'])
            idx_list = idx_spec if isinstance(idx_spec, list) else [idx_spec]

            frame_indices = []
            for element_symbol in idx_list:
                if isinstance(element_symbol, str):
                    if element_symbol in allElements:
                        # Find all atoms with this symbol
                        indices = [atom.index for atom in atoms if atom.symbol == element_symbol]
                        frame_indices.extend(indices)
                # We already checked for ints, so no need for else

            # Apply exclusions and remove duplicates from this frame's list
            frame_indices = [idx for idx in frame_indices if idx not in exclude_idx]
            current_idx[i] = list(set(frame_indices))

        current_idx1_list, current_idx2_list = current_idx

        # Skip frame if one of the lists is empty
        if not current_idx1_list or not current_idx2_list:
            continue

        # --- 2c. Get distances ---
        pos1 = pos[current_idx1_list]
        pos2 = pos[current_idx2_list]

        if surface:
            # Project positions to Z-axis for g(z)
            pos1 = pos1 * [0, 0, 1]
            pos2 = pos2 * [0, 0, 1]

        _, dist = get_distances(pos1, pos2, cell=cell, pbc=atoms.pbc)

        # --- 2d. Apply mask for self-interaction ---
        dist_masked = None
        if is_self_comparison:
            # Check if the generated index lists are identical
            if current_idx1_list == current_idx2_list:
                if dist.shape[0] == dist.shape[1]:
                    # Remove diagonal (self-pairs)
                    mask = ~np.eye(dist.shape[0], dtype=bool)
                    dist_masked = dist[mask]
                else:
                    dist_masked = dist.flatten()  # Should not happen, but safe
            else:
                # e.g., idx1=['O'], idx2=['O','H'] -> not a true self-comparison
                dist_masked = dist.flatten()
        else:
            # Different atom types (e.g., 'O' vs 'H'), no self-pairs
            dist_masked = dist.flatten()

        # --- 2e. Bin histogram ---
        hist, _ = np.histogram(dist_masked, bins=bin_edges)
        bins += hist
        total_pairs += len(dist_masked)

    # --- 3. Normalization ---
    if nimages == 0 or total_pairs == 0:
        print("Warning: No pairs found or no images processed. Returning g(r) = 0.")
        return np.array(list(zip(r_centers, np.zeros(nbins)))).T

    avg_vol = total_volume / nimages
    avg_pairs_per_frame = total_pairs / nimages

    # Calculate average pair density
    if avg_vol == 0:
        print("Warning: Average volume is zero. Cannot normalize.")
        return np.array(list(zip(r_centers, np.zeros(nbins)))).T
    pair_density = avg_pairs_per_frame / avg_vol

    if pair_density == 0:
        print("Warning: Zero pair density. Returning g(r) = 0.")
        return np.array(list(zip(r_centers, np.zeros(nbins)))).T

    # Get average counts per bin
    avg_counts_per_bin = bins / nimages

    if surface:
        # g(z) normalization
        avg_area = total_area / nimages
        if avg_area == 0:
            print("Warning: Average cell area is zero. Cannot calculate g(z).")
            return np.array(list(zip(r_centers, np.zeros(nbins)))).T

        # Ideal count in a 1D slab of thickness 'dr'
        # N_ideal = (Avg Pairs / Avg Vol) * Avg Area * dr
        ideal_counts = (avg_pairs_per_frame / avg_vol) * avg_area * dr
        if ideal_counts == 0:  # Handle constant
            print("Warning: Ideal count for g(z) is zero. Cannot normalize.")
            return np.array(list(zip(r_centers, np.zeros(nbins)))).T
        g = avg_counts_per_bin / ideal_counts  # Normalize all bins by the same 1D density

    else:
        # g(r) normalization: Volume of spherical shells: 4 * pi * r^2 * dr
        vol_shells = 4.0 * np.pi * r_centers**2 * dr

        # Ideal number of pairs in each shell
        ideal_counts = pair_density * vol_shells

        g = np.zeros_like(avg_counts_per_bin)

        # Avoid division by zero at r=0
        non_zero = ideal_counts > 1e-9
        g[non_zero] = avg_counts_per_bin[non_zero] / ideal_counts[non_zero]

    g_r = np.array(list(zip(r_centers, g))).T

    return g_r
