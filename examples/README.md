# Examples

The examples script in this directory demonstrate how to use the FLAMES package for various tasks, such as running GCMC simulations and performing Widom insertion tests, etc.

### Basic

[1. Widom Insertion](https://github.com/lipelopesoliveira/flames/tree/main/examples/Basic/1-Widom/run_widom.py)

The Widom insertion method is a powerful and computationally efficient technique in statistical mechanics used to calculate the excess chemical potential of a species at infinite dilution. The method operates by inserting a "ghost" or "test" particle at numerous random positions and orientations within a static configuration of a host system, such as a porous material. For each insertion, the interaction energy between the ghost particle and the host is calculated, but the particle is not actually added, so the host's configuration remains unchanged. By averaging the Boltzmann factor of these interaction energies over thousands or millions of trials, one can directly compute the Henry's constant, which is fundamentally related to the material's affinity for the adsorbate at low pressures. From this simulation the isosteric heat, or enthalpy of adsorption, at zero coverage can also be determined. Furthermore, the distribution of insertion energies reveals the potential energy landscape, identifying the coordinates of the most stable adsorption sites within the framework.

This makes the Widom method arguably the easiest and most direct way to test a new Machine Learning Potential (MLP) for adsorption applications. Unlike full Grand Canonical Monte Carlo simulations, it does not require lengthy equilibration or the simulation of multiple interacting guest molecules. The method yields two crucial, physically meaningful metrics: the most stable binding configurations and the enthalpy of adsorption. These two outputs can be directly compared with high-fidelity experimental data from microcalorimetry and diffraction techniques, or with results from expensive quantum mechanical calculations, providing a robust and computationally inexpensive first-pass validation of the MLP's accuracy in describing host-guest interactions.

[2. Rigid GCMC Simulation](https://github.com/lipelopesoliveira/flames/tree/main/examples/Basic/2-Rigid_GCMC/run_GCMC.py)

The Grand Canonical Monte Carlo (GCMC) method is a powerful computational technique used to simulate the adsorption of guest molecules in porous materials, such as metal-organic frameworks (MOFs), covalent organic frameworks (COFs), or zeolites. It operates under the grand canonical ensemble, allowing for the exchange of particles between the system and an ideal reservoir at a fixed temperature and chemical potential (that can be calculated from the pressure).

[3. Rigid GCMC Isotherm](https://github.com/lipelopesoliveira/flames/tree/main/examples/Basic/3-Rigid_GCMC_Isotherm/run_GCMC_Isotherm.py)

This example demonstrates how to run a GCMC simulation to generate adsorption isotherms for a given framework and adsorbate. The script allows you to specify a range of pressures and run the GCMC simulation for each pressure point, saving the results for further analysis. It starts each new pressure point from the last saved state, allowing for efficient continuation of simulations.

### Intermediate

[1. Geometry Optimization + GCMC](https://github.com/lipelopesoliveira/flames/tree/main/examples/Intermediate/1-Geometry_Optimization_GCMC/run_GCMC.py)

This example also demonstrate how to perform geometry optimization of the framework structure and adsorbate before running the GCMC simulation. The script uses the LBFGS optimizer from ASE to optimize the framework structure, but any other optimizer can be used as well. After the optimization, it runs the GCMC simulation using the optimized framework structure and the specified adsorbate.

[2. Molecular Dynamics (MD) with GCMC](https://github.com/lipelopesoliveira/flames/tree/main/examples/Intermediate/2-MD_GCMC/run_MD_GCMC.py)

This example demonstrates how to run a GCMC simulation with molecular dynamics (MD) steps. It allows you to perform GCMC simulations while also incorporating MD to explore the dynamic behavior of the system. The script initializes the GCMC simulation and runs it for a specified number of Monte Carlo steps. Them, it performs MD steps to simulate the motion of atoms in the framework and adsorbate at the specified temperature and pressure. After a few iterations of GCMC and MD, it runs a final GCMC simulation to ensure the system reaches equilibrium.
