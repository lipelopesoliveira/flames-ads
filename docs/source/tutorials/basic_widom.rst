===============
Widom insertion
===============


Breafly introduction about Widom insertion method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Widom insertion method is a powerful and computationally efficient technique in statistical mechanics used to calculate the excess chemical potential of a species at infinite dilution. The method operates by inserting a "ghost" or "test" particle at random positions and orientations within a static configuration of a host system, such as a porous material. For each insertion, the interaction energy between the ghost particle and the host is calculated, but the particle is not actually added, so the host's configuration remains unchanged. By averaging the Boltzmann factor of these interaction energies over thousands or millions of trials, one can directly compute the **Henry's constant**, which is fundamentally related to the material's affinity for the adsorbate at low pressures, and the **Isosteric Heat of Adsorption at zero coverage (Qst)**. Furthermore, the distribution of insertion energies reveals the potential energy landscape, identifying the coordinates of the most stable adsorption sites within the framework.

This makes the Widom method arguably the easiest and most direct way to test a new Machine Learning Potential (MLP) for adsorption applications. Unlike full Grand Canonical Monte Carlo simulations, it does not require lengthy equilibration or the simulation of multiple interacting guest molecules. The method yields two crucial, physically meaningful metrics: the most stable binding configurations and the enthalpy of adsorption. These two outputs can be directly compared with high-fidelity experimental data from microcalorimetry and diffraction techniques, or with results from expensive quantum mechanical calculations, providing a robust and computationally inexpensive first-pass validation of the MLP's accuracy in describing host-guest interactions.

----

Setting the stage
~~~~~~~~~~~~~~~~~

In this example we will evaluate the Mg-MOF-74, a material well-known for having adsorption of CO :sub:`2` on Open-Metal sites, which is very hard to model with classical force fields and usually require electronic structure methods, such DFT, to describe the interaction properly. 

On the ``flames/examples/Basic/1-Widom`` folder you will find the ``cif`` file for the Mg-MOF-74 [1]_. The simulation will be executed with the machine learning potential MACE [2]_, using the ``medium-0b2`` pre-trained foundation model with ``D3(0)`` dispersion correction. 

.. warning:: This model is used here just as an example, you should look carefully into the mace documentation to undestand how the potential work and which one is the best for your specific case.

The script below will run the simulation at 298 K (25°C). 

.. code:: python

    import os

    # Hide UserWarning and RuntimeWarning messages
    import warnings

    import ase
    import torch
    from ase.data import vdw_radii
    from ase.io import read
    from mace.calculators import mace_mp

    from flames.widom import Widom

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = mace_mp(
        model="medium-0b2",
        dispersion=True,
        damping="zero",  # choices: ["zero", "bj", "zerom", "bjm"]
        dispersion_xc="pbe",
        default_dtype="float32",
        device=device,
    )

    FrameworkPath = "mg-mof-74.cif"
    AdsorbatePath = "co2.xyz"

    # Load the framework structure
    framework: ase.Atoms = read(FrameworkPath)  # type: ignore

    # Load the adsorbate structure
    adsorbate: ase.Atoms = read(AdsorbatePath)  # type: ignore

    Temperature = 298.0

    NSteps = 3000

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
        cutoff_radius=6.0,
        automatic_supercell=True,
    )

    widom.logger.print_header()

    widom.run(NSteps)
    widom.logger.print_summary()
    widom.save_results()

----

Breaking down the input script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Importing libraries
-------------------


.. code:: python

    import os

    # Hide UserWarning and RuntimeWarning messages
    import warnings

    import ase
    import torch
    from ase.data import vdw_radii
    from ase.io import read
    from mace.calculators import mace_mp

    from flames.widom import Widom

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    
This first part only import the libraries and make the code to ignore a few non-important warnings from mace.
The ``os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"`` command is ony important on Windows machines. If you are running on Linux, this command will just be ignored.

Here the ``vdw_radii`` from ``ase.data`` is imported to calculate the smaller distance allowed between two atoms.

----

Defining the calculator
-----------------------

.. code:: python

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = mace_mp(
        model="medium-0b2",
        dispersion=True,
        damping="zero",  # choices: ["zero", "bj", "zerom", "bjm"]
        dispersion_xc="pbe",
        default_dtype="float32",
        device=device,
    )


This is an extremelly important part, as the ``calculator`` will be used to calculate the energy of interaction between the adsorbate (CO2) and the adsorbent (Mg-MOF-74). 
If you are running on GPU (which is extremelly recommended) you need to define the device correctly, so the calculator use the GPU. Currently, basically all MLPs only use a single GPU.

Here the pre-trained model ``medium-0b2`` is used, but currently there are several other variations available. For a complete list, check the mace-foundation repository: https://github.com/ACEsuit/mace-foundations.

For a list of available models, check https://github.com/ACEsuit/mace/blob/100a29149d90a5945eddec1f0940bd88a6e3b363/mace/calculators/foundations_models.py#L19. You can also run a local model, just passing the path of the model instead of the name to the variable ``model``.

Reading the structure
---------------------


.. code:: python

    FrameworkPath = "mg-mof-74.cif"
    AdsorbatePath = "co2.xyz"

    # Load the framework structure
    framework: ase.Atoms = read(FrameworkPath)  # type: ignore

    # Load the adsorbate structure
    adsorbate: ase.Atoms = read(AdsorbatePath)  # type: ignore

This part of the code will read the framework and the adsorbate files. It uses the ``read`` function from ``ase``, which is very powerful and can read most file formats. 
Here, just make sure that the framework file is in a file format that is compatible with periodic boundary conditions and has a unt cell correctly defined.

In case you want to make sure that your framework was read properly, you can create a separate python script and visualize the structure with the ``ase`` visualizator:

.. code:: python

    from ase.io import read
    from ase.visualize import view

    FrameworkPath = "mg-mof-74.cif"

    # Load the framework structure
    framework: ase.Atoms = read(FrameworkPath)  # type: ignore

    # Visualize the framework structure
    view(framework)


Running the simulation
----------------------

.. code:: python

    Temperature = 298.0
    NSteps = 3000

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
        cutoff_radius=6.0,
        automatic_supercell=True,
    )

    widom.logger.print_header()

    widom.run(NSteps)
    widom.logger.print_summary()
    widom.save_results()


Finally, this block defines the ``widom`` simulator and runs the code. You can define the ``Temperature`` and the number of Monte Carlo steps ``NSteps``.

Here we are using 3.000 for example, but it is recommended that you use way more steps for a serious simulations, and check if in fact your results are equilibrated. Usually 100.000 is a good starting point, but it strongly depends on the system.

You should change the ``cutoff_radius`` according to your system and potential used, usually 6.0 Å is a good starting point but check carefully the cutoff of your potential.
The ``automatic_supercell`` option will create a supercell big enough to fit the cutoff radius, so you don't need to worry about that.


Analyzing the output
~~~~~~~~~~~~~~~~~~~~

The header
----------

.. code:: none
    
    ===========================================================================
             _______  __          ___      .___  ___.  _______     _______.
            |   ____||  |        /   \     |   \/   | |   ____|   /       |
            |  |__   |  |       /  ^  \    |  \  /  | |  |__     |   (----`
            |   __|  |  |      /  /_\  \   |  |\/|  | |   __|     \   \    
            |  |     |  `----./  _____  \  |  |  |  | |  |____.----)   |   
            |__|     |_______/__/     \__\ |__|  |__| |_______|_______/    
                                                                
            Flexible Lattice Adsorption by Monte Carlo Engine Simulation
                            powered by Python + ASE
                        Author: Felipe Lopes de Oliveira
    ===========================================================================

    Code version: 0.4.4
    Simulation started at 2025-11-17 23:06:46
    Hostname: felipe-Dell-G16-7630
    OS type: Linux
    OS release: 6.7.10-060710-generic
    OS version: #202403151538 SMP PREEMPT_DYNAMIC Fri Mar 15 20:14:18 UTC 2024

    Python version: 3.12.11
    Numpy version: 1.26.4
    ASE version: 3.25.0

    Current directory: /home/felipe/PRs/flames/examples/Basic/1-Widom
    Random Seed: 42

    Model: sumcalculator
    Running on device: cuda

The header will print some information about the code version, the host machine, the python and ASE versions, the current directory, the random seed used, the model and the device used. It is always important to check this information to make sure everything is correct.

The simulation summary
----------------------

.. code:: none

    ===========================================================================

    Constants used:
    Boltzmann constant:     8.617330337217213e-05 eV/K
    Beta (1/kT):            38.941 eV^-1
    Fugacity coefficient:   0.000000000 (dimensionless)

    ===========================================================================

    Simulation Parameters:
    Temperature: 298.0 K
    Pressure: 0.00000 bar
    Fugacity: 0.000 Pa
    Fugacity: 0.00000e+00 eV/m^3
    (1/kB.T) * V * f = 0.0 [-]

    ===========================================================================

This block presents the constants used in the simulation, mostly for reference and debugging purposes. It also summarizes the simulation parameters, including temperature, pressure, and fugacity values. For Widom simulations, there is no pressure or fugacity involved, so these values are zero.


The system information
----------------------

.. code:: none

    System Information:
    Framework: C144H36Mg36O108
    Framework: 324 atoms,
    Framework mass: 4368.743999999999 g/mol, 7.254469967765757e-24 kg
    Framework energy: -2365.8930653513903 eV
    Framework volume: 8.24197326482685e-27 m^3
    Framework density: 880.186059171615 kg/m^3, 0.8801860591716151 g/cm^3
    Framework cell:
        26.1701000    0.0000000    0.0000000
        13.0850500   22.6639714    0.0000000
        0.0000000    0.0000000   13.8960000

    Perpendicular cell:
        22.6639714    0.0000000    0.0000000
        0.0000000   22.6639714    0.0000000
        0.0000000    0.0000000   13.8960000

    Atomic positions:
    Mg   16.5873945   14.4378564    3.7075223
    O     4.8947247    5.7482631    2.3059717
    C     5.6694905    4.8079349    1.7135852
    ...

     H    25.8864161   22.0824139    8.0335555

    ===========================================================================
    Adsorbate: CO2
    Adsorbate: 3 atoms, 7.307866261136e-26 kg
    Adsorbate energy: -22.788235797845456 eV

    Atomic positions:
    C     0.0000000    0.0000000    0.0000000
    O     1.1606430    0.0000000    0.0000000
    O    -1.1606430    0.0000000    0.0000000

    ===========================================================================
    Shortest distances:
    O  - H : 1.632 Å
    O  - Mg: 1.950 Å
    O  - C : 1.932 Å
    H  - Mg: 1.758 Å
    H  - C : 1.740 Å
    Mg - C : 2.058 Å

    ===========================================================================
    Conversion factors:
        Conversion factor molecules/unit cell -> mol/kg:         0.228898741
        Conversion factor molecules/unit cell -> mg/g:           10.073604679
        Conversion factor molecules/unit cell -> cm^3 STP/gr:    5.130527703
        Conversion factor molecules/unit cell -> cm^3 STP/cm^3:  4.515818960
        Conversion factor molecules/unit cell -> %wt:            1.007360468

    Partial pressure:
                    0.000000000000000 Pascal
                    0.000000000000000 bar
                    0.000000000000000 atm
                    0.000000000000000 Torr
    ===========================================================================

This section provides detailed information about the framework and adsorbate, including their compositions, masses, energies, volumes, densities, and atomic positions.

It also lists the shortest distances between different atom types in the system, which can be useful for understanding potential interactions. Always check these distances to make sure there are no unphysical overlaps or if it does not prevent desired interactions.

Additionally, conversion factors are provided to facilitate the interpretation of simulation results in various units. For Widom simulations these factores are not used, but they are printed for reference.

The simulation
--------------

.. code:: none

    +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    Starting Widom simulation
    +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    Iteration  |  dE (eV)  |  dE (kJ/mol)  | kH [mol kg-1 Pa-1]  |  dH (kJ/mol) | Time (s)
    ---------------------------------------------------------------------------------------
        1      | -0.049759 |         -4.80 |           1.821e-06 |        -6.67 |     0.52
        2      | -0.118381 |        -11.42 |           1.657e-05 |       -13.37 |     0.61
        3      | -0.156906 |        -15.14 |           6.406e-05 |       -16.79 |     0.15
        4      | -0.031338 |         -3.02 |           5.156e-05 |       -16.72 |     0.15
        5      | -0.034721 |         -3.35 |           4.326e-05 |       -16.65 |     0.30
        6      | -0.117306 |        -11.32 |           4.339e-05 |       -16.24 |     0.15
        7      | -0.077221 |         -7.45 |           3.913e-05 |       -16.05 |     0.15
        8      | -0.076447 |         -7.38 |           3.578e-05 |       -15.88 |     0.30
        9      | -0.077040 |         -7.43 |           3.312e-05 |       -15.71 |     0.15
        10     | -0.142258 |        -13.73 |           4.073e-05 |       -15.84 |     0.15
        11     | -0.184081 |        -17.76 |           8.692e-05 |       -18.35 |     0.17
        12     | -0.074968 |         -7.23 |           8.089e-05 |       -18.28 |     0.15
        13     | -0.233352 |        -22.52 |           3.646e-04 |       -23.61 |     0.15
        14     | -0.084846 |         -8.19 |           3.412e-04 |       -23.58 |     0.16
        15     | -0.068756 |         -6.63 |           3.203e-04 |       -23.56 |     0.31
        16     | -0.313437 |        -30.24 |           5.694e-03 |       -32.23 |     0.15
    ...


Here the main simulation output is presented. For each Widom insertion step, the change in energy (dE) upon inserting the ghost particle is reported in both eV and kJ/mol.
The Henry's constant (kH) and isosteric heat of adsorption at zero coverage (dH) are calculated based on the average Boltzmann factor of the insertion energies up to that point. The time taken for each insertion step is logged, allowing users to monitor performance.


The results
-----------

.. code:: none

    ...
    2998    | -0.083372 |         -8.04 |           2.046e+00 |       -53.78 |     0.43
    2999    | -0.146298 |        -14.12 |           2.045e+00 |       -53.78 |     0.42
    3000    | -0.156466 |        -15.10 |           2.044e+00 |       -53.78 |     0.33

    ===========================================================================

    +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    Finishing Widom simulation
    +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        Average properties of the system:
        ------------------------------------------------------------------------------
        Henry coefficient: [mol/kg/Pa]       2.04418e+00 +/-  3.33591e-01 [-]
        Enthalpy of adsorption: [kJ/mol]       -53.78391 +/-      0.34888 [-]

    ===========================================================================
    Simulation finished successfully!
    ===========================================================================

    Simulation finished at 2025-11-17 23:29:43
    Simulation duration: 0:13:37.214066
    ===========================================================================



Finally the simulation result is printed, with the average values reported.

Also, a json file is saved with the main results, that can be easily read and processed for further analysis.

.. code:: python

    {
        "code_version": "0.4.4",
        "random_seed": 42,
        "enlapsed_time_hours": 0.22700391277777776,
        "total_insertions": 3001,
        "temperature_K": 298.0,
        "henry_coefficient_mol_kg-1_Pa-1": 2.044178342815808,
        "henry_coefficient_std_mol_kg-1_Pa-1": 0.3335905351383779,
        "enthalpy_of_adsorption_kJ_mol-1": -53.78390928504431,
        "enthalpy_of_adsorption_std_kJ_mol-1": 0.3488799686455756
    }


References
~~~~~~~~~~

.. [1] Mason, J.A., Sumida, K., Herm, Z.R., Krishna, R. and Long, J.R., 2011. Evaluating metal-organic frameworks for post-combustion carbon dioxide capture via temperature swing adsorption. Energy & Environmental Science, 4(8), pp.3030-3040. https://pubs.rsc.org/en/content/articlelanding/2011/ee/c1ee01720a
.. [2] Batatia, I., Benner, P., Chiang, Y., Elena, A.M., Kovács, D.P., Riebesell, J., Advincula, X.R., Asta, M., Avaylon, M., Baldwin, W.J. and Berger, F., 2025. A foundation model for atomistic materials chemistry. The Journal of Chemical Physics, 163(18). https://pubs.aip.org/aip/jcp/article/163/18/184110/3372267/A-foundation-model-for-atomistic-materials