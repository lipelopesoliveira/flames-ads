# GCMC + MD of CO2 adsorption on MIL-120 using DeepMD

This tutorial shows how to use DeepMD as a calculator in FLAMES to perform GCMC + MD simulations of CO2 adsorption on MIL-120.

This tutorial is part of the paper: Decoding local framework dynamics in the ultra-small pore MOF MIL-120(Al) CO2 sorbent with machine-learning potential

First, you need to download the DeepMD trained potential from the Zenodo repository: https://zenodo.org/records/17618381/files/mlp-deepmd.zip?download=1

Unzip the downloaded file and place the folder `mlp-deepmd` inside the `examples/Calculators/DeepMD/` directory.

Then, run the script `run_GCMC_MD.py`. This script performs GCMC + MD simulations of CO2 adsorption on MIL-120 using the DeepMD potential.

You can use the environment.yml to create the conda environment with the DeepMD package installed.
