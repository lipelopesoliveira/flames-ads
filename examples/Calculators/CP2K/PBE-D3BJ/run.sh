#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --partition=chess
#SBATCH --output=slurm.log
#SBATCH --error=slurm.err
#SBATCH --time=700:00:00
#SBATCH --job-name=PBE_D3_TZVP

echo "The used cpu number is $SLURM_NTASKS"
echo "The used node name is $SLURM_JOB_NODELIST"
echo "The job ID is $SLURM_JOB_ID"
echo "The job started at $(date)"
echo "Working directory: $SLURM_SUBMIT_DIR"

# Load necessary modules, if needed
#module load intel/2023.1.0
#module load impi/2021.9.0
#module load mpi/2021.9.0

cd $SLURM_SUBMIT_DIR

ulimit -s unlimited
ulimit -m unlimited
export OMP_NUM_THREADS=1

source /opt/cp2k-2023.1/tools/toolchain/install/setup

export ASE_CP2K_COMMAND="mpirun -np 24 /opt/cp2k-2023.1/exe/local/cp2k_shell.psmp"

python widom.py
