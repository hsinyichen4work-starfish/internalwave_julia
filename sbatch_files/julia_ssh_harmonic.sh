#!/bin/bash
#SBATCH --job-name=ssh_harm
#SBATCH --account=uso102
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=ssh_harm_%j.out
#SBATCH --error=ssh_harm_%j.err

# usage: sbatch julia_ssh_harmonic.sh                         full grid (~5.5 GB of SSH in memory)
#        sbatch --export=ALL,STRIDE=10 julia_ssh_harmonic.sh  every 10th grid point (quick test, outputs _s10)
set -o pipefail
export PATH=/home/hchen54/.juliaup/bin${PATH:+:${PATH}}
cd /home/hchen54/internalwave_julia
mkdir -p log_files
julia -t 8 --project=. model_validation/tidal/ssh_harmonic.jl 2>&1 | tee log_files/ssh_harmonic_${STRIDE:+s${STRIDE}_}${SLURM_JOB_ID}.log
