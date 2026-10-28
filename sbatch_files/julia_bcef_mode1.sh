#!/bin/bash
#SBATCH --job-name=bcef_mode1
#SBATCH --account=uso102
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=runcode_%j.out
#SBATCH --error=runcode_%j.err
#SBATCH --mail-user=HsinYi.Chen@usm.edu
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90

# mode-1 p1, u1, v1 from the his files (DFF/bcef_mode1.jl): 15 workers read and project the files in parallel;
# the main process holds p1, u1, v1 for the whole record (~16.5 GB for the 3-month run)
set -o pipefail   # job shows FAILED (not COMPLETED) when julia fails, despite the tee
export PATH=/home/hchen54/.juliaup/bin${PATH:+:${PATH}}
cd /home/hchen54/internalwave_julia
mkdir -p log_files
export JULIA_PROJECT=/home/hchen54/internalwave_julia   # the worker processes inherit this
julia --heap-size-hint=24G DFF/bcef_mode1.jl 2>&1 | tee log_files/BCEFMODE1_${SLURM_JOB_ID}.log
