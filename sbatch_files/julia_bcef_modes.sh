#!/bin/bash
#SBATCH --job-name=bcef_modes
#SBATCH --account=uso102
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=runcode_%j.out
#SBATCH --error=runcode_%j.err
#SBATCH --mail-user=HsinYi.Chen@usm.edu
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90

# pn, un, vn of the first NMODE modes from the his files (DFF/bcef_modes_puv.jl): 15 workers read and project the files
# in parallel; the main process writes each batch and frees it. Run (default 3 modes, 3-month run):
#   sbatch --export=ALL,DFF_RUN=amazon_900m_pure_or,NMODE=3 julia_bcef_modes.sh
#   sbatch --export=ALL,DFF_RUN=amazon_2700m_cmems_0801,START_DATE=2022-08-24,BCEF_OUT=/expanse/lustre/projects/uso102/hchen54/amazon_2700m_cmems_0801/ddf_bcef_0824 julia_bcef_modes.sh   (skip the spin-up)
set -o pipefail   # job shows FAILED (not COMPLETED) when julia fails, despite the tee
export PATH=/home/hchen54/.juliaup/bin${PATH:+:${PATH}}
cd /home/hchen54/internalwave_julia
mkdir -p log_files
export JULIA_PROJECT=/home/hchen54/internalwave_julia   # the worker processes inherit this
julia --heap-size-hint=8G DFF/bcef_modes_puv.jl 2>&1 | tee log_files/BCEFMODES_${SLURM_JOB_ID}.log
