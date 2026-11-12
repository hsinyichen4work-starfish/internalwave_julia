#!/bin/bash
#SBATCH --job-name=w_bt
#SBATCH --account=uso102
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=runcode_%j.out
#SBATCH --error=runcode_%j.err
#SBATCH --mail-user=HsinYi.Chen@usm.edu
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90

# barotropic bottom vertical velocity input (DFF/W_BT_make.jl): (1) ubar, vbar, zeta of the his files -> rho points ->
# <run>/ddf_bcef_0824/bt_uvz.nc (~7 GB, ~10 min); (2) harmonic fit of ubar, vbar -> bt_harmonic.nc; (3) grad h -> bt_gradh.nc.
# Too big for the login node (8 GB limit: the netCDF library's memory grows with the output file).
# Run: sbatch julia_wbt.sh                                   (amazon_2700m_cmems_0801, from 2022-08-24, steps 1-3)
#      sbatch --export=ALL,BT_STEPS=2 julia_wbt.sh           (the fit only, bt_uvz.nc already there; ":" in lists)
set -o pipefail   # job shows FAILED (not COMPLETED) when julia fails, despite the tee
export PATH=/home/hchen54/.juliaup/bin${PATH:+:${PATH}}
cd /home/hchen54/internalwave_julia
mkdir -p log_files
export JULIA_PROJECT=/home/hchen54/internalwave_julia
julia --heap-size-hint=24G DFF/W_BT_make.jl 2>&1 | tee log_files/WBT_${SLURM_JOB_ID}.log
