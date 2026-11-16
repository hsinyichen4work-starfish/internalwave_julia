#!/bin/bash
#SBATCH --job-name=harm_dff
#SBATCH --account=uso102
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=runcode_%j.out
#SBATCH --error=runcode_%j.err
#SBATCH --mail-user=HsinYi.Chen@usm.edu
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90
# directional Fourier filter of the harmonic fit (DFF/DFF_on_harmonic.jl): half (north/south) and quad (N/W/S/E) sectors
# of every constituent and mode -> <bcef_dir>/harm_dff_flux.nc; test maps of TEST_CONS (default M2) in ~/figure_check.
# Needs only bcef_modes3_harmonic.nc (not W_BT), so it can run at the same time as julia_wbt.sh.
# Run: sbatch julia_harm_dff.sh                                                          (amazon_2700m_cmems_0801)
#      sbatch --export=ALL,DFF_RUN=amazon_900m_3mon_1008_2 julia_harm_dff.sh             (900m, ddf_bcef)
#      sbatch --mem=64G --export=ALL,DFF_RUN=...,TEST_CONS=M2:K1 julia_harm_dff.sh       (":" in lists)
set -o pipefail   # job shows FAILED (not COMPLETED) when julia fails, despite the tee
export PATH=/home/hchen54/.juliaup/bin${PATH:+:${PATH}}
cd /home/hchen54/internalwave_julia
mkdir -p log_files
export JULIA_PROJECT=/home/hchen54/internalwave_julia
julia --heap-size-hint=24G DFF/DFF_on_harmonic.jl 2>&1 | tee log_files/HDFF_${SLURM_JOB_ID}.log
