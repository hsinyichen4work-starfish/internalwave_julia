#!/bin/bash
#SBATCH --job-name=binplot
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
# strip (±200 km) bin-average line plots (DFF/mode_dff_binplot.jl) -> figure/<run>/bin_avg_lineplot (+ harm_dff/, dff/).
# Needs bcef_modes3_{structure,harmonic,puv}.nc, bcef_dff_flux_modes3*.nc (+ _transect_strip_*), bt_conv_harmonic.nc
# (W_BT_make.jl, steps 7-9) and harm_dff_flux.nc (DFF_on_harmonic.jl, step 11).
# Run: sbatch julia_binplot.sh                                                            (amazon_2700m_cmems_0801, all steps)
#      sbatch --export=ALL,DFF_RUN=amazon_900m_3mon_1008_2 julia_binplot.sh               (900m, ddf_bcef)
#      sbatch --export=ALL,DFF_RUN=...,BIN_STEPS=7:8:9 julia_binplot.sh                   (":" in lists)
set -o pipefail   # job shows FAILED (not COMPLETED) when julia fails, despite the tee
export PATH=/home/hchen54/.juliaup/bin${PATH:+:${PATH}}
cd /home/hchen54/internalwave_julia
mkdir -p log_files
export JULIA_PROJECT=/home/hchen54/internalwave_julia
julia --heap-size-hint=24G DFF/mode_dff_binplot.jl 2>&1 | tee log_files/BINPLOT_${SLURM_JOB_ID}.log
