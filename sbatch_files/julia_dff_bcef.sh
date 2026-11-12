#!/bin/bash
#SBATCH --job-name=dff_bcef
#SBATCH --account=uso102
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --output=runcode_%j.out
#SBATCH --error=runcode_%j.err
#SBATCH --mail-user=HsinYi.Chen@usm.edu
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90

# directional Fourier filter of the flux of modes 1..NMODE (DFF/DFF_on_bcef.jl, reads bcef_modes<NMODE>_puv.nc;
# NMODE from --export, default 3), one mode at a time: one process, FFTs threaded over the 8 CPUs;
# peak ~3 spectra + 3 filtered fields + FFT temporaries of one mode (~40 GB for the 3-month run); ~20 min per mode
# Also writes the hourly flux averaged across the ±200 km strip (STRIP_KM) around the mooring line (..._transect_strip_*.nc).
# Run: sbatch julia_dff_bcef.sh                                       (3-month 900 m run, ddf_bcef)
#      sbatch --export=ALL,DFF_RUN=amazon_2700m_cmems_0801,BCEF_OUT=/expanse/lustre/projects/uso102/hchen54/amazon_2700m_cmems_0801/ddf_bcef_0824 julia_dff_bcef.sh
set -o pipefail   # job shows FAILED (not COMPLETED) when julia fails, despite the tee
export PATH=/home/hchen54/.juliaup/bin${PATH:+:${PATH}}
cd /home/hchen54/internalwave_julia
mkdir -p log_files
export JULIA_PROJECT=/home/hchen54/internalwave_julia
julia --heap-size-hint=80G DFF/DFF_on_bcef.jl 2>&1 | tee log_files/DFFBCEF_${SLURM_JOB_ID}.log
