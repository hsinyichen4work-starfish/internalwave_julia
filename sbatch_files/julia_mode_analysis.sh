#!/bin/bash
#SBATCH --job-name=mode_analysis
#SBATCH --account=uso102
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=runcode_%j.out
#SBATCH --error=runcode_%j.err
#SBATCH --mail-user=HsinYi.Chen@usm.edu
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90

# vertical-mode analysis figures (DFF/mode_analysis.jl). Steps 1-2 always run; MODE_STEPS picks the others
# (default here 5,6,7,8: the sections along the mooring line -- pressure variance per band, residual, KE, KE/p ratio).
# The first run reads pn, un, vn along the line from the puv file once (~1-2 h on a slow Lustre) and caches them in
# ddf_bcef/bcef_modes<N>_{p,uv}_line_xi<ix>.nc (~75 MB); later runs take a few minutes.
# Run: sbatch julia_mode_analysis.sh
#      sbatch --export=ALL,MODE_STEPS=3:4:5:6:7:8:9:10 julia_mode_analysis.sh   all steps (":" not ",": --export splits on commas)
#      sbatch --export=ALL,DFF_RUN=amazon_2700m_cmems_0801,MODE_STEPS=5:7 julia_mode_analysis.sh   another run
set -o pipefail   # job shows FAILED (not COMPLETED) when julia fails, despite the tee
export PATH=/home/hchen54/.juliaup/bin${PATH:+:${PATH}}
cd /home/hchen54/internalwave_julia
mkdir -p log_files
export JULIA_PROJECT=/home/hchen54/internalwave_julia
export MODE_STEPS=${MODE_STEPS:-5,6,7,8}
echo "DFF_RUN=${DFF_RUN:-default} MODE_STEPS=$MODE_STEPS"
julia --heap-size-hint=12G DFF/mode_analysis.jl 2>&1 | tee log_files/MODEANALYSIS_${SLURM_JOB_ID}.log
