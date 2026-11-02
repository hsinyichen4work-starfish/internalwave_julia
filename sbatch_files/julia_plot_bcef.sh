#!/bin/bash
#SBATCH --job-name=plot_bcef
#SBATCH --account=uso102
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=runcode_%j.out
#SBATCH --error=runcode_%j.err
#SBATCH --mail-user=HsinYi.Chen@usm.edu
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90

# figures of the north / south mode-1 flux (DFF/DFF_ploting_bcef.jl). Run: sbatch julia_plot_bcef.sh
# (3-month run), or sbatch --export=ALL,DFF_RUN=amazon_900m_pure_or julia_plot_bcef.sh
set -o pipefail   # job shows FAILED (not COMPLETED) when julia fails, despite the tee
export PATH=/home/hchen54/.juliaup/bin${PATH:+:${PATH}}
cd /home/hchen54/internalwave_julia
mkdir -p log_files
export JULIA_PROJECT=/home/hchen54/internalwave_julia
julia --heap-size-hint=24G DFF/DFF_ploting_bcef.jl 2>&1 | tee log_files/PLOTBCEF_${SLURM_JOB_ID}.log
