#!/bin/bash
#SBATCH --job-name=meanflow
#SBATCH --account=uso102
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=runcode_%j.out
#SBATCH --error=runcode_%j.err
#SBATCH --mail-user=HsinYi.Chen@usm.edu
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90
# mean flow of the analysis window and of each month (DFF/mean_flow_month.jl): (1) top-level u, v and ubar, vbar of
# every his file in 2022-08-27 .. 11-30 -> <bcef_dir>/mean_flow_month.nc (~6.5 s per file on the 2.7 km grid: ~1 h);
# (2) figures -> ~/figure_check (MF_FIG=... for another folder).
# Run: sbatch julia_meanflow.sh                                                              (amazon_2700m_cmems_0801)
#      sbatch --time=08:00:00 --export=ALL,DFF_RUN=amazon_900m_3mon_1008_2 julia_meanflow.sh   (900m: bigger files)
#      sbatch --export=ALL,MF_STEPS=2 julia_meanflow.sh                                      (figures only; ":" in lists)
set -o pipefail   # job shows FAILED (not COMPLETED) when julia fails, despite the tee
export PATH=/home/hchen54/.juliaup/bin${PATH:+:${PATH}}
cd /home/hchen54/internalwave_julia
mkdir -p log_files
export JULIA_PROJECT=/home/hchen54/internalwave_julia
julia --heap-size-hint=12G DFF/mean_flow_month.jl 2>&1 | tee log_files/MEANFLOW_${SLURM_JOB_ID}.log
