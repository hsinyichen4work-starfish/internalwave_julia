#!/bin/bash
#SBATCH --job-name=his_plots_cmems
#SBATCH --account=uso102
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=his_plots_cmems_%j.out
#SBATCH --error=his_plots_cmems_%j.err
#SBATCH --mail-user=HsinYi.Chen@usm.edu
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90
 
export PATH=/home/hchen54/.juliaup/bin${PATH:+:${PATH}}  # replace with however Julia gets loaded on this cluster
cd /home/hchen54/internalwave_julia
mkdir -p log_files 
CHECK_RUN=cmems_2700m julia output_check/check_output_parallel.jl 2>&1 | tee log_files/check_output_cmems_2700m.log