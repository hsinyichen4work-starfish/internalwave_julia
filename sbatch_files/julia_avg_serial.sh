#!/bin/bash
#SBATCH --job-name=zeta_plots_dist
#SBATCH --account=uso102
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=zeta_plots_dist_%j.out
#SBATCH --error=zeta_plots_dist_%j.err
#SBATCH --mail-user=HsinYi.Chen@usm.edu
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90
 
export PATH=/home/hchen54/.juliaup/bin${PATH:+:${PATH}}  # replace with however Julia gets loaded on this cluster
 cd /home/hchen54/internalwave_julia
mkdir -p log_files 
julia check_output.jl  2>&1 | tee log_files/serial.log  # adjust to wherever you keep the script