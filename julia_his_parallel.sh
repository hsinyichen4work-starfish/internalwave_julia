#!/bin/bash
#SBATCH --job-name=zeta_plots_dist
#SBATCH --account=uso101
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=his_plots_dist_%j.out
#SBATCH --error=his_plots_dist_%j.err
#SBATCH --mail-user=HsinYi.Chen@usm.edu
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90
 
export PATH=/home/hchen54/.juliaup/bin${PATH:+:${PATH}}  # replace with however Julia gets loaded on this cluster
 
julia check_output_parallel.jl  > parallel.log 2>&1  # adjust to wherever you keep the script