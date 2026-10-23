#!/bin/bash
#SBATCH --job-name=runcode
#SBATCH --account=uso102
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=runcode_%j.out
#SBATCH --error=runcode_%j.err
#SBATCH --mail-user=HsinYi.Chen@usm.edu
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90
 
export PATH=/home/hchen54/.juliaup/bin${PATH:+:${PATH}}  # replace with however Julia gets loaded on this cluster
cd /home/hchen54/internalwave_julia
mkdir -p log_files
export JULIA_PROJECT=/home/hchen54/internalwave_julia   # the worker processes inherit this
julia DFF_ploting.jl 2>&1 | tee log_files/DFFPLOT_${SLURM_JOB_ID}.log
 # adjust to wherever you keep the script