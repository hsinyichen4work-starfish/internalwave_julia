#!/bin/bash
#SBATCH --job-name=dff_plot
#SBATCH --account=uso102
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=dff_plot_%j.out
#SBATCH --error=dff_plot_%j.err
#SBATCH --mail-user=HsinYi.Chen@usm.edu
#SBATCH --mail-type=END,FAIL

set -o pipefail
export PATH=/home/hchen54/.juliaup/bin${PATH:+:${PATH}}
cd /home/hchen54/internalwave_julia
mkdir -p log_files
export JULIA_PROJECT=/home/hchen54/internalwave_julia
julia DFF/DFF_ploting.jl 2>&1 | tee log_files/DFF_plot_${SLURM_JOB_ID}.log
