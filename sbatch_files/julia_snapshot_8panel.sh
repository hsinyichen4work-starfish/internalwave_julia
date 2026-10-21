#!/bin/bash
#SBATCH --job-name=snap8
#SBATCH --account=uso102
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=runcode_%j.out
#SBATCH --error=runcode_%j.err

export PATH=/home/hchen54/.juliaup/bin${PATH:+:${PATH}}
cd /home/hchen54/internalwave_julia
mkdir -p log_files
julia --project=. snapshot_8panel.jl 2>&1 | tee log_files/snapshot_8panel.log
