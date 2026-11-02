#!/bin/bash
#SBATCH --job-name=time_capture_roms
#SBATCH --account=uso102
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=time_capture_roms_%j.out
#SBATCH --error=time_capture_roms_%j.err

export PATH=/home/hchen54/.juliaup/bin${PATH:+:${PATH}}
cd /home/hchen54/internalwave_julia
mkdir -p log_files
julia --project=. model_validation/HYCOM/time_capture_ROMS.jl 2>&1 | tee log_files/time_capture_ROMS.log
