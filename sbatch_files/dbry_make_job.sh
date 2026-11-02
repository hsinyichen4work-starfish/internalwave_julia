#!/bin/bash
#SBATCH --job-name=dbry_make
#SBATCH --account=uso102
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/expanse/lustre/projects/uso102/hchen54/input_cmems/dbry_make_%j.log

export PATH=/home/hchen54/.juliaup/bin${PATH:+:${PATH}}
cd /home/hchen54/internalwave_julia
julia -t 16 --project=. input_file_make/dbry_make.jl
