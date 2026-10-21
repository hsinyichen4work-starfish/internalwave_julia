#!/bin/bash
#SBATCH --job-name=cmems_input
#SBATCH --account=uso102
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/expanse/lustre/projects/uso101/hchen54/input_cmems/cmems_input_%j.log

export PATH=/home/hchen54/.juliaup/bin${PATH:+:${PATH}}
cd /home/hchen54/internalwave_julia
julia --project=. -e "using Pkg; Pkg.instantiate(); Pkg.precompile()"
julia -t 16 --project=. input_file_make/cmems_input_make.jl
