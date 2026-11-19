#!/bin/bash
#SBATCH --job-name=wmf_bandplot
#SBATCH --account=uso102
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=wmf_bandplot_%j.out
#SBATCH --error=wmf_bandplot_%j.err
export PATH=/home/hchen54/.juliaup/bin${PATH:+:${PATH}}
cd /home/hchen54/internalwave_julia
julia --project=. --heap-size-hint=24G wave_meanflow/tests/test_wave_bands_plot.jl
