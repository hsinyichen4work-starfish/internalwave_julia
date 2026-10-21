#!/bin/bash
#SBATCH --job-name=samegrid_comp
#SBATCH --account=uso102
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=samegrid_comp_%j.out
#SBATCH --error=samegrid_comp_%j.err

# usage: sbatch julia_samegrid_comp.sh
# 32 CPUs = 31 workers + the main process; ~10 plots x ~20 s per time step,
# 721 time steps -> ~1.3 h. Each worker peaks around 3 GB.
# re-running skips snapshots that already exist and only redraws the time mean
export PATH=/home/hchen54/.juliaup/bin${PATH:+:${PATH}}
cd /home/hchen54/internalwave_julia
mkdir -p log_files
julia --project=. model_samegrid_comp.jl 2>&1 | tee log_files/samegrid_comp_${SLURM_JOB_ID}.log
