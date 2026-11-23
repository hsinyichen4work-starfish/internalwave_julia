#!/bin/bash
#SBATCH --job-name=wmf_proj
#SBATCH --account=uso102
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=1-10
#SBATCH --output=wmf_proj_%A_%a.out
#SBATCH --error=wmf_proj_%A_%a.err
# Section 5 of wave_meanflow/wave_mean_flow.jl (fixed-depth p', u', v' projected on the window's modes), one array
# task per window. Needs sections 2 (means of rho, u, v, zeta) and 4 (modes) done. Finished windows are skipped.
#   sbatch julia_wave_proj.sh                 # windows 1-10
#   sbatch --array=1 julia_wave_proj.sh       # window 1 only (test)

set -o pipefail   # a julia error fails the job (tee alone would hide it)
export PATH=/home/hchen54/.juliaup/bin${PATH:+:${PATH}}
export WMF_VARS=zeta WMF_STEPS=5 WMF_WINDOWS=$SLURM_ARRAY_TASK_ID
cd /home/hchen54/internalwave_julia
mkdir -p log_files
julia --project=. --heap-size-hint=10G wave_meanflow/wave_mean_flow.jl 2>&1 | tee log_files/wave_meanflow_proj_win${WMF_WINDOWS}.log
