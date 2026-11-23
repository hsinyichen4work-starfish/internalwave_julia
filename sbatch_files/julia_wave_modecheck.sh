#!/bin/bash
#SBATCH --job-name=wmf_check
#SBATCH --account=uso102
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=wmf_check_%A_%a.out
#SBATCH --error=wmf_check_%A_%a.err
# Step 9.5 of wave_meanflow/wave_mean_flow.jl: mode checks, one array task per band (hp28 bp9_15 hp9).
#   sbatch julia_wave_modecheck.sh                                       # all bands, all windows
#   sbatch --array=1 --export=ALL,WMF_WINDOWS=1 julia_wave_modecheck.sh  # bp9_15, window 1 (test)

set -o pipefail   # a julia error fails the job (tee alone would hide it)
export PATH=/home/hchen54/.juliaup/bin${PATH:+:${PATH}}
BANDS=(hp28 bp9_15 hp9)
export WMF_VARS=zeta WMF_STEPS=9.5 WMF_MODAL_BANDS=${BANDS[$SLURM_ARRAY_TASK_ID]}
cd /home/hchen54/internalwave_julia
mkdir -p log_files
julia --project=. --heap-size-hint=20G wave_meanflow/wave_mean_flow.jl 2>&1 | tee log_files/wave_meanflow_modecheck_${WMF_MODAL_BANDS}_win${WMF_WINDOWS:-all}.log
