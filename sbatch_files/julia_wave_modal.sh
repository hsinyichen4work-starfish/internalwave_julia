#!/bin/bash
#SBATCH --job-name=wmf_modal
#SBATCH --account=uso102
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-8
#SBATCH --output=wmf_modal_%A_%a.out
#SBATCH --error=wmf_modal_%A_%a.err
# Steps 8-9 of wave_meanflow/wave_mean_flow.jl: u_bc, v_bc, p_bc (= wave − bt) projected on the window's modes,
# one array task per field x band (task = 3 * field + band; fields u v p, bands hp28 bp9_15 hp9).
# Needs steps 4 (modes10), 5-6 (bands) and 7 (bt) done. Finished windows are skipped.
#   sbatch julia_wave_modal.sh                                          # all 9
#   sbatch --array=1 --export=ALL,WMF_WINDOWS=1 julia_wave_modal.sh    # u, bp9_15, window 1 (test)

set -o pipefail   # a julia error fails the job (tee alone would hide it)
export PATH=/home/hchen54/.juliaup/bin${PATH:+:${PATH}}
FIELDS=(u v p); BANDS=(hp28 bp9_15 hp9)
export WMF_VARS=zeta WMF_STEPS=9
export WMF_MODAL_FIELDS=${FIELDS[$((SLURM_ARRAY_TASK_ID / 3))]} WMF_MODAL_BANDS=${BANDS[$((SLURM_ARRAY_TASK_ID % 3))]}
cd /home/hchen54/internalwave_julia
mkdir -p log_files
julia --project=. --heap-size-hint=12G wave_meanflow/wave_mean_flow.jl 2>&1 | tee log_files/wave_meanflow_modal_${WMF_MODAL_FIELDS}_${WMF_MODAL_BANDS}_win${WMF_WINDOWS:-all}.log
