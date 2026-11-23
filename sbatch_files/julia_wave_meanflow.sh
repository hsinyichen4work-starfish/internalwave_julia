#!/bin/bash
#SBATCH --job-name=wmf_mean
#SBATCH --account=uso102
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --array=0-8
#SBATCH --output=wmf_mean_%A_%a.out
#SBATCH --error=wmf_mean_%A_%a.err
# wave_meanflow/wave_mean_flow.jl, one array task per variable: its window means (section 2) and, for rho / u / v,
# its wave part at every record (section 5). Sections 3-4 (B, N², modes) need rho, temp, salt, zeta together, so they
# run as one job after the array (WMF_STEPS=3,4). Finished windows are skipped, so any job can be resubmitted.
#   sbatch julia_wave_meanflow.sh                                  # all nine variables, sections 2 + 5
#   sbatch --array=3,4,8 julia_wave_meanflow.sh                    # only u, v, rho
#   sbatch --array=0 --export=ALL,WMF_STEPS=3,4 --mem=16G julia_wave_meanflow.sh   # sections 3-4 afterwards
#          (add --dependency=afterok:<array job id> to start it when the array is done)

set -o pipefail   # a julia error fails the job (tee alone would hide it)
export PATH=/home/hchen54/.juliaup/bin${PATH:+:${PATH}}
VARS=(zeta ubar vbar u v w temp salt rho)
export WMF_VARS=${VARS[$SLURM_ARRAY_TASK_ID]}
export WMF_STEPS=${WMF_STEPS:-2,5}
cd /home/hchen54/internalwave_julia
mkdir -p log_files
julia --project=. --heap-size-hint=4G wave_meanflow/wave_mean_flow.jl 2>&1 | tee log_files/wave_meanflow_${WMF_VARS}_steps${WMF_STEPS//,/}.log
