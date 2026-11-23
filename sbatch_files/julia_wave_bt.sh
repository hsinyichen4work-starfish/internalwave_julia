#!/bin/bash
#SBATCH --job-name=wmf_bt
#SBATCH --account=uso102
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --array=0-2
#SBATCH --output=wmf_bt_%A_%a.out
#SBATCH --error=wmf_bt_%A_%a.err
# Step 7 of wave_meanflow/wave_mean_flow.jl: u_bt, v_bt, p_bt of each band, one array task per field.
# Needs steps 5-6 done (u_wave_*, v_wave_*, p_wave_*, zeta_wave_* bands). Finished band-windows are skipped.
#   sbatch julia_wave_bt.sh                                    # u, v, p
#   sbatch --dependency=afterok:<bands job id> julia_wave_bt.sh  # start when steps 5-6 are done

export PATH=/home/hchen54/.juliaup/bin${PATH:+:${PATH}}
FIELDS=(u v p)
export WMF_VARS=zeta WMF_STEPS=7 WMF_BT_FIELDS=${FIELDS[$SLURM_ARRAY_TASK_ID]}
cd /home/hchen54/internalwave_julia
mkdir -p log_files
julia --project=. --heap-size-hint=6G wave_meanflow/wave_mean_flow.jl 2>&1 | tee log_files/wave_meanflow_bt_${WMF_BT_FIELDS}.log
