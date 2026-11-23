#!/bin/bash
#SBATCH --job-name=meanflow_loop
#SBATCH --account=uso102
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=meanflow_loop_%A_%a.out
#SBATCH --error=meanflow_loop_%A_%a.err
# Monthly + half-monthly means (3-D file + surface maps) (meanflow/mean_flow_avg_loop.jl), one array task per source.
#   sbatch julia_meanflow_avg_loop.sh                 # 3 ROMS runs + CMEMS
#   sbatch --array=3 julia_meanflow_avg_loop.sh       # only CMEMS
#   sbatch --export=ALL,MF_DEPTH_ONLY=1 --time=04:00:00 julia_meanflow_avg_loop.sh   # only depth maps from existing 3-D files

export PATH=/home/hchen54/.juliaup/bin${PATH:+:${PATH}}
SRCS=(design_1113_hycom_2700 design_1116_cmems_2700 amazon_900m_dbry_0927 CMEMS)   # HYCOM, NCOM: run on aurelia
export MF_SRC=${SRCS[$SLURM_ARRAY_TASK_ID]}
cd /home/hchen54/internalwave_julia
mkdir -p log_files
SUF=$([ "${MF_DEPTH_ONLY:-0}" = 1 ] && echo _depths)
julia --project=. --heap-size-hint=8G meanflow/mean_flow_avg_loop.jl 2>&1 | tee log_files/meanflow_avg_loop_${MF_SRC}${SUF}.log
