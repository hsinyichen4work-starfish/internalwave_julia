#!/bin/bash
#SBATCH --job-name=moor_all
#SBATCH --account=uso102
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=moor_all_%A_%a.out
#SBATCH --error=moor_all_%A_%a.err
# Moorings + M1-CPIES7 transect for every model, same method (model_validation/mooring/extract_mooring_all.jl),
# one array task per source. HYCOM / NCOM: run on aurelia.
#   sbatch julia_extract_mooring_all.sh                  # 3 ROMS runs + CMEMS
#   sbatch --array=3 julia_extract_mooring_all.sh        # only CMEMS

export PATH=/home/hchen54/.juliaup/bin${PATH:+:${PATH}}
SRCS=(design_1113_hycom_2700 design_1116_cmems_2700 amazon_900m_dbry_0927 CMEMS)
export MX_SRC=${SRCS[$SLURM_ARRAY_TASK_ID]}
export MX_T0=${MX_T0:-2022-08-01} MX_T1=${MX_T1:-2022-12-01}   # same window as the mean-flow loop
cd /home/hchen54/internalwave_julia
mkdir -p log_files
julia --project=. --heap-size-hint=8G model_validation/mooring/extract_mooring_all.jl 2>&1 | tee log_files/extract_mooring_all_${MX_SRC}.log
