#!/bin/bash
#SBATCH --job-name=meanflow_avg
#SBATCH --account=uso102
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=meanflow_avg_%A_%a.out
#SBATCH --error=meanflow_avg_%A_%a.err
# Monthly + total surface mean maps (meanflow/mean_flow_avg.jl), one array task per run.
#   sbatch julia_meanflow_avg.sh                  # all four runs
#   sbatch --array=1 julia_meanflow_avg.sh        # only amazon_900m_3mon_1008

export PATH=/home/hchen54/.juliaup/bin${PATH:+:${PATH}}
RUNS=(amazon_2700m_cmems_0801 amazon_900m_3mon_1008 design_1113_hycom_2700 design_1116_cmems_2700)
export MF_RUN=${RUNS[$SLURM_ARRAY_TASK_ID]}
cd /home/hchen54/internalwave_julia
mkdir -p log_files
julia --project=. --heap-size-hint=8G meanflow/mean_flow_avg.jl 2>&1 | tee log_files/meanflow_avg_${MF_RUN}.log
