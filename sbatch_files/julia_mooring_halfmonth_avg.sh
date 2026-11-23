#!/bin/bash
#SBATCH --job-name=moor_hmavg
#SBATCH --account=uso102
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=moor_hmavg_%A_%a.out
#SBATCH --error=moor_hmavg_%A_%a.err
# Half-month / month / total means of the extract_mooring_all.jl output (model_validation/mooring/mooring_halfmonth_avg.jl).
#   J=$(sbatch --parsable julia_extract_mooring_all.sh); sbatch --dependency=aftercorr:$J julia_mooring_halfmonth_avg.sh

export PATH=/home/hchen54/.juliaup/bin${PATH:+:${PATH}}
SRCS=(design_1113_hycom_2700 design_1116_cmems_2700 amazon_900m_dbry_0927 CMEMS)
export MX_SRC=${SRCS[$SLURM_ARRAY_TASK_ID]}
export MX_T0=${MX_T0:-2022-08-01} MX_T1=${MX_T1:-2022-12-01}
cd /home/hchen54/internalwave_julia
mkdir -p log_files
julia --project=. model_validation/mooring/mooring_halfmonth_avg.jl 2>&1 | tee log_files/mooring_halfmonth_avg_${MX_SRC}.log
