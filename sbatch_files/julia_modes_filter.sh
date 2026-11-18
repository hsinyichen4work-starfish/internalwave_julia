#!/bin/bash
#SBATCH --job-name=mode_filt
#SBATCH --account=uso102
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=runcode_%j.out
#SBATCH --error=runcode_%j.err
#SBATCH --mail-user=HsinYi.Chen@usm.edu
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90
# Butterworth filter of the modal pn, un, vn (modes 1, 2) and W_BT, bands M2 9-15 h / hp28 / hp9 (DFF/modes_filter_analysis.jl,
# step 2) -> <bcef_dir>/model_and_filter/filt_<band>_<field>.nc (~34 GB on the 2.7 km grid). Finished files are skipped,
# so a job that ran out of time can be resubmitted as it is.
# Run: sbatch julia_modes_filter.sh                                                        (amazon_2700m_cmems_0801)
#      sbatch --mem=64G --time=08:00:00 --export=ALL,DFF_RUN=amazon_900m_3mon_1008_2 julia_modes_filter.sh   (900 m, ~110 GB)
#      sbatch --export=ALL,MF_MODES=1:2:3 julia_modes_filter.sh                            (modes 1-3; ":" in lists)
set -o pipefail   # job shows FAILED (not COMPLETED) when julia fails, despite the tee
export PATH=/home/hchen54/.juliaup/bin${PATH:+:${PATH}}
cd /home/hchen54/internalwave_julia
mkdir -p log_files
export JULIA_PROJECT=/home/hchen54/internalwave_julia
export MF_STEPS=${MF_STEPS:-2}
julia -t ${SLURM_CPUS_PER_TASK} --heap-size-hint=24G DFF/modes_filter_analysis.jl 2>&1 | tee log_files/MODEFILT_${SLURM_JOB_ID}.log
