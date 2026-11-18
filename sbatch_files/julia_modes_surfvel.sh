#!/bin/bash
#SBATCH --job-name=mode_surfvel
#SBATCH --account=uso102
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=runcode_%j.out
#SBATCH --error=runcode_%j.err
#SBATCH --mail-user=HsinYi.Chen@usm.edu
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90
# RMS surface speed √<u_s² + v_s²> of mode 1 / mode 2 / modes 1+2, raw record and the bands M2 / hp28 / hp9, by period
# (DFF/modes_filter_analysis.jl step 10) -> <bcef_dir>/model_and_filter/filt_surfvel.nc and figure/<run>/mode_filter/filt_Urms_*.png
# Run: sbatch julia_modes_surfvel.sh
#      sbatch --export=ALL,MF_SV_BANDS=M2:hp9 julia_modes_surfvel.sh      (some bands only; MF_E_RECOMPUTE=1 to redo the file)
set -o pipefail
export PATH=/home/hchen54/.juliaup/bin${PATH:+:${PATH}}
cd /home/hchen54/internalwave_julia
mkdir -p log_files
export JULIA_PROJECT=/home/hchen54/internalwave_julia
export MF_STEPS=${MF_STEPS:-10}
julia -t 1 --heap-size-hint=12G DFF/modes_filter_analysis.jl 2>&1 | tee log_files/MODESURF_${SLURM_JOB_ID}.log
