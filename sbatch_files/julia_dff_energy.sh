#!/bin/bash
#SBATCH --job-name=dff_energy
#SBATCH --account=uso102
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=runcode_%j.out
#SBATCH --error=runcode_%j.err
#SBATCH --mail-user=HsinYi.Chen@usm.edu
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90
# modal KE and APE of each direction from the time-series DFF (DFF/DFF_energy_on_bcef.jl) -> <bcef_dir>/bcef_dff_energy_modes3.nc
# (time-mean maps) and bcef_dff_energy_modes3_strip_xi<a>-<b>.nc (strip mean every hour). Peak memory as DFF_on_bcef.jl:
# ~14 GB on the 2.7 km grid, ~40 GB on the 900 m grid (use --mem=96G --time=05:00:00 there).
# Run: sbatch julia_dff_energy.sh                                                           (2700 m, every mode and band)
#      sbatch --export=ALL,ENERGY_MODES=1,ENERGY_BANDS=M2 julia_dff_energy.sh               (test: mode 1, M2 -> *_test.nc)
#      sbatch --mem=96G --time=05:00:00 --export=ALL,DFF_RUN=amazon_900m_3mon_1008_2 julia_dff_energy.sh
set -o pipefail   # job shows FAILED (not COMPLETED) when julia fails, despite the tee
export PATH=/home/hchen54/.juliaup/bin${PATH:+:${PATH}}
cd /home/hchen54/internalwave_julia
mkdir -p log_files
export JULIA_PROJECT=/home/hchen54/internalwave_julia
julia --heap-size-hint=40G DFF/DFF_energy_on_bcef.jl 2>&1 | tee log_files/DFFENERGY_${SLURM_JOB_ID}.log
