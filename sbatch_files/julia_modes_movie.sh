#!/bin/bash
#SBATCH --job-name=mode_mov
#SBATCH --account=uso102
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=runcode_%j.out
#SBATCH --error=runcode_%j.err
#SBATCH --mail-user=HsinYi.Chen@usm.edu
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90
# surface velocity movies of the modes (DFF/modes_filter_analysis.jl step 3, MF_SNAP empty): u | v | speed, mode 1 / 2 / 1+2,
# raw record and the filtered bands (step 2 files) -> figure/<run>/mode_filter/mode_surf_uv_<band>_mode<set>.mp4
# (png frames in a temporary folder, deleted after ffmpeg). One band per job, the 4 bands in parallel:
#   for b in all M2 hp28 hp9; do sbatch --export=ALL,MF_BANDS=$b julia_modes_movie.sh; done
# Options: MF_SETS=1:2:12, MF_MOVIE_STRIDE=1 [h], MF_FPS=12, MF_MOVIE_DIR=...
set -o pipefail   # job shows FAILED (not COMPLETED) when julia fails, despite the tee
export PATH=/home/hchen54/.juliaup/bin${PATH:+:${PATH}}
cd /home/hchen54/internalwave_julia
mkdir -p log_files
export JULIA_PROJECT=/home/hchen54/internalwave_julia
export MF_STEPS=3 MF_SNAP=""
julia --heap-size-hint=12G DFF/modes_filter_analysis.jl 2>&1 | tee log_files/MODEMOV_${MF_BANDS:-all}_${SLURM_JOB_ID}.log
