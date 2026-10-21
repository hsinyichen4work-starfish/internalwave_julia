#!/bin/bash
#SBATCH --job-name=tidalcomp
#SBATCH --account=uso102
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=tidalcomp_%j.out
#SBATCH --error=tidalcomp_%j.err

# usage: sbatch julia_tidalcomp.sh            full record
#        sbatch --export=ALL,TEST=true julia_tidalcomp.sh   first 30 days, M2 S2 N2 K1 O1
#        add REFIT=true to redo the harmonic fit instead of reusing the saved constants
#        sbatch --export=ALL,SMOOTH=true julia_tidalcomp.sh   smooth the 1/30 deg atlas (sigma 20 km), outputs _s20
set -o pipefail
export PATH=/home/hchen54/.juliaup/bin${PATH:+:${PATH}}
cd /home/hchen54/internalwave_julia
mkdir -p log_files
julia -t 8 ROMS_TPXO_tidalcomp.jl 2>&1 | tee log_files/tidalcomp_${TEST:+test_}${SLURM_JOB_ID}.log
