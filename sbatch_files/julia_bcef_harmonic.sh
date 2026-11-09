#!/bin/bash
#SBATCH --job-name=bcef_harm
#SBATCH --account=uso102
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=runcode_%j.out
#SBATCH --error=runcode_%j.err
#SBATCH --mail-user=HsinYi.Chen@usm.edu
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90

# harmonic analysis of the modal pn, un, vn and the constituent energy flux (DFF/mode_bcef_harmonic.jl), full grid,
# one field (~5.5 GB on the 900 m grid) at a time. Output: <run>/ddf_bcef/bcef_modes<N>_harmonic.nc
# Run: sbatch julia_bcef_harmonic.sh                                            (3-month run, modes 1 and 2)
#      sbatch --export=ALL,DFF_RUN=amazon_900m_pure_or julia_bcef_harmonic.sh
#      sbatch --export=ALL,HARM_MODES=1,2,3,HARM_CONS=M2,S2,N2,K1,O1 julia_bcef_harmonic.sh   (default also fits M4, MS4, MN4, M6)
set -o pipefail   # job shows FAILED (not COMPLETED) when julia fails, despite the tee
export PATH=/home/hchen54/.juliaup/bin${PATH:+:${PATH}}
cd /home/hchen54/internalwave_julia
mkdir -p log_files
export JULIA_PROJECT=/home/hchen54/internalwave_julia
julia --heap-size-hint=48G DFF/mode_bcef_harmonic.jl 2>&1 | tee log_files/BCEFHARM_${SLURM_JOB_ID}.log
