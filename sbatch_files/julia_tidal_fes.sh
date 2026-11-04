#!/bin/bash
#SBATCH --job-name=tidal_fes
#SBATCH --account=uso102
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=tidal_fes_%j.out
#SBATCH --error=tidal_fes_%j.err

# FES chain, outputs in .../figure/amazon_900m_3mon_1008/tidal_comp/:
#   1. ssh_harmonic.jl        ROMS SSH harmonic fit         -> ROMS_amazon900m_sshfit.nc
#   2. ssh_tidalcomp_fes.jl   comparison with FES2022b      -> ROMS_FES_sshfit_comp.nc, ROMS_FES_sshfit_rms.txt
#   3. tidal_comp_plotting.jl figures                       -> ROMS_FES_*.png, ROMS_IT_ssh_maps.png
# usage: sbatch julia_tidal_fes.sh                 all three steps
#        sbatch --export=ALL,SKIPFIT=true julia_tidal_fes.sh   steps 2 and 3 only (fit file already there)
set -eo pipefail
export PATH=/home/hchen54/.juliaup/bin${PATH:+:${PATH}}
cd /home/hchen54/internalwave_julia
mkdir -p log_files
LOG=log_files/tidal_fes_${SLURM_JOB_ID}.log
if [ "${SKIPFIT}" != "true" ]; then
    echo "=== 1. ssh_harmonic.jl ===" | tee -a $LOG
    julia -t 8 --project=. model_validation/tidal/ssh_harmonic.jl 2>&1 | tee -a $LOG
fi
echo "=== 2. ssh_tidalcomp_fes.jl ===" | tee -a $LOG
julia -t 8 --project=. model_validation/tidal/ssh_tidalcomp_fes.jl 2>&1 | tee -a $LOG
echo "=== 3. tidal_comp_plotting.jl ===" | tee -a $LOG
julia -t 8 --project=. model_validation/tidal/tidal_comp_plotting.jl 2>&1 | tee -a $LOG
echo "=== done ===" | tee -a $LOG
