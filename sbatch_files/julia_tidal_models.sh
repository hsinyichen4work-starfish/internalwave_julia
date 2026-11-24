#!/bin/bash
#SBATCH --job-name=tidal_models
#SBATCH --account=uso102
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=runcode_%j.out
#SBATCH --error=runcode_%j.err
#SBATCH --mail-user=HsinYi.Chen@usm.edu
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90

# SSH tide validation vs FES2022b for the 2700 m ROMS runs (model_validation/tidal/, settings in tidal_models_setup.jl):
#   1. ssh_harmonic_models.jl       harmonic fit of hourly zeta, full grid (~3.2 GB per run)
#   2. ssh_tidalcomp_fes_models.jl  FES comparison on the 2700 m grid; also picks up HYCOM / NCOM2700 fits made on aurelia
#   3. tidal_comp_models_plot.jl    common-area table and figures for every model found
# Output: /expanse/lustre/projects/uso102/hchen54/figure/tidal_comp_models/
# Run: sbatch julia_tidal_models.sh
#      sbatch --export=ALL,TC_SKIP_FIT=1 julia_tidal_models.sh                       (comparison only, all 4 models: fits already made)
#      sbatch --export=ALL,TC_SKIP_FIT=1,TC_COMP_SRC=HYCOM:NCOM2700 julia_tidal_models.sh
#      lists use ":" (sbatch --export splits on ","); TC_FIT_SRC likewise
set -o pipefail
export PATH=/home/hchen54/.juliaup/bin${PATH:+:${PATH}}
cd /home/hchen54/internalwave_julia
mkdir -p log_files
export JULIA_PROJECT=/home/hchen54/internalwave_julia
LOG=log_files/TIDALMODELS_${SLURM_JOB_ID}.log
FIT_SRC=${TC_FIT_SRC:-ROMS_HYCOMparent:ROMS_GLORYSparent}
if [ -n "$TC_SKIP_FIT" ]; then COMP_DEF=ROMS_HYCOMparent:ROMS_GLORYSparent:HYCOM:NCOM2700; else COMP_DEF=$FIT_SRC; fi
COMP_SRC=${TC_COMP_SRC:-$COMP_DEF}
FIT_SRC=${FIT_SRC//:/,}; COMP_SRC=${COMP_SRC//:/,}
echo "fit: ${TC_SKIP_FIT:+skipped }$FIT_SRC   comparison: $COMP_SRC" | tee -a $LOG
if [ -z "$TC_SKIP_FIT" ]; then
    TC_SRC=$FIT_SRC julia --heap-size-hint=24G model_validation/tidal/ssh_harmonic_models.jl 2>&1 | tee -a $LOG || exit 1
fi
TC_SRC=$COMP_SRC julia -t 4 --heap-size-hint=24G model_validation/tidal/ssh_tidalcomp_fes_models.jl 2>&1 | tee -a $LOG || exit 1
julia --heap-size-hint=8G model_validation/tidal/tidal_comp_models_plot.jl 2>&1 | tee -a $LOG
