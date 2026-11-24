#!/bin/bash
#SBATCH --job-name=m2_modes
#SBATCH --account=uso102
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --output=runcode_%j.out
#SBATCH --error=runcode_%j.err
#SBATCH --mail-user=HsinYi.Chen@usm.edu
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90

# M2 internal tide in the models: vertical modes + harmonic fit of p', u', v' (DFF/m2_modes_models.jl), then the
# common-area comparison and figures (DFF/m2_modes_models_plot.jl; it also reads HYCOM / NCOM2700 results made on aurelia).
# Output: /expanse/lustre/projects/uso102/hchen54/figure/m2_modes_models/
# Run: sbatch julia_m2_modes.sh                                         (ROMS_HYCOMparent, ROMS_GLORYSparent, GLORYS)
#      sbatch --export=ALL,M2_SRC=ROMS:design_1201_xxx_2700 julia_m2_modes.sh   (a new ROMS run on the 2700 m grid)
#      sbatch --export=ALL,M2_PLOT_ONLY=1 julia_m2_modes.sh               (figures only, e.g. after the aurelia runs)
#      lists use ":" (sbatch --export splits on ",")
set -o pipefail
export PATH=/home/hchen54/.juliaup/bin${PATH:+:${PATH}}
cd /home/hchen54/internalwave_julia
mkdir -p log_files
export JULIA_PROJECT=/home/hchen54/internalwave_julia
LOG=log_files/M2MODES_${SLURM_JOB_ID}.log
export M2_SRC=${M2_SRC:-ROMS_HYCOMparent:ROMS_GLORYSparent:GLORYS}
export M2_SRC=${M2_SRC//:/,}
export M2_SRC=${M2_SRC//ROMS,/ROMS:}                    # ROMS:<run> survives the ":" -> "," swap
export M2_NWORK=${M2_NWORK:-15}
export M2_HEAP=${M2_HEAP:-8G}
echo "sources: $M2_SRC, workers $M2_NWORK" | tee -a $LOG
if [ -z "$M2_PLOT_ONLY" ]; then
    julia --heap-size-hint=16G -t 8 DFF/m2_modes_models.jl 2>&1 | tee -a $LOG || exit 1
fi
M2P_SRC=${M2P_SRC:-ROMS_HYCOMparent:ROMS_GLORYSparent:HYCOM:NCOM2700:GLORYS}
M2P_SRC=${M2P_SRC//:/,} julia --heap-size-hint=12G -t 4 DFF/m2_modes_models_plot.jl 2>&1 | tee -a $LOG
