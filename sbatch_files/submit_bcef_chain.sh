#!/bin/bash
# Submit the mode-1 energy-flux DFF as a chain, for one or more ROMS runs (the ssh DFF is not redone; that is
# submit_dff_chain.sh). The second job starts only after the first finished successfully (afterok); if the first
# fails, SLURM cancels the second. The chains of different runs are independent and can run at the same time.
#   1. bcef_mode1.jl    julia_bcef_mode1.sh   (128G, 16 CPUs)   his files -> p1, u1, v1   (ddf_bcef/bcef_mode1_*.nc)
#   2. DFF_on_bcef.jl   julia_dff_bcef.sh     (96G, 8 CPUs)     north / south flux       (ddf_bcef/bcef_dff_flux*.nc)
# The run is passed to the scripts as DFF_RUN (read by DFF/dff_run.jl): input /expanse/.../<run>/after_joint,
# output /expanse/.../<run>/ddf_bcef.
# Usage: bash sbatch_files/submit_bcef_chain.sh [run ...]      (run it with bash, not sbatch)
#   e.g. bash sbatch_files/submit_bcef_chain.sh amazon_900m_pure_or
#        bash sbatch_files/submit_bcef_chain.sh amazon_900m_3mon_1008_2 amazon_900m_pure_or
#   no run given: amazon_900m_3mon_1008_2
set -e
cd "$(dirname "$0")"   # the jobs' .out/.err files land in sbatch_files/, as when submitting by hand

runs=("$@")
[ ${#runs[@]} -eq 0 ] && runs=(amazon_900m_3mon_1008_2)

for run in "${runs[@]}"; do
    if [ ! -d "/expanse/lustre/projects/uso101/hchen54/$run/after_joint" ]; then
        echo "skipping $run: no /expanse/lustre/projects/uso101/hchen54/$run/after_joint"
        continue
    fi
    tag=${run#amazon_900m_}   # short name for squeue, e.g. pure_or
    env="--export=ALL,DFF_RUN=$run"
    j1=$(sbatch --parsable $env --job-name=bcef1_$tag julia_bcef_mode1.sh)
    j2=$(sbatch --parsable $env --job-name=dffbcef_$tag --dependency=afterok:$j1 --kill-on-invalid-dep=yes julia_dff_bcef.sh)
    echo "$run: bcef_mode1 $j1 -> DFF_on_bcef $j2"
done
echo "check with: squeue -u $USER   (waiting jobs show reason 'Dependency')"
