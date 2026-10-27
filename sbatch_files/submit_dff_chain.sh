#!/bin/bash
# Submit the three DFF steps as a chain, for one or more ROMS runs: each job starts only after the previous one of the
# same run finished successfully (afterok); if one fails, the later ones are cancelled by SLURM instead of running on
# stale files. The chains of different runs are independent and can run at the same time.
#   1. DDF_on_ssh.jl              julia_code_job2.sh     (80G, 8 CPUs)
#   2. DFF_ploting.jl             julia_code_job.sh      (32G, 8 CPUs; time raised to 8 h here for the small-sector frames)
#   3. DFF_transect_analysis.jl   julia_dff_transect.sh  (16G, 2 CPUs)
# The run is passed to the scripts as DFF_RUN (read by DFF/dff_run.jl): input /expanse/.../<run>/after_joint,
# output /expanse/.../<run>/ddf_ssh, figures /expanse/.../figure/<run>/DFF.
# Usage: bash sbatch_files/submit_dff_chain.sh [run ...]
#   e.g. bash sbatch_files/submit_dff_chain.sh amazon_900m_pure_or
#        bash sbatch_files/submit_dff_chain.sh amazon_900m_3mon_1008_2 amazon_900m_pure_or
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
    j1=$(sbatch --parsable $env --job-name=ddf_$tag julia_code_job2.sh)
    j2=$(sbatch --parsable $env --job-name=dffplot_$tag --dependency=afterok:$j1 --kill-on-invalid-dep=yes --time=08:00:00 julia_code_job.sh)
    j3=$(sbatch --parsable $env --job-name=dfftr_$tag --dependency=afterok:$j2 --kill-on-invalid-dep=yes julia_dff_transect.sh)
    echo "$run: DDF_on_ssh $j1 -> DFF_ploting $j2 -> DFF_transect_analysis $j3"
done
echo "check with: squeue -u $USER   (waiting jobs show reason 'Dependency')"
