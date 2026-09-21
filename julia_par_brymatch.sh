#!/bin/bash
#SBATCH --job-name=dbry_brymatch
#SBATCH --account=uso102
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=30G
#SBATCH --time=24:00:00
#SBATCH --output=dbry_brymatch_dist_%j.out
#SBATCH --error=dbry_brymatch_dist_%j.err
#SBATCH --mail-user=HsinYi.Chen@usm.edu
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90
 
export PATH=/home/hchen54/.juliaup/bin${PATH:+:${PATH}}  # replace with however Julia gets loaded on this cluster
 
julia check_output_brymatch_hpc.jl 2>&1 | tee dbry_match.log  # adjust to wherever you keep the script