#!/bin/bash
#SBATCH --job-name=tpxo_tide_1104
#SBATCH --account=uso102
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/expanse/lustre/projects/uso102/hchen54/new_design_cmems/tpxo_tide_%j.log

~/.venvs/utide/bin/python -I /home/hchen54/internalwave_julia/input_file_make/tpxo_tide_make_1104.py
