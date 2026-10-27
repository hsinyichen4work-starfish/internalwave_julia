#!/bin/bash
#SBATCH --job-name=tpxo_tide
#SBATCH --account=uso102
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/expanse/lustre/projects/uso101/hchen54/input_cmems/tpxo_tide_%j.log

~/.venvs/utide/bin/python -I /home/hchen54/internalwave_julia/input_file_make/tpxo_tide_make.py
