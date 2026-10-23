#!/bin/bash
#SBATCH --job-name=era5_frc
#SBATCH --account=uso102
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/expanse/lustre/projects/uso101/hchen54/input_cmems/era5_frc_%j.log

~/.venvs/utide/bin/python -I /home/hchen54/internalwave_julia/input_file_make/era5_frc_make.py
