#!/bin/bash
#SBATCH --job-name=era5_dl
#SBATCH --account=uso102
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=/expanse/lustre/projects/uso101/hchen54/era5_download_%j.log

# CDS requests wait in a queue; the job mostly sleeps
~/envs/cmems/bin/python -I /home/hchen54/internalwave_julia/input_file_make/era5_download.py
