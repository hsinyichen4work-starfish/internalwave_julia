#!/bin/bash
#SBATCH --job-name=cmems_dl_long
#SBATCH --account=uso102
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/expanse/lustre/projects/uso101/hchen54/cmems_download_%j.log

# CMEMS for 2022-08-22..2022-11-30 (+1 day margin each side)
T0=2022-08-21T00:00:00 T1=2022-12-01T00:00:00 SUF=_0822_1130 \
bash /home/hchen54/internalwave_julia/input_file_make/cmems_download.sh
