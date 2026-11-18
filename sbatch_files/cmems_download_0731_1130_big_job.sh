#!/bin/bash
#SBATCH --job-name=cmems_dl_big
#SBATCH --account=uso102
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=/expanse/lustre/projects/uso102/hchen54/cmems_download_%j.log

# CMEMS GLO12 for 2022-07-31..2022-11-30 (+ 12-01 00:00 for interpolation),
# box covering the new design 1104 grid (lon -54.3..-34.8, lat -6.3..14.0) + ~0.7 deg.
# Supersedes all earlier downloads (smaller box -51.5..-35.5 / -3.5..14.0).
LON0=-55 LON1=-34 LAT0=-7 LAT1=15 \
T0=2022-07-31T00:00:00 T1=2022-12-01T00:00:00 SUF=_0731_1130_big \
bash /home/hchen54/internalwave_julia/input_file_make/cmems_download.sh
