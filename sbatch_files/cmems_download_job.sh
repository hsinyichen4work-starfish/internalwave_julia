#!/bin/bash
#SBATCH --job-name=cmems_dl
#SBATCH --account=uso102
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/expanse/lustre/projects/uso101/hchen54/cmems_download_%j.log

bash /home/hchen54/internalwave_julia/input_file_make/cmems_download.sh
