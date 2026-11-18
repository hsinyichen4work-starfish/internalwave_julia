#!/bin/bash
#SBATCH --job-name=era5_dl_new
#SBATCH --account=uso102
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=/expanse/lustre/projects/uso102/hchen54/era5_download_%j.log

# ERA5 07-31 .. 12-01, area N16 W-56 S-8 E-33, one CDS request per month
# (202207 .. 202212) into ERA5_DATA_new/. Months already there are skipped, so
# just resubmit if a request fails or the job times out in the CDS queue.
~/envs/cmems/bin/python -I /home/hchen54/internalwave_julia/input_file_make/era5_download.py
