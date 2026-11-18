#!/bin/bash
#SBATCH --job-name=era5_swap
#SBATCH --account=uso102
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/expanse/lustre/projects/uso102/hchen54/era5_swap_%j.log

# Run after era5_download_new_job.sh (sbatch --dependency=afterok:<id>).
# Checks ERA5_DATA_new/ holds every hour 2022-07-31 00 .. 12-01 23 (2976 h) in
# both streams, then deletes the old ERA5_DATA/ and renames ERA5_DATA_new -> ERA5_DATA.
set -e
P=/expanse/lustre/projects/uso102/hchen54
module load netcdf-c 2>/dev/null || true
for kind in instant avg; do
    n=0
    for m in 202207 202208 202209 202210 202211 202212; do
        f=$P/ERA5_DATA_new/$m/data_stream-oper_stepType-$kind.nc
        [ -s "$f" ] || { echo "MISSING $f"; exit 1; }
        k=$(ncdump -h "$f" | sed -n 's/^\s*valid_time = [^0-9]*\([0-9]\+\).*/\1/p')
        echo "$m $kind: $k hours"; n=$((n + k))
    done
    [ "$n" -eq 2976 ] || { echo "$kind: $n hours, expected 2976 -> not swapping"; exit 1; }
done
rm -rf "$P/ERA5_DATA"
mv "$P/ERA5_DATA_new" "$P/ERA5_DATA"
echo "swapped:"; ls "$P/ERA5_DATA"; du -sh "$P/ERA5_DATA"
