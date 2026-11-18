#!/bin/bash
#SBATCH --job-name=cmems_cleanup
#SBATCH --account=uso102
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/expanse/lustre/projects/uso102/hchen54/cmems_cleanup_%j.log

# Run after cmems_download_0731_1130_big_job.sh (sbatch --dependency=afterok:<id>).
# Checks the new _0731_1130_big files, then deletes the older, smaller-box downloads.
set -e
module load netcdf-c 2>/dev/null || true
D=/expanse/lustre/projects/uso102/hchen54/CMEMS_DATA
S=_0731_1130_big

# new files must open and hold the full 07-31 00:00 .. 12-01 00:00 record
check () {   # file  min_time_records
    local f=$D/$1 n
    [ -s "$f" ] || { echo "MISSING $f"; exit 1; }
    ncdump -h "$f" > /dev/null || { echo "UNREADABLE $f"; exit 1; }
    if [ -n "$2" ]; then
        n=$(ncdump -h "$f" | sed -n 's/^\s*time = [^0-9]*\([0-9]\+\).*/\1/p')
        [ "${n:-0}" -ge "$2" ] || { echo "SHORT $f: $n time records < $2"; exit 1; }
        echo "ok $1  ($n time records)"
    else
        echo "ok $1"
    fi
}
check cmems_uv_6h$S.nc   493    # 6-hourly, 123 days (07-31 00 .. 12-01 00) + 1
check cmems_temp_6h$S.nc 493
check cmems_salt_6h$S.nc 493
check cmems_zos_1h$S.nc  2953   # hourly, 123*24 + 1
check cmems_static$S.nc

old=(cmems_uv_6h.nc cmems_temp_6h.nc cmems_salt_6h.nc cmems_zos_1h.nc cmems_static.nc
     cmems_uv_6h_0731_0824.nc cmems_temp_6h_0731_0824.nc cmems_salt_6h_0731_0824.nc cmems_zos_1h_0731_0824.nc
     cmems_uv_6h_0822_1130.nc cmems_temp_6h_0822_1130.nc cmems_salt_6h_0822_1130.nc cmems_zos_1h_0822_1130.nc)
for f in "${old[@]}"; do
    [ -e "$D/$f" ] && rm -v "$D/$f"
done
echo "--- remaining:"; ls -lh $D; du -sh $D
