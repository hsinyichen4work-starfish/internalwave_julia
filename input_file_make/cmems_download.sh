#!/bin/bash
# cmems_download.sh
#
# Downloads the CMEMS GLOBAL_ANALYSISFORECAST_PHY_001_024 (GLO12, 1/12 deg)
# fields needed to build the ROMS initial + boundary files for the 2700 m
# grid roms_cmems_grd_2700m.nc (lon -50.8..-36.2, lat -2.3..13.0, h <= 5940 m).
#
#   cmems_uv_6h.nc    uo, vo    6-hourly instantaneous, all depths
#   cmems_temp_6h.nc  thetao    6-hourly instantaneous, all depths
#   cmems_salt_6h.nc  so        6-hourly instantaneous, all depths
#   cmems_zos_1h.nc   zos       hourly mean SSH (no tides)
#   cmems_static.nc   deptho, mask
#
# One-time setup (credentials are stored in ~/.copernicusmarine):
#     ~/envs/cmems/bin/copernicusmarine login
# Then run (can take a while; better inside screen/tmux or a batch job):
#     bash input_file_make/cmems_download.sh

set -e
CM=~/envs/cmems/bin/copernicusmarine
OUT=/expanse/lustre/projects/uso101/hchen54/CMEMS_DATA
mkdir -p "$OUT"

# grid extent + ~0.7 deg margin; one extra day on each side of the run.
# Defaults: 2022-08-24..2022-09-30 run, no file suffix. Override, e.g.
#     T0=2022-08-21T00:00:00 T1=2022-12-01T00:00:00 SUF=_0822_1130 bash cmems_download.sh
BOX="--minimum-longitude -51.5 --maximum-longitude -35.5 --minimum-latitude -3.5 --maximum-latitude 14.0"
: "${T0:=2022-08-23T00:00:00}"; : "${T1:=2022-10-01T00:00:00}"; : "${SUF:=}"
DEP="--minimum-depth 0 --maximum-depth 6000"
COMMON="$BOX -o $OUT --skip-existing"

$CM subset -i cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i    -v uo -v vo -t $T0 -T $T1 $DEP $COMMON -f cmems_uv_6h$SUF.nc
$CM subset -i cmems_mod_glo_phy-thetao_anfc_0.083deg_PT6H-i -v thetao   -t $T0 -T $T1 $DEP $COMMON -f cmems_temp_6h$SUF.nc
$CM subset -i cmems_mod_glo_phy-so_anfc_0.083deg_PT6H-i     -v so       -t $T0 -T $T1 $DEP $COMMON -f cmems_salt_6h$SUF.nc
$CM subset -i cmems_mod_glo_phy_anfc_0.083deg_PT1H-m        -v zos      -t $T0 -T $T1 $COMMON -f cmems_zos_1h$SUF.nc
$CM subset -i cmems_mod_glo_phy_anfc_0.083deg_static        -v deptho -v mask $DEP $COMMON -f cmems_static.nc

ls -lh "$OUT"
