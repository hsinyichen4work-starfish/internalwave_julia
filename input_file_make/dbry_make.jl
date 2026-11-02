# dbry_make.jl
#
# Ported from dbry_make.m — computes the high-frequency baroclinic energy
# flux Fx/Fy at the child-grid open boundaries from the daily
# roms_bry_<dx>m_<fod>.nc files, and writes one daily flux file
# (roms_dbry_flux_<dx>m_<fod>.nc, rho0-normalized, what ROMS reads) plus
# one daily MAT file (bryfile_dynamic<dx>_<yyyymmdd>.mat, raw W/m, for
# diagnostics/plotting) per day. Existing files are overwritten.
#
# Run on Expanse as a batch job (sbatch_files/dbry_make_job.sh), i.e.
#     julia -t 16 --project=. input_file_make/dbry_make.jl
#
# Set up for the 2700 m CMEMS run; the 900 m NCOM settings are kept as comments.

using Dates

include("/home/hchen54/internalwave_julia/function/dbry_fun.jl")

##
# 900 m NCOM (hsinyi machine): dx = 900, grid /home/hsinyi/roms_data/grid/roms_grd_900m.nc,
# bry /home/hsinyi/roms_data/bry/, flux /home/hsinyi/roms_data/bry_dynamic_flux/,
# mat /home/hsinyi/matlab_file/dbry_saving/, dating 2022-08-22 .. 11-30
proj = "/expanse/lustre/projects/uso102/hchen54/input_cmems/"
dx = 2700
child_grid_file = proj * "grid/roms_cmems_grd_$(dx)m.nc"
child_bry_path = proj * "bry/"
flux_out_path = proj * "bry/"   # daily nc files (rho0-normalized) that feed ROMS, next to the bry files (make_run_folder_cmems.jl looks there)
daily_mat_path = proj * "dbry_mat/"   # daily MAT files (NOT rho0-normalized), read by dbry_check_plotting.jl

# CMEMS bry files are 6-hourly (5 records/day), so the highpass keeps 12..28 h
dating = Date(2022, 8, 24):Day(1):Date(2022, 11, 30)

cutoff_days = 28 / 24   # highpass cutoff period (~28 hours)
N_butter = 5

rho0 = 1027.5   # ROMS reference density (kg/m^3); only the nc files are divided by it

chunk_days = 15   # days of *output* produced per chunk
pad_days = 6      # extra days read (and discarded) on each side of a chunk

t1 = DateTime(1994, 1, 1)   # model bry_time epoch, only used to print real dates

save_mat = true   # also write the daily MAT files

cfg = (; dx, child_grid_file, child_bry_path, flux_out_path, daily_mat_path, dating,
    cutoff_days, N_butter, rho0, chunk_days, pad_days, save_mat, t1)

##
dbry_make(cfg)
