# dbry_make.jl
#
# Ported from dbry_make.m — computes the high-frequency baroclinic energy
# flux Fx/Fy at the child-grid open boundaries from the daily
# roms_bry_<dx>m_<fod>.nc files, and writes one daily flux file
# (roms_dbry_flux_<dx>m_<fod>.nc, rho0-normalized, what ROMS reads) plus
# one daily MAT file (bryfile_dynamic<dx>_<yyyymmdd>.mat, raw W/m, for
# diagnostics/plotting) per day. Existing files are overwritten.
#
# Run from the Documents/Julia project with threads, e.g.
#     julia -t auto --project=/home/hsinyi/Documents/Julia input_file_make/dbry_make.jl

using Dates

include("/home/hsinyi/Documents/Julia/function/dbry_fun.jl")

##
dx = 900
child_grid_file = "/home/hsinyi/roms_data/grid/roms_grd_$(dx)m.nc"
child_bry_path = "/home/hsinyi/roms_data/bry/"
flux_out_path = "/home/hsinyi/roms_data/bry_dynamic_flux/"   # daily nc files (rho0-normalized) that feed ROMS
daily_mat_path = "/home/hsinyi/matlab_file/dbry_saving/"     # daily MAT files (NOT rho0-normalized)

dating = Date(2022, 8, 22):Day(1):Date(2022, 11, 30)

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
