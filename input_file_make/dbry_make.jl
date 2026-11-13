# dbry_make.jl
#
# Ported from dbry_make.m — computes the high-frequency baroclinic energy
# flux Fx/Fy at the child-grid open boundaries from the ROMS bry files and
# writes, for each bry file, one flux file (roms_dbry_flux_..., rho0-
# normalized, what ROMS reads) holding the same bry_time records, plus one
# MAT file (raw W/m, for diagnostics/plotting). Existing files are overwritten.
#
# Set up for the new_design 2700 m grid with HYCOM boundaries
# (input_make.jl): one bry file per HYCOM 12Z cycle, 3-hourly records.
#
#     julia -t auto --project=/home/hsinyi/Documents/Julia /home/hsinyi/Documents/Julia/input_file_make/dbry_make.jl
#
# Earlier setups (daily 00Z bry files, run with dbry_make(cfg)):
#   2700 m CMEMS (Expanse): proj = /expanse/lustre/projects/uso102/hchen54/input_cmems/,
#     grid proj/grid/roms_cmems_grd_2700m.nc, bry + flux proj/bry/, mat proj/dbry_mat/,
#     dating 2022-08-24 .. 11-30, chunk_days = 15, pad_days = 6
#   900 m NCOM: grid /home/hsinyi/roms_data/grid/roms_grd_900m.nc, bry /home/hsinyi/roms_data/bry/,
#     flux /home/hsinyi/roms_data/bry_dynamic_flux/, mat /home/hsinyi/matlab_file/dbry_saving/

using Dates

include("/home/hsinyi/Documents/Julia/function/dbry_fun.jl")

##
dx = 2700
run_tag = "2700m_1104_hycom"
proj = "/home/hsinyi/roms_data/new_design/"
child_grid_file = proj * "roms_grd_2700m_1104.nc"
child_bry_path = proj * "bry/"
flux_out_path = proj * "bry/"       # flux nc files (rho0-normalized) that feed ROMS, next to the bry files
daily_mat_path = proj * "dbry_mat/" # MAT files (NOT rho0-normalized), for dbry_check_plotting.jl

# bry files: roms_bry_<run_tag>_<cycle>.nc, sorted by cycle
bry_files = sort(filter(f -> occursin(Regex("^roms_bry_$(run_tag)_\\d{10}\\.nc\$"), f), readdir(child_bry_path)))
bry_files = joinpath.(child_bry_path, bry_files)
_cycle(f) = match(r"_(\d{10})\.nc$", basename(f))[1]
flux_name(f) = "roms_dbry_flux_$(run_tag)_$(_cycle(f)).nc"
mat_name(f) = "bryfile_dynamic$(dx)_hycom_$(_cycle(f)).mat"

cutoff_days = 28 / 24   # highpass cutoff period (~28 hours)
N_butter = 5

rho0 = 1027.5   # ROMS reference density (kg/m^3); only the nc files are divided by it

chunk_files = 15  # bry files (= HYCOM cycles, ~1 day each) of *output* per chunk
pad_files = 6     # extra files read (and discarded) on each side of a chunk

t1 = DateTime(1994, 1, 1)   # model bry_time epoch, only used to print real dates

save_mat = true   # also write the MAT files

cfg = (; dx, child_grid_file, bry_files, flux_out_path, daily_mat_path,
    cutoff_days, N_butter, rho0, chunk_files, pad_files, save_mat, t1, flux_name, mat_name)

##
dbry_make_files(cfg)
