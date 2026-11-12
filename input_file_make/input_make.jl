# input_make.jl
#
# Ported from nc_make_all.m — builds the ROMS input files for the child
# grid: grid, initial, and one boundary + one surface forcing file per
# day. Files that already exist are left alone.
#
# parent_datatype picks the ocean parent for the initial and boundary files:
#   "NCOM"  : NCOM flat files, daily 00Z cycles (hourly, 0..24 h)
#   "HYCOM" : HYCOM GOFS 3.5 glbz04 z-level NetCDF, daily 12Z cycles
#             (3-hourly, 0..21 or 0..24 h). One boundary file per HYCOM
#             cycle, named with "hycom" so the NCOM files are kept.
# The surface forcing always comes from NCOM (HYCOM has no atmosphere).
#
# Run from the Documents/Julia project with threads, e.g.
#     julia -t auto --project=/home/hsinyi/Documents/Julia input_file_make/input_make.jl

using Dates

include("/home/hsinyi/Documents/Julia/function/input_nc/load_input_nc.jl")
using .InputNC

##
# child grid: an existing, already smoothed grid file (no grid build here)
grid_path = "/home/hsinyi/roms_data/new_design/"
grd_name = "roms_grd_2700m_1104"          # file grid_path/grd_name.nc
run_tag = "2700m_1104"                    # used in the ini/bry/frc file names
parent_datatype = "HYCOM"   # "NCOM" OR "HYCOM" (ocean parent for ini + bry)
is_hycom = uppercase(parent_datatype) == "HYCOM"
src_tag = is_hycom ? "_hycom" : "_ncom"   # ocean parent, in the ini/bry file names

chd_thetas = 6; chd_thetab = 3; chd_hc = 250; chd_N = 128
chd_ang = "rad"; chdscoord = "new2008"   # child 'new' or 'old' type scoord
ndomx = 2; ndomy = 2     # -> chunking needed to avoid OOM.
# ndomx = 4; ndomy = 4   # -> chunking needed to avoid OOM.

dating = Date(2022, 8, 22):Day(1):Date(2022, 11, 30)

## path settings
parent_data_path = "/home/mbui/ModelOutput/NCOM/data/"
hycom_data_path = "/home/mbui/ModelOutput/NCOM/HYCOM_data/subset/"
par_name = "2022080200"; ini_par_path = parent_data_path * par_name
# intermediate parent NetCDF files. NCOM (also used by the forcing step):
nc_path_ncom = "/home/hsinyi/roms_data/NCOM_DATA_NC_2700/"
nc_path_frc = nc_path_ncom
# HYCOM files are cut to the child grid, so they get their own folder per grid
nc_path_hycom = "/home/hsinyi/roms_data/HYCOM_DATA_NC_$(run_tag)/"
nc_path_ini_bry = is_hycom ? nc_path_hycom : nc_path_ncom
mkpath(nc_path_ini_bry)

initial_path = joinpath(grid_path, "ini")
mkpath(initial_path)
boundary_path = joinpath(grid_path, "bry")
mkpath(boundary_path)
forcing_path = joinpath(grid_path, "frc")
mkpath(forcing_path)

## name settings
# initial time = par_name (yyyymmddHH), a valid time. HYCOM: the cycle
# holding it is found from the `time` variable of the files (most recent
# analysis if several hold it); ini_build then picks the record whose
# valid time equals ini_time.
ini_time = DateTime(par_name, "yyyymmddHH")
ini_par_name = is_hycom ? hycom_find_time(ini_time; hycom_path = hycom_data_path).cycle : par_name
ini_name = "roms_ini_$(run_tag)$(src_tag)_$(par_name).nc"

# the settings the build steps (ini_build, bry_build, frc_build) read
cfg = (; chd_thetas, chd_thetab, chd_hc, chd_N, chd_ang, chdscoord, ndomx, ndomy,
    parent_datatype, parent_data_path, hycom_data_path, nc_path_ini_bry, nc_path_frc,
    grid_path, initial_path, boundary_path, forcing_path, grd_name, ini_name)

## child grid (already built and smoothed)
chdgrd_file = joinpath(grid_path, grd_name * ".nc")
isfile(chdgrd_file) || error("child grid not found: $chdgrd_file")
child_grid = read_nc_fun(chdgrd_file)
rx1_max, rx1_field, loc = compute_rx1(child_grid["h"], chd_thetas, chd_thetab, chd_hc, chd_N)

## initial build
if !isfile(joinpath(initial_path, ini_name))
    ini_build(cfg, nothing, ini_par_name; ini_time = ini_time)
end
child_ini = read_nc_fun(joinpath(initial_path, ini_name))

## boundary and forcing build
fod = Dates.format.(DateTime.(dating), "yyyymmddHH")   # NCOM 00Z cycles (forcing, NCOM bry)
# HYCOM 12Z cycles: start the day before so the first one covers dating[1] 00Z
hycom_cycles = Dates.format.(DateTime.(first(dating) - Day(1):Day(1):last(dating)) .+ Hour(12), "yyyymmddHH")
# what is the same for every day (parent subgrids, interpolation coefficients) is kept between days
bry_state = Dict{Symbol,Any}()
frc_state = Dict{Symbol,Any}()

if is_hycom
    for hpar in hycom_cycles
        bry_filename = "roms_bry_$(run_tag)$(src_tag)_$(hpar).nc"
        if !isfile(joinpath(boundary_path, bry_filename))
            bry_build(cfg, hpar, bry_filename; state = bry_state)
        end
        global child_bry = read_nc_fun(joinpath(boundary_path, bry_filename))
    end
end

# NCOM forcing (and the NCOM boundary files when parent_datatype is NCOM)
cfg_ncom = merge(cfg, (; parent_datatype = "NCOM", nc_path_ini_bry = nc_path_ncom))
for folder_num in eachindex(fod)
    global par_name = fod[folder_num]
    if !is_hycom
        bry_filename = "roms_bry_$(run_tag)$(src_tag)_$(par_name).nc"   # bry filename
        if !isfile(joinpath(boundary_path, bry_filename))
            bry_build(cfg_ncom, par_name, bry_filename; state = bry_state)
        end
        global child_bry = read_nc_fun(joinpath(boundary_path, bry_filename))
    end
    frc_filename = "roms_frc_$(run_tag)_ncom_$(par_name).nc"
    if !isfile(joinpath(forcing_path, frc_filename))
        frc_build(cfg_ncom, par_name, frc_filename; state = frc_state)
    end
    global child_frc = read_nc_fun(joinpath(forcing_path, frc_filename))
end
