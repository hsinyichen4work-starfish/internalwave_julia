# input_make.jl
#
# Ported from nc_make_all.m — builds the ROMS input files for the child
# grid from NCOM output: grid, initial, and one boundary + one surface
# forcing file per day. Files that already exist are left alone.
#
# Run from the Documents/Julia project with threads, e.g.
#     julia -t auto --project=/home/hsinyi/Documents/Julia input_file_make/input_make.jl

using Dates

include("/home/hsinyi/Documents/Julia/function/input_nc/load_input_nc.jl")
using .InputNC

##
mid_iter = 2; dx = 900; nx = 684; ny = 854      # 900 m
# mid_iter = 2; dx = 300; nx = 2048; ny = 2560  # 300 m
smooth_var = (rmax = 0.15, hmin = 2, offset = 2.2)
parent_datatype = "NCOM"   # "NCOM" OR "ROMS"

chd_thetas = 6; chd_thetab = 3; chd_hc = 250; chd_N = 128
chd_ang = "rad"; chdscoord = "new2008"   # child 'new' or 'old' type scoord
ndomx = 2; ndomy = 2     # -> chunking needed to avoid OOM.
# ndomx = 4; ndomy = 4   # -> chunking needed to avoid OOM.

dating = Date(2022, 8, 22):Day(1):Date(2022, 11, 30)

## path settings
parent_grid = "/home/mbui/ModelOutput/NCOM/grid/ohgrd_2.nc"
parent_data_path = "/home/mbui/ModelOutput/NCOM/data/"
par_name = "2022082400"; ini_par_path = parent_data_path * par_name
nc_path_ini_bry = "/home/hsinyi/roms_data/NCOM_DATA_NC/"
nc_path_frc = "/home/hsinyi/roms_data/NCOM_DATA_NC/"

path_figure = nothing   # no grid figures
# path_figure = "/home/hsinyi/figure/20260903_300_test/"; mkpath(path_figure)
grid_path = "/home/hsinyi/roms_data/grid/"
initial_path = "/home/hsinyi/roms_data/ini/"
mkpath(initial_path)
boundary_path = "/home/hsinyi/roms_data/bry/"
mkpath(boundary_path)
forcing_path = "/home/hsinyi/roms_data/frc/"

bath_path = "/home/hsinyi/data_notm/"

## name settings
grd_name = "roms_grd_$(dx)m"
ini_name = "roms_ini_$(dx)m$(par_name).nc"

# the settings the build steps (grd_build, ini_build, bry_build, frc_build) read
cfg = (; mid_iter, dx, nx, ny, smooth_var, chd_thetas, chd_thetab, chd_hc, chd_N, chd_ang, chdscoord,
    ndomx, ndomy, parent_grid, parent_data_path, nc_path_ini_bry, nc_path_frc, path_figure,
    grid_path, initial_path, boundary_path, forcing_path, bath_path, grd_name, ini_name)

## grid build
pgrid = read_nc_fun(parent_grid)
pgrid = standardize_name(pgrid)
if !isfile(joinpath(grid_path, grd_name * ".nc"))
    grd_build(cfg)
end
child_grid = read_nc_fun(joinpath(grid_path, grd_name * ".nc"))
rx1_max, rx1_field, loc = compute_rx1(child_grid["h"], chd_thetas, chd_thetab, chd_hc, chd_N)

## initial build
if !isfile(joinpath(initial_path, ini_name))
    ini_build(cfg, pgrid, par_name)
end
child_ini = read_nc_fun(joinpath(initial_path, ini_name))

## boundary and forcing build
fod = Dates.format.(DateTime.(dating), "yyyymmddHH")
# what is the same for every day (parent subgrids, interpolation coefficients) is kept between days
bry_state = Dict{Symbol,Any}()
frc_state = Dict{Symbol,Any}()
for folder_num in eachindex(fod)
    global par_name = fod[folder_num]
    bry_filename = "roms_bry_$(dx)m_$(par_name).nc"   # bry filename
    if !isfile(joinpath(boundary_path, bry_filename))
        bry_build(cfg, par_name, bry_filename; state = bry_state)
    end
    global child_bry = read_nc_fun(joinpath(boundary_path, bry_filename))
    frc_filename = "roms_frc_$(dx)m_$(par_name).nc"
    if !isfile(joinpath(forcing_path, frc_filename))
        frc_build(cfg, par_name, frc_filename; state = frc_state)
    end
    global child_frc = read_nc_fun(joinpath(forcing_path, frc_filename))
end
