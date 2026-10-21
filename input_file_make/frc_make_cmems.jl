# frc_make_cmems.jl
#
# Surface forcing part of input_make.jl for the 2700 m cmems grid
# (grid_make_cmems.jl): one forcing file per day from the nest-1 (2700 m)
# NCOM NetCDF files made by ncom2nc_make.jl. Files that already exist are
# left alone.
#
# Run from the Documents/Julia project with threads, e.g.
#     julia -t auto --project=/home/hsinyi/Documents/Julia input_file_make/frc_make_cmems.jl

using Dates

include("/home/hsinyi/Documents/Julia/function/input_nc/load_input_nc.jl")
using .InputNC

##
dx = 2700
chd_ang = "rad"
ndomx = 1; ndomy = 1     # the 2700 m grid is small, no chunking needed

dating = Date(2022, 8, 22):Day(1):Date(2022, 11, 30)

## path settings
parent_data_path = "/home/mbui/ModelOutput/NCOM/data/"
nc_path_ini_bry = "/home/hsinyi/roms_data/NCOM_DATA_NC_2700/"
nc_path_frc = "/home/hsinyi/roms_data/NCOM_DATA_NC_2700/"

grid_path = "/home/hsinyi/roms_data/grid/"
forcing_path = "/home/hsinyi/roms_data/roms_cmems/"
mkpath(forcing_path)

## name settings
grd_name = "roms_cmems_grd_$(dx)m"

# the settings frc_build reads
cfg = (; chd_ang, ndomx, ndomy, parent_data_path, nc_path_ini_bry, nc_path_frc, grid_path, forcing_path, grd_name)

## forcing build
fod = Dates.format.(DateTime.(dating), "yyyymmddHH")
# the parent subgrid limits are the same for every day, so they are kept between days
frc_state = Dict{Symbol,Any}()
for par_name in fod
    # make_frc_need_nc would fall back to the 900 m (nest 2) NCOM data if a file were missing
    isfile(joinpath(nc_path_frc, par_name * "_flx.nc")) || error("no 2700 m NCOM forcing data for $par_name")
    frc_filename = "roms_cmems_frc_$(dx)m_$(par_name).nc"
    if !isfile(joinpath(forcing_path, frc_filename))
        frc_build(cfg, par_name, frc_filename; state = frc_state)
    end
end
