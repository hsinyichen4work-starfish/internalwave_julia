# grid_make_cmems.jl
#
# Grid-only part of input_make.jl: builds the 2700 m ROMS child grid,
# blending the boundary bathymetry with the ohgrd_1 parent grid.
# An existing grid file is left alone.
#
# Run from the Documents/Julia project, e.g.
#     julia -t auto --project=/home/hsinyi/Documents/Julia input_file_make/grid_make_cmems.jl

using Dates

include("/home/hsinyi/Documents/Julia/function/input_nc/load_input_nc.jl")
using .InputNC

##
mid_iter = 2; dx = 2700; nx = 376; ny = 507      # 2700 m
# middle of the 900m ROMS grid extended +222 cells W/E, +666 N (figure_of_region_new.jl)
mid = [-43.542425679936414, 5.380053609976857]
smooth_var = (rmax = 0.15, hmin = 2, offset = 2.2)
parent_datatype = "NCOM"   # "NCOM" OR "ROMS"

chd_thetas = 6; chd_thetab = 3; chd_hc = 250; chd_N = 128
chd_ang = "rad"; chdscoord = "new2008"   # child 'new' or 'old' type scoord

## path settings
parent_grid = "/home/mbui/ModelOutput/NCOM/grid/ohgrd_1.nc"

path_figure = "/home/hsinyi/figure/20261020/cemes_run/"; mkpath(path_figure)
grid_path = "/home/hsinyi/roms_data/grid/"
bath_path = "/home/hsinyi/data_notm/"

## name settings
grd_name = "roms_cmems_grd_$(dx)m"

# the settings grd_build reads
cfg = (; mid_iter, mid, dx, nx, ny, smooth_var, parent_grid, path_figure, grid_path, bath_path, grd_name)

## grid build
if !isfile(joinpath(grid_path, grd_name * ".nc"))
    grd_build(cfg)
end
child_grid = read_nc_fun(joinpath(grid_path, grd_name * ".nc"))
rx1_max, rx1_field, loc = compute_rx1(child_grid["h"], chd_thetas, chd_thetab, chd_hc, chd_N)
println("rx1_max = ", rx1_max)
