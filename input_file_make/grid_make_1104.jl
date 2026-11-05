# grid_make_1104.jl
#
# Builds the four nested ROMS grids of the 2026-11-04 grid design
# (/home/hsinyi/data_notm/grid_design.txt, regions drawn on regiontest.png by
# other/figure_of_region_new.jl). Each grid's boundary bathymetry is blended
# with its parent: 2700 m <- NCOM ohgrd_1, 900 m <- 2700 m, 300 m <- 900 m,
# 100 m <- 300 m. An existing grid file is left alone.
#
# Run from the Documents/Julia project, e.g.
#     julia -t auto --project=/home/hsinyi/Documents/Julia input_file_make/grid_make_1104.jl

using Dates

include("/home/hsinyi/Documents/Julia/function/input_nc/load_input_nc.jl")
using .InputNC

##
mid_iter = 2   # rotation (-34.2112 deg) from the mooring line; every grid shares it
smooth_var = (rmax = 0.15, hmin = 2, offset = 2.2)
chd_thetas = 6; chd_thetab = 3; chd_hc = 250; chd_N = 128

## path settings
ncom_grid = "/home/mbui/ModelOutput/NCOM/grid/ohgrd_1.nc"
grid_path = "/home/hsinyi/roms_data/grid/new_design/"; mkpath(grid_path)
bath_path = "/home/hsinyi/data_notm/"
path_figure = nothing   # no grid figures

## the four grids, coarse to fine (mid, nx, ny from grid_design.txt)
grids = [
    (dx = 2700, nx = 524,  ny = 655,  mid = [-44.560093, 3.892079], parent = ncom_grid),
    (dx = 900,  nx = 796,  ny = 966,  mid = [-45.064075, 3.152625], parent = joinpath(grid_path, "roms_grd_2700m_1104.nc")),
    (dx = 300,  nx = 1584, ny = 2475, mid = [-45.059439, 3.151076], parent = joinpath(grid_path, "roms_grd_900m_1104.nc")),
    (dx = 100,  nx = 2304, ny = 6144, mid = [-45.064075, 3.152625], parent = joinpath(grid_path, "roms_grd_300m_1104.nc")),
]

## grid build
for g in grids
    grd_name = "roms_grd_$(g.dx)m_1104"
    println("\n===== ", grd_name, " (parent: ", basename(g.parent), ") =====")
    cfg = (; mid_iter, mid = g.mid, dx = g.dx, nx = g.nx, ny = g.ny, smooth_var,
        parent_grid = g.parent, path_figure, grid_path, bath_path, grd_name)
    if !isfile(joinpath(grid_path, grd_name * ".nc"))
        grd_build(cfg)
    end
    child_grid = read_nc_fun(joinpath(grid_path, grd_name * ".nc"))
    rx1_max, rx1_field, loc = compute_rx1(child_grid["h"], chd_thetas, chd_thetab, chd_hc, chd_N)
    println(grd_name, ": size ", size(child_grid["h"]), ", rx1_max = ", rx1_max)
end
