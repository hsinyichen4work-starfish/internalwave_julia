# grid_make_1104.jl
#
# Builds the four nested ROMS grids of the 2026-11-04 grid design
# (/home/hsinyi/data_notm/grid_design.txt, regions drawn on regiontest.png by
# other/figure_of_region_new.jl). Each grid's boundary bathymetry is blended
# with its parent: 2700 m <- NCOM ohgrd_1, 900 m <- 2700 m, 300 m <- 900 m,
# 100 m <- 300 m. Only the grids in build_dx are built (an existing file is
# left alone unless remake). A built grid is also copied to
# <name>_before_riverfix.nc if it is in riverfix_dx; then run river_input.jl
# (2700 m: river mask + river_flux) and child_mask_match.jl (900 m: follows the
# 2700 m river mask), which rebuild <name>.nc from that copy.
#
# Run from the Documents/Julia project, e.g.
#     julia -t auto --project=/home/hsinyi/Documents/Julia input_file_make/grid_make_1104.jl

using Dates

include("/home/hsinyi/Documents/Julia/function/input_nc/load_input_nc.jl")
using .InputNC

##
mid_iter = 2   # rotation (-34.2112 deg) from the mooring line; every grid shares it
# hmin = minimum depth (m) of the written h, also on land; offset = 0 so the
# rx0 < rmax test is on the real h. (Was hmin = 2, offset = 2.2: that gave
# h down to -0.2 m, wet cells of 0.16 m and a real rx0 of 0.67 -> ROMS blew up
# on the first step where h + zeta < 0.) This ROMS has no wet/dry, so hmin
# must stay wet through the lowest tide (Amazon shelf: several metres).
smooth_var = (rmax = 0.15, hmin = 10, offset = 0)
chd_thetas = 6; chd_thetab = 3; chd_hc = 250; chd_N = 128

## path settings
ncom_grid = "/home/mbui/ModelOutput/NCOM/grid/ohgrd_1.nc"
grid_path = "/home/hsinyi/roms_data/new_design/"; mkpath(grid_path)
build_dx = [900, 300, 100]   # grids to (re)build; others are only checked
riverfix_dx = [2700, 900]    # grids whose mask is edited afterwards (2700: river_input.jl, 900: child_mask_match.jl)
remake = true       # true: rebuild those grids even if the file exists (back it up first!)
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
    gfile = joinpath(grid_path, grd_name * ".nc")
    if g.dx in build_dx && (remake || !isfile(gfile))
        grd_build(cfg)
        # river_input.jl / child_mask_match.jl start from this untouched copy and rebuild gfile
        g.dx in riverfix_dx && cp(gfile, joinpath(grid_path, grd_name * "_before_riverfix.nc"); force = true)
    end
    isfile(gfile) || continue
    child_grid = read_nc_fun(joinpath(grid_path, grd_name * ".nc"))
    rx1_max, rx1_field, loc = compute_rx1(child_grid["h"], chd_thetas, chd_thetab, chd_hc, chd_N)
    println(grd_name, ": size ", size(child_grid["h"]), ", rx1_max = ", rx1_max)
end
