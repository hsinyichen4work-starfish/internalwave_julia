# load_input_nc.jl
#
# Loads everything needed to build ROMS input files (grid, initial,
# boundary, surface forcing) from NCOM output — the Julia port of the
# MATLAB functions behind nc_make_all.m. Use it with
#
#     include("/home/hsinyi/Documents/Julia/function/input_nc/load_input_nc.jl")
#     using .InputNC
#
# Requires NCDatasets, NearestNeighbors and MAT (all in Project.toml).
# The interpolation loops are threaded: start Julia with `-t auto`.

module InputNC

using NCDatasets, NearestNeighbors, MAT
using SparseArrays, LinearAlgebra, Statistics, Dates, Printf

include(joinpath(@__DIR__, "..", "zlevs3.jl"))   # zlevs3, CSF

include("nc_tools.jl")        # ncread, ncwrite, ncsize, read_nc_fun, standardize_name, datenum, num2str
include("compute_rx1.jl")     # compute_rx1
include("tri_coef.jl")        # gnomonic, tri_locate, fix_outside_child, get_tri_coef, apply_tri_coef, fillmask
include("hv_coef.jl")         # get_hv_coef, apply_hv_coef
include("inpaint_nans.jl")    # inpaint_nans, fillmissing_linear!
include("ncom_flatfile.jl")   # read_ncom_flatfile, extract_ncom_name, read_ohgrd, read_ovgrdA, avg_face_to_center, vel_rot
include("ncom2nc.jl")         # make_bry_need_nc, make_frc_need_nc and the per-file converters
include("easy_grid.jl")       # easy_grid, rot_sphere, tra_sphere, gc_dist, make_grid
include("grid_tools.jl")      # gid_middle, grid_setting, bathy_interp, lsmooth_fun, make_roms_ncgrid, match_boundary_topo
include("h2r_ini.jl")         # h2r_create_ini, h2r_make_ini, ncom_zgrid
include("h2r_bry.jl")         # h2r_create_bry, h2r_bry_subgrid, h2r_bry_hv
include("h2r_frc.jl")         # h2r_create_frc, h2r_frc_subgrid, h2r_make_frc
include("build_steps.jl")     # grd_build, ini_build, bry_build, frc_build

export grd_build, ini_build, bry_build, frc_build
export read_nc_fun, standardize_name, ncread, ncwrite, ncsize, datenum, num2str
export compute_rx1, zlevs3
export gnomonic, tri_locate, fix_outside_child, get_tri_coef, apply_tri_coef, fillmask
export get_hv_coef, apply_hv_coef, inpaint_nans, fillmissing_linear!
export extract_ncom_name, read_ncom_flatfile, ncom_file_list, read_ohgrd, read_ovgrdA, avg_face_to_center, vel_rot
export make_bry_need_nc, make_bry_need_grid, make_bry_need_ssh, make_bry_need_temp, make_bry_need_vel
export make_frc_need_nc, make_frc_need_wind, make_frc_need_flux, make_frc_need_pres
export easy_grid, rot_sphere, tra_sphere, gc_dist, make_grid
export gid_middle, midpoints, lonlat2xy, lonlat_rad2deg, grid_setting, read_topo_subset, bathy_interp
export rfact, lsmooth_fun, make_roms_ncgrid, match_boundary_topo
export ncom_zgrid, h2r_create_ini, h2r_make_ini
export h2r_create_bry, h2r_bry_subgrid, h2r_bry_hv
export h2r_create_frc, h2r_frc_subgrid, h2r_make_frc

end # module
