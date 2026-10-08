using NCDatasets, CairoMakie, Dates, Statistics
CairoMakie.activate!()
include("/home/hchen54/internalwave_julia/function/load_all_hpc.jl")
include("/home/hchen54/internalwave_julia/function/plotting_fun.jl")

grid_fname = "/expanse/lustre/projects/uso101/hchen54/input/grid/roms_grd_900m.nc"   # the grid_file listed in the .nc's global attributes
datadir = "/expanse/lustre/projects/uso101/hchen54/amazon_900m_dbry_0927"   # HPC output dir — contains avg/dia/his/rst files mixed together
figure_path = "/home/hchen54/figure/dbry_0927/energy_flux"   # separate from his_rejoint so the two scripts' skip checks don't interfere
mkpath(figure_path)

##
fname = joinpath(datadir, "roms_dia.20220829010000.nc")
ocean_time, up,vp = NCDataset(fname) do ds
    show(ds)
    ds["ocean_time"][:], ds["up"][:,:,:], ds["vp"][:,:,:]
end

mask_rho, lon_rho, lat_rho, h ,pm, pn, grid_angle = NCDataset(grid_fname) do ds
    ds["mask_rho"][:, :], ds["lon_rho"][:, :], ds["lat_rho"][:, :],
    ds["h"][:,:], ds["pm"][:,:] , ds["pn"][:,:], ds["angle"][:,:]
end

fname = joinpath(datadir, "roms_his.20220829000000.nc")
his_time,zeta,u,v,rho = NCDataset(fname) do ds
    show(ds)
    ds["ocean_time"][:], ds["zeta"][:,:,:], ds["u"][:,:,:,:], ds["v"][:,:,:,:], ds["rho"][:,:,:,:]
end