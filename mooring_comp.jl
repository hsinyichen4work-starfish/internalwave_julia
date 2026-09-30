using NCDatasets, CairoMakie, Dates, Statistics
CairoMakie.activate!()
include("/home/hchen54/internalwave_julia/function/load_all_hpc.jl")
include("/home/hchen54/internalwave_julia/function/plotting_fun.jl")

grid_fname = "/expanse/lustre/projects/uso101/hchen54/input/grid/roms_grd_900m.nc"   # the grid_file listed in the .nc's global attributes
datadir = "/expanse/lustre/projects/uso101/hchen54/amazon_900m_dbry_0927"   # HPC output dir — contains avg/dia/his/rst files mixed together
figure_path = "/home/hchen54/figure/dbry_0927/mooring"
joint_ext_path = "/expanse/lustre/projects/uso101/hchen54/amazon_900m_dbry_0927/mooring_merged"


##
fname = joinpath(joint_ext_path, "french.nc")
NCDataset(fname) do ds
    show(ds)
end

