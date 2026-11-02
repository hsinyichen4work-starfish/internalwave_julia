
##general 
# include("/home/hchen54/internalwave_julia/function/loadfun/load_interp_fun.jl")
include("/home/hchen54/internalwave_julia/function/zlevs3.jl")   # path relative to where you launch julia (your Documents/Julia folder)
include("/home/hchen54/internalwave_julia/function/depth_slice.jl")
include("/home/hchen54/internalwave_julia/function/uvp_masks.jl")
include("/home/hchen54/internalwave_julia/function/rho2uvp.jl")
include("/home/hchen54/internalwave_julia/function/uv2rho.jl")
isdefined(@__MODULE__, :Interp1D) || include("/home/hchen54/internalwave_julia/function/interp_1d.jl")   # already loaded by depth_slice.jl — re-including replaces the module and breaks `interp_1d` on a second include of this file
include("/home/hchen54/internalwave_julia/function/interp_2d.jl")
include("/home/hchen54/internalwave_julia/function/interp_3d.jl")
include("/home/hchen54/internalwave_julia/function/interp_bry.jl")

include("/home/hchen54/internalwave_julia/function/loadfun/load_usefultool_fun.jl")   # figpath, max_min
include("/home/hchen54/internalwave_julia/function/fig_setup.jl")   # figkw(w_cm, h_cm): size + 10 pt font on an 18 cm page; figsave
include("/home/hchen54/internalwave_julia/function/time_dir.jl")     # time_dir(data_folder): which file/record holds each time step; records_by_file
include("/home/hchen54/internalwave_julia/function/data_load.jl")    # data_load(data_folder, var, td; ix, jy, iz, grid_file); region_index
include("/home/hchen54/internalwave_julia/function/harm_fit.jl")     # harm_fit(Y, tsec, cons; scale): plain sinusoid tidal fit; harm_predict; TIDE_FREQ

include("/home/hchen54/internalwave_julia/function/vorticity_cal.jl")
