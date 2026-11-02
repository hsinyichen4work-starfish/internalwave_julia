
##general 
include("/home/hsinyi/Documents/Julia/function/loadfun/load_interp_fun.jl")
include("/home/hsinyi/Documents/Julia/function/loadfun/load_usefultool_fun.jl")
include("/home/hsinyi/Documents/Julia/function/fig_setup.jl")   # figkw(w_cm, h_cm): size + 10 pt font on an 18 cm page; figsave
include("/home/hsinyi/Documents/Julia/function/time_dir.jl")     # time_dir(data_folder): which file/record holds each time step; records_by_file
include("/home/hsinyi/Documents/Julia/function/data_load.jl")    # data_load(data_folder, var, td; ix, jy, iz, grid_file); region_index
include("/home/hsinyi/Documents/Julia/function/harm_fit.jl")     # harm_fit(Y, tsec, cons; scale): plain sinusoid tidal fit; harm_predict; TIDE_FREQ

include("/home/hsinyi/Documents/Julia/function/vorticity_cal.jl")
include("/home/hsinyi/Documents/Julia/function/butter_filters.jl")   # needs `using DSP` in the calling script