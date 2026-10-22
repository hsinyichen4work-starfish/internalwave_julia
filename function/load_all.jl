
##general 
include("/home/hsinyi/Documents/Julia/function/loadfun/load_interp_fun.jl")
include("/home/hsinyi/Documents/Julia/function/loadfun/load_usefultool_fun.jl")
include("/home/hsinyi/Documents/Julia/function/fig_setup.jl")   # figkw(w_cm, h_cm): size + 10 pt font on an 18 cm page; figsave

include("/home/hsinyi/Documents/Julia/function/vorticity_cal.jl")
include("/home/hsinyi/Documents/Julia/function/butter_filters.jl")   # needs `using DSP` in the calling script