
include(joinpath(@__DIR__, "..", "max_min.jl"))   # relative to this file, so it works on any machine's checkout

# Builds figure_path/subfolder/filename, creating subfolder if needed.
# Expects a global `figure_path` to already be set by the calling script
# (e.g. figure_path = "/home/hchen54/figure/dbry900m/bry_match").
# e.g. figpath("zeta_east_bry", "zeta_east_bry_20220901.png")
#   -> figure_path/zeta_east_bry/zeta_east_bry_20220901.png
function figpath(subfolder, filename)
    dir = joinpath(figure_path, subfolder)
    mkpath(dir)
    return joinpath(dir, filename)
end
