# grid_figures.jl
#
# Diagnostic figures for the boundary-matched bathymetry: the plot at the
# end of match_boundary_topo.m and grid_boundary_match_figure.m.
# Included by grd_build only when a figure folder is given, so CairoMakie
# is not loaded otherwise. Figures are saved as .png (MATLAB saved
# .fig + .jpg).

using CairoMakie
include(joinpath(@__DIR__, "..", "plotting_fun.jl"))   # topdown_axis3, plot_curvilinear!

"""
    match_topo_figures(grd_nc, diag, dx, path_figure)

Save `match_topo_check_<dx>.png` (interpolated parent topo, matched child
topo, their difference, transition weight) and `match_topo_check_<dx>_2.png`
(original, after match, after match - original) in `path_figure`.
"""
function match_topo_figures(grd_nc, diag, dx, path_figure)
    mkpath(path_figure)
    lon, lat = grd_nc["lon_rho"], grd_nc["lat_rho"]
    hcn, h_orig = grd_nc["h"], grd_nc["h_orig"]

    panel!(pos, data, title; kwargs...) = begin
        ax = topdown_axis3(pos; title = title)
        hm = plot_curvilinear!(ax, lon, lat, data; kwargs...)
        return hm
    end

    # ---- match_boundary_topo diagnostic plot ----
    sc = extrema(filter(!isnan, hcn))
    fig = Figure(size = (1400, 1300))
    hm = panel!(fig[1, 1], diag.hpi, "Interpolated Parent Topo"; colorrange = sc)
    Colorbar(fig[1, 2], hm)
    hm = panel!(fig[1, 3], hcn, "Boundary-Matched Child Topo"; colorrange = sc)
    Colorbar(fig[1, 4], hm)
    hm = panel!(fig[2, 1], hcn .- diag.hpi, "Difference (Child - Parent)")
    Colorbar(fig[2, 2], hm)
    hm = panel!(fig[2, 3], diag.alpha, "Parent/Child Transition Weight (α)")
    Colorbar(fig[2, 4], hm)
    save(joinpath(path_figure, "match_topo_check_$(dx).png"), fig)

    # ---- grid_boundary_match_figure ----
    levels = vcat([0, 100], 500:500:5000)
    fig = Figure(size = (2000, 800))
    hm = panel!(fig[1, 1], h_orig, "original"; colorrange = (0, 5000), contour_levels = levels)
    Colorbar(fig[1, 2], hm)
    hm = panel!(fig[1, 3], hcn, "after match"; colorrange = (0, 5000), contour_levels = levels)
    Colorbar(fig[1, 4], hm)
    hm = panel!(fig[1, 5], hcn .- h_orig, "after match - original"; colorrange = (-10, 10), colormap = :balance)
    Colorbar(fig[1, 6], hm)
    save(joinpath(path_figure, "match_topo_check_$(dx)_2.png"), fig)
    return nothing
end
