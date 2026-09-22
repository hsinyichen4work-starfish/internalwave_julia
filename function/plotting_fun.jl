using CairoMakie, Dates

# Your ROMS grid is curvilinear (lon_rho/lat_rho are full 2D fields, not
# separable 1D axes) — that's exactly why the old code used PyPlot's
# pcolormesh(lon, lat, data) instead of imshow/pcolor. Makie's own
# `heatmap` cannot take full 2D coordinate arrays (only 1D edges/centers,
# or a rectilinear x,y grid), so there is no direct pcolormesh drop-in.
#
# The documented workaround (see "2D Surface" in the Makie `surface` docs)
# is to (ab)use a flat `surface` plot: pass lon/lat as the x,y arguments,
# an all-zero array as z (so the surface is perfectly flat), and the real
# field through `color`. Viewed from directly overhead with perspective
# turned off, this reproduces what pcolormesh draws — including the true
# rotated quadrilateral cells — without needing any extra packages.
function topdown_axis3(pos; title = "", xlabel = "Longitude", ylabel = "Latitude", aspect = :data,
                        ylabeloffset = 40)
    return Axis3(pos;
        title = title, xlabel = xlabel, ylabel = ylabel,
        aspect = aspect,                 # :data is right when x/y share real units (e.g. lon/lat);
                                          # override for mismatched-unit axes (e.g. time vs depth)
        ylabeloffset = ylabeloffset,     # 40 is Makie's own default; bump it when tick labels are wide
                                          # enough to collide with the axis label (e.g. "-150", "-200")
        elevation = pi / 2,             # look straight down the z-axis
        azimuth = -pi / 2,
        perspectiveness = 0,            # orthographic, i.e. no perspective skew
        zticksvisible = false, zticklabelsvisible = false,
        zlabelvisible = false, zgridvisible = false, zspinesvisible = false)
end

function plot_curvilinear!(ax, x, y, data; contour_levels = nothing, contour_color = :black,
                            contour_linewidth = 1, kwargs...)
    hm = surface!(ax, x, y, zeros(size(data)); color = data, shading = NoShading, kwargs...)
    # contour! (unlike heatmap!) accepts full 2D x/y, so it can trace the
    # same curvilinear mesh directly — no need for the flat-surface trick
    if contour_levels !== nothing
        contour!(ax, x, y, data; levels = contour_levels, color = contour_color, linewidth = contour_linewidth)
    end
    return hm
end

# One panel of a curvilinear 2D field where the y-coordinate itself varies
# in 2D — e.g. ROMS z-levels, which shift with the tide via zeta, not a
# plain rectilinear heatmap. Combines topdown_axis3 + plot_curvilinear! +
# optional ylims! into a single call for the common case of a grid of such
# panels sharing one Colorbar (side views, time-vs-depth comparisons, etc.).
# Returns (ax, hm) — hm so the caller can wire up a shared Colorbar.
function curvilinear_panel!(fig_pos, x, y, data; colormap, colorrange, contour_levels = nothing,
                             contour_color = RGBf(0.6, 0.6, 0.6), title = "", xlabel = "", ylabel = "",
                             aspect = (3, 1, 1), ylabeloffset = 40, ylims = nothing)
    ax = topdown_axis3(fig_pos; title = title, xlabel = xlabel, ylabel = ylabel,
        aspect = aspect, ylabeloffset = ylabeloffset)
    hm = plot_curvilinear!(ax, x, y, data; colormap = colormap, colorrange = colorrange,
        contour_levels = contour_levels, contour_color = contour_color)
    ylims !== nothing && ylims!(ax, ylims...)
    return ax, hm
end

# Perimeter of a curvilinear grid (bottom row, right column, top row
# reversed, left column reversed) as one closed loop — for overlaying a
# child-grid outline on top of a parent-grid plot.
function grid_boundary(x::AbstractMatrix, y::AbstractMatrix)
    bx = vcat(x[:, 1], x[end, 2:end], reverse(x[1:end-1, end]), reverse(x[1, 1:end-1]))
    by = vcat(y[:, 1], y[end, 2:end], reverse(y[1:end-1, end]), reverse(y[1, 1:end-1]))
    return bx, by
end

# quiver arrows on top of a topdown_axis3 plot — arrows2d! is the current
# Makie quiver function (arrows! is deprecated). We flatten everything to
# vectors and add an explicit z=0 / w=0 so it plots flat inside the 3D axis.
#
# xlim/ylim (optional, e.g. from `extrema(lon_chd)`) restrict which arrows
# are kept, applied *after* downsampling by `skip` — useful when the field
# being plotted spans a much larger domain than the axis is zoomed to
# (xlims!/ylims!), since arrows outside the view can otherwise still poke
# in from just past the visible edge, and most of a full-grid `skip` budget
# would otherwise be spent on arrows the zoom never shows.
function quiver_curvilinear!(ax, x, y, u, v; skip = 1, lengthscale = 1.0,
                              xlim = nothing, ylim = nothing, kwargs...)
    xs = vec(x[1:skip:end, 1:skip:end])
    ys = vec(y[1:skip:end, 1:skip:end])
    us = vec(u[1:skip:end, 1:skip:end])
    vs = vec(v[1:skip:end, 1:skip:end])

    keep = .!isnan.(us) .& .!isnan.(vs)
    xlim !== nothing && (keep .&= xlim[1] .<= xs .<= xlim[2])
    ylim !== nothing && (keep .&= ylim[1] .<= ys .<= ylim[2])
    xs, ys, us, vs = xs[keep], ys[keep], us[keep], vs[keep]

    zs = zeros(length(xs))
    ws = zeros(length(xs))
    return arrows2d!(ax, xs, ys, zs, us, vs, ws; lengthscale = lengthscale, kwargs...)
end
