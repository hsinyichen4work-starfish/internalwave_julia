using CairoMakie

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
function topdown_axis3(pos; title = "", xlabel = "Longitude", ylabel = "Latitude")
    return Axis3(pos;
        title = title, xlabel = xlabel, ylabel = ylabel,
        aspect = :data,                 # equivalent of ax.set_aspect("equal")
        elevation = pi / 2,             # look straight down the z-axis
        azimuth = -pi / 2,
        perspectiveness = 0,            # orthographic, i.e. no perspective skew
        zticksvisible = false, zticklabelsvisible = false,
        zlabelvisible = false, zgridvisible = false, zspinesvisible = false)
end

function plot_curvilinear!(ax, x, y, data; kwargs...)
    return surface!(ax, x, y, zeros(size(data)); color = data, shading = NoShading, kwargs...)
end

# quiver arrows on top of a topdown_axis3 plot — arrows2d! is the current
# Makie quiver function (arrows! is deprecated). We flatten everything to
# vectors and add an explicit z=0 / w=0 so it plots flat inside the 3D axis.
function quiver_curvilinear!(ax, x, y, u, v; skip = 1, lengthscale = 1.0, kwargs...)
    xs = vec(x[1:skip:end, 1:skip:end])
    ys = vec(y[1:skip:end, 1:skip:end])
    us = vec(u[1:skip:end, 1:skip:end])
    vs = vec(v[1:skip:end, 1:skip:end])
    zs = zeros(length(xs))
    ws = zeros(length(xs))
    return arrows2d!(ax, xs, ys, zs, us, vs, ws; lengthscale = lengthscale, kwargs...)
end
