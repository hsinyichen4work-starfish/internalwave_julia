using NCDatasets, CairoMakie, Dates, Statistics
CairoMakie.activate!()
include("/home/hsinyi/Documents/Julia/function/load_all.jl")
include("/home/hsinyi/Documents/Julia/function/plotting_fun.jl")

## path setting
datadir = "/home/mbui/ModelOutput/NCOM/HYCOM_data/"   # GOFS 3.5 HYCOM 1/25° global: <yyyymmddhh>_t0000_{ssh,ts3z,uv3z}.nc
child_grid = "/home/hsinyi/roms_data/grid/roms_grd_900m.nc"   # ROMS 900m grid — boundary line
parent_grid = "/home/mbui/ModelOutput/NCOM/grid/ohgrd_2.nc"    # NCOM 900m grid — boundary line + map extent
mooring_loc = "/home/hsinyi/roms_data/grid/roms_grd_900m_mor_edata.nc"   # mooring locations in its global attributes
figure_path = "/home/hsinyi/figure/HYCOM_data"
mkpath(figure_path)
colorbar_file = joinpath(figure_path, "colorbar_nc.nc")   # color range of every plot, read by time_capture_NCOM.jl

##
# fname = "/home/mbui/ModelOutput/NCOM/HYCOM_data/gofs35_hycom_glbz04_2022080112_t0000_ts3z.nc"
# NCDataset(fname) do ds
#     show(ds)
# end

## settings
plot_depths = [0, 100, 200, 250]   # depths (m) to plot temp/salt/speed at — must be HYCOM z-levels (printed below)

## list the files, grouped by date
files = sort(filter(f -> endswith(f, ".nc"), readdir(datadir)))
dates = unique([match(r"glbz04_(\d{10})_", f)[1] for f in files])   # "2022080112", ...
println("$(length(files)) files, $(length(dates)) dates in $datadir")
for d in dates
    println("  ", d, ": ", join(filter(f -> occursin(d, f), files), ", "))
end

## print the header (dims, variables, units, time) of one file of each type
for kind in ["ssh", "ts3z", "uv3z"]
    fname = joinpath(datadir, "gofs35_hycom_glbz04_$(dates[1])_t0000_$(kind).nc")
    NCDataset(fname) do ds
        println("\n==== $kind  ($(basename(fname))) ====")
        println("dims: ", Dict(k => v for (k, v) in ds.dim))
        for v in keys(ds)
            var = variable(ds, v)   # raw variable: avoids CF time parsing of tau ("hours since analysis")
            a = var.attrib
            println("  ", rpad(v, 12), size(var), "  ", get(a, "long_name", ""), " [", get(a, "units", ""), "]")
        end
        println("time: ", ds["time"][:])
    end
end

## grid (same for all files): lon 0..360, lat -80.48..90, 40 z-levels
lon, lat, depth = NCDataset(joinpath(datadir, "gofs35_hycom_glbz04_$(dates[1])_t0000_ts3z.nc")) do ds
    ds["lon"][:], ds["lat"][:], ds["depth"][:]
end
lon180 = ifelse.(lon .> 180, lon .- 360, lon)   # -180..180 (east/west hemisphere)
ilon = sortperm(lon180)                         # reorder columns so lon180 is monotonic
lon180 = lon180[ilon]
println("\ndepth levels (m): ", depth)
k_depths = [findfirst(==(z), depth) for z in plot_depths]   # fixed z grid, so each depth is one index
any(isnothing, k_depths) && error("plot_depths $plot_depths not all in HYCOM depth levels")
println("plotting depths ", plot_depths, " m = indices ", k_depths)

## map region: the NCOM 900m grid, padded by 0.5 degrees, with the NCOM 900m and
# ROMS 900m grid outlines drawn on every map
lon_chd, lat_chd = NCDataset(child_grid) do ds
    ds["lon_rho"][:, :], ds["lat_rho"][:, :]
end
lon_chd[lon_chd .> 180] .-= 360   # 0-360 -> -180/180, same as lon180
lon_chd_b, lat_chd_b = grid_boundary(lon_chd, lat_chd)
lon_par, lat_par = NCDataset(parent_grid) do ds   # already -180/180
    ds["lon"][:, :], ds["lat"][:, :]
end
lon_par_b, lat_par_b = grid_boundary(lon_par, lat_par)
lon_lim = extrema(lon_par) .+ (-0.5, 0.5)
lat_lim = extrema(lat_par) .+ (-0.5, 0.5)
ix = findall(x -> lon_lim[1] <= x <= lon_lim[2], lon180)   # HYCOM indices inside the map region
iy = findall(y -> lat_lim[1] <= y <= lat_lim[2], lat)

## mooring locations (lon -180..180, lat), from the global attributes of mooring_loc, e.g.
#   :M1_mor_info = "indices for M1_mor in roms_grd_900m.nc , location:(-45.4802;2.5415)"
mor_loc = NCDataset(mooring_loc) do ds
    out = Dict{String,NTuple{2,Float64}}()
    for k in keys(ds.attrib)
        endswith(k, "_mor_info") || continue
        m = match(r"location:\((-?[\d.]+);(-?[\d.]+)\)", ds.attrib[k])
        m === nothing && continue
        out[replace(k, "_mor_info" => "")] = (parse(Float64, m[1]), parse(Float64, m[2]))
    end
    out
end
moor_name  = sort(collect(keys(mor_loc)))
moor_lon   = [mor_loc[n][1] for n in moor_name]
moor_lat   = [mor_loc[n][2] for n in moor_name]
moor_group = [replace(n, r"\d+$" => "") for n in moor_name]   # "M3" -> "M", "CPIES7" -> "CPIES"

# one marker style per mooring type (same as figure_of_region.jl)
moor_style = Dict("french" => (marker = :star5,   markersize = 11, color = :magenta),
                  "M"      => (marker = :diamond, markersize = 7,  color = :cyan),
                  "CPIES"  => (marker = :circle,  markersize = 6,  color = :yellow))

# marks every mooring on the map, one legend entry per mooring type
function add_mooring!(ax)
    for g in unique(moor_group)
        k = findall(==(g), moor_group)
        st = get(moor_style, g, (marker = :rect, markersize = 6, color = :white))
        scatter!(ax, moor_lon[k], moor_lat[k]; marker = st.marker, markersize = st.markersize,
                 color = st.color, strokecolor = :black, strokewidth = 0.75, label = g)
    end
end

# read one horizontal slice (lon x lat) of a variable; only that depth level is
# read from disk (the 3D files are ~12 GB each). Fill values become NaN.
function read_slice(fname, var; k = 1)
    NCDataset(fname) do ds
        v = ds[var]
        a = ndims(v) == 4 ? v[:, :, k, 1] : v[:, :, 1]
        Float32.(coalesce.(a, NaN32))
    end
end

## quiver settings (same style as check_output_parallel.jl)
skip = 18   # every 18th HYCOM point (~0.7 deg) — ~18 arrows across the map
# lengthscale = degrees of lon/lat drawn per 1 m/s
quiver_kw = (lengthscale = 0.6, color = :black, shaftwidth = 1.5, tipwidth = 5, tiplength = 6)
# reference arrow for the quiver scale, in the lower-left corner (over land)
ref_speed = 1.0   # m/s
ref_lon = lon_lim[1] + 0.04 * (lon_lim[2] - lon_lim[1])
ref_lat = lat_lim[1] + 0.05 * (lat_lim[2] - lat_lim[1])

# velocity arrows + their reference arrow. HYCOM u/v are already eastward/northward
# on a regular lon/lat grid, so no rotation is needed (unlike ROMS)
function add_quiver!(ax, u, v)
    xs = vec([x for x in lon180[ix][1:skip:end], _ in lat[iy][1:skip:end]])
    ys = vec([y for _ in lon180[ix][1:skip:end], y in lat[iy][1:skip:end]])
    us = vec(u[1:skip:end, 1:skip:end])
    vs = vec(v[1:skip:end, 1:skip:end])
    keep = .!isnan.(us) .& .!isnan.(vs)
    arrows2d!(ax, xs[keep], ys[keep], us[keep], vs[keep]; quiver_kw...)
    arrows2d!(ax, [ref_lon], [ref_lat], [ref_speed], [0.0]; quiver_kw...)
    text!(ax, ref_lon, ref_lat; text = "$(ref_speed) m/s", fontsize = 14,
          align = (:left, :bottom), offset = (0, 6))
end

# SSH gradient magnitude |∇ssh| (m/m), same method as check_NCOM_zeta.jl / ROMS:
# differences between neighboring cells (U/V points), masked across land/sea
# boundaries, then averaged back over wet neighbors only. HYCOM is a regular
# lon/lat grid, so dx = R cos(lat) dlon and dy = R dlat. Land (NaN) is zero-filled
# before differencing, since NaN * 0 is still NaN.
const R_earth = 6371e3   # m
function ssh_gradient(ssh)
    wet = .!isnan.(ssh)
    ssh0 = ifelse.(wet, ssh, 0.0)
    umask = wet[1:end-1, :] .& wet[2:end, :]
    vmask = wet[:, 1:end-1] .& wet[:, 2:end]
    dx_u = R_earth .* cosd.(lat') .* deg2rad.(diff(lon))      # (lon-1) x lat, m
    dy_v = R_earth .* deg2rad.(diff(lat))'                     # 1 x (lat-1), m
    dzdx_u = diff(ssh0, dims = 1) ./ dx_u .* umask
    dzdy_v = diff(ssh0, dims = 2) ./ dy_v .* vmask
    dzdx = u2rho(dzdx_u) ./ u2rho(Float64.(umask))
    dzdy = v2rho(dzdy_v) ./ v2rho(Float64.(vmask))
    return Float32.(ifelse.(wet, sqrt.(dzdx .^ 2 .+ dzdy .^ 2), NaN))
end
grad_scale = 1e6        # plot in units of 10^-6, i.e. mm of SSH change per km
grad_range = (0, 10)    # same color range as the NCOM/ROMS zeta gradient plots

## color range of every plot, keyed by the figure name without .png (e.g.
# "temp_100m_2022090112"), saved to colorbar_file so time_capture_NCOM.jl can draw
# the NCOM snapshot of the same variable/depth/time with the same colorbar.
# Existing entries are read first, so a partial run (e.g. one date) only
# updates its own plots instead of dropping the others.
colorbars = isfile(colorbar_file) ? NCDataset(colorbar_file) do ds
    Dict(n => (lo, hi) for (n, lo, hi) in zip(ds["name"][:], ds["cmin"][:], ds["cmax"][:]))
end : Dict{String,NTuple{2,Float64}}()

function save_colorbars(fname, colorbars)
    names = sort(collect(keys(colorbars)))
    NCDataset(fname, "c") do ds
        defDim(ds, "plot", length(names))
        defVar(ds, "name", names, ("plot",); attrib = ["long_name" => "figure name (file name without .png)"])
        defVar(ds, "cmin", [Float64(colorbars[n][1]) for n in names], ("plot",); attrib = ["long_name" => "colorbar minimum"])
        defVar(ds, "cmax", [Float64(colorbars[n][2]) for n in names], ("plot",); attrib = ["long_name" => "colorbar maximum"])
        ds.attrib["source"] = "HYCOM_data.jl"
    end
end

# regional map (full resolution) of one field, with the ROMS 900m grid outline
# and the velocity arrows (u, v: full lon x lat slices, as from read_slice)
function plot_field(field, u, v, title, label, cmap, outfile; colorrange = nothing)
    sub = field[ilon, :][ix, iy]
    cr = something(colorrange, quantile(filter(!isnan, vec(sub)), (0.02, 0.98)))
    colorbars[splitext(basename(outfile))[1]] = Float64.(Tuple(cr))
    fig = Figure(size = (800, 760))
    ax = Axis(fig[1, 1], title = title, xlabel = "lon", ylabel = "lat", aspect = DataAspect())
    hm = heatmap!(ax, lon180[ix], lat[iy], sub; colormap = cmap, colorrange = cr, nan_color = :gray80)
    add_quiver!(ax, u[ilon, :][ix, iy], v[ilon, :][ix, iy])
    lines!(ax, lon_par_b, lat_par_b; color = :black, linewidth = 2, linestyle = :dash, label = "NCOM 900m grid")
    lines!(ax, lon_chd_b, lat_chd_b; color = :red, linewidth = 2, label = "ROMS 900m grid")
    add_mooring!(ax)   # last, so the markers sit on top of the arrows and lines
    limits!(ax, lon_lim..., lat_lim...)
    Colorbar(fig[1, 2], hm, label = label)
    Legend(fig[2, 1:2], ax; orientation = :horizontal, framevisible = false)
    save(outfile, fig)
    println("saved ", outfile)
end

## plot for every date: SSH and SSH gradient (with surface arrows) once, then temperature,
# salinity and current speed at each of plot_depths, with that depth's arrows
for d in dates
    f(kind) = joinpath(datadir, "gofs35_hycom_glbz04_$(d)_t0000_$(kind).nc")
    t = DateTime(d, dateformat"yyyymmddHH")

    for (z, k) in zip(plot_depths, k_depths)
        zstr = "$(z)m"
        u = read_slice(f("uv3z"), "water_u"; k = k)
        v = read_slice(f("uv3z"), "water_v"; k = k)

        if k == 1
            ssh = read_slice(f("ssh"), "surf_el")
            plot_field(ssh, u, v, "HYCOM SSH, vel at $zstr  $t", "SSH (m)", :balance,
                       joinpath(figure_path, "ssh_$(d).png"))
            plot_field(ssh_gradient(ssh) .* grad_scale, u, v, "HYCOM SSH gradient |∇ζ|, vel at $zstr  $t",
                       "10⁻⁶ (mm/km)", :amp, joinpath(figure_path, "ssh_grad_$(d).png");
                       colorrange = grad_range)
        end

        temp = read_slice(f("ts3z"), "water_temp"; k = k)
        plot_field(temp, u, v, "HYCOM temperature at $zstr  $t", "T (°C)", :thermal,
                   joinpath(figure_path, "temp_$(zstr)_$(d).png"))

        salt = read_slice(f("ts3z"), "salinity"; k = k)
        plot_field(salt, u, v, "HYCOM salinity at $zstr  $t", "S (psu)", :haline,
                   joinpath(figure_path, "salt_$(zstr)_$(d).png"))

        plot_field(hypot.(u, v), u, v, "HYCOM current speed at $zstr  $t", "|u| (m/s)", :speed,
                   joinpath(figure_path, "speed_$(zstr)_$(d).png"))
    end
    save_colorbars(colorbar_file, colorbars)   # after every date, so a crash keeps what's done
    println("saved color ranges to ", colorbar_file)
end
