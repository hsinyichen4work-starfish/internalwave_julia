using NCDatasets, CairoMakie, Dates, Statistics
CairoMakie.activate!()
include("/home/hsinyi/Documents/Julia/function/load_all.jl")
include("/home/hsinyi/Documents/Julia/function/plotting_fun.jl")

# Same maps as HYCOM_data.jl / time_capture_NCOM.jl (SSH, temperature, salinity,
# current speed, with velocity arrows, grid outlines and moorings), but from
# CMEMS GLOBAL_ANALYSISFORECAST_PHY_001_024 (Mercator GLO12, 1/12°), at every
# HYCOM time that the CMEMS files also cover.

## path setting
hycom_dir = "/home/mbui/ModelOutput/NCOM/HYCOM_data/"   # only used for its time stamps
datadir = "/home/hsinyi/data_notm/"
# instantaneous snapshots (same as HYCOM/NCOM), lon -50..-40, lat -3..10:
# 6-hourly at the surface (0.5 m) and at depth (92-222 m), hourly for sea level
cur_surf_file  = joinpath(datadir, "cmems_cur_surf_snapshot.nc")    # uo, vo
temp_surf_file = joinpath(datadir, "cmems_temp_surf_snapshot.nc")   # thetao
salt_surf_file = joinpath(datadir, "cmems_salt_surf_snapshot.nc")   # so
cur_file       = joinpath(datadir, "cmems_cur_mid_snapshot.nc")     # uo, vo
temp_file      = joinpath(datadir, "cmems_temp_mid_snapshot.nc")    # thetao
salt_file      = joinpath(datadir, "cmems_salt_mid_snapshot.nc")    # so
sl_file        = joinpath(datadir, "cmems_ssh_snapshot.nc")         # total_sea_level, sea_surface_height, ocean_tide
# total_sea_level includes the ocean tide, like HYCOM's surf_el
# (sea_surface_height alone is computed without tides)
ssh_var = "total_sea_level"
child_grid = "/home/hsinyi/roms_data/grid/roms_grd_900m.nc"   # ROMS 900m grid — boundary line
parent_grid = "/home/mbui/ModelOutput/NCOM/grid/ohgrd_2.nc"    # NCOM 900m grid — boundary line + map extent
mooring_loc = "/home/hsinyi/roms_data/grid/roms_grd_900m_mor_edata.nc"   # mooring locations in its global attributes
figure_path = "/home/hsinyi/figure/20261016_hycomdata/CMEMS"   # same file names as the HYCOM figures one folder up
mkpath(figure_path)
colorbar_file = "/home/hsinyi/figure/20261016_hycomdata/colorbar_nc.nc"   # HYCOM color ranges, written by HYCOM_data.jl

## settings
# depths (m) to plot at: 0 = the surface files (0.49 m); the others are linearly
# interpolated between the deep files' levels (92, 110, 131, 156, 186, 222 m).
plot_depths = [0, 100, 200]

## times: every HYCOM time with a matching record in all CMEMS files
hycom_times = sort(unique([DateTime(match(r"glbz04_(\d{10})_", f)[1], dateformat"yyyymmddHH")
                           for f in readdir(hycom_dir) if endswith(f, ".nc")]))
file_times = Dict(f => NCDataset(ds -> ds["time"][:], f)
                  for f in (cur_surf_file, temp_surf_file, salt_surf_file, cur_file, temp_file, salt_file, sl_file))
# (time, Dict file => record index) for each overlapping time
captures = Tuple{DateTime,Dict{String,Int}}[]
for t in hycom_times
    recs = Dict(f => findfirst(==(t), tt) for (f, tt) in file_times)
    if any(isnothing, values(recs))
        println("CMEMS has no record at HYCOM time $t in every file, skipping")
        continue
    end
    push!(captures, (t, recs))
end
println("CMEMS records at HYCOM times (6-hourly / hourly sea level):")
foreach(c -> println("  ", c[1], " -> ", c[2][cur_file], " / ", c[2][sl_file]), captures)

## CMEMS grid: regular lon/lat (lon already -180/180); data are (lon, lat, depth, time)
lon, lat, depth_deep = NCDataset(cur_file) do ds
    Float64.(ds["longitude"][:]), Float64.(ds["latitude"][:]), Float64.(ds["depth"][:])
end
println("deep levels (m): ", round.(depth_deep, digits = 1))

## vertical interpolation: the two deep levels around z and the weight w so that
# F(z) = (1 - w) * F[k] + w * F[k+1]. Where the lower level is below the
# seafloor (NaN), the result is NaN too.
function vertical_weight(z)
    k = findlast(<=(z), depth_deep)
    (k === nothing || k == length(depth_deep)) && error("$z m is outside the deep levels $depth_deep")
    return k, (z - depth_deep[k]) / (depth_deep[k+1] - depth_deep[k])
end

# one horizontal slice (lon x lat) of var from fname at time record rec; for a
# surface file (one level) z is ignored, for a deep file the slice is
# interpolated to depth z (m)
function read_slice(fname, var, rec; z = 0)
    if z == 0
        return NCDataset(ds -> Float32.(coalesce.(ds[var][:, :, 1, rec], NaN)), fname)
    end
    k, w = vertical_weight(z)
    F = NCDataset(ds -> Float64.(coalesce.(ds[var][:, :, k:k+1, rec], NaN)), fname)
    return Float32.((1 - w) .* F[:, :, 1] .+ w .* F[:, :, 2])
end

# SSH gradient magnitude |∇ssh| (m/m), same method as HYCOM_data.jl / check_NCOM_zeta.jl:
# differences between neighboring cells (U/V points), masked across land/sea
# boundaries, then averaged back over wet neighbors only. CMEMS is a regular
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
grad_range = (0, 10)    # same color range as the HYCOM/NCOM/ROMS zeta gradient plots

## NCOM 900m and ROMS 900m grid outlines; the map extent is the NCOM grid's, as in
# HYCOM_data.jl / time_capture_NCOM.jl, so all three models' maps line up
lon_par, lat_par = NCDataset(parent_grid) do ds
    ds["lon"][:, :], ds["lat"][:, :]
end
lon_par_b, lat_par_b = grid_boundary(lon_par, lat_par)
lon_lim = extrema(lon_par) .+ (-0.5, 0.5)
lat_lim = extrema(lat_par) .+ (-0.5, 0.5)

lon_chd, lat_chd = NCDataset(child_grid) do ds
    ds["lon_rho"][:, :], ds["lat_rho"][:, :]
end
lon_chd[lon_chd .> 180] .-= 360   # 0-360 -> -180/180
lon_chd_b, lat_chd_b = grid_boundary(lon_chd, lat_chd)

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

## quiver settings (same arrow style/scale as HYCOM_data.jl)
skip = 9   # every 9th CMEMS point (0.75 deg), about the HYCOM arrow spacing
# lengthscale = degrees of lon/lat drawn per 1 m/s
quiver_kw = (lengthscale = 0.6, color = :black, shaftwidth = 1.5, tipwidth = 5, tiplength = 6)
# reference arrow for the quiver scale, in the lower-left corner (over land)
ref_speed = 1.0   # m/s
ref_lon = lon_lim[1] + 0.04 * (lon_lim[2] - lon_lim[1])
ref_lat = lat_lim[1] + 0.05 * (lat_lim[2] - lat_lim[1])

# velocity arrows + their reference arrow. CMEMS uo/vo are eastward/northward
# on a regular lon/lat grid, so no rotation is needed
function add_quiver!(ax, u, v)
    r = skip ÷ 2 + 1   # start half a step in, so arrows don't sit on the domain edge
    xs = vec([x for x in lon[r:skip:end], _ in lat[r:skip:end]])
    ys = vec([y for _ in lon[r:skip:end], y in lat[r:skip:end]])
    us = vec(u[r:skip:end, r:skip:end])
    vs = vec(v[r:skip:end, r:skip:end])
    keep = .!isnan.(us) .& .!isnan.(vs)
    arrows2d!(ax, xs[keep], ys[keep], us[keep], vs[keep]; quiver_kw...)
    arrows2d!(ax, [ref_lon], [ref_lat], [ref_speed], [0.0]; quiver_kw...)
    text!(ax, ref_lon, ref_lat; text = "$(ref_speed) m/s", fontsize = 10,
          align = (:left, :bottom), offset = (0, 6))
end

## HYCOM color range of every plot, keyed by the figure name without .png
# (e.g. "temp_100m_2022090112"); CMEMS figures have the same names, so each CMEMS
# snapshot is drawn with the colorbar of the HYCOM one at the same time/variable/depth
hycom_colorbars = isfile(colorbar_file) ? NCDataset(colorbar_file) do ds
    Dict(n => (lo, hi) for (n, lo, hi) in zip(ds["name"][:], ds["cmin"][:], ds["cmax"][:]))
end : Dict{String,NTuple{2,Float64}}()
isempty(hycom_colorbars) && println("no HYCOM color ranges in $colorbar_file — run HYCOM_data.jl first; ",
                                    "falling back to each map's own range")

# map of one CMEMS field, with the NCOM/ROMS grid outlines, velocity arrows and moorings
function plot_field(field, u, v, title, label, cmap, outfile; colorrange = nothing)
    key = splitext(basename(outfile))[1]
    cr = if colorrange !== nothing
        colorrange
    elseif haskey(hycom_colorbars, key)
        hycom_colorbars[key]
    else
        println("no HYCOM color range for $key, using this map's own range")
        quantile(filter(!isnan, vec(field)), (0.02, 0.98))
    end
    fig = Figure(; figkw(18, 17.1)...)
    ax = Axis(fig[1, 1], title = title, xlabel = "lon", ylabel = "lat", aspect = DataAspect(),
              backgroundcolor = :gray80,   # land / below seafloor / outside the CMEMS subset shows as gray
              xgridvisible = false, ygridvisible = false)
    hm = heatmap!(ax, lon, lat, field; colormap = cmap, colorrange = cr, nan_color = :transparent)
    add_quiver!(ax, u, v)
    lines!(ax, lon_par_b, lat_par_b; color = :black, linewidth = 2, linestyle = :dash, label = "NCOM 900m grid")
    lines!(ax, lon_chd_b, lat_chd_b; color = :red, linewidth = 2, label = "ROMS 900m grid")
    add_mooring!(ax)   # last, so the markers sit on top of the arrows and lines
    limits!(ax, lon_lim..., lat_lim...)
    Colorbar(fig[1, 2], hm, label = label)
    Legend(fig[2, 1:2], ax; orientation = :horizontal, framevisible = false)
    figsave(outfile, fig)
    println("saved ", outfile)
end

## plot for every captured time: SSH and SSH gradient (with surface arrows)
# once, then temperature, salinity and current speed at each of plot_depths,
# with that depth's arrows
for (t, rec) in captures
    d = Dates.format(t, "yyyymmddHH")   # same date stamp as the HYCOM figure names

    for z in plot_depths
        zstr = "$(z)m"
        cf, tf, sf = z == 0 ? (cur_surf_file, temp_surf_file, salt_surf_file) : (cur_file, temp_file, salt_file)
        u = read_slice(cf, "uo", rec[cf]; z = z)
        v = read_slice(cf, "vo", rec[cf]; z = z)

        if z == 0
            ssh = read_slice(sl_file, ssh_var, rec[sl_file])
            plot_field(ssh, u, v, "CMEMS SSH, vel at $zstr  $t", "SSH (m)", :balance,
                       joinpath(figure_path, "ssh_$(d).png"))
            plot_field(ssh_gradient(ssh) .* grad_scale, u, v, "CMEMS SSH gradient |∇ζ|, vel at $zstr  $t",
                       "10⁻⁶ (mm/km)", :amp, joinpath(figure_path, "ssh_grad_$(d).png");
                       colorrange = grad_range)
        end

        temp = read_slice(tf, "thetao", rec[tf]; z = z)
        plot_field(temp, u, v, "CMEMS temperature at $zstr  $t", "T (°C)", :thermal,
                   joinpath(figure_path, "temp_$(zstr)_$(d).png"))

        salt = read_slice(sf, "so", rec[sf]; z = z)
        plot_field(salt, u, v, "CMEMS salinity at $zstr  $t", "S (psu)", :haline,
                   joinpath(figure_path, "salt_$(zstr)_$(d).png"))

        plot_field(hypot.(u, v), u, v, "CMEMS current speed at $zstr  $t", "|u| (m/s)", :speed,
                   joinpath(figure_path, "speed_$(zstr)_$(d).png"))
    end
end
