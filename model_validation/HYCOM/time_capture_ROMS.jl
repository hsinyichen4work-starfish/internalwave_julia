using NCDatasets, CairoMakie, Dates, Statistics
CairoMakie.activate!()
include("/home/hchen54/internalwave_julia/function/load_all_hpc.jl")
include("/home/hchen54/internalwave_julia/function/plotting_fun.jl")

# Same maps as HYCOM_data.jl / time_capture_NCOM.jl / time_capture_CMEMS.jl
# (SSH, SSH gradient, temperature, salinity, current speed, with velocity arrows,
# grid outlines and moorings), but from our ROMS 2700m CMEMS run on Expanse, at
# every HYCOM time that the ROMS history files also cover.

## path setting
proj = "/expanse/lustre/projects/uso102/hchen54/"
run_fold = "amazon_2700m_cmems_3mon"
datadir = proj * run_fold * "/after_joint/"   # roms_his.<yyyymmddHH>0000.nc, 4 hourly records each
grid_fname = proj * "input/grid/roms_cmems_grd_2700m.nc"            # ROMS 2700m grid — data grid
child_grid = proj * "input/grid/roms_grd_900m.nc"                   # ROMS 900m grid — boundary line
parent_grid = proj * "input/grid/ohgrd_2.nc"   # NCOM 900m grid — boundary line + map extent (copy it from the USM
                                               # server, /home/mbui/ModelOutput/NCOM/grid/; without it the ROMS 2700m extent is used)
mooring_loc = proj * "input/grid/roms_cmems_grd_2700m_mor_edata.nc"   # mooring locations in its global attributes
figure_path = proj * "figure/20261016_hycomdata/ROMS_2700m"   # same file names as the HYCOM figures
mkpath(figure_path)
colorbar_file = proj * "data_upload/colorbar_nc.nc"   # HYCOM color ranges, written by HYCOM_data.jl

## settings
plot_depths = [0, 100, 200]   # depths (m) to plot at: 0 = the top sigma layer, others interpolated

# HYCOM snapshot times (12:00 UTC); the HYCOM files aren't on Expanse, so they're listed here
hycom_times = [DateTime(2022, 8, 1, 12), DateTime(2022, 9, 1, 12), DateTime(2022, 10, 1, 12),
               DateTime(2022, 11, 1, 12), DateTime(2022, 11, 28, 12)]

# ocean_time in the ROMS files is seconds since 1994-01-01
ot2datetime(ot) = DateTime(1994, 1, 1) + Millisecond(round(Int, ot * 1000))

## times: every HYCOM time with a matching ROMS history record. A history file
# holds the 4 hourly records starting at the time in its name, so 12:00 is in
# the file named ...120000
# (time, file, record index) for each overlapping time
captures = Tuple{DateTime,String,Int}[]
for t in hycom_times
    fname = joinpath(datadir, "roms_his." * Dates.format(t, "yyyymmddHH") * "0000.nc")
    if !isfile(fname)
        println("no ROMS history file for HYCOM time $t, skipping")
        continue
    end
    rec = NCDataset(ds -> findfirst(==(t), ot2datetime.(ds["ocean_time"][:])), fname)
    if rec === nothing
        println("$(basename(fname)) has no record at $t, skipping")
        continue
    end
    push!(captures, (t, fname, rec))
end
println("ROMS records at HYCOM times:")
foreach(c -> println("  ", c[1], " -> ", basename(c[2]), " record ", c[3]), captures)

## ROMS grid; data are (xi, eta[, s]), s = 1 at the bottom
mask_rho, lon_rho, lat_rho, h, pm, pn, grid_angle = NCDataset(grid_fname) do ds
    ds["mask_rho"][:, :], ds["lon_rho"][:, :], ds["lat_rho"][:, :],
    ds["h"][:, :], ds["pm"][:, :], ds["pn"][:, :], ds["angle"][:, :]
end
lon_rho[lon_rho .> 180] .-= 360   # 0-360 -> -180/180
land = mask_rho .== 0
cos_ang, sin_ang = Float32.(cos.(grid_angle)), Float32.(sin.(grid_angle))   # Float32 keeps the 3D velocities Float32
theta_s, theta_b, hc, N = NCDataset(captures[1][2]) do ds
    ds.attrib["theta_s"], ds.attrib["theta_b"], ds.attrib["hc"], size(ds["temp"], 3)
end

## NCOM 900m and ROMS 900m grid outlines; the map extent is the NCOM grid's, as in
# the HYCOM/NCOM/CMEMS scripts, so all models' maps line up
if isfile(parent_grid)
    lon_par, lat_par = NCDataset(parent_grid) do ds
        ds["lon"][:, :], ds["lat"][:, :]
    end
    lon_par_b, lat_par_b = grid_boundary(lon_par, lat_par)
    lon_lim = extrema(lon_par) .+ (-0.5, 0.5)
    lat_lim = extrema(lat_par) .+ (-0.5, 0.5)
else
    println("no NCOM grid at $parent_grid — no NCOM outline, map extent = the ROMS 2700m grid")
    lon_par_b = lat_par_b = nothing
    lon_lim = extrema(lon_rho)
    lat_lim = extrema(lat_rho)
end

lon_chd, lat_chd = NCDataset(child_grid) do ds
    ds["lon_rho"][:, :], ds["lat_rho"][:, :]
end
lon_chd[lon_chd .> 180] .-= 360   # 0-360 -> -180/180
lon_chd_b, lat_chd_b = grid_boundary(lon_chd, lat_chd)

## one time record: zeta, and temp/salt/east-north velocity (xi, eta, s) on rho
# points, land set to NaN, plus the rho-point depths z (negative down, ascending)
function read_record(fname, rec)
    zeta, temp, salt, u, v = NCDataset(fname) do ds
        Float64.(ds["zeta"][:, :, rec]), ds["temp"][:, :, :, rec], ds["salt"][:, :, :, rec],
        ds["u"][:, :, :, rec], ds["v"][:, :, :, rec]
    end
    z_dum, _ = zlevs3(h, zeta, theta_s, theta_b, hc, N, "r", "new2008")
    z = permutedims(z_dum, (2, 3, 1))
    # average raw (unmasked) u, v onto rho points first, then rotate from the
    # grid's xi/eta directions to east/north, then mask
    u_rho, v_rho = u2rho(u), v2rho(v)
    u_east  = u_rho .* cos_ang .- v_rho .* sin_ang
    v_north = u_rho .* sin_ang .+ v_rho .* cos_ang
    m(F) = ifelse.(land, NaN32, Float32.(F))
    return (zeta = m(zeta), z = z, temp = m(temp), salt = m(salt), u = m(u_east), v = m(v_north))
end

# horizontal slice of a 3D field F at depth zd (m): the top sigma layer for 0,
# otherwise interpolated in each column (NaN below the seafloor)
slice(z, F, zd) = zd == 0 ? F[:, :, end] : slice_at_depth(z, F, -float(zd))

# SSH gradient magnitude |∇ζ| (m/m), same method as HYCOM_data.jl / check_NCOM_zeta.jl:
# differences between neighboring cells (U/V points), masked across land/sea
# boundaries, then averaged back over wet neighbors only. Land (NaN) is
# zero-filled before differencing, since NaN * 0 is still NaN.
function ssh_gradient(ssh)
    wet = .!isnan.(ssh)
    ssh0 = ifelse.(wet, ssh, 0.0)
    umask = wet[1:end-1, :] .& wet[2:end, :]
    vmask = wet[:, 1:end-1] .& wet[:, 2:end]
    dzdx_u = diff(ssh0, dims = 1) .* rho2u(pm) .* umask   # pm = 1/dx
    dzdy_v = diff(ssh0, dims = 2) .* rho2v(pn) .* vmask   # pn = 1/dy
    dzdx = u2rho(dzdx_u) ./ u2rho(Float64.(umask))
    dzdy = v2rho(dzdy_v) ./ v2rho(Float64.(vmask))
    return Float32.(ifelse.(wet, sqrt.(dzdx .^ 2 .+ dzdy .^ 2), NaN))
end
grad_scale = 1e6        # plot in units of 10^-6, i.e. mm of SSH change per km
grad_range = (0, 10)    # same color range as the HYCOM/NCOM/CMEMS zeta gradient plots

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
skip = 26   # every 26th ROMS 2700m point (~70 km), about the HYCOM arrow spacing
# lengthscale = degrees of lon/lat drawn per 1 m/s
quiver_kw = (lengthscale = 0.6, color = :black, shaftwidth = 1.5, tipwidth = 5, tiplength = 6)
# reference arrow for the quiver scale, in the lower-left corner (over land)
ref_speed = 1.0   # m/s
ref_lon = lon_lim[1] + 0.04 * (lon_lim[2] - lon_lim[1])
ref_lat = lat_lim[1] + 0.05 * (lat_lim[2] - lat_lim[1])

# velocity arrows (already east/north) + their reference arrow
function add_quiver!(ax, u, v)
    # start half a step in from the edges: the open-boundary rows/columns
    # (sponge) don't represent the interior flow
    r = skip ÷ 2
    xs = vec(lon_rho[r:skip:end, r:skip:end])
    ys = vec(lat_rho[r:skip:end, r:skip:end])
    us = vec(u[r:skip:end, r:skip:end])
    vs = vec(v[r:skip:end, r:skip:end])
    keep = .!isnan.(us) .& .!isnan.(vs)
    arrows2d!(ax, xs[keep], ys[keep], us[keep], vs[keep]; quiver_kw...)
    arrows2d!(ax, [ref_lon], [ref_lat], [ref_speed], [0.0]; quiver_kw...)
    text!(ax, ref_lon, ref_lat; text = "$(ref_speed) m/s", fontsize = 10,
          align = (:left, :bottom), offset = (0, 6))
end

## HYCOM color range of every plot, keyed by the figure name without .png
# (e.g. "temp_100m_2022090112"); ROMS figures have the same names, so each ROMS
# snapshot is drawn with the colorbar of the HYCOM one at the same time/variable/depth
hycom_colorbars = isfile(colorbar_file) ? NCDataset(colorbar_file) do ds
    Dict(n => (lo, hi) for (n, lo, hi) in zip(ds["name"][:], ds["cmin"][:], ds["cmax"][:]))
end : Dict{String,NTuple{2,Float64}}()
isempty(hycom_colorbars) && println("no HYCOM color ranges in $colorbar_file — run HYCOM_data.jl first; ",
                                    "falling back to each map's own range")

# map of one ROMS field on its curvilinear grid (flat surface, as plot_curvilinear!
# does), with the NCOM/ROMS 900m grid outlines, velocity arrows and moorings
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
              backgroundcolor = :gray80,   # land / below seafloor / outside the ROMS domain shows as gray
              xgridvisible = false, ygridvisible = false)
    hm = surface!(ax, lon_rho, lat_rho, zeros(size(field)); color = field, shading = NoShading,
                  colormap = cmap, colorrange = cr, nan_color = :transparent)
    add_quiver!(ax, u, v)
    lon_par_b === nothing ||
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
for (t, fname, rec) in captures
    d = Dates.format(t, "yyyymmddHH")   # same date stamp as the HYCOM figure names
    r = read_record(fname, rec)

    for zd in plot_depths
        zstr = "$(zd)m"
        u = slice(r.z, r.u, zd)
        v = slice(r.z, r.v, zd)

        if zd == 0
            plot_field(r.zeta, u, v, "ROMS 2700m SSH, vel at $zstr  $t", "SSH (m)", :balance,
                       joinpath(figure_path, "ssh_$(d).png"))
            plot_field(ssh_gradient(r.zeta) .* grad_scale, u, v, "ROMS 2700m SSH gradient |∇ζ|, vel at $zstr  $t",
                       "10⁻⁶ (mm/km)", :amp, joinpath(figure_path, "ssh_grad_$(d).png");
                       colorrange = grad_range)
        end

        plot_field(slice(r.z, r.temp, zd), u, v, "ROMS 2700m temperature at $zstr  $t", "T (°C)", :thermal,
                   joinpath(figure_path, "temp_$(zstr)_$(d).png"))
        plot_field(slice(r.z, r.salt, zd), u, v, "ROMS 2700m salinity at $zstr  $t", "S (psu)", :haline,
                   joinpath(figure_path, "salt_$(zstr)_$(d).png"))
        plot_field(hypot.(u, v), u, v, "ROMS 2700m current speed at $zstr  $t", "|u| (m/s)", :speed,
                   joinpath(figure_path, "speed_$(zstr)_$(d).png"))
    end
    r = nothing
    GC.gc()   # free this record's 3D fields before reading the next one
end
