using NCDatasets, CairoMakie, Dates, Statistics
CairoMakie.activate!()
include("/home/hsinyi/Documents/Julia/function/load_all.jl")
include("/home/hsinyi/Documents/Julia/function/plotting_fun.jl")

# Same maps as HYCOM_data.jl (SSH, temperature, salinity, current speed, with
# velocity arrows, grid outlines and moorings), but from NCOM, at every HYCOM
# time that the NCOM files also cover.

## path setting
hycom_dir = "/home/mbui/ModelOutput/NCOM/HYCOM_data/"   # only used for its time stamps
datadir = "/home/hsinyi/roms_data/NCOM_DATA_NC/"        # NCOM daily files: <yyyymmdd>00_{ssh,ts,uv}.nc, 25 hourly records
child_grid = "/home/hsinyi/roms_data/grid/roms_grd_900m.nc"   # ROMS 900m grid — boundary line
parent_grid = "/home/mbui/ModelOutput/NCOM/grid/ohgrd_2.nc"    # NCOM 900m grid — data grid, boundary line + map extent
mooring_loc = "/home/hsinyi/roms_data/grid/roms_grd_900m_mor_edata.nc"   # mooring locations in its global attributes
figure_path = "/home/hsinyi/figure/HYCOM_data/NCOM"   # same file names as the HYCOM figures one folder up
mkpath(figure_path)

## settings
plot_depths = [0, 100, 200, 250]   # depths (m) to plot temp/salt/speed at, same as HYCOM_data.jl

# MT in the NCOM files is days since 1900-12-31 00:00
mt2datetime(mt) = DateTime(1900, 12, 31) + Millisecond(round(Int, mt * 86_400_000))

## times: every HYCOM time with a matching NCOM record
hycom_times = sort(unique([DateTime(match(r"glbz04_(\d{10})_", f)[1], dateformat"yyyymmddHH")
                           for f in readdir(hycom_dir) if endswith(f, ".nc")]))
# (time, NCOM file tag, record index) for each overlapping time
captures = Tuple{DateTime,String,Int}[]
for t in hycom_times
    tag = Dates.format(t, "yyyymmdd") * "00"
    fname = joinpath(datadir, tag * "_ssh.nc")
    if !isfile(fname)
        println("no NCOM file for HYCOM time $t, skipping")
        continue
    end
    rec = NCDataset(fname) do ds
        findfirst(==(t), mt2datetime.(ds["MT"][:]))
    end
    if rec === nothing
        println("NCOM $tag has no record at $t, skipping")
        continue
    end
    push!(captures, (t, tag, rec))
end
println("NCOM records at HYCOM times:")
foreach(c -> println("  ", c[1], " -> ", c[2], " record ", c[3]), captures)

## NCOM grid. NCDatasets gives lon(dim_j, dim_i) as lon[i, j]; data are (xi, eta) = (i, j)
lon_par, lat_par, zm3 = NCDataset(parent_grid) do ds   # lon already -180/180
    Float64.(ds["lon"][:, :]), Float64.(ds["lat"][:, :]),
    Float64.(coalesce.(ds["zm3"][:, :, :], NaN))   # layer-center depth (i, j, k), negative down,
end                                                  # k = 1 at the surface, NaN below kb / on land
lon_par_b, lat_par_b = grid_boundary(lon_par, lat_par)
lon_lim = extrema(lon_par) .+ (-0.5, 0.5)
lat_lim = extrema(lat_par) .+ (-0.5, 0.5)
Ni, Nj, Nk = size(zm3)

## vertical interpolation weights, built once (zm3 is fixed in time):
# for each column and target depth z, the layer k above and the weight w so that
# F(z) = (1 - w) * F[k] + w * F[k+1]. Above the first layer center (z = 0 vs
# -0.5 m) the top layer is used; below the deepest wet layer the column is NaN.
function vertical_weights(zm3, z)
    kk = zeros(Int, size(zm3, 1), size(zm3, 2))   # 0 = no data (land / below seafloor)
    ww = zeros(size(zm3, 1), size(zm3, 2))
    for j in axes(zm3, 2), i in axes(zm3, 1)
        zc = view(zm3, i, j, :)
        isnan(zc[1]) && continue
        if z >= zc[1]
            kk[i, j], ww[i, j] = 1, 0.0
            continue
        end
        for k in 1:size(zm3, 3)-1
            isnan(zc[k+1]) && break
            if zc[k] >= z >= zc[k+1]
                kk[i, j], ww[i, j] = k, (zc[k] - z) / (zc[k] - zc[k+1])
                break
            end
        end
    end
    return kk, ww
end
vw = Dict(z => vertical_weights(zm3, -float(z)) for z in plot_depths)

# read one horizontal slice (xi x eta) of a 3D variable at depth z (m) and time
# record rec. Only the layers spanned by that depth are read from disk.
function read_slice(fname, var, rec; z = 0)
    kk, ww = vw[z]
    wet = kk .> 0
    k0, k1 = extrema(kk[wet])
    k1 = min(k1 + 1, Nk)
    F = NCDataset(fname) do ds
        Float64.(coalesce.(ds[var][:, :, k0:k1, rec], NaN))
    end
    out = fill(NaN32, Ni, Nj)
    for I in CartesianIndices(out)
        wet[I] || continue
        k = kk[I] - k0 + 1
        w = ww[I]
        out[I] = w == 0 ? F[I, k] : (1 - w) * F[I, k] + w * F[I, k+1]
    end
    return out
end
read_ssh(fname, rec) = NCDataset(ds -> Float32.(coalesce.(ds["ssh"][:, :, rec], NaN)), fname)

## ROMS 900m grid outline
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
skip = 80   # every 80th NCOM point (~70 km), about the HYCOM arrow spacing
# lengthscale = degrees of lon/lat drawn per 1 m/s
quiver_kw = (lengthscale = 0.6, color = :black, shaftwidth = 1.5, tipwidth = 5, tiplength = 6)
# reference arrow for the quiver scale, in the lower-left corner (over land)
ref_speed = 1.0   # m/s
ref_lon = lon_lim[1] + 0.04 * (lon_lim[2] - lon_lim[1])
ref_lat = lat_lim[1] + 0.05 * (lat_lim[2] - lat_lim[1])

# velocity arrows + their reference arrow. NCOM u_velocity/v_velocity are already
# east/north components (see check_NCOM_parallel.jl), so no rotation is needed
function add_quiver!(ax, u, v)
    # start half a step in from the edges: the boundary rows/columns of the
    # NCOM domain have unrealistic velocities
    r = skip ÷ 2
    xs = vec(lon_par[r:skip:end, r:skip:end])
    ys = vec(lat_par[r:skip:end, r:skip:end])
    us = vec(u[r:skip:end, r:skip:end])
    vs = vec(v[r:skip:end, r:skip:end])
    keep = .!isnan.(us) .& .!isnan.(vs)
    arrows2d!(ax, xs[keep], ys[keep], us[keep], vs[keep]; quiver_kw...)
    arrows2d!(ax, [ref_lon], [ref_lat], [ref_speed], [0.0]; quiver_kw...)
    text!(ax, ref_lon, ref_lat; text = "$(ref_speed) m/s", fontsize = 14,
          align = (:left, :bottom), offset = (0, 6))
end

# map of one NCOM field on its curvilinear grid (flat surface, as plot_curvilinear!
# does), with the NCOM/ROMS grid outlines, velocity arrows and moorings
function plot_field(field, u, v, title, label, cmap, outfile; colorrange = nothing)
    cr = something(colorrange, quantile(filter(!isnan, vec(field)), (0.02, 0.98)))
    fig = Figure(size = (800, 760))
    ax = Axis(fig[1, 1], title = title, xlabel = "lon", ylabel = "lat", aspect = DataAspect(),
              backgroundcolor = :gray80,   # land / below seafloor (NaN) shows as gray, like HYCOM
              xgridvisible = false, ygridvisible = false)
    hm = surface!(ax, lon_par, lat_par, zeros(size(field)); color = field, shading = NoShading,
                  colormap = cmap, colorrange = cr, nan_color = :transparent)
    add_quiver!(ax, u, v)
    lines!(ax, lon_par_b, lat_par_b; color = :black, linewidth = 2, linestyle = :dash, label = "NCOM 900m grid")
    lines!(ax, lon_chd_b, lat_chd_b; color = :red, linewidth = 2, label = "ROMS 900m grid")
    add_mooring!(ax)   # last, so the markers sit on top of the arrows and lines
    limits!(ax, lon_lim..., lat_lim...)
    Colorbar(fig[1, 2], hm, label = label)
    Legend(fig[2, 1:2], ax; orientation = :horizontal, framevisible = false)
    save(outfile, fig)
    println("saved ", outfile)
end

## plot for every captured time: SSH (with surface arrows) once, then temperature,
# salinity and current speed at each of plot_depths, with that depth's arrows
for (t, tag, rec) in captures
    f(kind) = joinpath(datadir, "$(tag)_$(kind).nc")
    d = Dates.format(t, "yyyymmddHH")   # same date stamp as the HYCOM figure names

    for z in plot_depths
        zstr = "$(z)m"
        u = read_slice(f("uv"), "u_velocity", rec; z = z)
        v = read_slice(f("uv"), "v_velocity", rec; z = z)

        if z == 0
            ssh = read_ssh(f("ssh"), rec)
            plot_field(ssh, u, v, "NCOM SSH, vel at $zstr  $t", "SSH (m)", :thermal,
                       joinpath(figure_path, "ssh_$(d).png"))
        end

        temp = read_slice(f("ts"), "layer_temperature", rec; z = z)
        plot_field(temp, u, v, "NCOM temperature at $zstr  $t", "T (°C)", :thermal,
                   joinpath(figure_path, "temp_$(zstr)_$(d).png"))

        salt = read_slice(f("ts"), "layer_salinity", rec; z = z)
        plot_field(salt, u, v, "NCOM salinity at $zstr  $t", "S (psu)", :haline,
                   joinpath(figure_path, "salt_$(zstr)_$(d).png"))

        plot_field(hypot.(u, v), u, v, "NCOM current speed at $zstr  $t", "|u| (m/s)", :speed,
                   joinpath(figure_path, "speed_$(zstr)_$(d).png"))
    end
end
