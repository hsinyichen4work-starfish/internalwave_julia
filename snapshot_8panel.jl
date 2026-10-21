using NCDatasets, CairoMakie, Dates, Statistics
CairoMakie.activate!()
include("/home/hchen54/internalwave_julia/function/load_all_hpc.jl")
include("/home/hchen54/internalwave_julia/function/plotting_fun.jl")

# 2x4 report figure: row 1 = speed at 1 m (with arrows), row 2 = temperature,
# one column per snapshot time. Same styling as check_output_parallel.jl.

grid_fname = "/expanse/lustre/projects/uso101/hchen54/input/grid/roms_grd_900m.nc"
mooring_file = "/expanse/lustre/projects/uso101/hchen54/input/grid/roms_grd_900m_mor_edata.nc"
datadir = "/expanse/lustre/projects/uso101/hchen54/amazon_900m_3mon_1008_2/after_joint"
figure_path = "/expanse/lustre/projects/uso101/hchen54/figure/amazon_900m_3mon_1008"
mkpath(figure_path)

snap_times = [DateTime(2022, 9, 1, 12), DateTime(2022, 10, 1, 12),
              DateTime(2022, 11, 1, 12), DateTime(2022, 11, 28, 12)]
speed_depth = -1.0                 # m, negative below the surface
temp_depth = -1.0                  # switch to -100.0 for the subsurface field
speed_range = (0, 2)
temp_range = temp_depth > -50 ? (24, 30) : (15, 30)   # same limits as the single-snapshot plots

##
mask_rho, lon_rho, lat_rho, h, grid_angle = NCDataset(grid_fname) do ds
    ds["mask_rho"][:, :], ds["lon_rho"][:, :], ds["lat_rho"][:, :], ds["h"][:, :], ds["angle"][:, :]
end
lon_rho[lon_rho .> 180] .-= 360
lon_lim = extrema(lon_rho)
lat_lim = extrema(lat_rho)
bathy_levels = [500, 1000, 2000]

# every overlay size in check_output_parallel.jl is for an 800x600 panel;
# the panels here are smaller, so shrink markers/arrows/text by this ratio
sc = 0.6

moor_name, moor_lon, moor_lat = NCDataset(mooring_file) do ds
    names = [replace(v, "_mor" => "") for v in keys(ds) if endswith(v, "_mor")]
    locs  = [match(r"location:\(([-\d.]+);([-\d.]+)\)", ds.attrib["$(n)_mor_info"]) for n in names]
    names, [parse(Float64, m[1]) for m in locs], [parse(Float64, m[2]) for m in locs]
end
moor_group = [replace(n, r"\d+$" => "") for n in moor_name]
moor_style = Dict("french" => (marker = :star5,   markersize = 18, color = :magenta),
                  "M"      => (marker = :diamond, markersize = 11, color = :cyan),
                  "CPIES"  => (marker = :circle,  markersize = 10, color = :yellow))

function add_mooring!(ax; scale = 1)
    for g in unique(moor_group)
        k = findall(==(g), moor_group)
        st = get(moor_style, g, (marker = :rect, markersize = 10, color = :white))
        scatter!(ax, Point3f.(moor_lon[k], moor_lat[k], 0);
                 marker = st.marker, markersize = st.markersize * scale, color = st.color,
                 strokecolor = :black, strokewidth = 1.2 * scale, overdraw = true)
    end
end

skip = 40
quiver_kw(scale) = (lengthscale = 0.25, color = :black, shaftwidth = 1.5 * scale,
                    tipwidth = 5 * scale, tiplength = 6 * scale)
ref_speed = 1.0
ref_lon = lon_lim[2] - 0.22 * (lon_lim[2] - lon_lim[1])
ref_lat = lat_lim[1] + 0.07 * (lat_lim[2] - lat_lim[1])

function add_quiver!(ax, u_east, v_north; scale = 1)
    quiver_curvilinear!(ax, lon_rho, lat_rho, u_east, v_north; skip = skip, quiver_kw(scale)...)
    arrows2d!(ax, [ref_lon], [ref_lat], [0.0], [ref_speed], [0.0], [0.0]; quiver_kw(scale)...)
    text!(ax, Point3f(ref_lon, ref_lat, 0); text = "$(ref_speed) m/s", fontsize = 22 * scale,
          align = (:left, :bottom), offset = (0, 6 * scale), overdraw = true)
end

function map_axis(pos; title = "", show_x = true, show_y = true)
    ax = topdown_axis3(pos; title = title)
    ax.azimuth = -pi / 2 - 1e-4
    xlims!(ax, lon_lim...)
    ylims!(ax, lat_lim...)
    ax.xlabelvisible = show_x
    ax.ylabelvisible = show_y
    return ax
end

# reads one ocean_time record and returns the 2D fields for both rows
function load_snapshot(t_target)
    t_ref = DateTime(1994, 1, 1)
    fname = joinpath(datadir, "roms_his.$(Dates.format(t_target, "yyyymmddHH"))0000.nc")
    NCDataset(fname) do ds
        realtime = t_ref .+ Millisecond.(round.(Int, ds["ocean_time"][:] .* 1000))
        it = findfirst(==(t_target), realtime)
        it === nothing && error("$t_target not found in $fname (has $(realtime))")

        zeta = ds["zeta"][:, :, it]
        N = size(ds["temp"], 3)
        z_dum, _ = zlevs3(h, zeta, ds.attrib["theta_s"], ds.attrib["theta_b"], ds.attrib["hc"], N, "r", "new2008")
        z = permutedims(z_dum, (2, 3, 1))

        temp = ifelse.(mask_rho .== 0, NaN32, ds["temp"][:, :, :, it])
        u_rho = u2rho(ds["u"][:, :, :, it])
        v_rho = v2rho(ds["v"][:, :, :, it])
        speed = ifelse.(mask_rho .== 0, NaN32, sqrt.(u_rho .^ 2 .+ v_rho .^ 2))
        u_east = ifelse.(mask_rho .== 0, NaN32, u_rho .* cos.(grid_angle) .- v_rho .* sin.(grid_angle))
        v_north = ifelse.(mask_rho .== 0, NaN32, u_rho .* sin.(grid_angle) .+ v_rho .* cos.(grid_angle))

        (speed = slice_at_depth(z, speed, speed_depth),
         u_east = slice_at_depth(z, u_east, speed_depth),
         v_north = slice_at_depth(z, v_north, speed_depth),
         temp = slice_at_depth(z, temp, temp_depth))
    end
end

##
fig = Figure(size = (2000, 1050), fontsize = 20, figure_padding = (60, 10, 40, 10))
panel = 'a'
sp_speed = sp_temp = nothing
for (col, t) in enumerate(snap_times)
    println("loading ", t)
    s = load_snapshot(t)
    tstr = Dates.format(t, "yyyy-mm-dd HH:MM")

    ax = map_axis(fig[1, col]; title = "($panel) Speed at $(Int(-speed_depth)) m, $tstr",
                  show_x = false, show_y = col == 1)
    global sp_speed = plot_curvilinear!(ax, lon_rho, lat_rho, s.speed; colormap = :speed, colorrange = speed_range)
    contour!(ax, lon_rho, lat_rho, h; levels = bathy_levels, color = RGBf(0.3, 0.3, 0.3), linewidth = 1)
    add_quiver!(ax, s.u_east, s.v_north; scale = sc)
    add_mooring!(ax; scale = sc)
    global panel += 1

    ax = map_axis(fig[2, col]; title = "($(panel + 3)) Temperature at $(Int(-temp_depth)) m, $tstr",
                  show_y = col == 1)
    global sp_temp = plot_curvilinear!(ax, lon_rho, lat_rho, s.temp; colormap = :thermal, colorrange = temp_range)
    contour!(ax, lon_rho, lat_rho, h; levels = bathy_levels, color = RGBf(0.3, 0.3, 0.3), linewidth = 1)
    add_mooring!(ax; scale = sc)
end
Colorbar(fig[1, 5], sp_speed, label = "m/s")
Colorbar(fig[2, 5], sp_temp, label = "°C")
colgap!(fig.layout, 0)
rowgap!(fig.layout, 0)

outname = joinpath(figure_path, "snapshot_speed_temp_8panel.png")
save(outname, fig; px_per_unit = 2)
println("saved plot to ", outname)
