using NCDatasets, CairoMakie, Dates, Statistics, GibbsSeaWater, DSP
CairoMakie.activate!()
include("/home/hsinyi/Documents/Julia/function/load_all.jl")
include("/home/hsinyi/Documents/Julia/function/plotting_fun.jl")
##
grid_fname = "/home/hsinyi/roms_data/grid/roms_grd_900m.nc"   # the grid_file listed in the .nc's global attributes
grid100_fname = "/home/hsinyi/roms_data/grid/roms_grd_100m.nc" 
mooring_loc = "/home/hsinyi/roms_data/grid/roms_grd_900m_mor_edata.nc"   # HPC output dir — contains avg/dia/his/rst files mixed together
figure_path = "/home/hsinyi/figure/20261009_region/new_region"
parent_grid = "/home/mbui/ModelOutput/NCOM/grid/ohgrd_2.nc"
parent_27 = "/home/mbui/ModelOutput/NCOM/grid/ohgrd_1.nc"
mkpath(figure_path)
##
lon_rho_roms, lat_rho_roms = NCDataset(grid_fname) do ds
     ds["lon_rho"][:, :], ds["lat_rho"][:, :]
end
lon_rho_roms100, lat_rho_roms100 = NCDataset(grid100_fname) do ds
    ds["lon_rho"][:, :], ds["lat_rho"][:, :]
end
# ROMS lon is 0–360 (degrees East); NCOM is -180–180. Convert ROMS to -180–180
lon_rho_roms    = mod.(lon_rho_roms .+ 180, 360) .- 180
lon_rho_roms100 = mod.(lon_rho_roms100 .+ 180, 360) .- 180

lon_rho_27, lat_rho_27, h, mask  = NCDataset(parent_27) do ds
    ds["lon"][:, :], ds["lat"][:, :],ds["h"][:, :],ds["mask"][:, :]
end
# mask out land (mask: 0=land, 1=water); NaN so Makie leaves it blank
h = Float32.(coalesce.(h, NaN32))
h[coalesce.(mask, 0) .== 0] .= NaN32

lon_rho_9, lat_rho_9 ,h9, mask9 = NCDataset(parent_grid) do ds
    ds["lon"][:, :], ds["lat"][:, :], ds["h"][:, :],ds["mask"][:, :]
end
# mask out land (mask: 0=land, 1=water); NaN so Makie leaves it blank
h9 = Float32.(coalesce.(h9, NaN32))
h9[coalesce.(mask9, 0) .== 0] .= NaN32

## 900m ROMS grid extended by 222 cells (~200 km) on its west, east and north sides
# the grid is rotated: dim 1 (xi) runs ~ESE, dim 2 (eta) runs ~NNE, so
# west = i = 1, east = i = end, north = j = end. Spacing is near-uniform
# (899–900 m), so new cells are linearly extrapolated from the edge spacing.
function extend_grid(x; west = 0, east = 0, south = 0, north = 0)
    ni, nj = size(x)
    out = similar(x, ni + west + east, nj + south + north)
    I, J = west+1:west+ni, south+1:south+nj
    out[I, J] = x
    for k in 1:west;  out[west+1-k,  J] = x[1, :]   .- k .* (x[2, :]     .- x[1, :]);   end
    for k in 1:east;  out[west+ni+k, J] = x[end, :] .+ k .* (x[end, :]   .- x[end-1, :]); end
    y = out[:, J]   # rows now span the full extended i range
    for k in 1:south; out[:, south+1-k]  = y[:, 1]   .- k .* (y[:, 2]   .- y[:, 1]);   end
    for k in 1:north; out[:, south+nj+k] = y[:, end] .+ k .* (y[:, end] .- y[:, end-1]); end
    return out
end
n_ext = 222   # 222 × 0.9 km ≈ 200 km
lon_rho_roms_ext = extend_grid(Float64.(coalesce.(lon_rho_roms, NaN)); west = n_ext, east = n_ext, north = n_ext)
lat_rho_roms_ext = extend_grid(Float64.(coalesce.(lat_rho_roms, NaN)); west = n_ext, east = n_ext, north = n_ext)
# same extension, but on all four sides (south too)
lon_rho_roms_ext4 = extend_grid(Float64.(coalesce.(lon_rho_roms, NaN)); west = n_ext, east = n_ext, south = n_ext, north = n_ext)
lat_rho_roms_ext4 = extend_grid(Float64.(coalesce.(lat_rho_roms, NaN)); west = n_ext, east = n_ext, south = n_ext, north = n_ext)
# +222 cells west/east, none south, +666 cells (~600 km) north: the region for the 2700 m grid
n_ext_n = 666
lon_rho_roms_extN = extend_grid(Float64.(coalesce.(lon_rho_roms, NaN)); west = n_ext, east = n_ext, north = n_ext_n)
lat_rho_roms_extN = extend_grid(Float64.(coalesce.(lat_rho_roms, NaN)); west = n_ext, east = n_ext, north = n_ext_n)
# middle point of that region (average of the center rho points), to use as `mid` for the 2700 m grid
ni_N, nj_N = size(lon_rho_roms_extN)
ic, jc = (ni_N + 1) ÷ 2 : ni_N ÷ 2 + 1, (nj_N + 1) ÷ 2 : nj_N ÷ 2 + 1
mid_extN = [mean(lon_rho_roms_extN[ic, jc]), mean(lat_rho_roms_extN[ic, jc])]
# same physical size in 2700 m cells (the 900 m grid has nx = 684, ny = 854 interior cells)
nx_2700 = round(Int, (684 + 2n_ext) * 900 / 2700)
ny_2700 = round(Int, (854 + n_ext_n) * 900 / 2700)
println("extended region middle point: lon = ", mid_extN[1], ", lat = ", mid_extN[2])
println("same size at 2700 m: nx = ", nx_2700, ", ny = ", ny_2700)

##
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

##
moor_name  = sort(collect(keys(mor_loc)))
moor_lon   = [mor_loc[n][1] for n in moor_name]
moor_lat   = [mor_loc[n][2] for n in moor_name]
moor_group = [replace(n, r"\d+$" => "") for n in moor_name]   # "M3" -> "M", "CPIES7" -> "CPIES"

# one marker style per mooring type, so the three arrays can be told apart
moor_style = Dict("french" => (marker = :star5,   markersize = 11, color = :magenta),
                  "M"      => (marker = :diamond, markersize = 7,  color = :cyan),
                  "CPIES"  => (marker = :circle,  markersize = 6,  color = :yellow))

# name labels sit to the right of each marker by default; these few go on
# the left instead so neighboring labels don't run into each other
moor_label_left = ("french", "M1", "CPIES1", "CPIES5", "CPIES9")

# marks every mooring on the map; z = 0 puts them on the same flat plane as
# plot_curvilinear!'s surface, and overdraw keeps them from being hidden
# behind that surface. text! writes each mooring's own name next to its marker.
function add_mooring!(ax; labels = true, scale = 1)
    for g in unique(moor_group)
        k = findall(==(g), moor_group)
        st = get(moor_style, g, (marker = :rect, markersize = 6, color = :white))
        scatter!(ax, Point3f.(moor_lon[k], moor_lat[k], 0);
                 marker = st.marker, markersize = scale * st.markersize, color = st.color,
                 strokecolor = :black, strokewidth = 0.75, overdraw = true, label = g)
    end
    labels || return
    for (n, lon, lat) in zip(moor_name, moor_lon, moor_lat)
        left = n in moor_label_left
        kw = (text = n, fontsize = 7, font = :bold, overdraw = true,
              align = (left ? :right : :left, :center), offset = (left ? -5 : 5, 0))
        # white outline drawn first, underneath, so the black name stays
        # readable where it crosses dark bathymetry
        text!(ax, Point3f(lon, lat, 0); kw..., color = :white, strokecolor = :white, strokewidth = 2)
        text!(ax, Point3f(lon, lat, 0); kw..., color = :black)
    end
end

## bathymetry of the 27 km NCOM parent grid, with the mooring array on top
fig = Figure(; figkw(18, 17.2)...)
ax = topdown_axis3(fig[1, 1]; title = "Bathymetry")
# h is positive up, so -h is depth below the surface
sp = plot_curvilinear!(ax, lon_rho_27, lat_rho_27, -h;
                       colormap = :deep, colorrange = (0, 5500), nan_color = :gray80)
Colorbar(fig[1, 2], sp, label = "Depth (m)")
# outlines of the nested grids, drawn at z = 0 on the same flat plane as the surface
for (lon, lat, color, label) in ((lon_rho_9,       lat_rho_9,       :black, "900m resolution NCOM"),
                                 (lon_rho_roms,    lat_rho_roms,    :red,   "300m resolution ROMS"),
                                 (lon_rho_roms100, lat_rho_roms100, :orange, "100m resolution ROMS"))
    lon_b, lat_b = grid_boundary(lon, lat)
    lines!(ax, lon_b, lat_b, zeros(length(lon_b)); color = color, linewidth = 2, label = label)
end
# the whole array spans only ~2 degrees here, so the per-mooring names would
# pile up on each other; a legend by mooring type is used instead
add_mooring!(ax; labels = false)
Legend(fig[2, 1:2], ax; orientation = :horizontal, nbanks = 2, framevisible = false)
ax.protrusions = (30, 30, 50, 30)   # extra room at the bottom so the x label isn't clipped
figsave(joinpath(figure_path, "bathymetry_mooring.png"), fig)

## same map zoomed in to the 900m resolution NCOM grid, using its own bathymetry
fig = Figure(; figkw(18, 17.2)...)
ax = topdown_axis3(fig[1, 1]; title = "Bathymetry")
sp = plot_curvilinear!(ax, lon_rho_9, lat_rho_9, -h9;
                       colormap = :deep, colorrange = (0, 5500), nan_color = :gray80)
Colorbar(fig[1, 2], sp, label = "Depth (m)")
for (lon, lat, color, label) in ((lon_rho_roms,    lat_rho_roms,    :red,    "300m resolution ROMS"),
                                 (lon_rho_roms100, lat_rho_roms100, :orange, "100m resolution ROMS"))
    lon_b, lat_b = grid_boundary(lon, lat)
    lines!(ax, lon_b, lat_b, zeros(length(lon_b)); color = color, linewidth = 2, label = label)
end
add_mooring!(ax; labels = false)
Legend(fig[2, 1:2], ax; orientation = :horizontal, nbanks = 2, framevisible = false)
ax.protrusions = (30, 30, 50, 30)
figsave(joinpath(figure_path, "bathymetry_mooring_ncom900m.png"), fig)

## proposed extended 900m ROMS domain (+222 cells / ~200 km west/east/north) over the 27 km bathymetry
fig = Figure(; figkw(18, 17.2)...)
ax = topdown_axis3(fig[1, 1]; title = "Bathymetry, extended 900m ROMS domain")
sp = plot_curvilinear!(ax, lon_rho_27, lat_rho_27, -h;
                       colormap = :deep, colorrange = (0, 5500), nan_color = :gray80)
Colorbar(fig[1, 2], sp, label = "Depth (m)")
for (lon, lat, color, linestyle, label) in
        ((lon_rho_roms_ext4, lat_rho_roms_ext4, :green, :solid, "900m ROMS extended (+222 cells all sides)"),
         (lon_rho_roms_ext, lat_rho_roms_ext, :purple, :solid, "900m ROMS extended (+222 cells W/E/N)"),
         (lon_rho_9,       lat_rho_9,       :black,   :dash,  "900m resolution NCOM"),
         (lon_rho_roms,    lat_rho_roms,    :red,     :solid, "300m resolution ROMS"),
         (lon_rho_roms100, lat_rho_roms100, :orange,  :solid, "100m resolution ROMS"))
    lon_b, lat_b = grid_boundary(lon, lat)
    lines!(ax, lon_b, lat_b, zeros(length(lon_b)); color, linestyle, linewidth = 2, label)
end
add_mooring!(ax; labels = false)
Legend(fig[2, 1:2], ax; orientation = :horizontal, nbanks = 3, framevisible = false)
ax.protrusions = (30, 30, 50, 30)
figsave(joinpath(figure_path, "bathymetry_mooring_roms900m_ext.png"), fig)

## region for the 2700 m grid: 900m ROMS extended +222 cells W/E, +666 cells N, none S
fig = Figure(; figkw(18, 17.2)...)
ax = topdown_axis3(fig[1, 1]; title = "Bathymetry, region for the 2700 m grid")
sp = plot_curvilinear!(ax, lon_rho_27, lat_rho_27, -h;
                       colormap = :deep, colorrange = (0, 5500), nan_color = :gray80)
Colorbar(fig[1, 2], sp, label = "Depth (m)")
for (lon, lat, color, linestyle, label) in
        ((lon_rho_roms_extN, lat_rho_roms_extN, :purple, :solid, "900m ROMS extended (+$(n_ext) W/E, +$(n_ext_n) N)"),
         (lon_rho_9,         lat_rho_9,         :black,  :dash,  "900m resolution NCOM"),
         (lon_rho_roms,      lat_rho_roms,      :red,    :solid, "900m resolution ROMS"))
    lon_b, lat_b = grid_boundary(lon, lat)
    lines!(ax, lon_b, lat_b, zeros(length(lon_b)); color, linestyle, linewidth = 2, label)
end
add_mooring!(ax; labels = false)
scatter!(ax, Point3f(mid_extN[1], mid_extN[2], 0); marker = :xcross, markersize = 12, color = :red,
         strokecolor = :black, strokewidth = 0.75, overdraw = true, label = "middle point")
Legend(fig[2, 1:2], ax; orientation = :horizontal, nbanks = 2, framevisible = false)
ax.protrusions = (30, 30, 50, 30)
figsave(joinpath(figure_path, "bathymetry_mooring_roms900m_extN$(n_ext_n).png"), fig)

## test region: 900m ROMS extended +222 cells W/E/S plus another ~200 km on each
# of those sides (to take in the river to the south), and +666 cells N.
# The extra cells keep nx and ny multiples of 3, so the 2700 m grid has a whole number of points.
n_ext_we = n_ext + 222   # +199.8 km beyond the +222
n_ext_s  = n_ext + 223   # +200.7 km beyond the +222
lon_rho_roms_ext200 = extend_grid(Float64.(coalesce.(lon_rho_roms, NaN)); west = n_ext_we, east = n_ext_we, south = n_ext_s, north = n_ext_n)
lat_rho_roms_ext200 = extend_grid(Float64.(coalesce.(lat_rho_roms, NaN)); west = n_ext_we, east = n_ext_we, south = n_ext_s, north = n_ext_n)
ni_200, nj_200 = size(lon_rho_roms_ext200)
ic, jc = (ni_200 + 1) ÷ 2 : ni_200 ÷ 2 + 1, (nj_200 + 1) ÷ 2 : nj_200 ÷ 2 + 1
mid_ext200 = [mean(lon_rho_roms_ext200[ic, jc]), mean(lat_rho_roms_ext200[ic, jc])]
nx_900_200, ny_900_200 = 684 + 2n_ext_we, 854 + n_ext_s + n_ext_n
nx_2700_200, ny_2700_200 = nx_900_200 ÷ 3, ny_900_200 ÷ 3
println("test region middle point: lon = ", mid_ext200[1], ", lat = ", mid_ext200[2])
println("test region size at 900 m: ", nx_900_200, " × ", ny_900_200, "; at 2700 m: ", nx_2700_200, " × ", ny_2700_200)
println("test region extent: lon ", extrema(lon_rho_roms_ext200), ", lat ", extrema(lat_rho_roms_ext200))

# 900m ROMS domain extended on all four sides by +56 cells (~50 km)
n_red = 56
lon_rho_roms_red50 = extend_grid(Float64.(coalesce.(lon_rho_roms, NaN)); west = n_red, east = n_red, south = n_red, north = n_red)
lat_rho_roms_red50 = extend_grid(Float64.(coalesce.(lat_rho_roms, NaN)); west = n_red, east = n_red, south = n_red, north = n_red)

# test 300 m grid (cyan): each side halfway between the +50 km red domain and the orange box.
# Both grids share the same rotation (~-34°), so the boundaries are worked out in a local
# km frame (equirectangular around the red domain's center) with axes along the grid's xi/eta.
lon0, lat0 = mean(lon_rho_roms_red50), mean(lat_rho_roms_red50)
R_km = 6371.0
toxy(lon, lat) = (deg2rad.(lon .- lon0) .* R_km .* cosd(lat0), deg2rad.(lat .- lat0) .* R_km)
tolonlat(x, y) = (lon0 .+ rad2deg.(x ./ (R_km * cosd(lat0))), lat0 .+ rad2deg.(y ./ R_km))
jm = size(lon_rho_roms_red50, 2) ÷ 2
x1, y1 = toxy(lon_rho_roms_red50[1, jm], lat_rho_roms_red50[1, jm])
x2, y2 = toxy(lon_rho_roms_red50[end, jm], lat_rho_roms_red50[end, jm])
u = [x2 - x1, y2 - y1]; u ./= hypot(u...)   # along xi
v = [-u[2], u[1]]                            # along eta
function st_bounds(lon, lat)
    x, y = toxy(lon, lat)
    s, t = x .* u[1] .+ y .* u[2], x .* v[1] .+ y .* v[2]
    return extrema(s), extrema(t)
end
(s_r, t_r), (s_o, t_o) = st_bounds(lon_rho_roms_red50, lat_rho_roms_red50), st_bounds(lon_rho_roms100, lat_rho_roms100)
s_c = ((s_r[1] + s_o[1]) / 2, (s_r[2] + s_o[2]) / 2)
t_c = ((t_r[1] + t_o[1]) / 2, (t_r[2] + t_o[2]) / 2)
corner(s, t) = tolonlat(s * u[1] + t * v[1], s * u[2] + t * v[2])
cyan_c = [corner(s_c[1], t_c[1]), corner(s_c[2], t_c[1]), corner(s_c[2], t_c[2]), corner(s_c[1], t_c[2]), corner(s_c[1], t_c[1])]
lon_cyan, lat_cyan = first.(cyan_c), last.(cyan_c)
mid_cyan = corner(sum(s_c) / 2, sum(t_c) / 2)
nx_cyan, ny_cyan = round(Int, (s_c[2] - s_c[1]) / 0.3), round(Int, (t_c[2] - t_c[1]) / 0.3)
println("cyan 300 m grid: ", round(s_c[2] - s_c[1], digits = 1), " km × ", round(t_c[2] - t_c[1], digits = 1),
        " km -> nx × ny = ", nx_cyan, " × ", ny_cyan, "; middle point lon = ", mid_cyan[1], ", lat = ", mid_cyan[2])

# interior grid points (nx × ny, as passed to the grid build) = rho points minus the 2 edge points
npts(lon) = "$(size(lon, 1) - 2) × $(size(lon, 2) - 2)"
# the orange box redesigned at 300 m: the 100 m grid's footprint in 300 m cells
npts300 = "$(round(Int, (size(lon_rho_roms100, 1) - 2) * 100 / 300)) × $(round(Int, (size(lon_rho_roms100, 2) - 2) * 100 / 300))"
println("orange region at 300 m: ", npts300)
fig = Figure(; figkw(18, 17.2)...)
ax = topdown_axis3(fig[1, 1]; title = "Bathymetry, test region")
sp = plot_curvilinear!(ax, lon_rho_27, lat_rho_27, -h;
                       colormap = :deep, colorrange = (0, 5500), nan_color = :gray80)
Colorbar(fig[1, 2], sp, label = "Depth (m)")
for (lon, lat, color, linestyle, label) in
        ((lon_rho_roms_ext200, lat_rho_roms_ext200, :green, :solid, "2700m ROMS ($(nx_2700_200) × $(ny_2700_200) grid points)"),
         (lon_rho_roms_red50, lat_rho_roms_red50, :red,    :solid,  "900m ROMS, +$(round(Int, n_red * 0.9)) km all sides ($(npts(lon_rho_roms_red50)) grid points)"),
         (lon_rho_roms100,    lat_rho_roms100,    :orange, :solid, "100m resolution ROMS ($(npts(lon_rho_roms100)) grid points)"))
    lon_b, lat_b = grid_boundary(lon, lat)
    lines!(ax, lon_b, lat_b, zeros(length(lon_b)); color, linestyle, linewidth = 1, label)
end
lines!(ax, lon_cyan, lat_cyan, zeros(length(lon_cyan)); color = :blue, linewidth = 1,
       label = "300m ROMS, new ($(nx_cyan) × $(ny_cyan) grid points)")
add_mooring!(ax; labels = false, scale = 0.6)
Legend(fig[2, 1:2], ax; orientation = :horizontal, nbanks = 4, framevisible = false)
ax.protrusions = (30, 30, 50, 30)
figsave(joinpath(figure_path, "regiontest.png"), fig)

## grid design summary for the four regions on regiontest.png -> /home/hsinyi/data_notm/grid_design.txt
# angle of the grid's xi axis from east (deg), from the edge running through the middle row
function xi_angle(lon, lat)
    jm = size(lon, 2) ÷ 2
    dlon, dlat = lon[end, jm] - lon[1, jm], lat[end, jm] - lat[1, jm]
    return atand(dlat, dlon * cosd((lat[end, jm] + lat[1, jm]) / 2))
end
grid_center(lon, lat) = begin
    ni, nj = size(lon)
    ic, jc = (ni + 1) ÷ 2 : ni ÷ 2 + 1, (nj + 1) ÷ 2 : nj ÷ 2 + 1
    (mean(lon[ic, jc]), mean(lat[ic, jc]))
end
# corners in grid order: (xi, eta) = (1, 1), (end, 1), (end, end), (1, end)
grid_corners(lon, lat) = [(lon[i, j], lat[i, j]) for (i, j) in ((1, 1), (size(lon, 1), 1), (size(lon, 1), size(lon, 2)), (1, size(lon, 2)))]
rot_design = -34.21122878190815   # rot_ang from gid_middle(2) in function/input_nc (InputNC is not loaded here)
design = [
    (name = "2700m ROMS (green)", dx = 2700, nx = nx_2700_200, ny = ny_2700_200,
     ang = xi_angle(lon_rho_roms_ext200, lat_rho_roms_ext200), mid = Tuple(mid_ext200),
     corners = grid_corners(lon_rho_roms_ext200, lat_rho_roms_ext200),
     note = "900m ROMS grid extended +$(n_ext_we) cells W/E, +$(n_ext_s) S, +$(n_ext_n) N (900 m cells)"),
    (name = "900m ROMS, +50 km all sides (red)", dx = 900, nx = size(lon_rho_roms_red50, 1) - 2, ny = size(lon_rho_roms_red50, 2) - 2,
     ang = xi_angle(lon_rho_roms_red50, lat_rho_roms_red50), mid = grid_center(lon_rho_roms_red50, lat_rho_roms_red50),
     corners = grid_corners(lon_rho_roms_red50, lat_rho_roms_red50),
     note = "900m ROMS grid (roms_grd_900m.nc) extended +$(n_red) cells on all four sides"),
    (name = "300m ROMS, new (blue)", dx = 300, nx = nx_cyan, ny = ny_cyan,
     ang = atand(u[2], u[1]), mid = Tuple(mid_cyan),
     corners = [(lon_cyan[k], lat_cyan[k]) for k in 1:4],
     note = "each side halfway between the red and orange boxes; size estimated in a flat local km frame"),
    (name = "100m ROMS (orange)", dx = 100, nx = size(lon_rho_roms100, 1) - 2, ny = size(lon_rho_roms100, 2) - 2,
     ang = xi_angle(lon_rho_roms100, lat_rho_roms100), mid = grid_center(lon_rho_roms100, lat_rho_roms100),
     corners = grid_corners(lon_rho_roms100, lat_rho_roms100),
     note = "existing grid roms_grd_100m.nc"),
]
fmt(x) = @sprintf("%.6f", x)
open("/home/hsinyi/data_notm/grid_design.txt", "w") do io
    println(io, "# Grid design for the regions on regiontest.png (written by other/figure_of_region_new.jl, ", Dates.today(), ")")
    println(io, "# lon in degrees East (-180..180), lat in degrees North")
    println(io, "# rotation: angle of the grid's xi axis from east, measured from each outline (deg)")
    println(io, "# design rotation from gid_middle(2) (rot_ang the grid build uses): ", fmt(rot_design), " deg")
    println(io, "# corners: grid order (xi, eta) = (1,1) (end,1) (end,end) (1,end); outer rho points for grids, box corners for the new 300 m")
    println(io, "# nx, ny: interior grid points, as passed to the grid build")
    for d in design
        println(io)
        println(io, "[", d.name, "]")
        println(io, "resolution       = ", d.dx, " m")
        println(io, "nx x ny          = ", d.nx, " x ", d.ny, "   (", d.nx * d.ny, " points)")
        println(io, "rotation         = ", fmt(d.ang), " deg")
        println(io, "mid (lon, lat)   = ", fmt(d.mid[1]), ", ", fmt(d.mid[2]))
        for (k, c) in zip(("corner (1,1)  ", "corner (end,1)", "corner (end,end)", "corner (1,end)"), d.corners)
            println(io, rpad(k, 17), "= ", fmt(c[1]), ", ", fmt(c[2]))
        end
        println(io, "note             = ", d.note)
    end
end
println("wrote /home/hsinyi/data_notm/grid_design.txt")
