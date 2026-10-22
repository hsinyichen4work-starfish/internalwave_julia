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
function add_mooring!(ax; labels = true)
    for g in unique(moor_group)
        k = findall(==(g), moor_group)
        st = get(moor_style, g, (marker = :rect, markersize = 6, color = :white))
        scatter!(ax, Point3f.(moor_lon[k], moor_lat[k], 0);
                 marker = st.marker, markersize = st.markersize, color = st.color,
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
