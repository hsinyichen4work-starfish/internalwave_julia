using NCDatasets, CairoMakie, Dates, Statistics
CairoMakie.activate!()
include("/home/hchen54/internalwave_julia/function/load_all_hpc.jl")
include("/home/hchen54/internalwave_julia/function/plotting_fun.jl")

grid_fname = "/expanse/lustre/projects/uso101/hchen54/input/grid/roms_grd_900m.nc"   # the grid_file listed in the .nc's global attributes
datadir = "/expanse/lustre/projects/uso101/hchen54/amazon_900m_vsponge_400/"   # HPC output dir — contains avg/dia/his/rst files mixed together
figure_path = "/home/hchen54/figure/dbry_0927_vsponge400/wavy_vorticity"   # separate from his_rejoint so the two scripts' skip checks don't interfere
mkpath(figure_path)

## GRID
mask_rho, lon_rho, lat_rho, h ,pm, pn, grid_angle = NCDataset(grid_fname) do ds
    ds["mask_rho"][:, :], ds["lon_rho"][:, :], ds["lat_rho"][:, :],
    ds["h"][:,:], ds["pm"][:,:] , ds["pn"][:,:], ds["angle"][:,:]
end
lon_rho[lon_rho .> 180] .-= 360   # convert 0-360 convention to -180/180 (east/west hemisphere)
_, _, mask_p = uvp_masks(mask_rho)
lon_psi, lat_psi = rho2p(lon_rho), rho2p(lat_rho)
# no native lon_psi/lat_psi in this grid file — approximate via corner averaging

lon_lim = extrema(lon_rho)   # shared x/y axis limits for every plot below
lat_lim = extrema(lat_rho)

bathy_levels = [500, 1000, 2000]

skip = 30   # downsample for legible quiver arrows — plotting all 686x856 would be unreadable


## SECTION LINE
# the grid is rotated ~34°, so a line parallel to the north/south boundaries
# is a single eta row (fixed j, all i), running west boundary -> east boundary.
# vorticity lives on the psi grid, so pick the psi row closest to the target
sec_lon, sec_lat = -42.15, 2.95   # point the section should pass through (2.95°N, 42.15°W)
i_sec, j_sec = Tuple(argmin((lon_psi .- sec_lon) .^ 2 .+ (lat_psi .- sec_lat) .^ 2))
sec_line_lon, sec_line_lat = lon_psi[:, j_sec], lat_psi[:, j_sec]
println("section: psi row j = $j_sec, nearest point i = $i_sec at ",
        round(sec_line_lat[i_sec], digits = 3), "°N, ", round(-sec_line_lon[i_sec], digits = 3), "°W")

# along-line distance (km) from the west end, flat-earth approximation —
# fine at 900 m spacing near the equator
dx_km = 111.32 .* cosd.(sec_line_lat[1:end-1]) .* diff(sec_line_lon)
dy_km = 110.57 .* diff(sec_line_lat)
sec_dist = [0.0; cumsum(sqrt.(dx_km .^ 2 .+ dy_km .^ 2))]
sec_target_km = sec_dist[i_sec]

# keep only the easternmost stretch of the line, where the wavy structures are;
# distances stay measured from the west boundary. set sec_len_km = Inf for the full line
sec_len_km = 100.0
sec_i = findall(sec_dist .>= sec_dist[end] - sec_len_km)   # psi i-indices kept along row j_sec
sec_line_lon, sec_line_lat, sec_dist = sec_line_lon[sec_i], sec_line_lat[sec_i], sec_dist[sec_i]
println("keeping i = $(first(sec_i))-$(last(sec_i)) ($(round(sec_dist[1], digits = 1))-$(round(sec_dist[end], digits = 1)) km)")

clim = 5e-5   # symmetric vorticity color limits (s⁻¹), shared by every plot below

t_ref = DateTime(1994, 1, 1, 0, 0, 0)   # ROMS ocean_time reference (seconds since)

sec_zmin = -500.0   # bottom of the depth-section plots (m); set to nothing to show full depth

###
## BATHYMETRY MAP — where the section line sits
h_masked = ifelse.(mask_rho .== 0, NaN, h)
fig = Figure(size = (800, 600))
ax = topdown_axis3(fig[1, 1]; title = "Bathymetry + section line (psi row j = $j_sec)")
xlims!(ax, lon_lim...)
ylims!(ax, lat_lim...)
sp = plot_curvilinear!(ax, lon_rho, lat_rho, h_masked; colormap = :deep)   # darker = deeper
Colorbar(fig[1, 2], sp, label = "depth (m)")
contour!(ax, lon_rho, lat_rho, h; levels = bathy_levels,
         color = RGBf(0.3, 0.3, 0.3), linewidth = 1)
# full row dashed, kept stretch solid, target point as a dot;
# z = 0 / overdraw so the flat surface doesn't hide them
lines!(ax, Point3f.(lon_psi[:, j_sec], lat_psi[:, j_sec], 0); color = :black,
       linewidth = 1, linestyle = :dash, overdraw = true)
lines!(ax, Point3f.(sec_line_lon, sec_line_lat, 0); color = :red, linewidth = 3, overdraw = true)
scatter!(ax, [Point3f(sec_lon, sec_lat, 0)]; marker = :circle, markersize = 10,
         color = :yellow, strokecolor = :black, strokewidth = 1, overdraw = true)
outname_bathy = joinpath(figure_path, "bathymetry_section_line.png")
save(outname_bathy, fig)
println("saved plot to ", outname_bathy)
fig

## HOVMÖLLER — 1 m vorticity along the section vs time, across many his files,
# to see whether the structures propagate across the line.
# Only reads the thin block each psi point on row j_sec actually needs —
# psi (i, j) = dv/dx - du/dy uses u at rows j, j+1 and v at row j (plus zeta
# at rows j, j+1 for the z-levels) — and only the top few sigma levels, since
# the 1 m slice never looks deeper. Way less I/O than the full 3D u/v above.
hov_files = sort(filter(f -> occursin(r"^roms_his\.\d{14}\.nc$", basename(f)), readdir(datadir, join = true)))
# hov_files = hov_files[1:30]   # e.g. a shorter window while testing

i1, i2 = first(sec_i), last(sec_i)
ir = i1:i2+1           # rho / v columns around the kept psi points
jr = j_sec:j_sec+1     # rho / u rows around psi row j_sec
nk_top = 10            # top sigma levels to read — plenty to bracket 1 m
h_sub, pm_sub, pn_sub = h[ir, jr], pm[ir, jr], pn[ir, jr]
mask_sec = mask_p[sec_i, j_sec]

hov_cols = Vector{Vector{Float32}}()
hov_time = DateTime[]
for f in hov_files
    NCDataset(f) do ds
        ot = ds["ocean_time"][:]
        N_f = ds.dim["s_rho"]
        ks = N_f-nk_top+1:N_f
        ts, tb, hcf = ds.attrib["theta_s"], ds.attrib["theta_b"], ds.attrib["hc"]
        zeta_s = ds["zeta"][ir, jr, :]
        u_s = ds["u"][i1:i2, jr, ks, :]
        v_s = ds["v"][ir, j_sec:j_sec, ks, :]   # range (not j_sec) keeps the eta dim
        vor_s = vorticity_cal(u_s, v_s, pm_sub, pn_sub)   # (ni, 1, nk_top, nt)
        for t in eachindex(ot)
            maximum(abs, zeta_s[:, :, t]) > 1e30 && continue   # incomplete/fill-valued record
            z_dum, _ = zlevs3(h_sub, zeta_s[:, :, t], ts, tb, hcf, N_f, "r", "new2008")
            z_s = rho2p(permutedims(z_dum, (2, 3, 1))[:, :, ks])   # (ni, 1, nk_top)
            vor_1m_sec = slice_at_depth(z_s, vor_s[:, :, :, t], -1.0)[:, 1]
            push!(hov_cols, ifelse.(mask_sec .== 0, NaN32, vor_1m_sec))
            push!(hov_time, t_ref + Millisecond(round(Int, ot[t] * 1000)))
        end
    end
    println("hovmöller: read ", basename(f))
end

# drop any duplicate records where consecutive files overlap, then sort by time
keep = unique(i -> hov_time[i], sortperm(hov_time))
hov_time = hov_time[keep]
hov = reduce(hcat, hov_cols[keep])   # (ni, ntime_total)
days = Dates.value.(hov_time .- hov_time[1]) ./ 8.64e7

fig = Figure(size = (1000, 1000))
ax = Axis(fig[1, 1]; title = "1 m vorticity along psi row j = $j_sec, $(Dates.format(hov_time[1], "yyyy-mm-dd HH:MM")) to $(Dates.format(hov_time[end], "yyyy-mm-dd HH:MM"))",
          xlabel = "Distance from west boundary along section (km)",
          ylabel = "Days since $(Dates.format(hov_time[1], "yyyy-mm-dd HH:MM"))")
hm = heatmap!(ax, sec_dist, days, hov; colormap = Reverse(:RdBu), colorrange = (-clim, clim))
#vlines!(ax, sec_target_km; color = :orange)
Colorbar(fig[1, 2], hm, label = "s⁻¹")
outname_hov = joinpath(figure_path, "vorticity_hovmoller_$(Dates.format(hov_time[1], "yyyymmdd_HHMM"))_$(Dates.format(hov_time[end], "yyyymmdd_HHMM")).png")
save(outname_hov, fig)
println("saved plot to ", outname_hov)

# same figure zoomed in on a time window (days since hov_time[1])
zoom_days = (10, 20)
zoom_t0, zoom_t1 = hov_time[1] + Hour(24 * zoom_days[1]), hov_time[1] + Hour(24 * zoom_days[2])
ax.title = "1 m vorticity along psi row j = $j_sec, $(Dates.format(zoom_t0, "yyyy-mm-dd HH:MM")) to $(Dates.format(zoom_t1, "yyyy-mm-dd HH:MM"))"
ylims!(ax, zoom_days...)
outname_hov_zoom = joinpath(figure_path, "vorticity_hovmoller_$(Dates.format(zoom_t0, "yyyymmdd_HHMM"))_$(Dates.format(zoom_t1, "yyyymmdd_HHMM")).png")
save(outname_hov_zoom, fig)
println("saved plot to ", outname_hov_zoom)

##
