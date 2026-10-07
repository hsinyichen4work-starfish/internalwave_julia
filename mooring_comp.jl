using NCDatasets, CairoMakie, Dates, Statistics, GibbsSeaWater, DSP
CairoMakie.activate!()
include("/home/hsinyi/Documents/Julia/function/load_all.jl")
include("/home/hsinyi/Documents/Julia/function/plotting_fun.jl")
##
grid_fname = "/home/hsinyi/roms_data/grid/roms_grd_900m.nc"   # the grid_file listed in the .nc's global attributes
mooring_loc = "/home/hsinyi/roms_data/grid/roms_grd_900m_mor_edata.nc"   # HPC output dir — contains avg/dia/his/rst files mixed together
figure_path = "/home/hsinyi/figure/20260929_output/full3mon/mooring_with_cemes"
joint_ext_path = "/home/hsinyi/roms_data/mooring_ext_model/full_3mon/mooring_merged"
mooring_path = "/home/hsinyi/data_notm/French_mooring"
NCOM_mooring_path = "/home/hsinyi/roms_data/NCOM_mooring/NCOM_french.nc"
CMEMS_mooring_path = "/home/hsinyi/roms_data/GLOBAL_mooring/GLOBAL_french.nc"
mkpath(figure_path)
##
mask_rho, lon_rho, lat_rho, h ,pm, pn, grid_angle = NCDataset(grid_fname) do ds
    ds["mask_rho"][:, :], ds["lon_rho"][:, :], ds["lat_rho"][:, :],
    ds["h"][:,:], ds["pm"][:,:] , ds["pn"][:,:], ds["angle"][:,:]
end
lon_rho[lon_rho .> 180] .-= 360   # convert 0-360 convention to -180/180 (east/west hemisphere)
t_ref = DateTime(1994, 1, 1, 0, 0, 0)
# the mooring extract (french.nc) has no S-coordinate parameters, so take them from the initial file
ini_fname = "/home/hsinyi/roms_data/ini/roms_ini_900m2022082400.nc"
theta_s, theta_b, hc = NCDataset(ini_fname) do ds
    # theta_s/theta_b/hc live on a size-1 "one" dimension in this file, so [1]
    ds["theta_s"][1], ds["theta_b"][1], ds["hc"][1]
end
##
f_mor = NCDataset(mooring_loc) do ds
    show(ds)
    ds.attrib["french_mor_info"][:]
end
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
# --- bathymetry map with the mooring positions ---
# The grid is rotated, so lon_rho/lat_rho are 2-D and heatmap! (which needs 1-D regular axes) cannot be used:
# a flat surface! coloured by h draws every cell at its true lon/lat instead.
st = 4                                                    # plot every 4th point: much faster, looks the same
h_plot = Float32.(ifelse.(mask_rho .== 1, h, NaN))        # land -> NaN (left blank)
fig = Figure(size = (900, 800))
ax = Axis(fig[1, 1]; xlabel = "Longitude (°E)", ylabel = "Latitude (°N)", aspect = DataAspect(),
    title = "Bathymetry (ROMS 900 m grid) and mooring positions")
sf = surface!(ax, lon_rho[1:st:end, 1:st:end], lat_rho[1:st:end, 1:st:end], zeros(Float32, size(h_plot[1:st:end, 1:st:end]));
    color = h_plot[1:st:end, 1:st:end], colormap = :deep, shading = NoShading)
Colorbar(fig[1, 2], sf; label = "Depth (m)")
# isobaths; translate! lifts them above the surface so they are not hidden by it
ct = contour!(ax, lon_rho[1:st:end, 1:st:end], lat_rho[1:st:end, 1:st:end], h_plot[1:st:end, 1:st:end];
    levels = [100, 500, 1000], color = RGBf(0.4, 0.4, 0.4), linewidth = 1)
translate!(ct, 0, 0, 1)
# one colour per mooring type: french / M1–M4 / CPIES1–9
mor_groups = (("French", n -> n == "french", :red), ("M", n -> startswith(n, "M"), :orange),
              ("CPIES", n -> startswith(n, "CPIES"), :magenta))
for (label, ingroup, color) in mor_groups
    names = sort([n for n in keys(mor_loc) if ingroup(n)])
    isempty(names) && continue
    lons, lats = [mor_loc[n][1] for n in names], [mor_loc[n][2] for n in names]
    sc = scatter!(ax, lons, lats; color = color, strokecolor = :white, strokewidth = 1, markersize = 10, label = label)
    tx = text!(ax, lons, lats; text = names, offset = (5, 5), fontsize = 11, color = :white)
    translate!(sc, 0, 0, 2); translate!(tx, 0, 0, 2)
end
axislegend(ax; position = :rb)
outname = joinpath(figure_path, "bathymetry_moorings.png")
save(outname, fig)
println("saved ", outname)
fig

##
fname = joinpath(joint_ext_path, "french.nc")
ocean_time, zeta, temp, salt, u, v = NCDataset(fname) do ds
    # Julia reads dims reversed: zeta is (np, time), 3-D vars are (np, s_rho, time); np = 1 so drop it
    ds["ocean_time"][:], ds["zeta"][1, :],
    ds["temp"][1, :, :], ds["salt"][1, :, :], ds["u"][1, :, :], ds["v"][1, :, :]
end
model_time = t_ref .+ Millisecond.(round.(Int, ocean_time .* 1000))   # ocean_time is in SECONDS since t_ref
println(fname, " => ", model_time[1], " to ", model_time[end], ", size(temp) = ", size(temp))

##
# --- NCOM interpolated to the French mooring: hourly, 201 fixed z levels (0 to -1000 m every 5 m) ---
# 2-D vars come out as (z, time) = (201, nt). z is negative down, and index 1 is the deepest (-1000 m).
ncom_time, ncom_z, ncom_temp, ncom_salt, ncom_u, ncom_v, ncom_ssh = NCDataset(NCOM_mooring_path) do ds
    ds["time"][:],        # already decoded to DateTime (units are "hours since 2022-01-01")
    ds["z"][:],           # level depths (m, positive up)
    ds["temp"][:, :], ds["salt"][:, :],
    ds["u"][:, :], ds["v"][:, :],   # eastward / northward velocity (m/s), no rotation needed
    ds["ssh"][:]
end
println(NCOM_mooring_path, " => ", ncom_time[1], " to ", ncom_time[end], ", size(ncom_temp) = ", size(ncom_temp))

##
# --- CMEMS (GLOBAL_ANALYSISFORECAST_PHY_001_024) interpolated to the French mooring: 6-hourly, 30 z levels ---
# 2-D vars come out as (z, time) = (30, nt). z is negative down, and index 1 is the deepest (-380 m); the levels
# are the product's own, unevenly spaced (0.5 m apart at the surface, 60 m apart at the bottom).
# The file has temp, u and v only: no salinity and no sea surface height.
cmems_time, cmems_z, cmems_temp, cmems_u, cmems_v = NCDataset(CMEMS_mooring_path) do ds
    ds["time"][:],        # already decoded to DateTime (units are "hours since 2022-01-01")
    ds["z"][:],           # level depths (m, positive up)
    ds["temp"][:, :],     # potential temperature (°C)
    ds["u"][:, :], ds["v"][:, :]    # eastward / northward velocity (m/s), no rotation needed
end
println(CMEMS_mooring_path, " => ", cmems_time[1], " to ", cmems_time[end], ", size(cmems_temp) = ", size(cmems_temp))

##
# --- mooring CTD: 4 instruments, 6-min sampling, 2021-10-07 to 2024-06-02 ---
# 2-D vars come out as (nom_depth, time) = (4, nt). The file's _FillValue is NaN, which NCDatasets
# turns into `missing`, so nomissing(…, NaN) gives back plain Float64 arrays with NaN in the gaps.
fname = joinpath(mooring_path, "ctd_corrected_raw.nc")
ctd_time, nom_depth, ctd_temp, ctd_salt, ctd_CT, ctd_SA, ctd_pres, ctd_depth, ctd_rho = NCDataset(fname) do ds
    rd(name) = nomissing(ds[name][:, :], NaN)
    ds["time"][:],                         # already decoded to DateTime (units are "microseconds since …")
    nomissing(ds["nom_depth"][:], NaN),    # nominal instrument depths: 170, 220, 270, 315 m (labels only)
    rd("temperature"),                     # in-situ temperature (°C)
    rd("salinity"),                        # practical salinity (PSU) — compare with ROMS salt
    rd("conservative temperature"),        # TEOS-10 CT (°C)
    rd("absolute salinity"),               # TEOS-10 SA (g/kg)
    rd("pression"),                        # pressure (dbar)
    rd("depth"),                           # actual instrument depth (m, positive down) — use this, not nom_depth
    rd("density")                          # in-situ density (kg/m³)
end
println(fname, " => ", ctd_time[1], " to ", ctd_time[end], ", size(ctd_temp) = ", size(ctd_temp))

# --- mooring ADCP: upward-looking, 32 bins, 30-min sampling, 2021-10-07 to 2023-11-10 ---
# 2-D vars come out as (bin, time) = (32, nt). Bin 1 is the deepest (nearest the instrument).
fname = joinpath(mooring_path, "adcp_newalgo_raw_corr_angle.nc")
adcp_time, adcp_u, adcp_v, adcp_w, adcp_z, adcp_pg = NCDataset(fname) do ds
    rd(name) = nomissing(ds[name][:, :], NaN)
    ds["time"][:],
    rd("u"),     # eastward velocity (m/s)
    rd("v"),     # northward velocity (m/s)
    rd("w"),     # vertical velocity (m/s)
    rd("z"),     # bin depth (m, positive down), varies in time as the mooring moves
    rd("pg")     # percent good (0–100)
end
println(fname, " => ", adcp_time[1], " to ", adcp_time[end], ", size(adcp_u) = ", size(adcp_u))

##
# trim the mooring and NCOM records to the model period (safe to re-run: on already-trimmed data it keeps everything)
ctd_it  = findall(model_time[1] .<= ctd_time  .<= model_time[end])
adcp_it = findall(model_time[1] .<= adcp_time .<= model_time[end])
ncom_it = findall(model_time[1] .<= ncom_time .<= model_time[end])

ctd_time = ctd_time[ctd_it]
ctd_temp, ctd_salt, ctd_CT, ctd_SA, ctd_pres, ctd_depth, ctd_rho =
    (a[:, ctd_it] for a in (ctd_temp, ctd_salt, ctd_CT, ctd_SA, ctd_pres, ctd_depth, ctd_rho))

adcp_time = adcp_time[adcp_it]
adcp_u, adcp_v, adcp_w, adcp_z, adcp_pg =
    (a[:, adcp_it] for a in (adcp_u, adcp_v, adcp_w, adcp_z, adcp_pg))

ncom_time, ncom_ssh = ncom_time[ncom_it], ncom_ssh[ncom_it]
ncom_temp, ncom_salt, ncom_u, ncom_v =
    (a[:, ncom_it] for a in (ncom_temp, ncom_salt, ncom_u, ncom_v))

cmems_it = findall(model_time[1] .<= cmems_time .<= model_time[end])
cmems_time = cmems_time[cmems_it]
cmems_temp, cmems_u, cmems_v = (a[:, cmems_it] for a in (cmems_temp, cmems_u, cmems_v))

println("model: ", model_time[1], " to ", model_time[end], " (", length(model_time), " steps)")
println("NCOM : ", ncom_time[1], " to ", ncom_time[end], ", size(ncom_temp) = ", size(ncom_temp))
println("CMEMS: ", cmems_time[1], " to ", cmems_time[end], ", size(cmems_temp) = ", size(cmems_temp))
println("CTD  : ", ctd_time[1], " to ", ctd_time[end], ", size(ctd_temp) = ", size(ctd_temp))

# potential temperature (referenced to the surface) from the TEOS-10 CT and SA, to compare with the models'
# potential temperature. NaN in the gaps stays NaN. Here it is 0.03–0.05 °C colder than the in-situ temperature.
ctd_ptemp = gsw_pt_from_ct.(ctd_SA, ctd_CT)
println("ADCP : ", adcp_time[1], " to ", adcp_time[end], ", size(adcp_u) = ", size(adcp_u))

##
# Put mooring / ROMS / NCOM in one common frame so they can be compared directly:
#   z          : depth below the instantaneous sea surface (m, POSITIVE DOWN), size (nz, nt)
#   row order  : shallowest first, so z increases with the row index in every structure
#   u, v       : TRUE eastward / northward velocity (m/s) in all three
#                ROMS   : stored along the grid axes -> rotated below by the grid angle (-34.2°)
#                NCOM   : already east/north in the files, although the NCOM grid itself is turned ~-27°
#                         (checked against geostrophic flow from NCOM ssh) -> no rotation
#                mooring: ADCP u/v taken as east/north as delivered ("corr_angle" file) -> no rotation
#   temp, salt : °C / PSU
#   zeta       : sea surface height (m), models only; real height above the resting level is zeta - z
# Each structure keeps its own time axis (ROMS 10.5 min, NCOM 1 h, CTD 6 min, ADCP 30 min).

# --- ROMS ---
# french_mor = [i, j, angle]: fractional grid index of the mooring point and the angle its u/v are output at.
# That angle equals the local grid angle, so u/v in french.nc are still along the grid (xi/eta) directions.
i_mor, j_mor, ang_mor = NCDataset(mooring_loc) do ds
    ds["french_mor"][1, :]
end
# water depth at the mooring point: bilinear in h. The Julia (1-based) rho index is the object index + 1.5
# (checked: lon_rho/lat_rho interpolated this way give exactly -45.13, 3.95)
h_mor = let x = i_mor + 1.5, y = j_mor + 1.5
    i0, j0 = floor(Int, x), floor(Int, y)
    cx, cy = x - i0, y - j0
    (1 - cx) * (1 - cy) * h[i0, j0] + cx * (1 - cy) * h[i0+1, j0] +
    (1 - cx) * cy * h[i0, j0+1] + cx * cy * h[i0+1, j0+1]
end
N = size(temp, 1)
nt = length(model_time)
# zlevs3 wants 2-D h and zeta, so treat time as the second horizontal dimension: z_r is (N, 1, nt)
z_r, Cs = zlevs3(fill(h_mor, 1, nt), reshape(zeta, 1, nt), theta_s, theta_b, hc, N, "r", "new2008")
z_r = z_r[:, 1, :]                       # (N, nt), negative down, measured from the resting level
flipz(a) = reverse(a, dims=1)            # ROMS, NCOM and ADCP are stored deepest-first; make them surface-first
roms = (
    time = model_time,
    z    = flipz(zeta' .- z_r),          # depth below the moving surface
    u    = flipz(u .* cos(ang_mor) .- v .* sin(ang_mor)),   # rotate grid-aligned u/v to east/north
    v    = flipz(u .* sin(ang_mor) .+ v .* cos(ang_mor)),
    temp = flipz(temp),                  # potential temperature
    salt = flipz(salt),
    zeta = zeta,
)

# --- NCOM ---
# ncom_z are the nominal level depths (0, -5, …, -1000 m); take them as depth below the surface.
# The surface moving by ssh (up to ±1.4 m) is ignored here: it is smaller than the 5 m level spacing.
ncom = (
    time = ncom_time,
    z    = repeat(flipz(abs.(ncom_z)), 1, length(ncom_time)),   # (201, nt), same at every time
    u    = flipz(ncom_u),
    v    = flipz(ncom_v),
    temp = flipz(ncom_temp),
    salt = flipz(ncom_salt),
    zeta = ncom_ssh,
)

# --- CMEMS ---
# cmems_z are the product's level depths (-0.5 … -380 m); take them as depth below the surface, like NCOM.
# No salt and no zeta fields: the file does not have them.
cmems = (
    time = cmems_time,
    z    = repeat(flipz(abs.(cmems_z)), 1, length(cmems_time)),   # (30, nt), same at every time
    u    = flipz(cmems_u),
    v    = flipz(cmems_v),
    temp = flipz(cmems_temp),            # potential temperature, like ROMS
)

# --- mooring ---
# T/S come from the CTDs and u/v from the ADCP, on different times and depths, so each pair has its own axes
mooring = (
    time_ts = ctd_time,
    z_ts    = ctd_depth,                 # (4, nt), already positive down and shallowest-first
    temp    = ctd_ptemp,                 # potential temperature, like the models
    salt    = ctd_salt,
    time_uv = adcp_time,
    z_uv    = flipz(adcp_z),             # (32, nt); the first rows are the bins with no data (NaN)
    u       = flipz(adcp_u),
    v       = flipz(adcp_v),
)

println("h at mooring = ", round(h_mor, digits=1), " m, grid angle = ", round(rad2deg(ang_mor), digits=2), "°")
for (name, s) in (("roms", roms), ("ncom", ncom), ("cmems", cmems))
    println(name, ": size(z) = ", size(s.z), ", z from ", round(minimum(s.z), digits=1), " to ", round(maximum(s.z), digits=1), " m")
end
println("mooring: size(z_ts) = ", size(mooring.z_ts), ", size(z_uv) = ", size(mooring.z_uv))

##
# Put a (nz, nt) field onto one regular depth grid zg (m, positive down) by linear interpolation in depth,
# column by column. Depths outside the span of valid data in a column stay NaN (no extrapolation), so for
# the mooring only the range between the shallowest and deepest working instrument is filled.
function to_zgrid(z, f, zg)
    out = fill(NaN, length(zg), size(f, 2))
    for t in axes(f, 2)
        good = findall(.!isnan.(z[:, t]) .& .!isnan.(f[:, t]))
        length(good) < 2 && continue
        zt, ft = z[good, t], f[good, t]          # z increases with the row index in every structure
        for (k, d) in enumerate(zg)
            (isnan(d) || d < zt[1] || d > zt[end]) && continue
            i = clamp(searchsortedlast(zt, d), 1, length(zt) - 1)
            w = (d - zt[i]) / (zt[i+1] - zt[i])
            out[k, t] = (1 - w) * ft[i] + w * ft[i+1]
        end
    end
    return out
end

# time axis as days since the start of the model run, labelled with the real dates
t0 = model_time[1]
tday(t) = Dates.value.(t .- t0) ./ 86_400_000
date_ticks = DateTime(2022, 8, 25):Day(5):model_time[end]
xticks = (tday(date_ticks), Dates.format.(date_ticks, "mm/dd"))


# Model field AT the mooring CTDs: for every model time, take the depth each CTD had at that moment
# (linear in time between the 6-min CTD samples) and interpolate the model profile to that depth.
# Returns (4, nt_model); NaN where the instrument has no data, so the lines break exactly where the mooring's do.
function at_mooring_depth(s, f)
    out = fill(NaN, size(mooring.z_ts, 1), length(s.time))
    tm, ts = tday(mooring.time_ts), tday(s.time)
    for (n, t) in enumerate(ts)
        j = searchsortedlast(tm, t)
        (j < 1 || j >= length(tm)) && continue
        w = (t - tm[j]) / (tm[j+1] - tm[j])
        d = (1 - w) .* mooring.z_ts[:, j] .+ w .* mooring.z_ts[:, j+1]   # instrument depths at this model time
        out[:, n] = to_zgrid(s.z[:, n:n], f[:, n:n], d)
    end
    return out
end

##
# --- the two figure types, reused for temp / salt / u / v ---
site = "French mooring (3.95°N, 45.13°W)"
src_colors = (mooring = :black, roms = "#2a78d6", ncom = "#eb6834", cmems = "#2e9d5b")   # same colour = same data source in every line plot

# Time-vs-depth heatmaps, one panel per data source, sharing the axes and the colour scale.
# panels = ((title, time, z, field), …) with z and field of size (nz, nt); zlim = (shallow, deep) in m.
function plot_time_depth(panels; zlim, clim, colormap, levels, cblabel, title, fname,
                         contour_color = (:white, 0.6), ctd_lines = false, contours = true, zstep = 2.0)
    zg = zlim[1]:zstep:zlim[2]             # depths the panels are drawn on (fields are interpolated to them)
    fig = Figure(size = (1100, 900))
    axs, hms = Axis[], []
    for (row, (ptitle, time, z, f)) in enumerate(panels)
        ax = Axis(fig[row, 1]; title = ptitle, titlealign = :left, ylabel = "Depth (m)",
            yreversed = true, xticks = xticks, xgridvisible = false, ygridvisible = false)
        fg = to_zgrid(z, f, zg)
        push!(hms, heatmap!(ax, tday(time), zg, fg'; colormap = colormap, colorrange = clim))
        contours && contour!(ax, tday(time), zg, fg'; levels = levels, color = contour_color, linewidth = 0.8)
        push!(axs, ax)
    end
    if ctd_lines                         # where the CTDs actually were (first panel = mooring)
        for k in axes(mooring.z_ts, 1)
            lines!(axs[1], tday(mooring.time_ts), mooring.z_ts[k, :]; color = (:black, 0.7), linewidth = 0.6)
        end
    end
    linkaxes!(axs...)
    xlims!(axs[end], tday(model_time[1]), tday(model_time[end]))
    ylims!(axs[end], zlim[2], zlim[1])
    hidexdecorations!.(axs[1:end-1]; ticks = false)
    axs[end].xlabel = "Date (2022)"
    Colorbar(fig[1:length(panels), 2], hms[1]; label = cblabel, ticks = levels)
    Label(fig[0, :], title, fontsize = 18)
    outname = joinpath(figure_path, fname)
    save(outname, fig)
    println("saved ", outname)
    return fig
end

# Line plots, one panel per depth, one line per data source.
# sources = ((label, time, field, color), …) with field of size (npanel, nt), each on its own time axis.
function plot_lines(panel_titles, sources; ylabel, title, fname, zeroline = false)
    fig = Figure(size = (1100, 1100))
    axs = Axis[]
    for (k, ptitle) in enumerate(panel_titles)
        ax = Axis(fig[k, 1]; title = ptitle, titlealign = :left, ylabel = ylabel, xticks = xticks)
        zeroline && hlines!(ax, 0; color = (:black, 0.4), linewidth = 0.8)
        for (label, time, f, color) in sources
            lines!(ax, tday(time), f[k, :]; color = color, linewidth = 1, label = label)
        end
        push!(axs, ax)
    end
    linkxaxes!(axs...)                   # y is left free: each depth has its own range
    xlims!(axs[end], tday(model_time[1]), tday(model_time[end]))
    hidexdecorations!.(axs[1:end-1]; ticks = false, grid = false)
    axs[end].xlabel = "Date (2022)"
    n = length(panel_titles)
    Legend(fig[n+1, 1], axs[1]; orientation = :horizontal, framevisible = false, tellwidth = false)
    Label(fig[0, :], title, fontsize = 18, tellwidth = false)
    outname = joinpath(figure_path, fname)
    save(outname, fig)
    println("saved ", outname)
    return fig
end

##
# --- filtering: low-pass, high-pass and envelope of the high-pass, reused for every variable ---
Tcut = 28.0                # low/high-pass cutoff period (HOURS)
Tenv = 60.0                # low-pass cutoff period of the envelope (HOURS)
filt_order = 4

# Butterworth + filtfilt needs an evenly sampled series with no NaN, so for each row of a (npanel, nt) field:
# keep only the span between the first and last valid sample, fill the gaps inside it by linear interpolation
# in time, filter, then put NaN back where the data were missing. Values next to a long gap are less reliable.
# kind = "low" or "high" with Tcut one cutoff period, or kind = "band" with Tcut = (shortest, longest) period.
function filter_rows(time, f, Tcut, kind; N = filt_order)
    dt = median(diff(tday(time))) * 24             # sampling interval (hours), same unit as Tcut
    out = fill(NaN, size(f))
    for k in axes(f, 1)
        good = findall(!isnan, f[k, :])
        length(good) < 2 && continue
        y = f[k, good[1]:good[end]]
        for (a, b) in zip(good[1:end-1] .- good[1] .+ 1, good[2:end] .- good[1] .+ 1)   # fill the gaps
            for i in a+1:b-1
                y[i] = y[a] + (y[b] - y[a]) * (i - a) / (b - a)
            end
        end
        yf = kind == "band" ? bandpass_butter(y, Tcut[1], Tcut[2], dt, N) : lowhighpass_butter(y, Tcut, dt, N, kind)
        out[k, good] = yf[good .- good[1] .+ 1]
    end
    return out
end

# Envelope of the high-passed signal: sqrt( lowpass( hp^2 ) ).
# This is the running RMS amplitude of the high-frequency signal (a pure sine of amplitude A gives A/√2).
# The low-pass of hp^2 can ring slightly below zero, so it is clamped at 0 before the square root.
function hp_envelope(time, f)
    hp = filter_rows(time, f, Tcut, "high")
    return sqrt.(max.(filter_rows(time, hp .^ 2, Tenv, "low"), 0))
end

# The three filtered versions of a plot_lines figure: low-passed, high-passed, envelope of the high-pass.
# sources = the same ((label, time, field, color), …) as plot_lines; title and fname are the stems of the
# unfiltered figure, e.g. "Temperature at the CTDs" and "temp_lines_at_ctd".
# mask = true: the FIRST source is the reference (the mooring). Every source is filtered on its full series, then
# the others are blanked wherever the filtered reference is NaN (see mask_like, defined in the velocity section).
function plot_filtered(panel_titles, sources; ylabel, title, fname, mask = false)
    tc, te = round(Int, Tcut), round(Int, Tenv)
    function apply(g)
        out = [(label, time, g(time, f), color) for (label, time, f, color) in sources]
        if mask
            _, tref, fref, _ = out[1]
            out = [n == 1 ? o : (o[1], o[2], mask_like(o[2], o[3], tref, fref), o[4]) for (n, o) in enumerate(out)]
        end
        return Tuple(out)
    end
    plot_lines(panel_titles, apply((t, f) -> filter_rows(t, f, Tcut, "low")); ylabel = ylabel,
        title = "$title, $tc h low-passed, $site", fname = "$(fname)_lowpass$(tc)h.png")
    plot_lines(panel_titles, apply((t, f) -> filter_rows(t, f, Tcut, "high")); ylabel = ylabel, zeroline = true,
        title = "$title, $tc h high-passed, $site", fname = "$(fname)_highpass$(tc)h.png")
    plot_lines(panel_titles, apply(hp_envelope); ylabel = "Envelope: $ylabel",
        title = "$title, envelope of the $tc h high-pass ($te h low-pass), $site",
        fname = "$(fname)_hp$(tc)h_envelope$(te)h.png")
end

##
# ===== temperature and salinity: compared at the 4 CTDs =====
ctd_titles = map(axes(mooring.z_ts, 1)) do k
    zk = filter(!isnan, mooring.z_ts[k, :])       # depths this instrument really had during the period
    "CTD $(round(Int, nom_depth[k])) m nominal — actual depth $(round(Int, minimum(zk)))–$(round(Int, maximum(zk))) m"
end
zlim_ts = (150, 400)       # depth range of the heatmaps (m) — the range the mooring CTDs cover

# --- temperature ---
plot_time_depth((
        ("Mooring CTD (potential T, interpolated between the instruments; lines = instrument depths)",
            mooring.time_ts, mooring.z_ts, mooring.temp),
        ("ROMS 900 m (potential T)", roms.time, roms.z, roms.temp),
        ("NCOM", ncom.time, ncom.z, ncom.temp),
        ("CMEMS", cmems.time, cmems.z, cmems.temp));
    zlim = zlim_ts, clim = (8, 20), levels = 8:2:20, colormap = :thermal, ctd_lines = true,
    cblabel = "Temperature (°C)", title = "Temperature at the $site", fname = "temp_time_depth.png")

roms_temp_m = at_mooring_depth(roms, roms.temp)
ncom_temp_m = at_mooring_depth(ncom, ncom.temp)
cmems_temp_m = at_mooring_depth(cmems, cmems.temp)
plot_lines(ctd_titles, (
        ("Mooring CTD (potential T)", mooring.time_ts, mooring.temp, src_colors.mooring),
        ("ROMS 900 m (potential T)", roms.time, roms_temp_m, src_colors.roms),
        ("NCOM", ncom.time, ncom_temp_m, src_colors.ncom),
        ("CMEMS", cmems.time, cmems_temp_m, src_colors.cmems));
    ylabel = "Temperature (°C)", title = "Temperature at the CTDs, $site", fname = "temp_lines_at_ctd.png")
plot_lines(ctd_titles, (
#        ("Mooring CTD (potential T)", mooring.time_ts, mooring.temp, src_colors.mooring),
        ("ROMS 900 m (potential T)", roms.time, roms_temp_m, src_colors.roms),
        ("NCOM", ncom.time, ncom_temp_m, src_colors.ncom));
    ylabel = "Temperature (°C)", title = "Temperature at the CTDs, $site", fname = "temp_lines_at_ctd_model.png")

# plot_filtered(ctd_titles, (
#         ("Mooring CTD (potential T)", mooring.time_ts, mooring.temp, src_colors.mooring),
#         ("ROMS 900 m (potential T)", roms.time, roms_temp_m, src_colors.roms),
#         ("NCOM", ncom.time, ncom_temp_m, src_colors.ncom),
#         ("CMEMS", cmems.time, cmems_temp_m, src_colors.cmems));
#     ylabel = "Temperature (°C)", title = "Temperature at the CTDs", fname = "temp_lines_at_ctd")

# --- temperature at FIXED depths ---
# The CTDs move up and down with the mooring, so a series "at the CTD" mixes real changes in time with the
# instrument moving through the vertical gradient. Here every source is put on the same fixed depths first
# (linear in depth, each on its own time axis) and filtered after. Result: temp_z.mooring / .roms / .ncom / .cmems,
# each (length(ts_depths), nt). For the mooring this is an interpolation between the instruments just above and
# below, and it is NaN whenever the depth is outside the range the working CTDs span at that moment.
ts_depths = [200.0, 250.0, 300.0]
ts_titles = ["$(round(Int, d)) m" for d in ts_depths]
temp_z = (mooring = to_zgrid(mooring.z_ts, mooring.temp, ts_depths),
          roms    = to_zgrid(roms.z, roms.temp, ts_depths),
          ncom    = to_zgrid(ncom.z, ncom.temp, ts_depths),
          cmems   = to_zgrid(cmems.z, cmems.temp, ts_depths))
for (k, d) in enumerate(ts_depths)
    println("mooring T at ", round(Int, d), " m: ", round(100 * count(!isnan, temp_z.mooring[k, :]) / size(temp_z.mooring, 2), digits = 1), " % of the samples valid")
end

temp_z_src = (("Mooring CTD (potential T)", mooring.time_ts, temp_z.mooring, src_colors.mooring),
              ("ROMS 900 m (potential T)", roms.time, temp_z.roms, src_colors.roms),
              ("NCOM", ncom.time, temp_z.ncom, src_colors.ncom),
              ("CMEMS", cmems.time, temp_z.cmems, src_colors.cmems))
plot_lines(ts_titles, temp_z_src;
    ylabel = "Temperature (°C)", title = "Temperature at fixed depths, $site", fname = "temp_lines_at_depth.png")
plot_filtered(ts_titles, temp_z_src;
    ylabel = "Temperature (°C)", title = "Temperature at fixed depths", fname = "temp_lines_at_depth")


# --- filtered versions, kept so they can be called and plotted directly ---
# temp_zf.mooring / .roms / .ncom / .cmems, each with time and the (length(ts_depths), nt) matrices
#   raw : unfiltered          lp : Tcut low-pass          hp : Tcut high-pass
#   env : envelope of the high-pass, sqrt( lowpass( hp^2 ) ) with the Tenv low-pass
# e.g. temp_zf.roms.lp[2, :] is the low-passed ROMS temperature at ts_depths[2]. Each source is still on its own time axis.
src_time = (mooring = mooring.time_ts, roms = roms.time, ncom = ncom.time, cmems = cmems.time)
temp_zf = map(src_time, temp_z) do t, f
    (time = t, raw = f, lp = filter_rows(t, f, Tcut, "low"), hp = filter_rows(t, f, Tcut, "high"), env = hp_envelope(t, f))
end

# Average a (nrow, nt) field into time bins of length dt (any Dates period: Day(1), Hour(6), …), NaN ignored.
# The bins start at t_start and are the same for every source, so the results share one time axis and can be
# compared point by point. Returns (bin-centre times, (nrow, nbin) matrix). A bin is NaN when it holds fewer than
# minfrac of the samples a full bin would have (gaps, or the partly covered first/last bin).
function time_average(time, f, dt; t_start = floor(model_time[1], Day), t_end = model_time[end], minfrac = 0.5)
    ms = Dates.toms(dt)
    nb = floor(Int, Dates.value(t_end - t_start) / ms) + 1
    tb = t_start .+ Millisecond.(round.(Int, ((1:nb) .- 0.5) .* ms))
    nfull = ms / median(Dates.value.(diff(time)))          # samples in a full bin
    acc, cnt = zeros(size(f, 1), nb), zeros(Int, size(f, 1), nb)
    for (j, t) in enumerate(time)
        b = floor(Int, Dates.value(t - t_start) / ms) + 1
        1 <= b <= nb || continue
        for k in axes(f, 1)
            isnan(f[k, j]) && continue
            acc[k, b] += f[k, j]
            cnt[k, b] += 1
        end
    end
    out = acc ./ cnt
    out[cnt .< max(1, minfrac * nfull)] .= NaN
    return tb, out
end

# the same fields on a common time step: temp_za.roms.lp etc., all sources now (length(ts_depths), nbin) on temp_za.roms.time
avg_dt = Day(1)
temp_za = map(temp_zf) do s
    avg(f) = time_average(s.time, f, avg_dt)[2]
    (time = time_average(s.time, s.raw, avg_dt)[1], raw = avg(s.raw), lp = avg(s.lp), hp = avg(s.hp), env = avg(s.env))
end

# Scatter of each model against the mooring: one row per model, one column per depth, with the 1:1 line.
# time = the common time axis; x = (npanel, nt) mooring field; ys = ((label, field, color), …) with fields of the
# same size and on the same times as x. The points are coloured by date (one colour scale for the whole figure), and
# each panel's title takes the model's line colour. Only the times where both are valid are drawn; r is their
# correlation and N their number. xname = what x is (the x-axis label), the mooring unless said otherwise.
# angle = true is for directions in degrees (0–360): the axes are fixed to 0–360, and instead of r (meaningless for
# an angle, since 359° and 1° are neighbours) the title gives the mean absolute angular difference, taken the short
# way round. Points in the top-left and bottom-right corners are therefore close matches too, not outliers.
function plot_scatter(panel_titles, time, x, ys; unit, title, fname, xname = "Mooring", angle = false, markersize = 9)
    td = tday(time)
    crange = (tday(model_time[1]), tday(model_time[end]))
    fig = Figure(size = (370 * length(panel_titles) + 90, 370 * length(ys) + 60))
    for (row, (label, y, color)) in enumerate(ys), (k, ptitle) in enumerate(panel_titles)
        good = findall(.!isnan.(x[k, :]) .& .!isnan.(y[k, :]))
        xs, yv = x[k, good], y[k, good]
        stat = isempty(good) ? "no data" :
               angle ? "mean |Δ| = $(round(Int, mean(abs.(mod.(yv .- xs .+ 180, 360) .- 180))))°" :
               "r = $(length(good) > 2 ? round(cor(xs, yv), digits = 2) : NaN)"
        ax = Axis(fig[row, k]; title = "$label, $ptitle: $stat, N = $(length(good))", titlecolor = color,
            titlealign = :left, aspect = 1,
            xlabel = row == length(ys) ? "$xname ($unit)" : "", ylabel = "$label ($unit)")
        isempty(good) && continue
        lo, hi = angle ? (0.0, 360.0) : (min(minimum(xs), minimum(yv)), max(maximum(xs), maximum(yv)))
        pad = angle ? 0.0 : max(0.05 * (hi - lo), 1e-6)      # never zero, so the limits stay valid for a constant series
        angle && (ax.xticks = 0:90:360; ax.yticks = 0:90:360)
        lines!(ax, [lo - pad, hi + pad], [lo - pad, hi + pad]; color = (:black, 0.5), linewidth = 1)
        scatter!(ax, xs, yv; color = td[good], colormap = :viridis, colorrange = crange,
            markersize = markersize, strokewidth = markersize > 6 ? 0.5 : 0, strokecolor = :black)
        limits!(ax, lo - pad, hi + pad, lo - pad, hi + pad)
    end
    cb_dates = DateTime(2022, 9, 1):Day(15):model_time[end]
    Colorbar(fig[1:length(ys), length(panel_titles) + 1]; colormap = :viridis, limits = crange, label = "Date (2022)",
        ticks = (tday(cb_dates), Dates.format.(cb_dates, "mm/dd")))
    Label(fig[0, :], title, fontsize = 18, tellwidth = false)
    outname = joinpath(figure_path, fname)
    save(outname, fig)
    println("saved ", outname)
    return fig
end

let tc = round(Int, Tcut), te = round(Int, Tenv), dtlab = string(avg_dt)
    for (fld, long) in ((:raw, "unfiltered"), (:lp, "$tc h low-passed"), (:env, "envelope of the $tc h high-pass ($te h low-pass)"))
        plot_scatter(ts_titles, temp_za.mooring.time, getproperty(temp_za.mooring, fld), (
                ("ROMS 900 m", getproperty(temp_za.roms, fld), src_colors.roms),
                ("NCOM", getproperty(temp_za.ncom, fld), src_colors.ncom),
                ("CMEMS", getproperty(temp_za.cmems, fld), src_colors.cmems));
            unit = "°C", title = "Temperature at fixed depths, $long, $dtlab means: models vs mooring",
            fname = "temp_scatter_at_depth_$(fld).png")
    end
end

# --- ROMS against NCOM: the same scatter with NCOM on x instead of the mooring ---
# Both models have no gaps, so every day of the run is a point here (the mooring figures only have the days
# the CTDs cover the depth).
let tc = round(Int, Tcut), te = round(Int, Tenv), dtlab = string(avg_dt)
    for (fld, long) in ((:raw, "unfiltered"), (:lp, "$tc h low-passed"), (:env, "envelope of the $tc h high-pass ($te h low-pass)"))
        plot_scatter(ts_titles, temp_za.ncom.time, getproperty(temp_za.ncom, fld), (
                ("ROMS 900 m", getproperty(temp_za.roms, fld), src_colors.roms),);
            xname = "NCOM", unit = "°C", title = "Temperature at fixed depths, $long, $dtlab means: ROMS vs NCOM",
            fname = "temp_scatter_at_depth_roms_ncom_$(fld).png")
    end
end

# # --- salinity ---
# plot_time_depth((
#         ("Mooring CTD (interpolated between the instruments; lines = instrument depths)",
#             mooring.time_ts, mooring.z_ts, mooring.salt),
#         ("ROMS 900 m", roms.time, roms.z, roms.salt),
#         ("NCOM", ncom.time, ncom.z, ncom.salt));
#     zlim = zlim_ts, clim = (34.8, 36.0), levels = 34.8:0.2:36.0, colormap = :viridis, ctd_lines = true,
#     cblabel = "Salinity (PSU)", title = "Salinity at the $site", fname = "salt_time_depth.png")

# roms_salt_m = at_mooring_depth(roms, roms.salt)
# ncom_salt_m = at_mooring_depth(ncom, ncom.salt)
# plot_lines(ctd_titles, (
#         ("Mooring CTD", mooring.time_ts, mooring.salt, src_colors.mooring),
#         ("ROMS 900 m", roms.time, roms_salt_m, src_colors.roms),
#         ("NCOM", ncom.time, ncom_salt_m, src_colors.ncom));
#     ylabel = "Salinity (PSU)", title = "Salinity at the CTDs, $site", fname = "salt_lines_at_ctd.png")

# plot_lines(ctd_titles, (
#         ("ROMS 900 m", roms.time, roms_salt_m, src_colors.roms),
#         ("NCOM", ncom.time, ncom_salt_m, src_colors.ncom));
#     ylabel = "Salinity (PSU)", title = "Salinity at the CTDs, $site", fname = "salt_lines_at_ctd_model.png")

# plot_filtered(ctd_titles, (
#         ("Mooring CTD", mooring.time_ts, mooring.salt, src_colors.mooring),
#         ("ROMS 900 m", roms.time, roms_salt_m, src_colors.roms),
#         ("NCOM", ncom.time, ncom_salt_m, src_colors.ncom));
#     ylabel = "Salinity (PSU)", title = "Salinity at the CTDs", fname = "salt_lines_at_ctd")

# --- where the mooring CTDs were: pressure and depth of each instrument through time ---
# Both come straight from the CTD file ("pression" and "depth"); one line per instrument, deeper = lower.
let
    fig = Figure(size = (1100, 700))
    cols = Makie.wong_colors()
    axs = Axis[]
    for (row, (f, ylabel, ptitle)) in enumerate(((ctd_pres, "Pressure (dbar)", "Pressure"),
                                                 (mooring.z_ts, "Depth (m)", "Depth")))
        ax = Axis(fig[row, 1]; title = ptitle, titlealign = :left, ylabel = ylabel, yreversed = true, xticks = xticks)
        for k in axes(f, 1)
            lines!(ax, tday(mooring.time_ts), f[k, :]; color = cols[k], linewidth = 1,
                label = "CTD $(round(Int, nom_depth[k])) m nominal")
        end
        push!(axs, ax)
    end
    linkxaxes!(axs...)
    xlims!(axs[end], tday(model_time[1]), tday(model_time[end]))
    hidexdecorations!(axs[1]; ticks = false, grid = false)
    axs[end].xlabel = "Date (2022)"
    Legend(fig[3, 1], axs[1]; orientation = :horizontal, framevisible = false, tellwidth = false)
    Label(fig[0, :], "Pressure and depth of the mooring CTDs, $site", fontsize = 18, tellwidth = false)
    outname = joinpath(figure_path, "ctd_pressure_depth.png")
    save(outname, fig)
    println("saved ", outname)
    # zooms: the x axis is already in days since the start of the model run, so the limits are just day numbers
    # for (d0, d1) in ((3, 8), (33, 38))
    #     for ax in axs
    #         ax.xticks = (d0:d1, Dates.format.(t0 .+ Day.(d0:d1), "mm/dd"))   # one tick per day
    #     end
    #     xlims!(axs[end], d0, d1)
    #     outname = joinpath(figure_path, "ctd_pressure_depth_zoom$(d0)_$(d1).png")
    #     save(outname, fig)
    #     println("saved ", outname)
    # end
    fig
end

# --- where the ADCP bins were: depth of each of the 32 bins through time ---
# The ADCP file has no pressure, only the bin depths z, so this is one panel. Bin 1 is the deepest (nearest the
# instrument); all bins move together as the mooring is pulled down. Bins with no data (NaN) are simply not drawn.
let
    nbin = size(adcp_z, 1)
    fig = Figure(size = (1100, 500))
    ax = Axis(fig[1, 1]; ylabel = "Depth (m)", xlabel = "Date (2022)", yreversed = true, xticks = xticks,
        title = "Depth of the mooring ADCP bins, $site")
    for k in 1:nbin
        lines!(ax, tday(mooring.time_uv), adcp_z[k, :]; color = k, colormap = :viridis, colorrange = (1, nbin), linewidth = 0.8)
    end
    Colorbar(fig[1, 2]; colormap = :viridis, limits = (1, nbin), label = "Bin number (1 = deepest)")
    xlims!(ax, tday(model_time[1]), tday(model_time[end]))
    outname = joinpath(figure_path, "adcp_bin_depth.png")
    save(outname, fig)
    println("saved ", outname)
    # for (d0, d1) in ((3, 8), (33, 38))                 # same zooms as the CTD figure, one tick per day
    #     ax.xticks = (d0:d1, Dates.format.(t0 .+ Day.(d0:d1), "mm/dd"))
    #     xlims!(ax, d0, d1)
    #     outname = joinpath(figure_path, "adcp_bin_depth_zoom$(d0)_$(d1).png")
    #     save(outname, fig)
    #     println("saved ", outname)
    # end
    fig
end


##
# ===== velocity (true east / north): compared over the ADCP range =====
# The ADCP bins cover about 30–380 m, so the heatmaps show 0–400 m and the line plots use fixed depths
# (not the CTD depths): every source is interpolated to uv_depths on its own time axis.
zlim_uv = (0, 400)
uv_depths = [75.0, 150.0, 225.0, 300.0]   # 75 m is the shallowest depth the ADCP covers without gaps
uv_titles = ["$(round(Int, d)) m" for d in uv_depths]

# Blank a model field wherever the mooring has no data, so both are compared over the same times only.
# f = (npanel, nt) on `time`; ref_f = (npanel, nt_ref) on `ref_time`, same rows. Each model time takes the nearest
# mooring sample: if that one is NaN (or more than one mooring time step away, i.e. outside the record or in a
# time gap) the model value becomes NaN. Returns a copy, f itself is not changed.
function mask_like(time, f, ref_time, ref_f)
    out = copy(f)
    tm, tr = tday(time), tday(ref_time)
    dtr = median(diff(tr))
    for (n, t) in enumerate(tm)
        j = clamp(searchsortedlast(tr, t), 1, length(tr))
        j < length(tr) && tr[j+1] - t < t - tr[j] && (j += 1)       # nearest of the two samples around t
        far = abs(tr[j] - t) > dtr
        for k in axes(f, 1)
            (far || isnan(ref_f[k, j])) && (out[k, n] = NaN)
        end
    end
    return out
end

for (name, long, fm, fr, fn, fc) in (("u", "Eastward velocity u", mooring.u, roms.u, ncom.u, cmems.u),
                                     ("v", "Northward velocity v", mooring.v, roms.v, ncom.v, cmems.v))
    plot_time_depth((
            ("Mooring ADCP", mooring.time_uv, mooring.z_uv, fm),
            ("ROMS 900 m", roms.time, roms.z, fr),
            ("NCOM", ncom.time, ncom.z, fn),
            ("CMEMS", cmems.time, cmems.z, fc));
        zlim = zlim_uv, clim = (-1.5, 1.5), levels = -1.5:0.5:1.5, colormap = :balance,
        contour_color = (:black, 0.3),
        cblabel = "$name (m/s)", title = "$long at the $site", fname = "$(name)_time_depth.png")

    # every source at the fixed depths, each on its own time axis
    zm, zr, zn, zc = to_zgrid(mooring.z_uv, fm, uv_depths), to_zgrid(roms.z, fr, uv_depths),
                     to_zgrid(ncom.z, fn, uv_depths), to_zgrid(cmems.z, fc, uv_depths)
    masked(t, f) = mask_like(t, f, mooring.time_uv, zm)

    # masked heatmap: mask_like compares row by row, so every source has to be on the SAME depths first.
    # zg_hm is the 2 m grid plot_time_depth draws on; each panel is (label, time, z, field) with z = zg_hm at every time.
    zg_hm = collect(zlim_uv[1]:2.0:zlim_uv[2])
    hm_m = to_zgrid(mooring.z_uv, fm, zg_hm)
    hm_panel(label, s, f) = (label, s.time, repeat(zg_hm, 1, length(s.time)),
                             mask_like(s.time, to_zgrid(s.z, f, zg_hm), mooring.time_uv, hm_m))
    plot_time_depth((
            ("Mooring ADCP", mooring.time_uv, repeat(zg_hm, 1, length(mooring.time_uv)), hm_m),
            hm_panel("ROMS 900 m", roms, fr),
            hm_panel("NCOM", ncom, fn),
            hm_panel("CMEMS", cmems, fc));
        zlim = zlim_uv, clim = (-1.5, 1.5), levels = -1.5:0.5:1.5, colormap = :balance,
        contour_color = (:black, 0.3),
        cblabel = "$name (m/s)", title = "$long at the $site (models shown only where the ADCP has data)",
        fname = "$(name)_time_depth_mask.png")


    # with the mooring: the models are drawn only where the ADCP has data at that depth
    plot_lines(uv_titles, (
            ("Mooring ADCP", mooring.time_uv, zm, src_colors.mooring),
            ("ROMS 900 m", roms.time, masked(roms.time, zr), src_colors.roms),
            ("NCOM", ncom.time, masked(ncom.time, zn), src_colors.ncom),
            ("CMEMS", cmems.time, masked(cmems.time, zc), src_colors.cmems));
        ylabel = "$name (m/s)", zeroline = true,
        title = "$long at fixed depths (models shown only where the ADCP has data), $site", fname = "$(name)_lines_at_depth.png")

    # models only: not masked
    plot_lines(uv_titles, (
            ("ROMS 900 m", roms.time, zr, src_colors.roms),
            ("NCOM", ncom.time, zn, src_colors.ncom),
            ("CMEMS", cmems.time, zc, src_colors.cmems));
        ylabel = "$name (m/s)", zeroline = true,
        title = "$long at fixed depths, $site", fname = "$(name)_lines_at_depth_model.png")

    # filtered: not masked either — the filter needs the full model series
    plot_filtered(uv_titles, (
            ("Mooring ADCP", mooring.time_uv, zm, src_colors.mooring),
            ("ROMS 900 m", roms.time, zr, src_colors.roms),
            ("NCOM", ncom.time, zn, src_colors.ncom),
            ("CMEMS", cmems.time, zc, src_colors.cmems));
        ylabel = "$name (m/s)", title = "$long at fixed depths", fname = "$(name)_lines_at_depth")
end

##
# ===== filtered velocity on a regular depth grid: time-vs-depth heatmaps =====
# Each source's u and v are put on the same depths (10, 20, …, 400 m), then every depth row is low- and
# high-passed in time at Tcut. Each source keeps its own time axis. Result: uvf.mooring / uvf.roms / uvf.ncom /
# uvf.cmems, each with time, z (nz, nt) and the (nz, nt) matrices u, v, u_lp, u_hp, v_lp, v_hp.
# These are stored UNmasked: the models are filtered on their full series, the mask is only applied when plotting.
zg_uv = collect(10.0:10.0:400.0)
function uv_filtered(time, z, u, v)
    ug, vg = to_zgrid(z, u, zg_uv), to_zgrid(z, v, zg_uv)
    return (time = time, z = repeat(zg_uv, 1, length(time)), u = ug, v = vg,
        u_lp = filter_rows(time, ug, Tcut, "low"), u_hp = filter_rows(time, ug, Tcut, "high"),
        v_lp = filter_rows(time, vg, Tcut, "low"), v_hp = filter_rows(time, vg, Tcut, "high"))
end
uvf = (mooring = uv_filtered(mooring.time_uv, mooring.z_uv, mooring.u, mooring.v),
       roms    = uv_filtered(roms.time, roms.z, roms.u, roms.v),
       ncom    = uv_filtered(ncom.time, ncom.z, ncom.u, ncom.v),
       cmems   = uv_filtered(cmems.time, cmems.z, cmems.u, cmems.v))

zlim_uvf = (zg_uv[1], zg_uv[end])
# The panels of one figure: g picks the field out of a source, e.g. s -> s.u_lp. With mask = true the models are
# blanked wherever the mooring's own version of that field is NaN (same depths, so mask_like works row by row).
function uvf_panels(g; mask = true)
    fm = g(uvf.mooring)
    model(label, s) = (label, s.time, s.z, mask ? mask_like(s.time, g(s), uvf.mooring.time, fm) : g(s))
    return (("Mooring ADCP", uvf.mooring.time, uvf.mooring.z, fm),
            model("ROMS 900 m", uvf.roms), model("NCOM", uvf.ncom), model("CMEMS", uvf.cmems))
end
uvf_mask_note = " (models shown only where the ADCP has data)"
let tc = round(Int, Tcut)
    # colour limits (m/s): the high-passed current is much weaker than the low-passed one — adjust hp_max if it saturates
    hp_max = 0.3
    for (tag, long, cmax, step) in (("lp", "low-passed", 1.5, 0.5), ("hp", "high-passed", hp_max, hp_max / 3))
        for (name, vlong) in (("u", "Eastward velocity u"), ("v", "Northward velocity v"))
            plot_time_depth(uvf_panels(s -> getproperty(s, Symbol(name, "_", tag)));
                zlim = zlim_uvf, clim = (-cmax, cmax), levels = -cmax:step:cmax, colormap = :balance,
                contour_color = (:black, 0.3), contours = tag != "hp",   # no contour lines on the high-pass: they hide the colours
                cblabel = "$name (m/s)",
                title = "$vlong, $tc h $long, at the $site$uvf_mask_note", fname = "$(name)_time_depth_$(tag)$(tc)h.png")
        end
        # speed of the filtered current
        plot_time_depth(uvf_panels(s -> hypot.(getproperty(s, Symbol("u_", tag)), getproperty(s, Symbol("v_", tag))));
            zlim = zlim_uvf, clim = (0, cmax), levels = 0:step/2:cmax, colormap = :speed,
            contour_color = (:black, 0.3), cblabel = "Speed (m/s)",
            title = "Speed of the $tc h $long current at the $site$uvf_mask_note", fname = "speed_time_depth_$(tag)$(tc)h.png")
        # direction the filtered current flows TOWARD, as a compass heading: 0° = north, 90° = east, 180° = south, 270° = west.
        # :phase is a cyclic colormap, so 0° and 360° have the same colour. zstep = the data's own depth spacing, so no
        # interpolation in depth is done: averaging two angles across the 0/360 jump would give a wrong direction.
        # Where the current is weak the direction is not well defined and looks noisy (most of the high-pass figure).
        plot_time_depth(uvf_panels(s -> mod.(atand.(getproperty(s, Symbol("u_", tag)), getproperty(s, Symbol("v_", tag))), 360));
            zlim = zlim_uvf, clim = (0, 360), levels = 0:90:360, colormap = :phase, contours = false,
            zstep = zg_uv[2] - zg_uv[1], cblabel = "Direction toward (° clockwise from north)",
            title = "Direction of the $tc h $long current at the $site$uvf_mask_note", fname = "direction_time_depth_$(tag)$(tc)h.png")
    end
end

# envelope of the high-passed u and v on the depth grid: sqrt( lowpass( hp^2 ) ) with the Tenv low-pass, row by row.
# Result: uv_hp_env.mooring / .roms / .ncom / .cmems, each with u and v of size (nz, nt). Always >= 0, so one-sided colours.
uv_hp_env = map(s -> (u = sqrt.(max.(filter_rows(s.time, s.u_hp .^ 2, Tenv, "low"), 0)),
                      v = sqrt.(max.(filter_rows(s.time, s.v_hp .^ 2, Tenv, "low"), 0))), uvf)
let tc = round(Int, Tcut), te = round(Int, Tenv)
    env_max = 0.2          # colour limit (m/s) — a guess, adjust after the first look
    for (name, vlong) in (("u", "Eastward velocity u"), ("v", "Northward velocity v"))
        env(src) = getproperty(getproperty(uv_hp_env, src), Symbol(name))
        plot_time_depth((
                ("Mooring ADCP", uvf.mooring.time, uvf.mooring.z, env(:mooring)),
                ("ROMS 900 m", uvf.roms.time, uvf.roms.z, env(:roms)),
                ("NCOM", uvf.ncom.time, uvf.ncom.z, env(:ncom)));
            zlim = zlim_uvf, clim = (0, env_max), levels = 0:env_max/4:env_max, colormap = :amp, contours = false,
            cblabel = "Envelope of $name (m/s)",
            title = "$vlong, envelope of the $tc h high-pass ($te h low-pass), at the $site",
            fname = "$(name)_time_depth_hp$(tc)h_envelope$(te)h.png")
    end
end

##
# ===== current speed |u + iv| =====
# Speed does not depend on the direction the axes point, so a wrong rotation of u/v (in the ADCP, ROMS or NCOM)
# cannot change it: if only the angle were wrong, these three would agree even though u and v do not.
spd_m, spd_r, spd_n, spd_c = hypot.(mooring.u, mooring.v), hypot.(roms.u, roms.v), hypot.(ncom.u, ncom.v), hypot.(cmems.u, cmems.v)

plot_time_depth((
        ("Mooring ADCP", mooring.time_uv, mooring.z_uv, spd_m),
        ("ROMS 900 m", roms.time, roms.z, spd_r),
        ("NCOM", ncom.time, ncom.z, spd_n),
        ("CMEMS", cmems.time, cmems.z, spd_c));
    zlim = zlim_uv, clim = (0, 1.5), levels = 0:0.25:1.5, colormap = :speed, contour_color = (:black, 0.3),
    cblabel = "Speed (m/s)", title = "Current speed at the $site", fname = "speed_time_depth.png")

# masked heatmap: every source on the same 2 m depth grid first (mask_like compares row by row), as for u and v
let zg = collect(zlim_uv[1]:2.0:zlim_uv[2])
    hm_m = to_zgrid(mooring.z_uv, spd_m, zg)
    panel(label, s, f) = (label, s.time, repeat(zg, 1, length(s.time)),
                          mask_like(s.time, to_zgrid(s.z, f, zg), mooring.time_uv, hm_m))
    plot_time_depth((
            ("Mooring ADCP", mooring.time_uv, repeat(zg, 1, length(mooring.time_uv)), hm_m),
            panel("ROMS 900 m", roms, spd_r), panel("NCOM", ncom, spd_n), panel("CMEMS", cmems, spd_c));
        zlim = zlim_uv, clim = (0, 1.5), levels = 0:0.25:1.5, colormap = :speed, contour_color = (:black, 0.3),
        cblabel = "Speed (m/s)", title = "Current speed at the $site (models shown only where the ADCP has data)",
        fname = "speed_time_depth_mask.png")
end

# speed at the fixed depths, each source on its own time axis: spd_z.mooring / .roms / .ncom / .cmems, (length(uv_depths), nt)
spd_z = (mooring = to_zgrid(mooring.z_uv, spd_m, uv_depths), roms = to_zgrid(roms.z, spd_r, uv_depths),
         ncom = to_zgrid(ncom.z, spd_n, uv_depths), cmems = to_zgrid(cmems.z, spd_c, uv_depths))
spd_masked(t, f) = mask_like(t, f, mooring.time_uv, spd_z.mooring)

# with the mooring: the models are drawn only where the ADCP has data at that depth
plot_lines(uv_titles, (
        ("Mooring ADCP", mooring.time_uv, spd_z.mooring, src_colors.mooring),
        ("ROMS 900 m", roms.time, spd_masked(roms.time, spd_z.roms), src_colors.roms),
        ("NCOM", ncom.time, spd_masked(ncom.time, spd_z.ncom), src_colors.ncom),
        ("CMEMS", cmems.time, spd_masked(cmems.time, spd_z.cmems), src_colors.cmems));
    ylabel = "Speed (m/s)", title = "Current speed at fixed depths (models shown only where the ADCP has data), $site",
    fname = "speed_lines_at_depth.png")

# models only: not masked
plot_lines(uv_titles, (
        ("ROMS 900 m", roms.time, spd_z.roms, src_colors.roms),
        ("NCOM", ncom.time, spd_z.ncom, src_colors.ncom),
        ("CMEMS", cmems.time, spd_z.cmems, src_colors.cmems));
    ylabel = "Speed (m/s)", title = "Current speed at fixed depths, $site", fname = "speed_lines_at_depth_model.png")

# here the speed series itself is filtered (not the speed of the filtered u and v).
# mask = true: filtered on the full series first, then the models are blanked where the mooring has no data
plot_filtered(uv_titles, (
        ("Mooring ADCP", mooring.time_uv, spd_z.mooring, src_colors.mooring),
        ("ROMS 900 m", roms.time, spd_z.roms, src_colors.roms),
        ("NCOM", ncom.time, spd_z.ncom, src_colors.ncom),
        ("CMEMS", cmems.time, spd_z.cmems, src_colors.cmems));
    ylabel = "Speed (m/s)", title = "Current speed at fixed depths (models shown only where the ADCP has data)",
    fname = "speed_lines_at_depth", mask = true)

# speed of the FILTERED current: filter u and v first, then take the magnitude.
# The envelope is sqrt( lowpass( hp_u^2 + hp_v^2 ) ): the running RMS speed of the high-frequency current.
uv_src = (("Mooring ADCP", mooring.time_uv, mooring.z_uv, mooring.u, mooring.v, src_colors.mooring),
          ("ROMS 900 m", roms.time, roms.z, roms.u, roms.v, src_colors.roms),
          ("NCOM", ncom.time, ncom.z, ncom.u, ncom.v, src_colors.ncom))
filt_uv(kind) = Tuple((label, t, filter_rows(t, to_zgrid(z, u, uv_depths), Tcut, kind),
                       filter_rows(t, to_zgrid(z, v, uv_depths), Tcut, kind), c) for (label, t, z, u, v, c) in uv_src)
let tc = round(Int, Tcut), te = round(Int, Tenv)
    for (kind, long) in (("low", "low-passed"), ("high", "high-passed"))
        plot_lines(uv_titles, Tuple((label, t, hypot.(fu, fv), c) for (label, t, fu, fv, c) in filt_uv(kind));
            ylabel = "Speed (m/s)", title = "Speed of the $tc h $long current at fixed depths, $site",
            fname = "speed_of_$(kind)pass$(tc)h_uv_lines_at_depth.png")
    end
    plot_lines(uv_titles, Tuple((label, t, sqrt.(max.(filter_rows(t, fu .^ 2 .+ fv .^ 2, Tenv, "low"), 0)), c)
                                for (label, t, fu, fv, c) in filt_uv("high"));
        ylabel = "Envelope: Speed (m/s)",
        title = "Envelope of the $tc h high-passed current speed ($te h low-pass) at fixed depths, $site",
        fname = "speed_of_hp$(tc)h_uv_envelope$(te)h_lines_at_depth.png")
end

# time-mean speed at each depth, for a quick number to go with the figures
nanmean(x) = mean(filter(!isnan, x))
println("mean speed (m/s)   mooring   ROMS    NCOM")
for (k, d) in enumerate(uv_depths)
    println(lpad(round(Int, d), 6), " m       ",
        join((lpad(round(nanmean(to_zgrid(z, s, [d])), digits = 2), 6) for (z, s) in
              ((mooring.z_uv, spd_m), (roms.z, spd_r), (ncom.z, spd_n))), "  "))
end

##
# ===== scatter plots of the current: u, v, speed and direction, unfiltered / low-passed / high-passed =====
# Step 1, uvz: u and v of every source at the fixed depths uv_depths, each on its own time axis, with the Tcut
#   low- and high-pass. uvz.roms.lp.u is (length(uv_depths), nt); the fields are raw, lp, hp, each with u and v.
function uv_at_depths(time, z, u, v)
    ug, vg = to_zgrid(z, u, uv_depths), to_zgrid(z, v, uv_depths)
    band(kind) = (u = filter_rows(time, ug, Tcut, kind), v = filter_rows(time, vg, Tcut, kind))
    return (time = time, raw = (u = ug, v = vg), lp = band("low"), hp = band("high"))
end
uvz = (mooring = uv_at_depths(mooring.time_uv, mooring.z_uv, mooring.u, mooring.v),
       roms    = uv_at_depths(roms.time, roms.z, roms.u, roms.v),
       ncom    = uv_at_depths(ncom.time, ncom.z, ncom.u, ncom.v),
       cmems   = uv_at_depths(cmems.time, cmems.z, cmems.u, cmems.v))

# Step 2: average u and v into common time bins (time_average), THEN take the speed and the direction of the
#   averaged vector. A direction cannot be averaged as a number (359° and 1° would give 180°), and this way the
#   speed and the direction belong to the same vector. dir = compass heading the current flows toward (0° = north,
#   90° = east), as in the direction heatmaps. Result e.g. uvza.roms.hp.spd, (length(uv_depths), nbin).
function uv_binned(s, dt)
    function quantities(b)
        ub, vb = time_average(s.time, b.u, dt)[2], time_average(s.time, b.v, dt)[2]
        return (u = ub, v = vb, spd = hypot.(ub, vb), dir = mod.(atand.(ub, vb), 360))
    end
    return (time = time_average(s.time, s.raw.u, dt)[1], raw = quantities(s.raw), lp = quantities(s.lp), hp = quantities(s.hp))
end
# The time step depends on the field:
#   low-passed          : avg_dt (1 day, as for temperature) — a 28 h low-pass has nothing faster than a day left,
#                         so shorter bins would only add points that are copies of their neighbours. -> uvzd
#   unfiltered, high-passed: much shorter than a day, or the fast signal averages away to nothing:
#       with the mooring: 6 h, the CMEMS time step (CMEMS is one snapshot per bin, the others are 6 h means) -> uvza
#       ROMS vs NCOM    : 1 h, the NCOM time step -> uvzh
uv_dt, uv_dt_rn = Hour(6), Hour(1)
uvzd = map(s -> uv_binned(s, avg_dt), uvz)
uvza = map(s -> uv_binned(s, uv_dt), uvz)
uvzh = (roms = uv_binned(uvz.roms, uv_dt_rn), ncom = uv_binned(uvz.ncom, uv_dt_rn))

let tc = round(Int, Tcut)
    for (fld, flong) in ((:raw, "unfiltered"), (:lp, "$tc h low-passed"), (:hp, "$tc h high-passed")),
        (q, qlong, unit, qname) in ((:u, "Eastward velocity u", "m/s", "u"), (:v, "Northward velocity v", "m/s", "v"),
                                    (:spd, "Current speed", "m/s", "speed"),
                                    (:dir, "Current direction (toward, clockwise from north)", "°", "direction"))
        pick(s) = getproperty(getproperty(s, fld), q)
        # which binned set to use: daily for the low-pass, the short steps for the unfiltered and high-passed
        vm, dt_m, vrn, dt_rn, ms_rn = fld == :lp ? (uvzd, avg_dt, uvzd, avg_dt, 9) : (uvza, uv_dt, uvzh, uv_dt_rn, 4)
        # the three models against the mooring
        plot_scatter(uv_titles, vm.mooring.time, pick(vm.mooring), (
                ("ROMS 900 m", pick(vm.roms), src_colors.roms),
                ("NCOM", pick(vm.ncom), src_colors.ncom),
                ("CMEMS", pick(vm.cmems), src_colors.cmems));
            unit = unit, angle = q == :dir, title = "$qlong, $flong, $dt_m bins: models vs mooring",
            fname = "$(qname)_scatter_at_depth_$(fld).png")
        # ROMS against NCOM
        plot_scatter(uv_titles, vrn.ncom.time, pick(vrn.ncom), (
                ("ROMS 900 m", pick(vrn.roms), src_colors.roms),);
            xname = "NCOM", unit = unit, angle = q == :dir, markersize = ms_rn,
            title = "$qlong, $flong, $dt_rn bins: ROMS vs NCOM",
            fname = "$(qname)_scatter_at_depth_roms_ncom_$(fld).png")
    end
end

##
# ===== velocity split into three frequency bands, for the kinetic energy calculation =====
# u and v on the regular depth grid zg_uv (uvf.*.u / uvf.*.v), every depth row filtered in time:
#   lp : T_lp low-pass          (subtidal flow)
#   bp : T_bp band-pass         (semidiurnal band)
#   hp : T_hp high-pass         (higher frequencies)
# The cutoff periods are set ONLY on the T_lp, T_bp, T_hp line below: every calculation, title and file name follows it.
# Result: uvk.mooring / uvk.roms / uvk.ncom / uvk.cmems, each with time, z (nz, nt) and lp, bp, hp, each of which has u and v
# of size (nz, nt) — e.g. uvk.roms.bp.u. The three bands do not add up to the full signal: periods between the bands are left out.
T_lp, T_bp, T_hp = 30.0, (9.0, 15.0), 9.0       # cutoff periods (HOURS)
# A band can only be filtered if its shortest period is longer than 2 time steps (the Nyquist limit); otherwise the
# filter cannot be built and the band is returned as all NaN. This is the case for CMEMS (6-hourly, limit 12 h):
# its lp is real, its bp and hp are NaN.
function uv_bands(s)
    dt = median(diff(tday(s.time))) * 24             # sampling interval (hours)
    function band(kind, T)
        minimum(T) > 2 * dt || return (u = fill(NaN, size(s.u)), v = fill(NaN, size(s.v)))
        return (u = filter_rows(s.time, s.u, T, kind), v = filter_rows(s.time, s.v, T, kind))
    end
    return (time = s.time, z = s.z, lp = band("low", T_lp), bp = band("band", T_bp), hp = band("high", T_hp))
end
uvk = (mooring = uv_bands(uvf.mooring), roms = uv_bands(uvf.roms),ncom = uv_bands(uvf.ncom),cmems = uv_bands(uvf.cmems))
for (name, s) in pairs(uvk)
    println(name, ": size(lp.u) = ", size(s.lp.u), ", valid points lp/bp/hp = ",
        join((count(!isnan, b.u) for b in (s.lp, s.bp, s.hp)), " / "))
end

##
# ===== eddy kinetic energy of the low-frequency flow =====
# EKE(z, t) = 0.5 * ((u_lp - ubar)^2 + (v_lp - vbar)^2), per unit mass (m²/s²), where ubar(z), vbar(z) are the
# time means of the T_lp low-passed u and v at each depth (NaN ignored). The mean is taken over each source's own
# valid samples, so for the ADCP rows with gaps it covers less of the period than for the models.
rowmean(a) = [mean(filter(!isnan, a[k, :])) for k in axes(a, 1)]     # time mean of every depth row; NaN if no data
function eke_lp(s)
    ubar, vbar = rowmean(s.lp.u), rowmean(s.lp.v)
    return (time = s.time, z = s.z, ubar = ubar, vbar = vbar,
        eke = 0.5 .* ((s.lp.u .- ubar) .^ 2 .+ (s.lp.v .- vbar) .^ 2),     # (nz, nt)
        mke = 0.5 .* (ubar .^ 2 .+ vbar .^ 2))                             # (nz), kinetic energy of the mean flow
end
eke = (mooring = eke_lp(uvk.mooring), roms = eke_lp(uvk.roms), ncom = eke_lp(uvk.ncom), cmems = eke_lp(uvk.cmems))

println("time-mean EKE (m²/s²)   mooring    ROMS     NCOM    CMEMS")
for d in (50.0, 100.0, 200.0, 300.0)         # depths on the 10 m grid zg_uv
    k = findfirst(==(d), zg_uv)
    println(lpad(round(Int, d), 6), " m          ",
        join((lpad(round(rowmean(s.eke)[k], digits = 4), 7) for s in eke), "  "))
end

size(eke.roms.eke)      # (40, nt) = (depths, times)
size(eke.mooring.eke)
size(eke.ncom.eke)
size(eke.cmems.eke)

# time-vs-depth heatmap of the EKE, one panel per source (adjust eke_max if the colours saturate or look washed out)
eke_max = 0.3
plot_time_depth((
        ("Mooring ADCP", eke.mooring.time, eke.mooring.z, eke.mooring.eke),
        ("ROMS 900 m", eke.roms.time, eke.roms.z, eke.roms.eke),
        ("NCOM", eke.ncom.time, eke.ncom.z, eke.ncom.eke),
        ("CMEMS", eke.cmems.time, eke.cmems.z, eke.cmems.eke));
    zlim = zlim_uvf, clim = (0, eke_max), levels = 0:eke_max/6:eke_max, colormap = :amp, contour_color = (:black, 0.3),
    cblabel = "EKE (m²/s²)", title = "Eddy kinetic energy of the $(round(Int, T_lp)) h low-passed flow at the $site",
    fname = "eke_time_depth_lp$(round(Int, T_lp))h.png")


ke_band(b) = 0.5 .* (b.u .^ 2 .+ b.v .^ 2)        # (nz, nt), m²/s²
ke = map(s -> (time = s.time, z = s.z, bp = ke_band(s.bp), hp = ke_band(s.hp), lp = ke_band(s.lp)), uvk)
ke_smooth(s, k) = max.(filter_rows(s.time, k, T_lp, "low"), 0)   # T_lp low-pass, clamped at 0

# (band, title, colour limit in m²/s²) — the limits are guesses, adjust them after the first look
for (band, long, kmax) in ((:lp, "$(round(Int, T_lp)) h low-passed", 1.0), (:bp, "$(T_bp[1])–$(T_bp[2]) h band-passed", 0.05),
                           (:hp, "$(round(Int, T_hp)) h high-passed", 0.02))
    plot_time_depth((
            ("Mooring ADCP", ke.mooring.time, ke.mooring.z, getproperty(ke.mooring, band)),
            ("ROMS 900 m", ke.roms.time, ke.roms.z, getproperty(ke.roms, band)),
            ("NCOM", ke.ncom.time, ke.ncom.z, getproperty(ke.ncom, band)));
        zlim = zlim_uvf, clim = (0, kmax), levels = 0:kmax/5:kmax, colormap = :amp, contours = false,
        cblabel = "KE (m²/s²)", title = "Kinetic energy of the $long flow at the $site",
        fname = "ke_time_depth_$(band).png")
end
    
# envelope of the high-frequency KE: sqrt( lowpass( ke.hp^2 ) ) with a 24 h low-pass, i.e. the running RMS of the
# T_hp high-passed KE (m²/s²). The low-pass can ring slightly below zero, so it is clamped at 0 before the square root.
T_kenv = 24.0              # low-pass cutoff period of the KE envelope (HOURS)
ke_hp_env = map(s -> sqrt.(max.(filter_rows(s.time, s.hp .^ 2, T_kenv, "low"), 0)), ke)   # e.g. ke_hp_env.roms, (nz, nt)

ke_env_max = 0.02          # colour limit (m²/s²) — a guess, adjust after the first look
plot_time_depth((
        ("Mooring ADCP", ke.mooring.time, ke.mooring.z, ke_hp_env.mooring),
        ("ROMS 900 m", ke.roms.time, ke.roms.z, ke_hp_env.roms),
        ("NCOM", ke.ncom.time, ke.ncom.z, ke_hp_env.ncom));
    zlim = zlim_uvf, clim = (0, ke_env_max), levels = 0:ke_env_max/5:ke_env_max, colormap = :amp, contours = false,
    cblabel = "KE envelope (m²/s²)",
    title = "Envelope of the $(round(Int, T_hp)) h high-passed KE ($(round(Int, T_kenv)) h low-pass) at the $site",
    fname = "ke_hp_envelope$(round(Int, T_kenv))h_time_depth.png")


##
##
# ===== depth-integrated kinetic energy, 100–300 m =====
# The KE of each band (ke.lp / ke.bp / ke.hp) and the EKE of the low-passed flow, integrated over z_int with the
# trapezoid rule on the 10 m grid zg_uv. KE is per unit mass (m²/s²), so the integral is in m³/s²; multiply by a
# density (about 1025 kg/m³) for J/m², or divide by the layer thickness for the depth-mean KE.
# A time is NaN if ANY depth in the range is NaN there, so the integral always covers the whole layer.
# Result: kei.mooring / .roms / .ncom / .cmems, each with time and the vectors eke, lp, bp, hp (length nt).
# For CMEMS only eke and lp exist (6-hourly data, see uv_bands), its bp and hp are NaN.
z_int = (100.0, 300.0)     # depth range of the integral (m), limits included
function depth_integral(f; zr = z_int)
    k = findall(d -> zr[1] <= d <= zr[2], zg_uv)
    dz = diff(zg_uv[k])
    return [sum(dz .* (f[k[1:end-1], t] .+ f[k[2:end], t]) ./ 2) for t in axes(f, 2)]
end
kei = map(ke, eke) do s, e
    (time = s.time, eke = depth_integral(e.eke), lp = depth_integral(s.lp), bp = depth_integral(s.bp), hp = depth_integral(s.hp))
end
for (name, s) in pairs(kei)
    println(name, ": times with the full $(round(Int, z_int[1]))–$(round(Int, z_int[2])) m layer = ",
        round(100 * count(!isnan, s.lp) / length(s.lp), digits = 1), " %")
end

# one panel per quantity, one line per source; with the mooring the models are drawn only where the ADCP covers the layer
let zlab = "$(round(Int, z_int[1]))–$(round(Int, z_int[2])) m"
    titles = ["EKE of the $(round(Int, T_lp)) h low-passed flow", "KE, $(round(Int, T_lp)) h low-passed",
              "KE, $(T_bp[1])–$(T_bp[2]) h band-passed", "KE, $(round(Int, T_hp)) h high-passed"]
    rows(s) = vcat(s.eke', s.lp', s.bp', s.hp')           # (4, nt): the four panels
    ref = rows(kei.mooring)
    masked(s) = mask_like(s.time, rows(s), kei.mooring.time, ref)
    plot_lines(titles, (
            ("Mooring ADCP", kei.mooring.time, ref, src_colors.mooring),
            ("ROMS 900 m", kei.roms.time, masked(kei.roms), src_colors.roms),
            ("NCOM", kei.ncom.time, masked(kei.ncom), src_colors.ncom),
            ("CMEMS", kei.cmems.time, masked(kei.cmems), src_colors.cmems));
        ylabel = "∫KE dz (m³/s²)",
        title = "Kinetic energy integrated over $zlab (models shown only where the ADCP has data), $site",
        fname = "ke_depth_integrated_$(round(Int, z_int[1]))_$(round(Int, z_int[2]))m.png")
    plot_lines(titles, (
            ("ROMS 900 m", kei.roms.time, rows(kei.roms), src_colors.roms),
            ("NCOM", kei.ncom.time, rows(kei.ncom), src_colors.ncom),
            ("CMEMS", kei.cmems.time, rows(kei.cmems), src_colors.cmems));
        ylabel = "∫KE dz (m³/s²)", title = "Kinetic energy integrated over $zlab, $site",
        fname = "ke_depth_integrated_$(round(Int, z_int[1]))_$(round(Int, z_int[2]))m_model.png")
end

# --- envelope of the depth-integrated band-passed and high-passed KE ---
# Same definition as the other envelopes: sqrt( lowpass( x^2 ) ) with the T_kenv (24 h) low-pass, i.e. the running RMS
# of the series. It removes the fast pulsing (KE of a tidal current peaks twice per tidal period) and keeps the slow
# change in strength, e.g. the spring–neap cycle. Clamped at 0 before the square root, as the low-pass can ring below zero.
# Result: kei_env.mooring / .roms / .ncom / .cmems, each with time and the vectors bp and hp (CMEMS: all NaN).
kei_env = map(kei) do s
    env(x) = vec(sqrt.(max.(filter_rows(s.time, permutedims(x) .^ 2, T_kenv, "low"), 0)))
    (time = s.time, bp = env(s.bp), hp = env(s.hp))
end

let zlab = "$(round(Int, z_int[1]))–$(round(Int, z_int[2])) m", te = round(Int, T_kenv)
    titles = ["KE, $(T_bp[1])–$(T_bp[2]) h band-passed", "KE, $(round(Int, T_hp)) h high-passed"]
    rows(s) = vcat(s.bp', s.hp')                          # (2, nt): the two panels
    ref = rows(kei_env.mooring)
    masked(s) = mask_like(s.time, rows(s), kei_env.mooring.time, ref)   # filtered on the full series first, masked after
    # CMEMS is left out: it has no band-passed or high-passed KE
    plot_lines(titles, (
            ("Mooring ADCP", kei_env.mooring.time, ref, src_colors.mooring),
            ("ROMS 900 m", kei_env.roms.time, masked(kei_env.roms), src_colors.roms),
            ("NCOM", kei_env.ncom.time, masked(kei_env.ncom), src_colors.ncom));
        ylabel = "Envelope: ∫KE dz (m³/s²)",
        title = "Envelope ($te h low-pass) of the KE integrated over $zlab (models shown only where the ADCP has data), $site",
        fname = "ke_depth_integrated_$(round(Int, z_int[1]))_$(round(Int, z_int[2]))m_envelope$(te)h.png")
    plot_lines(titles, (
            ("ROMS 900 m", kei_env.roms.time, rows(kei_env.roms), src_colors.roms),
            ("NCOM", kei_env.ncom.time, rows(kei_env.ncom), src_colors.ncom));
        ylabel = "Envelope: ∫KE dz (m³/s²)",
        title = "Envelope ($te h low-pass) of the KE integrated over $zlab, $site",
        fname = "ke_depth_integrated_$(round(Int, z_int[1]))_$(round(Int, z_int[2]))m_envelope$(te)h_model.png")
end
