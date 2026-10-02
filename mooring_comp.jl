using NCDatasets, CairoMakie, Dates, Statistics, GibbsSeaWater
CairoMakie.activate!()
include("/home/hsinyi/Documents/Julia/function/load_all.jl")
include("/home/hsinyi/Documents/Julia/function/plotting_fun.jl")
##
grid_fname = "/home/hsinyi/roms_data/grid/roms_grd_900m.nc"   # the grid_file listed in the .nc's global attributes
mooring_loc = "/home/hsinyi/roms_data/grid/roms_grd_900m_mor_edata.nc"   # HPC output dir — contains avg/dia/his/rst files mixed together
figure_path = "/home/hsinyi/figure/20260929_output/dbry_0927/mooring"
joint_ext_path = "/home/hsinyi/roms_data/mooring_ext_model/dbry_0927/mooring_merged"
mooring_path = "/home/hsinyi/data_notm/French_mooring"
NCOM_mooring_path = "/home/hsinyi/roms_data/NCOM_mooring/NCOM_french.nc"
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
    ds.attrib["french_mor_info"][:]
end
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

println("model: ", model_time[1], " to ", model_time[end], " (", length(model_time), " steps)")
println("NCOM : ", ncom_time[1], " to ", ncom_time[end], ", size(ncom_temp) = ", size(ncom_temp))
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
for (name, s) in (("roms", roms), ("ncom", ncom))
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
src_colors = (mooring = :black, roms = "#2a78d6", ncom = "#eb6834")   # same colour = same data source in every line plot

# Time-vs-depth heatmaps, one panel per data source, sharing the axes and the colour scale.
# panels = ((title, time, z, field), …) with z and field of size (nz, nt); zlim = (shallow, deep) in m.
function plot_time_depth(panels; zlim, clim, colormap, levels, cblabel, title, fname,
                         contour_color = (:white, 0.6), ctd_lines = false)
    zg = zlim[1]:2.0:zlim[2]
    fig = Figure(size = (1100, 900))
    axs, hms = Axis[], []
    for (row, (ptitle, time, z, f)) in enumerate(panels)
        ax = Axis(fig[row, 1]; title = ptitle, titlealign = :left, ylabel = "Depth (m)",
            yreversed = true, xticks = xticks, xgridvisible = false, ygridvisible = false)
        fg = to_zgrid(z, f, zg)
        push!(hms, heatmap!(ax, tday(time), zg, fg'; colormap = colormap, colorrange = clim))
        contour!(ax, tday(time), zg, fg'; levels = levels, color = contour_color, linewidth = 0.8)
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
        ("NCOM", ncom.time, ncom.z, ncom.temp));
    zlim = zlim_ts, clim = (8, 20), levels = 8:2:20, colormap = :thermal, ctd_lines = true,
    cblabel = "Temperature (°C)", title = "Temperature at the $site", fname = "temp_time_depth.png")

roms_temp_m = at_mooring_depth(roms, roms.temp)
ncom_temp_m = at_mooring_depth(ncom, ncom.temp)
plot_lines(ctd_titles, (
        ("Mooring CTD (potential T)", mooring.time_ts, mooring.temp, src_colors.mooring),
        ("ROMS 900 m (potential T)", roms.time, roms_temp_m, src_colors.roms),
        ("NCOM", ncom.time, ncom_temp_m, src_colors.ncom));
    ylabel = "Temperature (°C)", title = "Temperature at the CTDs, $site", fname = "temp_lines_at_ctd.png")

# --- salinity ---
plot_time_depth((
        ("Mooring CTD (interpolated between the instruments; lines = instrument depths)",
            mooring.time_ts, mooring.z_ts, mooring.salt),
        ("ROMS 900 m", roms.time, roms.z, roms.salt),
        ("NCOM", ncom.time, ncom.z, ncom.salt));
    zlim = zlim_ts, clim = (34.8, 36.0), levels = 34.8:0.2:36.0, colormap = :viridis, ctd_lines = true,
    cblabel = "Salinity (PSU)", title = "Salinity at the $site", fname = "salt_time_depth.png")

roms_salt_m = at_mooring_depth(roms, roms.salt)
ncom_salt_m = at_mooring_depth(ncom, ncom.salt)
plot_lines(ctd_titles, (
        ("Mooring CTD", mooring.time_ts, mooring.salt, src_colors.mooring),
        ("ROMS 900 m", roms.time, roms_salt_m, src_colors.roms),
        ("NCOM", ncom.time, ncom_salt_m, src_colors.ncom));
    ylabel = "Salinity (PSU)", title = "Salinity at the CTDs, $site", fname = "salt_lines_at_ctd.png")

##
# ===== velocity (true east / north): compared over the ADCP range =====
# The ADCP bins cover about 30–380 m, so the heatmaps show 0–400 m and the line plots use fixed depths
# (not the CTD depths): every source is interpolated to uv_depths on its own time axis.
zlim_uv = (0, 400)
uv_depths = [75.0, 150.0, 225.0, 300.0]   # 75 m is the shallowest depth the ADCP covers without gaps
uv_titles = ["$(round(Int, d)) m" for d in uv_depths]

for (name, long, fm, fr, fn) in (("u", "Eastward velocity u", mooring.u, roms.u, ncom.u),
                                 ("v", "Northward velocity v", mooring.v, roms.v, ncom.v))
    plot_time_depth((
            ("Mooring ADCP", mooring.time_uv, mooring.z_uv, fm),
            ("ROMS 900 m", roms.time, roms.z, fr),
            ("NCOM", ncom.time, ncom.z, fn));
        zlim = zlim_uv, clim = (-1.5, 1.5), levels = -1.5:0.5:1.5, colormap = :balance,
        contour_color = (:black, 0.3),
        cblabel = "$name (m/s)", title = "$long at the $site", fname = "$(name)_time_depth.png")

    plot_lines(uv_titles, (
            ("Mooring ADCP", mooring.time_uv, to_zgrid(mooring.z_uv, fm, uv_depths), src_colors.mooring),
            ("ROMS 900 m", roms.time, to_zgrid(roms.z, fr, uv_depths), src_colors.roms),
            ("NCOM", ncom.time, to_zgrid(ncom.z, fn, uv_depths), src_colors.ncom));
        ylabel = "$name (m/s)", zeroline = true,
        title = "$long at fixed depths, $site", fname = "$(name)_lines_at_depth.png")
end

##
# ===== current speed |u + iv| =====
# Speed does not depend on the direction the axes point, so a wrong rotation of u/v (in the ADCP, ROMS or NCOM)
# cannot change it: if only the angle were wrong, these three would agree even though u and v do not.
spd_m, spd_r, spd_n = hypot.(mooring.u, mooring.v), hypot.(roms.u, roms.v), hypot.(ncom.u, ncom.v)

plot_time_depth((
        ("Mooring ADCP", mooring.time_uv, mooring.z_uv, spd_m),
        ("ROMS 900 m", roms.time, roms.z, spd_r),
        ("NCOM", ncom.time, ncom.z, spd_n));
    zlim = zlim_uv, clim = (0, 1.5), levels = 0:0.25:1.5, colormap = :speed, contour_color = (:black, 0.3),
    cblabel = "Speed (m/s)", title = "Current speed at the $site", fname = "speed_time_depth.png")

plot_lines(uv_titles, (
        ("Mooring ADCP", mooring.time_uv, to_zgrid(mooring.z_uv, spd_m, uv_depths), src_colors.mooring),
        ("ROMS 900 m", roms.time, to_zgrid(roms.z, spd_r, uv_depths), src_colors.roms),
        ("NCOM", ncom.time, to_zgrid(ncom.z, spd_n, uv_depths), src_colors.ncom));
    ylabel = "Speed (m/s)", title = "Current speed at fixed depths, $site", fname = "speed_lines_at_depth.png")

# time-mean speed at each depth, for a quick number to go with the figures
nanmean(x) = mean(filter(!isnan, x))
println("mean speed (m/s)   mooring   ROMS    NCOM")
for (k, d) in enumerate(uv_depths)
    println(lpad(round(Int, d), 6), " m       ",
        join((lpad(round(nanmean(to_zgrid(z, s, [d])), digits = 2), 6) for (z, s) in
              ((mooring.z_uv, spd_m), (roms.z, spd_r), (ncom.z, spd_n))), "  "))
end
