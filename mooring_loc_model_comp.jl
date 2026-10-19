using NCDatasets, CairoMakie, Dates, Statistics, GibbsSeaWater, DSP
CairoMakie.activate!()
include("/home/hsinyi/Documents/Julia/function/load_all.jl")
include("/home/hsinyi/Documents/Julia/function/plotting_fun.jl")
##
grid_fname = "/home/hsinyi/roms_data/grid/roms_grd_900m.nc"   # the grid_file listed in the .nc's global attributes
mooring_loc = "/home/hsinyi/roms_data/grid/roms_grd_900m_mor_edata.nc"   # HPC output dir — contains avg/dia/his/rst files mixed together
figure_path = "/home/hsinyi/figure/20261019_mooring_loc_comp"
joint_ext_path = "/home/hsinyi/roms_data/mooring_ext_model/1008/mooring_merged"

NCOM_mooring_path = "/home/hsinyi/roms_data/NCOM_mooring"
CMEMS_mooring_path = "/home/hsinyi/roms_data/GLOBAL_mooring"
HYCOM_mooring_path = "/home/hsinyi/roms_data/HYCOM_mooring"
mkpath(figure_path)

## switch for the compare loc
mooring_to_comp = ["M1","M3","CPIES7","M1_CPIES7"]
##
# --- grid, reference time and S-coordinate parameters (same as mooring_comp.jl) ---
h, = NCDataset(grid_fname) do ds
    (ds["h"][:, :],)
end
t_ref = DateTime(1994, 1, 1, 0, 0, 0)
ini_fname = "/home/hsinyi/roms_data/ini/roms_ini_900m2022082400.nc"
theta_s, theta_b, hc = NCDataset(ini_fname) do ds
    ds["theta_s"][1], ds["theta_b"][1], ds["hc"][1]
end
flipz(a) = reverse(a, dims=1)            # files are stored deepest-first; make them surface-first

##
# Every station ends up in the common frame of mooring_comp.jl:
#   z    : depth below the instantaneous sea surface (m, POSITIVE DOWN), (nz, nt), shallowest row first
#   u, v : TRUE eastward / northward velocity (m/s)
#   temp, salt : °C / PSU;  zeta : sea surface height (m)

# --- ROMS: <name>.nc in joint_ext_path, u/v along the grid axes -> rotated by the angle stored in <name>_mor ---
function load_roms(name)
    fname = joinpath(joint_ext_path, "$name.nc")
    ocean_time, zeta, temp, salt, u, v = NCDataset(fname) do ds
        # Julia reads dims reversed: zeta is (np, time), 3-D vars are (np, s_rho, time); np = 1 so drop it
        ds["ocean_time"][:], ds["zeta"][1, :],
        ds["temp"][1, :, :], ds["salt"][1, :, :], ds["u"][1, :, :], ds["v"][1, :, :]
    end
    i_mor, j_mor, ang_mor = NCDataset(mooring_loc) do ds
        ds["$(name)_mor"][1, :]
    end
    # water depth at the mooring point: bilinear in h; the Julia rho index is the object index + 1.5
    h_mor = let x = i_mor + 1.5, y = j_mor + 1.5
        i0, j0 = floor(Int, x), floor(Int, y)
        cx, cy = x - i0, y - j0
        (1 - cx) * (1 - cy) * h[i0, j0] + cx * (1 - cy) * h[i0+1, j0] +
        (1 - cx) * cy * h[i0, j0+1] + cx * cy * h[i0+1, j0+1]
    end
    N, nt = size(temp, 1), length(ocean_time)
    z_r, _ = zlevs3(fill(h_mor, 1, nt), reshape(zeta, 1, nt), theta_s, theta_b, hc, N, "r", "new2008")
    z_r = z_r[:, 1, :]
    return (
        time = t_ref .+ Millisecond.(round.(Int, ocean_time .* 1000)),   # ocean_time is in SECONDS since t_ref
        z    = flipz(zeta' .- z_r),
        u    = flipz(u .* cos(ang_mor) .- v .* sin(ang_mor)),
        v    = flipz(u .* sin(ang_mor) .+ v .* cos(ang_mor)),
        temp = flipz(temp),
        salt = flipz(salt),
        zeta = zeta,
        h    = h_mor,
    )
end

# --- NCOM: NCOM_<name>.nc, hourly, 201 fixed z levels (0 to -1000 m), u/v already east/north ---
function load_ncom(name)
    fname = joinpath(NCOM_mooring_path, "NCOM_$name.nc")
    NCDataset(fname) do ds
        time = ds["time"][:]
        (
            time = time,
            z    = repeat(flipz(abs.(ds["z"][:])), 1, length(time)),
            u    = flipz(ds["u"][:, :]),
            v    = flipz(ds["v"][:, :]),
            temp = flipz(ds["temp"][:, :]),
            salt = flipz(ds["salt"][:, :]),
            zeta = ds["ssh"][:],
            h    = ds["h"][],
        )
    end
end

# --- CMEMS: GLOBAL_<name>.nc, 6-hourly T/S/u/v on 30 z levels; ssh hourly on its own time axis ---
function load_cmems(name)
    fname = joinpath(CMEMS_mooring_path, "GLOBAL_$name.nc")
    NCDataset(fname) do ds
        time = ds["time"][:]
        (
            time     = time,
            z        = repeat(flipz(abs.(ds["z"][:])), 1, length(time)),
            u        = flipz(ds["u"][:, :]),
            v        = flipz(ds["v"][:, :]),
            temp     = flipz(ds["temp"][:, :]),
            salt     = flipz(ds["salt"][:, :]),
            # sea level: hourly, on its own time axis time_ssh
            time_ssh = ds["time_ssh"][:],
            zeta     = ds["ssh"][:],             # sea surface height without tides
            total_sl = ds["total_sea_level"][:], # tide + inverse barometer + ssh + global mean steric/mass terms
            tide     = ds["ocean_tide"][:],      # ocean tide elevation
        )
    end
end

# --- CMEMS transect (e.g. M1_CPIES7): 3-D vars come out as (z, x, time); z flipped to surface-first ---
function load_cmems_transect(name)
    fname = joinpath(CMEMS_mooring_path, "GLOBAL_$name.nc")
    tr = NCDataset(fname) do ds
        rd(v) = reverse(ds[v][:, :, :], dims=1)
        (
            time    = ds["time"][:],
            z       = flipz(abs.(ds["z"][:])),   # (30,), positive down
            lon     = ds["lon"][:], lat = ds["lat"][:],
            dist    = ds["dist"][:],             # km from M1
            angle   = ds["angle"][],             # transect direction, degrees ccw from east
            mooring = ds["mooring"][:], mooring_dist = ds["mooring_dist"][:], mooring_offset = ds["mooring_offset"][:],
            temp = rd("temp"), salt = rd("salt"), u = rd("u"), v = rd("v"),
            u_along = rd("u_along"), u_cross = rd("u_cross"),
        )
    end
    # sea level along the transect: separate file GLOBAL_<name>_ssh.nc, hourly, 2-D vars come out as (x, time)
    fssh = joinpath(CMEMS_mooring_path, "GLOBAL_$(name)_ssh.nc")
    isfile(fssh) || return tr
    ssh = NCDataset(fssh) do ds
        (time_ssh = ds["time"][:],
         zeta     = ds["sea_surface_height"][:, :],   # without tides
         total_sl = ds["total_sea_level"][:, :],
         tide     = ds["ocean_tide"][:, :])
    end
    return merge(tr, ssh)
end

##
# --- load every station in mooring_to_comp from each source that has it ---
# Point stations go into roms/ncom/cmems Dicts; transects (only CMEMS has them) go into cmems_tr.
roms, ncom, cmems, cmems_tr = Dict{String,Any}(), Dict{String,Any}(), Dict{String,Any}(), Dict{String,Any}()
for name in mooring_to_comp
    srcs = (("ROMS",  roms,  joinpath(joint_ext_path, "$name.nc"),               load_roms),
            ("NCOM",  ncom,  joinpath(NCOM_mooring_path, "NCOM_$name.nc"),       load_ncom),
            ("CMEMS", cmems, joinpath(CMEMS_mooring_path, "GLOBAL_$name.nc"),    load_cmems))
    for (label, store, fname, loader) in srcs
        if !isfile(fname)
            println("[$label] $name: no file ($fname), skipped")
            continue
        end
        if NCDataset(ds -> haskey(ds.dim, "x"), fname)   # transect file, not a point
            if label == "CMEMS"
                cmems_tr[name] = load_cmems_transect(name)
                println("[$label] $name: transect, size(temp) = ", size(cmems_tr[name].temp))
            else
                println("[$label] $name: transect file, read in the Hovmöller section")
            end
            continue
        end
        store[name] = loader(name)
        s = store[name]
        println("[$label] $name: ", s.time[1], " to ", s.time[end], ", size(temp) = ", size(s.temp))
    end
end


## ploting on M1 M3 CPIES7
# 3 columns (stations) x 2 rows (models), one figure per quantity: temp, salt, u, v, speed, direction.
# Every field is first put on one regular depth grid zg (no extrapolation, so CMEMS is blank below 380 m).
# Speed and direction are computed AFTER u and v are gridded: interpolating u, v is safe, while interpolating
# an angle across the 0/360 jump would not be.
plot_stations = ["M1", "M3", "CPIES7"]
plot_models   = (("CMEMS", cmems), ("NCOM", ncom))   # one row each, top to bottom
zlim_comp = (0.0, 400.0)                             # depth range drawn (m); CMEMS stops at 380 m
zg = collect(zlim_comp[1]:2.0:zlim_comp[2])

# same interpolation as mooring_comp.jl: linear in depth, column by column, NaN outside the valid data
function to_zgrid(z, f, zg)
    out = fill(NaN, length(zg), size(f, 2))
    for t in axes(f, 2)
        good = findall(.!isnan.(z[:, t]) .& .!isnan.(f[:, t]))
        length(good) < 2 && continue
        zt, ft = z[good, t], f[good, t]
        for (k, d) in enumerate(zg)
            (d < zt[1] || d > zt[end]) && continue
            i = clamp(searchsortedlast(zt, d), 1, length(zt) - 1)
            w = (d - zt[i]) / (zt[i+1] - zt[i])
            out[k, t] = (1 - w) * ft[i] + w * ft[i+1]
        end
    end
    return out
end

# gridded fields: gridded[model][station] = (time, temp, salt, u, v, speed, dir), each (length(zg), nt)
gridded = Dict(mname => Dict(name => let s = store[name]
        ug, vg = to_zgrid(s.z, s.u, zg), to_zgrid(s.z, s.v, zg)
        (time = s.time, temp = to_zgrid(s.z, s.temp, zg), salt = to_zgrid(s.z, s.salt, zg),
         u = ug, v = vg, speed = hypot.(ug, vg),
         dir = mod.(atand.(ug, vg), 360))    # direction toward, compass: 0° = north, 90° = east
    end for name in plot_stations) for (mname, store) in plot_models)

# time axis as days since t0, labelled with the dates
t0 = DateTime(2022, 8, 22)
t1 = DateTime(2022, 11, 30)
tday(t) = Dates.value.(t .- t0) ./ 86_400_000
date_ticks = DateTime(2022, 9, 1):Month(1):t1
xticks = (tday(date_ticks), Dates.format.(date_ticks, "mm/dd"))

# HYCOM (GOFS 3.5) snapshots: irregular in time (a few per month), 40 fixed z levels; variables come out as (z, time)
# with index 1 the deepest, so flipped to surface-first like the others. Only snapshots inside [t0, t1] are kept.
# Snapshots, not averages, and too few to filter: they are shown as-is in the low-pass figures too.
function load_hycom(name)
    NCDataset(joinpath(HYCOM_mooring_path, "HYCOM_$name.nc")) do ds
        time = ds["time"][:]
        keep = findall(t -> t0 <= t <= t1, time)
        rd(v) = flipz(nomissing(ds[v][:, keep], NaN))
        u, v = rd("u"), rd("v")
        (time = time[keep], z = flipz(abs.(ds["z"][:])),     # z: (40,), positive down
         temp = rd("temp"), salt = rd("salt"), u = u, v = v,
         speed = hypot.(u, v), dir = mod.(atand.(u, v), 360))
    end
end
hycom = Dict(name => load_hycom(name) for name in plot_stations)
hycom_time = Dict(name => hycom[name].time for name in plot_stations)   # marked as cyan lines on the heatmaps

# for a direction profile: put a NaN gap between neighbours more than 180° apart (a wrap across 0/360),
# so the line breaks there instead of cutting across the whole panel
function break_wraps(f, z)
    fo, zo = [f[1]], [z[1]]
    for k in 2:length(f)
        abs(f[k] - f[k-1]) > 180 && (push!(fo, NaN); push!(zo, NaN))
        push!(fo, f[k]); push!(zo, z[k])
    end
    return fo, zo
end

# data = gridded[model][station] structure and z = its depth axis (the 2 m grid by default)
function plot_station_grid(q; clim, colormap, levels, cblabel, title, fname,
                           contours = true, contour_color = (:white, 0.6), data = gridded, z = zg)
    fig = Figure(size = (1600, 1100))
    axs, hm = Axis[], nothing
    for (row, (mname, _)) in enumerate(plot_models), (col, name) in enumerate(plot_stations)
        g = data[mname][name]
        f = getproperty(g, q)
        ax = Axis(fig[row, col]; title = row == 1 ? name : "", titlesize = 18,
            ylabel = col == 1 ? "$mname\nDepth (m)" : "", yreversed = true, xticks = xticks,
            xgridvisible = false, ygridvisible = false)
        hm = heatmap!(ax, tday(g.time), z, f'; colormap = colormap, colorrange = clim,
                      lowclip = :black, highclip = :magenta, nan_color = :transparent)
        contours && contour!(ax, tday(g.time), z, f'; levels = levels, color = contour_color, linewidth = 0.8)
        vlines!(ax, tday(hycom_time[name]); color = :cyan, linewidth = 2, linestyle = :dash)   # HYCOM times
        push!(axs, ax)
    end
    linkaxes!(axs...)
    xlims!(axs[1], tday(t0), tday(t1))
    ylims!(axs[1], zlim_comp[2], zlim_comp[1])
    nr, nc = length(plot_models), length(plot_stations)
    for (k, ax) in enumerate(axs)
        row, col = divrem(k - 1, nc) .+ 1
        row < nr && hidexdecorations!(ax; ticks = false)
        col > 1  && hideydecorations!(ax; ticks = false)
        row == nr && (ax.xlabel = "Date (2022)")
    end
    # bottom row: HYCOM profiles, one small panel per snapshot under each station; x = value, y = depth.
    # The dots use the heatmap's colormap and range, so a dot has the colour that value has in the panels above.
    paxs = Axis[]
    for (col, name) in enumerate(plot_stations)
        hy = hycom[name]
        gl = fig[nr + 1, col] = GridLayout()
        for (k, t) in enumerate(hy.time)
            f = getproperty(hy, q)[:, k]
            ax = Axis(gl[1, k]; title = Dates.format(t, "mm/dd HH:MM"), titlesize = 12, titlecolor = :darkcyan,
                yreversed = true, xticks = range(clim..., length = 3), xticklabelsize = 10,
                xticklabelrotation = pi / 4, ylabel = (col == 1 && k == 1) ? "HYCOM\nDepth (m)" : "")
            lines!(ax, (q == :dir ? break_wraps(f, hy.z) : (f, hy.z))...; color = (:gray40, 0.8), linewidth = 1)
            scatter!(ax, f, hy.z; color = f, colormap = colormap, colorrange = clim,
                     lowclip = :black, highclip = :magenta, markersize = 16, strokecolor = :black, strokewidth = 0.3)
            xlims!(ax, clim...)
            (col > 1 || k > 1) && hideydecorations!(ax; ticks = false, grid = false)
            push!(paxs, ax)
        end
        colgap!(gl, 6)
    end
    linkyaxes!(axs[1], paxs...)
    ylims!(axs[1], zlim_comp[2], zlim_comp[1])   # again: linking would otherwise widen to the full HYCOM depth
    rowsize!(fig.layout, nr + 1, Relative(0.3))
    Colorbar(fig[1:nr+1, nc + 1], hm; label = cblabel, ticks = levels)
    Label(fig[0, :], title, fontsize = 20)
    outname = joinpath(figure_path, fname)
    save(outname, fig)
    println("saved ", outname)
    return fig
end

##
plot_station_grid(:temp; clim = (6, 30), levels = 6:4:30, colormap = :thermal,
    cblabel = "Potential temperature (°C)", title = "Temperature: CMEMS vs NCOM", fname = "temp_station_comp.png")
plot_station_grid(:salt; clim = (34.5, 36.7), levels = 34.5:0.5:36.5, colormap = :haline,
    cblabel = "Salinity (PSU)", title = "Salinity: CMEMS vs NCOM", fname = "salt_station_comp.png")
plot_station_grid(:u; clim = (-1.5, 1.5), levels = -1.5:0.5:1.5, colormap = :balance, contour_color = (:black, 0.3),
    cblabel = "u (m/s)", title = "Eastward velocity u: CMEMS vs NCOM", fname = "u_station_comp.png")
plot_station_grid(:v; clim = (-1.5, 1.5), levels = -1.5:0.5:1.5, colormap = :balance, contour_color = (:black, 0.3),
    cblabel = "v (m/s)", title = "Northward velocity v: CMEMS vs NCOM", fname = "v_station_comp.png")
plot_station_grid(:speed; clim = (0, 1.5), levels = 0:0.25:1.5, colormap = :speed, contour_color = (:black, 0.3),
    cblabel = "Speed (m/s)", title = "Current speed: CMEMS vs NCOM", fname = "speed_station_comp.png")
# :phase is cyclic, so 0° and 360° get the same colour; no contours (they would pile up on the 0/360 jump)
plot_station_grid(:dir; clim = (0, 360), levels = 0:90:360, colormap = :phase, contours = false,
    cblabel = "Direction toward (° clockwise from north)", title = "Current direction: CMEMS vs NCOM",
    fname = "direction_station_comp.png")


## 28h lp compare.
##
# --- 28 h low-pass on the CMEMS fixed depths ---
# The filter runs row by row (one depth = one time series), so every source has to be on the same fixed depths
# first. CMEMS already is: its 30 product levels (0.5 … 380 m, uneven). NCOM (201 levels every 5 m) is
# interpolated onto those depths, then both are low-passed. Speed and direction come from the low-passed u, v.
Tcut = 28.0                # low-pass cutoff period (HOURS)
filt_order = 4
z_fix = cmems[plot_stations[1]].z[:, 1]   # CMEMS level depths (m, positive down, shallowest first), same at every station

# Butterworth + filtfilt needs an evenly sampled series with no NaN, so for each row of a (nz, nt) field:
# keep only the span between the first and last valid sample, fill the gaps inside it by linear interpolation
# in time, filter, then put NaN back where the data were missing. (Same as filter_rows in mooring_comp.jl.)
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
        yf = lowhighpass_butter(y, Tcut, dt, N, kind)
        out[k, good] = yf[good .- good[1] .+ 1]
    end
    return out
end

# lp[model][station] = (time, temp, salt, u, v, speed, dir), each (length(z_fix), nt), 28 h low-passed
lp = Dict(mname => Dict(name => let s = store[name]
        lpf(f) = filter_rows(s.time, to_zgrid(s.z, f, z_fix), Tcut, "low")   # on CMEMS levels this is the identity
        ul, vl = lpf(s.u), lpf(s.v)
        (time = s.time, temp = lpf(s.temp), salt = lpf(s.salt),
         u = ul, v = vl, speed = hypot.(ul, vl), dir = mod.(atand.(ul, vl), 360))
    end for name in plot_stations) for (mname, store) in plot_models)

##
tc = round(Int, Tcut)
lp_kw = (data = lp, z = z_fix)
plot_station_grid(:temp; lp_kw..., clim = (6, 30), levels = 6:4:30, colormap = :thermal,
    cblabel = "Potential temperature (°C)", title = "Temperature, $tc h low-pass: CMEMS vs NCOM (on CMEMS levels)",
    fname = "temp_station_comp_lp$(tc)h.png")
plot_station_grid(:salt; lp_kw..., clim = (34.5, 36.7), levels = 34.5:0.5:36.5, colormap = :haline,
    cblabel = "Salinity (PSU)", title = "Salinity, $tc h low-pass: CMEMS vs NCOM (on CMEMS levels)",
    fname = "salt_station_comp_lp$(tc)h.png")
plot_station_grid(:u; lp_kw..., clim = (-1.5, 1.5), levels = -1.5:0.5:1.5, colormap = :balance, contour_color = (:black, 0.3),
    cblabel = "u (m/s)", title = "Eastward velocity u, $tc h low-pass: CMEMS vs NCOM (on CMEMS levels)",
    fname = "u_station_comp_lp$(tc)h.png")
plot_station_grid(:v; lp_kw..., clim = (-1.5, 1.5), levels = -1.5:0.5:1.5, colormap = :balance, contour_color = (:black, 0.3),
    cblabel = "v (m/s)", title = "Northward velocity v, $tc h low-pass: CMEMS vs NCOM (on CMEMS levels)",
    fname = "v_station_comp_lp$(tc)h.png")
plot_station_grid(:speed; lp_kw..., clim = (0, 1.5), levels = 0:0.25:1.5, colormap = :speed, contour_color = (:black, 0.3),
    cblabel = "Speed (m/s)", title = "Current speed, $tc h low-pass: CMEMS vs NCOM (on CMEMS levels)",
    fname = "speed_station_comp_lp$(tc)h.png")
plot_station_grid(:dir; lp_kw..., clim = (0, 360), levels = 0:90:360, colormap = :phase, contours = false,
    cblabel = "Direction toward (° clockwise from north)",
    title = "Current direction, $tc h low-pass: CMEMS vs NCOM (on CMEMS levels)", fname = "direction_station_comp_lp$(tc)h.png")

##
# ============================================================================================================
# Hovmöller diagrams along the M1 -> CPIES7 transect: x = longitude along the line, y = time
# ============================================================================================================
# Surface only for now. Both transect files store 3-D vars as (z, x, time) with z ASCENDING (bottom -> surface),
# so the surface is the level with the largest z (CMEMS -0.5 m, NCOM 0 m).
# Sea level:
#   NCOM ssh includes tides. CMEMS gives sea_surface_height WITHOUT tides and total_sea_level WITH them, on a
#   separate hourly file that starts on 09/01. The unfiltered figure uses CMEMS total_sea_level (closest to NCOM);
#   the low-passed one uses CMEMS sea_surface_height. Each model's own mean is removed (the two models' sea level
#   reference levels differ), so the panels show the sea level anomaly.
transect_name = "M1_CPIES7"

function load_transect_surface(fname; ssh_file = nothing)
    tr = NCDataset(fname) do ds
        z = ds["z"][:]
        ks = argmax(z)                                    # surface level
        rd(v) = Float64.(nomissing(ds[v][ks, :, :], NaN)) # (x, time)
        (lon = ds["lon"][:], lat = ds["lat"][:], dist = ds["dist"][:], z_surf = abs(z[ks]),
         mooring = ds["mooring"][:], mooring_dist = ds["mooring_dist"][:],
         time = ds["time"][:], temp = rd("temp"), salt = rd("salt"), u = rd("u"), v = rd("v"),
         time_ssh = ds["time"][:],
         ssh = haskey(ds, "ssh") ? Float64.(nomissing(ds["ssh"][:, :], NaN)) : nothing,
         ssh_lp_src = nothing)
    end
    ssh_file === nothing && return tr
    # CMEMS: sea level from its own hourly file; total (with tides) for the raw figure, ssh (no tides) for the low-pass
    NCDataset(ssh_file) do ds
        merge(tr, (time_ssh = ds["time"][:], ssh = Float64.(nomissing(ds["total_sea_level"][:, :], NaN)),
                   ssh_lp_src = Float64.(nomissing(ds["sea_surface_height"][:, :], NaN))))
    end
end

hov_files = (("CMEMS", joinpath(CMEMS_mooring_path, "GLOBAL_$transect_name.nc"),
                       joinpath(CMEMS_mooring_path, "GLOBAL_$(transect_name)_ssh.nc")),
             ("NCOM",  joinpath(NCOM_mooring_path, "NCOM_$transect_name.nc"), nothing))
hov_models = String[]
hov = Dict{String,Any}()
for (mname, f, fssh) in hov_files
    if !isfile(f)
        println("[$mname] no transect file ($f): left out of the Hovmöller figures")
        continue
    end
    hov[mname] = load_transect_surface(f; ssh_file = fssh)
    push!(hov_models, mname)
    println("[$mname] transect surface (", hov[mname].z_surf, " m): size(temp) = ", size(hov[mname].temp))
end

# 28 h low-pass of every surface field, point by point along the line: the fields are (x, time), so each row of
# filter_rows is the time series at one point.
# Sea level: low-pass of NCOM ssh (removes its tides) and of CMEMS sea_surface_height (has none to remove).
hov_lp = Dict(m => let h = hov[m]
        lpx(t, f) = filter_rows(t, f, Tcut, "low")
        (time = h.time, temp = lpx(h.time, h.temp), salt = lpx(h.time, h.salt),
         u = lpx(h.time, h.u), v = lpx(h.time, h.v), time_ssh = h.time_ssh,
         ssh = lpx(h.time_ssh, h.ssh_lp_src === nothing ? h.ssh : h.ssh_lp_src))
    end for m in hov_models)

# sea level anomaly: remove the model's own mean over [t0, t1] (one number per model)
function ssh_anom(time, ssh)
    keep = findall(t -> t0 <= t <= t1, time)
    return ssh .- mean(filter(!isnan, ssh[:, keep]))
end

# colour range from the data of all models shown: 0.5-99.5th percentile, symmetric about 0 for signed fields
function auto_clim(fields; symmetric = false)
    vals = filter(!isnan, reduce(vcat, vec.(fields)))
    symmetric && return (m = quantile(abs.(vals), 0.995); (-m, m))
    return (quantile(vals, 0.005), quantile(vals, 0.995))
end

# One panel per model, side by side; time runs down the page (top = t0). Moorings on the line are marked
# with grey dashed lines and named on a second x axis along the top; HYCOM snapshot times are cyan horizontal lines.
function plot_hovmoller(data, q; clim, colormap, cblabel, title, fname)
    fig = Figure(size = (550 * length(hov_models) + 150, 950))
    axs, hm = Axis[], nothing
    for (col, m) in enumerate(hov_models)
        h = data[m]
        time = q == :ssh ? h.time_ssh : h.time
        f = getproperty(h, q)
        q == :ssh && (f = ssh_anom(time, f))
        lon = hov[m].lon
        ax = Axis(fig[1, col]; xlabel = "Longitude (°E)",
            ylabel = col == 1 ? "Date (2022)" : "", yreversed = true,
            yticks = xticks, xgridvisible = false, ygridvisible = false)   # xticks = the (days, "mm/dd") date ticks
        hm = heatmap!(ax, lon, tday(time), f; colormap = colormap, colorrange = clim,
                      lowclip = :black, highclip = :magenta, nan_color = :transparent)
        # moorings on the line: longitude at their along-line distance
        mlon = [lon[clamp(searchsortedfirst(hov[m].dist, d), 1, length(lon))] for d in hov[m].mooring_dist]
        vlines!(ax, mlon; color = (:gray20, 0.6), linestyle = :dash, linewidth = 1)
        axt = Axis(fig[1, col]; xaxisposition = :top, xticks = (mlon, hov[m].mooring), xticklabelsize = 11,
            xticklabelrotation = pi / 3, xgridvisible = false, ygridvisible = false)   # mooring names on top
        hidespines!(axt); hideydecorations!(axt)
        linkxaxes!(ax, axt)
        hlines!(ax, tday(hycom_time[plot_stations[1]]); color = :cyan, linewidth = 2, linestyle = :dash)
        Label(fig[0, col], m, fontsize = 18, font = :bold, tellwidth = false)   # tellwidth = false: else the column shrinks to the label
        col > 1 && hideydecorations!(ax; ticks = false)
        push!(axs, ax)
    end
    linkaxes!(axs...)
    ylims!(axs[1], tday(t1), tday(t0))
    xlims!(axs[1], extrema(hov[hov_models[1]].lon)...)   # exactly the ends of the line, M1 to CPIES7
    Colorbar(fig[1, length(hov_models) + 1], hm; label = cblabel)
    Label(fig[-1, :], title, fontsize = 20)
    outname = joinpath(figure_path, fname)
    save(outname, fig)
    println("saved ", outname)
    return fig
end

##
# surface fields: unfiltered and 28 h low-pass; colour ranges shared by the two versions (taken from the raw data)
hov_vars = (
    (:temp, "Surface temperature",          "Potential temperature (°C)", :thermal, false),
    (:salt, "Surface salinity",             "Salinity (PSU)",             :haline,  false),
    (:ssh,  "Sea level anomaly",            "Sea level anomaly (m)",      :balance, true),
    (:u,    "Surface eastward velocity u",  "u (m/s)",                    :balance, true),
    (:v,    "Surface northward velocity v", "v (m/s)",                    :balance, true),
)
for (q, long, cblabel, cmap, sym) in hov_vars
    raw = [q == :ssh ? ssh_anom(hov[m].time_ssh, hov[m].ssh) : getproperty(hov[m], q) for m in hov_models]
    clim = auto_clim(raw; symmetric = sym)
    ssh_note = q == :ssh ? " (CMEMS: total sea level, incl. tides)" : ""
    plot_hovmoller(hov, q; clim = clim, colormap = cmap, cblabel = cblabel,
        title = "$long along $(replace(transect_name, "_" => " – "))$ssh_note", fname = "hov_surf_$(q).png")
    # low-pass: sea level gets its own (smaller) range, since the tides are gone
    clim_lp = q == :ssh ? auto_clim([ssh_anom(hov_lp[m].time_ssh, hov_lp[m].ssh) for m in hov_models]; symmetric = true) : clim
    plot_hovmoller(hov_lp, q; clim = clim_lp, colormap = cmap, cblabel = cblabel,
        title = "$long along $(replace(transect_name, "_" => " – ")), $tc h low-pass", fname = "hov_surf_$(q)_lp$(tc)h.png")
end
