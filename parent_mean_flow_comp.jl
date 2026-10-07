using Distributed

n_workers = max(1, parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "5")) - 1)   # -1 reserves a CPU for this main process
# only add workers once — re-running this cell in the same REPL session
# would otherwise stack another n_workers on top of the existing ones
nprocs() == 1 && addprocs(n_workers)
println("running with $(nworkers()) worker processes")

@everywhere begin
    using NCDatasets, CairoMakie, Dates, Statistics, LinearAlgebra
    CairoMakie.activate!()
    include("/home/hsinyi/Documents/Julia/function/load_all.jl")
    include("/home/hsinyi/Documents/Julia/function/plotting_fun.jl")

    ## path setting
    grid_fname = "/home/hsinyi/roms_data/grid/roms_grd_900m.nc"   # the grid_file listed in the .nc's global attributes
    parent_grid = "/home/mbui/ModelOutput/NCOM/grid/ohgrd_2.nc"
    figure_path = "/home/hsinyi/figure/20261006_model_comp"

    NCdatadir     = "/home/hsinyi/roms_data/NCOM_DATA_NC/"
    GLOBALdata    = "/home/hsinyi/data_notm/GLOBAL_ANALYSISFORECAST.nc"
end

# days to read (inclusive, through 00:00 of the day after day_end);
# the full period is Date(2022, 8, 22) to Date(2022, 11, 30)
day_start = Date(2022, 8, 24)
day_end   = Date(2022, 11, 29)

## grids and plot settings — needed by every worker (they read the NCOM files and draw the figures)
@everywhere begin
    # NCOM (parent) grid. u_velocity/v_velocity are (xi, eta, z, time) and ssh is
    # (xi, eta, time), all on the same unstaggered grid as lon_par/lat_par.
    lon_par, lat_par, mask_par, depth_par = NCDataset(parent_grid) do ds
        # h is negative-down with `missing` over land (see check_NCOM_parallel.jl)
        ds["lon"][:, :], ds["lat"][:, :], ds["mask"][:, :], abs.(coalesce.(ds["h"][:, :], NaN32))
    end
    lon_par_f = Float64.(lon_par)
    lat_par_f = Float64.(lat_par)
    bathy_levels = [500, 1000, 2000]

    # GLOBAL grid: 1D lon/lat vectors -> 2D (lon, lat) matrices, same layout as lon_par/lat_par
    GLOBAL_lon, GLOBAL_lat = NCDataset(GLOBALdata) do ds
        lon, lat = ds["longitude"][:], ds["latitude"][:]
        repeat(lon, 1, length(lat)), repeat(lat', length(lon), 1)
    end

    # child (ROMS) grid, only for its outline and extent
    lon_chd, lat_chd = NCDataset(grid_fname) do ds
        ds["lon_rho"][:, :], ds["lat_rho"][:, :]
    end
    lon_chd[lon_chd .> 180] .-= 360   # convert 0-360 convention to -180/180 (east/west hemisphere)
    lon_chd_b, lat_chd_b = grid_boundary(lon_chd, lat_chd)   # child-grid outline, overlaid on both panels
    lon_chd_lims = extrema(lon_chd)   # child-grid extent, both panels are zoomed to it
    lat_chd_lims = extrema(lat_chd)

    moor_lon, moor_lat = -45.13, 3.95   # French mooring

    zeta_clim = (-0.3, 0.3)   # detided and mooring-referenced SSH is only a few tens of cm
    temp_clim = (26, 31)
    speed_clim = (0, 2)
    uv_skip_par = 50   # quiver downsampling: every 50th NCOM point and every 5th GLOBAL point
    uv_skip_glb = 5    # are both roughly 45 km apart, so the two panels get a similar arrow density

    # MT in the NCOM files is days since 1900-12-31 00:00
    mt2datetime(mt) = DateTime(1900, 12, 31) + Millisecond(round(Int, mt * 86_400_000))
end

## GLOBAL surface data (small, read on the main process only)
GLOBAL_time, GLOBAL_dep, GLOBAL_surftemp,
 GLOBAL_surfu, GLOBAL_surfv , GLOBAL_zeta= NCDataset(GLOBALdata) do ds
    show(ds)
    t_all = ds["time"][:]
    it = findfirst(>=(DateTime(day_start)), t_all):findlast(<=(DateTime(day_end) + Day(1)), t_all)
    t_all[it], ds["depth"][1],
    ds["thetao"][:,:,1,it], ds["uo"][:,:,1,it], ds["vo"][:,:,1,it],
    ds["zos"][:,:,1,it]
end

## NCOM (parent) surface data — one daily file per worker
# surface = top layer k=1 (centred at ~0.5 m, cf. 0.494 m in the GLOBAL file).

# Runs on a worker: surface fields of one daily file. Each file holds 25 hourly
# records (00..24 h); record 25 is the same time as record 1 of the next day,
# so it is only kept (keep_last) for the last day.
@everywhere function load_ncom_day(datadir, tag, keep_last)
    ds_ts  = NCDataset(joinpath(datadir, tag * "_ts.nc"))
    ds_uv  = NCDataset(joinpath(datadir, tag * "_uv.nc"))
    ds_ssh = NCDataset(joinpath(datadir, tag * "_ssh.nc"))
    try
        MT = ds_ts["MT"][:]
        recs = keep_last ? eachindex(MT) : eachindex(MT)[1:end-1]
        nx, ny = size(ds_ssh["ssh"])[1:2]
        temp, u, v, ssh = (Array{Float32}(undef, nx, ny, length(recs)) for _ in 1:4)
        for (n, t) in enumerate(recs)
            temp[:, :, n] = ds_ts["layer_temperature"][:, :, 1, t]
            u[:, :, n]    = ds_uv["u_velocity"][:, :, 1, t]
            v[:, :, n]    = ds_uv["v_velocity"][:, :, 1, t]
            ssh[:, :, n]  = ds_ssh["ssh"][:, :, t]
        end
        # incomplete records are left at the raw NetCDF fill value (~9.97e36)
        for F in (temp, u, v, ssh)
            F[abs.(F) .> 1f30] .= NaN32
        end
        return mt2datetime.(MT[recs]), temp, u, v, ssh
    finally
        close(ds_ts); close(ds_uv); close(ds_ssh)
    end
end

# Runs on the main process: hands the daily files out to the workers and
# collects each day straight into its slot of the full arrays.
function load_ncom_surface(datadir, tags)
    nx, ny = NCDataset(joinpath(datadir, tags[1] * "_ssh.nc")) do ds
        size(ds["ssh"])[1:2]
    end
    nt = 24 * length(tags) + 1
    time_out = Vector{DateTime}(undef, nt)
    # Float32: the full grid is ~16 GB per variable for the whole period
    temp, u, v, ssh = (Array{Float32}(undef, nx, ny, nt) for _ in 1:4)

    pool = default_worker_pool()
    asyncmap(eachindex(tags); ntasks = nworkers()) do k
        keep_last = k == length(tags)
        t_d, temp_d, u_d, v_d, ssh_d = remotecall_fetch(load_ncom_day, pool, datadir, tags[k], keep_last)
        # each day's slot is fixed in advance, so every file must hold the full 25 records
        length(t_d) == (keep_last ? 25 : 24) ||
            error("$(tags[k]): expected 25 hourly records, found $(length(t_d) + !keep_last)")
        r = 24 * (k - 1) .+ (1:length(t_d))
        time_out[r] = t_d
        temp[:, :, r] = temp_d
        u[:, :, r]    = u_d
        v[:, :, r]    = v_d
        ssh[:, :, r]  = ssh_d
        println("done $(tags[k])")
    end
    return time_out, temp, u, v, ssh
end

ncom_tags = sort([replace(f, "_uv.nc" => "") for f in readdir(NCdatadir) if endswith(f, "_uv.nc")])
ncom_tags = filter(tag -> day_start <= Date(tag[1:8], "yyyymmdd") <= day_end, ncom_tags)
time_par, surftemp_par, surfu_par, surfv_par, zeta_par = load_ncom_surface(NCdatadir, ncom_tags)

## detide NCOM (GLOBAL has no tides) and put both SSH on the same reference
# The fit needs the whole time series of every point at once, so this part
# stays on the main process (it is a couple of matrix products, already
# multi-threaded through BLAS).
# main tidal constituents, periods in hours. These six can be fitted from a
# record as short as ~7 days; with a month or more, Q1 = 26.868350 and
# MS4 = 6.103339275 can be added.
tide_periods = (M2 = 12.4206012, S2 = 12.0, N2 = 12.65834751,
                K1 = 23.93447213, O1 = 25.81933871, M4 = 6.210300601)

"""
    detide(F, t; periods = values(tide_periods))

Remove the tides from `F` (nx, ny, nt) by a least-squares harmonic fit along
time at every grid point. Only the harmonic part is subtracted, so the time
mean and the non-tidal variability stay. `t` is the DateTime of each record.
A point that is NaN at any time comes back NaN at all times.
"""
function detide(F::AbstractArray{<:Real, 3}, t::AbstractVector{DateTime}; periods = values(tide_periods))
    th = Dates.value.(t .- t[1]) ./ 3.6e6                 # hours since the first record
    A = hcat(ones(length(th)), (f.(2pi .* th ./ p) for f in (cos, sin) for p in periods)...)   # nt x (1 + 2*nperiods)
    nx, ny, nt = size(F)
    Y = reshape(F, nx * ny, nt)                           # one row per grid point
    C = Y * Float32.(pinv(A))'                            # least-squares coefficients of every point
    tide = C[:, 2:end] * Float32.(A[:, 2:end])'           # column 1 is the mean, which is kept
    return reshape(Y .- tide, nx, ny, nt)
end

zeta_par_dt = detide(zeta_par, time_par)
GLOBAL_zeta_f = coalesce.(GLOBAL_zeta, NaN32)   # land is `missing` in the GLOBAL file

# The two models do not share an SSH reference level, so each one is taken
# relative to its own time-mean SSH at the French mooring.
nearest_idx(lon, lat, lon0, lat0) = argmin((lon .- lon0) .^ 2 .+ (lat .- lat0) .^ 2)
i_par = nearest_idx(lon_par, lat_par, moor_lon, moor_lat)
i_glb = nearest_idx(GLOBAL_lon, GLOBAL_lat, moor_lon, moor_lat)
zeta_ref_par = mean(zeta_par_dt[i_par, :])
zeta_ref_glb = mean(GLOBAL_zeta_f[i_glb, :])
println("mean SSH at the mooring: NCOM $(zeta_ref_par) m, GLOBAL $(zeta_ref_glb) m")
zeta_par_dt .-= zeta_ref_par
GLOBAL_zeta_f .-= zeta_ref_glb

# surface currents: NCOM u/v are detided the same way, speed is then computed from the detided components
surfu_par_dt = detide(surfu_par, time_par)
surfv_par_dt = detide(surfv_par, time_par)
GLOBAL_surfu_f = coalesce.(GLOBAL_surfu, NaN32)
GLOBAL_surfv_f = coalesce.(GLOBAL_surfv, NaN32)
GLOBAL_surftemp_f = coalesce.(GLOBAL_surftemp, NaN32)

## SSH, surface temperature and surface speed comparison: NCOM (left) vs GLOBAL (right), one hour per worker

# one two-panel figure (NCOM left, GLOBAL right) sharing one colorbar;
# uv_par / uv_glb are optional (u, v) tuples drawn as quiver arrows on top
@everywhere function comp_figure(F_par, F_glb, outname; title, par_title = "NCOM", colormap, colorrange, label,
                                 uv_par = nothing, uv_glb = nothing)
    fig = Figure(size = (1400, 650), figure_padding = (20, 20, 50, 20))   # extra bottom room: Axis3 labels sit outside the layout
    Label(fig[0, 1:2], title; fontsize = 20)
    ax1 = topdown_axis3(fig[1, 1]; title = par_title)
    ax2 = topdown_axis3(fig[1, 2]; title = "GLOBAL")
    sp = plot_curvilinear!(ax1, lon_par, lat_par, F_par; colormap = colormap, colorrange = colorrange)
    plot_curvilinear!(ax2, GLOBAL_lon, GLOBAL_lat, F_glb; colormap = colormap, colorrange = colorrange)
    Colorbar(fig[1, 3], sp, label = label)
    for (ax, lon, lat, uv, skip) in ((ax1, lon_par, lat_par, uv_par, uv_skip_par),
                                     (ax2, GLOBAL_lon, GLOBAL_lat, uv_glb, uv_skip_glb))
        uv === nothing && continue
        quiver_curvilinear!(ax, lon, lat, uv...;
                            skip = skip, xlim = lon_chd_lims, ylim = lat_chd_lims,
                            lengthscale = 0.5, color = :black,
                            shaftwidth = 1.5, tipwidth = 5, tiplength = 6)
    end
    for ax in (ax1, ax2)
        contour!(ax, lon_par_f, lat_par_f, depth_par; levels = bathy_levels,
                 color = RGBf(0.3, 0.3, 0.3), linewidth = 1)
        lines!(ax, lon_chd_b, lat_chd_b, zeros(length(lon_chd_b));
               color = :black, linewidth = 2)
        scatter!(ax, [moor_lon], [moor_lat], [0.0]; color = :black, marker = :star5, markersize = 15)
        xlims!(ax, lon_chd_lims...)
        ylims!(ax, lat_chd_lims...)
    end
    save(outname, fig)
    println("saved plot to ", outname)
end

# Runs on a worker: the three figures of one hour. The fields are 2D slices
# sent over by the main process (_p = NCOM, _g = GLOBAL).
@everywhere function plot_frame(tstamp, zeta_p, zeta_g, temp_p, temp_g, u_p, v_p, u_g, v_g)
    str1 = Dates.format(tstamp, "yyyy-mm-dd HH:MM:SS")
    str2 = Dates.format(tstamp, "yyyymmdd_HHMM")

    comp_figure(zeta_p, zeta_g, joinpath(figure_path, "zeta_comp_$(str2).png");
                title = "Sea surface height relative to the mooring mean, $str1", par_title = "NCOM (detided)",
                colormap = Reverse(:RdBu), colorrange = zeta_clim, label = "meters")
    comp_figure(temp_p, temp_g, joinpath(figure_path, "temp_comp_$(str2).png");
                title = "Surface temperature, $str1",
                colormap = :thermal, colorrange = temp_clim, label = "°C")
    comp_figure(sqrt.(u_p .^ 2 .+ v_p .^ 2), sqrt.(u_g .^ 2 .+ v_g .^ 2), joinpath(figure_path, "speed_comp_$(str2).png");
                title = "Surface speed, $str1", par_title = "NCOM (detided)",
                colormap = :speed, colorrange = speed_clim, label = "m/s",
                uv_par = (u_p, v_p), uv_glb = (u_g, v_g))
end

mkpath(figure_path)
# The full arrays only live on the main process. remotecall_fetch sends each
# worker just the 2D slices of its own hour — a pmap closure over the arrays
# would instead copy all of them to every worker.
pool = default_worker_pool()
asyncmap(eachindex(time_par); ntasks = nworkers()) do t
    tg = findfirst(==(time_par[t]), GLOBAL_time)   # same hour in the GLOBAL data
    if tg === nothing
        println("no GLOBAL record at $(time_par[t]), skipping")
        return
    end
    try
        remotecall_fetch(plot_frame, pool, time_par[t],
                         zeta_par_dt[:, :, t], GLOBAL_zeta_f[:, :, tg],
                         surftemp_par[:, :, t], GLOBAL_surftemp_f[:, :, tg],
                         surfu_par_dt[:, :, t], surfv_par_dt[:, :, t],
                         GLOBAL_surfu_f[:, :, tg], GLOBAL_surfv_f[:, :, tg])
    catch ex
        println("frame $(time_par[t]) failed: ", ex)
    end
end
