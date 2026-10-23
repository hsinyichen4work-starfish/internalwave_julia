using Distributed

n_workers = max(1, parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "4")) - 1)   # -1 reserves a CPU for this main process
# default of 6 (-> 5 workers) is sized for this workstation's RAM, not core count —
# each worker keeps its own full copy of the grid arrays plus ~1.3GB/timestep of
# u/v/temp data, so more workers means more concurrent memory, not just more CPU.
# SLURM_CPUS_PER_TASK still overrides this when running under a SLURM allocation.
addprocs(n_workers)
println("running with $(nprocs() - 1) worker processes")

@everywhere begin
    using NCDatasets, CairoMakie, Dates, Statistics
    CairoMakie.activate!()
    include("/home/hsinyi/Documents/Julia/function/load_all.jl")
    include("/home/hsinyi/Documents/Julia/function/plotting_fun.jl")

    ## path setting
    child_grid = "/home/hsinyi/roms_data/grid/roms_grd_900m.nc"
    parent_grid = "/home/mbui/ModelOutput/NCOM/grid/ohgrd_1.nc"   # 2700 m NCOM
    datadir = "/home/hsinyi/roms_data/NCOM_DATA_NC_2700/"
    figure_path = "/home/hsinyi/figure/20261023_NCOM2700"
    figure_path_full = joinpath(figure_path, "full_domain")   # same plots, whole NCOM domain

    ##
    lon_chd, lat_chd = NCDataset(child_grid) do ds
        ds["lon_rho"][:, :], ds["lat_rho"][:, :]
    end
    lon_chd[lon_chd .> 180] .-= 360   # convert 0-360 convention to -180/180 (east/west hemisphere)
    lon_chd_b, lat_chd_b = grid_boundary(lon_chd, lat_chd)   # child-grid outline, for overlaying on parent-grid plots
    lon_chd_lims = extrema(lon_chd)   # child-grid extent, to zoom parent-grid plots to it (xlim/ylim equivalent)
    lat_chd_lims = extrema(lat_chd)

    lon_par, lat_par, h_par, mask, zm3, kb, dx_par, dy_par = NCDataset(parent_grid) do ds
        ds["lon"][:, :], ds["lat"][:, :], ds["h"][:,:],
        ds["mask"][:, :], ds["zm3"][:,:,:], ds["kb"][:,:], ds["dx"][:,:], ds["dy"][:,:]
    end
    # h is negative-down (~-5 to -5078 m) with `missing` over land — contour!
    # can't dim-convert a Union{Missing,_} matrix, and levels are meant as
    # positive depths, so coalesce to NaN and flip sign before contouring.
    depth_par = abs.(coalesce.(h_par, NaN32))
    lon_par_f = Float64.(lon_par)
    lat_par_f = Float64.(lat_par)

    # -- grid metrics for speed/vorticity (see check_output_parallel.jl) --
    # NCOM's u_velocity/v_velocity sit on the same (unstaggered) grid as
    # lon_par/lat_par/mask, unlike ROMS's C-grid u/v, and are already given
    # as true east/north components (unlike ROMS's grid-relative u/v) — so
    # no u2rho/v2rho averaging or rotation is needed before computing speed.
    # vorticity_cal still auto-converts them onto native U/V points
    # internally since they're passed in at "RHO-grid" size.
    pm_par = 1 ./ coalesce.(dx_par, Inf32)                  # ROMS-style inverse grid spacing (1/meters)
    pn_par = 1 ./ coalesce.(dy_par, Inf32)
    umask, vmask, mask_p = uvp_masks(mask)
    lon_psi, lat_psi = rho2p(lon_par), rho2p(lat_par)
    zm3_p = rho2p(zm3)                                      # vertical grid at PSI points, for depth-slicing vorticity
    kb_valid = coalesce.(kb, 0)
    kb_p = min.(kb_valid[1:end-1, 1:end-1], kb_valid[2:end, 1:end-1],
                kb_valid[1:end-1, 2:end], kb_valid[2:end, 2:end])   # PSI point valid only as deep as its shallowest neighbor

    bathy_levels = [500, 1000, 2000]

    # -- SSH gradient (see check_NCOM_zeta.jl) --
    # differences between neighboring cells live "between" them (U/V
    # points), then get averaged back onto the grid over wet neighbors only
    pm_u = rho2u(pm_par)   # 1/dx between xi-neighbors (1/m)
    pn_v = rho2v(pn_par)   # 1/dy between eta-neighbors (1/m)

    # |∇ssh| (m/m). NCOM's ssh is NaN over land, and NaN * 0 is still NaN,
    # so land is zero-filled before differencing.
    function zeta_gradient(zeta)
        zeta0 = ifelse.(isnan.(zeta), 0.0, zeta)
        dzdx_u = diff(zeta0, dims = 1) .* pm_u .* umask
        dzdy_v = diff(zeta0, dims = 2) .* pn_v .* vmask
        dzdx = u2rho(dzdx_u) ./ u2rho(umask)
        dzdy = v2rho(dzdy_v) ./ v2rho(vmask)
        return sqrt.(dzdx .^ 2 .+ dzdy .^ 2)
    end

    grad_scale = 1e6        # plot in units of 10^-6, i.e. mm of SSH change per km
    grad_range = (0, 10)    # same color range as the ROMS plots, for direct comparison

    # -- two views of every plot, each saved to its own folder --
    #   zoom — zoomed to the ROMS child domain
    #   full — the whole NCOM domain, to see signals outside the child grid too
    # skip: quiver downsampling (~50 on the 900 m ohgrd_2 grid; 2700 m is ~3x
    # coarser, so 17 keeps the same arrow spacing in km for the zoom view).
    # lengthscale = degrees of lon/lat drawn per 1 m/s; the full view spans
    # ~3.5x more degrees, so its arrows are sparser and drawn longer.
    views = [
        (dir = figure_path, lon_lim = lon_chd_lims, lat_lim = lat_chd_lims, skip = 17,
         quiver_kw = (lengthscale = 0.5, color = :black, shaftwidth = 1.5, tipwidth = 5, tiplength = 6)),
        (dir = figure_path_full, lon_lim = extrema(lon_par), lat_lim = extrema(lat_par), skip = 45,
         quiver_kw = (lengthscale = 1.5, color = :black, shaftwidth = 1.5, tipwidth = 5, tiplength = 6)),
    ]
    outname(vw, prefix, str) = joinpath(vw.dir, "$(prefix)_$(str).png")

    # every map uses the 2D topdown_axis (not Axis3's topdown_axis3), so the
    # latitude label/ticks sit on the left; see plotting_fun.jl
    # bathymetry contours + child-grid outline + view limits, shared by every map.
    # col = the map's layout column; sizing it to the map's lon/lat ratio lets
    # savemap's resize_to_layout! shrink the figure snug around it.
    function decorate_map!(ax, vw; col = 1)
        contour!(ax, lon_par_f, lat_par_f, depth_par; levels = bathy_levels,
                 color = RGBf(0.3, 0.3, 0.3), linewidth = 1)
        lines!(ax, lon_chd_b, lat_chd_b, zeros(length(lon_chd_b));
               color = :black, linewidth = 2)
        xlims!(ax, vw.lon_lim...)
        ylims!(ax, vw.lat_lim...)
        colsize!(ax.parent.layout, col, Aspect(1, (vw.lon_lim[2] - vw.lon_lim[1]) / (vw.lat_lim[2] - vw.lat_lim[1])))
    end

    # shrink the figure to the maps' data aspect before saving, trimming the
    # white space a fixed-size figure leaves around a DataAspect() axis
    function savemap(path, fig)
        resize_to_layout!(fig)
        figsave(path, fig)
    end

    # xlim/ylim keep arrows from poking in from just outside the view
    add_quiver!(ax, vw, u, v) =
        quiver_curvilinear!(ax, lon_par, lat_par, u, v;
                             skip = vw.skip, xlim = vw.lon_lim, ylim = vw.lat_lim, vw.quiver_kw...)
end
mkpath(figure_path_full)   # once, on the main process, so workers don't race to create it

@everywhere function ncom_time(fname)
    ocean_time = NCDataset(fname) do ds
        ds["MT"][:]
    end
    ntime = length(ocean_time)   # length(), not size() — size() returns a Tuple like (4,), not a plain number
    t_ref = DateTime(1900, 12, 31, 0, 0, 0)
    realtime = t_ref .+ Millisecond.(round.(Int, ocean_time .* 86400 .* 1000))   # days -> ms
    str1 = Dates.format.(realtime, "yyyy-mm-dd HH:MM:SS")                # for plot titles
    str2 = Dates.format.(realtime, "yyyymmdd_HHMM")                      # for filenames, e.g. 20220824_0730
    return ntime, str1, str2
end

@everywhere function process_ssh(fname)
    ntime, str1, str2 = ncom_time(fname)
    println(fname, " => ntime = ", ntime)

    # check every timestep's own output, not just the last one — the last
    # timestep's filename is the same as the NEXT day's first timestep
    # (NCOM's daily files overlap at midnight: hour 24 of one file == hour 0
    # of the next), so checking only the last file risks a false "done" if
    # that neighboring day's worker happens to process it first (pmap does
    # not dispatch files in chronological order)
    prefixes = ("zeta_ncom_plot", "zeta_grad_ncom_plot", "zeta_grad_ncom_plot_novec")
    if all(isfile(outname(vw, p, str2[t])) for vw in views, p in prefixes, t in 1:ntime)
        println("all plots already exist for ", fname, ", skipping file entirely")
        return
    end

    zeta = NCDataset(fname) do ds
        ds["ssh"][:, :, :]   # sea surface height, dims xi_rho x eta_rho x time
    end
    zeta_masked = ifelse.(mask .== 0, NaN32, zeta)
    zeta_grad_masked = ifelse.(mask .== 0, NaN, zeta_gradient(zeta))

    # surface velocity for the gradient plot's arrows, from the matching
    # _uv.nc file (same MT times as _ssh.nc) — only level 1 (top layer,
    # zm3 ≈ -0.5 m) is read, since the full 100-level u/v is large per step
    uv_fname = replace(fname, r"_ssh\.nc$" => "_uv.nc")
    u_east, v_north = NCDataset(uv_fname) do ds
        ds["u_velocity"][:, :, 1, :], ds["v_velocity"][:, :, 1, :]   # dims xi x eta x time
    end
    u_east_masked = ifelse.(mask .== 0, NaN, coalesce.(u_east, NaN))
    v_north_masked = ifelse.(mask .== 0, NaN, coalesce.(v_north, NaN))

    for t in 1:ntime
        # skip incomplete/fill-valued records — e.g. the last time step of a
        # file still being actively written, where every variable is left at
        # the raw NetCDF fill value (~9.97e36) instead of real data. ssh is
        # NaN over land, so filter those out first (maximum() propagates NaN).
        finite_z = filter(isfinite, zeta[:, :, t])
        if !isempty(finite_z) && maximum(abs, finite_z) > 1e30
            println("skipping t=$t ($(str1[t])) — looks like an incomplete/fill-valued record")
            continue
        end

        for vw in views
            outname_zeta = outname(vw, "zeta_ncom_plot", str2[t])
            outname_grad = outname(vw, "zeta_grad_ncom_plot", str2[t])
            outname_grad_novec = outname(vw, "zeta_grad_ncom_plot_novec", str2[t])

            # -- ZETA GRADIENT -- same novec-then-vector save pattern as speed/vorticity
            if isfile(outname_grad) && isfile(outname_grad_novec)
                println("already exists, skipping: ", outname_grad, " and ", outname_grad_novec)
            else
                fig = Figure(; figkw(18, 13.5)...)
                ax = topdown_axis(fig[1, 1]; title = "SSH gradient |∇ζ|, $(str1[t])")
                sp = plot_curvilinear!(ax, lon_par, lat_par, zeta_grad_masked[:, :, t] .* grad_scale;
                                        colormap = :amp, colorrange = grad_range)
                Colorbar(fig[1, 2], sp, label = "10⁻⁶ (mm/km)")
                decorate_map!(ax, vw)
                savemap(outname_grad_novec, fig)
                println("saved plot to ", outname_grad_novec)

                add_quiver!(ax, vw, u_east_masked[:, :, t], v_north_masked[:, :, t])
                savemap(outname_grad, fig)
                println("saved plot to ", outname_grad)
            end

            # -- ZETA --
            if isfile(outname_zeta)
                println("already exists, skipping: ", outname_zeta)
                continue
            end

            fig = Figure(; figkw(18, 13.5)...)
            ax = topdown_axis(fig[1, 1]; title = "Sea surface height (zeta), $(str1[t])")
            sp = plot_curvilinear!(ax, lon_par, lat_par, zeta_masked[:, :, t];
                                    colormap = Reverse(:RdBu), colorrange = (-1, 1))
            Colorbar(fig[1, 2], sp, label = "meters")
            decorate_map!(ax, vw)
            savemap(outname_zeta, fig)
            println("saved plot to ", outname_zeta)
        end
    end
end

@everywhere function process_ts(fname)
    ntime, str1, str2 = ncom_time(fname)
    println(fname, " => ntime = ", ntime)

    # see process_ssh above for why this checks every timestep instead of
    # just the last one (adjacent days' files share a boundary timestamp)
    if all(isfile(outname(vw, "temp_ncom_plot", str2[t])) for vw in views, t in 1:ntime)
        println("all plots already exist for ", fname, ", skipping file entirely")
        return
    end

    # layer_temperature is (xi, eta, z, time) at ~0.7 GB per time step
    # (Float64) on this grid — reading all 25 steps at once would need
    # ~17 GB, so ds_ts is kept open and indexed one time step at a time
    # inside the loop below instead.
    ds_ts = NCDataset(fname)

    for t in 1:ntime
        if all(isfile(outname(vw, "temp_ncom_plot", str2[t])) for vw in views)
            println("already exists, skipping: temp_ncom_plot_$(str2[t]).png (all views)")
            continue
        end

        temp_t = ds_ts["layer_temperature"][:, :, :, t]

        # two-step mask: land (horizontal, mask==0) and below-seafloor
        # (vertical, level index > kb[i,j] at that column) — both are
        # already NaN in the raw file (checked directly), but masking
        # explicitly means temp_masked doesn't depend on that assumption
        # holding for every file.
        kb0 = coalesce.(kb, 0)                          # missing (land) -> 0, so every level there gets masked
        zidx = reshape(1:size(temp_t, 3), 1, 1, :)
        temp_masked = ifelse.((mask .== 0) .| (zidx .> kb0), NaN32, temp_t)

        # skip incomplete/fill-valued records, same idea as the ssh loop —
        # but layer_temperature is expected to already contain NaN below
        # kb[i,j] (below the seafloor at that column), so those NaNs must
        # be excluded first or they'd swallow a real 1e30 fill value too
        # (maximum() propagates NaN rather than ignoring it).
        finite_temp = filter(isfinite, temp_t)
        if !isempty(finite_temp) && maximum(abs, finite_temp) > 1e30
            println("skipping t=$t ($(str1[t])) — looks like an incomplete/fill-valued record")
            continue
        end

        temp_1m = slice_at_depth_ncom(zm3, kb, temp_masked, -1.0)
        temp_100m = slice_at_depth_ncom(zm3, kb, temp_masked, -100.0)

        for vw in views
            outname_temp = outname(vw, "temp_ncom_plot", str2[t])
            if isfile(outname_temp)
                println("already exists, skipping: ", outname_temp)
                continue
            end

            fig = Figure(; figkw(18, 7.7)...)
            for (col, (field, depth_label, crange)) in enumerate(((temp_1m, "1 m", (24, 30)), (temp_100m, "100 m", (15, 30))))
                ax = topdown_axis(fig[1, 2col - 1]; title = "Temperature at $depth_label")
                sp = plot_curvilinear!(ax, lon_par, lat_par, field; colormap = :thermal, colorrange = crange)
                Colorbar(fig[1, 2col], sp, label = "°C")
                decorate_map!(ax, vw; col = 2col - 1)
            end
            # date once above both panels — per-panel titles with it ran into the colorbars
            Label(fig[0, :], str1[t]; font = :bold)
            savemap(outname_temp, fig)
            println("saved plot to ", outname_temp)
        end
    end

    close(ds_ts)
end

@everywhere function process_uv(fname)
    ntime, str1, str2 = ncom_time(fname)
    println(fname, " => ntime = ", ntime)

    # see process_ssh above for why this checks every timestep instead of
    # just the last one (adjacent days' files share a boundary timestamp)
    speed_prefixes = ("speed_ncom_plot", "speed_ncom_plot_novec")
    vor_prefixes = ("vorticity_ncom_plot", "vorticity_ncom_plot_novec")
    done(prefixes, t) = all(isfile(outname(vw, p, str2[t])) for vw in views, p in prefixes)
    if all(t -> done(speed_prefixes, t) && done(vor_prefixes, t), 1:ntime)
        println("all plots already exist for ", fname, ", skipping file entirely")
        return
    end

    # u_velocity/v_velocity are (xi, eta, z, time) at ~0.7 GB per time step
    # (Float64) on this grid, same as layer_temperature — ds_uv is kept
    # open and indexed one time step at a time inside the loop below
    # instead of reading all 25 steps at once.
    ds_uv = NCDataset(fname)

    for t in 1:ntime
        speed_done = done(speed_prefixes, t)
        vor_done = done(vor_prefixes, t)
        if speed_done && vor_done
            println("already exists, skipping: speed/vorticity plots for $(str2[t]) (all views)")
            continue
        end

        u_t = ds_uv["u_velocity"][:, :, :, t]
        v_t = ds_uv["v_velocity"][:, :, :, t]

        # skip incomplete/fill-valued records, same idea as the ssh/temp loops
        finite_u = filter(isfinite, u_t)
        if !isempty(finite_u) && maximum(abs, finite_u) > 1e30
            println("skipping t=$t ($(str1[t])) — looks like an incomplete/fill-valued record")
            continue
        end

        # land (horizontal, mask==0) + below-seafloor (vertical, level index
        # > kb[i,j]) mask, same two-step logic as the temp loop above
        kb0 = coalesce.(kb, 0)
        zidx = reshape(1:size(u_t, 3), 1, 1, :)
        land_or_below = (mask .== 0) .| (zidx .> kb0)

        # NCOM's u_velocity/v_velocity are already true east/north
        # components, so no rotation is needed before masking (speed would
        # be rotation-invariant anyway, but the quiver arrows below need
        # the un-rotated east/north components too).
        u_east_masked = ifelse.(land_or_below, NaN32, u_t)
        v_north_masked = ifelse.(land_or_below, NaN32, v_t)
        u_east_1m = slice_at_depth_ncom(zm3, kb, u_east_masked, -1.0)
        v_north_1m = slice_at_depth_ncom(zm3, kb, v_north_masked, -1.0)

        # -- SPEED -- saved twice per view: once with the color field alone
        # (novec, easiest to read the speed itself), then again with quiver
        # arrows added on top of the same axis.
        if !speed_done
            speed_masked = ifelse.(land_or_below, NaN32, sqrt.(u_t .^ 2 .+ v_t .^ 2))
            speed_1m = slice_at_depth_ncom(zm3, kb, speed_masked, -1.0)

            for vw in views
                outname_speed = outname(vw, "speed_ncom_plot", str2[t])
                outname_speed_novec = outname(vw, "speed_ncom_plot_novec", str2[t])
                if isfile(outname_speed) && isfile(outname_speed_novec)
                    println("already exists, skipping: ", outname_speed, " and ", outname_speed_novec)
                    continue
                end

                fig = Figure(; figkw(18, 13.5)...)
                ax = topdown_axis(fig[1, 1]; title = "Speed at 1 m, $(str1[t])")
                sp = plot_curvilinear!(ax, lon_par, lat_par, speed_1m; colormap = :speed, colorrange = (0, 2))
                Colorbar(fig[1, 2], sp, label = "m/s")
                decorate_map!(ax, vw)
                savemap(outname_speed_novec, fig)
                println("saved plot to ", outname_speed_novec)

                add_quiver!(ax, vw, u_east_1m, v_north_1m)
                savemap(outname_speed, fig)
                println("saved plot to ", outname_speed)
            end
        end

        # -- VORTICITY -- same novec-then-vector save pattern as speed above
        if !vor_done
            # use the raw (unmasked) u_t, v_t here — masking with NaN before
            # differencing would spread NaN into neighboring cells; mask the
            # result afterward instead
            vor_psi = vorticity_cal(u_t, v_t, pm_par, pn_par)
            vor_psi_masked = ifelse.(mask_p .== 0, NaN32, vor_psi)
            vor_1m = slice_at_depth_ncom(zm3_p, kb_p, vor_psi_masked, -1.0)

            for vw in views
                outname_vor = outname(vw, "vorticity_ncom_plot", str2[t])
                outname_vor_novec = outname(vw, "vorticity_ncom_plot_novec", str2[t])
                if isfile(outname_vor) && isfile(outname_vor_novec)
                    println("already exists, skipping: ", outname_vor, " and ", outname_vor_novec)
                    continue
                end

                fig = Figure(; figkw(18, 13.5)...)
                ax = topdown_axis(fig[1, 1]; title = "Relative vorticity, surface, $(str1[t])")
                sp = plot_curvilinear!(ax, lon_psi, lat_psi, vor_1m;
                                        colormap = Reverse(:RdBu), colorrange = (-5e-5, 5e-5))
                Colorbar(fig[1, 2], sp, label = "s⁻¹")
                decorate_map!(ax, vw)
                savemap(outname_vor_novec, fig)
                println("saved plot to ", outname_vor_novec)

                add_quiver!(ax, vw, u_east_1m, v_north_1m)
                savemap(outname_vor, fig)
                println("saved plot to ", outname_vor)
            end
        end
    end

    close(ds_uv)
end

# dispatches on the type tag attached to each entry of `tasks` below —
# ssh/temp/uv come from entirely separate files for NCOM (unlike ROMS's
# single his.nc carrying zeta/temp/u/v together), so there's no shared
# read to lose by mixing all three types into one flat pmap list.
@everywhere function process_file((kind, fname))
    if kind == :ssh
        process_ssh(fname)
    elseif kind == :ts
        process_ts(fname)
    elseif kind == :uv
        process_uv(fname)
    end
end

## build one combined, type-tagged file list and pmap over all of it —
# work-stealing then lets a worker that just finished a quick ssh (2D)
# file immediately pick up the next available job regardless of type,
# instead of a worker dedicated to one variable sitting idle once its
# own queue drains early while the other queues still have a backlog.
files_ssh = sort(filter(f -> endswith(f, "_ssh.nc") &&
                         "2022082400_ssh.nc" <= basename(f) <= "2022092300_ssh.nc",
#                         "2022082400_ssh.nc" <= basename(f) <= "2022082400_ssh.nc",
                    readdir(datadir, join=true)))
files_ts = sort(filter(f -> endswith(f, "_ts.nc") &&
                         "2022082400_ts.nc" <= basename(f) <= "2022092300_ts.nc",
#                         "2022082400_ts.nc" <= basename(f) <= "2022082400_ts.nc",
                    readdir(datadir, join=true)))
files_uv = sort(filter(f -> endswith(f, "_uv.nc") &&
                         "2022082400_uv.nc" <= basename(f) <= "2022092300_uv.nc",
#                         "2022082400_uv.nc" <= basename(f) <= "2022082400_uv.nc",
                    readdir(datadir, join=true)))

tasks = [(:ssh, f) for f in files_ssh]
append!(tasks, [(:ts, f) for f in files_ts])
append!(tasks, [(:uv, f) for f in files_uv])
println("found $(length(tasks)) files total ($(length(files_ssh)) ssh, $(length(files_ts)) ts, $(length(files_uv)) uv)")

pmap(process_file, tasks; on_error = ex -> println("a file failed: ", ex))
