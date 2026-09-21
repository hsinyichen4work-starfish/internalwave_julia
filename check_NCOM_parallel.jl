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
    parent_grid = "/home/mbui/ModelOutput/NCOM/grid/ohgrd_2.nc"
    datadir = "/home/hsinyi/roms_data/NCOM_DATA_NC/"
    figure_path = "/home/hsinyi/figure/20260914_julia_outputtest/NCOM_FIG"

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
    _, _, mask_p = uvp_masks(mask)
    lon_psi, lat_psi = rho2p(lon_par), rho2p(lat_par)
    zm3_p = rho2p(zm3)                                      # vertical grid at PSI points, for depth-slicing vorticity
    kb_valid = coalesce.(kb, 0)
    kb_p = min.(kb_valid[1:end-1, 1:end-1], kb_valid[2:end, 1:end-1],
                kb_valid[1:end-1, 2:end], kb_valid[2:end, 2:end])   # PSI point valid only as deep as its shallowest neighbor
    uv_skip = 50   # downsample for legible quiver arrows on this ~1244x1334 grid

    bathy_levels = [500, 1000, 2000]
end

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

#=
@everywhere function process_ssh(fname)
    ntime, str1, str2 = ncom_time(fname)
    println(fname, " => ntime = ", ntime)

    # check every timestep's own output, not just the last one — the last
    # timestep's filename is the same as the NEXT day's first timestep
    # (NCOM's daily files overlap at midnight: hour 24 of one file == hour 0
    # of the next), so checking only the last file risks a false "done" if
    # that neighboring day's worker happens to process it first (pmap does
    # not dispatch files in chronological order)
    if all(t -> isfile(joinpath(figure_path, "zeta_ncom_plot_$(str2[t]).png")), 1:ntime)
        println("all plots already exist for ", fname, ", skipping file entirely")
        return
    end

    zeta = NCDataset(fname) do ds
        ds["ssh"][:, :, :]   # sea surface height, dims xi_rho x eta_rho x time
    end
    zeta_masked = ifelse.(mask .== 0, NaN32, zeta)

    for t in 1:ntime
        # skip incomplete/fill-valued records — e.g. the last time step of a
        # file still being actively written, where every variable is left at
        # the raw NetCDF fill value (~9.97e36) instead of real data
        if maximum(abs, zeta[:, :, t]) > 1e30
            println("skipping t=$t ($(str1[t])) — looks like an incomplete/fill-valued record")
            continue
        end

        outname_zeta = joinpath(figure_path, "zeta_ncom_plot_$(str2[t]).png")
        if isfile(outname_zeta)
            println("already exists, skipping: ", outname_zeta)
            continue
        end

        fig = Figure(size = (800, 600))
        ax = topdown_axis3(fig[1, 1]; title = "Sea surface height (zeta), $(str1[t])")
        sp = plot_curvilinear!(ax, lon_par, lat_par, zeta_masked[:, :, t];
                                colormap = Reverse(:RdBu), colorrange = (-1, 1))
        Colorbar(fig[1, 2], sp, label = "meters")
        contour!(ax, lon_par_f, lat_par_f, depth_par; levels = bathy_levels,
                 color = RGBf(0.3, 0.3, 0.3), linewidth = 1)
        lines!(ax, lon_chd_b, lat_chd_b, zeros(length(lon_chd_b));
               color = :black, linewidth = 2)
        xlims!(ax, lon_chd_lims...)
        ylims!(ax, lat_chd_lims...)
        save(outname_zeta, fig)
        println("saved plot to ", outname_zeta)
    end
end
=#

@everywhere function process_ts(fname)
    ntime, str1, str2 = ncom_time(fname)
    println(fname, " => ntime = ", ntime)

    # see process_ssh above for why this checks every timestep instead of
    # just the last one (adjacent days' files share a boundary timestamp)
    if all(t -> isfile(joinpath(figure_path, "temp_ncom_plot_$(str2[t]).png")), 1:ntime)
        println("all plots already exist for ", fname, ", skipping file entirely")
        return
    end

    # layer_temperature is (xi, eta, z, time) at ~1.3 GB per time step
    # (Float64) on this grid — reading all 25 steps at once would need
    # ~33 GB, so ds_ts is kept open and indexed one time step at a time
    # inside the loop below instead.
    ds_ts = NCDataset(fname)

    for t in 1:ntime
        outname_temp = joinpath(figure_path, "temp_ncom_plot_$(str2[t]).png")
        if isfile(outname_temp)
            println("already exists, skipping: ", outname_temp)
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

        fig = Figure(size = (1400, 600))
        for (col, (field, depth_label, crange)) in enumerate(((temp_1m, "1 m", (24, 30)), (temp_100m, "100 m", (15, 30))))
            ax = topdown_axis3(fig[1, 2col - 1]; title = "Temperature at $depth_label, $(str1[t])")
            sp = plot_curvilinear!(ax, lon_par, lat_par, field; colormap = :thermal, colorrange = crange)
            Colorbar(fig[1, 2col], sp, label = "°C")
            contour!(ax, lon_par_f, lat_par_f, depth_par; levels = bathy_levels,
                     color = RGBf(0.3, 0.3, 0.3), linewidth = 1)
            lines!(ax, lon_chd_b, lat_chd_b, zeros(length(lon_chd_b));
                   color = :black, linewidth = 2)
            xlims!(ax, lon_chd_lims...)
            ylims!(ax, lat_chd_lims...)
        end
        save(outname_temp, fig)
        println("saved plot to ", outname_temp)
    end

    close(ds_ts)
end

@everywhere function process_uv(fname)
    ntime, str1, str2 = ncom_time(fname)
    println(fname, " => ntime = ", ntime)

    # see process_ssh above for why this checks every timestep instead of
    # just the last one (adjacent days' files share a boundary timestamp)
    if all(1:ntime) do t
           isfile(joinpath(figure_path, "speed_ncom_plot_$(str2[t]).png")) &&
           isfile(joinpath(figure_path, "speed_ncom_plot_novec_$(str2[t]).png")) &&
           isfile(joinpath(figure_path, "vorticity_ncom_plot_$(str2[t]).png")) &&
           isfile(joinpath(figure_path, "vorticity_ncom_plot_novec_$(str2[t]).png"))
       end
        println("all plots already exist for ", fname, ", skipping file entirely")
        return
    end

    # u_velocity/v_velocity are (xi, eta, z, time) at ~1.3 GB per time step
    # (Float64) on this grid, same as layer_temperature — ds_uv is kept
    # open and indexed one time step at a time inside the loop below
    # instead of reading all 25 steps at once.
    ds_uv = NCDataset(fname)

    for t in 1:ntime
        outname_speed = joinpath(figure_path, "speed_ncom_plot_$(str2[t]).png")
        outname_speed_novec = joinpath(figure_path, "speed_ncom_plot_novec_$(str2[t]).png")
        outname_vor = joinpath(figure_path, "vorticity_ncom_plot_$(str2[t]).png")
        outname_vor_novec = joinpath(figure_path, "vorticity_ncom_plot_novec_$(str2[t]).png")
        if isfile(outname_speed) && isfile(outname_speed_novec) &&
           isfile(outname_vor) && isfile(outname_vor_novec)
            println("already exists, skipping: ", outname_speed, ", ", outname_speed_novec,
                    ", ", outname_vor, " and ", outname_vor_novec)
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
        speed_masked = ifelse.(land_or_below, NaN32, sqrt.(u_t .^ 2 .+ v_t .^ 2))
        u_east_masked = ifelse.(land_or_below, NaN32, u_t)
        v_north_masked = ifelse.(land_or_below, NaN32, v_t)

        # -- SPEED -- saved twice: once with the color field alone (novec,
        # easiest to read the speed itself), then again with quiver arrows
        # added on top of the same axis (arrows are restricted to the
        # child-grid box via xlim/ylim so none poke in from just outside
        # the zoomed view).
        if isfile(outname_speed) && isfile(outname_speed_novec)
            println("already exists, skipping: ", outname_speed, " and ", outname_speed_novec)
        else
            speed_1m = slice_at_depth_ncom(zm3, kb, speed_masked, -1.0)
            u_east_1m = slice_at_depth_ncom(zm3, kb, u_east_masked, -1.0)
            v_north_1m = slice_at_depth_ncom(zm3, kb, v_north_masked, -1.0)

            fig = Figure(size = (800, 600))
            ax = topdown_axis3(fig[1, 1]; title = "Speed at 1 m, $(str1[t])")
            sp = plot_curvilinear!(ax, lon_par, lat_par, speed_1m; colormap = :speed, colorrange = (0, 2))
            Colorbar(fig[1, 2], sp, label = "m/s")
            contour!(ax, lon_par_f, lat_par_f, depth_par; levels = bathy_levels,
                     color = RGBf(0.3, 0.3, 0.3), linewidth = 1)
            lines!(ax, lon_chd_b, lat_chd_b, zeros(length(lon_chd_b));
                   color = :black, linewidth = 2)
            xlims!(ax, lon_chd_lims...)
            ylims!(ax, lat_chd_lims...)
            save(outname_speed_novec, fig)
            println("saved plot to ", outname_speed_novec)

            quiver_curvilinear!(ax, lon_par, lat_par, u_east_1m, v_north_1m;
                                 skip = uv_skip, xlim = lon_chd_lims, ylim = lat_chd_lims,
                                 lengthscale = 0.5, color = :black,
                                 shaftwidth = 1.5, tipwidth = 5, tiplength = 6)
            save(outname_speed, fig)
            println("saved plot to ", outname_speed)
        end

        # -- VORTICITY -- same novec-then-vector save pattern as speed above
        if isfile(outname_vor) && isfile(outname_vor_novec)
            println("already exists, skipping: ", outname_vor, " and ", outname_vor_novec)
        else
            # speed's u_east_1m/v_north_1m may have been skipped above (if the
            # speed plots already existed), so recompute them here too — cheap
            # relative to the rest of this block, and keeps this block able to
            # run independently of the speed one.
            u_east_1m = slice_at_depth_ncom(zm3, kb, u_east_masked, -1.0)
            v_north_1m = slice_at_depth_ncom(zm3, kb, v_north_masked, -1.0)

            # use the raw (unmasked) u_t, v_t here — masking with NaN before
            # differencing would spread NaN into neighboring cells; mask the
            # result afterward instead
            vor_psi = vorticity_cal(u_t, v_t, pm_par, pn_par)
            vor_psi_masked = ifelse.(mask_p .== 0, NaN32, vor_psi)
            vor_1m = slice_at_depth_ncom(zm3_p, kb_p, vor_psi_masked, -1.0)

            fig = Figure(size = (800, 600))
            ax = topdown_axis3(fig[1, 1]; title = "Relative vorticity, surface, $(str1[t])")
            sp = plot_curvilinear!(ax, lon_psi, lat_psi, vor_1m;
                                    colormap = Reverse(:RdBu), colorrange = (-5e-5, 5e-5))
            Colorbar(fig[1, 2], sp, label = "s⁻¹")
            contour!(ax, lon_par_f, lat_par_f, depth_par; levels = bathy_levels,
                     color = RGBf(0.3, 0.3, 0.3), linewidth = 1)
            lines!(ax, lon_chd_b, lat_chd_b, zeros(length(lon_chd_b));
                   color = :black, linewidth = 2)
            xlims!(ax, lon_chd_lims...)
            ylims!(ax, lat_chd_lims...)
            save(outname_vor_novec, fig)
            println("saved plot to ", outname_vor_novec)

            quiver_curvilinear!(ax, lon_par, lat_par, u_east_1m, v_north_1m;
                                 skip = uv_skip, xlim = lon_chd_lims, ylim = lat_chd_lims,
                                 lengthscale = 0.5, color = :black,
                                 shaftwidth = 1.5, tipwidth = 5, tiplength = 6)
            save(outname_vor, fig)
            println("saved plot to ", outname_vor)
        end
    end

    close(ds_uv)
end

# dispatches on the type tag attached to each entry of `tasks` below —
# ssh/temp/uv come from entirely separate files for NCOM (unlike ROMS's
# single his.nc carrying zeta/temp/u/v together), so there's no shared
# read to lose by mixing all three types into one flat pmap list.
@everywhere function process_file((kind, fname))
    #=
    if kind == :ssh
        process_ssh(fname)
    end
    =#
    if kind == :ts
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
#=
files_ssh = sort(filter(f -> endswith(f, "_ssh.nc") &&
                         "2022082400_ssh.nc" <= basename(f) <= "2022092300_ssh.nc",
#                         "2022082400_ssh.nc" <= basename(f) <= "2022082400_ssh.nc",
                    readdir(datadir, join=true)))
=#
files_ts = sort(filter(f -> endswith(f, "_ts.nc") &&
                         "2022082400_ts.nc" <= basename(f) <= "2022092300_ts.nc",
#                         "2022082400_ts.nc" <= basename(f) <= "2022082400_ts.nc",
                    readdir(datadir, join=true)))
files_uv = sort(filter(f -> endswith(f, "_uv.nc") &&
                         "2022082400_uv.nc" <= basename(f) <= "2022092300_uv.nc",
#                         "2022082400_uv.nc" <= basename(f) <= "2022082400_uv.nc",
                    readdir(datadir, join=true)))

tasks = [(:ts, f) for f in files_ts]
append!(tasks, [(:uv, f) for f in files_uv])
println("found $(length(tasks)) files total ($(length(files_ts)) ts, $(length(files_uv)) uv)")

pmap(process_file, tasks; on_error = ex -> println("a file failed: ", ex))
