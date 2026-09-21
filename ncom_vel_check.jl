using Distributed

n_workers = max(1, parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "4")) - 1)   # -1 reserves a CPU for this main process
# default of 6 (-> 5 workers) is sized for this workstation's RAM, not core count —
# each worker keeps its own full copy of the grid arrays plus u/v data per timestep,
# so more workers means more concurrent memory, not just more CPU.
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
    figure_path = "/home/hsinyi/figure/20260914_julia_outputtest/NCOM_FIG/vel_check"
    mkpath(figure_path)   # no-op if it already exists

    ##
    lon_chd, lat_chd = NCDataset(child_grid) do ds
        ds["lon_rho"][:, :], ds["lat_rho"][:, :]
    end
    lon_chd[lon_chd .> 180] .-= 360   # convert 0-360 convention to -180/180 (east/west hemisphere)
    lon_chd_b, lat_chd_b = grid_boundary(lon_chd, lat_chd)   # child-grid outline, for overlaying on parent-grid plots
    lon_chd_lims = extrema(lon_chd)   # child-grid extent, to zoom parent-grid plots to it (xlim/ylim equivalent)
    lat_chd_lims = extrema(lat_chd)

    lon_par, lat_par, h_par, mask, zm3, kb = NCDataset(parent_grid) do ds
        ds["lon"][:, :], ds["lat"][:, :], ds["h"][:,:],
        ds["mask"][:, :], ds["zm3"][:,:,:], ds["kb"][:,:]
    end
    # h is negative-down (~-5 to -5078 m) with `missing` over land — contour!
    # can't dim-convert a Union{Missing,_} matrix, and levels are meant as
    # positive depths, so coalesce to NaN and flip sign before contouring.
    depth_par = abs.(coalesce.(h_par, NaN32))
    lon_par_f = Float64.(lon_par)
    lat_par_f = Float64.(lat_par)

    # NCOM's u_velocity/v_velocity sit on the same (unstaggered) grid as
    # lon_par/lat_par/mask, unlike ROMS's C-grid u/v, and are already given
    # as true east/north components (unlike ROMS's grid-relative u/v) — so
    # no u2rho/v2rho averaging or rotation is needed here.
    uv_skip = 50   # downsample for legible quiver arrows on this ~1244x1334 grid

    bathy_levels = [500, 1000, 2000]

    # one entry per depth slice: (depth in meters, label for titles/filenames,
    # filename prefix, symmetric color limit in m/s). Velocities are weaker at
    # depth, so 100 m gets a tighter color range than 1 m/10 m — adjust to taste.
    depth_slices = [
        (-1.0,   "1 m",   "vel_1m_ncom",   1.0),
        (-10.0,  "10 m",  "vel_10m_ncom",  1.0),
        (-100.0, "100 m", "vel_100m_ncom", 0.5),
    ]
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

@everywhere function process_file(fname)
    ntime, str1, str2 = ncom_time(fname)
    println(fname, " => ntime = ", ntime)

    # check every timestep's own output, not just the last one — NCOM's daily
    # files overlap at midnight (hour 24 of one file == hour 0 of the next),
    # so checking only the last file risks a false "done" if that neighboring
    # day's worker happens to process it first (pmap does not dispatch files
    # in chronological order)
    if all(1:ntime) do t
           all(isfile, [joinpath(figure_path, "$(prefix)_plot_$(str2[t]).png") for (_, _, prefix, _) in depth_slices])
       end
        println("all plots already exist for ", fname, ", skipping file entirely")
        return
    end

    # u_velocity/v_velocity are (xi, eta, z, time) at ~1.3 GB per time step
    # (Float64) on this grid — ds_uv is kept open and indexed one time step
    # at a time inside the loop below instead of reading all steps at once.
    ds_uv = NCDataset(fname)

    for t in 1:ntime
        outnames = [joinpath(figure_path, "$(prefix)_plot_$(str2[t]).png") for (_, _, prefix, _) in depth_slices]
        if all(isfile, outnames)
            println("already exists, skipping: ", join(outnames, ", "))
            continue
        end

        u_t = ds_uv["u_velocity"][:, :, :, t]
        v_t = ds_uv["v_velocity"][:, :, :, t]

        # skip incomplete/fill-valued records — e.g. the last time step of a
        # file still being actively written, where every variable is left at
        # the raw NetCDF fill value (~9.97e36) instead of real data
        finite_u = filter(isfinite, u_t)
        if !isempty(finite_u) && maximum(abs, finite_u) > 1e30
            println("skipping t=$t ($(str1[t])) — looks like an incomplete/fill-valued record")
            continue
        end

        # land (horizontal, mask==0) + below-seafloor (vertical, level index
        # > kb[i,j]) mask
        kb0 = coalesce.(kb, 0)
        zidx = reshape(1:size(u_t, 3), 1, 1, :)
        land_or_below = (mask .== 0) .| (zidx .> kb0)

        # NCOM's u_velocity/v_velocity are already true east/north
        # components, so no rotation is needed before masking.
        u_east_masked = ifelse.(land_or_below, NaN32, u_t)
        v_north_masked = ifelse.(land_or_below, NaN32, v_t)

        for (idx, (depth, depth_label, _, clim)) in enumerate(depth_slices)
            outname = outnames[idx]
            if isfile(outname)
                println("already exists, skipping: ", outname)
                continue
            end

            u_slice = slice_at_depth_ncom(zm3, kb, u_east_masked, depth)
            v_slice = slice_at_depth_ncom(zm3, kb, v_north_masked, depth)

            fig = Figure(size = (1400, 600))
            for (col, (field, comp_label)) in enumerate(((u_slice, "u (east)"), (v_slice, "v (north)")))
                ax = topdown_axis3(fig[1, 2col - 1]; title = "$comp_label at $depth_label, $(str1[t])")
                sp = plot_curvilinear!(ax, lon_par, lat_par, field;
                                        colormap = Reverse(:RdBu), colorrange = (-clim, clim))
                Colorbar(fig[1, 2col], sp, label = "m/s")
                contour!(ax, lon_par_f, lat_par_f, depth_par; levels = bathy_levels,
                         color = RGBf(0.3, 0.3, 0.3), linewidth = 1)
                lines!(ax, lon_chd_b, lat_chd_b, zeros(length(lon_chd_b));
                       color = :black, linewidth = 2)
                xlims!(ax, lon_chd_lims...)
                ylims!(ax, lat_chd_lims...)
            end
            save(outname, fig)
            println("saved plot to ", outname)
        end
    end

    close(ds_uv)
end

# the NCOM output dir keeps velocity in its own _uv.nc files (see
# check_NCOM_parallel.jl) — sorted so files are processed in chronological
# order, which works here because the filenames embed a sortable timestamp
files = sort(filter(f -> endswith(f, "_uv.nc"), readdir(datadir, join=true)))
println("found $(length(files)) uv files in $datadir")
# pmap hands files out to whichever worker is free, one at a time, and
# blocks here until every file is done — no manual scheduling needed.
pmap(process_file, files; on_error = ex -> println("a file failed: ", ex))


## MOVIES
# stitches each set of PNGs (across all uv files/timesteps) into an mp4,
# in chronological order (filenames sort correctly since str2 timestamps
# are lexicographically ordered). Requires ffmpeg on PATH.

using FFMPEG_jll
function make_movie(prefix::String; fps = 4)
    pngs = sort(filter(f -> startswith(basename(f), prefix) && endswith(f, ".png"),
                        readdir(figure_path, join = true)))
    if isempty(pngs)
        println("no PNGs found for prefix \"$prefix\", skipping movie")
        return
    end

    listfile = joinpath(figure_path, "$(prefix)_filelist.txt")
    open(listfile, "w") do io
        for p in pngs
            println(io, "file '$(p)'")
        end
    end

    outname = joinpath(figure_path, "$(prefix)_movie.mp4")
    FFMPEG_jll.ffmpeg() do ffmpeg_path
        run(`$ffmpeg_path -y -r $fps -f concat -safe 0 -i $listfile -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -vcodec libx264 -pix_fmt yuv420p $outname`)
    end
    rm(listfile)
    println("saved movie to ", outname)
end

make_movie("vel_1m_ncom_plot")
make_movie("vel_10m_ncom_plot")
make_movie("vel_100m_ncom_plot")
