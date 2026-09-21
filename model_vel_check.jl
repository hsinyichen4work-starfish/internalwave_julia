using Distributed

n_workers = max(1, parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "7")) - 1)   # -1 reserves a CPU for this main process
addprocs(n_workers)
println("running with $(nprocs() - 1) worker processes")

@everywhere begin
    using NCDatasets, CairoMakie, Dates, Statistics
    CairoMakie.activate!()
    include("/home/hchen54/internalwave_julia/function/load_all_hpc.jl")
    include("/home/hchen54/internalwave_julia/function/plotting_fun.jl")

    grid_fname = "/expanse/lustre/projects/uso101/hchen54/input/grid/roms_grd_900m.nc"   # the grid_file listed in the .nc's global attributes
    datadir = "/expanse/lustre/projects/uso101/hchen54/amazon_900m_dbry_2"   # HPC output dir — contains avg/dia/his/rst files mixed together
    figure_path = "/home/hchen54/figure/dbry900m/vel_check"
    mkpath(figure_path)   # no-op if it already exists

    ##
    mask_rho, lon_rho, lat_rho, h ,pm, pn, grid_angle = NCDataset(grid_fname) do ds
        ds["mask_rho"][:, :], ds["lon_rho"][:, :], ds["lat_rho"][:, :],
        ds["h"][:,:], ds["pm"][:,:] , ds["pn"][:,:], ds["angle"][:,:]
    end
    lon_rho[lon_rho .> 180] .-= 360   # convert 0-360 convention to -180/180 (east/west hemisphere)

    lon_lim = extrema(lon_rho)   # shared x/y axis limits for every plot below
    lat_lim = extrema(lat_rho)

    # h here is the ROMS grid's own bathymetry (positive-down, no missing values),
    # already on the same lon_rho/lat_rho grid as everything plotted below.
    bathy_levels = [500, 1000, 2000]

    # one entry per depth slice: (depth in meters, label for titles/filenames,
    # filename prefix, symmetric color limit in m/s). Velocities are weaker at
    # depth, so 100 m gets a tighter color range than 1 m/10 m — adjust to taste.
    depth_slices = [
        (-1.0,   "1 m",   "vel_1m",   1.0),
        (-10.0,  "10 m",  "vel_10m",  1.0),
        (-100.0, "100 m", "vel_100m", 0.5),
    ]
end

@everywhere function process_file(fname)
    ocean_time = NCDataset(fname) do ds
        ds["ocean_time"][:]
    end
    println(fname, " => size(ocean_time) = ", size(ocean_time))
    ntime = length(ocean_time)   # length(), not size() — size() returns a Tuple like (4,), not a plain number
    t_ref = DateTime(1994, 1, 1, 0, 0, 0)
    realtime = t_ref .+ Millisecond.(round.(Int, ocean_time .* 1000))   # Vector{DateTime}
    str1 = Dates.format.(realtime, "yyyy-mm-dd HH:MM:SS")                # for plot titles
    str2 = Dates.format.(realtime, "yyyymmdd_HHMM")                      # for filenames, e.g. 20220824_0730

    ## Whole-file skip — if this file's last time step already produced all
    # three depth plots, assume the whole file was fully processed last run
    # and skip reading zeta/u/v entirely (not just skip the plotting).
    last_outnames = [joinpath(figure_path, "$(prefix)_plot_$(str2[ntime]).png") for (_, _, prefix, _) in depth_slices]
    if all(isfile, last_outnames)
        println("all plots already exist for ", fname, ", skipping file entirely")
        return
    end

    ## SSH DATA — needed for the z levels used to slice u/v at fixed depths
    zeta = NCDataset(fname) do ds
        ds["zeta"][:, :, :]   # sea surface height, dims xi_rho x eta_rho x time
    end
    println(fname, " => size(zeta) = ", size(zeta))

    theta_s, theta_b, hc, N = NCDataset(fname) do ds
        ds.attrib["theta_s"], ds.attrib["theta_b"], ds.attrib["hc"], size(ds["temp"], 3)
    end

    z = zeros(size(zeta, 1), size(zeta, 2), N, ntime)
    for t in 1:ntime
        z_dum, _ = zlevs3(h, zeta[:, :, t], theta_s, theta_b, hc, N, "r", "new2008")
        z[:, :, :, t] = permutedims(z_dum, (2, 3, 1))
    end

    ## VEL DATA
    u,v = NCDataset(fname) do ds
        ds["u"][:, :, :,:]  , ds["v"][:, :, :,:]
    end
    println(fname, " => size(u) = ", size(u))
    println(fname, " => size(v) = ", size(v))

    # average raw (unmasked) u, v onto the rho grid first, then mask the
    # result — masking beforehand would spread NaN from land onto valid
    # ocean cells one row/column too far
    u_rho = u2rho(u)   # still grid-relative (xi/eta), not east/north
    v_rho = v2rho(v)

    # rotate out of the grid's local xi/eta directions into true east/north,
    # using the grid's rotation angle (radians) from the grid file
    u_east = u_rho .* cos.(grid_angle) .- v_rho .* sin.(grid_angle)
    v_north = u_rho .* sin.(grid_angle) .+ v_rho .* cos.(grid_angle)
    u_east_masked = ifelse.(mask_rho .== 0, NaN32, u_east)
    v_north_masked = ifelse.(mask_rho .== 0, NaN32, v_north)

    ## PLOTS — one set of files per ocean_time step
    for t in 1:ntime

        # skip incomplete/fill-valued records — e.g. the last time step of a
        # file still being actively written, where every variable is left at
        # the raw NetCDF fill value (~9.97e36) instead of real data
        if maximum(abs, zeta[:, :, t]) > 1e30
            println("skipping t=$t ($(str1[t])) — looks like an incomplete/fill-valued record")
            continue
        end

        for (depth, depth_label, prefix, clim) in depth_slices
            outname = joinpath(figure_path, "$(prefix)_plot_$(str2[t]).png")
            if isfile(outname)
                println("already exists, skipping: ", outname)
                continue
            end

            u_slice = slice_at_depth(z[:, :, :, t], u_east_masked[:, :, :, t], depth)
            v_slice = slice_at_depth(z[:, :, :, t], v_north_masked[:, :, :, t], depth)

            fig = Figure(size = (1400, 600))
            for (col, (field, comp_label)) in enumerate(((u_slice, "u (east)"), (v_slice, "v (north)")))
                ax = topdown_axis3(fig[1, 2col - 1]; title = "$comp_label at $depth_label, $(str1[t])")
                xlims!(ax, lon_lim...)
                ylims!(ax, lat_lim...)
                sp = plot_curvilinear!(ax, lon_rho, lat_rho, field;
                                        colormap = Reverse(:RdBu), colorrange = (-clim, clim))
                Colorbar(fig[1, 2col], sp, label = "m/s")
                contour!(ax, lon_rho, lat_rho, h; levels = bathy_levels,
                         color = RGBf(0.3, 0.3, 0.3), linewidth = 1)
            end
            save(outname, fig)
            println("saved plot to ", outname)
        end
    end
end

# the HPC output dir mixes roms_avg/dia/his/rst files together, so filter
# to just the "his" files (they carry zeta/u/v, everything the plots below
# need) — sorted so files are processed in chronological order, which works
# here because the filenames embed a sortable timestamp
files = sort(filter(f -> occursin("roms_his", basename(f)) && endswith(f, ".nc"),
                    readdir(datadir, join=true)))
println("found $(length(files)) his files in $datadir")
# pmap hands files out to whichever worker is free, one at a time, and
# blocks here until every file is done — no manual scheduling needed.
pmap(process_file, files; on_error = ex -> println("a file failed: ", ex))


## MOVIES
# stitches each set of PNGs (across all his files/timesteps) into an mp4,
# in chronological order (filenames sort correctly since str2 timestamps
# are lexicographically ordered). Requires ffmpeg on PATH — on the HPC
# you probably need `module load ffmpeg` (or similar) before launching julia.

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

make_movie("vel_1m_plot")
make_movie("vel_10m_plot")
make_movie("vel_100m_plot")
