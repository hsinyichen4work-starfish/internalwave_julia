using Distributed

# spin up worker *processes* (not threads) — PyPlot calls into Python's
# matplotlib via PyCall, which isn't thread-safe, so Threads.@threads would
# risk corrupted/crashed plots. Separate processes each get their own Python
# interpreter and don't share that risk.
#
# if running under SLURM with e.g. `--cpus-per-task=8` in your sbatch script,
# this uses 7 workers + 1 driver process to match your allocation; otherwise
# (e.g. testing interactively on a login node) it falls back to 4 workers.
cpus_per_task = tryparse(Int, get(ENV, "SLURM_CPUS_PER_TASK", ""))
nworkers_wanted = cpus_per_task === nothing ? 4 : max(cpus_per_task - 1, 1)
addprocs(nworkers_wanted)
println("running with $nworkers_wanted worker process(es) + 1 driver")

# `@everywhere` runs this on every worker (and the driver) — each process
# needs its own copy of the packages, helper functions, colormaps, and grid
# data since workers don't share memory with the driver.
@everywhere using NCDatasets, PyPlot, Dates, ColorSchemes, Statistics

@everywhere include("/home/hsinyi/Documents/Julia/function/load_all.jl")
# path relative to where you launch julia (your Documents/Julia folder)

@everywhere begin
    # cmocean's "thermal" colormap, via ColorSchemes.jl, converted to a
    # matplotlib colormap object for use with PyPlot's pcolormesh/cmap=...
    cmap_thermal = PyPlot.matplotlib.colors.ListedColormap([[c.r, c.g, c.b] for c in colorschemes[:thermal]])
    cmap_speed = PyPlot.matplotlib.colors.ListedColormap([[c.r, c.g, c.b] for c in colorschemes[:speed]])

    grid_fname = "/home/hsinyi/roms_data/grid/roms_grd_900m.nc"   # the grid_file listed in the .nc's global attributes
    figure_path = "/home/hsinyi/figure/20260914_julia_outputtest"
    skip = 30   # downsample for legible quiver arrows — plotting all 686x856 would be unreadable

    mask_rho, lon_rho, lat_rho, h ,pm, pn, grid_angle = NCDataset(grid_fname) do ds
        ds["mask_rho"][:, :], ds["lon_rho"][:, :], ds["lat_rho"][:, :],
        ds["h"][:,:], ds["pm"][:,:] , ds["pn"][:,:], ds["angle"][:,:]
    end
    lon_rho[lon_rho .> 180] .-= 360   # convert 0-360 convention to -180/180 (east/west hemisphere)
    _, _, mask_p = uvp_masks(mask_rho)
    lon_psi, lat_psi = rho2p(lon_rho), rho2p(lat_rho)   # no native lon_psi/lat_psi in this grid file — approximate via corner averaging
end

# the per-file work (same body as the serial check_zeta.jl's `for fname in
# his_files ... end` loop), wrapped in a function so `pmap` can dispatch one
# call per file to whichever worker is free.
#
# every stage below is wrapped with a `stage = "..."` label and the whole
# thing is inside one try/catch: if anything throws, we print which file AND
# which stage it happened in, plus the full stack trace, before rethrowing
# — that gets printed on the worker where it happened, and worker stdout is
# forwarded back to this driver's terminal/log automatically.
@everywhere function process_his_file(fname)
    stage = "startup"
    try
        stage = "reading ocean_time"
        ocean_time = NCDataset(fname) do ds
            ds["ocean_time"][:]
        end
        println(fname, " => size(ocean_time) = ", size(ocean_time))
        ntime = length(ocean_time)
        t_ref = DateTime(1994, 1, 1, 0, 0, 0)
        realtime = t_ref .+ Millisecond.(round.(Int, ocean_time .* 1000))
        str1 = Dates.format.(realtime, "yyyy-mm-dd HH:MM:SS")
        str2 = Dates.format.(realtime, "yyyymmdd_HHMM")

        stage = "reading zeta"
        zeta = NCDataset(fname) do ds
            ds["zeta"][:, :, :]
        end
        println(fname, " => size(zeta) = ", size(zeta))
        zeta_masked = ifelse.(mask_rho .== 0, NaN32, zeta)

        stage = "reading temp"
        temp = NCDataset(fname) do ds
            ds["temp"][:, :, :,:]
        end
        println(fname, " => size(temp) = ", size(temp))

        stage = "reading theta_s/theta_b/hc attributes"
        theta_s, theta_b, hc = NCDataset(fname) do ds
            ds.attrib["theta_s"], ds.attrib["theta_b"], ds.attrib["hc"]
        end
        N = size(temp,3)

        stage = "computing z levels (zlevs3)"
        z = zeros(size(temp))
        Cs = nothing
        for t in 1:ntime
            z_dum, Cs = zlevs3(h, zeta[:, :, t], theta_s, theta_b, hc, N, "r", "new2006")
            z[:, :, :, t] = permutedims(z_dum, (2, 3, 1))
        end
        z_p = rho2p(z)
        temp_masked = ifelse.(mask_rho .== 0, NaN32, temp)

        stage = "reading u/v"
        u,v = NCDataset(fname) do ds
            ds["u"][:, :, :,:]  , ds["v"][:, :, :,:]
        end
        println(fname, " => size(u) = ", size(u))
        println(fname, " => size(v) = ", size(v))

        stage = "computing speed"
        u_rho = u2rho(u)
        v_rho = v2rho(v)
        speed = sqrt.(u_rho .^ 2 .+ v_rho .^ 2)
        speed_masked = ifelse.(mask_rho .== 0, NaN32, speed)

        u_east = u_rho .* cos.(grid_angle) .- v_rho .* sin.(grid_angle)
        v_north = u_rho .* sin.(grid_angle) .+ v_rho .* cos.(grid_angle)
        u_east_masked = ifelse.(mask_rho .== 0, NaN32, u_east)
        v_north_masked = ifelse.(mask_rho .== 0, NaN32, v_north)

        stage = "computing vorticity"
        vor_psi = vorticity_cal(u, v, pm, pn)
        vor_psi_masked = ifelse.(mask_p .== 0, NaN32, vor_psi)

        for t in 1:ntime

            if maximum(abs, zeta[:, :, t]) > 1e30
                println("skipping t=$t ($(str1[t])) — looks like an incomplete/fill-valued record")
                continue
            end

            stage = "plotting SSH (t=$t)"
            fig, ax = subplots(figsize = (8, 6))
            pc = ax.pcolormesh(lon_rho, lat_rho, zeta_masked[:, :, t], shading = "auto", cmap = "RdBu_r", vmin = -1, vmax = 1)
            ax.set_aspect("equal")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_title("Sea surface height (zeta), $(str1[t])")
            colorbar(pc, ax = ax, label = "meters")
            outname = joinpath(figure_path, "zeta_plot_$(str2[t]).png")
            savefig(outname)
            println("saved plot to ", outname)
            close(fig)

            stage = "plotting temp (t=$t)"
            temp_1m = slice_at_depth(z[:, :, :, t], temp_masked[:, :, :, t], -1.0)
            temp_10m = slice_at_depth(z[:, :, :, t], temp_masked[:, :, :, t], -10.0)

            fig, axs = subplots(1, 2, figsize = (14, 6))
            for (ax, field, depth_label) in ((axs[1], temp_1m, "1 m"), (axs[2], temp_10m, "10 m"))
                pc = ax.pcolormesh(lon_rho, lat_rho, field, shading = "auto", cmap = cmap_thermal)
                ax.set_aspect("equal")
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")
                ax.set_title("Temperature at $depth_label, $(str1[t])")
                colorbar(pc, ax = ax, label = "°C")
            end
            outname = joinpath(figure_path, "temp_depth_plot_$(str2[t]).png")
            savefig(outname)
            println("saved plot to ", outname)
            close(fig)

            stage = "plotting speed (t=$t)"
            speed_1m = slice_at_depth(z[:, :, :, t], speed_masked[:, :, :, t], -1.0)
            u_east_1m = slice_at_depth(z[:, :, :, t], u_east_masked[:, :, :, t], -1.0)
            v_north_1m = slice_at_depth(z[:, :, :, t], v_north_masked[:, :, :, t], -1.0)

            fig, ax = subplots(figsize = (8, 6))
            pc = ax.pcolormesh(lon_rho, lat_rho, speed_1m, shading = "auto", cmap = cmap_speed)
            colorbar(pc, ax = ax, label = "m/s")
            ax.quiver(lon_rho[1:skip:end, 1:skip:end], lat_rho[1:skip:end, 1:skip:end],
                      u_east_1m[1:skip:end, 1:skip:end], v_north_1m[1:skip:end, 1:skip:end],
                      color = "k", scale = 40)
            ax.set_aspect("equal")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_title("Speed at 1 m, $(str1[t])")
            outname = joinpath(figure_path, "speed_plot_$(str2[t]).png")
            savefig(outname)
            println("saved plot to ", outname)
            close(fig)

            stage = "plotting vorticity (t=$t)"
            vor_1m = slice_at_depth(z_p[:, :, :, t], vor_psi_masked[:, :, :, t], -1.0)
            clim = 5 * mean(abs, filter(!isnan, vor_1m))
            fig, ax = subplots(figsize = (8, 6))
            pc = ax.pcolormesh(lon_psi, lat_psi, vor_1m, shading = "auto", cmap = "RdBu_r", vmin = -clim, vmax = clim)
            ax.quiver(lon_rho[1:skip:end, 1:skip:end], lat_rho[1:skip:end, 1:skip:end],
                      u_east_1m[1:skip:end, 1:skip:end], v_north_1m[1:skip:end, 1:skip:end],
                      color = "k", scale = 40)
            ax.set_aspect("equal")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_title("Relative vorticity, surface, $(str1[t])")
            colorbar(pc, ax = ax, label = "s⁻¹")
            outname = joinpath(figure_path, "vorticity_plot_$(str2[t]).png")
            savefig(outname)
            println("saved plot to ", outname)
            close(fig)
        end

        return :ok

    catch e
        println(stderr, "="^60)
        println(stderr, "ERROR on worker $(myid()) while processing $fname")
        println(stderr, "  stage: $stage")
        showerror(stderr, e, catch_backtrace())
        println(stderr)
        println(stderr, "="^60)
        return (:failed, fname, stage, sprint(showerror, e))
    end
end

# --- driver process only from here on ---

datadir = "/expanse/lustre/projects/uso101/hchen54/amazon_900m_dbry"   # HPC output dir — contains avg/dia/his/rst files mixed together

his_files = sort(filter(f -> occursin("roms_his", basename(f)) && endswith(f, ".nc"), readdir(datadir, join=true)))
println("found $(length(his_files)) his files in $datadir")

# each call returns :ok or (:failed, fname, stage, message) instead of
# throwing — one bad file won't abort the whole run, and afterward we print
# a clean summary of exactly which files/stages failed
results = pmap(process_his_file, his_files)

failed = [r for r in results if r isa Tuple && r[1] == :failed]
println()
println("finished: $(length(results) - length(failed)) succeeded, $(length(failed)) failed")
for (_, fname, stage, msg) in failed
    println("  FAILED: $fname (stage: $stage) — $msg")
end

## MOVIES
# stitches each set of PNGs (across all his files/timesteps) into an mp4,
# in chronological order. Requires ffmpeg on PATH — on the HPC you probably
# need `module load ffmpeg` (or similar) before launching julia.
# Runs on the driver only, after every worker has finished its files.
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
    run(`ffmpeg -y -r $fps -f concat -safe 0 -i $listfile -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -vcodec libx264 -pix_fmt yuv420p $outname`)
    rm(listfile)
    println("saved movie to ", outname)
end

if isempty(failed)
    make_movie("zeta_plot")
    make_movie("temp_depth_plot")
    make_movie("speed_plot")
    make_movie("vorticity_plot")
else
    println("skipping movies since some files failed — fix the errors above and rerun")
end
