using Distributed

n_workers = max(1, parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "5")) - 1)   # -1 reserves a CPU for this main process
addprocs(n_workers)
println("running with $(nprocs() - 1) worker processes")

@everywhere begin
    using NCDatasets, CairoMakie, Dates, Statistics
    CairoMakie.activate!()
    include("/home/hchen54/internalwave_julia/function/load_all_hpc.jl")
    include("/home/hchen54/internalwave_julia/function/plotting_fun.jl")

    grid_fname = "/expanse/lustre/projects/uso101/hchen54/input/grid/roms_grd_900m.nc"   # the grid_file listed in the .nc's global attributes
    datadir = "/expanse/lustre/projects/uso101/hchen54/amazon_900m_dbry_2"   # HPC output dir — contains avg/dia/his/rst files mixed together
    figure_path = "/home/hchen54/figure/dbry900m/hisfile"

    ##
    mask_rho, lon_rho, lat_rho, h ,pm, pn, grid_angle = NCDataset(grid_fname) do ds
        ds["mask_rho"][:, :], ds["lon_rho"][:, :], ds["lat_rho"][:, :],
        ds["h"][:,:], ds["pm"][:,:] , ds["pn"][:,:], ds["angle"][:,:]
    end
    lon_rho[lon_rho .> 180] .-= 360   # convert 0-360 convention to -180/180 (east/west hemisphere)
    _, _, mask_p = uvp_masks(mask_rho)
    lon_psi, lat_psi = rho2p(lon_rho), rho2p(lat_rho)
    # no native lon_psi/lat_psi in this grid file — approximate via corner averaging

    # h here is the ROMS grid's own bathymetry (positive-down, no missing values),
    # already on the same lon_rho/lat_rho grid as everything plotted below — unlike
    # check_NCOM_comp.jl, no sign flip / coalesce-to-NaN is needed before contouring.
    bathy_levels = [500, 1000, 2000]

    skip = 30   # downsample for legible quiver arrows — plotting all 686x856 would be unreadable


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
    # four plot types, assume the whole file was fully processed last run and
    # skip reading zeta/temp/u/v entirely (not just skip the plotting).
    outname_zeta_last = joinpath(figure_path, "zeta_plot_$(str2[ntime]).png")
    outname_temp_last = joinpath(figure_path, "temp_depth_plot_$(str2[ntime]).png")
    outname_speed_last = joinpath(figure_path, "speed_plot_$(str2[ntime]).png")
    outname_vor_last = joinpath(figure_path, "vorticity_plot_$(str2[ntime]).png")
 
    if isfile(outname_zeta_last) && isfile(outname_temp_last) &&
       isfile(outname_speed_last) && isfile(outname_vor_last)
        println("all plots already exist for ", fname, ", skipping file entirely")
        return
    end
 
    ## SSH DATA
    # NCDataset(fname) opens the file; the `do...end` block auto-closes it when done
    # (like fopen/fclose in MATLAB, but you don't have to remember to close it)
    # the last line of the block is what gets returned into `zeta` below
    zeta = NCDataset(fname) do ds
        ds["zeta"][:, :, :]   # sea surface height, dims xi_rho x eta_rho x time
    end
    println(fname, " => size(zeta) = ", size(zeta))
 
    zeta_masked = ifelse.(mask_rho .== 0, NaN32, zeta)
 
    # theta_s/theta_b/hc/N are all cheap metadata — attributes, plus a
    # variable's shape without reading its actual data — so we can always
    # fetch them without waiting on the (possibly skippable) full temp read
    # below. This is what z/z_p (needed by every plot type) depend on.
    theta_s, theta_b, hc, N = NCDataset(fname) do ds
        ds.attrib["theta_s"], ds.attrib["theta_b"], ds.attrib["hc"], size(ds["temp"], 3)
    end
 
    z = zeros(size(zeta, 1), size(zeta, 2), N, ntime)
    Cs = nothing
    for t in 1:ntime
        z_dum, Cs = zlevs3(h, zeta[:, :, t], theta_s, theta_b, hc, N, "r", "new2006")
        z[:, :, :, t] = permutedims(z_dum, (2, 3, 1))
    end
    z_p = rho2p(z)
 
    ## TEMP DATA — only actually read the (large) temp array if the temp
    # plot for this file's last time step isn't already sitting on disk
    if isfile(outname_temp_last)
        println("temp plots already exist for ", fname, ", skipping temp read")
    else
        temp = NCDataset(fname) do ds
            ds["temp"][:, :, :,:]
        end
        println(fname, " => size(temp) = ", size(temp))
        temp_masked = ifelse.(mask_rho .== 0, NaN32, temp)
    end
 
    ## vel data for speed + vorticity — same idea, skip the read if both
    # plots that need it already exist for this file's last time step
    if isfile(outname_speed_last) && isfile(outname_vor_last)
        println("speed/vorticity plots already exist for ", fname, ", skipping u/v read")
    else
        u,v = NCDataset(fname) do ds
            ds["u"][:, :, :,:]  , ds["v"][:, :, :,:]
        end
        println(fname, " => size(u) = ", size(u))
        println(fname, " => size(v) = ", size(v))
 
        ## speed
        # average raw (unmasked) u, v onto the rho grid first, then mask the
        # result — masking beforehand would spread NaN from land onto valid
        # ocean cells one row/column too far, same reasoning as vorticity below
        u_rho = u2rho(u)   # still grid-relative (xi/eta), not east/north
        v_rho = v2rho(v)
 
        # speed is rotation-invariant, so no need to rotate before this
        speed = sqrt.(u_rho .^ 2 .+ v_rho .^ 2)
        speed_masked = ifelse.(mask_rho .== 0, NaN32, speed)
 
        # but quiver arrows plotted against true lon/lat DO need rotating out of
        # the grid's local xi/eta directions into true east/north, using the
        # grid's rotation angle (radians) from the grid file
        u_east = u_rho .* cos.(grid_angle) .- v_rho .* sin.(grid_angle)
        v_north = u_rho .* sin.(grid_angle) .+ v_rho .* cos.(grid_angle)
        u_east_masked = ifelse.(mask_rho .== 0, NaN32, u_east)
        v_north_masked = ifelse.(mask_rho .== 0, NaN32, v_north)
 
        ## VORTICITY
        # use the raw (unmasked) u, v here — masking with NaN before differencing
        # would spread NaN into neighboring cells; mask the result afterward instead
        vor_psi = vorticity_cal(u, v, pm, pn)   # relative vorticity on the PSI grid
        vor_psi_masked = ifelse.(mask_p .== 0, NaN32, vor_psi)
    end
 
 
    ## PLOTS — one set of files per ocean_time step
    for t in 1:ntime
 
        # skip incomplete/fill-valued records — e.g. the last time step of a
        # file still being actively written, where every variable is left at
        # the raw NetCDF fill value (~9.97e36) instead of real data
        if maximum(abs, zeta[:, :, t]) > 1e30
            println("skipping t=$t ($(str1[t])) — looks like an incomplete/fill-valued record")
            continue
        end
 
        # -- SSH --
        outname_zeta = joinpath(figure_path, "zeta_plot_$(str2[t]).png")
        if isfile(outname_zeta)
            println("already exists, skipping: ", outname_zeta)
        else
            fig = Figure(size = (800, 600))
            ax = topdown_axis3(fig[1, 1]; title = "Sea surface height (zeta), $(str1[t])")
            sp = plot_curvilinear!(ax, lon_rho, lat_rho, zeta_masked[:, :, t];
                                    colormap = Reverse(:RdBu), colorrange = (-1, 1))
            Colorbar(fig[1, 2], sp, label = "meters")
            contour!(ax, lon_rho, lat_rho, h; levels = bathy_levels,
                     color = RGBf(0.3, 0.3, 0.3), linewidth = 1)
            save(outname_zeta, fig)
            println("saved plot to ", outname_zeta)
        end
 
        # -- TEMP --
        # slice_at_depth does per-column vertical interpolation only (same
        # horizontal grid in/out) — z and temp_masked here are one time step's
        # (M, L, N) fields; target depth is in meters, negative below the surface.
        outname_temp = joinpath(figure_path, "temp_depth_plot_$(str2[t]).png")
        if isfile(outname_temp)
            println("already exists, skipping: ", outname_temp)
        else
            temp_1m = slice_at_depth(z[:, :, :, t], temp_masked[:, :, :, t], -1.0)
            temp_10m = slice_at_depth(z[:, :, :, t], temp_masked[:, :, :, t], -10.0)
 
            fig = Figure(size = (1400, 600))
            for (col, (field, depth_label)) in enumerate(((temp_1m, "1 m"), (temp_10m, "10 m")))
                ax = topdown_axis3(fig[1, 2col - 1]; title = "Temperature at $depth_label, $(str1[t])")
                sp = plot_curvilinear!(ax, lon_rho, lat_rho, field;
                                        colormap = :thermal, colorrange = (24, 30))
                Colorbar(fig[1, 2col], sp, label = "°C")
                contour!(ax, lon_rho, lat_rho, h; levels = bathy_levels,
                         color = RGBf(0.3, 0.3, 0.3), linewidth = 1)
            end
            save(outname_temp, fig)
            println("saved plot to ", outname_temp)
        end
 
        # -- SPEED --
        outname_speed = joinpath(figure_path, "speed_plot_$(str2[t]).png")
        if isfile(outname_speed)
            println("already exists, skipping: ", outname_speed)
        else
            speed_1m = slice_at_depth(z[:, :, :, t], speed_masked[:, :, :, t], -1.0)
            u_east_1m = slice_at_depth(z[:, :, :, t], u_east_masked[:, :, :, t], -1.0)
            v_north_1m = slice_at_depth(z[:, :, :, t], v_north_masked[:, :, :, t], -1.0)
 
            fig = Figure(size = (800, 600))
            ax = topdown_axis3(fig[1, 1]; title = "Speed at 1 m, $(str1[t])")
            sp = plot_curvilinear!(ax, lon_rho, lat_rho, speed_1m; colormap = :speed)
            Colorbar(fig[1, 2], sp, label = "m/s")
            contour!(ax, lon_rho, lat_rho, h; levels = bathy_levels,
                     color = RGBf(0.3, 0.3, 0.3), linewidth = 1)
            quiver_curvilinear!(ax, lon_rho, lat_rho, u_east_1m, v_north_1m;
                                 skip = skip, lengthscale = 0.5, color = :black)
            save(outname_speed, fig)
            println("saved plot to ", outname_speed)
        end
 
        # -- VORTICITY --
        outname_vor = joinpath(figure_path, "vorticity_plot_$(str2[t]).png")
        if isfile(outname_vor)
            println("already exists, skipping: ", outname_vor)
        else
            # speed's u_east_1m/v_north_1m may have been skipped above (if the
            # speed plot already existed), so recompute them here too — cheap
            # relative to the rest of this block, and keeps this block able to
            # run independently of the speed one.
            u_east_1m = slice_at_depth(z[:, :, :, t], u_east_masked[:, :, :, t], -1.0)
            v_north_1m = slice_at_depth(z[:, :, :, t], v_north_masked[:, :, :, t], -1.0)
            vor_1m = slice_at_depth(z_p[:, :, :, t], vor_psi_masked[:, :, :, t], -1.0)
 
            clim = 5 * mean(abs, filter(!isnan, vor_1m))   # symmetric color limits centered on 0
            fig = Figure(size = (800, 600))
            ax = topdown_axis3(fig[1, 1]; title = "Relative vorticity, surface, $(str1[t])")
            sp = plot_curvilinear!(ax, lon_psi, lat_psi, vor_1m;
                                    colormap = Reverse(:RdBu), colorrange = (-clim, clim))
            Colorbar(fig[1, 2], sp, label = "s⁻¹")
            contour!(ax, lon_rho, lat_rho, h; levels = bathy_levels,
                     color = RGBf(0.3, 0.3, 0.3), linewidth = 1)
            quiver_curvilinear!(ax, lon_rho, lat_rho, u_east_1m, v_north_1m;
                                 skip = skip, lengthscale = 0.5, color = :black)
            save(outname_vor, fig)
            println("saved plot to ", outname_vor)
        end
    end
end

# the HPC output dir mixes roms_avg/dia/his/rst files together, so filter
# to just the "his" files (they carry zeta/temp/u/v, everything the plots
# below need) — sorted so files are processed in chronological order,
# which works here because the filenames embed a sortable timestamp
# files = sort(filter(f -> occursin("roms_avg", basename(f)) && endswith(f, ".nc"), readdir(datadir, join=true)))
# files = [joinpath(datadir, "roms_avg.20220908210000.nc")]
files = sort(filter(f -> occursin("roms_his", basename(f)) &&
                         endswith(f, ".nc") &&
                         basename(f) <= "roms_his.20220923210000.nc",
                    readdir(datadir, join=true)))
println("found $(length(files)) his files in $datadir")
# pmap hands files out to whichever worker is free, one at a time, and
# blocks here until every file is done — no manual scheduling needed.
# pmap(process_file, files)
pmap(process_file, files; on_error = ex -> println("a file failed: ", ex))


## MOVIES
# stitches each set of PNGs (across all his files/timesteps) into an mp4,
# in chronological order (filenames sort correctly since str2 timestamps
# are lexicographically ordered). Requires ffmpeg on PATH — on the HPC
# you probably need `module load ffmpeg` (or similar) before launching julia.
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

make_movie("zeta_plot")
make_movie("temp_depth_plot")
make_movie("speed_plot")
make_movie("vorticity_plot")